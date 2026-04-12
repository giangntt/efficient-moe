import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn.functional as F
from utils.model_utils import apply_pruning
from utils.common_utils import get_experts_to_prune_from_json

# -------------------
# Argument parsing
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen MoE Evaluation Script")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen1.5-MoE-A2.7B", help='Model name or path to load')
    parser.add_argument('--tasks', type=str, nargs='+', default=['mmlu'], help='List of evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples for quick testing')
    parser.add_argument('--use_pruned_model', action='store_true', help='Whether to use pruned model')
    parser.add_argument('--pruned_metadata', type=str, default=None, help='Path to pruned expert metadata JSON')
    parser.add_argument('--k', type=int, default=20, help='Maximum number of experts to prune per layer')
    parser.add_argument('--pruning_method', type=str, choices=['mask', 'zero', 'dynamic'], default='zero', help='Method to use for pruning experts')
    parser.add_argument('--dynamic_routing_threshold', type=float, default=0.8, help='Cumulative probability threshold for dynamic routing')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model')
    parser.add_argument('--output_file', type=str, default=None, help='File to save results JSON')
    parser.add_argument('--confirm_run_unsafe_code', action='store_true',
                        help='Allow lm_eval to execute generated code (required for humaneval)')

    return parser.parse_args()

# -------------------
# Main
# -------------------
def main():
    args = parse_args()
    model = HFLM(args.model_name, parallelize=True, dtype="bfloat16")

    if args.use_pruned_model and args.pruned_metadata:
        experts_to_prune = get_experts_to_prune_from_json(
            path=args.pruned_metadata,
            k=args.k
        )
        apply_pruning(
            model.model, experts_to_prune,
            mode=args.pruning_method,
            dynamic_routing_threshold=args.dynamic_routing_threshold
        )

    # Prepare arguments for simple_evaluate
    eval_kwargs = dict(
        model=model,
        tasks=args.tasks,
        log_samples=False,
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.limit:
        eval_kwargs['limit'] = args.limit
    if args.confirm_run_unsafe_code:
        eval_kwargs['confirm_run_unsafe_code'] = True

    t0 = time.perf_counter()
    results = simple_evaluate(**eval_kwargs)
    eval_s = time.perf_counter() - t0

    n_samples = args.limit if args.limit else None
    ms_per_sample = eval_s / n_samples * 1000 if n_samples else None

    print(f"[latency] eval: {eval_s:.2f}s"
          + (f"  ({ms_per_sample:.1f} ms/question)" if ms_per_sample else ""))
    print(results)

    # Collect expert activation stats for dynamic routing
    expert_activation_report = {}
    if args.pruning_method == "dynamic":
        actual_model = model.model.model if hasattr(model.model, 'model') else model.model
        total_avg = 0
        num_layers = 0
        for i, layer in enumerate(actual_model.layers):
            moe_block = layer.mlp
            if hasattr(moe_block, "num_activated_experts_log") and moe_block.num_activated_experts_log:
                avg_experts = sum(moe_block.num_activated_experts_log) / len(moe_block.num_activated_experts_log)
                expert_activation_report[f"layer_{i}"] = f"{avg_experts:.2f}"
                total_avg += avg_experts
                num_layers += 1
        if num_layers > 0:
            expert_activation_report["overall_average"] = f"{total_avg / num_layers:.2f}"

    if args.output_file:
        output_data = {
            "config": vars(args),
            "latency": {
                "eval_s": eval_s,
                **({"ms_per_question": ms_per_sample} if ms_per_sample else {}),
            },
            "results": results["results"]
        }
        if expert_activation_report:
            output_data["expert_activation_report"] = expert_activation_report
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results and config saved to {args.output_file}")

if __name__ == "__main__":
    main()
