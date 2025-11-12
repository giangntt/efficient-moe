import argparse
import json
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import torch
import torch.nn.functional as F
from model_utils import apply_pruning
from common_utils import get_topk_experts_from_json

# -------------------
# Argument parsing
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen MoE Evaluation Script")
    parser.add_argument('--tasks', type=str, nargs='+', default=['wikitext'], help='List of evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples for quick testing')
    parser.add_argument('--use_pruned_model', action='store_true', help='Whether to use pruned model')
    parser.add_argument('--pruned_metadata', type=str, default=None, help='Path to pruned expert metadata JSON')
    parser.add_argument('--mode', type=str, choices=['least', 'mode'], default='least', help='Strategy for selecting experts to prune')
    parser.add_argument('--pruning_method', type=str, choices=['mask', 'zero', 'dynamic'], default='zero', help='Method to use for pruning experts')
    parser.add_argument('--k', type=int, default=5, help='Number of experts to prune per layer')
    parser.add_argument('--dynamic_routing_threshold', type=float, default=0.8, help='Cumulative probability threshold for dynamic routing')
    parser.add_argument('--device', type=str, default='cuda', help='Device for model')
    parser.add_argument('--output_file', type=str, default=None, help='File to save results JSON')
    
    return parser.parse_args()

# -------------------
# Main
# -------------------
def main():
    args = parse_args()
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    model = HFLM(model_name, device=args.device, dtype="bfloat16")

    if args.use_pruned_model and args.pruned_metadata:
        experts_to_prune = get_topk_experts_from_json(
            path=args.pruned_metadata,
            top_k=args.k,
            mode=args.mode
        )
        apply_pruning(
            model.model, 
            experts_to_prune, 
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

    results = simple_evaluate(**eval_kwargs)
    print(results)

    if args.output_file:
        output_file = args.output_file
        output_data = {
            "config": vars(args),
            "results": results["results"]
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results and config saved to {output_file}")

if __name__ == "__main__":
    main()
