import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from lm_eval import simple_evaluate


# -------------------
# Argument parsing
# -------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen MoE Evaluation Script")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen1.5-MoE-A2.7B", help='Model name or path to load')
    parser.add_argument('--tasks', type=str, nargs='+', default=['mmlu'], help='List of evaluation tasks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples for quick testing')
    parser.add_argument('--output_file', type=str, default=None, help='File to save results JSON')

    # Backend selection
    parser.add_argument('--backend', type=str, choices=['hf', 'vllm'], default='vllm',
                        help='Inference backend: "vllm" (fast, no pruning) or "hf" (slower, supports pruning)')

    # Pruning options (HF backend only)
    parser.add_argument('--use_pruned_model', action='store_true', help='Whether to use pruned model (HF backend only)')
    parser.add_argument('--pruned_metadata', type=str, default=None, help='Path to pruned expert metadata JSON')
    parser.add_argument('--k', type=int, default=20, help='Maximum number of experts to prune per layer')
    parser.add_argument('--pruning_method', type=str, choices=['mask', 'zero'], default='zero',
                        help='Method to use for pruning experts')

    # vLLM options
    parser.add_argument('--tensor_parallel_size', type=int, default=None,
                        help='Number of GPUs for tensor parallelism (vLLM only, default: all available)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use (vLLM only, default: 0.9)')

    return parser.parse_args()


def create_hf_model(args):
    """Create an HF model with multi-GPU support via Accelerate pipeline parallelism."""
    from lm_eval.models.huggingface import HFLM
    from utils.model_utils import apply_pruning
    from utils.common_utils import get_experts_to_prune_from_json

    model = HFLM(
        args.model_name,
        parallelize=True,
        dtype="bfloat16",
    )

    if args.use_pruned_model and args.pruned_metadata:
        experts_to_prune = get_experts_to_prune_from_json(
            path=args.pruned_metadata,
            k=args.k
        )
        apply_pruning(model.model, experts_to_prune, mode=args.pruning_method)
        print(f"Applied {args.pruning_method} pruning with k={args.k}")

    return model


def create_vllm_model(args):
    """Create a vLLM model with multi-GPU support via tensor parallelism."""
    from lm_eval.models.vllm_causallms import VLLM

    tp_size = args.tensor_parallel_size or torch.cuda.device_count()

    model = VLLM(
        pretrained=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size="auto",
        trust_remote_code=True,
    )

    print(f"vLLM model loaded with tensor_parallel_size={tp_size}, "
          f"gpu_memory_utilization={args.gpu_memory_utilization}")
    return model


# -------------------
# Main
# -------------------
def main():
    args = parse_args()

    # Validate: pruning requires HF backend
    if args.use_pruned_model and args.backend == 'vllm':
        print("WARNING: Pruning is not supported with vLLM backend. Switching to HF backend.")
        args.backend = 'hf'

    print(f"Using backend: {args.backend}")

    if args.backend == 'vllm':
        model = create_vllm_model(args)
    else:
        model = create_hf_model(args)

    # Prepare arguments for simple_evaluate
    eval_kwargs = dict(
        model=model,
        tasks=args.tasks,
        log_samples=False,
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
