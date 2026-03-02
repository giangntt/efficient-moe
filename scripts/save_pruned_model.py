#!/usr/bin/env python3
"""
Prune a MoE model offline and save it to disk.

Instead of applying pruning at runtime (which requires the HF backend), this
script permanently mutates the model weights and saves a standard Hugging Face
model directory that can be loaded by any inference engine, including vLLM.

Two pruning modes are supported:
  zero  — Expert MLP weights are zeroed. Tokens may still be routed to pruned
           experts, but their output will be all zeros.
  mask  — Expert MLP weights are zeroed AND the router gate gets a large
           negative bias for pruned experts, so they are never selected.

Usage
-----
python scripts/save_pruned_model.py \\
    --model_name Qwen/Qwen3-30B-A3B \\
    --pruned_metadata expert_stats_stem_pruned.json \\
    --k 20 \\
    --pruning_mode mask \\
    --output_dir outputs/pruned_models/Qwen3-30B-A3B-stem-k20-mask

Then evaluate with vLLM:
python scripts/evaluation.py \\
    --model_name outputs/pruned_models/Qwen3-30B-A3B-stem-k20-mask \\
    --tasks mmlu \\
    --backend vllm
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.model_utils import prune_weights_inplace
from utils.common_utils import get_experts_to_prune_from_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune a MoE model offline and save it to disk."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="HuggingFace model ID or local path to load"
    )
    parser.add_argument(
        "--pruned_metadata", type=str, required=True,
        help="Path to the JSON file produced by profile_and_prune.py"
    )
    parser.add_argument(
        "--k", type=int, default=20,
        help="Maximum number of experts to prune per layer (default: 20)"
    )
    parser.add_argument(
        "--pruning_mode", type=str, choices=["zero", "mask"], default="zero",
        help=(
            "Pruning mode: "
            "'zero' zeros expert weights (router unchanged); "
            "'mask' also biases the gate so pruned experts are never selected "
            "(default: zero)"
        )
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the pruned model and tokenizer"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load tokenizer
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Loading tokenizer...")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ------------------------------------------------------------------ #
    # 2. Load model across all available GPUs
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    if hasattr(model, "hf_device_map"):
        devices = sorted(set(str(d) for d in model.hf_device_map.values()))
        print(f"Model loaded across devices: {', '.join(devices)}")
    else:
        print("Model loaded on single device")

    # ------------------------------------------------------------------ #
    # 3. Load pruning metadata
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Loading pruning metadata...")
    print(f"{'='*60}")
    experts_to_prune = get_experts_to_prune_from_json(
        path=args.pruned_metadata,
        k=args.k,
    )
    total = sum(len(v) for v in experts_to_prune.values())
    print(f"Will prune {total} experts across {len(experts_to_prune)} layers "
          f"(k={args.k}, mode={args.pruning_mode})")

    # ------------------------------------------------------------------ #
    # 4. Prune weights in-place
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Pruning model (mode={args.pruning_mode})...")
    print(f"{'='*60}")
    prune_weights_inplace(model, experts_to_prune, mode=args.pruning_mode)

    # ------------------------------------------------------------------ #
    # 5. Save pruned model + tokenizer to disk
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Saving pruned model to: {args.output_dir}")
    print(f"{'='*60}")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # save_pretrained with safe serialization (safetensors)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Pruned model saved to : {args.output_dir}")
    print(f"Pruning mode          : {args.pruning_mode}")
    print(f"Experts pruned        : {total} across {len(experts_to_prune)} layers")
    print()
    print("To evaluate with vLLM:")
    print(f"  python scripts/evaluation.py \\")
    print(f"      --model_name {args.output_dir} \\")
    print(f"      --tasks mmlu \\")
    print(f"      --backend vllm")


if __name__ == "__main__":
    main()
