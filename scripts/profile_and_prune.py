#!/usr/bin/env python3
"""
Script to profile a model on given prompts and determine experts to prune.
This script collects router logits and expert activations, computes statistics,
and saves a list of experts to prune based on mean and variance activation norms.
"""

import os
import json
import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

STATISTICS_DIR = "outputs/statistics"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hook_utils import ExpertActivationHook
from utils.router_utils import collect_router_logits
from utils.analysis_utils import compute_all_stats
from utils.data_utils import prepare_mmlu_prompts, prepare_gsm8k_prompts, MMLU_CATEGORIES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile model and determine experts to prune"
    )
    
    # Data source options
    parser.add_argument(
        '--prompts_file',
        type=str,
        default=None,
        help='Path to JSON file containing list of prompt strings'
    )
    parser.add_argument(
        '--mmlu_topic',
        type=str,
        choices=list(MMLU_CATEGORIES.keys()) + [None],
        default=None,
        help='MMLU topic to use for prompts (humanities, other, social_sciences, stem)'
    )
    parser.add_argument(
        '--gsm8k',
        action='store_true',
        help='Use GSM8K dataset for prompts'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=50,
        help='Number of samples to use (for GSM8K or for max samples per subject of MMLU)'
    )
    
    # Pruning criteria
    parser.add_argument(
        '--max_pruned_experts_per_layer',
        type=int,
        default=30,
        help='Maximum number of experts to prune per layer'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        choices=['dynamic', 'absolute', 'percentile', 'std'],
        default='dynamic',
        help='Method to determine thresholds: dynamic (normalize to 0-1 and use midpoints), absolute (normalize to 0-1 and use provided thresholds), percentile (use percentiles), std (use std-based thresholds)'
    )
    parser.add_argument(
        '--mean_act_threshold',
        type=float,
        default=None,
        help='Threshold for mean activation norm: for absolute (0-1 range after normalization), for std (multiplier of std, negative = below mean). If None, uses percentile.'
    )
    parser.add_argument(
        '--var_act_threshold',
        type=float,
        default=None,
        help='Threshold for variance activation norm: for absolute (0-1 range after normalization), for std (multiplier of std, negative = below mean). If None, uses percentile.'
    )
    parser.add_argument(
        '--mean_act_percentile',
        type=float,
        default=50.0,
        help='Percentile threshold for mean activation norm (0-100, for percentile method or fallback)'
    )
    parser.add_argument(
        '--var_act_percentile',
        type=float,
        default=50.0,
        help='Percentile threshold for variance activation norm (0-100, for percentile method or fallback)'
    )
    parser.add_argument(
        '--layers_to_prune',
        type=int,
        nargs='+',
        default=None,
        help='List of layer indices to prune (default: all layers except 0 and last)'
    )
    
    # Model and device
    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen/Qwen1.5-MoE-A2.7B",
        help='Model name or path to load'
    )
    parser.add_argument(
        '--cuda_visible_devices',
        type=str,
        default=None,
        help='CUDA visible devices (e.g., "0,1")'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help='Device to use (overrides cuda_visible_devices); ignored when using device_map=auto'
    )

    # Generation parameters
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=10000,
        help='Maximum new tokens to generate per prompt during profiling'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (set to None/omit for greedy)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Nucleus sampling top-p'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--enable_thinking',
        action='store_true',
        default=False,
        help='Enable Qwen3 thinking mode (default: disabled)'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: STATISTICS_DIR)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSON file path (default: auto-generated)'
    )
    parser.add_argument(
        '--save_stats',
        action='store_true',
        help='Save computed statistics to JSON'
    )
    
    return parser.parse_args()


def load_prompts(args):
    """Load prompts based on arguments."""
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict) and 'prompts' in data:
                prompts = data['prompts']
            else:
                raise ValueError(f"Invalid prompts file format: {args.prompts_file}")
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        return prompts
    
    elif args.mmlu_topic:
        prompts = prepare_mmlu_prompts(
            topic=args.mmlu_topic,
            max_samples_per_subject=args.sample_size
        )
        print(f"Loaded {len(prompts)} prompts from MMLU topic: {args.mmlu_topic}")
        return prompts
    
    elif args.gsm8k:
        prompts = prepare_gsm8k_prompts(sample_size=args.sample_size)
        print(f"Loaded {len(prompts)} prompts from GSM8K")
        return prompts
    
    else:
        raise ValueError(
            "Must specify one of: --prompts_file, --mmlu_topic, or --gsm8k"
        )


def profile_model(model, tokenizer, prompts, device, args):
    """Profile model by collecting router logits and expert activations."""
    print("\n" + "="*60)
    print("Profiling model...")
    print("="*60)

    # Register hooks and collect data
    hook_manager = ExpertActivationHook()
    hook_manager.register_hooks(model)

    # Collect router logits via generation
    result = collect_router_logits(
        model, tokenizer, prompts, device,
        output_final_logits=False,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    router_logits = result['router_logits']
    
    # Get expert activations and clear hooks
    expert_activations = hook_manager.expert_activations
    hook_manager.clear_hooks()
    
    print(f"Collected data for {len(router_logits)} layers")
    print(f"Total expert activation entries: {len(expert_activations)}")
    
    return router_logits, expert_activations


def compute_statistics(router_logits, expert_activations, top_k=4):
    """Compute router and expert statistics."""
    print("\n" + "="*60)
    print("Computing statistics...")
    print("="*60)
    
    stats = compute_all_stats(
        router_logits=router_logits,
        expert_activations=expert_activations,
        top_k=top_k,
        device='cpu'
    )
    
    print(f"Computed statistics for {len(stats)} layers")
    return stats


def determine_experts_to_prune(stats, router_logits, args):
    """Determine which experts to prune based on criteria."""
    print("\n" + "="*60)
    print("Determining experts to prune...")
    print("="*60)
    
    # Determine layers to prune
    if args.layers_to_prune is None:
        # Default: all layers except first and last
        all_layers = sorted(stats.keys())
        layers_to_prune = [l for l in all_layers if l != 0 and l != all_layers[-1]]
    else:
        layers_to_prune = args.layers_to_prune
    
    print(f"Pruning layers: {layers_to_prune}")
    
    # Prune experts based on mean and variance activation norm thresholds
    experts_to_prune = {}
    
    for layer_id in layers_to_prune:
        if layer_id not in stats:
            continue
        
        layer_stats = stats[layer_id]
        mean_act = np.array(layer_stats.get('mean_act', []))
        var_act = np.array(layer_stats.get('var_act', []))
        
        if len(mean_act) == 0 or len(var_act) == 0:
            continue
        
        # Determine thresholds based on method
        if args.threshold_method == 'dynamic':
            # 1️⃣ Normalize to 0-1
            mean_act_min, mean_act_max = mean_act.min(), mean_act.max()
            var_act_min, var_act_max = var_act.min(), var_act.max()
            
            mean_act_norm = (mean_act - mean_act_min) / (mean_act_max - mean_act_min + 1e-12)
            var_act_norm = (var_act - var_act_min) / (var_act_max - var_act_min + 1e-12)
            
            # 2️⃣ Middle of value range
            mean_act_mid = (mean_act_norm.min() + mean_act_norm.max()) / 2 / 2  # then divide by 2
            var_act_mid = (var_act_norm.min() + var_act_norm.max()) / 2
            
            # Pruning rule: both normalized values below their midpoints
            pruned_mask = (mean_act_norm < mean_act_mid) & (var_act_norm < var_act_mid)
            
        elif args.threshold_method == 'absolute':
            # Normalize to 0-1, then use absolute thresholds in 0-1 range
            mean_act_min, mean_act_max = mean_act.min(), mean_act.max()
            var_act_min, var_act_max = var_act.min(), var_act.max()
            
            mean_act_norm = (mean_act - mean_act_min) / (mean_act_max - mean_act_min + 1e-12)
            var_act_norm = (var_act - var_act_min) / (var_act_max - var_act_min + 1e-12)
            
            # Use absolute thresholds (or fallback to percentile)
            if args.mean_act_threshold is not None:
                mean_threshold = args.mean_act_threshold
            else:
                mean_threshold = np.percentile(mean_act_norm, args.mean_act_percentile)
            
            if args.var_act_threshold is not None:
                var_threshold = args.var_act_threshold
            else:
                var_threshold = np.percentile(var_act_norm, args.var_act_percentile)
            
            # Pruning rule: both normalized values below their thresholds
            pruned_mask = (mean_act_norm < mean_threshold) & (var_act_norm < var_threshold)
            
        elif args.threshold_method == 'std':
            # Use std-based thresholds (mean - N*std)
            mean_act_mean = mean_act.mean()
            mean_act_std = mean_act.std()
            var_act_mean = var_act.mean()
            var_act_std = var_act.std()
            
            # Use thresholds as multipliers of std (negative means below mean)
            if args.mean_act_threshold is not None:
                mean_threshold = mean_act_mean + args.mean_act_threshold * mean_act_std
            else:
                # Fallback to percentile
                mean_threshold = np.percentile(mean_act, args.mean_act_percentile)
            
            if args.var_act_threshold is not None:
                var_threshold = var_act_mean + args.var_act_threshold * var_act_std
            else:
                # Fallback to percentile
                var_threshold = np.percentile(var_act, args.var_act_percentile)
            
            # Pruning rule: both values below their thresholds
            pruned_mask = (mean_act < mean_threshold) & (var_act < var_threshold)
            
        else:  # percentile
            # Use percentile thresholds
            mean_threshold = np.percentile(mean_act, args.mean_act_percentile)
            var_threshold = np.percentile(var_act, args.var_act_percentile)
            
            # Pruning rule: both values below their thresholds
            pruned_mask = (mean_act < mean_threshold) & (var_act < var_threshold)
        
        # Limit to max_pruned_experts_per_layer
        pruned_indices = np.where(pruned_mask)[0]
        if len(pruned_indices) > args.max_pruned_experts_per_layer:
            # Keep the ones with lowest mean_act (or mean_act_norm if using normalized methods)
            if args.threshold_method in ['dynamic', 'absolute']:
                sort_values = mean_act_norm[pruned_indices]
            else:
                sort_values = mean_act[pruned_indices]
            idx_sort = np.argsort(sort_values)
            pruned_indices = pruned_indices[idx_sort[:args.max_pruned_experts_per_layer]]
        
        if len(pruned_indices) > 0:
            experts_to_prune[layer_id] = pruned_indices.tolist()
    
    print(f"Mean-var activation pruning: Found experts below thresholds in {len(experts_to_prune)} layers")
    
    # Filter to only layers we want to prune (if layers_to_prune is specified)
    if layers_to_prune is not None:
        experts_to_prune = {
            k: v for k, v in experts_to_prune.items()
            if k in layers_to_prune and len(v) > 0
        }
    else:
        # Remove empty layers
        experts_to_prune = {
            k: v for k, v in experts_to_prune.items()
            if len(v) > 0
        }
    
    total_experts = sum(len(v) for v in experts_to_prune.values())
    print(f"Selected {total_experts} experts to prune across {len(experts_to_prune)} layers")
    
    return experts_to_prune


def save_results(experts_to_prune, stats, router_logits, args):
    """Save results to files."""
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = STATISTICS_DIR
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename if not provided
    if args.output_file is None:
        if args.threshold_method == 'dynamic':
            method_str = f"mean_var_act_dynamic_max{args.max_pruned_experts_per_layer}"
        elif args.threshold_method == 'absolute':
            if args.mean_act_threshold is not None and args.var_act_threshold is not None:
                method_str = f"mean_var_act_abs{args.mean_act_threshold:.2f}_{args.var_act_threshold:.2f}_max{args.max_pruned_experts_per_layer}"
            else:
                method_str = f"mean_var_act_abs_p{args.mean_act_percentile:.1f}_{args.var_act_percentile:.1f}_max{args.max_pruned_experts_per_layer}"
        elif args.threshold_method == 'std':
            if args.mean_act_threshold is not None and args.var_act_threshold is not None:
                method_str = f"mean_var_act_std{args.mean_act_threshold:.2f}_{args.var_act_threshold:.2f}_max{args.max_pruned_experts_per_layer}"
            else:
                method_str = f"mean_var_act_std_p{args.mean_act_percentile:.1f}_{args.var_act_percentile:.1f}_max{args.max_pruned_experts_per_layer}"
        else:  # percentile
            method_str = f"mean_var_act_p{args.mean_act_percentile:.1f}_{args.var_act_percentile:.1f}_max{args.max_pruned_experts_per_layer}"
        filename = f"experts_to_prune_{method_str}.json"
        output_file = os.path.join(output_dir, filename)
    else:
        output_file = args.output_file
    
    # Save experts to prune
    # Convert layer keys to strings for JSON
    experts_to_prune_str = {str(k): v for k, v in experts_to_prune.items()}
    with open(output_file, 'w') as f:
        json.dump(experts_to_prune_str, f, indent=2)
    print(f"Saved experts to prune to: {output_file}")
    
    # Save statistics if requested
    if args.save_stats:
        stats_file = output_file.replace('experts_to_prune', 'stats').replace('.json', '_stats.json')
        # Convert numpy arrays to lists for JSON serialization
        stats_json = {}
        for layer_id, layer_stats in stats.items():
            stats_json[str(layer_id)] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in layer_stats.items()
            }
        with open(stats_file, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"Saved statistics to: {stats_file}")
    
    return output_file


def main():
    """Main function."""
    args = parse_args()
    
    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    # Set CUDA visible devices if specified
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    
    # Determine device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",   # distributes layers across all visible GPUs
    )
    model.eval()

    # Input tensors go to the device that hosts the embedding layer
    device = next(model.parameters()).device
    print(f"Model loaded with device_map=auto; input device: {device}")
    
    # Load prompts
    prompts = load_prompts(args)
    
    # Profile model
    router_logits, expert_activations = profile_model(model, tokenizer, prompts, device, args)
    
    # Compute statistics
    stats = compute_statistics(router_logits, expert_activations, top_k=4)
    
    # Determine experts to prune
    experts_to_prune = determine_experts_to_prune(stats, router_logits, args)
    
    # Save results
    output_file = save_results(experts_to_prune, stats, router_logits, args)
    
    print("\n" + "="*60)
    print("Done! Profiling and pruning analysis complete.")
    print("="*60)
    print(f"Output file: {output_file}")
    print(f"Total experts to prune: {sum(len(v) for v in experts_to_prune.values())}")
    print(f"Layers affected: {len(experts_to_prune)}")


if __name__ == '__main__':
    main()

