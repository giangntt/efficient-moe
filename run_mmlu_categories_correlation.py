#!/usr/bin/env python3
"""
Script to run forward passes for each MMLU category, calculate stats,
and plot router-expert activation correlations for all 4 categories in one plot.
Creates two plots: one for Spearman correlation and one for Pearson correlation.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = "outputs/plots"
from utils.data_utils import MMLU_CATEGORIES, prepare_mmlu_prompts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hook_utils import ExpertActivationHook
from utils.router_utils import collect_router_logits
from utils.analysis_utils import compute_all_stats, compute_correlations_by_layer


def process_category(model, tokenizer, category_name, device, max_samples_per_subject=5):
    """Process one MMLU category: load samples, run forward passes, compute stats."""
    print(f"\n{'='*60}")
    print(f"Processing category: {category_name}")
    print(f"{'='*60}")
    
    # Prepare prompts using utility function
    prompts = prepare_mmlu_prompts(topic=category_name, max_samples_per_subject=max_samples_per_subject)
    print(f"Loaded {len(prompts)} prompts")
    
    # Register hooks and collect router logits
    hook_manager = ExpertActivationHook()
    hook_manager.register_hooks(model)
    
    # Collect router logits and final logits
    result = collect_router_logits(model, tokenizer, prompts, device, output_final_logits=True)
    router_logits = result['router_logits']
    
    # Get expert activations and clear hooks
    expert_activations = hook_manager.expert_activations
    hook_manager.clear_hooks()
    
    # Compute stats
    print(f"Computing stats for {category_name}...")
    stats = compute_all_stats(
        router_logits=router_logits,
        expert_activations=expert_activations,
        top_k=4,
        device='cpu'
    )
    
    return stats


def plot_correlations(all_correlations, method='spearman', output_dir=None):
    """Plot correlations for all categories in one figure."""
    if output_dir is None:
        output_dir = PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Get common layers across all categories
    all_layers = set()
    for corrs in all_correlations.values():
        all_layers.update(corrs.keys())
    layers = sorted(all_layers)
    
    # Prepare data for plotting
    category_names = list(all_correlations.keys())
    colors = ['C0', 'C1', 'C2', 'C3']  # Blue, orange, green, red
    markers = ['o', 's', 'D', '^']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    for i, category_name in enumerate(category_names):
        corrs = all_correlations[category_name]
        correlations = [corrs.get(layer, np.nan) for layer in layers]
        
        plt.plot(
            layers,
            correlations,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=category_name.replace('_', ' ').title(),
            linewidth=2,
            markersize=6,
            alpha=0.8
        )
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(f'{method.capitalize()} Correlation', fontsize=12)
    plt.title(f'Router-Expert Activation Correlation by Layer ({method.capitalize()})', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'mmlu_router_activation_{method}_all_categories.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved {method.capitalize()} correlation plot to: {output_path}")
    plt.close()


def main():
    # Configuration
    cuda_visible_devices = "1"
    
    # Load model
    print("\nLoading model...")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Process each category
    all_stats = {}
    for category_name in MMLU_CATEGORIES.keys():
        stats = process_category(
            model=model,
            tokenizer=tokenizer,
            category_name=category_name,
            device=device,
            max_samples_per_subject=5
        )
        all_stats[category_name] = stats
    
    # Compute correlations for each category
    print("\n" + "="*60)
    print("Computing correlations...")
    print("="*60)
    
    spearman_correlations = {}
    pearson_correlations = {}
    
    for category_name, stats in all_stats.items():
        print(f"\nComputing correlations for {category_name}...")
        # Use utility function to compute both correlations at once
        spearman_by_layer, pearson_by_layer = compute_correlations_by_layer(stats)
        
        spearman_correlations[category_name] = spearman_by_layer
        pearson_correlations[category_name] = pearson_by_layer
        
        # Print summary
        valid_spearman = [v for v in spearman_by_layer.values() if not np.isnan(v)]
        valid_pearson = [v for v in pearson_by_layer.values() if not np.isnan(v)]
        
        if valid_spearman:
            print(f"  Spearman: mean={np.mean(valid_spearman):.4f}, "
                  f"min={np.min(valid_spearman):.4f}, max={np.max(valid_spearman):.4f}")
        if valid_pearson:
            print(f"  Pearson: mean={np.mean(valid_pearson):.4f}, "
                  f"min={np.min(valid_pearson):.4f}, max={np.max(valid_pearson):.4f}")
    
    # Create plots
    print("\n" + "="*60)
    print("Creating plots...")
    print("="*60)
    
    plot_correlations(spearman_correlations, method='spearman')
    plot_correlations(pearson_correlations, method='pearson')
    
    print("\n" + "="*60)
    print("Done! All processing complete.")
    print("="*60)


if __name__ == '__main__':
    main()

