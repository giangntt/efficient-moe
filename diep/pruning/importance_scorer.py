"""
Expert Importance Scoring for DiEP
Implements various metrics to evaluate expert importance for pruning decisions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class ExpertImportanceScorer:
    """
    Computes importance scores for experts using various metrics.
    Higher scores indicate more important experts (should be kept).
    """
    
    def __init__(self, prunable_model, device: str = "cuda"):
        """
        Args:
            prunable_model: PrunableQwenMoE model instance
            device: Device for computations
        """
        self.model = prunable_model
        self.device = device
        self.moe_layers = prunable_model.prunable_layers
        
        # Storage for accumulated statistics
        self.router_stats = defaultdict(lambda: defaultdict(float))
        self.gradient_stats = defaultdict(lambda: defaultdict(float))
        self.activation_stats = defaultdict(lambda: defaultdict(float))
        
    def reset_statistics(self):
        """Reset all accumulated statistics."""
        self.router_stats.clear()
        self.gradient_stats.clear()
        self.activation_stats.clear()
    
    def compute_router_based_importance(
        self,
        dataloader,
        num_batches: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance based on router selection frequency.
        Experts selected more often are considered more important.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to process (None = all)
            
        Returns:
            Dict mapping layer_idx -> importance scores [num_experts]
        """
        print("Computing router-based importance scores...")
        self.model.enable_usage_tracking()
        
        # Accumulate router selections
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Router stats")):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                _ = self.model(**batch)
        
        self.model.disable_usage_tracking()
        
        # Collect usage statistics
        model_stats = self.model.get_statistics()
        importance_scores = {}
        
        for layer_idx, layer_stats in model_stats.items():
            usage_counts = layer_stats['expert_usage_count'].float()
            # Normalize to [0, 1]
            if usage_counts.sum() > 0:
                importance = usage_counts / usage_counts.sum()
            else:
                importance = torch.ones_like(usage_counts) / len(usage_counts)
            
            importance_scores[layer_idx] = importance
        
        return importance_scores
    
    def compute_gradient_based_importance(
        self,
        dataloader,
        num_batches: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance based on gradient magnitude (Taylor expansion).
        Importance = |weight * gradient| accumulated over calibration set.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to process
            
        Returns:
            Dict mapping layer_idx -> importance scores [num_experts]
        """
        print("Computing gradient-based importance scores...")
        
        # Storage for gradient accumulation
        importance_accum = {
            layer_idx: torch.zeros(layer.num_experts, device=self.device)
            for layer_idx, layer in self.moe_layers.items()
        }
        
        self.model.model.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Gradient stats")):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with loss
            outputs = self.model(**batch, labels=batch.get('input_ids'))
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Accumulate |weight * gradient| for each expert
            for layer_idx, layer in self.moe_layers.items():
                experts = layer.experts
                
                for expert_idx in range(layer.num_experts):
                    expert = experts[expert_idx]
                    expert_importance = 0.0
                    
                    # Accumulate over all parameters in the expert
                    for param in expert.parameters():
                        if param.grad is not None:
                            # Taylor expansion: importance ~ |w * grad|
                            importance_contrib = (param * param.grad).abs().sum()
                            expert_importance += importance_contrib.item()
                    
                    importance_accum[layer_idx][expert_idx] += expert_importance
            
            # Clear gradients
            self.model.model.zero_grad()
        
        # Normalize importance scores per layer
        importance_scores = {}
        for layer_idx, scores in importance_accum.items():
            if scores.sum() > 0:
                importance_scores[layer_idx] = scores / scores.sum()
            else:
                importance_scores[layer_idx] = torch.ones_like(scores) / len(scores)
        
        self.model.model.eval()
        return importance_scores
    
    def compute_activation_magnitude_importance(
        self,
        dataloader,
        num_batches: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance based on expert output magnitude.
        Experts with larger outputs are considered more important.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to process
            
        Returns:
            Dict mapping layer_idx -> importance scores [num_experts]
        """
        print("Computing activation-based importance scores...")
        
        # Storage for activation accumulation
        activation_accum = {
            layer_idx: torch.zeros(layer.num_experts, device=self.device)
            for layer_idx, layer in self.moe_layers.items()
        }
        
        # Hook to capture expert outputs
        hooks = []
        expert_outputs = defaultdict(list)
        
        def create_hook(layer_idx, expert_idx):
            def hook(module, input, output):
                # Store mean absolute activation
                expert_outputs[(layer_idx, expert_idx)].append(
                    output.abs().mean().item()
                )
            return hook
        
        # Register hooks
        for layer_idx, layer in self.moe_layers.items():
            for expert_idx, expert in enumerate(layer.experts):
                hook = expert.register_forward_hook(
                    create_hook(layer_idx, expert_idx)
                )
                hooks.append(hook)
        
        # Run forward passes
        self.model.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Activation stats")):
                if num_batches and batch_idx >= num_batches:
                    break
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                _ = self.model(**batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Aggregate statistics
        for (layer_idx, expert_idx), activations in expert_outputs.items():
            if activations:
                activation_accum[layer_idx][expert_idx] = np.mean(activations)
        
        # Normalize per layer
        importance_scores = {}
        for layer_idx, scores in activation_accum.items():
            if scores.sum() > 0:
                importance_scores[layer_idx] = scores / scores.sum()
            else:
                importance_scores[layer_idx] = torch.ones_like(scores) / len(scores)
        
        return importance_scores
    
    def compute_fisher_information_importance(
        self,
        dataloader,
        num_batches: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance using Fisher Information approximation.
        Importance ~ E[gradient^2] over the calibration set.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to process
            
        Returns:
            Dict mapping layer_idx -> importance scores [num_experts]
        """
        print("Computing Fisher Information importance scores...")
        
        # Storage for Fisher information accumulation
        fisher_accum = {
            layer_idx: torch.zeros(layer.num_experts, device=self.device)
            for layer_idx, layer in self.moe_layers.items()
        }
        
        self.model.model.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Fisher stats")):
            if num_batches and batch_idx >= num_batches:
                break
            
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch, labels=batch.get('input_ids'))
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for layer_idx, layer in self.moe_layers.items():
                experts = layer.experts
                
                for expert_idx in range(layer.num_experts):
                    expert = experts[expert_idx]
                    expert_fisher = 0.0
                    
                    for param in expert.parameters():
                        if param.grad is not None:
                            # Fisher ~ E[grad^2]
                            fisher_contrib = (param.grad ** 2).sum()
                            expert_fisher += fisher_contrib.item()
                    
                    fisher_accum[layer_idx][expert_idx] += expert_fisher
            
            self.model.model.zero_grad()
        
        # Normalize per layer
        importance_scores = {}
        for layer_idx, scores in fisher_accum.items():
            if scores.sum() > 0:
                importance_scores[layer_idx] = scores / scores.sum()
            else:
                importance_scores[layer_idx] = torch.ones_like(scores) / len(scores)
        
        self.model.model.eval()
        return importance_scores
    
    def compute_combined_importance(
        self,
        dataloader,
        num_batches: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance using a weighted combination of multiple metrics.
        
        Args:
            dataloader: DataLoader with calibration data
            num_batches: Number of batches to process
            weights: Weights for each metric (router, gradient, activation, fisher)
            
        Returns:
            Dict mapping layer_idx -> importance scores [num_experts]
        """
        if weights is None:
            weights = {
                'router': 0.3,
                'gradient': 0.3,
                'activation': 0.2,
                'fisher': 0.2
            }
        
        print(f"Computing combined importance with weights: {weights}")
        
        # Compute individual importance scores
        importance_dict = {}
        
        if weights.get('router', 0) > 0:
            importance_dict['router'] = self.compute_router_based_importance(
                dataloader, num_batches
            )
        
        if weights.get('gradient', 0) > 0:
            importance_dict['gradient'] = self.compute_gradient_based_importance(
                dataloader, num_batches
            )
        
        if weights.get('activation', 0) > 0:
            importance_dict['activation'] = self.compute_activation_magnitude_importance(
                dataloader, num_batches
            )
        
        if weights.get('fisher', 0) > 0:
            importance_dict['fisher'] = self.compute_fisher_information_importance(
                dataloader, num_batches
            )
        
        # Combine scores
        combined_scores = {}
        layer_indices = list(self.moe_layers.keys())
        
        for layer_idx in layer_indices:
            combined = torch.zeros(
                self.moe_layers[layer_idx].num_experts,
                device=self.device
            )
            
            for metric_name, metric_scores in importance_dict.items():
                weight = weights.get(metric_name, 0)
                if weight > 0 and layer_idx in metric_scores:
                    combined += weight * metric_scores[layer_idx]
            
            # Normalize
            if combined.sum() > 0:
                combined = combined / combined.sum()
            else:
                combined = torch.ones_like(combined) / len(combined)
            
            combined_scores[layer_idx] = combined
        
        return combined_scores
    
    def rank_experts_by_importance(
        self,
        importance_scores: Dict[int, torch.Tensor],
        top_k: Optional[int] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Rank experts within each layer by importance.
        
        Args:
            importance_scores: Importance scores per layer
            top_k: Return only top-k experts (None = all)
            
        Returns:
            Dict mapping layer_idx -> list of (expert_idx, score) tuples
        """
        rankings = {}
        
        for layer_idx, scores in importance_scores.items():
            # Sort by importance (descending)
            sorted_indices = torch.argsort(scores, descending=True)
            
            ranked_experts = [
                (idx.item(), scores[idx].item())
                for idx in sorted_indices
            ]
            
            if top_k:
                ranked_experts = ranked_experts[:top_k]
            
            rankings[layer_idx] = ranked_experts
        
        return rankings
    
    def get_pruning_mask_from_importance(
        self,
        importance_scores: Dict[int, torch.Tensor],
        target_sparsity: float = 0.5,
        uniform: bool = False
    ) -> Dict[int, torch.Tensor]:
        """
        Convert importance scores to binary pruning masks.
        
        Args:
            importance_scores: Importance scores per layer
            target_sparsity: Target overall sparsity
            uniform: If True, use uniform sparsity across layers
            
        Returns:
            Dict mapping layer_idx -> binary mask [num_experts]
        """
        masks = {}
        
        if uniform:
            # Uniform sparsity across all layers
            for layer_idx, scores in importance_scores.items():
                num_experts = len(scores)
                num_to_keep = int(num_experts * (1 - target_sparsity))
                num_to_keep = max(1, num_to_keep)  # Keep at least 1 expert
                
                # Keep top-k important experts
                top_k_indices = torch.topk(scores, num_to_keep).indices
                mask = torch.zeros_like(scores)
                mask[top_k_indices] = 1.0
                masks[layer_idx] = mask
        else:
            # Non-uniform: prune globally based on importance
            # Flatten all scores
            all_scores = []
            score_to_layer_expert = []
            
            for layer_idx, scores in importance_scores.items():
                for expert_idx, score in enumerate(scores):
                    all_scores.append(score.item())
                    score_to_layer_expert.append((layer_idx, expert_idx))
            
            all_scores = np.array(all_scores)
            total_experts = len(all_scores)
            num_to_keep = int(total_experts * (1 - target_sparsity))
            
            # Find global threshold
            threshold = np.partition(all_scores, -num_to_keep)[-num_to_keep]
            
            # Create masks
            for layer_idx, scores in importance_scores.items():
                mask = (scores >= threshold).float()
                # Ensure at least one expert per layer
                if mask.sum() == 0:
                    best_expert = scores.argmax()
                    mask[best_expert] = 1.0
                masks[layer_idx] = mask
        
        return masks


def visualize_importance_scores(
    importance_scores: Dict[int, torch.Tensor],
    title: str = "Expert Importance Scores"
):
    """
    Print a visualization of expert importance scores.
    
    Args:
        importance_scores: Importance scores per layer
        title: Title for the visualization
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    for layer_idx in sorted(importance_scores.keys()):
        scores = importance_scores[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Num experts: {len(scores)}")
        print(f"  Mean importance: {scores.mean().item():.4f}")
        print(f"  Std importance: {scores.std().item():.4f}")
        print(f"  Min importance: {scores.min().item():.4f}")
        print(f"  Max importance: {scores.max().item():.4f}")
        
        # Show top-3 and bottom-3 experts
        top_3_idx = torch.topk(scores, min(3, len(scores))).indices
        bottom_3_idx = torch.topk(scores, min(3, len(scores)), largest=False).indices
        
        print(f"  Top-3 experts: {top_3_idx.tolist()} "
              f"(scores: {[f'{scores[i].item():.4f}' for i in top_3_idx]})")
        print(f"  Bottom-3 experts: {bottom_3_idx.tolist()} "
              f"(scores: {[f'{scores[i].item():.4f}' for i in bottom_3_idx]})")