"""
Differentiable Mask Module for DiEP
Implements learnable binary masks with continuous relaxation for gradient-based pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class GumbelSoftmaxMask(nn.Module):
    """
    Differentiable binary mask using Gumbel-Softmax trick.
    Allows gradient flow through discrete pruning decisions.
    """
    
    def __init__(
        self,
        num_experts: int,
        initial_temp: float = 5.0,
        min_temp: float = 0.1,
        anneal_rate: float = 0.95,
        use_straight_through: bool = True
    ):
        """
        Args:
            num_experts: Number of experts to generate masks for
            initial_temp: Initial temperature for Gumbel-Softmax
            min_temp: Minimum temperature (after annealing)
            anneal_rate: Temperature decay rate per step
            use_straight_through: Use straight-through estimator
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.use_straight_through = use_straight_through
        
        # Learnable logits for each expert (keep/prune decision)
        # Shape: [num_experts, 2] where [:, 0] = prune, [:, 1] = keep
        self.logits = nn.Parameter(torch.zeros(num_experts, 2))
        
        # Initialize with bias toward keeping experts
        nn.init.constant_(self.logits[:, 1], 2.0)  # Favor "keep"
        nn.init.constant_(self.logits[:, 0], -2.0)  # Disfavor "prune"
        
        # Temperature parameter
        self.register_buffer('temperature', torch.tensor(initial_temp))
        
    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, device=self.logits.device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature):
        """Draw a sample from the Gumbel-Softmax distribution"""
        gumbel_noise = self.sample_gumbel(logits.shape)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    def forward(self, hard: bool = False) -> torch.Tensor:
        """
        Generate differentiable masks.
        
        Args:
            hard: If True, return hard binary masks (for evaluation)
            
        Returns:
            masks: Tensor of shape [num_experts] with values in [0, 1]
        """
        if hard or not self.training:
            # Hard decision: argmax
            masks = (self.logits[:, 1] > self.logits[:, 0]).float()
            return masks
        
        # Gumbel-Softmax sampling
        y_soft = self.gumbel_softmax_sample(self.logits, self.temperature)
        
        if self.use_straight_through:
            # Straight-through estimator
            # Forward: hard decisions, Backward: soft gradients
            y_hard = torch.zeros_like(y_soft)
            y_hard[torch.arange(self.num_experts), y_soft.argmax(dim=1)] = 1.0
            masks = (y_hard - y_soft).detach() + y_soft
            # Extract "keep" probability
            masks = masks[:, 1]
        else:
            # Soft masks (continuous)
            masks = y_soft[:, 1]
        
        return masks
    
    def anneal_temperature(self):
        """Decrease temperature for annealing."""
        self.temperature = torch.clamp(
            self.temperature * self.anneal_rate,
            min=self.min_temp
        )
    
    def get_keep_probabilities(self) -> torch.Tensor:
        """Get probability of keeping each expert (without sampling)."""
        probs = F.softmax(self.logits / self.temperature, dim=-1)
        return probs[:, 1]  # Probability of "keep"
    
    def get_expected_sparsity(self) -> float:
        """Calculate expected sparsity based on current logits."""
        keep_probs = self.get_keep_probabilities()
        return 1.0 - keep_probs.mean().item()


class LayerWiseMaskController(nn.Module):
    """
    Controls masks for all MoE layers with layer-wise importance learning.
    Implements non-uniform, adaptive pruning across layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        experts_per_layer: List[int],
        target_sparsity: float = 0.5,
        initial_temp: float = 5.0,
        min_temp: float = 0.1,
        anneal_rate: float = 0.95
    ):
        """
        Args:
            num_layers: Number of MoE layers
            experts_per_layer: List of expert counts for each layer
            target_sparsity: Target overall sparsity (0.5 = prune 50% of experts)
            initial_temp: Initial Gumbel-Softmax temperature
            min_temp: Minimum temperature
            anneal_rate: Temperature annealing rate
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.target_sparsity = target_sparsity
        self.total_experts = sum(experts_per_layer)
        
        # Create mask generators for each layer
        self.layer_masks = nn.ModuleList([
            GumbelSoftmaxMask(
                num_experts=num_experts,
                initial_temp=initial_temp,
                min_temp=min_temp,
                anneal_rate=anneal_rate
            )
            for num_experts in experts_per_layer
        ])
        
        # Layer importance scores (learnable)
        # Higher score = more important layer = less pruning
        self.layer_importance = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, hard: bool = False) -> Dict[int, torch.Tensor]:
        """
        Generate masks for all layers.
        
        Args:
            hard: Generate hard binary masks
            
        Returns:
            Dict mapping layer_idx -> mask tensor
        """
        masks = {}
        for layer_idx, mask_gen in enumerate(self.layer_masks):
            masks[layer_idx] = mask_gen(hard=hard)
        return masks
    
    def get_layer_sparsity_targets(self) -> torch.Tensor:
        """
        Compute per-layer sparsity targets based on layer importance.
        More important layers get lower sparsity (fewer experts pruned).
        """
        # Normalize importance scores to [0, 1]
        importance_normalized = torch.sigmoid(self.layer_importance)
        
        # Compute per-layer sparsity: inverse of importance
        # Important layers (high score) -> low sparsity
        # Less important layers (low score) -> high sparsity
        
        # Total experts to prune
        target_pruned = int(self.target_sparsity * self.total_experts)
        
        # Distribute pruning inversely proportional to importance
        importance_inv = 1.0 - importance_normalized
        total_inv_importance = importance_inv.sum()
        
        layer_sparsities = []
        for layer_idx in range(self.num_layers):
            num_experts = self.experts_per_layer[layer_idx]
            # Portion of pruning allocated to this layer
            layer_portion = importance_inv[layer_idx] / total_inv_importance
            layer_pruned = layer_portion * target_pruned
            layer_sparsity = layer_pruned / num_experts
            layer_sparsities.append(torch.clamp(layer_sparsity, 0.0, 1.0))
        
        return torch.stack(layer_sparsities)
    
    def compute_sparsity_loss(self, masks: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss to enforce target sparsity constraint.
        
        Args:
            masks: Current masks from forward pass
            
        Returns:
            Sparsity loss (L1 distance from target)
        """
        # Overall sparsity
        total_kept = sum(mask.sum() for mask in masks.values())
        current_sparsity = 1.0 - (total_kept / self.total_experts)
        overall_loss = torch.abs(current_sparsity - self.target_sparsity)
        
        # Per-layer sparsity (optional, for non-uniform pruning)
        layer_targets = self.get_layer_sparsity_targets()
        layer_loss = 0.0
        
        for layer_idx, mask in masks.items():
            num_experts = self.experts_per_layer[layer_idx]
            current_layer_sparsity = 1.0 - (mask.sum() / num_experts)
            target_layer_sparsity = layer_targets[layer_idx]
            layer_loss += torch.abs(current_layer_sparsity - target_layer_sparsity)
        
        layer_loss = layer_loss / self.num_layers
        
        # Combine losses
        total_loss = overall_loss + 0.1 * layer_loss
        
        return total_loss
    
    def anneal_temperature(self):
        """Anneal temperature for all layer masks."""
        for mask_gen in self.layer_masks:
            mask_gen.anneal_temperature()
    
    def get_current_temperature(self) -> float:
        """Get current temperature (assumes all layers have same temp)."""
        return self.layer_masks[0].temperature.item()
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about current pruning configuration."""
        masks = self.forward(hard=True)
        
        stats = {
            'overall_sparsity': 0.0,
            'target_sparsity': self.target_sparsity,
            'layer_stats': [],
            'layer_importance': self.layer_importance.detach().cpu().tolist(),
            'temperature': self.get_current_temperature()
        }
        
        total_kept = 0
        for layer_idx, mask in masks.items():
            num_experts = self.experts_per_layer[layer_idx]
            num_kept = mask.sum().item()
            total_kept += num_kept
            
            stats['layer_stats'].append({
                'layer_idx': layer_idx,
                'num_experts': num_experts,
                'num_active': int(num_kept),
                'sparsity': 1.0 - (num_kept / num_experts)
            })
        
        stats['overall_sparsity'] = 1.0 - (total_kept / self.total_experts)
        stats['total_active_experts'] = int(total_kept)
        stats['total_experts'] = self.total_experts
        
        return stats
    
    def initialize_from_importance(self, importance_scores: Dict[int, torch.Tensor]):
        """
        Initialize mask logits based on pre-computed expert importance scores.
        
        Args:
            importance_scores: Dict mapping layer_idx -> importance tensor [num_experts]
        """
        for layer_idx, scores in importance_scores.items():
            if layer_idx >= len(self.layer_masks):
                continue
            
            mask_gen = self.layer_masks[layer_idx]
            
            # Normalize scores to [-2, 2] range
            scores_norm = (scores - scores.mean()) / (scores.std() + 1e-8)
            scores_norm = torch.clamp(scores_norm * 2, -2, 2)
            
            # Set logits: higher importance -> higher "keep" logit
            with torch.no_grad():
                mask_gen.logits[:, 1] = scores_norm  # Keep
                mask_gen.logits[:, 0] = -scores_norm  # Prune


class UniformMaskController(nn.Module):
    """
    Simpler baseline: uniform pruning across all layers.
    Useful for ablation studies comparing uniform vs non-uniform pruning.
    """
    
    def __init__(
        self,
        num_layers: int,
        experts_per_layer: List[int],
        target_sparsity: float = 0.5,
        initial_temp: float = 5.0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.target_sparsity = target_sparsity
        self.total_experts = sum(experts_per_layer)
        
        # All experts share same mask distribution
        self.layer_masks = nn.ModuleList([
            GumbelSoftmaxMask(
                num_experts=num_experts,
                initial_temp=initial_temp
            )
            for num_experts in experts_per_layer
        ])
    
    def forward(self, hard: bool = False) -> Dict[int, torch.Tensor]:
        masks = {}
        for layer_idx, mask_gen in enumerate(self.layer_masks):
            masks[layer_idx] = mask_gen(hard=hard)
        return masks
    
    def compute_sparsity_loss(self, masks: Dict[int, torch.Tensor]) -> torch.Tensor:
        total_kept = sum(mask.sum() for mask in masks.values())
        current_sparsity = 1.0 - (total_kept / self.total_experts)
        return torch.abs(current_sparsity - self.target_sparsity)
    
    def anneal_temperature(self):
        for mask_gen in self.layer_masks:
            mask_gen.anneal_temperature()


def create_mask_controller(
    moe_layer_indices: List[int],
    experts_per_layer: List[int],
    target_sparsity: float = 0.5,
    use_layerwise: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate mask controller.
    
    Args:
        moe_layer_indices: Indices of MoE layers
        experts_per_layer: Number of experts in each layer
        target_sparsity: Target overall sparsity
        use_layerwise: Use layer-wise adaptive pruning (vs uniform)
        
    Returns:
        Mask controller module
    """
    num_layers = len(moe_layer_indices)
    
    if use_layerwise:
        return LayerWiseMaskController(
            num_layers=num_layers,
            experts_per_layer=experts_per_layer,
            target_sparsity=target_sparsity,
            **kwargs
        )
    else:
        return UniformMaskController(
            num_layers=num_layers,
            experts_per_layer=experts_per_layer,
            target_sparsity=target_sparsity,
            **kwargs
        )