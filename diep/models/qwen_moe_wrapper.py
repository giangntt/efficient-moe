"""
Qwen MoE Wrapper for DiEP (Differentiable Expert Pruning)
Wraps Qwen1.5-MoE layers to support differentiable expert pruning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeSparseMoeBlock,
    Qwen2MoeConfig
)


class PrunableQwenMoEBlock(nn.Module):
    """
    Wraps a Qwen2MoeSparseMoeBlock to support differentiable expert pruning.
    Applies learned binary masks to experts before/after routing.
    """
    
    def __init__(
        self,
        original_moe_block: Qwen2MoeSparseMoeBlock,
        layer_idx: int,
        num_experts: int,
        pruning_masks: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.original_block = original_moe_block
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        
        # Store references to the experts
        self.experts = original_moe_block.experts
        self.gate = original_moe_block.gate
        
        # Configuration
        self.config = original_moe_block.config
        self.top_k = self.config.num_experts_per_tok
        self.norm_topk_prob = self.config.norm_topk_prob
        
        # Pruning masks (will be set externally)
        # Shape: [num_experts], values in [0, 1]
        if pruning_masks is not None:
            self.register_buffer('pruning_masks', pruning_masks)
        else:
            self.register_buffer('pruning_masks', torch.ones(num_experts))
        
        # Track expert usage for analysis
        self.register_buffer(
            'expert_usage_count',
            torch.zeros(num_experts, dtype=torch.long)
        )
        self.track_usage = False
    
    def set_pruning_masks(self, masks: torch.Tensor):
        """Set the pruning masks for this layer's experts."""
        assert masks.shape[0] == self.num_experts, \
            f"Expected {self.num_experts} masks, got {masks.shape[0]}"
        self.pruning_masks = masks.to(self.pruning_masks.device)
    
    def get_active_experts(self) -> torch.Tensor:
        """Returns indices of active (non-pruned) experts."""
        # Threshold at 0.5 for soft masks, or use argmax for hard masks
        return (self.pruning_masks > 0.5).nonzero(as_tuple=True)[0]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with expert pruning.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            output: Processed hidden states
            router_logits: Router logits for auxiliary loss (optional)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Router logits: [batch_size * seq_len, num_experts]
        router_logits = self.gate(hidden_states)
        
        # Apply pruning masks to router logits
        # Masked experts get very negative logits (effectively pruned)
        masked_router_logits = router_logits + (1 - self.pruning_masks) * (-1e9)
        
        # Top-k routing with masked logits
        routing_weights = torch.nn.functional.softmax(masked_router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        
        # Normalize routing weights if configured
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Track expert usage
        if self.track_usage:
            unique_experts = torch.unique(selected_experts)
            for expert_idx in unique_experts:
                self.expert_usage_count[expert_idx] += (selected_experts == expert_idx).sum()
        
        # Prepare for expert computation
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # One-hot encoding of expert indices
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        
        # Compute outputs for each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            
            # Get tokens routed to this expert
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
            
            # Apply expert mask (soft pruning during training)
            expert_multiplier = self.pruning_masks[expert_idx]
            
            # Forward through expert
            top_x_list = top_x.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx, None]
            
            # Apply pruning mask
            current_hidden_states = current_hidden_states * expert_multiplier
            
            # Accumulate results
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        
        return final_hidden_states, router_logits
    
    def get_expert_statistics(self) -> Dict[str, torch.Tensor]:
        """Return statistics about expert usage and pruning."""
        return {
            'layer_idx': self.layer_idx,
            'pruning_masks': self.pruning_masks.clone(),
            'active_experts': self.get_active_experts(),
            'num_active_experts': (self.pruning_masks > 0.5).sum().item(),
            'expert_usage_count': self.expert_usage_count.clone(),
            'sparsity': 1.0 - (self.pruning_masks.mean().item())
        }
    
    def reset_usage_tracking(self):
        """Reset expert usage counters."""
        self.expert_usage_count.zero_()


class PrunableQwenMoE(nn.Module):
    """
    Wraps entire Qwen MoE model to support layer-wise differentiable expert pruning.
    """
    
    def __init__(self, model, moe_layer_indices: List[int]):
        """
        Args:
            model: Pretrained Qwen MoE model
            moe_layer_indices: List of layer indices that contain MoE blocks
        """
        super().__init__()
        self.model = model
        self.moe_layer_indices = moe_layer_indices
        self.prunable_layers: Dict[int, PrunableQwenMoEBlock] = {}
        
        # Wrap MoE layers
        self._wrap_moe_layers()
        
    def _wrap_moe_layers(self):
        """Replace original MoE blocks with prunable versions."""
        for layer_idx in self.moe_layer_indices:
            # Access the MoE block in the layer
            # Path: model.model.layers[layer_idx].mlp (if MoE) or .block_sparse_moe
            layer = self.model.model.layers[layer_idx]
            
            # Check if this layer has MoE
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                original_moe = layer.mlp
                num_experts = original_moe.config.num_experts
                
                # Create prunable wrapper
                prunable_moe = PrunableQwenMoEBlock(
                    original_moe_block=original_moe,
                    layer_idx=layer_idx,
                    num_experts=num_experts
                )
                
                # Replace the original block
                layer.mlp = prunable_moe
                self.prunable_layers[layer_idx] = prunable_moe
                
                print(f"Wrapped layer {layer_idx} with {num_experts} experts")
    
    def set_all_masks(self, masks_dict: Dict[int, torch.Tensor]):
        """
        Set pruning masks for all MoE layers.
        
        Args:
            masks_dict: Dictionary mapping layer_idx -> mask tensor
        """
        for layer_idx, masks in masks_dict.items():
            if layer_idx in self.prunable_layers:
                self.prunable_layers[layer_idx].set_pruning_masks(masks)
    
    def get_all_masks(self) -> Dict[int, torch.Tensor]:
        """Get current pruning masks for all layers."""
        return {
            layer_idx: layer.pruning_masks.clone()
            for layer_idx, layer in self.prunable_layers.items()
        }
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def enable_usage_tracking(self):
        """Enable expert usage tracking for analysis."""
        for layer in self.prunable_layers.values():
            layer.track_usage = True
            layer.reset_usage_tracking()
    
    def disable_usage_tracking(self):
        """Disable expert usage tracking."""
        for layer in self.prunable_layers.values():
            layer.track_usage = False
    
    def get_statistics(self) -> Dict[int, Dict]:
        """Get pruning and usage statistics for all layers."""
        return {
            layer_idx: layer.get_expert_statistics()
            for layer_idx, layer in self.prunable_layers.items()
        }
    
    def get_total_sparsity(self) -> float:
        """Calculate overall expert sparsity across all MoE layers."""
        total_experts = 0
        total_pruned = 0
        
        for layer in self.prunable_layers.values():
            masks = layer.pruning_masks
            total_experts += len(masks)
            total_pruned += (masks < 0.5).sum().item()
        
        return total_pruned / total_experts if total_experts > 0 else 0.0
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio (remaining parameters / original parameters)."""
        return 1.0 - self.get_total_sparsity()
    
    def finalize_pruning(self, threshold: float = 0.5):
        """
        Convert soft masks to hard binary masks and optionally remove pruned experts.
        
        Args:
            threshold: Threshold for binarizing soft masks
        """
        for layer_idx, layer in self.prunable_layers.items():
            # Binarize masks
            hard_masks = (layer.pruning_masks > threshold).float()
            layer.set_pruning_masks(hard_masks)
            
            active_experts = layer.get_active_experts()
            print(f"Layer {layer_idx}: {len(active_experts)}/{layer.num_experts} active experts")


def create_prunable_qwen_model(model_name_or_path: str, device: str = "cuda"):
    """
    Load Qwen MoE model and wrap with prunable layers.
    
    Args:
        model_name_or_path: HuggingFace model identifier or path
        device: Device to load model on
        
    Returns:
        PrunableQwenMoE: Wrapped model ready for pruning
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # Identify MoE layers
    moe_layer_indices = []
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            moe_layer_indices.append(idx)
    
    print(f"Found {len(moe_layer_indices)} MoE layers: {moe_layer_indices}")
    
    # Wrap model
    prunable_model = PrunableQwenMoE(model, moe_layer_indices)
    
    return prunable_model, tokenizer, moe_layer_indices