"""
Utilities for registering and managing hooks to capture expert activations.
"""
import torch
from typing import Dict, List, Tuple, Any


class ExpertActivationHook:
    """Manages hooks for capturing expert activations."""
    
    def __init__(self):
        self.expert_activations: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self.hook_handles: List[Any] = []
    
    def save_expert_activation(self, module, input, output):
        """Hook function to save expert activations."""
        layer_id = getattr(module, 'layer_id', None)
        expert_id = getattr(module, 'expert_id', None)
        if layer_id is not None and expert_id is not None:
            key = (layer_id, expert_id)
            self.expert_activations.setdefault(key, []).append(output.detach().cpu())
    
    def register_hooks(self, model):
        """Register forward hooks on all expert down_proj layers."""
        # Handle both model.model.layers (wrapped) and model.layers (unwrapped)
        actual_model = model.model if hasattr(model, 'model') else model
        for layer_idx, layer in enumerate(actual_model.layers):
            moe_block = layer.mlp
            moe_block.layer_id = layer_idx

            if hasattr(moe_block, 'experts'):
                for expert_idx, expert in enumerate(moe_block.experts):
                    down_proj_layer = expert.down_proj
                    down_proj_layer.layer_id = layer_idx
                    down_proj_layer.expert_id = expert_idx
                    handle = down_proj_layer.register_forward_hook(self.save_expert_activation)
                    self.hook_handles.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def clear_activations(self):
        """Clear stored activations."""
        self.expert_activations.clear()


def create_hook_manager() -> ExpertActivationHook:
    """Create a new hook manager instance."""
    return ExpertActivationHook()


