"""
Utilities for registering and managing hooks to capture expert activations.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any


class ExpertActivationHook:
    """Manages hooks for capturing expert activations."""

    def __init__(self):
        self.expert_activations: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self.hook_handles: List[Any] = []

    def save_expert_activation(self, module, input, output):
        """Hook for individual down_proj modules (classic ModuleList style)."""
        layer_id = getattr(module, 'layer_id', None)
        expert_id = getattr(module, 'expert_id', None)
        if layer_id is not None and expert_id is not None:
            key = (layer_id, expert_id)
            self.expert_activations.setdefault(key, []).append(output.detach().cpu())

    def _make_fused_hook(self, layer_idx):
        """
        Returns a forward hook for fused Qwen3_5MoeExperts modules.

        The hook re-runs the gate+up projection (first half of each expert's
        computation) to capture per-expert intermediate activations without
        modifying the module's forward path.
        """
        hook_store = self

        def hook_fn(module, inputs, output):
            hidden_states, top_k_index, top_k_weights = inputs
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=module.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)  # [E, top_k, T]
                expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()
                for expert_idx_t in expert_hit:
                    eidx = expert_idx_t[0].item()
                    if eidx >= module.num_experts:
                        continue
                    _, token_idx = torch.where(expert_mask[eidx])
                    current_state = hidden_states[token_idx]
                    gate, up = F.linear(
                        current_state, module.gate_up_proj[eidx]
                    ).chunk(2, dim=-1)
                    act = module.act_fn(gate) * up  # [selected_tokens, intermediate_dim]
                    key = (layer_idx, eidx)
                    hook_store.expert_activations.setdefault(key, []).append(
                        act.detach().cpu()
                    )

        return hook_fn

    def register_hooks(self, model):
        """Register forward hooks on all MoE expert layers."""
        for layer_idx, layer in enumerate(model.model.layers):
            moe_block = layer.mlp
            moe_block.layer_id = layer_idx

            if not hasattr(moe_block, 'experts'):
                continue

            experts = moe_block.experts
            # Classic style: experts is a list or nn.ModuleList of individual modules
            try:
                expert_list = list(experts)
                for expert_idx, expert in enumerate(expert_list):
                    down_proj_layer = expert.down_proj
                    down_proj_layer.layer_id = layer_idx
                    down_proj_layer.expert_id = expert_idx
                    handle = down_proj_layer.register_forward_hook(
                        self.save_expert_activation
                    )
                    self.hook_handles.append(handle)
            except TypeError:
                # Fused style (e.g. Qwen3_5MoeExperts): not directly iterable
                handle = experts.register_forward_hook(
                    self._make_fused_hook(layer_idx)
                )
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


class RouterLogitHook:
    """Manages hooks for capturing router logits from MoE gate layers."""

    def __init__(self):
        self.router_logits: Dict[int, List[torch.Tensor]] = {}
        self.hook_handles: List[Any] = []

    def _save(self, module, input, output):
        layer_id = getattr(module, 'layer_id', None)
        if layer_id is None:
            return
        # Gate output may be a tuple (router_logits, scores, indices)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        self.router_logits.setdefault(layer_id, []).append(
            logits.detach().cpu().to(torch.float32)
        )

    def register_hooks(self, model):
        for layer_idx, layer in enumerate(model.model.layers):
            moe_block = layer.mlp
            if hasattr(moe_block, 'gate'):
                moe_block.gate.layer_id = layer_idx
                handle = moe_block.gate.register_forward_hook(self._save)
                self.hook_handles.append(handle)

    def get_router_logits(self) -> Dict[int, torch.Tensor]:
        return {lid: torch.cat(ts, dim=0) for lid, ts in self.router_logits.items()}

    def clear_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def clear_logits(self):
        self.router_logits.clear()
