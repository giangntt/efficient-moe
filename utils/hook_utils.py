"""
Utilities for registering and managing hooks to capture expert activations.
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any


class ExpertActivationHook:
    """Manages hooks for capturing expert activations.

    Two modes:

    - ``streaming=False`` (default): every forward pass appends the raw
      activation tensor to ``self.expert_activations[(layer, expert)]``.
      Convenient for downstream analysis (SVD/overlap, max-norm, etc.) but
      memory grows linearly with prompts × tokens × hidden_dim and risks
      OOM on large profiling runs.

    - ``streaming=True``: for each forward pass, compute the per-token L2
      norm of the captured activation and fold it into a Welford running
      mean/variance per ``(layer, expert)``. Raw tensors are discarded
      after the update, so memory usage is O(num_layers × num_experts).
      Use ``get_streaming_stats()`` to retrieve the per-expert mean and
      variance of activation L2 norms.
    """

    def __init__(self, streaming: bool = False):
        self.streaming = streaming
        self.expert_activations: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        # Welford state per (layer, expert): (count, mean, M2)
        self._stream_state: Dict[Tuple[int, int], Tuple[int, float, float]] = {}
        self.hook_handles: List[Any] = []

    def _record(self, key: Tuple[int, int], activation: torch.Tensor) -> None:
        """Record a captured activation either as raw tensor or streaming stats."""
        if self.streaming:
            # Per-token L2 norms over the last (hidden) dim → 1D tensor.
            with torch.no_grad():
                norms = activation.detach().float().norm(dim=-1).flatten().cpu()
            self._update_welford(key, norms)
        else:
            self.expert_activations.setdefault(key, []).append(activation.detach().cpu())

    def _update_welford(self, key: Tuple[int, int], values: torch.Tensor) -> None:
        """Fold a 1D tensor of new observations into the Welford state for `key`."""
        n_new = values.numel()
        if n_new == 0:
            return
        count, mean, m2 = self._stream_state.get(key, (0, 0.0, 0.0))
        new_mean = float(values.mean().item())
        # population sum-of-squared-deviations for the incoming batch
        new_m2 = float(((values - new_mean) ** 2).sum().item())

        total = count + n_new
        delta = new_mean - mean
        combined_mean = mean + delta * n_new / total
        combined_m2 = m2 + new_m2 + (delta * delta) * count * n_new / total
        self._stream_state[key] = (total, combined_mean, combined_m2)

    def save_expert_activation(self, module, input, output):
        """Hook for individual down_proj modules (classic ModuleList style)."""
        layer_id = getattr(module, 'layer_id', None)
        expert_id = getattr(module, 'expert_id', None)
        if layer_id is not None and expert_id is not None:
            self._record((layer_id, expert_id), output)

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
                    hook_store._record((layer_idx, eidx), act)

        return hook_fn

    def get_streaming_stats(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Return per-expert running mean / variance of activation L2 norms.

        Only meaningful when ``streaming=True``. Variance is the population
        variance (matching ``tensor.var(unbiased=False)``), to keep the
        downstream pruning math identical to the raw-tensor path.
        """
        stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        for key, (count, mean, m2) in self._stream_state.items():
            var = (m2 / count) if count > 0 else 0.0
            stats[key] = {"mean": mean, "var": var, "count": count}
        return stats

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
        """Clear stored activations and any streaming state."""
        self.expert_activations.clear()
        self._stream_state.clear()


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
