import torch
import torch.nn.functional as F

# -------------------
# Patched forward & pruning
# -------------------

import torch
import torch.nn.functional as F

def patched_forward_masked_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method that masks pruned experts by setting their logits to -inf
    before softmax to prevent routing tokens to them.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    # # Mask pruned experts
    pruned_experts = getattr(self, "pruned_experts", None)
    if pruned_experts is not None and len(pruned_experts) > 0:
        idxs = torch.tensor(pruned_experts, device=router_logits.device, dtype=torch.long)
        router_logits[:, idxs] = float('-inf')

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def patched_forward_zeroed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method that zeros out pruned experts' outputs after routing.
    Tokens may still be routed to pruned experts, but their outputs are zeroed.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    pruned_experts = set(getattr(self, "pruned_experts", []))

    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        expert_idx_int = int(expert_idx)
        if expert_idx_int in pruned_experts:
            # Zero output for pruned experts
            current_hidden_states = torch.zeros(len(top_x), hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            expert_layer = self.experts[expert_idx]
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def patched_forward_dynamic_routing(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method for dynamic routing based on cumulative probability threshold.
    Selects a variable number of experts up to self.top_k.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    pruned_experts = getattr(self, "pruned_experts", None)
    if pruned_experts is not None and len(pruned_experts) > 0:
        idxs = torch.tensor(pruned_experts, device=router_logits.device, dtype=torch.long)
        router_logits[:, idxs] = float('-inf')

    routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    # self.top_k is used as the max number of experts to consider
    routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)

    # Dynamic routing based on cumulative probability
    # The threshold is set on the MoE block during the patching process.
    threshold = self.dynamic_routing_threshold
    cumulative_weights = torch.cumsum(routing_weights, dim=-1)
    
    # Create a mask to select experts until the cumulative probability exceeds the threshold
    shifted_cumsum = torch.zeros_like(cumulative_weights)
    shifted_cumsum[..., 1:] = cumulative_weights[..., :-1]
    selection_mask = shifted_cumsum < threshold

    # Apply the mask to the routing weights
    routing_weights = routing_weights * selection_mask

    # DEBUG: Log routing info for the first token to verify the logic
    if not hasattr(self, '_debug_logged'):
        token_idx = 0
        print("\n" + "="*50)
        print(f"DEBUG: Dynamic Routing for first token in batch (Layer: {self.layer_id if hasattr(self, 'layer_id') else 'Unknown'})")
        print(f"  Threshold: {threshold}")
        print(f"  Top-k routing weights: {routing_weights[token_idx].tolist()}")
        print(f"  Cumulative weights: {cumulative_weights[token_idx].tolist()}")
        print(f"  Selection mask: {selection_mask[token_idx].tolist()}")
        num_selected = selection_mask[token_idx].sum().item()
        print(f"  => Number of experts selected: {num_selected}")
        print("="*50 + "\n")
        self._debug_logged = True

    if self.norm_topk_prob:
        # Normalize routing weights for the selected experts
        # Add a small epsilon for numerical stability to prevent division by zero.
        routing_weights_sum = routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights / (routing_weights_sum + 1e-6)

    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    
    # Apply the dynamic selection mask to the expert_mask
    selection_mask_broadcast = selection_mask.permute(1, 0).unsqueeze(0)
    expert_mask = expert_mask * selection_mask_broadcast

    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def apply_pruning(model, experts_to_prune, mode="zero", dynamic_routing_threshold=0.8):
    """
    Monkey patch each OlmoeSparseMoeBlock forward and assign pruned experts per layer.

    Args:
        model: The model to prune.
        experts_to_prune: A dictionary where keys are layer indices and values are lists of expert indices to prune.
        mode: "mask" (mask logits), "zero" (zero out expert outputs), or "dynamic" (dynamic routing).
        dynamic_routing_threshold (float): Cumulative probability threshold for dynamic routing.
    """
    if mode == "mask":
        patch_fn = patched_forward_masked_experts
    elif mode == "zero":
        patch_fn = patched_forward_zeroed_experts
    elif mode == "dynamic":
        patch_fn = patched_forward_dynamic_routing
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mask', 'zero', or 'dynamic'.")

    for layer_idx, layer in enumerate(model.model.layers):
        moe_block = layer.mlp  
        if hasattr(moe_block, "gate") and hasattr(moe_block, "experts"):
            moe_block.layer_id = layer_idx  # For debug logging
            print(f"INFO: Patching MoE layer {layer_idx} with mode '{mode}'.")
            moe_block.pruned_experts = experts_to_prune.get(layer_idx, [])
            if mode == "dynamic":
                print(f"INFO: Setting dynamic_routing_threshold to {dynamic_routing_threshold} for layer {layer_idx}.")
                moe_block.dynamic_routing_threshold = dynamic_routing_threshold
            moe_block.forward = patch_fn.__get__(moe_block, moe_block.__class__)
