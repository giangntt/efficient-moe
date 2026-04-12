import torch
import torch.nn.functional as F

# -------------------
# Patched forward & pruning
# -------------------

def patched_forward_masked_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method that masks pruned experts by setting their logits to -inf
    before softmax to prevent routing tokens to them.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    # Mask pruned experts (pruned_experts_tensor is pre-created during apply_pruning)
    pruned_experts_tensor = getattr(self, "pruned_experts_tensor", None)
    if pruned_experts_tensor is not None and pruned_experts_tensor.numel() > 0:
        router_logits[:, pruned_experts_tensor] = float('-inf')

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

    if hasattr(self, 'shared_expert') and self.shared_expert is not None:
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

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0).float()
    pruned_experts_tensor = getattr(self, "pruned_experts_tensor", None)
    if pruned_experts_tensor is not None and pruned_experts_tensor.numel() > 0:
        # Drop all dispatch slots for pruned experts; they will not appear in expert_hit.
        expert_mask[pruned_experts_tensor] = 0

    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        expert_layer = self.experts[expert_idx]
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    if hasattr(self, 'shared_expert') and self.shared_expert is not None:
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

def apply_pruning(model, experts_to_prune, mode="zero"):
    """
    Monkey patch each OlmoeSparseMoeBlock forward and assign pruned experts per layer.

    Args:
        model: The model to prune.
        experts_to_prune: A dictionary where keys are layer indices and values are lists of expert indices to prune.
        mode: "mask" (mask logits) or "zero" (zero out expert outputs)
    """
    if mode == "mask":
        patch_fn = patched_forward_masked_experts
    elif mode == "zero":
        patch_fn = patched_forward_zeroed_experts
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'mask' or 'zero'.")

    # Get the actual model (handle both model and model.model cases)
    actual_model = model.model if hasattr(model, 'model') else model
    # Get device from model parameters
    device = next(actual_model.parameters()).device
    
    for layer_idx, layer in enumerate(actual_model.layers):
        moe_block = layer.mlp  
        if hasattr(moe_block, "gate") and hasattr(moe_block, "experts"):
            pruned_experts = experts_to_prune.get(layer_idx, [])
            # Pre-create tensor once during pruning setup to avoid recreating it every forward pass
            pruned_experts_tensor = torch.tensor(list(pruned_experts), device=device, dtype=torch.long) if pruned_experts else torch.tensor([], device=device, dtype=torch.long)
            moe_block.pruned_experts_tensor = pruned_experts_tensor
            moe_block.forward = patch_fn.__get__(moe_block, moe_block.__class__)


def evaluate_model(model, val_loader):
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model: The model to evaluate.
        val_loader: DataLoader for the validation dataset.
    
    Returns:
        avg_loss: Average token-level loss.
        accuracy: Token-level accuracy.
    """
    from tqdm import tqdm
    
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            # Ignore padding tokens in loss
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # averaged per token (excluding -100)
            logits = outputs.logits  # (batch, seq_len, vocab)

            # Shift inputs for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Compute accuracy only on non-padding tokens
            predictions = shift_logits.argmax(dim=-1)
            correct = ((predictions == shift_labels) * shift_mask).sum().item()
            tokens = shift_mask.sum().item()

            total_loss += loss.item() * tokens  # total loss over tokens
            total_correct += correct
            total_tokens += tokens
            num_batches += 1

    # Final metrics
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    return avg_loss, accuracy
