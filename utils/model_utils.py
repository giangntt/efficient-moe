import torch
import torch.nn.functional as F

# -------------------
# Patched forward & pruning
# -------------------

def patched_forward_masked_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method that masks pruned experts by setting their logits to -inf
    before softmax to prevent routing tokens to them.

    Works with Qwen3MoeSparseMoeBlock where self.gate is Qwen3MoeTopKRouter
    (returns (router_logits, router_scores, router_indices)) and self.experts
    is Qwen3MoeExperts (batched tensor, not a list of modules).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    # Compute raw logits directly from gate weights so we can mask before softmax
    raw_logits = F.linear(hidden_states_reshaped, self.gate.weight)

    # Mask pruned experts (pruned_experts_tensor is pre-created during apply_pruning)
    pruned_experts_tensor = getattr(self, "pruned_experts_tensor", None)
    if pruned_experts_tensor is not None and pruned_experts_tensor.numel() > 0:
        pruned_experts_tensor = pruned_experts_tensor.to(raw_logits.device)
        raw_logits[:, pruned_experts_tensor] = float('-inf')

    # Replicate Qwen3MoeTopKRouter routing logic on masked logits
    router_logits = F.softmax(raw_logits, dtype=torch.float, dim=-1)
    routing_weights, selected_experts = torch.topk(router_logits, self.gate.top_k, dim=-1)
    if self.gate.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def patched_forward_zeroed_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Patched forward method that zeros out pruned experts' routing weights after
    normal routing. Tokens may still be selected for pruned experts but their
    contribution is zeroed (no renormalization).

    Works with Qwen3MoeSparseMoeBlock where self.gate is Qwen3MoeTopKRouter
    (returns (router_logits, router_scores, router_indices)) and self.experts
    is Qwen3MoeExperts (batched tensor, not a list of modules).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    # Gate returns (softmaxed_logits, top_k_weights, top_k_indices)
    _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

    # Zero routing weights for pruned experts (output zeroed, no renorm)
    pruned_experts_tensor = getattr(self, "pruned_experts_tensor", None)
    if pruned_experts_tensor is not None and pruned_experts_tensor.numel() > 0:
        pruned_experts_tensor = pruned_experts_tensor.to(selected_experts.device)
        is_pruned = (selected_experts[..., None] == pruned_experts_tensor).any(dim=-1)
        routing_weights = routing_weights.masked_fill(is_pruned, 0.0)

    final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

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
