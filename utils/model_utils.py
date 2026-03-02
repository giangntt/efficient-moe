import torch
import torch.nn.functional as F
from utils.device_utils import get_input_device, get_layer_device

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

    # Check if experts are pruned and mask them out
    pruned_experts_tensor = getattr(self, "pruned_experts_tensor", None)
    if pruned_experts_tensor is not None and pruned_experts_tensor.numel() > 0:
        is_pruned = (selected_experts[..., None] == pruned_experts_tensor).any(dim=-1)
    else:
        is_pruned = torch.zeros_like(selected_experts, dtype=torch.bool)
    selected_experts = selected_experts.masked_fill(is_pruned, -1)

    # Build mask (ignore -1)
    expert_mask = torch.nn.functional.one_hot(selected_experts.clamp(min=0), num_classes=self.num_experts).permute(2, 1, 0).float()
    expert_mask *= (selected_experts >= 0).float().permute(1, 0)[None, :, :]

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
    
    for layer_idx, layer in enumerate(actual_model.layers):
        moe_block = layer.mlp  
        if hasattr(moe_block, "gate") and hasattr(moe_block, "experts"):
            pruned_experts = experts_to_prune.get(layer_idx, [])
            # Get the device for THIS layer (may differ across GPUs with device_map="auto")
            layer_device = get_layer_device(moe_block)
            # Pre-create tensor once during pruning setup to avoid recreating it every forward pass
            pruned_experts_tensor = torch.tensor(list(pruned_experts), device=layer_device, dtype=torch.long) if pruned_experts else torch.tensor([], device=layer_device, dtype=torch.long)
            moe_block.pruned_experts_tensor = pruned_experts_tensor
            moe_block.forward = patch_fn.__get__(moe_block, moe_block.__class__)


def prune_weights_inplace(model, experts_to_prune, mode="zero"):
    """
    Permanently prune experts by mutating model weights in-place.

    Unlike apply_pruning() (which monkey-patches forward at runtime), this
    function directly modifies the weight tensors so the result can be saved
    with model.save_pretrained() and loaded by any inference engine (vLLM, etc.).

    Two modes (matching the runtime apply_pruning modes):

      "zero"  — zeros all weight tensors in each pruned expert's MLP
                (gate_proj, up_proj, down_proj). The router may still route
                tokens to those experts, but their output will be all zeros.

      "mask"  — additionally sets a large negative bias (-1e9) on the gate
                linear layer's columns for pruned experts, so softmax assigns
                them ~0 probability and tokens are never routed to them.
                Since the gate has no bias by default, a zero bias is added
                first. Expert weights are also zeroed (belt-and-suspenders).

    Args:
        model: A Hugging Face AutoModelForCausalLM (or its inner .model).
        experts_to_prune: dict mapping layer_idx (int) -> list of expert indices.
        mode: "zero" or "mask".
    """
    if mode not in ("zero", "mask"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'zero' or 'mask'.")

    actual_model = model.model if hasattr(model, "model") else model
    total_zeroed = 0
    layers_affected = 0

    for layer_idx, layer in enumerate(actual_model.layers):
        moe_block = layer.mlp
        if not (hasattr(moe_block, "gate") and hasattr(moe_block, "experts")):
            continue

        to_prune = experts_to_prune.get(layer_idx, [])
        if not to_prune:
            continue

        layers_affected += 1

        # Both modes: zero out the expert MLP weights
        for expert_idx in to_prune:
            for param in moe_block.experts[expert_idx].parameters():
                param.data.zero_()
            total_zeroed += 1

        # mask mode: bias the gate so pruned experts get ~0 routing probability
        if mode == "mask":
            gate = moe_block.gate  # nn.Linear(hidden_size, num_experts, bias=False)
            if gate.bias is None:
                gate.bias = torch.nn.Parameter(
                    torch.zeros(gate.out_features,
                                device=gate.weight.device,
                                dtype=gate.weight.dtype)
                )
            gate.bias.data[list(to_prune)] = -1e9

    print(f"[prune_weights_inplace] mode={mode} | "
          f"layers affected: {layers_affected} | experts zeroed: {total_zeroed}")


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
            device = get_input_device(model)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

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
