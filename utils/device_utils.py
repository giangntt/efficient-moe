"""
Utilities for handling device placement in multi-GPU setups.

When using device_map="auto" with Hugging Face Accelerate, different layers of
the model may reside on different GPUs. These helpers detect the correct device
for placing inputs and per-layer tensors.
"""
import torch


def get_input_device(model):
    """
    Return the device where model inputs (token IDs, attention masks) should be placed.

    For device_map="auto" models this is the device of the embedding layer.
    Falls back to the first parameter's device if no embedding is found.

    Args:
        model: A Hugging Face model (AutoModelForCausalLM or its inner .model)

    Returns:
        torch.device
    """
    # Unwrap common wrappers
    inner = model.model if hasattr(model, "model") else model

    # Try the embedding layer first (most reliable for input placement)
    if hasattr(inner, "embed_tokens"):
        return next(inner.embed_tokens.parameters()).device

    # Fallback: first parameter in the model
    return next(inner.parameters()).device


def get_layer_device(layer):
    """
    Return the device of a specific model layer.

    Useful for placing per-layer tensors (e.g. pruned expert indices) on the
    correct GPU when the model is sharded across devices.

    Args:
        layer: A torch.nn.Module (typically a transformer layer or MoE block)

    Returns:
        torch.device
    """
    return next(layer.parameters()).device
