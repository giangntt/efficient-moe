"""
Utilities for collecting and analyzing router logits.
"""
import torch
from typing import Dict, List, Optional
from tqdm import tqdm


def collect_router_logits(
    model,
    tokenizer,
    prompts,
    device,
    output_final_logits=False,
    max_new_tokens: int = 512,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    enable_thinking: bool = False,
):
    """
    Collect router logits from model generation on prompts.

    Uses model.generate() so that generation parameters are applied and router
    logits are captured via a hook on each MoE gate layer (works with
    device_map="auto" multi-GPU layouts).

    Args:
        model: The model to run inference on
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        device: Device for input tensors (first device of the model)
        output_final_logits: Unused; kept for API compatibility
        max_new_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (None = greedy)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling
        enable_thinking: Qwen3 chat-template flag (False disables thinking mode)

    Returns:
        dict: {
            'router_logits': dict[layer_idx] -> tensor [total_tokens, num_experts],
            'final_logits': [] (placeholder for API compatibility)
        }
    """
    from utils.hook_utils import RouterLogitHook

    router_hook = RouterLogitHook()
    router_hook.register_hooks(model)

    # Build generate() kwargs
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
        gen_kwargs["do_sample"] = True
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
        gen_kwargs["do_sample"] = True

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Collecting router logits"):
            # Apply chat template (handles enable_thinking for Qwen3)
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                    )
                except TypeError:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                inputs = tokenizer(
                    formatted, return_tensors="pt", truncation=True, max_length=4096
                )
            else:
                inputs = tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            model.generate(**inputs, **gen_kwargs)

    router_logits = router_hook.get_router_logits()
    router_hook.clear_hooks()

    return {"router_logits": router_logits, "final_logits": []}


def collect_router_logits_from_loader(model, loader, output_final_logits=False):
    """
    Collect router logits from model inference on a DataLoader.
    
    Args:
        model: The model to run inference on
        loader: DataLoader with batches
        output_final_logits: Whether to also collect final output logits
    
    Returns:
        dict: {
            'router_logits': dict[layer_idx] -> tensor [total_tokens, num_experts],
            'final_logits': list of tensors (if output_final_logits=True)
        }
    """
    router_logits = {}
    final_logits = [] if output_final_logits else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting router logits")):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
            
            for layer_idx, logits in enumerate(outputs.router_logits):
                if layer_idx not in router_logits:
                    router_logits[layer_idx] = []
                router_logits[layer_idx].append(logits.cpu().to(torch.float32))
            
            if output_final_logits:
                final_logits.append(outputs.logits.cpu().to(torch.float32))
    
    # Concatenate all router logits per layer
    for layer_idx in router_logits:
        router_logits[layer_idx] = torch.cat(router_logits[layer_idx], dim=0)
    
    result = {'router_logits': router_logits}
    if output_final_logits:
        result['final_logits'] = final_logits
    
    return result

