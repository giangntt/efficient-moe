"""
Utilities for collecting and analyzing router logits.
"""
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from utils.device_utils import get_input_device


def collect_router_logits(model, tokenizer, prompts, device=None, output_final_logits=False):
    """
    Collect router logits from model inference on prompts.
    
    Args:
        model: The model to run inference on
        tokenizer: Tokenizer for the model
        prompts: List of prompt strings
        device: Device to run inference on (auto-detected from model if None)
        output_final_logits: Whether to also collect final output logits
    
    Returns:
        dict: {
            'router_logits': dict[layer_idx] -> tensor [total_tokens, num_experts],
            'final_logits': list of tensors (if output_final_logits=True)
        }
    """
    if device is None:
        device = get_input_device(model)
    router_logits = {}
    final_logits = [] if output_final_logits else None
    
    with torch.no_grad():
        for idx, prompt in enumerate(tqdm(prompts, desc="Collecting router logits")):
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs, output_router_logits=True)
            
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
    
    device = get_input_device(model)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting router logits")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
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

