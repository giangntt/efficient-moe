"""
Analysis utilities for MoE models: router statistics, expert usage, correlations, etc.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from scipy.stats import spearmanr, pearsonr
    def spearman_safe(a, b):
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return np.nan
        return spearmanr(a, b).correlation
    def pearson_safe(a, b):
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return np.nan
        return pearsonr(a, b).correlation
except ImportError:
    # Fallback implementations
    def spearman_fallback(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        if rx.std() == 0 or ry.std() == 0:
            return np.nan
        return np.corrcoef(rx, ry)[0, 1]
    
    def pearson_fallback(a, b):
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    
    spearman_safe = spearman_fallback
    pearson_safe = pearson_fallback


def cat_if_list(x):
    """Concatenate if list, otherwise return as-is."""
    return torch.cat(x, dim=0) if isinstance(x, list) else x


def compute_router_stats(router_logits_layer, expert_acts_layer, top_k=None, device='cpu'):
    """
    Compute router statistics for a single layer.
    
    Args:
        router_logits_layer: tensor [T, E] or list of tensors
        expert_acts_layer: dict[expert_id] -> list of tensors [num_tokens, H]
        top_k: k used for selection (None for full softmax)
        device: device to use for computation
    
    Returns:
        dict with keys: mean_over_all, select_freq, cond_mean, mean_act, var_act, approx_contrib
    """
    logits_cat = cat_if_list(router_logits_layer).to(device)   # [T, E]
    probs = torch.softmax(logits_cat, dim=-1)                 # [T, E]

    T, E = probs.shape
    mean_over_all = probs.mean(dim=0).cpu().numpy()           # global mean weight per expert
    select_freq = None
    cond_mean = None
    
    if top_k is not None:
        # compute selection mask per token (top-k)
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        # boolean mask [T, E]
        sel_mask = torch.zeros_like(probs, dtype=torch.bool)
        sel_mask.scatter_(1, topk_idx, True)
        select_freq = sel_mask.float().mean(dim=0).cpu().numpy()
        # conditional mean: mean of probs where selected (avoid div0)
        sum_selected = (probs * sel_mask.float()).sum(dim=0)                # sum of probs for selected tokens
        denom = sel_mask.sum(dim=0).clamp(min=1.0)                          # number of selected tokens per expert
        cond_mean = (sum_selected / denom).cpu().numpy()
    else:
        # if no topk, define selection as prob > small threshold
        sel_mask = probs > 1e-6
        select_freq = sel_mask.float().mean(dim=0).cpu().numpy()
        sum_selected = (probs * sel_mask.float()).sum(dim=0)
        denom = sel_mask.sum(dim=0).clamp(min=1.0)
        cond_mean = (sum_selected / denom).cpu().numpy()

    # mean activation norm & variance per expert.
    # `acts` may be either a list of raw tensors (legacy hook mode) or a
    # pre-computed dict {'mean': float, 'var': float} from streaming hook mode.
    mean_act = []
    var_act = []

    for e in range(E):
        acts = expert_acts_layer.get(e, None)
        if acts is None:
            mean_act.append(0.0)
            var_act.append(0.0)
        elif isinstance(acts, dict):
            mean_act.append(float(acts.get('mean', 0.0)))
            var_act.append(float(acts.get('var', 0.0)))
        elif len(acts) == 0:
            mean_act.append(0.0)
            var_act.append(0.0)
        else:
            acts_cat = torch.cat(acts, dim=0).to(device)  # [N_e, H]
            norms = acts_cat.norm(dim=-1)                 # [N_e]
            mean_act.append(norms.mean().item())
            var_act.append(norms.var(unbiased=False).item())

    mean_act = np.array(mean_act)
    var_act = np.array(var_act)

    # approximate contribution (coarse)
    approx_contrib = cond_mean * mean_act

    return {
        "mean_over_all": mean_over_all,
        "select_freq": select_freq,
        "cond_mean": cond_mean,
        "mean_act": mean_act,
        "var_act": var_act,
        "approx_contrib": approx_contrib
    }


def compute_all_stats(router_logits, expert_activations, top_k=None, device='cpu'):
    """
    Compute stats per layer per expert.

    Args:
        router_logits: dict[layer_id] -> tensor [T, E] or list of tensors per batch
        expert_activations: dict[(layer, expert)] -> list of tensors [num_tokens, H]
        top_k: int or None
        device: device to use for computation
    
    Returns:
        dict[layer_id] -> stats dict
    """
    all_stats = {}

    for layer_id, logits_layer in router_logits.items():
        # extract activations for this layer: dict[expert] -> list of tensors
        num_experts = logits_layer[0].shape[1] if isinstance(logits_layer, list) else logits_layer.shape[1]
        expert_acts_layer = {
            e: expert_activations.get((layer_id, e), None)
            for e in range(num_experts)
        }

        stats_layer = compute_router_stats(
            router_logits_layer=logits_layer,
            expert_acts_layer=expert_acts_layer,
            top_k=top_k,
            device=device
        )
        all_stats[layer_id] = stats_layer

    return all_stats


def analyze_expert_usage_entropy(router_logits, top_k=4):
    """
    Analyze expert usage and entropy from router logits.

    Args:
        router_logits: dict[layer_idx] = tensor([total_tokens, num_experts])
        top_k: number of experts activated per token

    Returns:
        layer_entropy: list [num_layers] of avg gating entropy
        expert_usage: list [num_layers] of tensors [num_experts] (usage fractions)
    """
    num_layers = len(router_logits)
    layer_entropy = []
    expert_usage = []

    for layer_id in range(num_layers):
        logits = router_logits[layer_id]  # [total_tokens, num_experts]
        probs = F.softmax(logits, dim=-1)     # [total_tokens, num_experts]

        # ---- entropy per token
        ent = -(probs * probs.clamp(min=1e-9).log()).sum(-1)  # [total_tokens]
        layer_entropy.append(ent.mean().item())

        # ---- expert usage (top-k activation counts)
        topk_idx = probs.topk(top_k, dim=-1).indices  # [total_tokens, k]
        counts = torch.bincount(topk_idx.view(-1), minlength=probs.size(-1)).float()
        usage_frac = counts / counts.sum()

        expert_usage.append(usage_frac.cpu())

    return layer_entropy, expert_usage


def compute_correlations_by_layer(all_stats):
    """
    Compute Spearman and Pearson correlations between cond_mean and mean_act per layer.
    
    Args:
        all_stats: dict[layer_id] -> stats dict from compute_all_stats
    
    Returns:
        spearman_by_layer: dict[layer_id] -> float or np.nan
        pearson_by_layer: dict[layer_id] -> float or np.nan
    """
    spearman_by_layer = {}
    pearson_by_layer = {}
    layers = sorted(list(all_stats.keys()))

    for layer in layers:
        stats = all_stats[layer]
        cond_mean = np.array(stats.get('cond_mean', []))
        mean_act = np.array(stats.get('mean_act', []))
        
        # require at least two values and non-zero variation
        if cond_mean.size < 2 or mean_act.size < 2 or np.nanstd(cond_mean) == 0 or np.nanstd(mean_act) == 0:
            spearman_by_layer[layer] = np.nan
            pearson_by_layer[layer] = np.nan
            continue
        
        sp = spearman_safe(cond_mean, mean_act)
        pe = pearson_safe(cond_mean, mean_act)
        spearman_by_layer[layer] = float(sp) if not np.isnan(sp) else np.nan
        pearson_by_layer[layer] = float(pe) if not np.isnan(pe) else np.nan

    return spearman_by_layer, pearson_by_layer


def aggregate_norm_stats(expert_activations):
    """
    Aggregate L2 norm statistics per expert.
    
    Args:
        expert_activations: dict[(layer_id, expert_id), acts_list]
    
    Returns:
        dict[(layer_id, expert_id)] -> mean L2 norm
    """
    stats = {}
    for (layer_id, expert_id), acts_list in expert_activations.items():
        # Concatenate all activations for this expert
        if isinstance(acts_list, list):
            acts_cat = torch.cat(acts_list, dim=0)
        else:
            acts_cat = acts_list
        # Compute mean L2 norm per token
        norm_val = acts_cat.norm(p=2, dim=1).mean().item()
        stats[(layer_id, expert_id)] = norm_val
    return stats


def compare_logits(full_logits_list, pruned_logits_list):
    """
    Compare full and pruned logits for a list of sequences.
    
    Args:
        full_logits_list: list of tensors [1, seq_len, vocab]
        pruned_logits_list: list of tensors [1, seq_len, vocab]
    
    Returns:
        result: dict with keys:
            - per_token_cos: list of tensors [seq_len] per sequence
            - avg_token_cos: list of float per sequence
            - sequence_cos: list of float per sequence
            - sequence_kl: list of float per sequence
    """
    per_token_cos = []
    avg_token_cos = []
    sequence_cos = []
    sequence_kl = []

    for full, pruned in zip(full_logits_list, pruned_logits_list):
        # remove batch dim
        full = full.squeeze(0)   # [seq_len, vocab]
        pruned = pruned.squeeze(0)
        
        # --- per-token cosine similarity ---
        token_sim = F.cosine_similarity(full, pruned, dim=1)  # [seq_len]
        per_token_cos.append(token_sim)
        avg_token_cos.append(token_sim.mean().item())
        
        # --- sequence-level cosine similarity ---
        seq_sim = F.cosine_similarity(full.view(1, -1), pruned.view(1, -1)).item()
        sequence_cos.append(seq_sim)
        
        # --- sequence-level KL divergence ---
        # treat full as target
        kl = F.kl_div(
            F.log_softmax(pruned, dim=-1),
            F.softmax(full, dim=-1),
            reduction='batchmean'
        ).item()
        sequence_kl.append(kl)

    result = {
        "per_token_cos": per_token_cos,
        "avg_token_cos": avg_token_cos,
        "sequence_cos": sequence_cos,
        "sequence_kl": sequence_kl
    }
    return result


def summarize_pruned_experts(full_expert_activations, expert_activations, top_k_per_layer=5):
    """
    Summarize which experts were pruned and their impact.
    
    Args:
        full_expert_activations: dict[(layer, expert)] -> list of tensors
        expert_activations: dict[(layer, expert)] -> list of tensors (after pruning)
        top_k_per_layer: number of top disabled experts to show per layer
    
    Returns:
        per_layer: dict[layer] -> summary dict
        disabled_keys: set of (layer, expert) tuples that were disabled
    """
    full_keys = set(full_expert_activations.keys())
    pruned_keys = set(expert_activations.keys())
    disabled_keys = full_keys - pruned_keys

    per_layer = defaultdict(lambda: {
        'num_experts_original': 0,
        'num_disabled': 0,
        'tokens_original': 0,
        'tokens_disabled': 0,
        'disabled_details': []  # list of (expert_id, tokens_assigned)
    })

    # Aggregate counts from full_expert_activations
    for (layer, expert) in full_keys:
        per_layer[layer]['num_experts_original'] += 1
        acts_list = full_expert_activations[(layer, expert)]
        # Count tokens for this expert as sum of first-dim sizes of saved tensors
        tokens = sum(getattr(a, 'shape', (0,))[0] for a in acts_list)
        per_layer[layer]['tokens_original'] += tokens
        if (layer, expert) in disabled_keys:
            per_layer[layer]['num_disabled'] += 1
            per_layer[layer]['tokens_disabled'] += tokens
            per_layer[layer]['disabled_details'].append((expert, int(tokens)))

    # Print a readable per-layer summary
    print('Per-layer pruning summary:')
    layers = sorted(per_layer.keys())
    for layer in layers:
        d = per_layer[layer]
        tot = d['tokens_original']
        disabled = d['tokens_disabled']
        frac = (disabled / tot * 100.0) if tot else 0.0
        print(f"- Layer {layer}: experts_original={d['num_experts_original']}, disabled={d['num_disabled']}, "
              f"tokens_original={d['tokens_original']}, tokens_disabled={d['tokens_disabled']} ({frac:.2f}% of layer tokens)")
        # show top disabled experts by tokens (if any)
        if d['disabled_details']:
            top = sorted(d['disabled_details'], key=lambda x: x[1], reverse=True)[:top_k_per_layer]
            print('   top disabled experts (expert_id, tokens):', top)

    # Global totals
    total_experts_orig = sum(per_layer[l]['num_experts_original'] for l in per_layer)
    total_disabled = sum(per_layer[l]['num_disabled'] for l in per_layer)
    total_tokens_orig = sum(per_layer[l]['tokens_original'] for l in per_layer)
    total_tokens_disabled = sum(per_layer[l]['tokens_disabled'] for l in per_layer)
    total_frac = (total_tokens_disabled / total_tokens_orig * 100.0) if total_tokens_orig else 0.0
    print('\nOverall:')
    print(f"  experts_original={total_experts_orig}, disabled={total_disabled}")
    print(f"  tokens_original={total_tokens_orig}, tokens_disabled={total_tokens_disabled} ({total_frac:.2f}% of all tokens)")

    return per_layer, disabled_keys


def analyze_expert_interactions(router_logits, top_k_routing=4,
                                           metrics=("pattern_similarity", "jaccard_similarity", "conditional_probability")):
    """
    Analyze MoE expert interactions efficiently.

    Args:
        router_logits: dict[layer_idx] -> tensor [num_tokens, num_experts]
        top_k_routing: number of top experts per token
        metrics: tuple of metrics to compute

    Returns:
        results: dict[layer_idx] -> dict of metric matrices [num_experts, num_experts]
    """
    from tqdm import tqdm
    
    results = {}

    for layer_idx, logits in tqdm(router_logits.items(), desc="Analyzing expert overlap"):
        num_tokens, num_experts = logits.shape
        probs = F.softmax(logits, dim=-1)

        # Top-k indices and probabilities
        topk_vals, topk_idx = torch.topk(probs, k=top_k_routing, dim=-1, largest=True, sorted=True)

        layer_res = {}

        # -------------------------
        # Pattern similarity (weighted by top-k probabilities)
        # -------------------------
        if "pattern_similarity" in metrics:
            expert_patterns = torch.zeros(num_experts, num_tokens, device=logits.device, dtype=probs.dtype)
            # scatter top-k probabilities into expert rows
            expert_patterns.scatter_add_(0, topk_idx.T, topk_vals.T)
            # normalize
            norms = expert_patterns.norm(dim=1, keepdim=True).clamp(min=1e-12)
            patterns_norm = expert_patterns / norms
            pattern_sim = (patterns_norm @ patterns_norm.T)
            # Zero out diagonal
            pattern_sim.fill_diagonal_(0.0)
            layer_res["pattern_similarity"] = pattern_sim.cpu().numpy()

        # -------------------------
        # Jaccard similarity (binary top-k)
        # -------------------------
        if "jaccard_similarity" in metrics:
            topk_mask = torch.zeros_like(probs)
            topk_mask.scatter_(1, topk_idx, 1.0)  # binary mask [num_tokens, num_experts]
            expert_patterns_bin = topk_mask.T      # [num_experts, num_tokens]
            intersection = expert_patterns_bin @ expert_patterns_bin.T  # [num_experts, num_experts]
            expert_counts = expert_patterns_bin.sum(dim=1, keepdim=True) # [num_experts, 1]
            union = expert_counts + expert_counts.T - intersection
            jaccard_sim = (intersection / (union + 1e-8))
            jaccard_sim.fill_diagonal_(0.0)
            layer_res["jaccard_similarity"] = jaccard_sim.cpu().numpy()

        # -------------------------
        # Conditional probability (i comes before j)
        # -------------------------
        if "conditional_probability" in metrics:
            num_tokens, top_k = topk_idx.shape
            # Generate all pair indices per token where i < j
            # Use broadcasting to generate pairs
            i_idx = topk_idx[:, :, None].expand(num_tokens, top_k, top_k)  # [tokens, top_k, top_k]
            j_idx = topk_idx[:, None, :].expand(num_tokens, top_k, top_k)  # [tokens, top_k, top_k]

            # Mask to keep only upper-triangular (i before j)
            mask = torch.triu(torch.ones(top_k, top_k, dtype=torch.bool, device=topk_idx.device), diagonal=1)
            i_flat = i_idx[:, mask].reshape(-1)
            j_flat = j_idx[:, mask].reshape(-1)

            # Accumulate counts
            linear_idx = i_flat * num_experts + j_flat
            counts = torch.bincount(linear_idx, minlength=num_experts*num_experts).reshape(num_experts, num_experts)

            # Normalize by occurrence of i in top-k
            occur = torch.bincount(topk_idx.reshape(-1), minlength=num_experts).float().clamp(min=1e-12)
            cond_prob = counts.float() / occur.view(-1,1)

            layer_res["conditional_probability"] = cond_prob.cpu().numpy()

        results[layer_idx] = layer_res

    return results


def get_top_pairs(metrics, top_k=1, symmetric_metrics=("pattern_similarity", "jaccard_similarity"), ignore_diagonal=True):
    """
    Extract top expert pairs for each layer and each metric.

    Args:
        metrics: dict[layer_idx] -> dict of metric matrices
        top_k: number of top pairs to return per metric
        symmetric_metrics: metrics where mat[i,j] == mat[j,i]
        ignore_diagonal: whether to ignore diagonal entries (i==j)

    Returns:
        top_pairs: dict[layer_idx][metric_name] -> list of tuples (i, j, score)
    """
    top_pairs = {}

    for layer, layer_metrics in metrics.items():
        top_pairs[layer] = {}
        for metric_name, mat in layer_metrics.items():
            # Convert to numpy if torch
            if hasattr(mat, "cpu"):
                mat = mat.cpu().numpy()

            # Optionally ignore diagonal
            if ignore_diagonal:
                mat = mat.copy()
                np.fill_diagonal(mat, -np.inf)

            # For symmetric metrics, use upper triangle only
            if metric_name in symmetric_metrics:
                triu_idx = np.triu_indices_from(mat, k=1)  # k=1 skips diagonal
                scores = mat[triu_idx]
                # Get top_k indices
                top_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                pairs = [(int(triu_idx[0][i]), int(triu_idx[1][i]), float(scores[i])) for i in top_idx]
            else:
                # Non-symmetric metrics: flatten full matrix
                flat_idx = np.argpartition(mat.flatten(), -top_k)[-top_k:]
                flat_idx = flat_idx[np.argsort(mat.flatten()[flat_idx])[::-1]]
                n_rows, n_cols = mat.shape
                pairs = [(int(idx // n_cols), int(idx % n_cols), float(mat[idx // n_cols, idx % n_cols])) for idx in flat_idx]

            top_pairs[layer][metric_name] = pairs

    return top_pairs


def sort_experts_by_usage_or_prob(router_logits, top_k_routing=4, criterion="freq", descending=True):
    """
    Sort experts by usage frequency or average routing probability.

    Args:
        router_logits: dict[layer_id] = tensor [num_tokens, num_experts]
        top_k_routing: number of experts routed per token
        criterion: 'freq' (usage count) or 'prob' (mean probability)
        descending: True = most→least, False = least→most

    Returns:
        sorted_dict: dict[layer_id] = list of expert indices sorted by the chosen criterion
    """
    assert criterion in ("freq", "prob"), "criterion must be 'freq' or 'prob'"
    sorted_dict = {}

    for layer_id, logits in router_logits.items():
        probs = torch.softmax(logits, dim=-1)

        if criterion == "freq":
            topk_vals, topk_idx = torch.topk(probs, k=top_k_routing, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, 1.0)
            usage_counts = mask.sum(dim=0)
            scores = usage_counts
        else:
            scores = probs.mean(dim=0)

        sorted_indices = torch.argsort(scores, descending=descending)
        sorted_dict[layer_id] = sorted_indices.tolist()

    return sorted_dict


def compute_asymmetric_overlap_matrix(
    expert_activations,
    layer_id,
    rank=64,
    sample_tokens=2000,
    device='cuda'
):
    """
    Compute asymmetric overlap matrix for a given layer.

    This measures how much each expert's activations (source) are captured
    by the low-rank subspace of other experts (target). High values indicate
    redundancy: source expert i is largely represented by target expert j.

    Args:
        expert_activations: dict[(layer_id, expert_id)] -> list of tensors
            Each tensor: [num_tokens_batch, hidden_dim] for activations of that expert
        layer_id: int, the layer to analyze
        rank: int, rank of the subspace for projection
        sample_tokens: int, maximum tokens to sample per expert to speed up computation
        device: 'cuda' or 'cpu'

    Returns:
        asymmetric_overlap: torch.Tensor [num_experts, num_experts]
            asymmetric_overlap[i, j] = fraction of activations of expert i
            captured by the low-rank subspace of expert j
    """
    # Collect all expert IDs in this layer
    expert_ids = sorted([e for (l, e) in expert_activations.keys() if l == layer_id])
    num_experts = len(expert_ids)

    # Dictionaries to store low-rank bases and full activations
    expert_bases = {}
    expert_full_X = {}

    # Step 1: Construct low-rank bases and store full activations for each expert
    for e in expert_ids:
        # Concatenate all batches for this expert
        X = torch.cat(expert_activations[(layer_id, e)], dim=0)  # [num_tokens_total, hidden_dim]

        # Optional: randomly sample tokens for efficiency
        if sample_tokens and X.size(0) > sample_tokens:
            idx = torch.randperm(X.size(0))[:sample_tokens]
            X = X[idx]

        # Move to device
        X = X.to(device)

        # Center activations
        X_centered = X - X.mean(dim=0, keepdim=True)
        X_centered = X_centered.float()  # ensure float32 for SVD

        # Step 1a: Low-rank basis using SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        basis = Vh[:rank].T  # [hidden_dim, rank]

        expert_bases[e] = basis         # store low-rank basis
        expert_full_X[e] = X_centered   # store full centered activations

    # Step 2: Initialize asymmetric overlap matrix
    # Rows: source experts (i), Columns: target experts (j)
    asymmetric_overlap = torch.zeros((num_experts, num_experts), device=device)

    # Step 3: Compute overlap for each pair of experts
    for i in range(num_experts):
        Xi = expert_full_X[expert_ids[i]]  # full activations of source expert i
        for j in range(num_experts):
            Bj = expert_bases[expert_ids[j]]  # low-rank basis of target expert j

            # Project source activations onto target basis
            X_proj = Xi @ Bj @ Bj.T  # [num_tokens, hidden_dim]

            # Overlap = fraction of Xi captured by Bj
            overlap = torch.norm(X_proj, p='fro')**2 / torch.norm(Xi, p='fro')**2
            asymmetric_overlap[i, j] = overlap

    return asymmetric_overlap


def get_super_experts(expert_activations, num_layers, num_experts_per_layer, top_k_ratio=0.2):
    """
    Compute expert scores and select super experts per layer.

    Args:
        expert_activations: dict[(layer_id, expert_id)] -> list of tensors
            Each tensor shape: [num_tokens_batch, hidden_dim] for activations of that expert
        num_layers: int, Total number of MoE layers
        num_experts_per_layer: int, Number of experts per layer
        top_k_ratio: float, Fraction of experts to keep as super experts per layer

    Returns:
        super_experts_per_layer: dict[layer_id] -> list of (expert_id, score) tuples
    """
    from collections import defaultdict
    
    # Compute per-expert scores
    expert_scores = {}
    for (layer_id, expert_id), batches in expert_activations.items():
        # Compute the max activation across batches in a memory-efficient way
        max_val = max(batch.abs().max().item() for batch in batches)
        expert_scores[(layer_id, expert_id)] = max_val

    # Sort experts per layer
    super_experts_per_layer = defaultdict(list)
    for layer in range(num_layers):
        # Filter experts for this layer
        layer_experts = [(expert_id, score) for (l, expert_id), score in expert_scores.items() if l == layer]
        # Sort by score descending
        layer_experts.sort(key=lambda x: x[1], reverse=True)
        # Keep top-k
        keep_num = int(num_experts_per_layer * top_k_ratio)
        super_experts_per_layer[layer] = layer_experts[:keep_num]

    return super_experts_per_layer


