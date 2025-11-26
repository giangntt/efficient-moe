import os
import json

def get_experts_to_prune_from_json(path, k=20):
    """
    Load experts to prune from JSON. Select top k experts per layer.
    
    Args:
        path: Path to JSON file containing experts to prune
        k: Number of experts to select per layer (default: 20)
    
    Returns:
        dict: Dictionary mapping layer_id (int) to list of expert indices
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the profile and prune script to produce it first.")
    
    with open(path, "r") as f:
        experts_to_prune = json.load(f)

    # Select top k experts per layer
    experts_to_prune = {int(layer): experts[:k] for layer, experts in experts_to_prune.items()}

    return experts_to_prune

def get_topk_experts_from_json(path, top_k=5, mode="most", criterion=None, out_path=None):
    """
    Load sorted experts from JSON and select top-k most/least used/important per layer.

    Args:
        path: Path to sorted experts JSON file.
        top_k: Number of experts to select per layer.
        mode: 'most' or 'least'.
        criterion: 'freq', 'prob' or 'magnitude'.
        out_path: Optional path to save selected experts.

    Returns:
        experts_to_prune: dict[layer_idx] = list of expert indices.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the sorting cell to produce it first.")

    with open(path, "r") as f:
        sorted_experts = json.load(f)

    experts_to_prune = {}
    for k, v in sorted_experts.items():
        k_int = int(k)
        v_int = [int(x) for x in v]
        if mode == "most":
            selected = v_int[:top_k]
        else:
            selected = v_int[-top_k:]
        experts_to_prune[k_int] = selected

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in experts_to_prune.items()}, f, indent=4)

    num_layers = len(experts_to_prune)
    total_experts = sum(len(v) for v in experts_to_prune.values())
    print(f"Loaded '{path}' and produced top-{top_k} {mode} used/important experts per layer (criterion={criterion})" +
          (f" -> saved to '{out_path}'" if out_path else ""))
    print(f"Layers: {num_layers}, total experts selected: {total_experts}")

    return experts_to_prune


def save_top_target_experts_for_pruning(top_pairs, metric="conditional_probability", out_file=None):
    """
    Save top target experts per layer for pruning purposes.
    Duplicates are removed so each expert appears at most once per layer.

    Args:
        top_pairs: dict[layer_idx][metric_name] -> list of (i, j, score)
        metric: metric to extract
        out_file: path to JSON file

    Returns:
        prune_targets: dict[layer_id] -> list of expert IDs
    """
    prune_targets = {}

    for layer, metrics_dict in top_pairs.items():
        if metric not in metrics_dict:
            continue
        # Extract only expert_j and remove duplicates
        targets = [int(j) for i, j, score in metrics_dict[metric]]  # ensure Python int
        targets = sorted(list(set(targets)))  # unique & sorted
        prune_targets[str(layer)] = targets

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(prune_targets, f, indent=4)
        print(f"Top target experts per layer saved for pruning in '{out_file}'")
    return prune_targets


def prune_by_threshold(
    all_stats,
    method='percentile',
    router_param=20,
    act_param=20,
    save_path=None,
    layers_to_prune=None,
    max_prune_ratio=0.5
):
    """
    Select experts to prune using either percentile or std-based thresholds for cond_mean and mean_act,
    but limit pruning to at most max_prune_ratio of experts per layer.

    Args:
        all_stats: dict[layer_id] -> stats dict from compute_all_stats
        method: 'percentile' or 'std'
        router_param: percentile (0-100) or std multiplier for cond_mean
        act_param: percentile (0-100) or std multiplier for mean_act
        save_path: optional path to save JSON
        layers_to_prune: list of layer IDs to consider
        max_prune_ratio: maximum allowed ratio of experts to prune per layer (default 0.5)

    Returns:
        dict[layer_id] -> list of expert IDs to prune
    """
    import numpy as np
    
    experts_to_prune = {}

    if layers_to_prune is None:
        layers_to_prune = all_stats.keys()

    for layer_id, stats in all_stats.items():
        if layer_id not in layers_to_prune:
            continue

        cond_mean = np.array(stats["cond_mean"])
        mean_act = np.array(stats["mean_act"])
        expert_ids = np.arange(len(cond_mean))
        num_experts = len(expert_ids)

        # compute thresholds
        if method == 'percentile':
            router_thresh = np.percentile(cond_mean, router_param)
            act_thresh = np.percentile(mean_act, act_param)
        elif method == 'std':
            router_thresh = cond_mean.mean() - router_param * cond_mean.std()
            act_thresh = mean_act.mean() - act_param * mean_act.std()
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'percentile' or 'std'.")

        # select experts below both thresholds
        selected = expert_ids[(cond_mean <= router_thresh) & (mean_act <= act_thresh)]

        # enforce max prune ratio
        max_prune = int(num_experts * max_prune_ratio)
        if len(selected) > max_prune:
            # sort by combined score (low cond_mean & low mean_act)
            combined_score = cond_mean + mean_act
            sorted_idx = np.argsort(combined_score)
            selected = sorted_idx[:max_prune]

        experts_to_prune[layer_id] = selected.tolist()

    # remove empty layers
    experts_to_prune = {k: v for k, v in experts_to_prune.items() if len(v) > 0}

    # save to JSON if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(experts_to_prune, f, indent=2)
        print(f"Saved experts to prune to {save_path}")

    return experts_to_prune