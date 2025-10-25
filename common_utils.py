import os
import json

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