# Coding Conventions

**Analysis Date:** 2026-04-10

## Naming Patterns

**Files:**
- `snake_case.py` for all modules: `hook_utils.py`, `router_utils.py`, `analysis_utils.py`, `model_utils.py`, `common_utils.py`, `data_utils.py`, `visualization_utils.py`
- `snake_case.py` for scripts: `profile_and_prune.py`, `evaluation.py`, `run_mmlu_categories_correlation.py`
- Shell scripts: `snake_case.sh` for runners: `run_evaluation.sh`, `submit_profile_jobs.sh`

**Classes:**
- `PascalCase` for all classes: `ExpertActivationHook`, `RouterLogitHook`, `PackedTextDataset`

**Functions:**
- `snake_case` for all functions: `compute_router_stats`, `collect_router_logits`, `apply_pruning`, `parse_args`, `profile_model`
- Functions in the same file are grouped by logical responsibility (e.g., `patched_forward_masked_experts` and `patched_forward_zeroed_experts` are together before `apply_pruning`)

**Variables:**
- `snake_case` throughout: `router_logits`, `expert_activations`, `experts_to_prune`, `hook_handles`
- Constants: `UPPER_CASE` at module level — `STATISTICS_DIR`, `PLOTS_DIR`, `MMLU_CATEGORIES`
- Loop variable shorthand `k` is avoided as a dict key variable name; prefer `layer`, `layer_id`, `expert_id` for clarity

**Type Hints:**
- Used selectively in class-based code and public function signatures in `hook_utils.py`, `router_utils.py`, `analysis_utils.py`
- Standard library types: `Dict`, `List`, `Tuple`, `Optional` imported from `typing` (Python 3.12 style but old-style `typing` imports used throughout)
- Examples:
  ```python
  self.expert_activations: Dict[Tuple[int, int], List[torch.Tensor]] = {}
  def collect_router_logits(..., temperature: Optional[float] = None, ...) -> dict:
  ```

**Parameters with defaults:**
- Optional parameters default to `None` when not applicable: `criterion=None`, `out_path=None`, `temperature=None`
- Boolean flags default `False` and use `action='store_true'` in argparse: `--use_pruned_model`, `--save_stats`, `--gsm8k`, `--enable_thinking`, `--prefill_only`

## Code Style

**Formatting:**
- No `.prettierrc`, `pyproject.toml`, or `ruff.toml` detected — no formatter enforced
- 4-space indentation throughout (PEP 8 standard)
- Line length is pragmatic — long lines in print statements and f-strings are not broken

**Linting:**
- No `.flake8`, `.pylintrc`, or linting config detected — no linter enforced

**Docstrings:**
- Google-style docstrings used consistently for all public functions and classes:
  ```python
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
  ```
- Module-level docstrings for all utility modules and scripts (e.g., `hook_utils.py`, `profile_and_prune.py`)
- Private/internal helpers (`_save`, `_make_fused_hook`) have brief inline comments rather than full docstrings

**Section Comments:**
- Visual section separators used in scripts and modules:
  ```python
  # -------------------
  # Argument parsing
  # -------------------
  ```
- In `profile_and_prune.py`: print banners with `"="*60` to mark phases at runtime

## Import Organization

**Order:**
1. Standard library: `os`, `json`, `sys`, `argparse`, `time`, `random`, `pathlib`
2. Third-party: `torch`, `numpy`, `transformers`, `datasets`, `tqdm`, `matplotlib`, `seaborn`, `scipy`
3. Local project: `from utils.hook_utils import ...`, `from utils.router_utils import ...`

**Path manipulation:**
- Scripts use `sys.path.insert(0, str(Path(__file__).parent.parent))` to add the project root:
  ```python
  sys.path.insert(0, str(Path(__file__).parent.parent))
  ```
- This pattern appears in `scripts/evaluation.py` and `scripts/profile_and_prune.py`
- Notebooks and root-level scripts import directly (they run from project root)

**Deferred imports:**
- Heavy or optional imports placed inside functions to avoid overhead at module load: `from tqdm import tqdm` inside `evaluate_model`, `from collections import defaultdict` inside `get_super_experts`
- `scipy.stats` uses a `try/except ImportError` pattern with pure-numpy fallback in `analysis_utils.py`

## Error Handling

**Patterns:**
- `raise ValueError(f"...")` for invalid enum-like arguments (mode, method, criterion, topic):
  ```python
  raise ValueError(f"Unknown mode: {mode}. Use 'mask' or 'zero'.")
  raise ValueError(f"Unknown method '{method}'. Choose 'percentile' or 'std'.")
  ```
- `raise FileNotFoundError(f"...")` for missing required files with guidance:
  ```python
  raise FileNotFoundError(f"{path} not found. Run the profile and prune script to produce it first.")
  ```
- `assert` for internal pre-conditions in analysis functions:
  ```python
  assert criterion in ("freq", "prob"), "criterion must be 'freq' or 'prob'"
  ```
- `try/except TypeError` for architecture branching (classic vs. fused expert layout) in `hook_utils.py` and `router_utils.py`:
  ```python
  try:
      expert_list = list(experts)
      ...
  except TypeError:
      # Fused style (e.g. Qwen3_5MoeExperts): not directly iterable
      handle = experts.register_forward_hook(...)
  ```
- `try/except ImportError` for optional dependencies with pure fallback (`scipy` in `analysis_utils.py`)
- No bare `except:` clauses — all are typed (`TypeError`, `ImportError`)

## Logging

**Framework:** `print()` — no logging framework used

**Patterns:**
- Progress banners with `"="*60` separators in long-running scripts
- `[latency]` prefixed timing lines for performance output:
  ```python
  print(f"[latency] eval: {eval_s:.2f}s" + ...)
  ```
- `tqdm` progress bars for loops over prompts and data batches:
  ```python
  for prompt in tqdm(prompts, desc="Collecting router logits"):
  ```
- Descriptive result counts: `f"Collected data for {len(router_logits)} layers"`

## Comments

**When to Comment:**
- Inline comments explain non-obvious PyTorch indexing and tensor operations
- Architecture-specific comments note MoE model variants (classic `ModuleList` vs. fused `Qwen3_5MoeExperts`)
- Numbered step comments in complex algorithms:
  ```python
  # Step 1: Construct low-rank bases and store full activations for each expert
  # Step 2: Initialize asymmetric overlap matrix
  # Step 3: Compute overlap for each pair of experts
  ```
- Emoji used in one comment (`# 1️⃣ Normalize to 0-1`) in `profile_and_prune.py` — non-standard; not adopted elsewhere

## Function Design

**Size:** Functions are medium-length (20–80 lines); long computations in `analysis_utils.py` stay in single functions rather than being split

**Parameters:** 
- Keyword arguments with defaults preferred for optional configuration
- `args` namespace from `argparse` passed through as a single object in scripts rather than unpacked individually
- Tensor operations operate directly on passed tensors without copying unless necessary (`.detach().cpu()` on activations)

**Return Values:**
- Functions return typed dicts with descriptive string keys for multi-value returns:
  ```python
  return {
      "mean_over_all": mean_over_all,
      "select_freq": select_freq,
      "cond_mean": cond_mean,
      ...
  }
  ```
- Hooks and hook managers return `None`; state is stored on `self`
- `getattr(module, 'attr', None)` pattern used for optional attributes set by monkey-patching

## Module Design

**Exports:** No `__all__` declarations — all public symbols are importable by name

**Barrel Files:** `utils/__init__.py` is empty — importers use explicit module paths:
```python
from utils.hook_utils import ExpertActivationHook
from utils.router_utils import collect_router_logits
from utils.analysis_utils import compute_all_stats
```

**Class design:**
- Hook manager classes (`ExpertActivationHook`, `RouterLogitHook`) follow a consistent interface:
  - `register_hooks(model)` to attach
  - `clear_hooks()` to detach
  - `clear_activations()` / `clear_logits()` to reset state
- Factory function `create_hook_manager()` provided as convenience wrapper in `hook_utils.py`

**Constants:**
- Module-level constants declared at top after imports: `STATISTICS_DIR = "outputs/statistics"`, `PLOTS_DIR = "outputs/plots"`, `MMLU_CATEGORIES = {...}`
- Hardcoded device/model settings in `run_mmlu_categories_correlation.py` main block (not parameterized — research script)

---

*Convention analysis: 2026-04-10*
