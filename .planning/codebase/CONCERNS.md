# Concerns

**Analysis Date:** 2026-04-10

## Tech Debt

### TD-01: No reproducible environment specification
- **File:** project root (missing `requirements.txt` / `pyproject.toml`)
- **Issue:** `transformers-5.6.0.dev0` is a dev build; no pinned dependencies anywhere in source control
- **Risk:** Environment cannot be reproduced from scratch; silent breakage when dev build is upgraded
- **Fix:** Add `requirements.txt` or `pyproject.toml` with pinned versions

### TD-02: Monkey-patching via `__get__` binding ✅ RESOLVED (2026-04-12)
- **File:** `utils/model_utils.py`
- **Issue:** `moe_block.forward` replaced using `patched_fn.__get__(moe_block, type(moe_block))` — breaks `torch.compile`, `DataParallel`, and model serialization
- **Resolution:** `apply_pruning` now reassigns `moe_block.__class__` to a cached dynamic subclass of the block's original type whose `forward` is the pruning implementation. The patched method lives on a real class, so torch.compile / DataParallel / serialization paths see it as a normal method. Commit `9492ad0`.

### TD-03: Asymmetric threshold formula with no justification
- **File:** `scripts/profile_and_prune.py:308` (dynamic pruning method)
- **Issue:** `threshold = mean_act_mid / 2` applied twice with no principled derivation; magic constant
- **Risk:** Produces arbitrary pruning decisions; results are not reproducible across model scales
- **Fix:** Document formula derivation or replace with principled criterion

### TD-04: Dead `output_final_logits` parameter ✅ RESOLVED (2026-04-12)
- **File:** `utils/router_utils.py`, `run_mmlu_categories_correlation.py`, `scripts/profile_and_prune.py`, `analyze_expert_dynamics.ipynb`
- **Issue:** `collect_router_logits` accepted `output_final_logits` but always returned an empty list under that key.
- **Resolution:** Removed the parameter and the placeholder return key from `collect_router_logits`; updated every call site (analysis script, profiling script, two notebook cells) to stop passing it. `collect_router_logits_from_loader` (a separate function that genuinely uses the flag) is left as-is. Commit `e21ce43`.

### TD-05: Hard-coded paths and constants ✅ RESOLVED (2026-04-12)
- **File:** `scripts/profile_and_prune.py`, `run_mmlu_categories_correlation.py`
- **Issue:** `STATISTICS_DIR` was a relative path; `cuda_visible_devices = "1"`, `model_name`, and `max_samples_per_subject` were hard-coded.
- **Resolution:** Both scripts now define `PROJECT_ROOT = Path(__file__).resolve().parent[.parent]` and resolve `STATISTICS_DIR` / `PLOTS_DIR` against it. `run_mmlu_categories_correlation.py` now exposes `--model_name`, `--cuda_visible_devices`, `--max_samples_per_subject`, and `--output_dir` via argparse with the original values as defaults. Commit `4754b19`.

### TD-06: `top_k=4` hard-coded for pruning ✅ RESOLVED (2026-04-12)
- **File:** `scripts/profile_and_prune.py`
- **Issue:** `compute_statistics` was being called with `top_k=4`, which only matches Qwen1.5-MoE.
- **Resolution:** `main()` now reads `top_k = model.config.num_experts_per_tok` and passes it to `compute_statistics`; the helper's default was removed so callers must pass it explicitly. Commit `b868442`.

## Known Bugs

### BUG-01: `plt.show()` dead code in visualization utils
- **File:** `utils/visualization_utils.py:79,115` (`plot_bar`, `plot_histogram`)
- **Issue:** `if ax is None: plt.show()` branch is unreachable — `ax` is always provided by callers
- **Risk:** No functional bug, but misleading code; interactive display never triggered

### BUG-02: Unbounded hook activation accumulation (OOM risk)
- **File:** `utils/hook_utils.py` — `ExpertActivationHook.expert_activations`
- **Issue:** Tensors accumulate across ALL prompts in a single dict without clearing between batches; no streaming stats
- **Risk:** Out-of-memory on large profiling runs (many prompts × many experts × activation size)
- **Fix:** Accumulate running statistics (mean/variance) online; discard raw tensors after each forward pass

## Performance Issues

### PERF-01: O(E² × H²) SVD loop in overlap matrix
- **File:** `utils/analysis_utils.py:522` (`compute_asymmetric_overlap_matrix`)
- **Issue:** Nested loop over expert pairs with full SVD per pair; no batching
- **Risk:** Prohibitively slow for large expert counts (e.g., 64+ experts)
- **Fix:** Batch SVD or use approximate overlap metrics

### PERF-02: `--max_new_tokens` defaults to 10,000
- **File:** `scripts/profile_and_prune.py` (argparse default)
- **Issue:** Default is orders of magnitude larger than needed for prefill-only profiling
- **Risk:** Massively inflated profiling runtime if default not overridden
- **Fix:** Default to 1 (prefill-only) with explicit opt-in for generation profiling

## Fragile Areas

### FRAG-01: Silent failure on unknown expert module types
- **File:** `utils/hook_utils.py` — hook registration
- **Issue:** `try/except TypeError` catches failures silently; hooks simply not registered for unknown architectures
- **Risk:** Zero activations collected with no warning; misleading empty outputs
- **Fix:** Log a warning with the unrecognised module type; raise for strict mode

### FRAG-02: Pruning only supports Qwen1.5/Qwen3.5-MoE structure
- **File:** `utils/model_utils.py`
- **Issue:** `patched_forward_masked_experts` and `patched_forward_zeroed_experts` assume specific MoE block API
- **Risk:** Silent incorrect behavior or crash on other MoE architectures (Mixtral, DeepSeek, etc.)
- **Fix:** Add architecture detection and model-specific patching strategies

### FRAG-03: `pruned_experts_tensor` device mismatch under `device_map="auto"` ✅ RESOLVED (2026-04-12)
- **File:** `utils/model_utils.py`
- **Issue:** Tensor created on `next(model.parameters()).device`; may not match device of the specific MoE layer.
- **Resolution:** Fixed incidentally as part of TD-02. `apply_pruning` now reads `next(moe_block.parameters()).device` per layer, so `pruned_experts_tensor` lands on the same device as the block under `device_map="auto"`. Commit `9492ad0`.

## Missing Capabilities

- **Zero test coverage** — no `tests/` directory, no pytest, no CI
- **No cross-architecture support** — profiling and pruning tested only on Qwen family
- **No statistical baselines** for evaluating pruning decisions (random pruning baseline, oracle upper bound)
- **No checkpoint/resume** for long profiling jobs — a failed PBS job loses all progress
- **No structured logging** — `print()` only; impossible to filter or redirect output

---

*Concerns analysis: 2026-04-10*
*Updated 2026-04-12: TD-02, TD-04, TD-05, TD-06, FRAG-03 resolved.*
