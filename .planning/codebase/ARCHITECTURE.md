# Architecture

**Analysis Date:** 2026-04-10

## Pattern Overview

**Overall:** Research pipeline / script-oriented with a shared utility library

**Key Characteristics:**
- No application server or web layer; all entry points are standalone Python scripts or Jupyter notebooks
- Two distinct pipeline modes: (1) profiling/pruning and (2) evaluation, both driven from `scripts/`
- All stateful model interaction is done through PyTorch forward hooks; no global singletons
- Outputs are persisted as JSON files and PNG plots under `outputs/`; no database

## Layers

**Entry Points (scripts / notebooks):**
- Purpose: Orchestrate end-to-end workflows; load models, invoke utilities, write results
- Location: `scripts/`, project root (`run_mmlu_categories_correlation.py`), `*.ipynb`
- Contains: Argument parsing (`argparse`), top-level pipeline orchestration, result serialization
- Depends on: All `utils/` modules, HuggingFace `transformers`, `lm_eval`
- Used by: Users directly via CLI or PBS job scheduler

**Utility Library (`utils/`):**
- Purpose: Reusable, independently importable functions and classes with no cross-dependencies between utility modules (except `router_utils` importing `hook_utils`)
- Location: `utils/`
- Contains: Hook management, statistics computation, model patching, data loading, visualization
- Depends on: `torch`, `transformers`, `datasets`, `numpy`, `scipy`, `matplotlib`, `seaborn`
- Used by: All scripts and notebooks

**Outputs (artifacts):**
- Purpose: Persist profiling statistics and evaluation results for later analysis
- Location: `outputs/statistics/`, `outputs/evaluation_results/`, `outputs/plots/`, `outputs/prune_experts/`
- Contains: JSON files (expert stats, prune lists, eval results), PNG plots
- Generated: Yes
- Committed: Only `.gitkeep` placeholders; data files are gitignored by convention (not explicitly listed in `.gitignore` but the directory is untracked)

## Data Flow

**Profiling and Expert Selection Pipeline:**

1. `scripts/profile_and_prune.py` loads an HuggingFace `AutoModelForCausalLM` with `device_map="auto"` (multi-GPU)
2. `utils/data_utils.py` prepares prompts from MMLU (`cais/mmlu`) or GSM8K (`openai/gsm8k`) datasets
3. `utils/hook_utils.ExpertActivationHook` registers `register_forward_hook` on each MoE expert's `down_proj` layer (classic style) or on the fused `Qwen3_5MoeExperts` module; simultaneously `RouterLogitHook` hooks each `gate` layer
4. `utils/router_utils.collect_router_logits` drives `model.generate()` or a single `model()` forward pass (prefill-only mode) to trigger hooks
5. `utils/analysis_utils.compute_all_stats` aggregates collected tensors into per-layer statistics: `mean_over_all`, `select_freq`, `cond_mean`, `mean_act`, `var_act`, `approx_contrib`
6. `scripts/profile_and_prune.determine_experts_to_prune` applies a threshold method (dynamic / absolute / percentile / std) to select low-importance expert indices per layer
7. Results are written as JSON to `outputs/statistics/experts_to_prune_<method>.json`

**Evaluation Pipeline:**

1. `scripts/evaluation.py` loads an `HFLM` wrapper (from `lm_eval`) around HuggingFace model
2. Optionally reads expert prune list from JSON via `utils/common_utils.get_experts_to_prune_from_json`
3. `utils/model_utils.apply_pruning` monkey-patches each MoE block's `forward` method in-place:
   - `mask` mode: sets pruned expert logits to `-inf` before softmax
   - `zero` mode: routes normally but zeroes out pruned expert outputs
4. `lm_eval.simple_evaluate` runs the benchmark task(s)
5. Results + latency metadata written as JSON to `outputs/evaluation_results/`

**Correlation Analysis Pipeline (`run_mmlu_categories_correlation.py`):**

1. Load model once; iterate over all four MMLU category groups
2. For each category: register hooks, run forward passes, clear hooks
3. `utils/analysis_utils.compute_correlations_by_layer` computes per-layer Spearman and Pearson correlations between `cond_mean` (router weight) and `mean_act` (expert activation norm)
4. `run_mmlu_categories_correlation.plot_correlations` saves PNG plots to `outputs/plots/`

**State Management:**
- No persistent runtime state; each script loads model fresh
- Hook data (tensors) lives in `ExpertActivationHook.expert_activations` dict keyed by `(layer_id, expert_id)` and `RouterLogitHook.router_logits` keyed by `layer_id`
- Hooks are always explicitly cleared after use via `clear_hooks()` to prevent GPU memory leaks

## Key Abstractions

**ExpertActivationHook (`utils/hook_utils.py`):**
- Purpose: Captures intermediate activations from each expert's feed-forward computation during a forward pass
- Pattern: PyTorch `register_forward_hook`; stores `dict[(layer_id, expert_id)] -> List[Tensor]`
- Handles two model styles: classic `nn.ModuleList` of individual expert modules (Qwen1.5-MoE) and fused `Qwen3_5MoeExperts` modules (Qwen3.5-MoE) via `_make_fused_hook`

**RouterLogitHook (`utils/hook_utils.py`):**
- Purpose: Captures gate/router output logits per MoE layer
- Pattern: `register_forward_hook` on `moe_block.gate`; stores `dict[layer_id] -> List[Tensor]`; concatenated on `get_router_logits()`

**apply_pruning (`utils/model_utils.py`):**
- Purpose: Non-destructive runtime pruning by monkey-patching `forward` on each `SparseMoeBlock`
- Pattern: Stores `pruned_experts_tensor` as an attribute on the block; replaces `moe_block.forward` with either `patched_forward_masked_experts` or `patched_forward_zeroed_experts` using `__get__` binding
- Does not modify model weights; pruning is purely routing-level

**compute_all_stats / compute_router_stats (`utils/analysis_utils.py`):**
- Purpose: Single source of truth for per-layer, per-expert statistics used by both pruning selection and correlation analysis
- Returns dict with keys: `mean_over_all`, `select_freq`, `cond_mean`, `mean_act`, `var_act`, `approx_contrib`

**MMLU_CATEGORIES (`utils/data_utils.py`):**
- Purpose: Central mapping of the four MMLU super-categories (`humanities`, `other`, `social_sciences`, `stem`) to their constituent subject strings
- Used by `prepare_mmlu_prompts` and referenced as valid CLI choices in `profile_and_prune.py`

## Entry Points

**`scripts/profile_and_prune.py`:**
- Location: `scripts/profile_and_prune.py`
- Triggers: `python scripts/profile_and_prune.py [OPTIONS]` or via PBS job via `scripts/submit_profile_jobs.sh`
- Responsibilities: Load model, prepare prompts, run hooks, compute statistics, select experts to prune, write JSON output

**`scripts/evaluation.py`:**
- Location: `scripts/evaluation.py`
- Triggers: `python scripts/evaluation.py [OPTIONS]` or via `scripts/run_evaluation.sh`
- Responsibilities: Load model (optionally patched with pruning), run `lm_eval.simple_evaluate`, write JSON results

**`run_mmlu_categories_correlation.py`:**
- Location: `run_mmlu_categories_correlation.py` (project root)
- Triggers: `python run_mmlu_categories_correlation.py`
- Responsibilities: Run hooks across all four MMLU categories, compute Spearman/Pearson correlations, save correlation PNG plots

**`analyze_expert_dynamics.ipynb` / `analyze_routing_statistics.ipynb`:**
- Location: project root
- Triggers: Jupyter notebook execution
- Responsibilities: Interactive exploratory analysis using `utils/` functions; ad-hoc visualization

**`scripts/run_evaluation.sh`:**
- Location: `scripts/run_evaluation.sh`
- Triggers: `bash scripts/run_evaluation.sh`
- Responsibilities: Shell wrapper to evaluate full model and multiple pruning configurations in sequence; prints latency summary

**`scripts/submit_profile_jobs.sh`:**
- Location: `scripts/submit_profile_jobs.sh`
- Triggers: `bash scripts/submit_profile_jobs.sh`
- Responsibilities: Submit PBS (`qsub`) batch jobs for profiling each MMLU topic on an HPC cluster with 2 GPUs

## Error Handling

**Strategy:** Minimal; errors propagate as unhandled exceptions terminating the script

**Patterns:**
- `utils/common_utils.get_experts_to_prune_from_json` raises `FileNotFoundError` with an informative message if the JSON path does not exist
- `utils/data_utils.prepare_mmlu_prompts` raises `ValueError` for unknown topic strings
- `utils/router_utils.collect_router_logits` silently falls back from `apply_chat_template` with `chat_template_kwargs` to without, catching `TypeError`, to support models without thinking-mode templates
- `utils/analysis_utils` catches missing `scipy` at import time and falls back to pure-numpy correlation implementations
- No retry logic or logging framework; stdout `print` statements used for progress reporting

## Cross-Cutting Concerns

**Logging:** `print()` statements throughout; no structured logging framework
**Validation:** CLI argument validation via `argparse` choices (e.g., `threshold_method`, `mmlu_topic`, `pruning_method`); minimal runtime validation
**GPU/Device management:** `device_map="auto"` in profiling scripts for automatic multi-GPU distribution; `CUDA_VISIBLE_DEVICES` set via `os.environ` before model load; input tensors moved to embedding-layer device obtained from `next(model.parameters()).device`
**Model compatibility:** Hook registration in `utils/hook_utils.py` explicitly handles both classic (Qwen1.5-MoE style) and fused (Qwen3.5-MoE style) MoE expert architectures via try/except on `list(experts)`

---

*Architecture analysis: 2026-04-10*
