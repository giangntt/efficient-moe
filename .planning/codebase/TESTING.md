# Testing Patterns

**Analysis Date:** 2026-04-10

## Test Framework

**Runner:**
- None detected. No `pytest`, `unittest`, or any test runner is installed in `.venv` or present in the codebase.
- No `pytest.ini`, `setup.cfg [tool:pytest]`, `pyproject.toml [tool.pytest]`, or `conftest.py` exists.

**Assertion Library:**
- Not applicable — no automated tests exist.

**Run Commands:**
```bash
# No test commands defined
```

## Test File Organization

**Location:**
- No test files exist. `find . -name "test_*.py" -o -name "*_test.py"` returns empty.
- The `.gitignore` includes standard `htmlcov/`, `.coverage`, `*.cover` sections, suggesting tests were anticipated at project scaffolding time but never written.

**Naming:**
- Not established.

**Structure:**
- Not established.

## Test Structure

**Suite Organization:**
- Not applicable.

**Patterns:**
- Not applicable.

## Mocking

**Framework:** None detected.

**Patterns:**
- Not applicable.

**What to Mock:**
- If tests were added, the following would require mocking:
  - `AutoModelForCausalLM.from_pretrained` / `AutoTokenizer.from_pretrained` — heavy model loads
  - `load_dataset` from `datasets` — network calls
  - `torch.cuda.is_available()` — GPU availability
  - `model.generate()` — GPU inference

**What NOT to Mock:**
- Pure computation functions in `analysis_utils.py` (e.g., `compute_router_stats`, `compute_correlations_by_layer`) — these are CPU/NumPy operations that can be tested with synthetic tensors directly
- `PackedTextDataset.__getitem__` — pure Python logic with no external dependencies

## Fixtures and Factories

**Test Data:**
- Not applicable — no test infrastructure exists.
- If added, synthetic data could be generated with:
  ```python
  # Example synthetic router logits for a 2-layer, 4-expert model
  router_logits = {
      0: torch.randn(100, 4),  # [num_tokens, num_experts]
      1: torch.randn(100, 4),
  }
  expert_activations = {
      (0, 0): [torch.randn(10, 64)],
      (0, 1): [torch.randn(8, 64)],
  }
  ```

**Location:**
- Not established.

## Coverage

**Requirements:** None enforced.

**View Coverage:**
```bash
# Not configured
```

## Test Types

**Unit Tests:**
- Not present. Priority candidates for unit tests:
  - `utils/common_utils.py`: `get_experts_to_prune_from_json` (file I/O + dict slicing), `prune_by_threshold` (numpy logic)
  - `utils/analysis_utils.py`: `compute_router_stats`, `compute_correlations_by_layer`, `sort_experts_by_usage_or_prob`, `get_top_pairs`
  - `utils/model_utils.py`: `apply_pruning` (monkey-patching behavior)
  - `utils/data_utils.py`: `PackedTextDataset` (chunking logic, `__len__`, `__getitem__`), `format_mmlu_sample`

**Integration Tests:**
- Not present. Priority candidates:
  - End-to-end profiling pipeline: `profile_model` → `compute_statistics` → `determine_experts_to_prune`
  - Pruning + evaluation round-trip using a tiny model or stub

**E2E Tests:**
- Not used.

## Common Patterns

**Async Testing:**
- Not applicable — no async code.

**Error Testing:**
- Not applicable — no tests exist.
- Key error paths that warrant coverage if tests are added:
  - `FileNotFoundError` from `get_experts_to_prune_from_json` when path is missing
  - `ValueError` from `apply_pruning` with invalid `mode`
  - `ValueError` from `prepare_mmlu_prompts` with unknown topic
  - `ValueError` from `load_prompts` when no data source flag is provided

## Validation Approach (Current Practice)

In the absence of automated tests, the project relies on:

1. **Manual script execution** — scripts print descriptive progress and result summaries to stdout (e.g., `f"Collected data for {len(router_logits)} layers"`, layer-by-layer pruning summaries from `summarize_pruned_experts`)
2. **Visual inspection** — Jupyter notebooks (`analyze_expert_dynamics.ipynb`, `analyze_routing_statistics.ipynb`) serve as exploratory validation environments
3. **Output file inspection** — results are saved as JSON to `outputs/statistics/` and `outputs/evaluation_results/` for post-hoc review
4. **Latency reporting** — `[latency]` prefixed timing lines and per-question millisecond reporting in evaluation scripts give a proxy for correctness (unexpected slowdowns surface bugs)

---

*Testing analysis: 2026-04-10*
