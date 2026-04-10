# Structure

**Analysis Date:** 2026-04-10

## Directory Layout

```
efficient-moe/
├── scripts/                    # CLI entry points and HPC job scripts
│   ├── profile_and_prune.py    # Main profiling + expert selection pipeline
│   ├── evaluation.py           # lm-eval benchmark runner (with optional pruning)
│   ├── run_evaluation.sh       # Shell wrapper: run eval suite sequentially
│   └── submit_profile_jobs.sh  # PBS job submission for HPC profiling
│
├── utils/                      # Reusable utility library
│   ├── __init__.py             # Empty package marker
│   ├── hook_utils.py           # PyTorch forward hook classes (ExpertActivationHook, RouterLogitHook)
│   ├── model_utils.py          # Model patching / runtime pruning (apply_pruning)
│   ├── analysis_utils.py       # Statistics computation and correlation analysis
│   ├── data_utils.py           # Dataset loading and prompt preparation (MMLU, GSM8K)
│   ├── router_utils.py         # Forward pass execution and router logit collection
│   ├── visualization_utils.py  # Matplotlib/Seaborn plot helpers
│   └── common_utils.py         # Shared helpers (JSON loading, etc.)
│
├── outputs/                    # Generated artifacts (gitignored data, only .gitkeep committed)
│   ├── statistics/             # Expert activation stats JSON (profile_and_prune output)
│   ├── evaluation_results/     # lm-eval benchmark results JSON
│   ├── plots/                  # Correlation / analysis PNG plots
│   └── prune_experts/          # Pruning configuration JSON files
│
├── logs/                       # HPC PBS job stdout/stderr logs
│
├── analyze_expert_dynamics.ipynb       # Notebook: interactive expert analysis
├── analyze_routing_statistics.ipynb    # Notebook: routing statistics exploration
├── run_mmlu_categories_correlation.py  # Standalone script: cross-category correlation analysis
│
├── README.md
├── .gitignore
└── .venv/                      # Python virtualenv (not committed)
```

## Key File Locations

| What | Where |
|------|-------|
| Main profiling script | `scripts/profile_and_prune.py` |
| Evaluation runner | `scripts/evaluation.py` |
| Forward hook classes | `utils/hook_utils.py` |
| Runtime pruning logic | `utils/model_utils.py` |
| Statistics computation | `utils/analysis_utils.py` |
| Dataset preparation | `utils/data_utils.py` |
| MMLU category mapping | `utils/data_utils.py` → `MMLU_CATEGORIES` dict |
| Expert prune JSON outputs | `outputs/statistics/experts_to_prune_*.json` |
| Evaluation result JSONs | `outputs/evaluation_results/*.json` |
| HPC job submission | `scripts/submit_profile_jobs.sh` |

## Naming Conventions

- **Scripts:** `snake_case.py`, imperative verb-noun (`profile_and_prune`, `run_evaluation`)
- **Utils:** `snake_case.py`, noun-describing domain (`hook_utils`, `analysis_utils`)
- **Output files:** `<description>_<method>_<variant>.json` (e.g., `experts_to_prune_mean_var_act_dynamic_max30.json`)
- **Classes:** `PascalCase` (e.g., `ExpertActivationHook`, `RouterLogitHook`)
- **Functions:** `snake_case` verb-noun (e.g., `compute_all_stats`, `apply_pruning`, `prepare_mmlu_prompts`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MMLU_CATEGORIES`, `STATISTICS_DIR`)

## Notes

- No `src/` layout — flat project root with `scripts/` and `utils/` top-level
- No `tests/` directory — zero test coverage
- `outputs/` subdirectories tracked via `.gitkeep` only; actual data files are untracked
- PBS log files in `logs/` are untracked (listed in `??` git status)

---

*Structure analysis: 2026-04-10*
