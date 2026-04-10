# Stack

**Analysis Date:** 2026-04-10

## Language & Runtime

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Via `.venv` virtualenv at project root |
| CUDA | 12.4 | `cu124` build of PyTorch |
| Platform | Linux (HPC cluster) | PBS/Torque job scheduler; multi-GPU nodes |

## Core Frameworks

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.6.0+cu124 | Tensor computation, model inference, forward hooks |
| Transformers (HF) | 5.6.0.dev0 | Model loading (`AutoModelForCausalLM`), tokenization |
| lm-eval | 0.4.11 | Standardized LLM benchmark evaluation (`simple_evaluate`) |
| Datasets (HF) | Latest stable | MMLU and GSM8K dataset loading |
| Accelerate (HF) | Latest | `device_map="auto"` multi-GPU distribution |

## Scientific / Numeric

| Library | Purpose |
|---------|---------|
| NumPy | Array operations, statistics |
| SciPy | Pearson/Spearman correlation (`scipy.stats`); soft import with numpy fallback |
| Matplotlib | Plot generation (PNG output) |
| Seaborn | Statistical plot styling |

## Environment Management

- **Virtualenv:** `.venv/` at project root (Python 3.12)
- **No `requirements.txt` or `pyproject.toml`** — environment not pinned in source control
- **Dev install note:** `transformers` is installed as a dev build (`5.6.0.dev0`), not a stable release

## Configuration

- No configuration files (YAML/TOML/JSON) — all configuration passed via CLI arguments (`argparse`)
- `CUDA_VISIBLE_DEVICES` set programmatically via `os.environ` before model loading
- HPC job configuration embedded in `scripts/submit_profile_jobs.sh` (PBS directives)

## HPC / Job Scheduling

- **PBS/Torque:** `qsub` used to submit profiling jobs per MMLU category (4 parallel jobs)
- GPU allocation: `select=1:ncpus=8:mem=40gb:ngpus=2` per job
- Wall time: 24h per profiling job

## Key CLI Entry Points

```bash
# Profile experts
python scripts/profile_and_prune.py --model_name <model> --mmlu_topic stem [OPTIONS]

# Run evaluation
python scripts/evaluation.py --model_name <model> [--prune_experts_path <json>] [OPTIONS]

# Submit HPC profiling jobs
bash scripts/submit_profile_jobs.sh

# Run evaluation suite
bash scripts/run_evaluation.sh
```

---

*Stack analysis: 2026-04-10*
