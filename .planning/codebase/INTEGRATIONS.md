# Integrations

**Analysis Date:** 2026-04-10

## External Services

### Hugging Face Hub

- **Type:** Model weights & dataset hosting
- **Access:** At runtime via `transformers.AutoModelForCausalLM.from_pretrained()` and `datasets.load_dataset()`
- **Authentication:** No auth required for public models/datasets used in this project
- **Network dependency:** Internet access required on first run; weights/datasets cached locally by HF cache mechanisms
- **Models used:** Qwen1.5-MoE, Qwen3.5-MoE (and variants, specified via CLI `--model_name`)

## Datasets

| Dataset | HF ID | Usage |
|---------|-------|-------|
| MMLU | `cais/mmlu` | Primary profiling and evaluation benchmark; 4 super-categories (humanities, other, social_sciences, stem) |
| GSM8K | `openai/gsm8k` | Secondary evaluation benchmark (math reasoning) |

- Loaded via `utils/data_utils.py` → `datasets.load_dataset()`
- Prompts formatted in `prepare_mmlu_prompts()` / `prepare_gsm8k_prompts()`

## HPC / Compute Platform

- **Scheduler:** PBS/Torque (`qsub`)
- **Integration:** `scripts/submit_profile_jobs.sh` emits PBS directives; no API — shell-level integration only
- **Node spec:** 8 CPUs, 40 GB RAM, 2 GPUs per job

## No Other External Integrations

This project has:
- No databases (relational or vector)
- No REST APIs (outbound or inbound)
- No authentication providers
- No cloud storage (S3, GCS, etc.)
- No message queues or streaming infrastructure
- No monitoring or observability services
- No webhooks

All outputs are written to local filesystem under `outputs/`.

---

*Integrations analysis: 2026-04-10*
