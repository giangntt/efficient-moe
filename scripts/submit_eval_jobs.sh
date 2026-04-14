#!/bin/bash

# Submit PBS batch jobs to evaluate each MMLU topic on original and pruned models.
# Usage: bash scripts/submit_eval_jobs.sh
#
# For each MMLU topic, evaluates three variants:
#   1. Original model (no pruning)
#   2. Pruned model — zero method
#   3. Pruned model — mask method
# Pruned variants use the per-topic statistics produced by submit_profile_jobs.sh.

MODEL="$HOME/scratch/models/Qwen3-30B-A3B"
BATCH_SIZE=8
LIMIT=100
DEVICES="0,1"
MAX_PRUNED_EXPERTS_PER_LAYER=50
OUTPUT_DIR="outputs/evaluation_results"
# Reuse k=60 metadata; the eval loader slices experts[:k] so k<=60 needs no reprofiling.
STATS_DIR="outputs/statistics/max_pruned_experts_60"

TOPICS=("stem" "humanities" "other" "social_sciences")
EXTRA_TASKS=("humaneval")

PROJECT="13004345"
QUEUE="normal"
WALLTIME="2:00:00"
SELECT="1:ngpus=2"

mkdir -p "$OUTPUT_DIR" logs

# ── Helper: submit one evaluation job ─────────────────────────────────────────
submit_eval_job() {
    local JOB_NAME="$1"
    local TASK="$2"
    local EXTRA_ARGS="$3"
    local OUTPUT_FILE="$4"

    JOB_SCRIPT=$(cat <<EOF
#!/bin/bash
#PBS -l select=${SELECT}
#PBS -l walltime=${WALLTIME}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N ${JOB_NAME}
#PBS -o logs/${JOB_NAME}.out
#PBS -e logs/${JOB_NAME}.err

cd \$PBS_O_WORKDIR

export CUDA_VISIBLE_DEVICES=${DEVICES}
export HF_ALLOW_CODE_EVAL=1

\$PBS_O_WORKDIR/.venv/bin/python scripts/evaluation.py \
    --model_name ${MODEL} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --limit ${LIMIT} \
    --confirm_run_unsafe_code \
    ${EXTRA_ARGS} \
    --output_file ${OUTPUT_FILE}
EOF
)

    echo "$JOB_SCRIPT" | qsub
    echo "Submitted job: ${JOB_NAME} -> ${OUTPUT_FILE}"
}

# ── Helper: submit all three variants (original + pruned zero/mask) ───────────
submit_variants() {
    local NAME="$1"   # short identifier used in job/output names
    local TASK="$2"   # lm_eval task name
    local METADATA="${STATS_DIR}/expert_stats_${NAME}.json"

    # 1. Original model
    submit_eval_job \
        "eval_${NAME}_original" \
        "${TASK}" \
        "" \
        "${OUTPUT_DIR}/${NAME}_original.json"

    # 2. Pruned model — zero method
    submit_eval_job \
        "eval_${NAME}_pruned_zero" \
        "${TASK}" \
        "--use_pruned_model --pruned_metadata ${METADATA} --pruning_method zero --k ${MAX_PRUNED_EXPERTS_PER_LAYER}" \
        "${OUTPUT_DIR}/${NAME}_pruned_zero.json"

    # 3. Pruned model — mask method
    submit_eval_job \
        "eval_${NAME}_pruned_mask" \
        "${TASK}" \
        "--use_pruned_model --pruned_metadata ${METADATA} --pruning_method mask --k ${MAX_PRUNED_EXPERTS_PER_LAYER}" \
        "${OUTPUT_DIR}/${NAME}_pruned_mask.json"
}

for TOPIC in "${TOPICS[@]}"; do
    submit_variants "${TOPIC}" "mmlu_${TOPIC}"
done

for TASK in "${EXTRA_TASKS[@]}"; do
    submit_variants "${TASK}" "${TASK}"
done
