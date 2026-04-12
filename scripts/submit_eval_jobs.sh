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
OUTPUT_DIR="outputs/evaluation_results"
STATS_DIR="outputs/statistics"

TOPICS=("stem" "humanities" "other" "social_sciences")

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

\$PBS_O_WORKDIR/.venv/bin/python scripts/evaluation.py \
    --model_name ${MODEL} \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --limit ${LIMIT} \
    --parallelize \
    ${EXTRA_ARGS} \
    --output_file ${OUTPUT_FILE}
EOF
)

    echo "$JOB_SCRIPT" | qsub
    echo "Submitted job: ${JOB_NAME} -> ${OUTPUT_FILE}"
}

for TOPIC in "${TOPICS[@]}"; do
    TASK="mmlu_${TOPIC}"
    METADATA="${STATS_DIR}/expert_stats_${TOPIC}.json"

    # 1. Original model
    submit_eval_job \
        "eval_${TOPIC}_original" \
        "${TASK}" \
        "" \
        "${OUTPUT_DIR}/${TOPIC}_original.json"

    # 2. Pruned model — zero method
    submit_eval_job \
        "eval_${TOPIC}_pruned_zero" \
        "${TASK}" \
        "--use_pruned_model --pruned_metadata ${METADATA} --pruning_method zero" \
        "${OUTPUT_DIR}/${TOPIC}_pruned_zero.json"

    # 3. Pruned model — mask method
    submit_eval_job \
        "eval_${TOPIC}_pruned_mask" \
        "${TASK}" \
        "--use_pruned_model --pruned_metadata ${METADATA} --pruning_method mask" \
        "${OUTPUT_DIR}/${TOPIC}_pruned_mask.json"
done
