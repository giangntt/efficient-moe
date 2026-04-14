#!/bin/bash

# Submit PBS batch jobs to evaluate dynamic routing at various thresholds.
# Usage: bash scripts/submit_dynamic_eval_jobs.sh
#
# For each MMLU topic + extra task, sweeps cumulative probability thresholds.
# Dynamic routing is applied to the full original model (no pruning).

MODEL="$HOME/scratch/models/Qwen3-30B-A3B"
BATCH_SIZE=8
LIMIT=100
DEVICES="0,1"
OUTPUT_DIR="outputs/evaluation_results/dynamic"
RUN_TAG="limit${LIMIT}"

TOPICS=("stem" "humanities" "other" "social_sciences")
EXTRA_TASKS=("humaneval" "aime25")
THRESHOLDS=("0.3" "0.4" "0.5" "0.6" "0.7")

PROJECT="13004345"
QUEUE="normal"
WALLTIME="3:00:00"
SELECT="1:ngpus=2"

mkdir -p "$OUTPUT_DIR" logs

# ── Helper: submit one dynamic routing evaluation job ────────────────────────
submit_dynamic_job() {
    local JOB_NAME="$1"
    local TASK="$2"
    local THRESHOLD="$3"
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
    --pruning_method dynamic \
    --dynamic_routing_threshold ${THRESHOLD} \
    --output_file ${OUTPUT_FILE}
EOF
)

    echo "$JOB_SCRIPT" | qsub
    echo "Submitted job: ${JOB_NAME} -> ${OUTPUT_FILE}"
}

# ── Submit threshold sweep for a given task ──────────────────────────────────
submit_threshold_sweep() {
    local NAME="$1"   # short identifier used in job/output names
    local TASK="$2"   # lm_eval task name

    for THRESHOLD in "${THRESHOLDS[@]}"; do
        submit_dynamic_job \
            "eval_${NAME}_dynamic_${THRESHOLD}_${RUN_TAG}" \
            "${TASK}" \
            "${THRESHOLD}" \
            "${OUTPUT_DIR}/${NAME}_dynamic_${THRESHOLD}_${RUN_TAG}.json"
    done
}

for TOPIC in "${TOPICS[@]}"; do
    submit_threshold_sweep "${TOPIC}" "mmlu_${TOPIC}"
done

for TASK in "${EXTRA_TASKS[@]}"; do
    submit_threshold_sweep "${TASK}" "${TASK}"
done
