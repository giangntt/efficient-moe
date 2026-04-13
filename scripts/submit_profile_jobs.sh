#!/bin/bash

# Submit PBS batch jobs for profiling each MMLU topic separately.
# Usage: bash scripts/submit_profile_jobs.sh

MODEL="$HOME/scratch/models/Qwen3-30B-A3B"
SAMPLE_SIZE=50
MAX_PRUNED_EXPERTS_PER_LAYER=60
OUTPUT_DIR="outputs/statistics"
DEVICES="0,1"
PROJECT="13004345"
QUEUE="normal"
WALLTIME="8:00:00"
SELECT="1:ngpus=2"

TOPICS=("stem" "humanities" "other" "social_sciences")
EXTRA_DATASETS=("aime25" "humaneval")

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

submit_job() {
    local NAME="$1"
    local DATA_FLAG="$2"
    local OUTPUT_FILE="$OUTPUT_DIR/expert_stats_${NAME}.json"

    local JOB_SCRIPT
    JOB_SCRIPT=$(cat <<EOF
#!/bin/bash
#PBS -l select=${SELECT}
#PBS -l walltime=${WALLTIME}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -N profile_${NAME}
#PBS -o logs/profile_${NAME}.out
#PBS -e logs/profile_${NAME}.err

cd \$PBS_O_WORKDIR

\$PBS_O_WORKDIR/.venv/bin/python scripts/profile_and_prune.py \
    --model_name ${MODEL} \
    ${DATA_FLAG} \
    --sample_size ${SAMPLE_SIZE} \
    --output_file ${OUTPUT_FILE} \
    --cuda_visible_devices ${DEVICES} \
    --max_pruned_experts_per_layer ${MAX_PRUNED_EXPERTS_PER_LAYER} \
    --prefill_only
EOF
)

    echo "$JOB_SCRIPT" | qsub
    echo "Submitted job: $NAME -> $OUTPUT_FILE"
}

for TOPIC in "${TOPICS[@]}"; do
    submit_job "$TOPIC" "--mmlu_topic ${TOPIC}"
done

for DATASET in "${EXTRA_DATASETS[@]}"; do
    submit_job "$DATASET" "--${DATASET}"
done
