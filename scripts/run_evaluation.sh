#!/bin/bash

# Configuration
# =============

# Evaluation tasks - customize this list as needed

# Full evaluation with all MMLU tasks:
# TASKS="mmlu_stem mmlu_social_sciences mmlu_humanities mmlu_other"

TASKS="mmlu_stem"
BATCH_SIZE=8
# Set LIMIT to a number to limit examples, or leave empty/unset for no limit
LIMIT=30
# LIMIT=""  # Uncomment this line and comment the line above to run without limit
DEVICE="cuda"
BASE_OUTPUT_DIR="outputs/evaluation_results"
PRUNED_METADATA_DIR="outputs/statistics"

# Set to "true" to evaluate full model, "false" to skip
EVALUATE_FULL_MODEL="false"

# Build limit argument (only if LIMIT is set and not empty)
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# 1. Evaluate full model (if enabled)
if [ "$EVALUATE_FULL_MODEL" = "true" ]; then
    echo "1. Evaluating FULL MODEL..."
    python3 scripts/evaluation.py \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        $LIMIT_ARG \
        --device $DEVICE \
        --output_file "$BASE_OUTPUT_DIR/full_model.json"
    
    echo "Full model evaluation completed."
    echo "=========================================="
else
    echo "Skipping full model evaluation (EVALUATE_FULL_MODEL=false)"
    echo "=========================================="
fi

# 2. Evaluate pruned models with different criteria
echo "2. Evaluating PRUNED MODELS..."

# Define pruning configurations
# Format: "metadata_file:pruning_method:k:output_suffix"
declare -a PRUNING_CONFIGS=(
    "experts_to_prune_mean_var_act_dynamic_max30.json:zero:20:experts_to_prune_mean_var_act_dynamic_zero"
)

# Run evaluations for each pruning configuration
for config in "${PRUNING_CONFIGS[@]}"; do
    IFS=':' read -r metadata_file pruning_method k output_suffix <<< "$config"
    
    echo "Evaluating: $output_suffix"
    echo "  - Metadata: $metadata_file"
    echo "  - Pruning method: $pruning_method"
    echo "  - K: $k"
    
    python3 scripts/evaluation.py \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        $LIMIT_ARG \
        --use_pruned_model \
        --pruned_metadata "$PRUNED_METADATA_DIR/$metadata_file" \
        --pruning_method $pruning_method \
        --k $k \
        --device $DEVICE \
        --output_file "$BASE_OUTPUT_DIR/${output_suffix}_results.json"
    
    echo "  Completed: $output_suffix"
    echo "  ----------------------------------------"
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "=========================================="

