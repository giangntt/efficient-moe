#!/bin/bash

# This script runs evaluation using the dynamic routing method on the original model.
#
# Why `--use_pruned_model` and `--pruned_metadata` are used:
# The `apply_pruning` function in `evaluation.py`, which patches the model for
# dynamic routing, is only called when the `--use_pruned_model` flag is set.
# This requires a metadata file, but we can prevent any experts from being
# pruned by setting `--k 0`.
#
# This setup allows us to apply the dynamic routing logic to the full, original model.

# Define the thresholds to test
THRESHOLDS="0.3 0.35 0.4 0.45 0.5"

# Loop over each threshold and run the evaluation
for threshold in $THRESHOLDS
do
    echo "Running evaluation with dynamic routing threshold: $threshold"
    
    # Construct the output filename
    output_filename="results_dynamic_mmlu_${threshold}.json"

    # Run the evaluation script
    python scripts/evaluation.py \
        --tasks mmlu \
        --batch_size 12 \
        --limit 300 \
        --use_pruned_model \
        --pruned_metadata "prune_experts/super_experts_ids.json" \
        --pruning_method dynamic \
        --k 0 \
        --dynamic_routing_threshold "$threshold" \
        --device cuda \
        --output_file "$output_filename"

    echo "Evaluation finished for threshold $threshold. Results saved to $output_filename"
    echo "----------------------------------------------------"
done

echo "All evaluations complete."




