#!/bin/bash

# Run dynamic routing evaluation on the full original model across a range
# of cumulative probability thresholds.

THRESHOLDS="0.3 0.35 0.4 0.45 0.5"

for threshold in $THRESHOLDS
do
    echo "Running evaluation with dynamic routing threshold: $threshold"

    output_filename="results_dynamic_mmlu_${threshold}.json"

    python scripts/evaluation.py \
        --tasks mmlu \
        --batch_size 8 \
        --limit 100 \
        --pruning_method dynamic \
        --dynamic_routing_threshold "$threshold" \
        --device cuda \
        --output_file "$output_filename"

    echo "Evaluation finished for threshold $threshold. Results saved to $output_filename"
    echo "----------------------------------------------------"
done

echo "All evaluations complete."




