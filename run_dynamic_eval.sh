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

python evaluation.py \
    --tasks mmlu \
    --batch_size 8 \
    --limit 1 \
    --use_pruned_model \
    --pruned_metadata "prune_experts/super_experts_ids.json" \
    --mode least \
    --pruning_method dynamic \
    --k 0 \
    --dynamic_routing_threshold 0.2 \
    --device cuda \
    --output_file "results_dynamic_mmlu.json"




