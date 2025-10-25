#!/bin/bash

python /mnt/ssd/shared/giangntt/expert_analysis/clean_code/evaluation.py \
    --tasks gsm8k \
    --batch_size 8 \
    --limit 100 \
    --use_pruned_model \
    --pruned_metadata "/mnt/ssd/shared/giangntt/expert_analysis/clean_code/prune_experts/super_experts_ids.json" \
    --mode least \
    --pruning_method zero \
    --k 10 \
    --device cuda \
    # --output_file "/mnt/ssd/shared/giangntt/expert_analysis/clean_code/results.json"
