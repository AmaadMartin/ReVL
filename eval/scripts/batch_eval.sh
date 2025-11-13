#!/bin/bash
# Example script to run batch evaluation on all models
# This will check wandb for existing runs and only evaluate models that haven't been tested yet

# Navigate to the ReVL directory
cd /ocean/projects/cis240092p/amartin1/ReVL

# Run the batch evaluation script
# By default, it will:
# - Check all models in finetune/model_output
# - Compare against runs in the ReVL_eval wandb project
# - Evaluate on screenspot_bbox_test dataset
# - Test with k values: 0, 1, 2, 3
# - Test with context: true and false

python eval/python_scripts/batch_eval.py \
    --model_dir /ocean/projects/cis240092p/amartin1/ReVL/finetune/model_output \
    --eval_script /ocean/projects/cis240092p/amartin1/ReVL/eval/scripts/eval-screenspot-test.sh \
    --project ReVL_eval \
    --dataset screenspot_bbox_test \
    --k_values "0,1,2,3" \
    --context_values "true,false" \
    --base_path /ocean/projects/cis240092p/amartin1/ReVL

# To do a dry run (see what would be evaluated without running):
# python eval/python_scripts/batch_eval.py --dry_run

# To evaluate only specific k values:
# python eval/python_scripts/batch_eval.py --k_values "0,1"

# To evaluate only with context enabled:
# python eval/python_scripts/batch_eval.py --context_values "true"

