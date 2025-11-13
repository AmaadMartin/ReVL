#!/bin/bash
# Simple wrapper script to run batch evaluation with default settings
# This will submit sbatch jobs for all models that haven't been evaluated yet

cd /ocean/projects/cis240092p/amartin1/ReVL

# Create logs directory if it doesn't exist
mkdir -p eval/logs

echo "Starting batch evaluation..."
echo "This will submit jobs to the GPU cluster using sbatch"
echo ""

# load anaconda environment
module load anaconda3
module load cuda
module load nvhpc
conda activate py310

# activate wandb login
wandb login d73de72f4a6e9d226499f9c6da0c361a04336fde

# Run the batch evaluation script with auto-detection
# This will automatically extract K and context values from model directory names
python eval/python_scripts/batch_eval.py \
    --model_dir /ocean/projects/cis240092p/amartin1/ReVL/finetune/model_output \
    --eval_script /ocean/projects/cis240092p/amartin1/ReVL/eval/scripts/eval-screenspot-test.sh \
    --project ReVL_eval \
    --dataset screenspot_bbox_test \
    --log_dir /ocean/projects/cis240092p/amartin1/ReVL/eval/logs \
    --sbatch_partition GPU-shared \
    --sbatch_time "48:00:00" \
    --sbatch_gpus "h100-80:1" \
    --sbatch_nodes 1 \
    --base_path /ocean/projects/cis240092p/amartin1/ReVL \
    --auto_detect_k \
    --auto_detect_context \
    --context_values "true"

# To manually specify k and context values instead of auto-detection, remove the auto_detect flags:
# python eval/python_scripts/batch_eval.py \
#     --k_values "0,1,2,3" \
#     --context_values "true,false" \
#     (other args...)

echo ""
echo "Batch evaluation submission complete!"
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: eval/logs/"

