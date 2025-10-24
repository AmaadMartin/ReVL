#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export CUDA_VISIBLE_DEVICES=0

# Defaults (can be overridden via CLI args)
MODEL="text_to_point"
K="3"
DATA="screenspot_bbox_test"
CONTEXT="True"

# Usage helper
usage() {
    echo "Usage: $0 [--model NAME|-m NAME] [--k K|-k K] [--data NAME|-d NAME] [--context True|False|-c True|False]"
    exit 1
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"; shift 2 ;;
        -k|--k)
            K="$2"; shift 2 ;;
        -d|--data)
            DATA="$2"; shift 2 ;;
        -c|--context)
            CONTEXT="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

module load anaconda3
module load cuda
module load nvhpc

conda activate py310

source env/bin/activate

wandb login d73de72f4a6e9d226499f9c6da0c361a04336fde

python /ocean/projects/cis240092p/amartin1/ReVL/eval/eval.py --data "$DATA" --model "$MODEL" --k "$K" --context "$CONTEXT"