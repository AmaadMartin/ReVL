#!/bin/bash
DIR=`pwd`

module load anaconda3
module load cuda
module load nvhpc
module load gcc

conda activate py310

source env/bin/activate

python experiment_construction.py