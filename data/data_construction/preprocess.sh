#!/bin/bash
DIR=`pwd`

source "../../env/bin/activate"
# pip3 install tqdm
python3 -u preprocess_images.py