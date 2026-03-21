#!/bin/bash
mkdir -p /cache/output

echo "=== SHELL STARTED ===" > /cache/output/shell_test.log
date >> /cache/output/shell_test.log
echo "USER=$(whoami)" >> /cache/output/shell_test.log
echo "PWD=$(pwd)" >> /cache/output/shell_test.log
echo "PYTHON=$(which python 2>&1)" >> /cache/output/shell_test.log

# Activate conda
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate PyTorch-2.1.0

echo "CONDA_ENV=$CONDA_DEFAULT_ENV" >> /cache/output/shell_test.log
echo "PYTHON_AFTER=$(which python 2>&1)" >> /cache/output/shell_test.log
python -c "print('python works')"
