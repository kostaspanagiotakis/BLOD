#!/bin/bash
#SBATCH --job-name=mi_tre_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mi_tre_%j.out
#SBATCH --error=logs/mi_tre_%j.err

set -euo pipefail

mkdir -p logs

cd /fs04/sq98/BLOD/TRE
source /home/kpanagio/sq98/BLOD/venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Install Python requirements for MI / TRE experiments
python -m pip install --upgrade pip
python -m pip install -r requirements_tre_1d.txt

# Sanity check: confirm the job sees CUDA + the expected packages
python - <<'PY'
import sys
import numpy as np
import torch
import matplotlib
print('python executable:', sys.executable)
print('numpy version:', np.__version__)
print('torch version:', torch.__version__)
print('matplotlib version:', matplotlib.__version__)
print('cuda available?:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu count:', torch.cuda.device_count())
    print('gpu name:', torch.cuda.get_device_name(0))
PY

# Replace this with your MI runner if the filename differs
python -u run_mi_tre.py
