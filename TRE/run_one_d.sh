#!/bin/bash
#SBATCH --job-name=toy_1d_tre
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/run_one_d_%j.out
#SBATCH --error=logs/run_one_d_%j.err

set -euo pipefail

mkdir -p logs

cd /fs04/sq98/BLOD/TRE
source /home/kpanagio/sq98/BLOD/venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Install the exact Python requirements for this 1D toy experiment
python -m pip install --upgrade pip
python -m pip install -r requirements_tre_1d.txt

# Sanity check: ensure the batch job is using the venv Python and can import NumPy
python - <<'PY'
import sys, numpy as np
print('python executable:', sys.executable)
print('numpy version:', np.__version__)
PY

python -u run_one_d.py
