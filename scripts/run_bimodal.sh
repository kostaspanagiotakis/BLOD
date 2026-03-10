#!/bin/bash
#SBATCH --job-name=blod_1d_gpu
#SBATCH --partition=gpu            # use a GPU node
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # host-side threads for BLAS/data prep
#SBATCH --mem=64G                   # keep as in your original (raise if needed)
#SBATCH --time=00:60:00            # keep 5 minutes (raise if needed)
#SBATCH --output=logs/blod_1d_%j.out
#SBATCH --error=logs/blod_1d_%j.err

set -euo pipefail

# Only logs here; plots/results are created by the Python script under the project root
mkdir -p logs

# Activate venv (ensure it has a CUDA-enabled PyTorch build)
source /home/kpanagio/sq98/BLOD/venv/bin/activate

# Match thread counts to Slurm allocation (speeds up host-side ops)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Make Python unbuffered so tqdm/progress appears live in the .err log
export PYTHONUNBUFFERED=1

# (Optional) CUDA allocator tweak to reduce fragmentation on long runs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run from project root (so outputs land in top-level plots/results)
cd /fs04/sq98/BLOD

# Quick device sanity check (goes to .out)
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
PY

# Launch the experiment (unbuffered)
python -u scripts/run_bimodal.py