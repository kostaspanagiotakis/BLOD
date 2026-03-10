#!/bin/bash
#SBATCH --job-name=gmm_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gmm%j.out
#SBATCH --error=logs/gmm%j.err

set -euo pipefail
mkdir -p logs

# Activate venv (must contain CUDA-enabled PyTorch)
source /home/kpanagio/sq98/BLOD/venv/bin/activate

# Threading (CPU side)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# Optional allocator tweak
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run from project root
cd /fs04/sq98/BLOD

echo "=== Slurm GPU env ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE-<unset>}"
echo "Hostname: $(hostname)"

echo "=== nvidia-smi (should show 1 GPU) ==="
nvidia-smi

echo "=== torch CUDA sanity check ==="
# Use srun so the job step inherits the allocated GPU reliably
srun --ntasks=1 --gres=gpu:1 python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda available?", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA not available in this environment (CPU-only torch or missing drivers).")
    sys.exit(1)
print("gpu count:", torch.cuda.device_count())
print("gpu name:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("GPU matmul ok. y.mean =", y.mean().item())
PY

echo "=== launching GMM experiment ==="
srun --ntasks=1 --gres=gpu:1 python -u scripts/run_gmm.py