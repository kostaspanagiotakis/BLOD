#!/bin/bash
#SBATCH --job-name=testpy
#SBATCH --output=logs/testpy_%j.out
#SBATCH --error=logs/testpy_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# (No --partition means default: comp on M3)

# Good practice: fail fast on errors
set -euo pipefail

# Create a logs dir for outputs
mkdir -p logs

# --- Activate your venv (absolute path) ---
# If your venv was created with python -m venv:
source /home/kpanagio/sq98/BLOD/venv/bin/activate

# (Optional) print which python/pip to confirm we're in the venv
which python
python --version
pip --version

# --- Run your script with the venv's Python ---
python test.py