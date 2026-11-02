#!/bin/bash
#SBATCH --job-name=chl_download
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/shared/home/dimitar.nakov/MasterT/MastersThesis/logs/chl_download_%j.out
#SBATCH --error=/shared/home/dimitar.nakov/MasterT/MastersThesis/logs/chl_download_%j.err

set -euo pipefail

PROJECT_DIR="/shared/home/dimitar.nakov/MasterT/MastersThesis"

cd "$PROJECT_DIR/data"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $(pwd)"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate your environment
conda activate chlorophyll-thesis-gpu

# Optional sanity checks
which python
python --version

# Run download
python download.py

echo "Finished: $(date)"
