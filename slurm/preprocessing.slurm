#!/bin/bash
#SBATCH --job-name=chl_preprocess
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/shared/home/dimitar.nakov/MasterT/MastersThesis
#SBATCH --output=/shared/home/dimitar.nakov/MasterT/MastersThesis/logs/chl_preprocess_%j.out
#SBATCH --error=/shared/home/dimitar.nakov/MasterT/MastersThesis/logs/chl_preprocess_%j.err

set -eo pipefail

# Locate Conda, including the installation inherited from the submitting shell.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate chlorophyll-thesis-gpu

# Dask uses the allocated CPUs; prevent nested numerical thread pools.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date -Is)"
echo "Python: $(command -v python)"

# Fail early if a required package or NetCDF backend is missing.
python -c 'import xarray, numpy, pandas, matplotlib, dask.array, scipy, netCDF4'

python -u data/dataPrep_chl_optics_temp.py

echo "Preprocessing finished $(date -Is)"
