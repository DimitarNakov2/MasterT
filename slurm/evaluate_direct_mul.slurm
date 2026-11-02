#!/bin/bash
#SBATCH --job-name=direct_sst_bbp_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/direct_sst_bbp_eval_%j.out
#SBATCH --error=logs/direct_sst_bbp_eval_%j.err

cd /d/hpc/home/dn9954/MasterT/MastersThesis || exit 1

source /cvmfs/sling.si/modules/el7/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate chlorophyll-thesis-gpu

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs
mkdir -p models/direct_sst_bbp443/evaluation

echo "========================================"
echo "DIRECT + SST + BBP443 EVALUATION"
echo "========================================"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

nvidia-smi
echo

python src/evaluate_direct_mul.py \
    --processed-dir /d/hpc/home/dn9954/MasterT/MastersThesis/data/processed \
    --checkpoint /d/hpc/home/dn9954/MasterT/MastersThesis/models/direct_sst_bbp443/best.pt \
    --output-dir /d/hpc/home/dn9954/MasterT/MastersThesis/models/direct_sst_bbp443/evaluation \
    --steps 50 \
    --ensemble-size 16 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --log-every 10

echo
echo "Finished: $(date)"
