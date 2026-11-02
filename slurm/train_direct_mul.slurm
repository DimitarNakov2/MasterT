#!/bin/bash
#SBATCH --job-name=direct_sst
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/direct_sst_%j.out
#SBATCH --error=logs/direct_sst_%j.err

cd /shared/home/dimitar.nakov/MasterT/MastersThesis || exit 1

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate chlorophyll-thesis-gpu

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs
mkdir -p models/direct_sst_bbp443

echo "========================================"
echo "DIRECT + SST TRAINING"
echo "========================================"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

nvidia-smi
echo

python src/train_direct_multivar.py \
    --processed-dir /shared/home/dimitar.nakov/MasterT/MastersThesis/data/processed \
    --model-dir /shared/home/dimitar.nakov/MasterT/MastersThesis/models/direct_sst_bbp443 \
    --aux-mode sst_bbp443 \
    --batch-size 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --epochs 200 \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-8 \
    --warmup-epochs 20 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --patience 30 \
    --seed 42 \
    --log-every 500 \
    --no-amp 

echo
echo "Finished: $(date)"
