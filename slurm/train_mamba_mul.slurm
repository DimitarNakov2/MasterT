#!/bin/bash
#SBATCH --job-name=mamba_sst
#SBATCH --partition=gpu
#SBATCH --nodelist=gwn08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/mamba_sst_%j.out
#SBATCH --error=logs/mamba_sst_%j.err

cd /d/hpc/home/dn9954/MasterT/MastersThesis || exit 1

source /cvmfs/sling.si/modules/el7/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate chlorophyll-thesis-gpu

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs
mkdir -p models/mamba_sst_bbp443

echo "========================================"
echo "MAMBA + SST TRAINING"
echo "========================================"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

nvidia-smi
echo

python src/train_mamba_multivar.py \
    --processed-dir /d/hpc/home/dn9954/MasterT/MastersThesis/data/processed \
    --model-dir /d/hpc/home/dn9954/MasterT/MastersThesis/models/mamba_sst_bbp443 \
    --sequence-length 25 \
    --aux-mode sst_bbp443 \
    --batch-size 1 \
    --num-workers 4 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-6 \
    --warmup-epochs 5 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --patience 20 \
    --seed 42 \
    --log-every 500 \
    --resume /d/hpc/home/dn9954/MasterT/MastersThesis/models/mamba_sst_bbp443/last.pt

echo
echo "Finished: $(date)"
