#!/bin/bash
#SBATCH --job-name=download_aux
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/download_aux_%j.out
#SBATCH --error=logs/download_aux_%j.err

cd /shared/home/dimitar.nakov/MasterT/MastersThesis || exit 1

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate chlorophyll-thesis-gpu

mkdir -p logs

echo "Starting temperature download..."
python data/download_temp.py > logs/temp_download_${SLURM_JOB_ID}.log 2>&1 &
TEMP_PID=$!

echo "Starting river download..."
python data/download_optics.py > logs/river_download_${SLURM_JOB_ID}.log 2>&1 &
RIVER_PID=$!

echo "Temperature PID: $TEMP_PID"
echo "River PID: $RIVER_PID"

wait $TEMP_PID
TEMP_EXIT=$?

wait $RIVER_PID
RIVER_EXIT=$?

echo "Temperature download exit code: $TEMP_EXIT"
echo "River download exit code: $RIVER_EXIT"

if [ $TEMP_EXIT -ne 0 ] || [ $RIVER_EXIT -ne 0 ]; then
    echo "One or more downloads failed."
    exit 1
fi

echo "Both downloads completed successfully."
