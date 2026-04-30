#!/bin/bash

#SBATCH --job-name=jose_completeness
#SBATCH --output=jose_completeness_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "========================================================"
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo

RUN_DIR_JOSE=~/spd_out/spd/jose
MODEL_PATH_JOSE=$(ls -t "$RUN_DIR_JOSE"/model_*.pth | head -n 1)

cd ~/SPD/spd

uv run python -m spd.scripts.validation.completeness \
    "$MODEL_PATH_JOSE" "$RUN_DIR_JOSE/alive_components.tsv" \
    --split=train --n-batches=10 --batch-size=8 --ci-threshold=0.1

echo "Job finished at: $(date)"
