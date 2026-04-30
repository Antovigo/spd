#!/bin/bash

#SBATCH --job-name=np_pd_ref
#SBATCH --output=np_pd_naive_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

cd ~/SPD/spd

uv run spd/experiments/lm/lm_decomposition.py \
  /mnt/nw/home/a.vigouroux/SPD/batch_commands/numpy/reference_4L/config_numpy_reference.yaml

echo "Job finished at: $(date)"
