#!/bin/bash

#SBATCH --job-name=css_eoa_target
#SBATCH --output=css_eoa_target_%j.log
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

RUN_DIR_CSS=~/spd_out/spd/s-429ea112
MODEL_PATH_CSS=$(ls -t "$RUN_DIR_CSS"/model_*.pth | head -n 1)

cd ~/SPD/spd

uv run python -m spd.scripts.validation.effect_of_ablation \
    "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_components.tsv" \
    --summary-only --n-batches=10 --quantile=0.9 \
    --batch-size=32 --checkpoint-every=1

echo "Job finished at: $(date)"
