#!/bin/bash

#SBATCH --job-name=hybridize
#SBATCH --output=hybridize_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

#MODEL_PATH=~/spd_out/spd/s-e790005d/model_50000.pth
#MODEL_PATH=~/Documents/MATS/spd_out/spd/s-429ea112/model_50000.pth
MODEL_PATH=~/spd_out/spd/s-429ea112/model_50000.pth

cd ~/SPD/spd_alt

uv run python -m spd.scripts.decomposition_stress_test.recon_distribution "$MODEL_PATH" \
    --n-batches 512 --batch-size 16

uv run python -m spd.scripts.decomposition_stress_test.per_matrix_recon "$MODEL_PATH" \
    --n-batches 256 --batch-size 16

uv run python -m spd.scripts.decomposition_stress_test.entropy_vs_metrics "$MODEL_PATH" \
    --n-batches 128 --batch-size 16
