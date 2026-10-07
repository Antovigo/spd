#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -J q-layers
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch v2_layers.sbatch <hypothesis set> <tag>. Runs a node-local copy of this folder.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities
W=$(mktemp -d); rsync -a --exclude figures $D/ $W/; mkdir -p $W/figures
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u v2_layers.py "$@" && cp $W/figures/v2_layers_* $D/figures/ && cp $W/layers_*.json $D/
