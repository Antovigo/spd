#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -J q-report
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Runs a node-local copy of this folder (NFS can serve stale .py), then copies the outputs back.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities
W=$(mktemp -d); rsync -a --exclude figures $D/ $W/; mkdir -p $W/figures
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u report_figures.py && cp $W/figures/walk_* $D/figures/ && cp $W/walkthrough.json $D/
