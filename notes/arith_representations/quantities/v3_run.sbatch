#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH -J q-v3
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities
W=$(mktemp -d); rsync -a --exclude figures $D/ $W/
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u v3_run.py && cp $W/v3_token_a.json $W/v3_recon.npz $D/
