#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -J vw-cimax
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/virtual_weights/logs/%x-%j.out
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 PYTHONPYCACHEPREFIX=$(mktemp -d)
source /mnt/nw/home/a.vigouroux/pd_scratch/dual_obj_jax/arith_repr/isa/fresh.sh
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/virtual_weights
cp $D/ci_max_check.py $FRESH/
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u $FRESH/ci_max_check.py
