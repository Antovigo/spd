#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 2:00:00
#SBATCH -J vw
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/virtual_weights/logs/%x-%j.out
# Usage: sbatch compute.sbatch build|check
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 PYTHONPYCACHEPREFIX=$(mktemp -d)
source /mnt/nw/home/a.vigouroux/pd_scratch/dual_obj_jax/arith_repr/isa/fresh.sh
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/virtual_weights/compute.py $FRESH/
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u $FRESH/compute.py "$@"
