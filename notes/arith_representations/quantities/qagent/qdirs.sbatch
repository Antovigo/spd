#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 4:00:00
#SBATCH -J q-dirs
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch qdirs.sbatch <t>
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/qagent
W=$(mktemp -d); cp $D/*.py $W/
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u qdirs.py "$@"
