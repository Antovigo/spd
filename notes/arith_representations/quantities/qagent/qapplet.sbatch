#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -J q-applet
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch qapplet.sbatch [--positions 1,2,3,4]
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/qagent
W=$(mktemp -d); cp $D/*.py $D/qapplet.html $W/
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u qapplet.py "$@"
