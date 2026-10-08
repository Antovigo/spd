#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH -J q-loop
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%A_%a.out
# Usage: sbatch qloop.sbatch <t> [args]          (full run, or --combine)
#        sbatch --array=0-9 qloop.sbatch <t>     (nested selection, one fold per task)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/qagent
W=$(mktemp -d); cp $D/*.py $W/
FOLD=${SLURM_ARRAY_TASK_ID:+--fold $SLURM_ARRAY_TASK_ID}
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u qloop.py "$@" $FOLD
