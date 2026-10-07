#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=10G
#SBATCH -t 4:00:00
#SBATCH -J q-prep
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch qprep.sbatch <t>
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/qagent
W=$(mktemp -d); cp $D/qprep.py $W/
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u qprep.py "$@"
