#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH -t 04:00:00
#SBATCH -J q-patch
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch qpatch.sbatch <t> [args]
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/nw/home/a.vigouroux/pd_scratch/jax_cache_arith PYTHONPYCACHEPREFIX=$(mktemp -d)
source /mnt/nw/home/a.vigouroux/pd_scratch/dual_obj_jax/arith_repr/isa/fresh.sh
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/qagent/*.py $FRESH/
cd $FRESH && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u qpatch.py "$@"
