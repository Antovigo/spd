#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH -t 4:00:00
#SBATCH -J q-patch
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# Usage: sbatch v3_patch.sbatch [n_prompts]. Node-local copies of the package and of this folder.
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/nw/home/a.vigouroux/pd_scratch/jax_cache_arith PYTHONDONTWRITEBYTECODE=1
source /mnt/nw/home/a.vigouroux/pd_scratch/dual_obj_jax/arith_repr/isa/fresh.sh
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities
W=$(mktemp -d); rsync -a --exclude figures $D/ $W/
cd $W && /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u v3_patch.py "$@" && cp $W/v3_patch.json $D/
