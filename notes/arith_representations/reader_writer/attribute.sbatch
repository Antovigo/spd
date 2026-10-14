#!/bin/bash
#SBATCH -J rw-attr
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%A_%a.out
# LAYER=18 sbatch attribute.sbatch, or sbatch --array=0-31 attribute.sbatch (layer = task id):
# runs attribute.py on extract_layer.py / extract_all_layers.py output
# (node-local copy: NFS can serve a stale .py).
set -euo pipefail
LAYER="${LAYER:-${SLURM_ARRAY_TASK_ID:-18}}"
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/param_decomp "$FRESH/"
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/reader_writer/attribute.py "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=$(mktemp -d)
# the allocated card can hold another user's orphan process: wait for >= 20000 MiB free
for ATTEMPT in $(seq 1 30); do
  FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "attempt $ATTEMPT: layer $LAYER, free MiB $FREE"
  [ "$FREE" -ge 20000 ] && break
  [ "$ATTEMPT" -eq 30 ] && { echo "no free card"; exit 1; }
  sleep 60
done
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u attribute.py --layer "$LAYER" --positions ${POSITIONS:-4 3}
