#!/bin/bash
#SBATCH -J rw-extract
#SBATCH --gres=gpu:l40:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# LAYER=18 sbatch extract_layer.sbatch — runs extract_layer.py against the dual_obj_jax package
# (node-local copy: NFS can serve a stale .py).
set -euo pipefail
LAYER="${LAYER:-18}"
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/param_decomp "$FRESH/"
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/reader_writer/extract_layer.py "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TMPDIR=$(mktemp -d)
# --gres reserves cards but does not make them idle: wait until both have >= 44000 MiB free
for ATTEMPT in $(seq 1 30); do
  FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "attempt $ATTEMPT: free MiB $(echo $FREE)"
  [ "$(echo "$FREE" | awk '$1 < 44000' | wc -l)" -eq 0 ] && break
  sleep 60
done
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u extract_layer.py --layer "$LAYER" ${LIMIT:+--limit $LIMIT}
