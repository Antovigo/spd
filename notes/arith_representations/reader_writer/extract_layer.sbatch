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
# --gres reserves cards but does not make them idle (2026-10-14: an orphan vLLM worker of another
# user held 41 GB on one allocated card). Use only the allocated cards with >= 44000 MiB free.
FREE_IDS=""
for ATTEMPT in $(seq 1 10); do
  FREE_IDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$2 >= 44000 {print $1}' | paste -sd, -)
  echo "attempt $ATTEMPT: allocated $CUDA_VISIBLE_DEVICES, free cards: ${FREE_IDS:-none}"
  [ -n "$FREE_IDS" ] && break
  sleep 60
done
[ -n "$FREE_IDS" ] || { echo "no free card"; exit 1; }
export CUDA_VISIBLE_DEVICES=$FREE_IDS
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u extract_layer.py --layer "$LAYER" ${LIMIT:+--limit $LIMIT}
