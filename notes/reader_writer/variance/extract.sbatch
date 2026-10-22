#!/bin/bash
#SBATCH -J rwv-extract
#SBATCH --gres=gpu:l40:2
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance/logs/%x-%j.out
# sbatch [--export=ALL,LIMIT=6000,OUT=<dir>,ATTN=1] extract.sbatch — alive-only activations (extract_alive_only.py)
# against a node-local copy of the arith_representations package (NFS can serve a stale .py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ "$W/param_decomp" "$FRESH/"
cp "$W/notes/reader_writer/variance/extract_alive_only.py" "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TMPDIR=$(mktemp -d)
# --gres reserves cards but does not make them idle: use only allocated cards with >= 44000 MiB free.
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
"$W/.venv/bin/python" -u extract_alive_only.py ${LIMIT:+--limit $LIMIT} ${OUT:+--out $OUT} ${PROFILE:+--profile} ${ATTN:+--attn}
