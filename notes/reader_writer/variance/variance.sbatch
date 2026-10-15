#!/bin/bash
#SBATCH -J rwv-var
#SBATCH --gres=gpu:l40:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance/logs/%x-%j.out
# sbatch --export=ALL,LAYER=<l>[,ARGS="--pilot --src <dir>"] variance.sbatch — variance.py for one block.
# Resumable: rerunning the same command continues from the saved accumulators.
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W/notes/reader_writer/variance/variance.py" "$FRESH/"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export TMPDIR=$(mktemp -d)
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
"$W/.venv/bin/python" -u variance.py --layer "${LAYER:?}" ${ARGS:-}
