#!/bin/bash
#SBATCH -J rw-attr
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# LAYER=18 sbatch attribute.sbatch — runs attribute.py on extract_layer.py's output
# (node-local copy: NFS can serve a stale .py).
set -euo pipefail
LAYER="${LAYER:-18}"
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/param_decomp "$FRESH/"
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/reader_writer/attribute.py "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TMPDIR=$(mktemp -d)
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u attribute.py --layer "$LAYER" --positions ${POSITIONS:-4 3}
