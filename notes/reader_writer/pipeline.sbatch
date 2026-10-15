#!/bin/bash
#SBATCH -J rw-pipeline
#SBATCH --gres=gpu:l40:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 24:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# One job, everything serial on one card:
#   EXTRACT=1     extract_all_layers.py first (skips blocks whose L<l>/meta.json exists)
#   ATTR="3 14"   attribute.py on these layers
#   BUILD="3 14"  build_app.py on these layers (default: = ATTR)
# e.g. ATTR="$(seq -s ' ' 0 31)" EXTRACT=1 sbatch pipeline.sbatch
set -euo pipefail
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer
PY=/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/param_decomp "$FRESH/"
cp "$D"/{extract_layer.py,extract_all_layers.py,attribute.py,build_app.py,app.html} "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=$(mktemp -d)
# the allocated card can hold another user's orphan process: wait for enough free memory
NEED=$([ "${EXTRACT:-0}" = 1 ] && echo 44000 || echo 20000)
for ATTEMPT in $(seq 1 30); do
  FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "attempt $ATTEMPT: card $CUDA_VISIBLE_DEVICES free MiB $FREE (need $NEED)"
  [ "$FREE" -ge "$NEED" ] && break
  [ "$ATTEMPT" -eq 30 ] && { echo "no free card"; exit 1; }
  sleep 60
done
cd "$FRESH"
[ "${EXTRACT:-0}" = 1 ] && $PY -u extract_all_layers.py
for L in ${ATTR:-}; do
  echo "=== attribute L$L $(date -u +%T)"
  $PY -u attribute.py --layer "$L" --positions ${POSITIONS:-4 3} 2>&1 | grep -v "pass 1b" || echo "L$L attribute FAILED"
done
for L in ${BUILD:-${ATTR:-}}; do
  echo "=== build_app L$L $(date -u +%T)"
  $PY -u build_app.py --layer "$L" --positions ${POSITIONS:-4 3} || echo "L$L build_app FAILED"
done
echo "=== pipeline done $(date -u +%T)"
