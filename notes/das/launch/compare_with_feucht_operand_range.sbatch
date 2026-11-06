#!/bin/bash
#SBATCH -J das-feucht-comparison
#SBATCH -p compute
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/pd_scratch/das_job_logs/%x-%j.out
set -euo pipefail
REPO=/mnt/nw/home/a.vigouroux/Code/param-decomp/das
PY=/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ "$REPO/param_decomp" "$REPO/notes/das/compare_with_feucht_operand_range.py" "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR=/ephemeral/$USER/das-$SLURM_JOB_ID; mkdir -p "$TMPDIR"
for ATTEMPT in $(seq 1 15); do
  FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "attempt $ATTEMPT: free MiB $FREE"
  [ "$FREE" -ge 30000 ] && break
  [ "$ATTEMPT" -eq 15 ] && { echo "GPU stays occupied"; exit 75; }
  sleep 60
done
cd "$FRESH"
"$PY" -u compare_with_feucht_operand_range.py --config "$REPO/notes/das/das_experiment_config.yaml" \
  --out /mnt/nw/home/a.vigouroux/pd_scratch/das_pilot/feucht_comparison
