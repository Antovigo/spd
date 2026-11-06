#!/bin/bash
#SBATCH -J das-pilot
#SBATCH -p compute
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --array=0-7%4
#SBATCH -o /mnt/nw/home/a.vigouroux/pd_scratch/das_job_logs/%x-%A_%a.out
# One DAS pilot site per array task. STAGE is fit_quality, search or drop_then_bisect.
set -euo pipefail
STAGE="${STAGE:?set STAGE}"
REPO=/mnt/nw/home/a.vigouroux/Code/param-decomp/das
PY=/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ "$REPO/param_decomp" "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR=/ephemeral/$USER/das-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p "$TMPDIR"
for ATTEMPT in $(seq 1 15); do
  FREE=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "attempt $ATTEMPT: free MiB $FREE"
  [ "$FREE" -ge 30000 ] && break
  [ "$ATTEMPT" -eq 15 ] && { echo "GPU stays occupied"; exit 75; }
  sleep 60
done
cd "$FRESH"
"$PY" -u -m param_decomp.das.run_das_pilot --config "$REPO/notes/das/das_experiment_config.yaml" \
  --stage "$STAGE" --site "$SLURM_ARRAY_TASK_ID"
