#!/usr/bin/env bash
# Run ONE targeted-decomposition process to completion, with the two protections the
# cluster sbatch scripts carried (no SLURM on the pod, so they live here):
#   * SIGTERM forwarding: a TERM to this script is forwarded to the TRAINER, which saves a
#     checkpoint and exits; we then wait for it to be really gone (bash's `wait` returns
#     >128 when a trap interrupts it, so re-wait). Without this the trainer dies mid-save.
#   * hang watchdog: a dp>1 OOM does NOT raise — one rank dies mid-allocation, the others
#     wait on the NCCL clique, and the process looks alive forever. If the log named by
#     $PD_LOG has not grown for $WATCHDOG_FUSE seconds, kill -9 the trainer (exit 124).
#
#     usage:  PD_LOG=<logfile> pd_run.sh <config.yaml> <run_id>       (env from env.sh)
#     env:    WATCHDOG_FUSE (s, default 3600; a cold 32-block compile is silent for a long
#             time), PD_LOG (the file this script's stdout is redirected to; watchdog off if
#             unset), REPO/DATA_ROOT/VENV_PY (env.sh).
# Exit code is the trainer's (0 = trained to pd.steps, or SIGTERM-saved cleanly), 124 on a
# watchdog kill. Re-running with the same run id RESUMES from the newest checkpoint.
set -euo pipefail
CONFIG="$(readlink -f "${1:?config.yaml}")"   # absolute: we cd into $REPO below
RUN_ID="${2:?run id (p-<8hex>)}"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
WATCHDOG_FUSE="${WATCHDOG_FUSE:-3600}"
[ -f "$CONFIG" ] || { echo "no such config: $1"; exit 2; }

cd "$REPO"
echo "=== pd_run $(date -Is) config=$CONFIG run_id=$RUN_ID commit=$(git rev-parse --short HEAD 2>/dev/null || echo tarball) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader

# Refuse a contended card rather than wedging on it (report §4): a previous trial that hung
# and was killed can hold memory for a while, and a pod is not a SLURM allocation.
APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)
if [ -n "$APPS" ]; then
  echo "=== ABORT: GPUs are not idle (other compute processes): ==="; echo "$APPS"; exit 75
fi

# local_device_count = this process's GPUs = replicate * fsdp * tp of the config (single host).
N_DEV=$("$VENV_PY" - "$CONFIG" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))["runtime"]
print(r["replicate"] * r["fsdp"] * r["tp"])
PY
)
N_GPU=$(nvidia-smi -L | wc -l)
[ "$N_DEV" -eq "$N_GPU" ] || { echo "=== ABORT: config mesh wants $N_DEV GPUs, pod has $N_GPU ==="; exit 2; }

RUN_NAME="$(awk -F': *' '/^run_name:/{print $2; exit}' "$CONFIG")"
mkdir -p "$DATA_ROOT/runs/by-name" "$DATA_ROOT/pids"
ln -sfn "../$RUN_ID" "$DATA_ROOT/runs/by-name/$RUN_NAME"
echo "run $RUN_NAME -> $DATA_ROOT/runs/$RUN_ID  (local_device_count=$N_DEV)"

"$VENV_PY" -m param_decomp.experiments.lm.run_targeted \
  "$CONFIG" "$DATA_ROOT" "$N_DEV" --run_id "$RUN_ID" &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$DATA_ROOT/pids/$RUN_ID.trainer.pid"
echo "trainer pid $TRAIN_PID"

WATCHDOG_PID=""
FIRED="$DATA_ROOT/pids/$RUN_ID.watchdog-fired"
rm -f "$FIRED"
if [ -n "${PD_LOG:-}" ]; then
  ( while kill -0 "$TRAIN_PID" 2>/dev/null; do
      sleep 60
      if [ -f "$PD_LOG" ]; then
        AGE=$(( $(date +%s) - $(stat -c %Y "$PD_LOG") ))
        if [ "$AGE" -gt "$WATCHDOG_FUSE" ]; then
          echo "=== WATCHDOG: no log growth in ${AGE}s — kill -9 (suspect OOM hang) ==="
          touch "$FIRED"
          kill -9 "$TRAIN_PID" 2>/dev/null || true
          exit 0
        fi
      fi
    done ) &
  WATCHDOG_PID=$!
else
  echo "(PD_LOG unset: hang watchdog disabled)"
fi

INTERRUPTED=0
forward_term() {
  INTERRUPTED=1
  echo "=== SIGTERM $(date -Is): forwarding to trainer $TRAIN_PID for a checkpoint save ==="
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
}
trap forward_term TERM INT

set +e
wait "$TRAIN_PID"; rc=$?
# A trapped signal makes `wait` return 128+sig at once WITHOUT reaping the child; re-wait
# until a wait completes uninterrupted, so rc is the trainer's real exit status (0 after a
# SIGTERM save) and bash cannot exit mid-save. A child that itself died of a signal never
# sets INTERRUPTED, so its 128+sig comes through unchanged.
while [ "$INTERRUPTED" -eq 1 ]; do
  INTERRUPTED=0
  wait "$TRAIN_PID"; rc=$?
done
set -e
[ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null || true
if [ -f "$FIRED" ]; then rc=124; rm -f "$FIRED"; fi
rm -f "$DATA_ROOT/pids/$RUN_ID.trainer.pid"
echo "=== RUN EXITED rc=$rc $(date -Is) ==="
exit "$rc"
