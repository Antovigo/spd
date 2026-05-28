#!/bin/bash
# Submit the 3-pool slow-eval deadlock repro to SLURM.
# Single-node, 8 GPUs, opportunistic QoS (preemptible, no fairshare cost).
#
# Output goes to:
#   $HOME/.slurm-logs/slurm-<job>.out  (slurm stdout)
#   $HOME/pd_3pool_debug/<job>/        (per-rank stacks, memory, heartbeat)
#
# To dump live stacks while a job is running:
#   srun --jobid=<job> --overlap bash -c 'pkill -USR1 -f "python.*run_debug.py"'

set -euo pipefail

REPO_ROOT="$HOME/param-decomp"
JOB_NAME="${JOB_NAME:-3pool-deadlock-repro}"
TIME="${TIME:-00:30:00}"
QOS="${QOS:-opportunistic}"

cd "$REPO_ROOT"
LOG_DIR="$HOME/.slurm-logs"
mkdir -p "$LOG_DIR"

# Single-node 8 GPU job.
SCRIPT=$(mktemp /tmp/3pool_repro_XXXX.sh)
cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=$TIME
#SBATCH --qos=$QOS
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err
#SBATCH --comment="3-pool slow-eval smoke (regression test for the bf16/numpy fix)"

set -euo pipefail
umask 002
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
export PD_DEBUG_DIR=$HOME/pd_3pool_debug
export PD_DEBUG_HEARTBEAT_S=10
export PD_DEBUG_FAULT_TIMEOUT_S=600

cd $REPO_ROOT
source .venv/bin/activate

# Pick a deterministic master port so concurrent debug runs don't collide.
PORT=\$((20000 + RANDOM % 20000))

torchrun \\
    --standalone \\
    --nproc_per_node=8 \\
    --master_port=\$PORT \\
    scripts/repro_3pool_eval_deadlock/run_debug.py \\
    scripts/repro_3pool_eval_deadlock/debug_config.yaml \\
    --no_snapshot

echo "==== JOB COMPLETED OR HUNG, debug dir: /tmp/pd_3pool_debug/\$SLURM_JOB_ID ===="
EOF

chmod +x "$SCRIPT"
echo "Submitting: $SCRIPT"
JOB_ID=$(sbatch --parsable "$SCRIPT")
echo "Job ID: $JOB_ID"
echo "Log: $LOG_DIR/slurm-$JOB_ID.out"
echo "Debug dir: $HOME/pd_3pool_debug/$JOB_ID"
echo
echo "Tail the log:"
echo "  tail -F $LOG_DIR/slurm-$JOB_ID.out"
echo
echo "Dump live stacks (after some progress):"
echo "  srun --jobid=$JOB_ID --overlap bash -c 'pkill -USR1 -f \"python.*run_debug.py\"'"
