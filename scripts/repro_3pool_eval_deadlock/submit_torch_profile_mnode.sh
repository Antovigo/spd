#!/bin/bash
# torch.profiler scale test for 3-pool. Multi-node (NODES × 8 GPUs).
#
# Now that PhaseProfiler wiring lives in `param_decomp_lab/experiments/lm/
# run.py` (`_maybe_build_torch_profiler`), this submit script is just:
#   - set PD_TORCH_PROFILE_RANKS + PD_TORCH_PROFILE_OUT to opt in
#   - run the slim debug wrapper (SIGUSR1 + faulthandler + heartbeat only;
#     no profile-trace, no distributed-debug env, no monkey-patches)
#
# That way the only confounder vs production is the profiler itself.
#
# Defaults to 2 nodes (16 GPUs).
# Override: NODES=4 CONFIG=... PROFILE_RANKS="0,X,Y" bash submit_torch_profile_mnode.sh
#
# Knobs the wiring honors:
#   PD_TORCH_PROFILE_RANKS         - comma-separated ranks to profile
#   PD_TORCH_PROFILE_OUT           - trace dump directory
#   PD_TORCH_PROFILE_SKIP_FIRST    - schedule.skip_first (default 20; we use SKIP_FIRST below)
#   PD_TORCH_PROFILE_ACTIVE        - schedule.active (default 3; we use ACTIVE below)
#
# Debug-only (does NOT affect the profiler):
#   PD_DEBUG_HEARTBEAT_S, PD_DEBUG_FAULT_TIMEOUT_S

set -euo pipefail

REPO_ROOT="$HOME/param-decomp"
JOB_NAME="${JOB_NAME:-3pool-tp-mnode}"
TIME="${TIME:-00:15:00}"
QOS="${QOS:-opportunistic}"
NODES="${NODES:-2}"
SKIP_FIRST="${SKIP_FIRST:-1}"
ACTIVE="${ACTIVE:-2}"
CONFIG="${CONFIG:-scripts/repro_3pool_eval_deadlock/debug_config_torch_profile_16r.yaml}"
PROFILE_RANKS="${PROFILE_RANKS:-0,8,12}"

cd "$REPO_ROOT"
LOG_DIR="$HOME/.slurm-logs"
mkdir -p "$LOG_DIR"

PORT=$((20000 + RANDOM % 20000))

SCRIPT=$(mktemp /tmp/3pool_tp_mnode_XXXX.sh)
cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=$TIME
#SBATCH --qos=$QOS
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.err
#SBATCH --comment="3-pool torch.profiler scale test"

set -euo pipefail
umask 002
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

# Profiler opt-in (read by _maybe_build_torch_profiler in lm/run.py).
export PD_TORCH_PROFILE_RANKS="$PROFILE_RANKS"
export PD_TORCH_PROFILE_OUT=\$HOME/pd_3pool_debug/\$SLURM_JOB_ID/torch_profile
export PD_TORCH_PROFILE_SKIP_FIRST=$SKIP_FIRST
export PD_TORCH_PROFILE_ACTIVE=$ACTIVE
export PD_TORCH_PROFILE_MEMORY=${PD_TORCH_PROFILE_MEMORY:-1}

# Hang diagnostics only (no perf effect):
export PD_DEBUG_DIR=\$HOME/pd_3pool_debug
export PD_DEBUG_HEARTBEAT_S=10
export PD_DEBUG_FAULT_TIMEOUT_S=300

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
echo "Master: \$MASTER_ADDR:$PORT, nodes=\$SLURM_JOB_NUM_NODES"

srun --nodes=$NODES --ntasks=$NODES --ntasks-per-node=1 \\
    bash -c "cd $REPO_ROOT && source .venv/bin/activate && \\
    torchrun \\
        --nnodes=$NODES \\
        --node_rank=\\\$SLURM_PROCID \\
        --nproc_per_node=8 \\
        --master_addr=\$MASTER_ADDR \\
        --master_port=$PORT \\
        scripts/repro_3pool_eval_deadlock/run_debug.py \\
        $CONFIG \\
        --no_snapshot"

echo "==== JOB COMPLETED OR HUNG, debug dir: \$HOME/pd_3pool_debug/\$SLURM_JOB_ID ===="
EOF

chmod +x "$SCRIPT"
echo "Submitting: $SCRIPT  (NODES=$NODES, total ranks=$((NODES * 8)))"
JOB_ID=$(sbatch --parsable "$SCRIPT")
echo "Job ID: $JOB_ID"
echo "Log: $LOG_DIR/slurm-$JOB_ID.{out,err}"
echo "Debug dir: $HOME/pd_3pool_debug/$JOB_ID"
echo "Trace target: $HOME/pd_3pool_debug/$JOB_ID/torch_profile/"
echo
echo "Tail the log:  tail -F $LOG_DIR/slurm-$JOB_ID.out"
