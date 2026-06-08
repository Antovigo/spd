"""Reusable SLURM <-> jax.distributed bring-up for the spike.

Canonical 1-process-per-GPU recipe: the job makes all 8 node GPUs visible to every task
(--gres=gpu:8), and each process claims exactly one via local_device_ids=[SLURM_LOCALID].
Coordinator address / num_processes / process_id are auto-detected from the SLURM env.

(The alternative -- per-task GPU binding so each process sees a single ordinal-0 GPU --
got through init but tripped 'invalid device ordinal' inside NCCL on this cluster; the
all-visible + LOCALID recipe is the robust one.)

Call once at process start. No-op (returns False) when not under SLURM.
"""

import os

import jax


def init_distributed() -> bool:
    if "SLURM_PROCID" not in os.environ:
        return False
    local_id = int(os.environ["SLURM_LOCALID"])
    jax.distributed.initialize(local_device_ids=[local_id])
    return True
