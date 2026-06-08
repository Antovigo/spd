"""Reusable SLURM <-> jax.distributed bring-up for the spike.

Under `srun --gpus-per-task=1`, each task sees exactly ONE GPU (CUDA_VISIBLE_DEVICES=0,
i.e. local ordinal 0). JAX's SLURM auto-detect otherwise sets local_device_ids to the
task's local rank (0..7) and tries to grab that ordinal -- which isn't visible -- so it
must be pinned to [0]. Coordinator address / num_processes / process_id are still
auto-detected from the SLURM env.

Call once at process start. No-op (returns False) when not under SLURM.
"""

import os

import jax


def init_distributed() -> bool:
    if "SLURM_PROCID" not in os.environ:
        return False
    jax.distributed.initialize(local_device_ids=[0])
    return True
