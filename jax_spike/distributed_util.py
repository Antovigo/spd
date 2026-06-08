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
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def init_distributed() -> bool:
    if "SLURM_PROCID" not in os.environ:
        return False
    local_id = int(os.environ["SLURM_LOCALID"])
    jax.distributed.initialize(local_device_ids=[local_id])
    return True


def dp_mesh() -> Mesh:
    """1-D data-parallel mesh over all (global) devices."""
    return Mesh(np.array(jax.devices()), axis_names=("dp",))


def shard_dp(full_global: jax.Array, mesh: Mesh) -> jax.Array:
    """Build a P('dp')-batch-sharded global array from a full array that every process
    generated identically (same seed). Assumes 1 device per process; device order is
    sorted by process index, so process p owns global batch slice p.
    """
    sharding = NamedSharding(mesh, P("dp"))
    n = mesh.devices.size
    assert full_global.shape[0] % n == 0, f"batch {full_global.shape[0]} not divisible by {n}"
    per = full_global.shape[0] // n
    idx = jax.process_index()
    local = full_global[idx * per : (idx + 1) * per]
    return jax.make_array_from_single_device_arrays(
        full_global.shape,
        sharding,
        [jax.device_put(local, d) for d in sharding.addressable_devices],
    )


def replicate(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Place a fully-replicated array (same on every device)."""
    return jax.device_put(x, NamedSharding(mesh, P()))
