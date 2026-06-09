"""GSPMD sharding helpers — the JAX analog of FSDP2.

The single-pool SPMD design (the recommended JAX target, see
`jax_spike/SYNTHESIS.md`): data is sharded `P('dp')` over a 1-D device mesh,
params + PGD sources are placed with an explicit sharding, and `jax.jit` inserts
every collective (the grad all-reduce, the source-grad reduction) automatically
because the mean-losses reduce over the sharded batch axis. No manual NCCL, no
pool-coordination code.

Two placement strategies, both expressed as `NamedSharding`:
  * replicate    — params on every device (the data-parallel memory floor).
  * shard_leading — shard a param's leading axis over 'dp' (the FSDP analog for
    the memory story: stacked sites / components split across devices). The
    step's einsums stay correct because XLA all-gathers on demand.

This generalizes `jax_spike/distributed_util.py` into an importable module the
trainer composes, rather than a per-stage copy.
"""

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def init_distributed() -> bool:
    """Bring up `jax.distributed` under SLURM. No-op (False) off SLURM.

    The cluster recipe (from the spike): all GPUs visible per task
    (`--gres=gpu:8`), each process claims `local_device_ids=[SLURM_LOCALID]`.
    """
    if "SLURM_PROCID" not in os.environ:
        return False
    local_id = int(os.environ["SLURM_LOCALID"])
    jax.distributed.initialize(local_device_ids=[local_id])
    return True


def dp_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("dp",))


def replicate(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P()))


def shard_leading(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Shard a param's leading axis over 'dp' (the FSDP-style param shard).

    The leading axis must be divisible by the mesh size. For stacked sites this
    splits the site bank across devices; for components, split the C axis instead
    by transposing first.
    """
    n = mesh.devices.size
    assert x.shape[0] % n == 0, f"leading dim {x.shape[0]} not divisible by mesh size {n}"
    return jax.device_put(x, NamedSharding(mesh, P("dp")))


def shard_batch(full_global: jax.Array, mesh: Mesh, batch_axis: int) -> jax.Array:
    """Shard `full_global` over 'dp' along `batch_axis`. Generated identically on
    every process (same seed), so each process slices out its process-local
    sub-batch and `make_array_from_process_local_data` does the device placement.

    Works for both topologies the spike uses: single-process / many-devices (CPU
    sim, or 1 process with N local GPUs — the process owns the whole batch and it
    splits across the local devices) and multi-process / 1-device-each (SLURM —
    each process owns its 1/n_processes slice). `batch_axis` is axis 1 for the
    stacked-site `[S, B, ..., d]` layout.
    """
    n_proc = jax.process_count()
    B = full_global.shape[batch_axis]
    assert B % mesh.devices.size == 0, (
        f"batch {B} (axis {batch_axis}) not divisible by mesh size {mesh.devices.size}"
    )
    spec: list[str | None] = [None] * full_global.ndim
    spec[batch_axis] = "dp"
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
