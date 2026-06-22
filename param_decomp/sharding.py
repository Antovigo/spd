"""GSPMD sharding helpers — the JAX analog of FSDP2.

The single-pool SPMD design (the recommended JAX target, see
`jax_spike/SYNTHESIS.md`): data is sharded `P('dp')` over a 1-D device mesh,
params + PGD sources are placed with an explicit sharding, and `jax.jit` inserts
every collective (the grad all-reduce, the source-grad reduction) automatically
because the mean-losses reduce over the sharded batch axis. No manual NCCL, no
pool-coordination code.

Placement is expressed as `NamedSharding`: target-specific plans (like
`llama8b_sharding.py`) place params with layer C-sharding; `shard_batch` shards the
data axis.
"""

import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def init_distributed(dp: int | None) -> bool:
    """Bring up `jax.distributed` iff `dp` is set. Distributedness is config-driven
    (`runtime.dp`), NEVER inferred from ambient SLURM env — `SLURM_PROCID` is present in
    every process on a SLURM box (incl. a pytest worker), so sniffing it would wrongly
    fire `jax.distributed.initialize` mid-test.

    `dp is None` → single device, no-op (return False). Otherwise the cluster recipe (from
    the spike): all GPUs visible per task (`--gres=gpu:8`), each process claims
    `local_device_ids=[SLURM_LOCALID]`, and the realized world size must equal `dp`. SLURM
    env is read ONLY for the rank info, once `dp` has decided we're distributed.
    """
    if dp is None:
        return False
    local_id = int(os.environ["SLURM_LOCALID"])
    n_visible = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    assert local_id < n_visible, (
        f"SLURM_LOCALID={local_id} >= {n_visible} visible GPUs — srun packed tasks onto "
        f"too few nodes (job 50416 failure mode); launch steps with an explicit "
        f"--ntasks-per-node=<gpus-per-node>"
    )
    jax.distributed.initialize(local_device_ids=[local_id])
    assert jax.process_count() == dp, (
        f"runtime.dp={dp} != realized world size {jax.process_count()} — the config's "
        f"declared world size must match the launch topology (nodes × 8)"
    )
    return True


def dp_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("dp",))


def batch_shard_leading(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    """In-jit `with_sharding_constraint` pinning the LEADING (batch) axis to `'dp'`, the
    rest replicated. `mesh is None` (single device) is a passthrough. Keeps the masked
    re-forwards on per-device sub-batches (activation memory 1/n_dev)."""
    if mesh is None:
        return x
    spec = ["dp"] + [None] * (x.ndim - 1)
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))


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
