"""GSPMD sharding helpers for the 3-D `(replicate, fsdp, tp)` device mesh.

`replicate` spans nodes. Each node is an `(fsdp, tp)` plane: `fsdp` shards the parameter
and optimizer-state dimensions used by HSDP, while `tp` tensor-parallelizes target
matrices that declare it. `tp=1` is the ordinary HSDP layout. Batches shard over
`(replicate, fsdp)` and are replicated across `tp`.
"""

import os
from typing import cast

import jax
import numpy as np
from jax.sharding import AbstractMesh, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.model import DecomposedModel


def init_distributed(dp: int, gpus_per_node: int) -> None:
    """The multi-node process bring-up: `jax.distributed` over `dp // gpus_per_node`
    nodes. Distributedness is config-DERIVED (`dp > gpus_per_node`), NEVER inferred from ambient
    SLURM env — `SLURM_PROCID` is present in every process on a SLURM box (incl. a pytest
    worker), so sniffing it would wrongly fire `jax.distributed.initialize` mid-test.

    The recipe: ONE process per node, each owning all its local GPUs (mirrors the torch
    torchrun model — under SLURM, srun `--ntasks-per-node=1`). jax auto-detects the SLURM
    topology (process_id = node rank, num_processes = node count) but its SLURM cluster
    env claims only ONE device per process by default, so we pass the full local device
    list explicitly (`CUDA_VISIBLE_DEVICES`, set to all 8 by `--gpus-per-node=8`). The
    realized total device count must equal `dp`. This also rules out the 8-tasks-per-node
    placement that a packing node-selection policy (`CR_Pack_Nodes`) would land on one
    node. `dp` (config) decides distributedness and world size; SLURM env only supplies
    the rank.
    """
    assert dp % gpus_per_node == 0, f"dp={dp} must be a multiple of {gpus_per_node} (GPUs/node)"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_local = len([d for d in cuda_visible.split(",") if d]) or gpus_per_node
    jax.distributed.initialize(local_device_ids=list(range(n_local)))
    assert jax.device_count() == dp, (
        f"runtime.dp={dp} != realized device count {jax.device_count()} "
        f"({jax.process_count()} procs × {jax.local_device_count()} local GPUs; "
        f"CUDA_VISIBLE_DEVICES={cuda_visible!r}) — the config's declared world size must "
        f"match the launch topology (nodes × {gpus_per_node})"
    )


def assert_inline_topology(dp: int) -> None:
    """The single-process startup gate (`dp <= gpus_per_node`): one process (no
    `jax.distributed`) whose local
    devices ARE the whole declared world. Lives at the config-consuming entry, NOT in
    `hsdp_mesh` — the CPU-sim tests deliberately call `hsdp_mesh` bare under forced host
    device counts."""
    assert jax.process_count() == 1, (
        f"a sub-node world (dp <= gpus_per_node) is single-process, found {jax.process_count()} processes"
    )
    assert jax.device_count() == dp, (
        f"runtime.dp={dp} != local device count {jax.device_count()} — a sub-node world "
        f"runs one process over exactly the devices the config declares; an ambient "
        f"mismatch is a mis-sized allocation, never absorbed"
    )


BATCH_AXES = ("replicate", "fsdp")
"""The full-mesh batch sharding: data shards over BOTH axes (per-rank batch = B/N)."""


def _hsdp_shape(n_devices: int, tp: int, gpus_per_node: int) -> tuple[int, int, int]:
    """`(replicate, fsdp, tp)` for a world of `n_devices` — the ONE shape rule both the
    concrete and abstract mesh constructors share. A multiple-of-`gpus_per_node` world
    splits into in-node blocks; a smaller world (an inline sub-node run, or CPU sim at a
    non-multiple count) IS one in-node block (so the divisibility asserts still bite on
    the real shard dims)."""
    assert n_devices % tp == 0, f"device count {n_devices} not divisible by tp={tp}"
    in_node = gpus_per_node if n_devices % gpus_per_node == 0 else n_devices
    assert in_node % tp == 0, f"in-node block {in_node} not divisible by tp={tp}"
    return (n_devices // in_node, in_node // tp, tp)


def hsdp_mesh(tp: int = 1, gpus_per_node: int = 8) -> Mesh:
    """The 3-D HSDP+TP device mesh `(replicate, fsdp, tp)`. Both `fsdp` and `tp` are
    intra-node NVLink axes carved from a node's GPUs (`fsdp * tp = GPUS_PER_NODE`); `tp` is
    the FAST-VARYING / minor axis so a tp group is adjacent GPUs, and a node's contiguous
    block in `jax.devices()` becomes one `(fsdp, tp)` plane. `replicate` (= n_devices //
    GPUS_PER_NODE) is the across-node axis. `tp = 1` is a degenerate `(replicate, fsdp, 1)`
    mesh — identical to the old 2-D mesh for any `("replicate","fsdp")`-only sharding, so
    behaviour-preserving."""
    devices = np.array(jax.devices())
    replicate, fsdp, tp = _hsdp_shape(devices.size, tp, gpus_per_node)
    return Mesh(devices.reshape(replicate, fsdp, tp), axis_names=("replicate", "fsdp", "tp"))


def single_device_mesh() -> Mesh:
    """The degenerate `(1, 1, 1)` mesh for a domain that is single-device BY CONSTRUCTION
    (the toys: one CPU process, seconds of training, so no `dp` to author). Asserts the
    world it assumes rather than absorbing whatever devices happen to be visible — a toy
    started inside someone's 8-GPU allocation is a mis-targeted job, not a free speedup."""
    assert jax.process_count() == 1 and jax.device_count() == 1, (
        f"single-device by construction, found {jax.process_count()} processes × "
        f"{jax.local_device_count()} local devices"
    )
    return hsdp_mesh()


def hsdp_abstract_mesh(dp: int, tp: int, gpus_per_node: int) -> AbstractMesh:
    """The mesh SHAPE a run config implies, with no devices — exactly the `hsdp_mesh` a
    run realizing `dp` devices builds, since `dp` declares the device count under BOTH
    launch modes. What config-build placement validation (`placement.from_config`) runs
    against, so a config refuses at submit validation, before any allocation."""
    return AbstractMesh(_hsdp_shape(dp, tp, gpus_per_node), ("replicate", "fsdp", "tp"))


def place_via_shardings[T](tree: T, shardings: T) -> T:
    """Place each array leaf of `tree` onto the matching `NamedSharding` leaf of `shardings`
    (a same-structure pytree, e.g. from a model's `.shardings(mesh)`). Static / non-array
    leaves pass through. The apply path for an already-loaded frozen model (vs the jitted
    `out_shardings` init path for freshly-seeded params).

    Every leaf goes through `make_array_from_callback`, never `device_put`. Each process
    loaded the same complete frozen leaf, so the callback can serve its addressable shards
    directly. `device_put(local, multi-process sharding)` first checks cross-host replica
    equality; for an FSDP target leaf replicated across the process axis that check gathers
    the full leaf on every process before slicing it (56 GiB for full32L's stacked MLP
    weights)."""
    is_array = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")  # noqa: E731
    place = lambda a, s: jax.make_array_from_callback(  # noqa: E731
        a.shape, s, lambda index: a[index]
    )
    return jax.tree.map(
        lambda a, s: place(a, s) if is_array(a) else a,
        tree,
        shardings,
        is_leaf=lambda x: isinstance(x, NamedSharding),
    )


def place_target[M: DecomposedModel](tgt: M, mesh: Mesh) -> M:
    """Eager placement of an already-loaded frozen target onto its own declared per-leaf
    shardings (`tgt.shardings(mesh)`). The apply path for loaded weights; freshly-seeded
    params go through the jitted `out_shardings` init path (`init_placed`) instead."""
    return place_via_shardings(tgt, cast(M, tgt.shardings(mesh)))


def assert_divisible(dim: int, mesh: Mesh, axis: str, what: str) -> None:
    """Fail loud if a dim sharded on mesh `axis` cannot tile that axis. Uniform across mesh
    sizes — at axis size 1 it is trivially true, so there is no single-device special case.
    `what` names the model / field / axis so a non-dividing dim crashes with a clear
    message rather than silently replicating."""
    n = mesh.shape[axis]
    assert dim % n == 0, f"{what}: dim {dim} not divisible by mesh axis '{axis}' size {n}"


def batch_shard_leading(x: jax.Array, mesh: Mesh | None) -> jax.Array:
    """In-jit `with_sharding_constraint` pinning the LEADING (batch) axis over the FULL mesh
    (`('replicate', 'fsdp')`), the rest replicated. `mesh is None` (single device) is a
    passthrough. Keeps the masked re-forwards on per-rank sub-batches (activation memory
    1/N)."""
    if mesh is None:
        return x
    spec = [BATCH_AXES] + [None] * (x.ndim - 1)
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))


def shard_batch(full_global: jax.Array, mesh: Mesh, batch_axis: int) -> jax.Array:
    """Shard `full_global` over the FULL mesh (`('replicate', 'fsdp')`) along `batch_axis`.
    Generated identically on every process (same seed), so each process slices out its
    process-local sub-batch and `make_array_from_process_local_data` does the device
    placement.

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
    spec: list[str | tuple[str, ...] | None] = [None] * full_global.ndim
    spec[batch_axis] = BATCH_AXES
    sharding = NamedSharding(mesh, P(*spec))

    per_proc = B // n_proc
    idx = jax.process_index()
    sl = [slice(None)] * full_global.ndim
    sl[batch_axis] = slice(idx * per_proc, (idx + 1) * per_proc)
    local = full_global[tuple(sl)]
    return jax.make_array_from_process_local_data(sharding, local, full_global.shape)
