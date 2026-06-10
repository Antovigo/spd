"""GSPMD sharding plan for the Llama-8B single-pool step — the FSDP-style memory story.

The memory consumers, and how each is placed on the 1-D `dp` mesh:

  * frozen suffix (`Target`): REPLICATED. ~3.6B bf16 params (14 blocks + lm_head) ~=
    7.3GB/device. Small relative to activations; replicating avoids all-gathering the
    target every forward.
  * components (V/U) + their Adam states: SHARDED over `dp` (the FSDP analog). V/U is
    ~2.1B params; fp32 Adam m+v doubles that to ~17GB replicated. Sharding the C axis
    splits both across devices -> the per-device component+optimizer footprint scales
    1/n_dev.
  * CI fn + Adam states: SHARDED over `dp` along the largest axis (out_head 3C, in_proj).
  * PGD source (broadcast, (1,T,L,C+1)): REPLICATED. A single adversarial source shared
    across the global batch; it combines elementwise with the batch-sharded CI and its
    grad is AVG-reduced across shards (torch `reduce_source_grads`). Tiny vs activations,
    so replicating costs nothing; the C+1 axis is odd and cannot tile the mesh anyway.
  * residual input + all activations: BATCH-sharded over `dp`. The masked suffix
    re-forwards then run on per-device sub-batches -> activation memory scales 1/n_dev.
    This is what unlocks a global batch that OOMs replicated on one device.

Sharding V/U over the C axis keeps every einsum valid: `x @ V` contracts d_in and
produces a C-sharded result; `(.) @ U` contracts the sharded C and `jax.jit` inserts
the reduce-scatter / all-reduce. No manual collectives.
"""

import equinox as eqx
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jax_single_pool.ci_fn import CIFn
from jax_single_pool.llama8b import DecompVU, Target


def dp_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()), axis_names=("dp",))


def _put(x, sharding):
    return jax.device_put(x, sharding) if eqx.is_array(x) else x


def replicate_target(tgt: Target, mesh: Mesh) -> Target:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: _put(a, repl), tgt)


def shard_decomp_vu(vu: DecompVU, mesh: Mesh) -> DecompVU:
    """Shard each V/U over its C axis. Arrays carry a leading layer axis: V is
    (L, d_in, C) -> shard axis 2; U is (L, C, d_out) -> shard axis 1. Both put C on `dp`."""
    n = mesh.devices.size
    assert vu.Vg.shape[2] % n == 0, f"C={vu.Vg.shape[2]} not divisible by mesh size {n}"
    shard_V = NamedSharding(mesh, P(None, None, "dp"))  # (L, d_in, C)
    shard_U = NamedSharding(mesh, P(None, "dp", None))  # (L, C, d_out)
    return DecompVU(
        Vg=jax.device_put(vu.Vg, shard_V),
        Ug=jax.device_put(vu.Ug, shard_U),
        Vu=jax.device_put(vu.Vu, shard_V),
        Uu=jax.device_put(vu.Uu, shard_U),
        Vd=jax.device_put(vu.Vd, shard_V),
        Ud=jax.device_put(vu.Ud, shard_U),
    )


def shard_ci_fn(ci_fn: CIFn, mesh: Mesh) -> CIFn:
    """Shard the CI fn's largest matrices over `dp`. out_head (d_model, 3C) shards the
    3C axis; in_proj (total_in, d_model) and per-block weights shard d_model where it
    is divisible; small vectors (lns, inv_freq) replicate."""
    n = mesh.devices.size
    repl = NamedSharding(mesh, P())
    shard_last = NamedSharding(mesh, P(None, "dp"))

    def place(a):
        if not eqx.is_array(a):
            return a
        if a.ndim == 2 and a.shape[-1] % n == 0:
            return jax.device_put(a, shard_last)
        return jax.device_put(a, repl)

    return jax.tree.map(place, ci_fn)


def shard_source(source: dict, mesh: Mesh) -> dict:
    """Broadcast PGD source (1, T, L, C+1) -> REPLICATED over `dp`.

    The source is a single adversarial source shared across the whole global batch
    (leading batch axis = 1, broadcast); it combines elementwise with the batch-sharded
    CI (`mask = ci + (1-ci)*source[..., :-1]`) and its grad is AVG-reduced across shards
    (torch `reduce_source_grads`). Replication is the semantically correct placement and
    the torch analog. Sharding the trailing C+1 axis is invalid anyway: with the
    weight-delta channel C+1 is odd (8193) and not divisible by the mesh size, and would
    also fight the batch-sharded elementwise combine."""
    repl = NamedSharding(mesh, P())
    return {s: jax.device_put(v, repl) for s, v in source.items()}


def shard_batch(resid_global, mesh: Mesh):
    """Batch-shard the residual input (b, t, d) over `dp` (axis 0). Built from a full
    global array each process generated identically; each process contributes its slice."""
    n = mesh.devices.size
    b = resid_global.shape[0]
    assert b % n == 0, f"global batch {b} not divisible by mesh size {n}"
    sharding = NamedSharding(mesh, P("dp"))
    n_proc = jax.process_count()
    per = b // n_proc
    idx = jax.process_index()
    local = resid_global[idx * per : (idx + 1) * per]
    return jax.make_array_from_process_local_data(sharding, local, resid_global.shape)
