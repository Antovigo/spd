"""GSPMD sharding plan for the Llama-8B single-pool step — the FSDP-style memory story.

The memory consumers, and how each is placed on the 1-D `dp` mesh:

  * frozen suffix (`Target`): REPLICATED. ~3.6B bf16 params (14 blocks + lm_head) ~=
    7.3GB/device. Small relative to activations; replicating avoids all-gathering the
    target every forward.
  * components (V/U) + their Adam states: SHARDED over `dp` (the FSDP analog). The fp32
    masters + fp32 Adam m/v are the dominant non-activation footprint; sharding each
    site's C axis splits all three across devices -> 1/n_dev per device.
  * CI fn + Adam states: SHARDED over `dp` along the largest axis (out head, in_proj).
  * PGD source (broadcast scope, `{site: (1,T,C+1)}`): REPLICATED. A single adversarial
    source shared across the global batch; it combines elementwise with the batch-sharded
    CI and its grad reduction falls out of the global-mean loss (torch
    `reduce_source_grads` analog). Tiny vs activations, so replicating costs nothing;
    the C+1 axis is odd and cannot tile the mesh anyway. `SrcAdamState` mirrors it.
  * residual input + all activations: BATCH-sharded over `dp`. The masked suffix
    re-forwards then run on per-device sub-batches -> activation memory scales 1/n_dev.
    This is what unlocks a global batch that OOMs replicated on one device.

Sharding V/U over the C axis keeps every einsum valid: `x @ V` contracts d_in and
produces a C-sharded result; `(.) @ U` contracts the sharded C and `jax.jit` inserts
the reduce-scatter / all-reduce. No manual collectives.
"""

from functools import partial
from typing import Any

import equinox as eqx
import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from param_decomp.adversary import init_persistent_sources
from param_decomp.ci_fn import CIFn, CIFnArch, build_ci_fn
from param_decomp.components import DecompVU, SiteSpec, init_decomp_vu
from param_decomp.configs import BSCScope, SCScope
from param_decomp.log import logger
from param_decomp.sharding import dp_mesh
from param_decomp.sharding import shard_batch as _generic_shard_batch
from param_decomp.targets.llama8b import LlamaDecomposedModel

__all__ = [
    "dp_mesh",
    "replicate_target",
    "init_decomp_vu_placed",
    "init_ci_fn_placed",
    "init_sources_sharded",
    "shard_batch",
]


def _put(x: Any, sharding: NamedSharding) -> Any:
    return jax.device_put(x, sharding) if eqx.is_array(x) else x


def replicate_target(tgt: LlamaDecomposedModel, mesh: Mesh) -> LlamaDecomposedModel:
    repl = NamedSharding(mesh, P())
    return jax.tree.map(lambda a: _put(a, repl), tgt)


def init_decomp_vu_placed(
    sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh, shardable: bool
) -> DecompVU:
    """Seeded per-site V/U init, placed by SCALE (not by CI-fn arch). `shardable` (the
    caller's scale decision: mesh > 1 and every C divides it) → C-sharded; else replicated.

    Either way the init runs under jit with `out_shardings`, so each device generates only
    its own shard and no host-side full tree exists — eager `device_put` of a host tree onto
    a multi-process non-replicated sharding triggers a `process_allgather` (a 168 GiB
    allocation for a 12-layer chunk at C=24576). When sharded, per site: V `(d_in, C_s)` on
    axis 1; U `(C_s, d_out)` on axis 0."""
    init = partial(init_decomp_vu, sites)
    if not shardable:
        return jax.jit(init, out_shardings=NamedSharding(mesh, P()))(key)
    site_c = {spec.name: spec.C for spec in sites}
    shard_V = NamedSharding(mesh, P(None, "dp"))
    shard_U = NamedSharding(mesh, P("dp", None))

    def place(path: tuple[Any, ...], shape: jax.ShapeDtypeStruct) -> NamedSharding:
        *_, site_key, vu_key = path
        assert isinstance(site_key, jax.tree_util.DictKey), path
        assert isinstance(vu_key, jax.tree_util.SequenceKey), path
        is_V = vu_key.idx == 0
        assert shape.shape[1 if is_V else 0] == site_c[site_key.key], (path, shape.shape)
        return shard_V if is_V else shard_U

    out_shardings = jax.tree_util.tree_map_with_path(place, jax.eval_shape(init, key))
    return jax.jit(init, out_shardings=out_shardings)(key)


def _shard_last_axis(mesh: Mesh, ndim: int) -> NamedSharding:
    """Tile an array's LAST axis over `dp`, replicating every earlier axis."""
    return NamedSharding(mesh, P(*([None] * (ndim - 1)), "dp"))


def init_ci_fn_placed(
    arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray, mesh: Mesh, shardable: bool
) -> CIFn:
    """Seeded CI-fn init (any arch, via `build_ci_fn`), placed by SCALE — not arch type.
    `shardable` → shard every 2-D+ matrix on its LAST axis where it tiles the mesh (the
    chunkwise `out_w` ΣC axis, attention / in_proj output axes; 1-D vectors replicate);
    else replicate everything. Same jit + `out_shardings` no-host-tree rationale as V/U.

    A 2-D+ matrix whose last axis does NOT tile the mesh falls back to replication — fine
    for correctness but a potential memory surprise at scale, so it's logged rather than
    silent."""
    n = mesh.devices.size
    repl = NamedSharding(mesh, P())
    init = partial(build_ci_fn, arch, sites)
    if not shardable:
        return jax.jit(init, out_shardings=repl)(key)

    untiled: list[tuple[int, ...]] = []  # 2-D+ matrices whose last axis isn't mesh-divisible

    def place(shape: jax.ShapeDtypeStruct) -> NamedSharding:
        if shape.ndim < 2:
            return repl  # 1-D vectors (biases, inv_freq) always replicate
        if shape.shape[-1] % n == 0:
            return _shard_last_axis(mesh, shape.ndim)
        untiled.append(shape.shape)
        return repl

    out_shardings = jax.tree.map(place, jax.eval_shape(init, key))
    if untiled:
        logger.info(
            "ci_fn placement: %d matrices replicated (last axis not divisible by mesh=%d): %s",
            len(untiled),
            n,
            untiled,
        )
    return jax.jit(init, out_shardings=out_shardings)(key)


def init_sources_sharded(
    site_names: tuple[str, ...],
    site_component_counts: tuple[int, ...],
    seq_len: int,
    scope: SCScope | BSCScope,
    global_batch: int,
    key: PRNGKeyArray,
    mesh: Mesh,
) -> dict[str, Array]:
    """Seeded PPGD-source init -> placed per scope (jit + `out_shardings`; same
    no-host-tree rationale as `init_decomp_vu_placed`).

    `sc`: `{site: (1, T, C+1)}` REPLICATED over `dp`. One adversarial source shared
    across the whole global batch (leading batch axis = 1, broadcast); it combines
    elementwise with the batch-sharded CI (`mask = ci + (1-ci)*source[..., :-1]`) and its
    grad is AVG-reduced across shards (torch `reduce_source_grads`).

    `bsc`: `{site: (B, T, C+1)}` BATCH-SHARDED over `dp` (axis 0), aligning each batch
    element's source with that element's `shard_batch`-placed residual/CI. The source is
    independent per element, so the per-element grad is already shard-local — NO
    cross-rank reduction, matching torch's `_skip_all_reduce`. (Requires
    `global_batch % n_dev == 0`, the same divisibility `shard_batch` needs.)

    Sharding the trailing C+1 axis is invalid for either scope: with the weight-delta
    channel C+1 is odd and not divisible by the mesh size, and would also fight the
    batch-sharded elementwise combine."""
    match scope:
        case SCScope():
            leading_shape, placement = (1, seq_len), NamedSharding(mesh, P())
        case BSCScope():
            leading_shape = (global_batch, seq_len)
            placement = NamedSharding(mesh, P("dp", None, None))
    init = partial(init_persistent_sources, site_names, site_component_counts, leading_shape)
    return jax.jit(init, out_shardings=placement)(key)


def shard_batch(resid_global: jax.Array, mesh: Mesh) -> jax.Array:
    """Batch-shard the residual input (b, t, d) over `dp` (axis 0)."""
    return _generic_shard_batch(resid_global, mesh, batch_axis=0)
