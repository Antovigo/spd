"""Persistent-PGD source scopes — leading-dim shape of the adversarial source.

Mirrors `param_decomp/metrics/persistent_pgd_state.py`'s scope classes. The
scope fixes the source's *batch* leading dims (the trailing `source_c` axis is
always `C` or `C + 1` with a weight-delta channel):

  single                 -> [1] * len(batch_dims)        one source for the whole batch
  broadcast_across_batch -> [1, *batch_dims[1:]]         shared over batch elems, free per position
  repeat_across_batch(n) -> [n, *batch_dims[1:]]         n sources tiled over the batch dim
  per_batch_per_position -> [*batch_dims]                independent source per element

`broadcast_across_batch` is the production LM default. Under pure data-parallel
SPMD, a *shared* source (single/broadcast/repeat) is replicated across the dp
mesh and its grad is reduced over the sharded batch axis automatically by the
mean-loss — the torch `replica_sync_group` broadcast/AVG-reduce dance is gone.
`per_batch_per_position` shards with the batch (one source per element).
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class SingleSourceScope:
    pass


@dataclass(frozen=True)
class BroadcastAcrossBatchScope:
    pass


@dataclass(frozen=True)
class RepeatAcrossBatchScope:
    n_sources: int


@dataclass(frozen=True)
class PerBatchPerPositionScope:
    pass


SourceScope = (
    SingleSourceScope
    | BroadcastAcrossBatchScope
    | RepeatAcrossBatchScope
    | PerBatchPerPositionScope
)


def source_leading_dims(scope: SourceScope, batch_dims: tuple[int, ...]) -> tuple[int, ...]:
    match scope:
        case SingleSourceScope():
            return (1,) * len(batch_dims)
        case BroadcastAcrossBatchScope():
            return (1, *batch_dims[1:])
        case RepeatAcrossBatchScope(n_sources=n):
            assert batch_dims[0] % n == 0, (
                f"n_sources={n} must divide the per-rank batch dim {batch_dims[0]}"
            )
            return (n, *batch_dims[1:])
        case PerBatchPerPositionScope():
            return batch_dims


def scope_is_batch_sharded(scope: SourceScope) -> bool:
    """Whether the source's leading batch dim shards with the data-parallel batch.

    Only `per_batch_per_position` has a per-element source that follows the batch
    split; the replicated scopes hold one shared source per replica.
    """
    return isinstance(scope, PerBatchPerPositionScope)


def expand_source_to_batch(
    source: Float[Array, "*leading source_c"],
    batch_dims: tuple[int, ...],
) -> Float[Array, "*batch_dims source_c"]:
    """Broadcast/repeat a scoped source up to the full per-batch shape.

    Mirrors `get_ppgd_mask_infos`: leading dim 1 or == B broadcasts; otherwise it
    must divide B and is tiled. Returns shape `[*batch_dims, source_c]`.
    """
    B = batch_dims[0]
    N = source.shape[0]
    if N == 1 or N == B:
        return jnp.broadcast_to(source, (*batch_dims, source.shape[-1]))
    assert B % N == 0, f"source leading dim {N} must divide batch dim {B}"
    reps = (B // N,) + (1,) * (source.ndim - 1)
    return jnp.tile(source, reps)
