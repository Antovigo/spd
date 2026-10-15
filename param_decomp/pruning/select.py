"""`H_c`: each `S₀` component's prompts of `P` with the highest max-over-positions output CI under
the CI filter's trained CI fn (lower leaky, ceiling and prune applied), in one streaming pass."""

import time
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.step import (
    CIConstraints,
    FilterCIFn,
    Prepared,
    gather_batch,
    output_ci,
)
from param_decomp.core.log import logger
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.pruning.model import Slots

type TopK = tuple[Float[Array, "A k"], Int[Array, "A k"]]


def make_update(slots: Slots, k: int) -> Callable[..., TopK]:
    """`(..., tokens_all, idx, best) -> best`: merge one batch into the running top-`k`."""
    by_site: dict[str, np.ndarray] = {}
    for site in dict.fromkeys(slots.sites):
        by_site[site] = slots.index[np.asarray(slots.sites) == site]
    columns = {site: jnp.asarray(ids) for site, ids in by_site.items()}
    order = list(by_site)

    @eqx.filter_jit
    def update(
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        valid: Float[Array, " B"],
        best: TopK,
    ) -> TopK:
        clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
        per_site = [
            jnp.max(
                jnp.take(
                    jax.sharding.reshard(lower[site].astype(jnp.float32), P()),
                    columns[site],
                    axis=-1,
                ),
                axis=1,
            )
            for site in order
        ]
        ci = jax.sharding.reshard(jnp.concatenate(per_site, axis=-1), P())  # (B, A)
        ci = jnp.where(valid[:, None] > 0, ci, -1.0)  # padding rows never enter the top-k
        values = jnp.concatenate([best[0], ci.T], axis=1)
        indices = jnp.concatenate(
            [best[1], jnp.broadcast_to(idx[None, :], (ci.shape[1], idx.size))], axis=1
        )
        top, where = jax.lax.top_k(values, k)
        return top, jnp.take_along_axis(indices, where, axis=1)

    return update


def select_top_ci(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    tokens: np.ndarray,
    slots: Slots,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(A, k)` prompt indices into `tokens` (descending CI) and their CI values."""
    update = make_update(slots, k)
    tokens_all = jax.sharding.reshard(jnp.asarray(tokens), P())
    best: TopK = (
        jnp.full((slots.n, k), -2.0, jnp.float32),
        jnp.zeros((slots.n, k), jnp.int32),
    )
    n = tokens.shape[0]
    t0 = time.time()
    for i, start in enumerate(range(0, n, batch_size)):
        real = min(batch_size, n - start)
        idx = np.zeros(batch_size, np.int32)
        idx[:real] = np.arange(start, start + real)
        valid = np.zeros(batch_size, np.float32)
        valid[:real] = 1.0
        best = update(
            placed,
            prepared,
            ci_fn,
            constraints,
            tokens_all,
            jnp.asarray(idx),
            jnp.asarray(valid),
            best,
        )
        if i % 50 == 0:
            logger.info(f"select: {start + real}/{n} prompts ({time.time() - t0:.0f}s)")
    values, indices = (np.asarray(a) for a in best)
    assert (values > -1.0).all(), "fewer than k real prompts"
    return indices, values
