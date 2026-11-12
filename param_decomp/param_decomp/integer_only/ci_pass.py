"""The run's own CI fn over the pool: the ORIGINAL model's clean forward, then the output head's
lower-leaky CI (`ci_filter.step.output_ci`, unconstrained), the CI the run's alive sets and
masks threshold. Pool arrays are replicated and gathered in-jit over the batch axes; every
model, prepared weight and CI fn is a traced argument."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    FilterCIFn,
    Prepared,
    gather_batch,
    output_ci,
)
from param_decomp.experiments.lm.load_run import PlacedTransformer


@eqx.filter_jit
def chunk_max_ci(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    n_real: Array,
) -> dict[str, Array]:
    """`{site: (T, C)}` fp32 max CI over the batch's first `n_real` rows (the rest pad)."""
    clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
    lower = output_ci(placed, prepared, ci_fn, UNCONSTRAINED, clean, remat=False).lower
    real = (jnp.arange(idx.shape[0]) < n_real)[:, None, None]
    return {
        site: jax.sharding.reshard(
            jnp.where(real, v.astype(jnp.float32), -jnp.inf).max(axis=0), P()
        )
        for site, v in lower.items()
    }


@eqx.filter_jit
def chunk_ci_columns(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    columns: dict[str, Array],
) -> Array:
    """`(B, T, K)` fp32 CI of the requested components, `K = Σ_site len(columns[site])`, the
    sites in sorted name order, each site's columns in the given order; replicated."""
    clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
    lower = output_ci(placed, prepared, ci_fn, UNCONSTRAINED, clean, remat=False).lower
    return jnp.concatenate(
        [
            jnp.take(jax.sharding.reshard(lower[s], P()), columns[s], axis=-1).astype(jnp.float32)
            for s in sorted(columns)
        ],
        axis=-1,
    )
