"""Sharding tests. Run under simulated multi-device CPU via the env in conftest.

These guard the harness pitfall (NOTES): `shard_batch` must reconstruct the FULL
global array across the mesh, not replicate a per-process slice. The GPU-count
invariance of the whole step is validated end-to-end by
`experiments/distributed_stacked_sites.py` at 1 vs N devices (bit-identical);
that needs distinct process-level device counts so it lives in the runnable
experiment, not here.
"""

import jax
import jax.numpy as jnp

from jax_single_pool.sharding import dp_mesh, shard_batch


def test_shard_batch_preserves_global_data():
    mesh = dp_mesh()
    n = mesh.devices.size
    B = 8 * n
    full = jax.random.normal(jax.random.PRNGKey(0), (3, B, 5))
    sharded = shard_batch(full, mesh, batch_axis=1)
    assert sharded.shape == full.shape
    # the sharded array must equal the original global array (the harness pitfall
    # replicated a single slice instead, which this catches when n > 1).
    assert jnp.allclose(jnp.asarray(sharded), full)


def test_shard_batch_requires_divisible_batch():
    mesh = dp_mesh()
    n = mesh.devices.size
    if n == 1:
        return  # any batch divides 1
    full = jax.random.normal(jax.random.PRNGKey(1), (2, n + 1, 4))
    try:
        shard_batch(full, mesh, batch_axis=1)
    except AssertionError:
        return
    raise AssertionError("expected non-divisible batch to fail")
