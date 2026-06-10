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


def test_jitted_sharded_inits_match_eager_values():
    """`init_*_sharded` must be a placement-only change: same values as the eager init
    fns (threefry is partitionable, so generating under jit with `out_shardings` cannot
    perturb the stream — only op fusion can reassociate the scaling, SPEC D4: rel ~1e-7),
    with the expected sharded placements."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jax_single_pool.ci_fn import CIArch, init_ci_fn
    from jax_single_pool.llama8b import LayerRange, LlamaConfig, init_decomp_vu
    from jax_single_pool.llama8b_sharding import (
        init_ci_fn_sharded,
        init_decomp_vu_sharded,
        init_sources_sharded,
    )
    from jax_single_pool.lm import SiteSpec
    from jax_single_pool.train import init_sources

    mesh = dp_mesh()
    n = mesh.devices.size
    cfg = LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )
    C = 8 * n
    layer_range = LayerRange(2, 3)

    vu_sharded = init_decomp_vu_sharded(cfg, C, layer_range.n_layers, jax.random.PRNGKey(1), mesh)
    vu_eager = init_decomp_vu(cfg, C, layer_range.n_layers, jax.random.PRNGKey(1))
    assert isinstance(vu_sharded.Vg.sharding, NamedSharding)
    assert isinstance(vu_sharded.Ug.sharding, NamedSharding)
    assert vu_sharded.Vg.sharding.spec == P(None, None, "dp")
    assert vu_sharded.Ug.sharding.spec == P(None, "dp", None)
    for got, want in zip(jax.tree.leaves(vu_sharded), jax.tree.leaves(vu_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    sites = (
        SiteSpec("layers.2.mlp.gate_proj", cfg.n_embd, cfg.n_intermediate, C),
        SiteSpec("layers.2.mlp.down_proj", cfg.n_intermediate, cfg.n_embd, C),
    )
    arch = CIArch(d_model=16, n_blocks=1, n_heads=2, mlp_hidden=8 * n)
    ci_sharded = init_ci_fn_sharded(arch, sites, jax.random.PRNGKey(2), mesh)
    ci_eager = init_ci_fn(arch, sites, jax.random.PRNGKey(2))
    for got, want in zip(jax.tree.leaves(ci_sharded), jax.tree.leaves(ci_eager), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert jnp.allclose(jnp.asarray(got), want, rtol=1e-6, atol=0)

    site_names = tuple(s.name for s in sites)
    site_Cs = tuple(s.C for s in sites)
    src_sharded = init_sources_sharded(site_names, site_Cs, 16, jax.random.PRNGKey(3), mesh)
    src_eager = init_sources(site_names, site_Cs, 16, jax.random.PRNGKey(3))
    for name in site_names:
        src_sharding = src_sharded[name].sharding
        assert isinstance(src_sharding, NamedSharding)
        assert src_sharding.spec == P()
        assert jnp.allclose(jnp.asarray(src_sharded[name]), src_eager[name], rtol=1e-6, atol=0)
