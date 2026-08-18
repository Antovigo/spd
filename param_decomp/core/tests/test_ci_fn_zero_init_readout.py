"""`zero_init_readout` zeroes every readout head (bias 0.5) and touches nothing else."""

import dataclasses

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    MHACIAttention,
    init_chunkwise_transformer_ci_fn,
)
from param_decomp.core.components import SiteSpec

_ARCH = ChunkwiseTransformerCIArch(
    chunks=(Chunk(input_taps=("t0",), output_sites=("s0", "s1")),),
    input_dim=16,
    d_model=32,
    n_blocks=1,
    attention=MHACIAttention(n_heads=4),
    ffn_hidden=64,
    ffn_kind="gelu",
    learned_norm_scale=False,
    zero_init_readout=True,
    dual=True,
)
_SITES = tuple(SiteSpec(name=f"s{i}", C=8, d_in=16, d_out=16) for i in range(2))


def test_zero_init_readout_heads_are_zero_with_half_bias() -> None:
    fn = init_chunkwise_transformer_ci_fn(_ARCH, _SITES, jax.random.PRNGKey(0), dual=True)
    chunk = fn.chunks
    for ws, bs in ((chunk.out_ws, chunk.out_bs), (chunk.hidden_out_ws, chunk.hidden_out_bs)):
        for w, b in zip(ws, bs, strict=True):
            assert jnp.all(w == 0)
            assert jnp.all(b == 0.5)


def test_zero_init_readout_leaves_the_trunk_bit_identical() -> None:
    zeroed = init_chunkwise_transformer_ci_fn(_ARCH, _SITES, jax.random.PRNGKey(0), dual=True)
    default = init_chunkwise_transformer_ci_fn(
        dataclasses.replace(_ARCH, zero_init_readout=False),
        _SITES,
        jax.random.PRNGKey(0),
        dual=True,
    )
    assert jnp.array_equal(zeroed.chunks.in_proj_w, default.chunks.in_proj_w)
    for zero_block, default_block in zip(zeroed.chunks.blocks, default.chunks.blocks, strict=True):
        assert jnp.array_equal(zero_block.w1, default_block.w1)
        assert jnp.array_equal(zero_block.wq, default_block.wq)
