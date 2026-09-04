"""The dual (shared-trunk) CI fn: two readout heads over ONE trunk (SPEC S37).

The claim these pin is that the trunk is shared BY CONSTRUCTION — it appears once in the
pytree, so one `eqx.filter_vjp` pullback with a `DualCI` cotangent returns a trunk gradient
that is exactly the sum of the two objectives' separate trunk gradients, and no optimizer can
double-count it. The torch analogue (`GlobalSharedTransformerCiFn.adopt_trunk`) shares by
module identity and needs a load-time value check to tell the two topologies apart; here the
pytree shape differs, so that trap is unreachable and there is nothing to check.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from param_decomp.core.ci_fn import (
    CI,
    DUAL_CI_ROLES,
    Chunk,
    ChunkwiseTransformerCIArch,
    CIFn,
    CIRole,
    DualCI,
    LayerwiseMLPCIArch,
    MHACIAttention,
    build_ci_fn,
    ci_for_role,
    is_dual,
)
from param_decomp.core.components import SiteSpec

D_TAP = 8
SITES = (
    SiteSpec(name="s0", d_in=D_TAP, d_out=4, C=3, group="s0"),
    SiteSpec(name="s1", d_in=D_TAP, d_out=4, C=5, group="s1"),
)


def _chunkwise_arch(dual: bool = False) -> ChunkwiseTransformerCIArch:
    return ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("t0",), output_sites=("s0", "s1")),),
        input_dim=D_TAP,
        d_model=8,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=16,
        ffn_kind="gelu",
        learned_norm_scale=False,
        dual=dual,
    )


def _mlp_arch(dual: bool = False) -> LayerwiseMLPCIArch:
    return LayerwiseMLPCIArch(
        hidden_dims=(6,), has_position_axis=False, input_names=("t0", "t1"), dual=dual
    )


ArchOf = Callable[..., ChunkwiseTransformerCIArch | LayerwiseMLPCIArch]
TapsOf = Callable[[], dict[str, Array]]


def _chunkwise_taps() -> dict[str, Array]:
    return {"t0": jax.random.normal(jax.random.key(1), (2, 4, D_TAP))}


def _mlp_taps() -> dict[str, Array]:
    keys = jax.random.split(jax.random.key(1), 2)
    return {
        "t0": jax.random.normal(keys[0], (2, D_TAP)),
        "t1": jax.random.normal(keys[1], (2, D_TAP)),
    }


ARCHES = [
    pytest.param(_chunkwise_arch, _chunkwise_taps, id="chunkwise"),
    pytest.param(_mlp_arch, _mlp_taps, id="layerwise_mlp"),
]


@pytest.mark.parametrize(("arch_of", "taps_of"), ARCHES)
def test_single_role_returns_bare_ci_and_refuses_the_hidden_head(
    arch_of: ArchOf, taps_of: TapsOf
) -> None:
    """A plain run never sees the dual vocabulary, and asking it for the hidden bundle is a
    wiring bug rather than a silent fallback to the output one."""
    ci_fn = build_ci_fn(arch_of(), SITES, jax.random.key(0))
    assert not is_dual(ci_fn) and ci_fn.roles == ("output",)
    ci = ci_fn(taps_of(), remat=False, placement=None)
    assert isinstance(ci, CI)
    assert ci_for_role(ci, "output") is ci
    with pytest.raises(AssertionError, match="no 'hidden' head"):
        ci_for_role(ci, "hidden")


@pytest.mark.parametrize(("arch_of", "taps_of"), ARCHES)
def test_dual_returns_two_distinct_bundles_over_every_site(
    arch_of: ArchOf, taps_of: TapsOf
) -> None:
    ci_fn = build_ci_fn(arch_of(dual=True), SITES, jax.random.key(0))
    assert is_dual(ci_fn) and ci_fn.roles == DUAL_CI_ROLES
    ci = ci_fn(taps_of(), remat=False, placement=None)
    assert isinstance(ci, DualCI)
    for role in DUAL_CI_ROLES:
        bundle = ci_for_role(ci, role)
        assert set(bundle.lower) == {s.name for s in SITES}
        for site in SITES:
            assert bundle.lower[site.name].shape[-1] == site.C
    # Two heads, not one tied to itself: a run whose heads agreed everywhere would report a
    # dual objective while training a single CI.
    for site in SITES:
        assert not jnp.allclose(
            ci.output.preactivations[site.name], ci.hidden.preactivations[site.name]
        )


@pytest.mark.parametrize(("arch_of", "taps_of"), ARCHES)
def test_trunk_gradient_is_the_sum_of_the_two_objectives(arch_of: ArchOf, taps_of: TapsOf) -> None:
    """The load-bearing property. One pullback with a `DualCI` cotangent must equal the sum of
    the two single-head pullbacks — that is what makes running the objectives sequentially and
    adding their gradients identical to fusing them, and what makes the trunk see both."""
    arch, taps = arch_of(dual=True), taps_of()
    ci_fn = build_ci_fn(arch, SITES, jax.random.key(0))

    def run(cf: CIFn) -> DualCI:
        result = cf(taps, remat=False, placement=None)
        assert isinstance(result, DualCI)
        return result

    ci, vjp = eqx.filter_vjp(run, ci_fn)
    zeros = jax.tree.map(jnp.zeros_like, ci)

    def cotangent_on(role: CIRole) -> DualCI:
        ones = jax.tree.map(jnp.ones_like, ci_for_role(ci, role))
        return DualCI(
            output=ones if role == "output" else zeros.output,
            hidden=ones if role == "hidden" else zeros.hidden,
        )

    both = jax.tree.map(jnp.ones_like, ci)
    (grad_both,) = vjp(both)
    (grad_output,) = vjp(cotangent_on("output"))
    (grad_hidden,) = vjp(cotangent_on("hidden"))

    summed = jax.tree.map(lambda a, b: a + b, grad_output, grad_hidden)
    for got, want in zip(
        jax.tree.leaves(eqx.filter(grad_both, eqx.is_inexact_array)),
        jax.tree.leaves(eqx.filter(summed, eqx.is_inexact_array)),
        strict=True,
    ):
        assert jnp.allclose(got, want, atol=1e-5), "trunk gradient is not the sum of the roles'"

    # And the split is real: each head's own gradient is reached by only its own objective.
    assert any(
        jnp.any(leaf != 0)
        for leaf in jax.tree.leaves(eqx.filter(grad_output, eqx.is_inexact_array))
    )
    assert any(
        jnp.any(leaf != 0)
        for leaf in jax.tree.leaves(eqx.filter(grad_hidden, eqx.is_inexact_array))
    )


@pytest.mark.parametrize(("arch_of", "taps_of"), ARCHES)
def test_dual_init_leaves_the_trunk_and_output_head_bit_identical(
    arch_of: ArchOf, taps_of: TapsOf
) -> None:
    """The hidden head folds off the output head's key instead of widening the split, so a dual
    run and a single-role run at the same seed start from the SAME trunk and output head — the
    two topologies are comparable from step 0, and existing goldens do not move."""
    arch, taps = arch_of(), taps_of()
    single = build_ci_fn(arch, SITES, jax.random.key(0))
    dual = build_ci_fn(arch_of(dual=True), SITES, jax.random.key(0))

    single_ci = single(taps, remat=False, placement=None)
    dual_ci = dual(taps, remat=False, placement=None)
    assert isinstance(single_ci, CI) and isinstance(dual_ci, DualCI)
    for site in SITES:
        assert jnp.array_equal(
            single_ci.preactivations[site.name], dual_ci.output.preactivations[site.name]
        ), "the dual fn's OUTPUT head diverged from the single-role fn at the same seed"
