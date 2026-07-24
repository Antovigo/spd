"""Tests for the `coupled` component initialization (`init_coupled_component_stacks`)."""

import jax
import jax.numpy as jnp

from param_decomp.components import SiteSpec, init_coupled_component_stacks


def _random_weight(d_out: int, d_in: int) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(0), (d_out, d_in))


def _component_sum(V: jax.Array, U: jax.Array) -> jax.Array:
    return (V @ U).T


def _col_space_residual(u: jax.Array, w: jax.Array) -> jax.Array:
    q_out, _, _ = jnp.linalg.svd(w, full_matrices=False)
    return u - (u @ q_out) @ q_out.T


def _row_space_residual(v: jax.Array, w: jax.Array) -> jax.Array:
    _, _, vh = jnp.linalg.svd(w, full_matrices=False)
    q_in = vh.T
    return v - q_in @ (q_in.T @ v)


def _init_single(w: jax.Array, c: int, seed: int) -> tuple[jax.Array, jax.Array]:
    d_out, d_in = w.shape
    sites = (SiteSpec(name="m", d_in=d_in, d_out=d_out, C=c),)
    vu = init_coupled_component_stacks(sites, {"m": w}, jax.random.PRNGKey(seed))
    return vu.site("m")


def test_coupled_seeds_are_unit_norm_and_derived_side_is_raw_w_image():
    w = _random_weight(10, 6)  # d_in < d_out: V seeded, U derived
    V, U = _init_single(w, c=8, seed=0)
    assert jnp.allclose(jnp.linalg.norm(V, axis=0), jnp.ones(8), atol=1e-5)
    assert jnp.allclose(U, (w @ V).T, atol=1e-6)
    assert jnp.abs(_col_space_residual(U, w)).max() < 1e-5

    w_t = w.T  # d_in > d_out: U seeded, V derived
    V_t, U_t = _init_single(w_t, c=8, seed=0)
    assert jnp.allclose(jnp.linalg.norm(U_t, axis=1), jnp.ones(8), atol=1e-5)
    assert jnp.allclose(V_t, w_t.T @ U_t.T, atol=1e-6)
    assert jnp.abs(_row_space_residual(V_t, w_t)).max() < 1e-5


def test_coupled_sum_is_w_restricted_to_seed_span():
    w = _random_weight(10, 6)
    V, U = _init_single(w, c=4, seed=0)
    # sum_c u_c v_c^T = W V V^T exactly.
    assert jnp.allclose(_component_sum(V, U), w @ V @ V.T, atol=1e-5)


def test_coupled_is_seed_deterministic():
    w = _random_weight(10, 6)
    V_a, U_a = _init_single(w, c=8, seed=7)
    V_b, U_b = _init_single(w, c=8, seed=7)
    assert jnp.array_equal(V_a, V_b) and jnp.array_equal(U_a, U_b)


def test_coupled_stacked_sites_get_independent_draws_coupled_to_their_own_w():
    # Two sites sharing one (d_in, d_out, C) shape group stack on one axis; each slice
    # must be coupled to ITS OWN W with an independent seed draw.
    key = jax.random.PRNGKey(3)
    w_a = jax.random.normal(jax.random.fold_in(key, 0), (10, 6))
    w_b = jax.random.normal(jax.random.fold_in(key, 1), (10, 6))
    sites = (
        SiteSpec(name="a", d_in=6, d_out=10, C=8),
        SiteSpec(name="b", d_in=6, d_out=10, C=8),
    )
    vu = init_coupled_component_stacks(sites, {"a": w_a, "b": w_b}, jax.random.PRNGKey(0))
    V_a, U_a = vu.site("a")
    V_b, U_b = vu.site("b")
    assert not jnp.allclose(V_a, V_b)
    assert jnp.allclose(U_a, (w_a @ V_a).T, atol=1e-6)
    assert jnp.allclose(U_b, (w_b @ V_b).T, atol=1e-6)
