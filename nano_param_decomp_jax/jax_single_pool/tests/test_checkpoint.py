"""Checkpoint round-trip + resume-continuity (the adversary state must persist)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from jax_single_pool.checkpoint import load_state, save_state
from jax_single_pool.pgd import PGDConfig, init_pgd_state
from jax_single_pool.scopes import BroadcastAcrossBatchScope
from jax_single_pool.step import LossCoeffs, init_train_state, make_step
from jax_single_pool.tests.test_single_pool import _toy


def _build():
    decomp, ci, pgd, source_c = _toy(jax.random.PRNGKey(0), use_delta=True)
    coeffs = LossCoeffs(faith=1.0, imp=1e-2, stoch=1.0, ppgd=1.0, p_imp=0.9)
    cfg = PGDConfig(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=3, use_delta_component=True)
    opt_main = optax.adam(3e-3)
    opt_ci = optax.adam(3e-3)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)
    step = make_step(coeffs, cfg, opt_main, opt_ci, source_c=source_c, use_delta_component=True)
    return state, step


def _fresh_reference():
    decomp, ci, _, source_c = _toy(jax.random.PRNGKey(0), use_delta=True)
    pgd = init_pgd_state(jax.random.PRNGKey(1), BroadcastAcrossBatchScope(), 3, source_c, (16,))
    return init_train_state(decomp, ci, pgd, optax.adam(3e-3), optax.adam(3e-3))


def test_resume_continues_trajectory(tmp_path: Path):
    state, step = _build()
    x = jax.random.normal(jax.random.PRNGKey(8), (3, 16, 8))

    for i in range(5):
        state, _ = step(state, x, jax.random.PRNGKey(200 + i))

    ckpt = tmp_path / "state.npz"
    save_state(ckpt, state)
    restored = load_state(ckpt, _fresh_reference())

    # The restored state must be leaf-identical (sources + adam moments included).
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        assert jnp.allclose(jnp.asarray(a), jnp.asarray(b))

    # And continuing from the restore must match continuing from the live state.
    s_live, s_rest = state, restored
    for i in range(5):
        k = jax.random.PRNGKey(300 + i)
        s_live, m_live = step(s_live, x, k)
        s_rest, m_rest = step(s_rest, x, k)
        assert jnp.allclose(m_live["total"], m_rest["total"])
