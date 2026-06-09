"""Pure-function unit tests for the single-pool VPD step. CPU-runnable."""

import jax
import jax.numpy as jnp
import optax
import pytest

from jax_single_pool.losses import (
    CIParams,
    Decomposition,
    ci_envelope,
    faithfulness_loss,
    interpolate_mask,
)
from jax_single_pool.pgd import (
    PGDConfig,
    _adam_ascend,
    adversarial_recon,
    init_pgd_state,
    pgd_warmup,
)
from jax_single_pool.scopes import (
    BroadcastAcrossBatchScope,
    PerBatchPerPositionScope,
    RepeatAcrossBatchScope,
    SingleSourceScope,
    expand_source_to_batch,
    source_leading_dims,
)
from jax_single_pool.step import LossCoeffs, init_train_state, make_step

jax.config.update("jax_enable_x64", True)


def _toy(key, S=3, D=8, C=4, B=16, use_delta=True):
    kW, kV, kU, kw, kb, kpgd = jax.random.split(key, 6)
    decomp = Decomposition(
        V=jax.random.normal(kV, (S, D, C)) * 0.2,
        U=jax.random.normal(kU, (S, C, D)) * 0.2,
        W_target=jax.random.normal(kW, (S, D, D)) * 0.2,
    )
    ci = CIParams(w=jax.random.normal(kw, (S, D, C)) * 0.2, b=jax.random.normal(kb, (S, C)) * 0.1)
    source_c = C + 1 if use_delta else C
    pgd = init_pgd_state(kpgd, BroadcastAcrossBatchScope(), S, source_c, (B,))
    return decomp, ci, pgd, source_c


def test_scope_leading_dims():
    bd = (32, 12)
    assert source_leading_dims(SingleSourceScope(), bd) == (1, 1)
    assert source_leading_dims(BroadcastAcrossBatchScope(), bd) == (1, 12)
    assert source_leading_dims(RepeatAcrossBatchScope(4), bd) == (4, 12)
    assert source_leading_dims(PerBatchPerPositionScope(), bd) == (32, 12)


def test_repeat_scope_must_divide_batch():
    with pytest.raises(AssertionError):
        source_leading_dims(RepeatAcrossBatchScope(5), (32,))


def test_expand_source_broadcast_and_repeat():
    # broadcast: leading 1 -> B
    s = jnp.ones((1, 3))
    assert expand_source_to_batch(s, (8,)).shape == (8, 3)
    # repeat: leading n | B
    s = jnp.arange(4 * 3).reshape(4, 3).astype(float)
    out = expand_source_to_batch(s, (8,))
    assert out.shape == (8, 3)
    # rows 0 and 4 (= 0 + n) tile from source row 0
    assert jnp.allclose(out[0], s[0]) and jnp.allclose(out[4], s[0])


def test_interpolate_mask_no_delta():
    ci = jnp.array([[0.2, 0.8]])
    src = jnp.array([[1.0, 0.0]])
    m = interpolate_mask(ci, src, use_delta_component=False)
    # ci + (1-ci)*src
    assert jnp.allclose(m, jnp.array([[1.0, 0.8]]))


def test_interpolate_mask_delta_channel_raw():
    ci = jnp.array([[0.2, 0.8]])
    src = jnp.array([[1.0, 0.0, 0.5]])  # last is the delta channel
    m = interpolate_mask(ci, src, use_delta_component=True)
    assert jnp.allclose(m, jnp.array([[1.0, 0.8, 0.5]]))


def test_ci_envelope_in_unit_interval():
    key = jax.random.PRNGKey(1)
    _, ci, _, _ = _toy(key)
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 16, 8))
    out = ci_envelope(ci, x)
    assert out.shape == (3, 16, 4)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_faithfulness_nonnegative_and_zero_at_exact():
    S, D, C = 2, 6, 6
    V = jnp.eye(D)[:, :C][None].repeat(S, 0)
    U = jnp.eye(C, D)[None].repeat(S, 0)
    W = jnp.einsum("sic,scd->sid", V, U)
    assert float(faithfulness_loss(Decomposition(V, U, W))) == pytest.approx(0.0, abs=1e-10)


def test_pgd_warmup_scan_equals_python_loop():
    """The load-bearing bit-exactness check (cf. stage6 check (a))."""
    key = jax.random.PRNGKey(3)
    decomp, ci_params, pgd, _ = _toy(key, use_delta=True)
    x = jax.random.normal(jax.random.PRNGKey(4), (3, 16, 8))
    ci = ci_envelope(ci_params, x)
    cfg = PGDConfig(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=5, use_delta_component=True)

    scanned = pgd_warmup(decomp, x, ci, pgd, (16,), cfg)

    # python-loop reference
    decomp_det = jax.tree.map(jax.lax.stop_gradient, decomp)
    src, adam = pgd.sources, pgd.adam
    for _ in range(cfg.n_warmup):
        g = jax.grad(lambda s: adversarial_recon(decomp_det, x, ci, s, (16,), True))(src)
        src, adam = _adam_ascend(src, g, adam, cfg)

    assert float(jnp.max(jnp.abs(scanned.sources - src))) < 1e-11


def test_pgd_ascends_recon():
    key = jax.random.PRNGKey(5)
    decomp, ci_params, pgd, _ = _toy(key, use_delta=True)
    x = jax.random.normal(jax.random.PRNGKey(6), (3, 16, 8))
    ci = ci_envelope(ci_params, x)
    cfg = PGDConfig(lr=0.2, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=10, use_delta_component=True)
    before = float(adversarial_recon(decomp, x, ci, pgd.sources, (16,), True))
    refined = pgd_warmup(decomp, x, ci, pgd, (16,), cfg)
    after = float(adversarial_recon(decomp, x, ci, refined.sources, (16,), True))
    assert after > before


def test_step_finite_and_faith_decreases():
    key = jax.random.PRNGKey(7)
    decomp, ci, pgd, source_c = _toy(key, use_delta=True)
    coeffs = LossCoeffs(faith=1.0, imp=1e-2, stoch=1.0, ppgd=1.0, p_imp=0.9)
    cfg = PGDConfig(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=3, use_delta_component=True)
    opt_main = optax.adam(3e-3)
    opt_ci = optax.adam(3e-3)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)
    step = make_step(coeffs, cfg, opt_main, opt_ci, source_c=source_c, use_delta_component=True)
    x = jax.random.normal(jax.random.PRNGKey(8), (3, 16, 8))

    state, m0 = step(state, x, jax.random.PRNGKey(100))
    for v in m0.values():
        assert bool(jnp.isfinite(v))
    m = m0
    for i in range(40):
        state, m = step(state, x, jax.random.PRNGKey(200 + i))
    assert float(m["faith"]) < float(m0["faith"])


def test_step_grad_does_not_move_frozen_target():
    key = jax.random.PRNGKey(9)
    decomp, ci, pgd, source_c = _toy(key, use_delta=False)
    coeffs = LossCoeffs(faith=1.0, imp=1e-2, stoch=1.0, ppgd=1.0, p_imp=0.9)
    cfg = PGDConfig(
        lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=2, use_delta_component=False
    )
    opt_main = optax.adam(1e-2)
    opt_ci = optax.adam(1e-2)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)
    step = make_step(coeffs, cfg, opt_main, opt_ci, source_c=source_c, use_delta_component=False)
    x = jax.random.normal(jax.random.PRNGKey(10), (3, 16, 8))
    W_before = state.decomp.W_target
    for i in range(5):
        state, _ = step(state, x, jax.random.PRNGKey(300 + i))
    assert jnp.array_equal(state.decomp.W_target, W_before)
