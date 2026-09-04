"""Smooth-L0 (Geman–McClure) importance-minimality penalty (SPEC S7/S8/S9).

The penalty sums per-component mean activity and optionally adds a frequency term. These
checks pin the properties that motivate it over the retired `L_p` penalty: flat at the
origin (`phi'(0)=0`, no singularity to clip), bounded gradient
(`|phi'| <= 0.65/gamma`),
redescent for clearly-on components, and the half-saturation crossover `phi(gamma) = 1/2`.
"""

import jax
import jax.numpy as jnp

from param_decomp.core.configs import (
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
)
from param_decomp.core.losses import (
    imp_min_terms,
    importance_minimality_terms,
    scheduled_value_at,
)
from param_decomp.core.schedule import Knot, ScheduleConfig, get_scheduled_value


def _phi(c: jax.Array, gamma: float) -> jax.Array:
    return c**2 / (c**2 + gamma**2)


def test_phi_shape_invariants():
    for gamma in (1.0, 0.1):
        assert float(_phi(jnp.array(0.0), gamma)) == 0.0  # off -> exactly 0
        assert abs(float(_phi(jnp.array(gamma), gamma)) - 0.5) < 1e-6  # half-saturation
        assert float(_phi(jnp.array(10.0 * gamma), gamma)) > 0.99  # clearly-on -> ~1


def test_phi_gradient_flat_at_origin_and_bounded():
    """phi'(0) = 0 (no L_p cliff) and the peak |phi'| ~ 0.65/gamma sits at c = gamma/sqrt(3)."""
    for gamma in (1.0, 0.1):
        dphi = jax.grad(lambda c, g=gamma: _phi(c, g))
        assert float(dphi(jnp.array(0.0))) == 0.0
        cs = jnp.linspace(0.0, 5.0 * gamma, 4096)
        grads = jnp.abs(jax.vmap(dphi)(cs))
        peak = float(grads.max())
        assert peak <= 0.65 / gamma + 1e-3
        c_peak = float(cs[jnp.argmax(grads)])
        assert abs(c_peak - gamma / jnp.sqrt(3.0)) < 0.02 * gamma
        # redescent: gradient at a clearly-on point is far below the peak.
        assert float(dphi(jnp.array(5.0 * gamma))) < 0.2 * peak


def test_terms_match_manual_per_site_structure():
    ci = {
        "a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]]),
        "b": jnp.array([[0.3], [0.7]]),
    }
    gamma = 0.1
    n_positions = 2  # both sites have 2 rows; a' = B·T reproduces the old `log2(1 + sum)`
    activity, freq = importance_minimality_terms(
        ci, jnp.asarray(gamma), reference_datapoint_count=n_positions
    )

    expected_activity = jnp.zeros(())
    exp_freq = jnp.zeros(())
    for v in ci.values():
        sums = _phi(v, gamma).sum(axis=0)
        means = sums / v.shape[0]
        expected_activity = expected_activity + means.sum()
        exp_freq = exp_freq + (means * jnp.log2(1.0 + n_positions * means)).sum()
    assert jnp.allclose(activity, expected_activity)
    assert jnp.allclose(freq, exp_freq)


def test_anneal_and_dispatch():
    cfg = ImportanceMinimalityLossConfig(
        coeff=2e-4,
        gamma=ScheduleConfig(max_val=1.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1))),
        frequency=FrequencyMinimalityConfig(coeff=1e-4, reference_datapoint_count=64),
    )
    total = 100
    for step in (0, 50, total - 1):
        param = scheduled_value_at(jnp.asarray(step / (total - 1), jnp.float32), cfg.gamma)
        expected = get_scheduled_value(step, total, cfg.gamma)
        assert abs(float(param) - expected) < 1e-6
    assert abs(get_scheduled_value(0, total, cfg.gamma) - 1.0) < 1e-6
    assert abs(get_scheduled_value(total - 1, total, cfg.gamma) - 0.1) < 1e-6

    last = total - 1
    ci = {"a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]])}
    param = scheduled_value_at(jnp.asarray(last / (total - 1), jnp.float32), cfg.gamma)
    via_dispatch = imp_min_terms(ci, cfg, param)
    direct = importance_minimality_terms(ci, param, reference_datapoint_count=64)
    assert jnp.allclose(via_dispatch[0], direct[0])
    assert jnp.allclose(via_dispatch[1], direct[1])


def test_normalize_at_one_pins_a_full_component_to_one():
    """`c = 1` contributes exactly 1 at every gamma, so the anneal stops carrying an
    implicit coefficient ramp."""
    ci = {"a": jnp.ones((1, 1))}
    for gamma in (1.0, 0.1, 0.01):
        lp, _ = importance_minimality_terms(
            ci, jnp.asarray(gamma), reference_datapoint_count=None, normalize_at_one=True
        )
        assert abs(float(lp) - 1.0) < 1e-6, gamma
    bare, _ = importance_minimality_terms(
        ci, jnp.asarray(1.0), reference_datapoint_count=None, normalize_at_one=False
    )
    assert abs(float(bare) - 0.5) < 1e-6


def test_auto_reference_count_is_the_raw_firing_count_on_any_geometry():
    """`"auto"` resolves `a'` to each call's own `B·T`, so `a' · f_c` is the raw firing
    count — the torch oracle's `log2(1 + layer_sums * world_size)`. The point is that ONE
    config is correct on both tPD passes (SPEC T6 shares the frequency block) even though
    their `B·T` differ; a literal `a'` is right for one stream and off by their ratio on
    the other."""
    gamma = jnp.asarray(0.1)
    target = {"a": jnp.array([[0.0, 0.5, 1.0], [0.2, 0.0, 0.9]])}  # B·T = 2
    nontarget = {"a": jnp.array([[0.4, 0.1, 0.8]] * 8)}  # B·T = 8

    for ci in (target, nontarget):
        rows = next(iter(ci.values())).shape[0]
        auto = importance_minimality_terms(
            ci, gamma, reference_datapoint_count="auto", normalize_at_one=False
        )[1]
        literal = importance_minimality_terms(
            ci, gamma, reference_datapoint_count=rows, normalize_at_one=False
        )[1]
        assert jnp.allclose(auto, literal), "auto must equal that stream's own B·T"

        sums = _phi(next(iter(ci.values())), 0.1).sum(axis=0)  # the raw-count form
        assert jnp.allclose(auto, (sums / rows * jnp.log2(1.0 + sums)).sum())

    # one literal cannot serve both passes: correct for the target, wrong for the other
    assert not jnp.allclose(
        importance_minimality_terms(
            nontarget, gamma, reference_datapoint_count=2, normalize_at_one=False
        )[1],
        importance_minimality_terms(
            nontarget, gamma, reference_datapoint_count="auto", normalize_at_one=False
        )[1],
    )
