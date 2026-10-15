"""Synthetic checks of `variance.py` (plan §10, checks 2-4), float64 on CPU.

JAX_PLATFORMS=cpu python -m pytest -q test_variance.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from variance import (  # noqa: E402
    Weights,
    analyse_in_memory,
    finalize,
    secant,
    silu,
    writer_activation,
)

M = 4000  # prompts per synthetic test


def weights(Ug: np.ndarray, Uu: np.ndarray, Vd: np.ndarray) -> Weights:
    return Weights(Ug=jnp.asarray(Ug, jnp.float64), Uu=jnp.asarray(Uu, jnp.float64),
                   Vd=jnp.asarray(Vd, jnp.float64), ud2=jnp.ones(Vd.shape[1], jnp.float64))  # fmt: skip


def random_block(seed: int, N: int = 7, G: int = 4, U: int = 5, D: int = 5):  # noqa: ANN201
    rng = np.random.default_rng(seed)
    W = weights(rng.normal(size=(G, N)), rng.normal(size=(U, N)), rng.normal(size=(N, D)))
    tg = jnp.asarray(rng.normal(size=(300, G)) + 0.5)
    tu = jnp.asarray(rng.normal(size=(300, U)) - 0.3)
    return W, tg, tu


def test_identities() -> None:
    """(4): reference + up terms + gate terms = t_d; (5): M^u0 + sum_c' J = M^u; shares sum to 1.
    D = 5 with wc = 3 also exercises the writer padding."""
    W, tg, tu = random_block(0)
    res, tm, refs = analyse_in_memory(tg, tu, W, wc=3)
    t_d = writer_activation(tg, tu, W)
    total = refs.tref[None] + tm["Mu"].sum(-1) + tm["Mg"].sum(-1)
    np.testing.assert_allclose(total, t_d, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(tm["t_d"], t_d, rtol=1e-12)
    np.testing.assert_allclose(tm["Mu0"] + tm["J_sum_gates"], tm["Mu"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(res["phi_Mu"].sum(1) + res["phi_Mg"].sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(res["phi_Mu0"] + res["phi_J"].sum(2), res["phi_Mu"], atol=1e-9)


def test_secant_taylor_branch() -> None:
    hbar = jnp.linspace(-6.0, 6.0, 25)
    for d in (1e-4, -5e-4, 9e-4, 2e-3):
        h = hbar + d
        exact = (silu(h) - silu(hbar)) / d
        np.testing.assert_allclose(secant(h, hbar), exact, rtol=1e-6, atol=1e-9)


def two_neurons(g1: np.ndarray, u1: np.ndarray, rng: np.random.Generator):  # noqa: ANN201
    """Neuron 0 holds the scenario: one gate reader writing `g1`, one up reader writing `u1`.
    Neuron 1 carries an independent signal of order-1 variance (gate held open at 3, up reader
    standard normal). One writer reads both neurons with weight 1."""
    W = weights(np.eye(2), np.eye(2), np.ones((2, 1)))
    tg = jnp.asarray(np.stack([g1, np.full(M, 3.0)], 1))
    tu = jnp.asarray(np.stack([u1, rng.normal(size=M)], 1))
    return analyse_in_memory(tg, tu, W, wc=1)[0]


LEAK = 1e-3
"""Bound on a leaking term's weight; the order-1 signals carry weights of 4 to 14."""


def test_property1_gate_always_off() -> None:
    rng = np.random.default_rng(1)
    res = two_neurons(np.full(M, -10.0), rng.normal(size=M), rng)
    assert abs(res["phi_Mu"][0, 0]) < 1e-3
    assert abs(res["a_Mu"][0, 0]) < LEAK


def test_property2_gate_varies_while_off() -> None:
    rng = np.random.default_rng(2)
    res = two_neurons(rng.uniform(-12, -8, size=M), np.ones(M), rng)
    assert abs(res["phi_Mg"][0, 0]) < 1e-3
    assert abs(res["a_Mg"][0, 0]) < LEAK


def test_property3_up_varies_only_while_gate_off() -> None:
    rng = np.random.default_rng(3)
    on = rng.random(M) < 0.5
    g1 = np.where(on, 4.0, -10.0)
    u1 = np.where(on, 1.0, rng.normal(3.0, 1.0, size=M))
    res = two_neurons(g1, u1, rng)
    assert abs(res["phi_Mu"][0, 0]) < 1e-3
    assert abs(res["a_Mu"][0, 0]) < LEAK
    assert res["phi_Mg"][0, 0] > 0.1  # the gate's switching is credited


def test_property4_gate_switch_credited() -> None:
    rng = np.random.default_rng(4)
    on = rng.random(M) < 0.5
    g1 = np.where(on, 4.0, rng.uniform(-12, -8, size=M))
    W = weights(np.eye(1), np.eye(1), np.ones((1, 1)))
    res = analyse_in_memory(jnp.asarray(g1[:, None]), jnp.ones((M, 1)), W, wc=1)[0]
    assert res["phi_Mg"][0, 0] > 0.999
    assert abs(res["phi_Mu"][0, 0]) < 1e-3


def test_zero_reference_example_constant_gate_reader() -> None:
    """Gate reader 0 writes the constant 5, gate reader 1 writes -10 or 0 (section 5)."""
    rng = np.random.default_rng(5)
    g = np.stack([np.full(M, 5.0), np.where(rng.random(M) < 0.5, -10.0, 0.0)], 1)
    W = weights(np.ones((2, 1)), np.eye(1), np.ones((1, 1)))
    res = analyse_in_memory(jnp.asarray(g), jnp.asarray(rng.uniform(1, 2, size=(M, 1))), W, wc=1)[0]
    assert abs(res["phi_Mg"][0, 0]) < 1e-12
    assert np.abs(res["phi_J"][0, :, 0]).max() < 1e-12
    assert res["phi_Mg"][0, 1] > 0.5


def test_excluded_writer() -> None:
    n = 10
    t = np.full((n, 2), 3.0)
    t[:, 1] = np.arange(n)
    res = finalize(n, t.sum(0), (t**2).sum(0), {"Mu": (np.zeros((2, 1)), np.zeros((2, 1)))},
                   np.ones(2))  # fmt: skip
    assert res["excluded"].tolist() == [True, False]
    assert np.isnan(res["phi_Mu"][0, 0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
