"""`core.ci_fn.ceiling.cap_ci`: the entrywise minimum and its gradient rule."""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.ci_fn.ceiling import cap_ci
from param_decomp.core.ci_fn.interface import CI, DualCI


def _ci(values: list[float]) -> CI:
    x = jnp.asarray(values, jnp.float32)
    return CI(preactivations={"s": x}, lower={"s": x}, upper={"s": x})


def test_cap_is_the_entrywise_minimum_on_every_leaf_and_head():
    ci = DualCI(output=_ci([0.2, 0.9]), hidden=_ci([0.5, 0.1]))
    cap = DualCI(output=_ci([0.5, 0.5]), hidden=_ci([0.3, 0.3]))
    capped = cap_ci(ci, cap)
    for got, want in zip(
        jax.tree.leaves(capped),
        jax.tree.leaves(jax.tree.map(jnp.minimum, ci, cap)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_gradient_passes_under_the_cap_and_over_it_only_when_it_lowers_the_ci():
    x = jnp.asarray([0.2, 0.9, 0.9], jnp.float32)
    cap = jnp.asarray([0.5, 0.5, 0.5], jnp.float32)
    # d loss / d capped = +1, -1, +1: entry 0 is under the cap (passes), entry 1 is over it
    # with a gradient that would RAISE it (blocked), entry 2 over it lowering it (passes).
    weights = jnp.asarray([1.0, -1.0, 1.0], jnp.float32)

    def loss(x: jax.Array, cap: jax.Array) -> jax.Array:
        capped = cap_ci(_ci_of(x), _ci_of(cap)).lower["s"]
        assert isinstance(capped, jax.Array)
        return jnp.sum(weights * capped)

    gx, gcap = jax.grad(loss, argnums=(0, 1))(x, cap)
    np.testing.assert_array_equal(np.asarray(gx), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(gcap), [0.0, 0.0, 0.0])


def _ci_of(x: jax.Array) -> CI:
    return CI(preactivations={"s": x}, lower={"s": x}, upper={"s": x})
