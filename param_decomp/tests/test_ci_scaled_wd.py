"""CI-scaled component weight decay (SPEC S40): keep-factor math + V/U application."""

import jax.numpy as jnp
import numpy as np

from param_decomp.components import DecompVU
from param_decomp.train import apply_ci_scaled_weight_decay, ci_scaled_weight_decay_keep


def test_keep_factor_matches_torch_formula():
    ci_max = {"site": jnp.array([0.0, 0.5, 1.0, 1.7, -0.3])}  # incl. out-of-[0,1] leaks
    keep = ci_scaled_weight_decay_keep(
        ci_max, lr=jnp.float32(0.001), coeff=0.2, active=jnp.array(True)
    )["site"]
    # keep = 1 - lr*coeff*(1 - clamp(ci, 0, 1))
    np.testing.assert_allclose(
        np.asarray(keep), [1 - 2e-4, 1 - 1e-4, 1.0, 1.0, 1 - 2e-4], rtol=1e-6
    )


def test_inactive_is_exact_identity():
    ci_max = {"site": jnp.array([0.0, 0.5])}
    keep = ci_scaled_weight_decay_keep(
        ci_max, lr=jnp.float32(0.001), coeff=0.2, active=jnp.array(False)
    )["site"]
    np.testing.assert_array_equal(np.asarray(keep), [1.0, 1.0])


def test_apply_scales_v_columns_and_u_rows():
    v = jnp.ones((3, 2))  # (d_in, C)
    u = jnp.ones((2, 4))  # (C, d_out)
    components = DecompVU(vu={"site": (v, u)})
    out = apply_ci_scaled_weight_decay(components, {"site": jnp.array([0.5, 1.0])})
    v_out, u_out = out.vu["site"]
    np.testing.assert_allclose(np.asarray(v_out), np.stack([np.full(3, 0.5), np.ones(3)], axis=1))
    np.testing.assert_allclose(np.asarray(u_out), np.stack([np.full(4, 0.5), np.ones(4)], axis=0))
