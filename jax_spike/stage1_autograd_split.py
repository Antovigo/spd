"""Stage 1: single-process autograd-split sanity check (JAX).

Mirrors the 2-pool cross-pool autograd seam without any distribution, to answer
the numerical-correctness crux: does shipping masks forward (A->B) and cotangents
back (B->A), then stitching with `jax.vjp`, reproduce a monolithic single-process
gradient?

Minimal faithful model of the 2-pool:

  Pool A  holds the CI-fn params `theta_ci`. Produces per-site masks
          `m = ci_fn(theta_ci, acts)`. Also computes a LOCAL importance-minimality
          loss on the masks (this is A-side in the real system).

  Pool B  holds the decomposed component params `theta_vu`. Receives masks `m` as a
          detached leaf, computes the recon loss (couples m and theta_vu) plus a
          LOCAL faithfulness loss (theta_vu only). Backward yields g_vu (local) and
          g_m = dL_recon/dm (shipped back to A).

  Stitch  A's mask cotangent = g_m (from B's recon) + d(imp_min)/dm (local). One
          vjp through ci_fn flows that combined seed into theta_ci. This is exactly
          the real system's "g_CI_chunk + imp_min fused in one backward through the
          CI-fn graph" (SUM_GRAD_CONVENTION.md / step_pool_a.py).

The reference is the same total loss as one pure function of (theta_ci, theta_vu)
differentiated by a single jax.grad.
"""

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)  # tight tolerance check

D = 16  # feature dim
N_SITES = 3
B = 8  # batch


def init_params(key):
    k1, k2, k3, k4, kx, ka = random.split(key, 6)
    theta_ci = {
        "w": random.normal(k1, (D, D)) * 0.1,
        "b": random.normal(k2, (D,)) * 0.1,
    }
    # one (U, V) per site
    theta_vu = {
        f"site{i}": {
            "U": random.normal(random.fold_in(k3, i), (D, D)) * 0.1,
            "V": random.normal(random.fold_in(k4, i), (D, D)) * 0.1,
        }
        for i in range(N_SITES)
    }
    acts = random.normal(kx, (B, D))  # frozen target activations (input to CI fn)
    x = random.normal(ka, (B, D))  # recon input
    return theta_ci, theta_vu, acts, x


# ---- the two pool-local computations, as pure functions ----


def ci_fn(theta_ci, acts):
    """Pool A: activations -> per-site masks in [0,1] (leaky-hard-sigmoid-ish)."""
    h = jnp.tanh(acts @ theta_ci["w"] + theta_ci["b"])
    masks = {f"site{i}": jax.nn.sigmoid(h * (i + 1.0)) for i in range(N_SITES)}
    return masks


def imp_min_loss(masks):
    """Pool A local: importance-minimality (push masks toward 0)."""
    return sum(jnp.mean(m) for m in masks.values())


def recon_loss(theta_vu, masks, x):
    """Pool B: masked reconstruction coupling masks (from A) and V/U (B)."""
    total = 0.0
    for i in range(N_SITES):
        s = f"site{i}"
        U, V = theta_vu[s]["U"], theta_vu[s]["V"]
        m = masks[s]
        # decomposed matrix W = U @ V, masked per-component by m
        recon = (x * m) @ U @ V
        total = total + jnp.mean((recon - x) ** 2)
    return total


def faith_loss(theta_vu, x):
    """Pool B local: faithfulness (theta_vu only, no masks)."""
    total = 0.0
    for i in range(N_SITES):
        U, V = theta_vu[f"site{i}"]["U"], theta_vu[f"site{i}"]["V"]
        total = total + jnp.mean((x @ U @ V - x) ** 2)
    return total


COEFF_IMP, COEFF_RECON, COEFF_FAITH = 0.7, 1.0, 0.3


# ---- reference: one monolithic pure function ----


def total_loss(theta_ci, theta_vu, acts, x):
    masks = ci_fn(theta_ci, acts)
    return (
        COEFF_IMP * imp_min_loss(masks)
        + COEFF_RECON * recon_loss(theta_vu, masks, x)
        + COEFF_FAITH * faith_loss(theta_vu, x)
    )


def reference_grads(theta_ci, theta_vu, acts, x):
    g = jax.grad(total_loss, argnums=(0, 1))(theta_ci, theta_vu, acts, x)
    return g[0], g[1]


# ---- two-pool split: masks forward, cotangents back, vjp stitch ----


def two_pool_grads(theta_ci, theta_vu, acts, x):
    # --- POOL A: forward CI fn, retain vjp graph (this is the "held graph") ---
    masks, vjp_A = jax.vjp(lambda tc: ci_fn(tc, acts), theta_ci)

    # ship `masks` over the wire to B as a detached leaf -> simulate by treating as
    # constants that B differentiates w.r.t.

    # --- POOL B: recon (couples masks+vu) + local faith. Need g_vu and g_m. ---
    def pool_b_loss(theta_vu_, masks_):
        return COEFF_RECON * recon_loss(theta_vu_, masks_, x) + COEFF_FAITH * faith_loss(
            theta_vu_, x
        )

    (g_vu, g_m_recon) = jax.grad(pool_b_loss, argnums=(0, 1))(theta_vu, masks)
    # g_m_recon == dL_recon/dm  (faith doesn't touch masks). Ship g_m_recon back to A.

    # --- POOL A: local imp_min cotangent on masks, summed with g_m from B ---
    g_m_imp = jax.grad(lambda m: COEFF_IMP * imp_min_loss(m))(masks)
    # combined mask seed = recon (from B) + imp_min (local A) -- the fused seed
    combined_mask_cotangent = jax.tree.map(lambda a, b: a + b, g_m_recon, g_m_imp)

    # one vjp through the held CI-fn graph -> g_ci
    (g_ci,) = vjp_A(combined_mask_cotangent)

    return g_ci, g_vu


def max_rel_err(a, b):
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    worst = 0.0
    for la, lb in zip(leaves_a, leaves_b, strict=False):
        denom = jnp.maximum(jnp.abs(la), jnp.abs(lb)) + 1e-12
        worst = max(worst, float(jnp.max(jnp.abs(la - lb) / denom)))
    return worst


def main():
    key = random.PRNGKey(0)
    theta_ci, theta_vu, acts, x = init_params(key)

    ref_ci, ref_vu = reference_grads(theta_ci, theta_vu, acts, x)
    tp_ci, tp_vu = two_pool_grads(theta_ci, theta_vu, acts, x)

    err_ci = max_rel_err(ref_ci, tp_ci)
    err_vu = max_rel_err(ref_vu, tp_vu)
    print(f"g_ci  worst rel err: {err_ci:.3e}")
    print(f"g_vu  worst rel err: {err_vu:.3e}")
    ok = err_ci < 1e-10 and err_vu < 1e-10
    print("STAGE 1:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
