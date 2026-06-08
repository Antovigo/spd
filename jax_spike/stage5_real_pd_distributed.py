"""Stage 5: the synthesis -- REAL PD algorithm through the cross-pool seam.

Stages 1-4 used toy couplings to test the distributed plumbing. The previous spike
(feature/nano-pd-jax) ported the REAL PD core single-device (Equinox+Optax). This
stage glues them: the genuine PD math -- V/U decomposition, the custom_vjp
lower-leaky-hard-sigmoid, stochastic masks, faith+imp+stoch losses -- run SPLIT
across two pools via ppermute, grad-checked against the single-process reference.

Reused verbatim from feature/nano-pd-jax (nano_pd_jax/{ci_sigmoids,masks,losses}.py):
  * lower_leaky_hard_sigmoid (custom_vjp, asymmetric leak)
  * sample_masks: m = ci + (1-ci)*u
  * faith / imp / stoch loss forms

Pool split mirrors the 2-pool:
  Pool A: CI fn (acts -> ci -> mask), local importance-minimality on ci.
  Pool B: decomposed recon ((x@V)*m)@U + x@W_delta vs target, local faithfulness.
  Transport: masks A->B via ppermute; cotangents return automatically.

The novel risk retired here: does a custom_vjp (the leaky sigmoid's hand-written
backward) compose correctly with ppermute's transpose across the mesh? And does the
stochastic-mask PRNG stay consistent so the split matches the reference bit-for-bit?
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)
assert jax.device_count() >= 2
mesh = Mesh(np.array(jax.devices()[:2]), axis_names=("pool",))

D_IN, D_OUT, C, B = 6, 6, 4, 8
N_SITES = 2
COEFF_FAITH, COEFF_IMP, COEFF_STOCH, P_IMP = 1.0, 0.3, 1.0, 0.9


# ---- stolen verbatim from nano_pd_jax/ci_sigmoids.py ----
@jax.custom_vjp
def lower_leaky_hard_sigmoid(x):
    return jnp.clip(x, 0.0, 1.0)


def _fwd(x):
    return jnp.clip(x, 0.0, 1.0), x


def _bwd(x, g):
    alpha = 0.01
    leak = jnp.where(g < 0, alpha * g, 0.0)
    grad = jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0))
    return (grad,)


lower_leaky_hard_sigmoid.defvjp(_fwd, _bwd)


def init(key):
    ks = random.split(key, 9)
    ci_params = {  # pool A: per-site CI MLP (1 layer: acts -> C logits)
        f"s{i}": {
            "w": random.normal(ks[i], (D_IN, C)) * 0.3,
            "b": random.normal(ks[i + 2], (C,)) * 0.1,
        }
        for i in range(N_SITES)
    }
    vu_params = {  # pool B: V/U + frozen target weight per site
        f"s{i}": {
            "V": random.normal(ks[i + 4], (D_IN, C)) * 0.2,
            "U": random.normal(ks[i + 6], (C, D_OUT)) * 0.2,
            "W_target": random.normal(random.fold_in(ks[8], i), (D_IN, D_OUT)) * 0.2,
        }
        for i in range(N_SITES)
    }
    x = random.normal(ks[8], (B, D_IN))
    return ci_params, vu_params, x


# ---- pool-local pure functions (real PD math) ----
def ci_forward(ci_params, x):
    """Pool A: activations -> ci in [0,1] via the leaky-hard sigmoid."""
    return {s: lower_leaky_hard_sigmoid(x @ p["w"] + p["b"]) for s, p in ci_params.items()}


def sample_masks(key, ci):
    names = sorted(ci)
    keys = random.split(key, len(names))
    return {
        n: ci[n] + (1.0 - ci[n]) * random.uniform(k, ci[n].shape)
        for k, n in zip(keys, names, strict=False)
    }


def recon_and_faith(vu_params, masks, x):
    """Pool B: decomposed forward + faithfulness."""
    stoch = 0.0
    faith_sq, faith_n = 0.0, 0
    for s, p in vu_params.items():
        W_delta = p["W_target"] - p["V"] @ p["U"]
        y_dec = ((x @ p["V"]) * masks[s]) @ p["U"] + x @ W_delta
        y_tgt = x @ p["W_target"]
        stoch = stoch + jnp.mean((y_dec - y_tgt) ** 2)
        resid = p["W_target"] - p["V"] @ p["U"]
        faith_sq = faith_sq + (resid**2).sum()
        faith_n += resid.size
    return COEFF_STOCH * (stoch / N_SITES) + COEFF_FAITH * (faith_sq / faith_n)


def imp_min(ci):
    per = [jnp.mean(jnp.clip(v, 0.0, 1.0) ** P_IMP) for v in ci.values()]
    return COEFF_IMP * jnp.mean(jnp.stack(per))


# ---- reference: monolithic single-process ----
def total_loss(ci_params, vu_params, x, key):
    ci = ci_forward(ci_params, x)
    masks = sample_masks(key, ci)
    return recon_and_faith(vu_params, masks, x) + imp_min(ci)


def reference_grads(ci_params, vu_params, x, key):
    return jax.grad(total_loss, argnums=(0, 1))(ci_params, vu_params, x, key)


# ---- two-pool: ci on A, recon on B, masks via ppermute, mask RNG fixed & shared ----
def two_pool_grads(ci_params, vu_params, x, key):
    def step(theta, x_pair):
        ci_p, vu_p = theta
        xb = x_pair[0]
        axis = jax.lax.axis_index("pool")

        # POOL A: CI forward (only meaningful on axis 0, but traced on both).
        ci = ci_forward(ci_p, xb)
        masks = sample_masks(key, ci)  # same key on both -> identical noise

        # ship masks A->B; cotangents auto-return via ppermute transpose.
        masks_b = {s: jax.lax.ppermute(m, "pool", perm=[(0, 1)]) for s, m in masks.items()}

        loss_b = recon_and_faith(vu_p, masks_b, xb)  # pool B term
        loss_a = imp_min(ci)  # pool A local term
        loss = jnp.where(axis == 1, loss_b, loss_a)
        return loss[None]

    x_pair = jnp.broadcast_to(x, (2, *x.shape))

    def total(theta):
        sm = shard_map(
            lambda th, xp: step(th, xp),
            mesh=mesh,
            in_specs=(P(), P("pool")),
            out_specs=P("pool"),
            check_rep=False,
        )
        return jnp.sum(sm(theta, x_pair))

    return jax.grad(total)((ci_params, vu_params))


def max_rel(a, b):
    worst = 0.0
    for la, lb in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=False):
        worst = max(
            worst,
            float(jnp.max(jnp.abs(la - lb) / (jnp.maximum(jnp.abs(la), jnp.abs(lb)) + 1e-12))),
        )
    return worst


def main():
    ci_params, vu_params, x = init(random.PRNGKey(0))
    mask_key = random.PRNGKey(123)

    ref_ci, ref_vu = reference_grads(ci_params, vu_params, x, mask_key)
    tp_ci, tp_vu = two_pool_grads(ci_params, vu_params, x, mask_key)

    e_ci = max_rel(ref_ci, tp_ci)
    e_vu = max_rel(ref_vu, tp_vu)
    print(f"CI-fn params (pool A) grad rel err: {e_ci:.3e}")
    print(f"V/U params  (pool B) grad rel err: {e_vu:.3e}")
    print("  (real PD: custom_vjp leaky sigmoid + stochastic mask + V/U decomp + faith/imp/stoch)")
    ok = e_ci < 1e-10 and e_vu < 1e-10
    print("STAGE 5:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
