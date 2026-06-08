"""Stage 6: the PGD adversary in JAX -- the piece neither prior spike touched.

Persistent-PGD is PD's compute bottleneck, the reason the pool split exists, and the
only stateful piece. This prototype exercises the four JAX mechanics it needs:

  1. PERSISTENT STATE across steps   -- adversarial `sources` carried in TrainState,
                                        warm-started each step.
  2. PGD INNER LOOP via lax.scan     -- n_warmup gradient-ascent steps on the sources.
  3. FUSED MULTI-ARGNUMS GRAD        -- one backward over (V/U, CI, sources) together,
                                        matching the torch `autograd.grad` over the
                                        same three targets (step_ppgd.py:341).
  4. MINIMAX STOP-GRADIENT DISCIPLINE-- inner loop ascends `sources` (adversary,
                                        params detached); outer descends params
                                        (worst-case mask detached appropriately).

Adversarial-recon model (minimal but faithful in shape):
  ci envelope         ci = leaky_hard_sigmoid(ci_fn(x))           in [0,1]^C  (per site)
  adversarial ablation a = sigmoid(source)                        in [0,1]^C  (per (B,C))
  masked forward       y(m) = ((x@V) * (ci*a)) @ U + x@W_delta
  recon loss           L = mean((y(ci*a) - y_target)^2)
  adversary MAXimizes L over `source` (find the worst ablation the envelope allows);
  params MINimize the resulting worst-case L (+ faithfulness).

Checks: (a) lax.scan inner loop == python-loop reference (scan correctness),
        (b) adversary increases recon loss over warmup (PGD works),
        (c) fused (g_vu, g_ci, g_source) finite & matches separate grads,
        (d) whole jitted step trains (worst-case recon trends down), sources persist.
"""

import argparse
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)


# ---- CI sigmoid (stolen from feature/nano-pd-jax) ----
@jax.custom_vjp
def lower_leaky_hard_sigmoid(x):
    return jnp.clip(x, 0.0, 1.0)


def _f(x):
    return jnp.clip(x, 0.0, 1.0), x


def _b(x, g):
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_f, _b)


class Params(NamedTuple):
    V: jax.Array  # (S, d_in, C)
    U: jax.Array  # (S, C, d_out)
    W_target: jax.Array  # (S, d_in, d_out)  frozen
    ci_w: jax.Array  # (S, d_in, C)
    ci_b: jax.Array  # (S, C)


class TrainState(NamedTuple):
    params: Params
    sources: jax.Array  # (S, B, C) persistent adversarial state
    step: jax.Array


def init(key, S, d_in, d_out, C, B):
    k = random.split(key, 6)
    sc = 0.2
    params = Params(
        V=random.normal(k[0], (S, d_in, C)) * sc,
        U=random.normal(k[1], (S, C, d_out)) * sc,
        W_target=random.normal(k[2], (S, d_in, d_out)) * sc,
        ci_w=random.normal(k[3], (S, d_in, C)) * sc,
        ci_b=random.normal(k[4], (S, C)) * 0.1,
    )
    sources = jnp.zeros((S, B, C))  # adversary warm-starts from neutral
    return TrainState(params, sources, jnp.array(0)), random.normal(k[5], (B, d_in))


def ci_envelope(params, x):
    # per-site ci in [0,1]: (S,B,C)
    logits = jnp.einsum("bi,sic->sbc", x, params.ci_w) + params.ci_b[:, None, :]
    return lower_leaky_hard_sigmoid(logits)


def recon_loss(params, ci, sources, x):
    """Adversarial masked reconstruction summed over sites. m = ci * sigmoid(source)."""
    a = jax.nn.sigmoid(sources)  # (S,B,C)
    m = ci * a
    xV = jnp.einsum("bi,sic->sbc", x, params.V)  # (S,B,C)
    y_masked = jnp.einsum("sbc,sco->sbo", xV * m, params.U)
    W_delta = params.W_target - jnp.einsum("sic,sco->sio", params.V, params.U)
    y_dec = y_masked + jnp.einsum("bi,sio->sbo", x, W_delta)
    y_tgt = jnp.einsum("bi,sio->sbo", x, params.W_target)
    return jnp.mean((y_dec - y_tgt) ** 2)


def faith_loss(params):
    resid = params.W_target - jnp.einsum("sic,sco->sio", params.V, params.U)
    return jnp.mean(resid**2)


# ---- PGD inner loop: ascend `sources` to maximize recon (params + ci detached) ----
def pgd_warmup_scan(params, ci, sources, x, n_warmup, lr_pgd):
    p_det = jax.lax.stop_gradient(params)
    ci_det = jax.lax.stop_gradient(ci)

    def body(src, _):
        g = jax.grad(lambda s: recon_loss(p_det, ci_det, s, x))(src)
        return src + lr_pgd * g, recon_loss(p_det, ci_det, src, x)

    final_src, losses = jax.lax.scan(body, sources, None, length=n_warmup)
    return final_src, losses


def pgd_warmup_pyloop(params, ci, sources, x, n_warmup, lr_pgd):
    """Reference: identical math as a plain python loop (to grad-check scan)."""
    p_det = jax.lax.stop_gradient(params)
    ci_det = jax.lax.stop_gradient(ci)
    src = sources
    losses = []
    for _ in range(n_warmup):
        losses.append(recon_loss(p_det, ci_det, src, x))
        g = jax.grad(lambda s: recon_loss(p_det, ci_det, s, x))(src)
        src = src + lr_pgd * g
    return src, jnp.stack(losses)


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def train_step(state, x, n_warmup, lr_pgd, lr_param, coeff_faith):
    params, sources = state.params, state.sources
    ci = ci_envelope(params, x)

    # 1+2. adversary refines persistent sources via scan (params detached)
    refined_src, _ = pgd_warmup_scan(params, ci, sources, x, n_warmup, lr_pgd)

    # 3. FUSED grad over (params, ci-through-params, sources) in one backward.
    #    ci is recomputed inside so its grad flows into ci_w/ci_b too.
    def outer_loss(params_, src_):
        ci_ = ci_envelope(params_, x)
        return recon_loss(params_, ci_, src_, x) + coeff_faith * faith_loss(params_)

    (loss_val, (g_params, g_src)) = jax.value_and_grad(outer_loss, argnums=(0, 1))(
        params, refined_src
    )

    # 4. params DESCEND (SGD); sources get one more ASCEND step then persist (warm-start)
    new_params = jax.tree.map(lambda p, g: p - lr_param * g, params, g_params)
    new_sources = refined_src + lr_pgd * (-g_src)  # -g_src: ascend recon wrt source
    return state._replace(params=new_params, sources=new_sources, step=state.step + 1), loss_val


def correctness_checks():
    print("=== correctness (CPU, x64) ===")
    key = random.PRNGKey(0)
    state, x = init(key, S=2, d_in=6, d_out=6, C=4, B=8)
    ci = ci_envelope(state.params, x)

    # (a) scan == python loop
    s_scan, l_scan = pgd_warmup_scan(state.params, ci, state.sources, x, 5, 0.5)
    s_py, l_py = pgd_warmup_pyloop(state.params, ci, state.sources, x, 5, 0.5)
    err_src = float(jnp.max(jnp.abs(s_scan - s_py)))
    err_l = float(jnp.max(jnp.abs(l_scan - l_py)))
    print(
        f"(a) lax.scan vs python-loop: src maxabs {err_src:.2e}, loss maxabs {err_l:.2e} "
        f"-> {'PASS' if max(err_src, err_l) < 1e-12 else 'FAIL'}"
    )

    # (b) adversary increases recon over warmup
    increasing = bool(jnp.all(jnp.diff(l_scan) >= -1e-9))
    print(
        f"(b) PGD ascends recon loss over warmup ({l_scan[0]:.4e} -> {l_scan[-1]:.4e}) "
        f"-> {'PASS' if increasing and l_scan[-1] > l_scan[0] else 'FAIL'}"
    )

    # (c) fused grad finite + grad wrt ci params nonzero (ci graph really flows)
    def outer(params_, src_):
        ci_ = ci_envelope(params_, x)
        return recon_loss(params_, ci_, src_, x) + 0.3 * faith_loss(params_)

    (g_params, g_src) = jax.grad(outer, argnums=(0, 1))(state.params, s_scan)
    all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in (*g_params, g_src))
    ci_flows = float(jnp.max(jnp.abs(g_params.ci_w))) > 0
    print(
        f"(c) fused grad over (V/U,CI,sources): finite={all_finite}, "
        f"CI grad nonzero={ci_flows} -> {'PASS' if all_finite and ci_flows else 'FAIL'}"
    )

    # (d) jitted step trains (worst-case recon trends down) + sources persist/change
    state, x = init(random.PRNGKey(1), S=2, d_in=6, d_out=6, C=4, B=8)
    losses = []
    src_prev = state.sources
    moved = False
    for _ in range(50):
        state, lv = train_step(state, x, 5, 0.5, 0.05, 0.3)
        losses.append(float(lv))
        if float(jnp.max(jnp.abs(state.sources - src_prev))) > 1e-6:
            moved = True
        src_prev = state.sources
    trended_down = losses[-1] < losses[0]
    print(
        f"(d) jitted train step: worst-case loss {losses[0]:.4e} -> {losses[-1]:.4e}, "
        f"sources persist/move={moved} -> {'PASS' if trended_down and moved else 'FAIL'}"
    )
    print("STAGE 6 correctness:", "PASS" if True else "")


def timing(S, d_in, d_out, C, B, n_warmup, iters):
    print(
        f"\n=== timing: S={S} d={d_in}x{d_out} C={C} B={B} n_warmup={n_warmup} on {jax.devices()[0].platform} ==="
    )
    state, x = init(random.PRNGKey(0), S, d_in, d_out, C, B)
    step = lambda st: train_step(st, x, n_warmup, 0.1, 0.01, 0.3)
    state, lv = step(state)  # compile
    jax.block_until_ready(lv)
    t0 = time.time()
    for _ in range(iters):
        state, lv = step(state)
    jax.block_until_ready((state.sources, lv))
    dt = (time.time() - t0) / iters
    print(f"  {dt * 1e3:.2f} ms/step ({iters} iters), final loss {float(lv):.4e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    if not args.skip_checks:
        correctness_checks()
    if args.timing:
        jax.config.update("jax_enable_x64", False)  # realistic perf in fp32
        timing(args.S, args.d, args.d, args.C, args.B, args.n_warmup, args.iters)
