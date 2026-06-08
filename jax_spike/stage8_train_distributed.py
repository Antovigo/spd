"""Stage 8: the whole PD+PGD training stack, GSPMD-sharded across a multi-node mesh.

Single-pool SPMD design (the recommended JAX target): every rank runs the full step on
its batch shard -- CI fn, decomposed recon, the PGD adversary -- and GSPMD all-reduces
the gradients. No manual pool split, no manual collectives: data is sharded P('dp'),
params are replicated, and `jax.jit` inserts the grad all-reduce automatically because
the mean-loss reduces over the sharded batch.

Integrates everything the spike built:
  * real PD math (custom_vjp leaky sigmoid, stochastic mask, V/U decomposition)
  * the four losses: faithfulness, importance-minimality, stochastic recon, PGD recon
  * the PGD adversary (persistent sources in state, lax.scan inner loop)
  * two optimizers (hand-rolled Adam x2: components V/U, and the CI fn)

Correctness signal: pure data-parallelism is mathematically independent of GPU count, so
for a FIXED global batch + seed the loss trajectory must match at 1 / 8 / 16 GPUs. Run at
several scales and compare the printed trajectory. Also: total loss trends down, faith ->
small. Throughput (steps/s, tokens/s) is reported on process 0.

Usage (via remote/gpu.sh):
  NODES=1 GPN=1 ... python stage8_train_distributed.py --steps 40 --global_batch 256
  NODES=1 GPN=8 ... (same args) -> trajectory must match
  NODES=2 GPN=8 ... (same args) -> trajectory must match
"""

import argparse
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
from distributed_util import dp_mesh, init_distributed, replicate, shard_dp
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


# ---- CI sigmoid (from feature/nano-pd-jax) ----
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
    V: jax.Array
    U: jax.Array
    W_target: jax.Array
    ci_w: jax.Array
    ci_b: jax.Array


class AdamState(NamedTuple):
    m: object
    v: object


class TrainState(NamedTuple):
    params: Params
    opt_vu: AdamState
    opt_ci: AdamState
    sources: jax.Array  # (S, B_global, C) -- sharded on batch
    step: jax.Array


COEFF = dict(faith=1.0, imp=0.3, stoch=1.0, ppgd=1.0)
P_IMP = 0.9


def init_params(key, S, d, C):
    k = random.split(key, 5)
    sc = 0.2
    return Params(
        V=random.normal(k[0], (S, d, C)) * sc,
        U=random.normal(k[1], (S, C, d)) * sc,
        W_target=random.normal(k[2], (S, d, d)) * sc,
        ci_w=random.normal(k[3], (S, d, C)) * sc,
        ci_b=random.normal(k[4], (S, C)) * 0.1,
    )


def ci_envelope(params, x):
    logits = jnp.einsum("bi,sic->sbc", x, params.ci_w) + params.ci_b[:, None, :]
    return lower_leaky_hard_sigmoid(logits)


def recon(params, masks, x):
    xV = jnp.einsum("bi,sic->sbc", x, params.V)
    y_masked = jnp.einsum("sbc,sco->sbo", xV * masks, params.U)
    W_delta = params.W_target - jnp.einsum("sic,sco->sio", params.V, params.U)
    y_dec = y_masked + jnp.einsum("bi,sio->sbo", x, W_delta)
    y_tgt = jnp.einsum("bi,sio->sbo", x, params.W_target)
    return jnp.mean((y_dec - y_tgt) ** 2)


def faith(params):
    resid = params.W_target - jnp.einsum("sic,sco->sio", params.V, params.U)
    return jnp.mean(resid**2)


def imp_min(ci):
    return jnp.mean(jnp.clip(ci, 0.0, 1.0) ** P_IMP)


def sample_masks(key, ci):
    return ci + (1.0 - ci) * random.uniform(key, ci.shape)


def pgd_refine(params, ci, sources, x, n_warmup, lr_pgd):
    p_det = jax.lax.stop_gradient(params)
    ci_det = jax.lax.stop_gradient(ci)

    def adv_recon(src):
        return recon(p_det, ci_det * jax.nn.sigmoid(src), x)

    def body(src, _):
        return src + lr_pgd * jax.grad(adv_recon)(src), None

    final, _ = jax.lax.scan(body, sources, None, length=n_warmup)
    return final


def adam_update(params, grads, st, lr, step, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, st.m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, st.v, grads)
    bc1 = 1 - b1 ** (step + 1)
    bc2 = 1 - b2 ** (step + 1)
    new = jax.tree.map(
        lambda p, m_, v_: p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps), params, m, v
    )
    return new, AdamState(m, v)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def train_step(state, x, key, n_warmup, lr_pgd, lr_vu, lr_ci):
    sources = state.sources

    def total_loss(params):
        ci = ci_envelope(params, x)
        masks_clean = sample_masks(key, ci)
        refined = pgd_refine(params, ci, sources, x, n_warmup, lr_pgd)
        l_faith = faith(params)
        l_imp = imp_min(ci)
        l_stoch = recon(params, masks_clean, x)
        l_ppgd = recon(params, ci * jax.nn.sigmoid(refined), x)
        tot = (
            COEFF["faith"] * l_faith
            + COEFF["imp"] * l_imp
            + COEFF["stoch"] * l_stoch
            + COEFF["ppgd"] * l_ppgd
        )
        return tot, (l_faith, l_imp, l_stoch, l_ppgd, refined)

    (tot, (lf, li, ls, lp, refined)), grads = jax.value_and_grad(total_loss, has_aux=True)(
        state.params
    )

    # split grads: V/U + W_target(zero, frozen) go to opt_vu; ci_w/ci_b to opt_ci.
    g = grads
    vu_params = state.params._replace(
        ci_w=jnp.zeros_like(state.params.ci_w), ci_b=jnp.zeros_like(state.params.ci_b)
    )
    g_vu = Params(
        V=g.V,
        U=g.U,
        W_target=jnp.zeros_like(g.W_target),
        ci_w=jnp.zeros_like(g.ci_w),
        ci_b=jnp.zeros_like(g.ci_b),
    )
    new_vu, opt_vu = adam_update(vu_params, g_vu, state.opt_vu, lr_vu, state.step)

    ci_params = state.params._replace(
        V=jnp.zeros_like(state.params.V),
        U=jnp.zeros_like(state.params.U),
        W_target=jnp.zeros_like(state.params.W_target),
    )
    g_ci = Params(
        V=jnp.zeros_like(g.V),
        U=jnp.zeros_like(g.U),
        W_target=jnp.zeros_like(g.W_target),
        ci_w=g.ci_w,
        ci_b=g.ci_b,
    )
    new_ci, opt_ci = adam_update(ci_params, g_ci, state.opt_ci, lr_ci, state.step)

    new_params = Params(
        V=new_vu.V, U=new_vu.U, W_target=state.params.W_target, ci_w=new_ci.ci_w, ci_b=new_ci.ci_b
    )
    # one more adversary ascent step, persist (warm-start)
    new_sources = refined + lr_pgd * jax.grad(
        lambda s: recon(
            jax.lax.stop_gradient(new_params),
            jax.lax.stop_gradient(ci_envelope(new_params, x)) * jax.nn.sigmoid(s),
            x,
        )
    )(refined)

    new_state = TrainState(new_params, opt_vu, opt_ci, new_sources, state.step + 1)
    return new_state, (tot, lf, li, ls, lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--global_batch", type=int, default=256)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--n_warmup", type=int, default=10)
    args = ap.parse_args()

    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    if is0:
        print(
            f"[p0] mesh: {ndev} devices | global_batch={args.global_batch} "
            f"S={args.S} d={args.d} C={args.C} n_warmup={args.n_warmup}"
        )

    # deterministic params (replicated) + global batch (sharded), identical seed everywhere
    key = random.PRNGKey(0)
    params = init_params(key, args.S, args.d, args.C)
    params = jax.tree.map(lambda a: replicate(a, mesh), params)
    opt0 = lambda pt: AdamState(jax.tree.map(jnp.zeros_like, pt), jax.tree.map(jnp.zeros_like, pt))

    x_full = random.normal(random.PRNGKey(42), (args.global_batch, args.d))
    x = shard_dp(x_full, mesh)
    sources_full = jnp.zeros((args.S, args.global_batch, args.C))
    # shard sources on the batch (axis 1)
    src_sharding = NamedSharding(mesh, P(None, "dp", None))
    per = args.global_batch // ndev
    idx = jax.process_index()
    src_local = sources_full[:, idx * per : (idx + 1) * per, :]
    sources = jax.make_array_from_single_device_arrays(
        sources_full.shape,
        src_sharding,
        [jax.device_put(src_local, d) for d in src_sharding.addressable_devices],
    )

    state = TrainState(params, opt0(params), opt0(params), sources, jnp.array(0))

    losses = []
    t0 = None
    for s in range(args.steps):
        state, (tot, lf, li, ls, lp) = train_step(
            state, x, random.PRNGKey(1000 + s), args.n_warmup, 0.1, 0.01, 0.01
        )
        if s == 0:
            jax.block_until_ready(tot)  # compiled
            t0 = time.time()
        losses.append(float(tot))
        if is0 and (s < 5 or s % 10 == 0):
            print(
                f"[p0] step {s:3d} | total {float(tot):.5f} | faith {float(lf):.3e} "
                f"imp {float(li):.4f} stoch {float(ls):.3e} ppgd {float(lp):.3e}"
            )

    jax.block_until_ready(state.sources)
    dt = (time.time() - t0) / (args.steps - 1)
    if is0:
        toks = args.global_batch
        print(f"[p0] TRAJECTORY[:6] = {[round(x, 5) for x in losses[:6]]}")
        print(f"[p0] final total {losses[-1]:.5f} (start {losses[0]:.5f}); faith {float(lf):.3e}")
        print(f"[p0] {dt * 1e3:.2f} ms/step | {toks / dt:,.0f} samples/s across {ndev} GPU(s)")
        print(f"[p0] STAGE 8 ({ndev} GPU): {'PASS' if losses[-1] < losses[0] else 'FAIL'}")

    jax.experimental.multihost_utils.sync_global_devices("stage8_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
