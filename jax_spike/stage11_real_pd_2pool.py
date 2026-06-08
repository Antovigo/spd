"""Stage 11: the REAL L18-MLP PD workload SPLIT across two pools — JAX throughput.

The 2-pool of the torch impl, faithfully in JAX via the stage-5 mechanism (validated
bit-exact there): a `shard_map` over a size-2 `pool` mesh axis, with the CI masks shipped
pool A -> pool B by a differentiable `jax.lax.ppermute` (cotangents return automatically on
the transpose — this is the whole point: it deletes the torch 2-pool's manual cross-pool
g_CI plumbing + deadlock-prone P2P ordering).

  Pool A (adversary + CI): CI fn forward (per-site input acts -> ci), importance-minimality,
    and the persistent-PGD adversary recon (a masked suffix forward).
  Pool B (component recon): faithfulness + layerwise stochastic recon + PPGD recon, all
    masked suffix forwards using the components V/U and the ci shipped from A.

Both pools process the SAME batch (work split, not data split) — so per-GPU throughput is
global_batch*seq / step / 2. At small scale the 2-pool is EXPECTED to trail single-pool
tok/s/GPU (work split across 2x GPUs for one batch, plus mask transport); its real win is
memory at scale (sharding V/U so the workload fits where replicated 1-pool OOMs).

Reuses every model component from stage10 (suffix, decomposed MLP, CI transformer).
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
import stage10_real_pd_bench as s
from distributed_util import init_distributed
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

SITES = s.SITES
DT = s.DT
COEFF = s.COEFF
P_IMP = s.P_IMP


class State(NamedTuple):
    vu: s.DecompVU
    ci_fn: s.CIFn
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    source: dict
    step: jax.Array


def make_step(mesh, opt_vu, opt_ci, lr_pgd, n_pgd):
    @jax.jit
    def step(state: State, frozen, resid, source, key):
        dm = {site: jnp.ones((1, 1, 1), DT) for site in SITES}
        nomask = {site: None for site in SITES}
        Wg, Wu, Wd = frozen.l18_Wg, frozen.l18_Wu, frozen.l18_Wd

        ckpt_suffix = jax.checkpoint(s.suffix_logits)

        def per_pool(trainable, resid_p, src_p):
            vu, ci_fn = trainable
            # shard_map keeps the mapped axis (size 1) rather than squeezing it (unlike vmap).
            resid_p = resid_p[0]
            src_p = {site: v[0] for site, v in src_p.items()}
            axis = jax.lax.axis_index("pool")
            clean = jax.lax.stop_gradient(s.suffix_logits(frozen, vu, resid_p, nomask, dm))
            mlp_in = s.l18_resid_to_mlp_input(frozen, resid_p)
            ci = ci_fn(s.mlp_site_inputs(Wg, Wu, mlp_in))

            # ship ci A->B (unconditional so both pools enter the collective); cotangents return
            # automatically through the ppermute transpose.
            ci_b = {site: jax.lax.ppermute(v, "pool", perm=[(0, 1)]) for site, v in ci.items()}

            def pool_a(_):
                # importance-minimality + persistent-PGD adversary recon.
                l_imp = jnp.mean(
                    jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()])
                )
                ppgd_a = {site: ci[site] * jax.nn.sigmoid(src_p[site]) for site in SITES}
                l_adv = s.recon(ckpt_suffix(frozen, vu, resid_p, ppgd_a, dm), clean)
                return COEFF["imp"] * l_imp + COEFF["ppgd"] * l_adv

            def pool_b(_):
                # faithfulness + layerwise stochastic recon + PPGD recon (uses ci_b).
                wd = s.weight_deltas(vu, Wg, Wu, Wd)
                l_faith = sum((d**2).sum() for d in wd.values()) / sum(d.size for d in wd.values())
                l_stoch = jnp.array(0.0)
                for i, site in enumerate(SITES):
                    u = random.uniform(random.fold_in(key, i), ci_b[site].shape, dtype=DT)
                    m = {**nomask, site: ci_b[site] + (1 - ci_b[site]) * u}
                    l_stoch = l_stoch + s.recon(ckpt_suffix(frozen, vu, resid_p, m, dm), clean)
                l_stoch = l_stoch / len(SITES)
                ppgd_b = {site: ci_b[site] * jax.nn.sigmoid(src_p[site]) for site in SITES}
                l_ppgd = s.recon(ckpt_suffix(frozen, vu, resid_p, ppgd_b, dm), clean)
                return COEFF["faith"] * l_faith + COEFF["stoch"] * l_stoch + COEFF["ppgd"] * l_ppgd

            # lax.cond so each pool runs ONLY its branch at runtime (jnp.where would compute both
            # on every device -> each GPU does the full 2-pool work -> OOM).
            return jax.lax.cond(axis == 0, pool_a, pool_b, operand=None)[None]

        def total(trainable):
            sm = shard_map(
                per_pool,
                mesh=mesh,
                in_specs=(P(), P("pool"), P("pool")),
                out_specs=P("pool"),
                check_rep=False,
            )
            return jnp.sum(sm(trainable, resid, source))

        loss, grads = jax.value_and_grad(total)((state.vu, state.ci_fn))
        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.vu)
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.ci_fn)
        new_vu = eqx.apply_updates(state.vu, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        # persistent-PGD source update on pool A's data (single shard, same on both here).
        vu_det = jax.lax.stop_gradient(state.vu)
        resid0 = resid[0]
        clean0 = jax.lax.stop_gradient(s.suffix_logits(frozen, vu_det, resid0, nomask, dm))
        ci0 = jax.lax.stop_gradient(
            state.ci_fn(s.mlp_site_inputs(Wg, Wu, s.l18_resid_to_mlp_input(frozen, resid0)))
        )

        def adv(src1):
            masks = {site: ci0[site] * jax.nn.sigmoid(src1[site]) for site in SITES}
            return s.recon(s.suffix_logits(frozen, vu_det, resid0, masks, dm), clean0)

        def body(src1, _):
            g = jax.grad(adv)(src1)
            return jax.tree.map(lambda a, b: a + lr_pgd * b, src1, g), None

        src0 = {site: source[site][0] for site in SITES}
        new_src0, _ = jax.lax.scan(body, src0, None, length=n_pgd)
        new_src0 = jax.lax.stop_gradient(new_src0)
        new_source = {
            site: jnp.broadcast_to(new_src0[site][None], source[site].shape) for site in SITES
        }
        return State(new_vu, new_ci, os_vu, os_ci, new_source, state.step + 1), loss

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--per_gpu_batch", type=int, default=4)
    ap.add_argument("--C", type=int, default=24576)
    ap.add_argument("--n_warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n_embd", type=int, default=4096)
    ap.add_argument("--n_intermediate", type=int, default=14336)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--suffix_layers", type=int, default=14)
    ap.add_argument("--n_head", type=int, default=32)
    ap.add_argument("--n_kv_head", type=int, default=8)
    ap.add_argument("--ci_d_model", type=int, default=4096)
    ap.add_argument("--ci_blocks", type=int, default=4)
    ap.add_argument("--ci_heads", type=int, default=64)
    ap.add_argument("--ci_mlp", type=int, default=16384)
    args = ap.parse_args()

    init_distributed()
    devices = jax.devices()
    assert len(devices) >= 2, "2-pool needs >= 2 devices"
    mesh = Mesh(devices[:2], axis_names=("pool",))
    is0 = jax.process_index() == 0

    cfg = s.LlamaConfig(
        vocab_size=args.vocab,
        n_layer=args.suffix_layers,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        n_intermediate=args.n_intermediate,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )
    total_in = cfg.n_embd + cfg.n_embd + cfg.n_intermediate
    if is0:
        print(
            f"[p0] STAGE11 2-pool real L18-MLP PD | 2 pools (1 GPU each) | per_pool_batch="
            f"{args.per_gpu_batch} seq={args.seq} C={args.C} suffix={cfg.n_layer} n_pgd={args.n_warmup}"
        )

    tgt, vu0 = s.init_target(cfg, args.C, random.PRNGKey(0))
    ci_fn = s.init_ci(
        args.ci_d_model,
        args.ci_blocks,
        args.ci_heads,
        args.ci_mlp,
        total_in,
        args.C,
        random.PRNGKey(1),
    )

    repl = NamedSharding(mesh, P())
    pool_sh = NamedSharding(mesh, P("pool"))
    tgt = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tgt)
    vu0 = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vu0)
    ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)

    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    # both pools see the same batch (work split); resid/source carry a leading pool=2 axis.
    resid1 = (
        random.normal(random.PRNGKey(42), (args.per_gpu_batch, args.seq, cfg.n_embd)) * 0.5
    ).astype(DT)
    resid = jax.device_put(jnp.broadcast_to(resid1[None], (2, *resid1.shape)), pool_sh)
    source = {
        site: jax.device_put(jnp.zeros((2, 1, args.seq, args.C), DT), pool_sh) for site in SITES
    }

    state = State(
        vu=vu0,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu0, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source=source,
        step=jnp.array(0),
    )
    step = make_step(mesh, opt_vu, opt_ci, lr_pgd=0.01, n_pgd=args.n_warmup)

    for _ in range(2):
        state, loss = step(state, tgt, resid, state.source, random.PRNGKey(7))
        jax.block_until_ready((state.source, loss))

    per = []
    for i in range(args.steps):
        t = time.time()
        state, loss = step(state, tgt, resid, state.source, random.PRNGKey(1000 + i))
        jax.block_until_ready((state.source, loss))
        per.append(time.time() - t)
    blocked = sum(per) / len(per)

    if is0:
        toks = (
            args.per_gpu_batch * args.seq
        )  # one batch, processed once (work split across 2 pools)
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | {toks / blocked:,.0f} tok/s total "
            f"| {toks / blocked / 2:,.0f} tok/s/GPU (2 GPU) | final loss {float(loss):.4f}"
        )
        print("[p0] STAGE11 2-pool: OK")

    jax.experimental.multihost_utils.sync_global_devices("stage11_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
