"""Stage 13: GPT-2-XL MLP PD workload SPLIT across two pools — JAX throughput, 8 GPU.

The 2-pool companion to stage12, on a 2-D mesh (pool=2, dp=4) so it runs at the SAME 8 GPU
and SAME global batch as the 1-pool cells (the clean equal-GPU/equal-batch A/B; lore
2026-06-08--4way-gpt2xl-clean-ab). Both pools process the FULL global batch (work split, not
data split); within each pool the batch is data-parallel over the dp axis.

  Pool A (adversary + CI): CI fn forward, importance-minimality, persistent-PGD adversary
    recon loss, and the persistent-PGD SOURCE update (PGD scan, source grad psum'd over dp).
  Pool B (component recon): faithfulness + layerwise stochastic recon (one masked forward per
    site) + PPGD recon, using components V/U and the ci shipped A->B via differentiable
    `jax.lax.ppermute` (cotangents return through the transpose — deletes torch's manual
    cross-pool g_CI plumbing).

Per-GPU throughput = global_batch*seq / step / 8 (one batch, work split across the 2 pools)
so at matched batch the 2-pool necessarily trails 1-pool tok/s/GPU; its win is memory at scale.

Reuses every GPT-2-XL model component + CI fn from stage12.
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
import optax
import stage12_gpt2xl_1pool as s
from distributed_util import init_distributed
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

DT = s.DT
COEFF = s.COEFF
P_IMP = s.P_IMP


class State(NamedTuple):
    vus: dict
    ci_fn: s.CIFn
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    source: dict
    step: jax.Array


def make_step(mesh, opt_vu, opt_ci, dec_layers, sites, lr_pgd, n_pgd):
    dm = {site: jnp.ones((1, 1, 1), DT) for site in sites}
    nomask = {site: None for site in sites}
    ckpt = jax.checkpoint(s.gpt2_logits, static_argnums=(2,))

    @jax.jit
    def step(state: State, frozen, idx, source, key):
        def per_pool(frozen, trainable, idx_p, src_p):
            vus, ci_fn = trainable
            idx_p = idx_p[0, 0]  # strip the (pool, dp) mapped axes (each size 1)
            src_p = {site: v[0, 0] for site, v in src_p.items()}
            pool = jax.lax.axis_index("pool")

            clean = jax.lax.stop_gradient(s.gpt2_logits(frozen, vus, dec_layers, idx_p, nomask, dm))
            ci = ci_fn(s.clean_site_inputs(frozen, dec_layers, idx_p))
            ci_b = {site: jax.lax.ppermute(v, "pool", perm=[(0, 1)]) for site, v in ci.items()}

            def pool_a(_):
                l_imp = jnp.mean(
                    jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()])
                )
                ppgd_a = {site: ci[site] * jax.nn.sigmoid(src_p[site]) for site in sites}
                l_adv = s.recon(ckpt(frozen, vus, dec_layers, idx_p, ppgd_a, dm), clean)
                return COEFF["imp"] * l_imp + COEFF["ppgd"] * l_adv

            def pool_b(_):
                wd = {
                    site: s.frozen_W(frozen, site) - (vus[site].V @ vus[site].U).T for site in sites
                }
                l_faith = sum((d**2).sum() for d in wd.values()) / sum(d.size for d in wd.values())
                l_stoch = jnp.array(0.0)
                for i, site in enumerate(sites):
                    u = random.uniform(random.fold_in(key, i), ci_b[site].shape, dtype=DT)
                    m = {**nomask, site: ci_b[site] + (1 - ci_b[site]) * u}
                    l_stoch = l_stoch + s.recon(ckpt(frozen, vus, dec_layers, idx_p, m, dm), clean)
                l_stoch = l_stoch / len(sites)
                ppgd_b = {site: ci_b[site] * jax.nn.sigmoid(src_p[site]) for site in sites}
                l_ppgd = s.recon(ckpt(frozen, vus, dec_layers, idx_p, ppgd_b, dm), clean)
                return COEFF["faith"] * l_faith + COEFF["stoch"] * l_stoch + COEFF["ppgd"] * l_ppgd

            return jax.lax.cond(pool == 0, pool_a, pool_b, operand=None)[None, None]

        def total(trainable):
            sm = shard_map(
                per_pool,
                mesh=mesh,
                in_specs=(P(), P(), P("pool", "dp"), P("pool", "dp")),
                out_specs=P("pool", "dp"),
                check_rep=False,
            )
            return jnp.sum(sm(frozen, trainable, idx, source))

        loss, grads = jax.value_and_grad(total)((state.vus, state.ci_fn))
        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.vus)
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.ci_fn)
        new_vu = eqx.apply_updates(state.vus, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        # persistent-PGD source update: pool A only (pool B contributes zeros), batch-aggregated
        # over dp; psum over pool broadcasts pool A's result back to both pools for next step.
        vu_det = jax.lax.stop_gradient(state.vus)
        ci_fn_det = jax.lax.stop_gradient(state.ci_fn)

        def update_source(frozen, idx_p, src_p):
            idx_p = idx_p[0, 0]
            src_p = {site: v[0, 0] for site, v in src_p.items()}
            pool = jax.lax.axis_index("pool")
            clean = jax.lax.stop_gradient(
                s.gpt2_logits(frozen, vu_det, dec_layers, idx_p, nomask, dm)
            )
            ci0 = jax.lax.stop_gradient(ci_fn_det(s.clean_site_inputs(frozen, dec_layers, idx_p)))

            def adv(src1):
                masks = {site: ci0[site] * jax.nn.sigmoid(src1[site]) for site in sites}
                return s.recon(s.gpt2_logits(frozen, vu_det, dec_layers, idx_p, masks, dm), clean)

            def body(src1, _):
                g = jax.lax.psum(jax.grad(adv)(src1), "dp")  # aggregate over the pool's batch
                return jax.tree.map(lambda a, b: a + lr_pgd * b, src1, g), None

            new_src1, _ = jax.lax.cond(
                pool == 0,
                lambda sp: jax.lax.scan(body, sp, None, length=n_pgd),
                lambda sp: (jax.tree.map(jnp.zeros_like, sp), None),
                src_p,
            )
            # broadcast pool A's result to both pools (pool B contributed zeros)
            new_src1 = {site: jax.lax.psum(v, "pool") for site, v in new_src1.items()}
            return {site: v[None, None] for site, v in new_src1.items()}

        sm_src = shard_map(
            update_source,
            mesh=mesh,
            in_specs=(P(), P("pool", "dp"), P("pool", "dp")),
            out_specs=P("pool", "dp"),
            check_rep=False,
        )
        new_source = jax.lax.stop_gradient(sm_src(frozen, idx, source))
        return State(new_vu, new_ci, os_vu, os_ci, new_source, state.step + 1), loss

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--global_batch", type=int, default=16)
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_dec", type=int, default=3)
    ap.add_argument("--dec_start", type=int, default=20)
    ap.add_argument("--d", type=int, default=1600)
    ap.add_argument("--ffn", type=int, default=6400)
    ap.add_argument("--n_head", type=int, default=25)
    ap.add_argument("--n_layer", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--ci_d_model", type=int, default=1024)
    ap.add_argument("--ci_blocks", type=int, default=5)
    ap.add_argument("--ci_heads", type=int, default=8)
    ap.add_argument("--ci_mlp", type=int, default=4096)
    args = ap.parse_args()

    init_distributed()
    ndev = len(jax.devices())
    assert ndev % 2 == 0, "2-pool needs an even device count"
    dp = ndev // 2
    mesh = Mesh(np.array(jax.devices()).reshape(2, dp), axis_names=("pool", "dp"))
    is0 = jax.process_index() == 0
    assert args.global_batch % dp == 0, f"global_batch {args.global_batch} not divisible by dp {dp}"
    per_rank = args.global_batch // dp

    dec_layers = tuple(range(args.dec_start, args.dec_start + args.n_dec))
    sites = tuple(f"h{i}.{site}" for i in dec_layers for site in ("c_fc", "c_proj"))
    total_in = args.n_dec * (args.d + args.ffn)
    if is0:
        print(
            f"[p0] STAGE13 GPT-2-XL 2-pool | {ndev} GPU (pool=2 x dp={dp}) | global_batch="
            f"{args.global_batch} per_rank={per_rank} seq={args.seq} C={args.C} "
            f"dec_layers={dec_layers} n_sites={len(sites)} n_pgd={args.n_warmup}"
        )

    tgt, vus = s.init_target(
        args.d,
        args.ffn,
        args.n_head,
        args.n_layer,
        args.vocab,
        args.block,
        1e-5,
        dec_layers,
        args.C,
        random.PRNGKey(0),
    )
    ci_fn = s.init_ci(
        args.ci_d_model,
        args.ci_blocks,
        args.ci_heads,
        args.ci_mlp,
        total_in,
        args.C,
        sites,
        random.PRNGKey(1),
    )

    repl = NamedSharding(mesh, P())
    pooldp = NamedSharding(mesh, P("pool", "dp"))
    tgt = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tgt)
    vus = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vus)
    ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)

    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    # both pools see the same global batch (work split); leading (pool=2, dp) axes carry it.
    idx1 = random.randint(random.PRNGKey(42), (dp, per_rank, args.seq), 0, args.vocab)
    idx = jax.device_put(jnp.broadcast_to(idx1[None], (2, dp, per_rank, args.seq)), pooldp)
    source = {
        site: jax.device_put(jnp.zeros((2, dp, 1, args.seq, args.C), DT), pooldp) for site in sites
    }

    state = State(
        vus=vus,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vus, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source=source,
        step=jnp.array(0),
    )
    step = make_step(mesh, opt_vu, opt_ci, dec_layers, sites, lr_pgd=0.01, n_pgd=args.n_warmup)

    for _ in range(2):
        state, loss = step(state, tgt, idx, state.source, random.PRNGKey(7))
        jax.block_until_ready((state.source, loss))

    losses = []
    per = []
    for i in range(args.steps):
        t = time.time()
        state, loss = step(state, tgt, idx, state.source, random.PRNGKey(1000 + i))
        jax.block_until_ready((state.source, loss))
        per.append(time.time() - t)
        losses.append(float(loss))
    blocked = sum(per) / len(per)

    if is0:
        toks = args.global_batch * args.seq  # one batch, processed once (work split across pools)
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | {toks / blocked:,.0f} tok/s total "
            f"| {toks / blocked / ndev:,.0f} tok/s/GPU ({ndev} GPU)"
        )
        print(
            f"[p0] loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} (down={losses[0] - losses[-1]:.4f})"
        )
        print(f"[p0] STAGE13 2-pool ({ndev} GPU): OK")

    jax.experimental.multihost_utils.sync_global_devices("stage13_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
