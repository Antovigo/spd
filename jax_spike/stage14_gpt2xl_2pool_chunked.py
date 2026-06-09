"""Stage 14: GPT-2-XL MLP PD CHUNK-PARALLEL stochastic recon — V/U REPLICATED.

stage13 (2-pool) generalized so the per-site stochastic recon forwards run on SEPARATE GPUs
concurrently. The mesh is a Cartesian (group, dp) grid:

  mesh = Mesh(devices.reshape(Nc+1, Nd), ("group", "dp"))

with group indices 0..Nc-1 = CHUNK groups, group index Nc = MAIN POOL (adversary + CI).
The sites are statically partitioned across the Nc chunks; each chunk group runs ONLY its
own sites' masked stochastic forwards (+ its sites' faith term) — that's the parallelism.
The main pool runs importance-minimality + the persistent-PGD adversary recon over ALL
sites, and owns the PGD source update.

V/U is REPLICATED here (compute-parallel only); sharding V/U is a later version.

The batch is sharded over dp ONLY and replicated over group: every group sees the full
global batch (work split across groups, data-parallel within a group over dp).

Reuses every GPT-2-XL model component + CI fn + recon helper from stage12.
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


def make_step(mesh, opt_vu, opt_ci, dec_layers, sites, site_chunks, lr_pgd, n_pgd):
    Nc = len(site_chunks)
    dm = {site: jnp.ones((1, 1, 1), DT) for site in sites}
    nomask = {site: None for site in sites}
    ckpt = jax.checkpoint(s.gpt2_logits, static_argnums=(2,))
    n_sites = len(sites)

    @jax.jit
    def step(state: State, frozen, idx, source, key):
        def per_device(frozen, trainable, idx_g, src_g):
            vus, ci_fn = trainable
            idx_g = idx_g[0, 0]  # strip the (group, dp) mapped axes (each size 1)
            src_g = {site: v[0, 0] for site, v in src_g.items()}
            g = jax.lax.axis_index("group")

            clean = jax.lax.stop_gradient(s.gpt2_logits(frozen, vus, dec_layers, idx_g, nomask, dm))
            # CI is computed ONLY on the main pool (g == Nc), then broadcast to every group:
            # main holds ci, the others hold zeros_like(ci); psum over "group" sums to ci.
            ci_local = ci_fn(s.clean_site_inputs(frozen, dec_layers, idx_g))
            ci_or_zeros = jax.tree.map(lambda v: jnp.where(g == Nc, v, jnp.zeros_like(v)), ci_local)
            ci = {site: jax.lax.psum(v, "group") for site, v in ci_or_zeros.items()}

            def chunk_branch(c):
                def f(_):
                    chunk_sites = site_chunks[c]
                    wd = {
                        site: s.frozen_W(frozen, site) - (vus[site].V @ vus[site].U).T
                        for site in chunk_sites
                    }
                    l_faith = sum((d**2).sum() for d in wd.values()) / sum(
                        d.size for d in wd.values()
                    )
                    m_stoch = {
                        site: ci[site]
                        + (1 - ci[site])
                        * random.uniform(random.fold_in(key, i), ci[site].shape, DT)
                        for i, site in enumerate(chunk_sites)
                    }
                    masks = dict(nomask)
                    masks.update(m_stoch)
                    l_stoch = s.recon(ckpt(frozen, vus, dec_layers, idx_g, masks, dm), clean)
                    return (
                        COEFF["stoch"] * l_stoch * (len(chunk_sites) / n_sites)
                        + COEFF["faith"] * l_faith
                    )

                return f

            def main_branch(_):
                l_imp = jnp.mean(
                    jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()])
                )
                ppgd = {site: ci[site] * jax.nn.sigmoid(src_g[site]) for site in sites}
                l_adv = s.recon(ckpt(frozen, vus, dec_layers, idx_g, ppgd, dm), clean)
                return COEFF["imp"] * l_imp + COEFF["ppgd"] * l_adv

            branches = [chunk_branch(c) for c in range(Nc)] + [main_branch]
            out = jax.lax.switch(g, branches, operand=None)
            return out[None, None]

        def total(trainable):
            sm = shard_map(
                per_device,
                mesh=mesh,
                in_specs=(P(), P(), P("group", "dp"), P("group", "dp")),
                out_specs=P("group", "dp"),
                check_rep=False,
            )
            return jnp.sum(sm(frozen, trainable, idx, source))

        loss, grads = jax.value_and_grad(total)((state.vus, state.ci_fn))
        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.vus)
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.ci_fn)
        new_vu = eqx.apply_updates(state.vus, upd_vu)
        new_ci = eqx.apply_updates(state.ci_fn, upd_ci)

        vu_det = jax.lax.stop_gradient(state.vus)
        ci_fn_det = jax.lax.stop_gradient(state.ci_fn)

        def update_source(frozen, idx_g, src_g):
            idx_g = idx_g[0, 0]
            src_g = {site: v[0, 0] for site, v in src_g.items()}
            g = jax.lax.axis_index("group")
            clean = jax.lax.stop_gradient(
                s.gpt2_logits(frozen, vu_det, dec_layers, idx_g, nomask, dm)
            )
            ci0 = jax.lax.stop_gradient(ci_fn_det(s.clean_site_inputs(frozen, dec_layers, idx_g)))

            def adv(src1):
                masks = {site: ci0[site] * jax.nn.sigmoid(src1[site]) for site in sites}
                return s.recon(s.gpt2_logits(frozen, vu_det, dec_layers, idx_g, masks, dm), clean)

            def body(src1, _):
                grad = jax.lax.psum(jax.grad(adv)(src1), "dp")  # aggregate over the pool's batch
                return jax.tree.map(lambda a, b: a + lr_pgd * b, src1, grad), None

            new_src1, _ = jax.lax.cond(
                g == Nc,
                lambda sp: jax.lax.scan(body, sp, None, length=n_pgd),
                lambda sp: (jax.tree.map(jnp.zeros_like, sp), None),
                src_g,
            )
            # broadcast the main pool's result to every group (others contributed zeros)
            new_src1 = {site: jax.lax.psum(v, "group") for site, v in new_src1.items()}
            return {site: v[None, None] for site, v in new_src1.items()}

        sm_src = shard_map(
            update_source,
            mesh=mesh,
            in_specs=(P(), P("group", "dp"), P("group", "dp")),
            out_specs=P("group", "dp"),
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
    ap.add_argument("--n_chunks", type=int, default=1)
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
    Nc = args.n_chunks
    assert ndev % (Nc + 1) == 0, f"ndev {ndev} not divisible by Nc+1 {Nc + 1}"
    Nd = ndev // (Nc + 1)
    mesh = Mesh(np.array(jax.devices()).reshape(Nc + 1, Nd), axis_names=("group", "dp"))
    is0 = jax.process_index() == 0
    assert args.global_batch % Nd == 0, f"global_batch {args.global_batch} not divisible by Nd {Nd}"
    per_rank = args.global_batch // Nd

    dec_layers = tuple(range(args.dec_start, args.dec_start + args.n_dec))
    sites = tuple(f"h{i}.{site}" for i in dec_layers for site in ("c_fc", "c_proj"))
    site_chunks = [tuple(sites[c::Nc]) for c in range(Nc)]
    assert all(len(sc) > 0 for sc in site_chunks), f"empty chunk in {site_chunks}"
    total_in = args.n_dec * (args.d + args.ffn)
    if is0:
        print(
            f"[p0] STAGE14 GPT-2-XL 2-pool CHUNKED | {ndev} GPU (group={Nc + 1} x dp={Nd}) | "
            f"Nc={Nc} global_batch={args.global_batch} per_rank={per_rank} seq={args.seq} "
            f"C={args.C} dec_layers={dec_layers} n_sites={len(sites)} n_pgd={args.n_warmup}"
        )
        print(f"[p0] site_chunks={site_chunks}")

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
    groupdp = NamedSharding(mesh, P("group", "dp"))
    tgt = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tgt)
    vus = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vus)
    ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)

    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    # every group sees the same global batch (work split); leading (group, dp) axes carry it.
    idx1 = random.randint(random.PRNGKey(42), (Nd, per_rank, args.seq), 0, args.vocab)
    idx = jax.device_put(jnp.broadcast_to(idx1[None], (Nc + 1, Nd, per_rank, args.seq)), groupdp)
    source = {
        site: jax.device_put(jnp.zeros((Nc + 1, Nd, 1, args.seq, args.C), DT), groupdp)
        for site in sites
    }

    state = State(
        vus=vus,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vus, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source=source,
        step=jnp.array(0),
    )
    step = make_step(
        mesh, opt_vu, opt_ci, dec_layers, sites, site_chunks, lr_pgd=0.01, n_pgd=args.n_warmup
    )

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
        toks = args.global_batch * args.seq
        peak_gb = max(d.memory_stats()["peak_bytes_in_use"] for d in jax.local_devices()) / 1e9
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | {toks / blocked:,.0f} tok/s total "
            f"| {toks / blocked / ndev:,.0f} tok/s/GPU ({ndev} GPU)"
        )
        print(f"[p0] peak_mem/device={peak_gb:.2f} GB | Nc={Nc} mesh=(group={Nc + 1} x dp={Nd})")
        print(
            f"[p0] loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} (down={losses[0] - losses[-1]:.4f})"
        )
        print(f"[p0] STAGE14 2-pool chunked ({ndev} GPU, Nc={Nc}): OK")

    jax.experimental.multihost_utils.sync_global_devices("stage14_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
