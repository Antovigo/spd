"""Run the full Llama-3.1-8B single-pool VPD step in JAX and measure tok/s/GPU + MFU.

The full PD step on the REAL 8B model: residual-start L18->L31 suffix, L18 MLP decomposed
(gate/up/down, C=24576, weight-delta), global_shared_transformer CI fn, 4 losses + persistent
PGD. GSPMD-sharded: frozen suffix replicated, V/U + CI + Adam + PGD source sharded over `dp`,
batch sharded over `dp`. Matches `param_decomp_lab/experiments/lm/_fsdp/llama8b_l18_mlp_fsdp.yaml`.

Usage (single B200, random weights, fast smoke):
  python -m jax_single_pool.experiments.llama8b_real --per_gpu_batch 4 --steps 10

Real HF weights + the full residual-start prefix harvest:
  python -m jax_single_pool.experiments.llama8b_real --real_weights --per_gpu_batch 4 --steps 10

Multi-GPU under SLURM (1 task/GPU): init_distributed() brings up the mesh; --shard splits
V/U + CI + Adam + batch across the mesh (the memory story).
"""

import argparse
import time

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from vendored_jax.llama import llama3_inv_freq

from jax_single_pool.ci_fn import CIFnDims, init_ci_fn
from jax_single_pool.llama8b import (
    DT,
    SITES,
    FrozenAttn,
    FrozenBlock,
    FrozenMLP,
    Target,
    init_decomp_vu,
    llama31_8b_config,
    load_target_from_hf,
    make_real_target_residual,
)
from jax_single_pool.llama8b_sharding import (
    dp_mesh,
    replicate_target,
    shard_batch,
    shard_ci_fn,
    shard_decomp_vu,
    shard_source,
)
from jax_single_pool.llama8b_step import Llama8BState, LossCoeffs, make_llama8b_step
from jax_single_pool.sharding import init_distributed


def suffix_flops_per_token(cfg, vocab: int) -> float:
    """Matmul FLOPs for ONE suffix forward, per token (2 * params; attention-quadratic
    and elementwise ignored — a slight under-count, standard for MFU)."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    per_block = 2 * (d * qd + 2 * d * kvd + qd * d) + 2 * (3 * d * di)  # attn + mlp
    n_suffix_blocks = cfg.n_layer - 18  # L18..L31 inclusive = 14 blocks
    head = 2 * d * vocab
    return 2 * (n_suffix_blocks * per_block + head)  # 2 flops per madd


def ci_flops_per_token(dims: CIFnDims) -> float:
    dm = dims.d_model
    per_block = 2 * (4 * dm * dm) + 2 * (2 * dm * dims.mlp_hidden)
    return 2 * (dims.total_in * dm + dims.n_blocks * per_block + dm * len(SITES) * dims.C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--C", type=int, default=24576)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--real_weights", action="store_true")
    ap.add_argument("--shard", action="store_true", help="FSDP-shard V/U + CI + Adam + batch")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    args = ap.parse_args()

    distributed = init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    gbatch = args.per_gpu_batch * ndev

    cfg = llama31_8b_config()
    dims = CIFnDims(
        d_model=4096,
        n_blocks=4,
        n_heads=64,
        mlp_hidden=16384,
        total_in=cfg.n_embd + cfg.n_embd + cfg.n_intermediate,
        C=args.C,
    )
    if is0:
        print(
            f"[p0] LLAMA8B single-pool PD | {ndev} GPU | gbatch={gbatch} seq={args.seq} "
            f"C={args.C} n_warmup={args.n_warmup} shard={args.shard} "
            f"weights={'HF' if args.real_weights else 'random'}"
        )

    idx_global = random.randint(random.PRNGKey(42), (gbatch, args.seq), 0, cfg.vocab_size)
    if args.real_weights:
        if is0:
            print("[p0] loading HF suffix + harvesting residual via prefix forward...")
        target = load_target_from_hf(args.model_name, cfg)
        resid_global = make_real_target_residual(
            args.model_name, cfg, idx_global, random.PRNGKey(0)
        )
    else:
        ks = iter(random.split(random.PRNGKey(0), 512))
        d, di = cfg.n_embd, cfg.n_intermediate
        qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

        def n(shape, s=None):
            return (random.normal(next(ks), shape) * (s or d**-0.5)).astype(DT)

        def fattn():
            return FrozenAttn(
                n((qd, d)),
                n((kvd, d)),
                n((kvd, d)),
                n((d, qd)),
                cfg.n_head,
                cfg.n_kv_head,
                cfg.head_dim,
                cfg.n_rep,
            )

        def fblock():
            return FrozenBlock(
                jnp.ones((d,), DT),
                jnp.ones((d,), DT),
                fattn(),
                FrozenMLP(n((di, d)), n((di, d)), n((d, di))),
                cfg.rms_norm_eps,
            )

        target = Target(
            l18_ln1=jnp.ones((d,), DT),
            l18_ln2=jnp.ones((d,), DT),
            l18_attn=fattn(),
            l18_Wg=n((di, d)),
            l18_Wu=n((di, d)),
            l18_Wd=n((d, di)),
            rest=[fblock() for _ in range(cfg.n_layer - 18 - 1)],
            norm=jnp.ones((d,), DT),
            lm_head=n((cfg.vocab_size, d), 0.02),
            inv_freq=llama3_inv_freq(cfg),
            eps=cfg.rms_norm_eps,
        )
        resid_global = (random.normal(random.PRNGKey(7), (gbatch, args.seq, d)) * 0.5).astype(DT)

    vu = init_decomp_vu(cfg, args.C, target, random.PRNGKey(1))
    ci_fn = init_ci_fn(dims, random.PRNGKey(2))
    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    target = replicate_target(target, mesh)
    if args.shard:
        vu = shard_decomp_vu(vu, mesh)
        ci_fn = shard_ci_fn(ci_fn, mesh)
        source = shard_source({s: jnp.zeros((1, args.seq, args.C), DT) for s in SITES}, mesh)
        resid = shard_batch(resid_global, mesh)
    else:
        repl = NamedSharding(mesh, P())
        vu = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vu)
        ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)
        source = {s: jax.device_put(jnp.zeros((1, args.seq, args.C), DT), repl) for s in SITES}
        resid = shard_batch(resid_global, mesh)

    state = Llama8BState(
        vu=vu,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source=source,
        step=jnp.array(0),
    )
    coeffs = LossCoeffs(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5, p_imp=0.4)
    step = make_llama8b_step(coeffs, opt_vu, opt_ci, pgd_lr=0.01, n_warmup=args.n_warmup)

    for _ in range(2):
        state, m = step(state, target, resid, random.PRNGKey(7))
        jax.block_until_ready((state.source, m["total"]))

    per = []
    for s in range(args.steps):
        t = time.time()
        state, m = step(state, target, resid, random.PRNGKey(1000 + s))
        jax.block_until_ready((state.source, m["total"]))
        per.append(time.time() - t)
    blocked = sum(per) / len(per)

    t = time.time()
    for s in range(args.steps):
        state, m = step(state, target, resid, random.PRNGKey(2000 + s))
    dispatch = (time.time() - t) / args.steps
    jax.block_until_ready((state.source, m["total"]))

    if is0:
        toks = gbatch * args.seq
        # FLOP model: clean(fwd) + 3 stoch(fwd+bwd) + 1 ppgd(fwd+bwd) suffix forwards through
        # the decomposed MLP + suffix, + (n_warmup+1) PGD ascent suffix fwd+bwd (params detached
        # -> 1 fwd + 1 bwd each), + CI fn fwd+bwd. fwd=1x, fwd+bwd=3x (bwd is 2x fwd).
        sf = suffix_flops_per_token(cfg, cfg.vocab_size)
        cf = ci_flops_per_token(dims)
        recon_fwds_3x = 4  # 3 stoch + 1 ppgd, each fwd+bwd
        clean_1x = 1
        pgd_3x = args.n_warmup + 1  # each ascent: fwd + bwd (3x)
        suffix_flops = (clean_1x * 1 + recon_fwds_3x * 3 + pgd_3x * 3) * sf
        ci_flops = 3 * cf  # CI fn fwd + bwd
        flops_per_token = suffix_flops + ci_flops
        total_flops = flops_per_token * toks
        PEAK = 1715e12  # B200 bf16 dense peak (lore)
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | dispatch {dispatch * 1e3:.1f} ms/step "
            f"| {'HOST-BOUND' if dispatch > 0.7 * blocked else 'device-bound'}"
        )
        print(
            f"[p0] {toks / blocked:,.0f} tok/s | {toks / blocked / ndev:,.0f} tok/s/GPU "
            f"| {total_flops / blocked / 1e12:,.1f} TFLOP/s "
            f"| MFU {100 * total_flops / blocked / (PEAK * ndev):.1f}% "
            f"| final loss {float(m['total']):.4f}"
        )
        print(
            f"[p0]   losses: faith {float(m['faith']):.4e} imp {float(m['imp']):.4f} "
            f"stoch {float(m['stoch']):.4e} ppgd {float(m['ppgd']):.4e}"
        )
        print(f"[p0] LLAMA8B ({ndev} GPU): OK")

    if distributed:
        jax.experimental.multihost_utils.sync_global_devices("llama8b_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
