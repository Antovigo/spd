"""Run the full Llama-3.1-8B single-pool VPD step in JAX and measure tok/s/GPU + MFU.

The full PD step on the REAL 8B model: residual-start suffix from `--first_layer`,
MLP (gate/up/down) decomposed on layers `[first_layer, last_layer]` (3N sites),
weight-delta, global_shared_transformer CI fn (ONE shared transformer over all sites),
4 losses + persistent PGD. GSPMD-sharded: frozen suffix replicated, V/U + CI + Adam +
PGD source sharded over `dp`, batch sharded over `dp`. Matches
`param_decomp_lab/experiments/lm/_llama8b/llama8b_l18_b512_2pool_lr_mid.yaml` extended
to a layer range.

Everything that varies for the matched / max-batch / min-GPU runs is a flag:
  --first_layer / --last_layer  which layers to decompose (default 20..31 = 12 layers)
  --C                           components per site (the torch agent fixes this)
  --per_gpu_batch               local batch; gbatch = per_gpu_batch * n_devices
  mesh / world size             = all visible jax devices (SLURM topology)

Usage (single B200, random weights, fast smoke, 12 layers):
  python -m jax_single_pool.experiments.llama8b_real --per_gpu_batch 1 --steps 6 --C 2048

Real HF weights + the residual-start prefix harvest:
  python -m jax_single_pool.experiments.llama8b_real --real_weights --first_layer 20 \
      --last_layer 31 --C 8192 --per_gpu_batch 1 --steps 6 --shard

Multi-GPU under SLURM (1 task/GPU): init_distributed() brings up the mesh.
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
    KINDS,
    DecompLayerFrozen,
    FrozenAttn,
    FrozenBlock,
    FrozenMLP,
    LayerRange,
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
from jax_single_pool.llama8b_step import (
    Llama8BState,
    LossCoeffs,
    make_llama8b_step,
    make_llama8b_step_shmap,
)
from jax_single_pool.sharding import init_distributed


def suffix_flops_per_token(cfg, vocab: int, rng: LayerRange) -> float:
    """Matmul FLOPs for ONE suffix forward, per token (2 * params; attention-quadratic
    and elementwise ignored). The suffix is `first..n_layer-1` blocks + lm_head."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    per_block = 2 * (d * qd + 2 * d * kvd + qd * d) + 2 * (3 * d * di)  # attn + mlp
    n_suffix_blocks = cfg.n_layer - rng.first  # first..n_layer-1 inclusive
    head = 2 * d * vocab
    return 2 * (n_suffix_blocks * per_block + head)


def ci_flops_per_token(dims: CIFnDims) -> float:
    dm = dims.d_model
    per_block = 2 * (4 * dm * dm) + 2 * (2 * dm * dims.mlp_hidden)
    total_c = len(KINDS) * dims.n_layers * dims.C
    return 2 * (dims.total_in * dm + dims.n_blocks * per_block + dm * total_c)


def _random_target(cfg, rng: LayerRange, key) -> Target:
    ks = iter(random.split(key, 4096))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape, s=None):
        return (random.normal(next(ks), shape) * (s or d**-0.5)).astype(DT)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep,
        )  # fmt: skip

    def dlayer():
        return DecompLayerFrozen(
            ln1=jnp.ones((d,), DT), ln2=jnp.ones((d,), DT), attn=fattn(),
            Wg=n((di, d)), Wu=n((di, d)), Wd=n((d, di)),
        )  # fmt: skip

    def fblock():
        return FrozenBlock(
            jnp.ones((d,), DT), jnp.ones((d,), DT), fattn(),
            FrozenMLP(n((di, d)), n((di, d)), n((d, di))), cfg.rms_norm_eps,
        )  # fmt: skip

    n_tail = cfg.n_layer - rng.last - 1
    return Target(
        decomp_layers=[dlayer() for _ in range(rng.n_layers)],
        tail=[fblock() for _ in range(n_tail)],
        norm=jnp.ones((d,), DT),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_gpu_batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument("--first_layer", type=int, default=20)
    ap.add_argument("--last_layer", type=int, default=31)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--real_weights", action="store_true")
    ap.add_argument("--shard", action="store_true", help="jit + C-shard V/U/CI/Adam + batch")
    ap.add_argument("--shmap", action="store_true", help="shard_map DP (params replicated)")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    args = ap.parse_args()

    rng = LayerRange(first=args.first_layer, last=args.last_layer)
    assert 0 <= rng.first <= rng.last < 32, f"bad layer range {rng}"

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
        total_in=rng.n_layers * (cfg.n_embd + cfg.n_embd + cfg.n_intermediate),
        C=args.C,
        n_layers=rng.n_layers,
    )
    if is0:
        print(
            f"[p0] LLAMA8B single-pool PD | {ndev} GPU | gbatch={gbatch} seq={args.seq} "
            f"layers={rng.first}..{rng.last} ({rng.n_layers}L, {3 * rng.n_layers} sites) "
            f"C={args.C} n_warmup={args.n_warmup} "
            f"mode={'shmap' if args.shmap else 'shard' if args.shard else 'replicated'} "
            f"weights={'HF' if args.real_weights else 'random'}"
        )

    idx_global = random.randint(random.PRNGKey(42), (gbatch, args.seq), 0, cfg.vocab_size)
    if args.real_weights:
        if is0:
            print("[p0] loading HF suffix + harvesting residual via prefix forward...")
        target = load_target_from_hf(args.model_name, cfg, rng)
        resid_global = make_real_target_residual(
            args.model_name, cfg, rng, idx_global, chunk=args.per_gpu_batch
        )
    else:
        target = _random_target(cfg, rng, random.PRNGKey(0))
        resid_global = (
            random.normal(random.PRNGKey(7), (gbatch, args.seq, cfg.n_embd)) * 0.5
        ).astype(DT)

    vu = init_decomp_vu(cfg, args.C, rng.n_layers, random.PRNGKey(1))
    ci_fn = init_ci_fn(dims, random.PRNGKey(2))
    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    assert not (args.shard and args.shmap), "pick one of --shard / --shmap"
    target = replicate_target(target, mesh)
    # +1 trailing channel = the weight-delta source (torch use_delta_component=true).
    src_shape = (1, args.seq, rng.n_layers, args.C + 1)
    if args.shard:
        vu = shard_decomp_vu(vu, mesh)
        ci_fn = shard_ci_fn(ci_fn, mesh)
        source = shard_source({k: jnp.zeros(src_shape, DT) for k in KINDS}, mesh)
        resid = shard_batch(resid_global, mesh)
    else:
        repl = NamedSharding(mesh, P())
        vu = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vu)
        ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)
        source = {k: jax.device_put(jnp.zeros(src_shape, DT), repl) for k in KINDS}
        resid = shard_batch(resid_global, mesh)

    state = Llama8BState(
        vu=vu,
        ci_fn=ci_fn,
        opt_vu=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source=source,
        step=jnp.array(0),
    )
    coeffs = LossCoeffs(
        faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5, p_imp=0.4, imp_beta=0.2, imp_eps=1e-12
    )
    if args.shmap:
        step = make_llama8b_step_shmap(
            coeffs, opt_vu, opt_ci, pgd_lr=0.01, n_warmup=args.n_warmup,
            n_layers=rng.n_layers, mesh=mesh,
        )  # fmt: skip
    else:
        step = make_llama8b_step(
            coeffs, opt_vu, opt_ci, pgd_lr=0.01, n_warmup=args.n_warmup,
            n_layers=rng.n_layers, mesh=mesh if args.shard else None,
        )  # fmt: skip

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

    peak_gb = max(
        d.memory_stats()["peak_bytes_in_use"] / 1e9
        for d in jax.local_devices()
        if d.memory_stats() is not None
    )

    if is0:
        toks = gbatch * args.seq
        sf = suffix_flops_per_token(cfg, cfg.vocab_size, rng)
        cf = ci_flops_per_token(dims)
        recon_fwds_3x = 4  # 3 stoch + 1 ppgd, each fwd+bwd
        clean_1x = 1
        pgd_3x = args.n_warmup + 1
        suffix_flops = (clean_1x * 1 + recon_fwds_3x * 3 + pgd_3x * 3) * sf
        ci_flops = 3 * cf
        flops_per_token = suffix_flops + ci_flops
        total_flops = flops_per_token * toks
        PEAK = 1715e12  # B200 bf16 dense peak (lore)
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | dispatch {dispatch * 1e3:.1f} ms/step "
            f"| {'HOST-BOUND' if dispatch > 0.7 * blocked else 'device-bound'} "
            f"| peak {peak_gb:.1f} GB/dev"
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
        print(f"[p0] LLAMA8B ({ndev} GPU, {rng.n_layers}L): OK")

    if distributed:
        jax.experimental.multihost_utils.sync_global_devices("llama8b_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
