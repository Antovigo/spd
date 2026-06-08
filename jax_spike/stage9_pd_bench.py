"""Stage 9: full PD training step on the (parity-checked) vendored Llama — JAX throughput.

Single-pool SPMD: every rank runs the whole PD step on its batch shard, GSPMD all-reduces
grads. Assembles the real pieces:
  * target = vendored ComponentLlama (random init at benchmark scale)
  * CI fn  = global MLP on token embeddings -> per-site causal importances (leaky-hard-sigmoid)
  * losses = faithfulness + importance-minimality + stochastic recon + PGD recon
  * PGD adversary = persistent sources, lax.scan warmup inner loop
  * two optimizers (optax adamw): components V/U  and  CI fn
Data-parallel: params replicated, batch + sources sharded P('dp'); grad all-reduce is implicit.

Prints tokens/s + ms/step. Same workload spec is mirrored by the torch baseline.
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
from distributed_util import dp_mesh, init_distributed, replicate, shard_dp
from jax import random
import vendored_jax.llama as llama_mod
from vendored_jax.llama import (
    ComponentLinear,
    ComponentLlama,
    LlamaConfig,
    MaskInfo,
    all_target_paths,
    random_init,
)

llama_mod.USE_FLASH_ATTENTION = True  # flash attention for the throughput benchmark

COEFF = dict(faith=1.0, imp=0.3, stoch=1.0, ppgd=1.0)
P_IMP = 0.9


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x):
    return jnp.clip(x, 0.0, 1.0)


def _f(x):
    return jnp.clip(x, 0.0, 1.0), x


def _b(x, g):
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_f, _b)


class CIFn(eqx.Module):
    """Global CI fn: token embeddings -> per-site causal importances. One hidden layer."""

    w1: jax.Array
    w2: jax.Array
    sites: tuple[str, ...] = eqx.field(static=True)
    C: int = eqx.field(static=True)

    def __call__(self, emb: jax.Array) -> dict:
        h = jax.nn.gelu(emb @ self.w1)
        flat = h @ self.w2  # (B, T, n_sites*C)
        out = {}
        for i, s in enumerate(self.sites):
            out[s] = lower_leaky_hard_sigmoid(flat[..., i * self.C : (i + 1) * self.C])
        return out


def make_ci_fn(cfg: LlamaConfig, sites, C, hidden, key) -> CIFn:
    k1, k2 = random.split(key)
    return CIFn(
        w1=random.normal(k1, (cfg.n_embd, hidden)) * (cfg.n_embd**-0.5),
        w2=random.normal(k2, (hidden, len(sites) * C)) * (hidden**-0.5),
        sites=tuple(sites),
        C=C,
    )


def each_clin(model: ComponentLlama):
    for blk in model.blocks:
        a, m = blk.self_attn, blk.mlp
        yield from (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj)


def trainable_filter(model: ComponentLlama):
    def per(node):
        if isinstance(node, ComponentLinear):
            f = jax.tree.map(lambda _: False, node)
            return eqx.tree_at(lambda m: (m.V, m.U), f, (True, True), is_leaf=lambda x: x is None)
        return jax.tree.map(lambda _: False, node)

    return jax.tree.map(per, model, is_leaf=lambda n: isinstance(n, ComponentLinear))


def faith_loss(model: ComponentLlama) -> jax.Array:
    tot, n = jnp.array(0.0), 0
    for cl in each_clin(model):
        resid = cl.target_weight - (cl.V @ cl.U).T
        tot = tot + (resid**2).sum()
        n += resid.size
    return tot / n


def imp_loss(ci: dict) -> jax.Array:
    return jnp.mean(jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()]))


def stoch_masks(key, ci: dict) -> dict:
    return {
        s: MaskInfo(v + (1 - v) * random.uniform(random.fold_in(key, i), v.shape))
        for i, (s, v) in enumerate(ci.items())
    }


def ppgd_masks(ci: dict, sources: dict) -> dict:
    return {s: MaskInfo(ci[s] * jax.nn.sigmoid(sources[s])) for s in ci}


class State(NamedTuple):
    trainable: ComponentLlama
    frozen: ComponentLlama
    ci: CIFn
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    sources: dict
    step: jax.Array


def make_step(opt_vu, opt_ci, n_warmup, lr_pgd):
    @jax.jit
    def step(state: State, idx, key):
        def loss_fn(params):
            trainable, ci_fn = params
            model = eqx.combine(trainable, state.frozen)
            clean = jax.lax.stop_gradient(model(idx, None))  # target logits
            emb = model.embed_tokens[idx]
            ci = ci_fn(emb)

            # PGD: refine persistent sources to maximize recon (model + ci detached)
            m_det = jax.lax.stop_gradient(model)
            ci_det = jax.lax.stop_gradient(ci)

            def adv(src):
                out = m_det(idx, ppgd_masks(ci_det, src))
                return jnp.mean((out - clean) ** 2)

            def body(src, _):
                return jax.tree.map(lambda s, g: s + lr_pgd * g, src, jax.grad(adv)(src)), None

            refined, _ = jax.lax.scan(body, state.sources, None, length=n_warmup)

            l_faith = faith_loss(model)
            l_imp = imp_loss(ci)
            l_stoch = jnp.mean((model(idx, stoch_masks(key, ci)) - clean) ** 2)
            l_ppgd = jnp.mean((model(idx, ppgd_masks(ci, refined)) - clean) ** 2)
            tot = (
                COEFF["faith"] * l_faith
                + COEFF["imp"] * l_imp
                + COEFF["stoch"] * l_stoch
                + COEFF["ppgd"] * l_ppgd
            )
            return tot, (refined, tot)

        (_, (refined, total)), (g_tr, g_ci) = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            (state.trainable, state.ci)
        )
        upd_vu, os_vu = opt_vu.update(g_tr, state.opt_vu, state.trainable)
        new_tr = eqx.apply_updates(state.trainable, upd_vu)
        upd_ci, os_ci = opt_ci.update(g_ci, state.opt_ci, state.ci)
        new_ci = eqx.apply_updates(state.ci, upd_ci)
        return State(new_tr, state.frozen, new_ci, os_vu, os_ci, refined, state.step + 1), total

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=2048)
    ap.add_argument("--n_head", type=int, default=16)
    ap.add_argument("--n_kv_head", type=int, default=8)
    ap.add_argument("--n_intermediate", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--per_gpu_batch", type=int, default=8)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--ci_hidden", type=int, default=2048)
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    gbatch = args.per_gpu_batch * ndev

    cfg = LlamaConfig(
        vocab_size=args.vocab,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        n_intermediate=args.n_intermediate,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=8192,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )
    sites = all_target_paths(cfg)
    if is0:
        print(
            f"[p0] {ndev} GPU | gbatch={gbatch} seq={args.seq} | L{cfg.n_layer} d{cfg.n_embd} "
            f"ffn{cfg.n_intermediate} C{args.C} n_warmup{args.n_warmup} | sites={len(sites)}"
        )

    model = random_init(cfg, args.C, random.PRNGKey(0))
    ci_fn = make_ci_fn(cfg, sites, args.C, args.ci_hidden, random.PRNGKey(1))
    trainable, frozen = eqx.partition(model, trainable_filter(model))

    opt_vu = optax.adamw(3e-4)
    opt_ci = optax.adamw(1e-4)
    state = State(
        trainable=replicate(trainable, mesh),
        frozen=replicate(frozen, mesh),
        ci=replicate(ci_fn, mesh),
        opt_vu=replicate(opt_vu.init(eqx.filter(trainable, eqx.is_array)), mesh),
        opt_ci=replicate(opt_ci.init(eqx.filter(ci_fn, eqx.is_array)), mesh),
        sources={s: shard_dp(jnp.zeros((gbatch, args.seq, args.C)), mesh) for s in sites},
        step=jnp.array(0),
    )

    idx_full = random.randint(random.PRNGKey(42), (gbatch, args.seq), 0, args.vocab)
    idx = shard_dp(idx_full, mesh)
    step = make_step(opt_vu, opt_ci, args.n_warmup, lr_pgd=0.1)

    t0 = None
    for s in range(args.steps):
        state, total = step(state, idx, random.PRNGKey(1000 + s))
        if s == 0:
            jax.block_until_ready(total)
            t0 = time.time()
    jax.block_until_ready(state.sources)
    dt = (time.time() - t0) / (args.steps - 1)
    if is0:
        toks = gbatch * args.seq
        print(
            f"[p0] {dt * 1e3:.1f} ms/step | {toks / dt:,.0f} tok/s | {toks / dt / ndev:,.0f} tok/s/GPU "
            f"| final loss {float(total):.4f}"
        )
        print(f"[p0] STAGE 9 ({ndev} GPU): OK")

    jax.experimental.multihost_utils.sync_global_devices("stage9_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
