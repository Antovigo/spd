"""Stage 12: clean-A/B GPT-2-XL MLP PD workload in JAX — single pool, throughput (tok/s/GPU).

Companion to stage10 (Llama-8B suffix) but on the FULL GPT-2-XL model so the 1-pool cell
fits replicated at 8 GPU — enabling the controlled {jax,torch}x{1,2-pool} A/B at equal GPU
count and equal global batch (lore 2026-06-08--4way-gpt2xl-clean-ab).

Workload (matched across all four cells):
  * FULL GPT-2-XL forward from token ids (learned wpe, LayerNorm, tanh-GELU, no RoPE, no GQA).
    48 layers, d 1600, 25 heads, ffn 6400, vocab 50257, block 1024.
  * decompose `n_dec` MLP layers' c_fc + c_proj (C components each), weight-delta on; every
    other linear frozen.
  * CI fn = global_shared_transformer over the per-site clean input acts (each c_fc_in 1600,
    c_proj_in 6400) concat -> Linear d_model -> bidirectional RoPE blocks -> Linear -> lhs.
  * losses: faithfulness (weight MSE over all sites) + importance-minimality
    + StochasticReconLayerwise (one masked full forward per site) + PersistentPGD recon
    (one masked full forward w/ broadcast source) + a persistent-PGD source Adam-ish update.
  * bf16 params/compute; optax adamw (fp32 states); TF32 matmul (matches torch).

Single-pool GSPMD: params replicated, token ids + PGD source sharded over 'dp'.
Reuses stage10's CIBlock / leaky-hard-sigmoid / _proj / recon (architecture-agnostic).
"""

import argparse
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
from distributed_util import dp_mesh, init_distributed
from jax import random
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from stage10_real_pd_bench import CIBlock, _proj, lhs, recon
from vendored_jax.gpt2 import layer_norm, new_gelu

jax.config.update("jax_default_matmul_precision", "tensorfloat32")  # match torch TF32 path

DT = jnp.bfloat16
COEFF = dict(faith=1e5, imp=5e-6, stoch=0.5, ppgd=0.5)
P_IMP = 0.4


# ----------------------------- frozen GPT-2 block -----------------------------
class GPT2Attn(eqx.Module):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    bq: jax.Array
    bk: jax.Array
    bv: jax.Array
    bo: jax.Array
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(self, x):
        b, t, c = x.shape
        # jax.nn.dot_product_attention wants (batch, seq, heads, head_dim) layout
        q = (x @ self.wq.T + self.bq).reshape(b, t, self.n_head, self.head_dim)
        k = (x @ self.wk.T + self.bk).reshape(b, t, self.n_head, self.head_dim)
        v = (x @ self.wv.T + self.bv).reshape(b, t, self.n_head, self.head_dim)
        y = jax.nn.dot_product_attention(q, k, v, is_causal=True).reshape(b, t, c)
        return y @ self.wo.T + self.bo


class GPT2Block(eqx.Module):
    ln1_w: jax.Array
    ln1_b: jax.Array
    ln2_w: jax.Array
    ln2_b: jax.Array
    attn: GPT2Attn
    # frozen MLP weights (also the weight-delta target for decomposed layers)
    Wfc: jax.Array  # (ffn, d)
    bfc: jax.Array  # (ffn,)
    Wproj: jax.Array  # (d, ffn)
    bproj: jax.Array  # (d,)
    eps: float = eqx.field(static=True)


class VU(eqx.Module):
    V: jax.Array
    U: jax.Array


class Target(eqx.Module):
    wte: jax.Array
    wpe: jax.Array
    blocks: list  # GPT2Block * n_layer
    lnf_w: jax.Array
    lnf_b: jax.Array
    eps: float = eqx.field(static=True)


def _fc_in(blk: GPT2Block, x):
    return layer_norm(x, blk.ln2_w, blk.ln2_b, blk.eps)


def frozen_mlp(blk: GPT2Block, fc_in):
    h = new_gelu(fc_in @ blk.Wfc.T + blk.bfc)
    return h @ blk.Wproj.T + blk.bproj


def decomp_mlp(blk: GPT2Block, vu_fc: VU, vu_proj: VU, fc_in, m_fc, m_proj, dm):
    # c_fc: weight-delta forward (bias handled in the delta path's frozen term via blk.bfc)
    fc = _proj(fc_in, vu_fc.V, vu_fc.U, blk.Wfc, m_fc, dm) + blk.bfc
    h = new_gelu(fc)
    return _proj(h, vu_proj.V, vu_proj.U, blk.Wproj, m_proj, dm) + blk.bproj


def gpt2_logits(tgt: Target, vus: dict, dec_layers, idx, masks, dms):
    t = idx.shape[1]
    x = tgt.wte[idx] + tgt.wpe[jnp.arange(t)]
    for i, blk in enumerate(tgt.blocks):
        x = x + blk.attn(layer_norm(x, blk.ln1_w, blk.ln1_b, blk.eps))
        fc_in = _fc_in(blk, x)
        if i in dec_layers:
            kf, kp = f"h{i}.c_fc", f"h{i}.c_proj"
            x = x + decomp_mlp(blk, vus[kf], vus[kp], fc_in, masks[kf], masks[kp], dms[kf])
        else:
            x = x + frozen_mlp(blk, fc_in)
    x = layer_norm(x, tgt.lnf_w, tgt.lnf_b, tgt.eps)
    return x @ tgt.wte.T


def clean_site_inputs(tgt: Target, dec_layers, idx):
    """Replay the clean forward, collecting each decomposed site's input activation."""
    t = idx.shape[1]
    x = tgt.wte[idx] + tgt.wpe[jnp.arange(t)]
    site_in = {}
    for i, blk in enumerate(tgt.blocks):
        x = x + blk.attn(layer_norm(x, blk.ln1_w, blk.ln1_b, blk.eps))
        fc_in = _fc_in(blk, x)
        if i in dec_layers:
            site_in[f"h{i}.c_fc"] = fc_in
            site_in[f"h{i}.c_proj"] = new_gelu(fc_in @ blk.Wfc.T + blk.bfc)
        x = x + frozen_mlp(blk, fc_in)
    return site_in


# ----------------------------- CI fn (global_shared_transformer) -----------------------------
class CIFn(eqx.Module):
    in_proj: jax.Array
    blocks: list  # CIBlock
    out_head: jax.Array
    inv_freq: jax.Array = eqx.field()
    C: int = eqx.field(static=True)
    sites: tuple = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, site_inputs: dict):
        normed = [
            layer_norm(
                site_inputs[s],
                jnp.ones((site_inputs[s].shape[-1],), DT),
                jnp.zeros((site_inputs[s].shape[-1],), DT),
                self.eps,
            )
            for s in self.sites
        ]
        x = jax.nn.relu(jnp.concatenate(normed, axis=-1) @ self.in_proj)
        for blk in self.blocks:
            x = blk(x, self.inv_freq)
        flat = x @ self.out_head
        return {s: lhs(flat[..., i * self.C : (i + 1) * self.C]) for i, s in enumerate(self.sites)}


# ----------------------------- init -----------------------------
def init_target(d, ffn, n_head, n_layer, vocab, block, eps, dec_layers, C, key):
    ks = iter(random.split(key, 8 * n_layer + 16))
    sc = d**-0.5

    def n(shape, s=sc):
        return (random.normal(next(ks), shape) * s).astype(DT)

    hd = d // n_head

    def mk_block():
        return GPT2Block(
            ln1_w=jnp.ones((d,), DT),
            ln1_b=jnp.zeros((d,), DT),
            ln2_w=jnp.ones((d,), DT),
            ln2_b=jnp.zeros((d,), DT),
            attn=GPT2Attn(
                n((d, d)),
                n((d, d)),
                n((d, d)),
                n((d, d)),
                jnp.zeros((d,), DT),
                jnp.zeros((d,), DT),
                jnp.zeros((d,), DT),
                jnp.zeros((d,), DT),
                n_head,
                hd,
            ),
            Wfc=n((ffn, d)),
            bfc=jnp.zeros((ffn,), DT),
            Wproj=n((d, ffn)),
            bproj=jnp.zeros((d,), DT),
            eps=eps,
        )

    tgt = Target(
        wte=n((vocab, d), 0.02),
        wpe=n((block, d), 0.02),
        blocks=[mk_block() for _ in range(n_layer)],
        lnf_w=jnp.ones((d,), DT),
        lnf_b=jnp.zeros((d,), DT),
        eps=eps,
    )
    vus = {}
    for i in dec_layers:
        vus[f"h{i}.c_fc"] = VU(n((d, C)), n((C, ffn), C**-0.5))
        vus[f"h{i}.c_proj"] = VU(n((ffn, C)), n((C, d), C**-0.5))
    return tgt, vus


def init_ci(d_model, n_blocks, n_heads, mlp_hidden, total_in, C, sites, key):
    ks = iter(random.split(key, 8 * n_blocks + 8))
    hd = d_model // n_heads

    def n(shape, s):
        return (random.normal(next(ks), shape) * s).astype(DT)

    def block():
        return CIBlock(
            ln1=jnp.ones((d_model,), DT),
            ln2=jnp.ones((d_model,), DT),
            wq=n((d_model, d_model), d_model**-0.5),
            wk=n((d_model, d_model), d_model**-0.5),
            wv=n((d_model, d_model), d_model**-0.5),
            wo=n((d_model, d_model), d_model**-0.5),
            w1=n((d_model, mlp_hidden), d_model**-0.5),
            w2=n((mlp_hidden, d_model), mlp_hidden**-0.5),
            n_head=n_heads,
            head_dim=hd,
            eps=1e-5,
        )

    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    return CIFn(
        in_proj=n((total_in, d_model), total_in**-0.5),
        blocks=[block() for _ in range(n_blocks)],
        out_head=n((d_model, len(sites) * C), d_model**-0.5),
        inv_freq=inv_freq,
        C=C,
        sites=tuple(sites),
        eps=1e-5,
    )


# ----------------------------- training step -----------------------------
class State(NamedTuple):
    trainable: tuple  # (vus, ci_fn)
    opt_vu: optax.OptState
    opt_ci: optax.OptState
    source: dict  # site -> (1,T,C) broadcast PGD source
    step: jax.Array


def make_step(opt_vu, opt_ci, dec_layers, sites, lr_pgd, n_pgd):
    @jax.jit
    def step(state: State, frozen: Target, idx, key):
        dm = {s: jnp.ones((1, 1, 1), DT) for s in sites}
        nomask = {s: None for s in sites}
        ckpt = jax.checkpoint(gpt2_logits, static_argnums=(2,))

        def loss_fn(trainable):
            vus, ci_fn = trainable
            clean = jax.lax.stop_gradient(gpt2_logits(frozen, vus, dec_layers, idx, nomask, dm))
            ci = ci_fn(clean_site_inputs(frozen, dec_layers, idx))

            wd = {s: frozen_W(frozen, s) - (vus[s].V @ vus[s].U).T for s in sites}
            l_faith = sum((d**2).sum() for d in wd.values()) / sum(d.size for d in wd.values())
            l_imp = jnp.mean(jnp.stack([jnp.mean(jnp.clip(v, 0, 1) ** P_IMP) for v in ci.values()]))

            l_stoch = jnp.array(0.0)
            for i, s in enumerate(sites):
                u = random.uniform(random.fold_in(key, i), ci[s].shape, dtype=DT)
                m = {**nomask, s: ci[s] + (1 - ci[s]) * u}
                l_stoch = l_stoch + recon(ckpt(frozen, vus, dec_layers, idx, m, dm), clean)
            l_stoch = l_stoch / len(sites)

            src = jax.lax.stop_gradient(state.source)
            ppgd_masks = {s: ci[s] * jax.nn.sigmoid(src[s]) for s in sites}
            l_ppgd = recon(ckpt(frozen, vus, dec_layers, idx, ppgd_masks, dm), clean)

            tot = (
                COEFF["faith"] * l_faith
                + COEFF["imp"] * l_imp
                + COEFF["stoch"] * l_stoch
                + COEFF["ppgd"] * l_ppgd
            )
            return tot, (clean, ci)

        (tot, (clean, ci)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            state.trainable
        )
        upd_vu, os_vu = opt_vu.update(grads[0], state.opt_vu, state.trainable[0])
        upd_ci, os_ci = opt_ci.update(grads[1], state.opt_ci, state.trainable[1])
        new_vu = eqx.apply_updates(state.trainable[0], upd_vu)
        new_ci = eqx.apply_updates(state.trainable[1], upd_ci)

        vu_det = jax.lax.stop_gradient(state.trainable[0])
        ci_det = jax.lax.stop_gradient(ci)

        def adv(src):
            masks = {s: ci_det[s] * jax.nn.sigmoid(src[s]) for s in sites}
            return recon(gpt2_logits(frozen, vu_det, dec_layers, idx, masks, dm), clean)

        def body(src, _):
            g = jax.grad(adv)(src)
            return jax.tree.map(lambda s, gg: s + lr_pgd * gg, src, g), None

        new_src, _ = jax.lax.scan(body, state.source, None, length=n_pgd)
        new_src = jax.lax.stop_gradient(new_src)
        return State((new_vu, new_ci), os_vu, os_ci, new_src, state.step + 1), tot

    return step


def frozen_W(tgt: Target, site: str):
    # site = "h{i}.c_fc" | "h{i}.c_proj"
    i = int(site[1:].split(".")[0])
    blk = tgt.blocks[i]
    return blk.Wfc if site.endswith("c_fc") else blk.Wproj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--per_gpu_batch", type=int, default=2)
    ap.add_argument("--C", type=int, default=8192)
    ap.add_argument(
        "--n_warmup", type=int, default=2, help="persistent-PGD inner steps per train step"
    )
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--n_dec", type=int, default=3, help="number of decomposed MLP layers")
    ap.add_argument("--dec_start", type=int, default=20)
    # GPT-2-XL dims
    ap.add_argument("--d", type=int, default=1600)
    ap.add_argument("--ffn", type=int, default=6400)
    ap.add_argument("--n_head", type=int, default=25)
    ap.add_argument("--n_layer", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--block", type=int, default=1024)
    # CI fn
    ap.add_argument("--ci_d_model", type=int, default=1024)
    ap.add_argument("--ci_blocks", type=int, default=5)
    ap.add_argument("--ci_heads", type=int, default=8)
    ap.add_argument("--ci_mlp", type=int, default=4096)
    args = ap.parse_args()

    init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0
    gbatch = args.per_gpu_batch * ndev
    dec_layers = tuple(range(args.dec_start, args.dec_start + args.n_dec))
    sites = [f"h{i}.{s}" for i in dec_layers for s in ("c_fc", "c_proj")]
    total_in = args.n_dec * (args.d + args.ffn)
    if is0:
        print(
            f"[p0] STAGE12 GPT-2-XL 1-pool | {ndev} GPU | gbatch={gbatch} seq={args.seq} C={args.C} "
            f"dec_layers={dec_layers} n_sites={len(sites)} n_pgd={args.n_warmup}"
        )

    tgt, vus = init_target(
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
    ci_fn = init_ci(
        args.ci_d_model,
        args.ci_blocks,
        args.ci_heads,
        args.ci_mlp,
        total_in,
        args.C,
        sites,
        random.PRNGKey(1),
    )

    opt_vu = optax.adamw(1.5e-4)
    opt_ci = optax.adamw(5e-5)

    repl = NamedSharding(mesh, P())
    shard_dp = NamedSharding(mesh, P("dp"))
    tgt = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, tgt)
    vus = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, vus)
    ci_fn = jax.tree.map(lambda a: jax.device_put(a, repl) if eqx.is_array(a) else a, ci_fn)

    state = State(
        trainable=(vus, ci_fn),
        opt_vu=opt_vu.init(eqx.filter(vus, eqx.is_array)),
        opt_ci=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        source={s: jnp.zeros((1, args.seq, args.C), DT) for s in sites},
        step=jnp.array(0),
    )

    idx_full = random.randint(random.PRNGKey(42), (gbatch, args.seq), 0, args.vocab)
    idx = jax.device_put(idx_full, shard_dp)
    step = make_step(opt_vu, opt_ci, dec_layers, tuple(sites), lr_pgd=0.01, n_pgd=args.n_warmup)

    for _ in range(2):
        state, tot = step(state, tgt, idx, random.PRNGKey(7))
        jax.block_until_ready((state.source, tot))

    losses = []
    per = []
    for st in range(args.steps):
        t = time.time()
        state, tot = step(state, tgt, idx, random.PRNGKey(1000 + st))
        jax.block_until_ready((state.source, tot))
        per.append(time.time() - t)
        losses.append(float(tot))
    blocked = sum(per) / len(per)

    t = time.time()
    for st in range(args.steps):
        state, tot = step(state, tgt, idx, random.PRNGKey(2000 + st))
    dispatch = (time.time() - t) / args.steps
    jax.block_until_ready((state.source, tot))

    if is0:
        toks = gbatch * args.seq
        print(
            f"[p0] blocked {blocked * 1e3:.1f} ms/step | dispatch {dispatch * 1e3:.1f} ms/step "
            f"| {'HOST-BOUND' if dispatch > 0.7 * blocked else 'device-bound'}"
        )
        print(f"[p0] {toks / blocked:,.0f} tok/s | {toks / blocked / ndev:,.0f} tok/s/GPU")
        print(
            f"[p0] loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} (down={losses[0] - losses[-1]:.4f})"
        )
        print(f"[p0] STAGE12 ({ndev} GPU): OK")

    jax.experimental.multihost_utils.sync_global_devices("stage12_done")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
