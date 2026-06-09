"""Roofline probe: GPT-2-XL fwd / fwd+bwd throughput + MFU vs per-GPU batch, on ONE B200.

Answers: how far below compute-bound is a single recon forward at a given per-GPU batch, and where
is the knee? If our real per-GPU batch is already above the knee, serial layer-at-a-time recon costs
~the same as the chunk-parallel version (the FLOPs/GPU are conserved), so we could consolidate.

Single device on purpose (no comm) — this is the pure compute roofline. FLOP count is the standard
6*P*tokens for fwd+bwd over the transformer matmul params (approx: ignores attention-quadratic and
embeddings, so MFU is a slight under-count at long seq — fine for locating the knee).
"""

import statistics
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import stage12_gpt2xl_1pool as s
from jax import random

dev = jax.devices()[0]
d, ffn, n_head, n_layer, vocab, block, seq = 1600, 6400, 25, 48, 50257, 1024, 1024
P = 1.5e9  # ~GPT-2-XL transformer matmul params (excl embeddings)
PEAK = 1715e12  # measured B200 bf16 dense peak (lore)

# No decomposition: pure frozen GPT-2-XL forward (representative recon-forward compute).
tgt, _ = s.init_target(d, ffn, n_head, n_layer, vocab, block, 1e-5, (), 0, random.PRNGKey(0))
tgt = jax.tree.map(lambda a: jax.device_put(a, dev) if eqx.is_array(a) else a, tgt)


def loss(tgt, idx):
    return (s.gpt2_logits(tgt, {}, (), idx, {}, {}).astype(jnp.float32) ** 2).sum()


fwd = jax.jit(lambda idx: s.gpt2_logits(tgt, {}, (), idx, {}, {}))
fwd_bwd = jax.jit(eqx.filter_grad(loss))  # grad wrt tgt -> full backward


def med(fn, *a, n=10):
    y = fn(*a)
    jax.block_until_ready(y)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        y = fn(*a)
        jax.block_until_ready(y)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print(f"{'b/gpu':>6} {'fwd_ms':>8} {'fb_ms':>9} {'tok/s':>10} {'TFLOP/s':>9} {'MFU%':>6}")
for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    try:
        idx = jax.device_put(random.randint(random.PRNGKey(b), (b, seq), 0, vocab), dev)
        t_f = med(fwd, idx)
        t_fb = med(fwd_bwd, idx)
    except Exception as e:  # noqa: BLE001 - want to see the OOM batch
        print(f"{b:>6}  stop: {type(e).__name__}: {str(e)[:60]}")
        break
    tokens = b * seq
    flops_fb = 6 * P * tokens
    print(
        f"{b:>6} {t_f * 1e3:>8.1f} {t_fb * 1e3:>9.1f} {tokens / t_fb:>10.0f} "
        f"{flops_fb / t_fb / 1e12:>9.1f} {100 * flops_fb / t_fb / PEAK:>6.1f}"
    )
