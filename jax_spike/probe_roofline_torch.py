"""Roofline (torch + compile): fwd / fwd+bwd throughput + MFU vs per-GPU batch, on ONE B200.

Decides how far from compute-bound a single recon forward is at a given per-GPU batch, and where the
knee is. If our real per-GPU batch is already above the knee, serial layer-at-a-time recon costs ~the
same as the chunk-parallel version (FLOPs/GPU are conserved) -> we could consolidate.

Random weights (timing only, no HF download). bf16 + TF32, torch.compile (production path). Single
GPU = pure compute roofline. FLOP = 6*P*tokens over non-embedding params (approx: ignores
attention-quadratic, so MFU is a slight under-count at long seq — fine for locating the knee).
"""

import argparse
import statistics

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["gpt2xl", "llama8b"], required=True)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--no-compile", action="store_true")
args = ap.parse_args()

torch.set_float32_matmul_precision("high")
dev = "cuda"

if args.model == "gpt2xl":
    from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (
        GPT2Simple,
        GPT2SimpleConfig,
    )

    cfg = GPT2SimpleConfig(
        n_layer=48, n_head=25, n_embd=1600, block_size=args.seq, vocab_size=50257
    )
    model = GPT2Simple(cfg)
    vocab, embed_numel = 50257, model.wte.weight.numel()
else:
    from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import VendoredLlamaConfig
    from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

    cfg = VendoredLlamaConfig()  # defaults = Llama-3.1-8B
    model = VendoredLlama(cfg)
    vocab, embed_numel = cfg.vocab_size, model.embed_tokens.weight.numel()

model = model.to(torch.bfloat16).to(dev)
p_flop = sum(p.numel() for p in model.parameters()) - embed_numel  # embedding lookup ~0 FLOP
PEAK = 1715e12  # measured B200 bf16 dense peak (lore)
fn = model if args.no_compile else torch.compile(model)


def med(do, n=8, warmup=3):
    for _ in range(warmup):
        do()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        do()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)


print(f"model={args.model} seq={args.seq} P_flop={p_flop / 1e9:.2f}B compile={not args.no_compile}")
print(
    f"{'b/gpu':>6} {'fwd_ms':>8} {'fb_ms':>9} {'tok/s':>11} {'TFLOP/s':>9} {'MFU%':>6} {'peak_GB':>8}"
)
for b in [1, 2, 4, 8, 16, 32, 64, 128]:
    idx = torch.randint(0, vocab, (b, args.seq), device=dev)

    def fwd(idx=idx):
        with torch.no_grad():
            return fn(idx)

    def fwd_bwd(idx=idx):
        model.zero_grad(set_to_none=True)
        fn(idx).float().pow(2).sum().backward()

    try:
        t_f = med(fwd)
        t_fb = med(fwd_bwd)
    except torch.cuda.OutOfMemoryError:
        print(f"{b:>6}  OOM")
        torch.cuda.empty_cache()
        break
    tokens = b * args.seq
    flops = 6 * p_flop * tokens
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(
        f"{b:>6} {t_f * 1e3:>8.1f} {t_fb * 1e3:>9.1f} {tokens / t_fb:>11.0f} "
        f"{flops / t_fb / 1e12:>9.1f} {100 * flops / t_fb / PEAK:>6.1f} {peak_gb:>8.1f}"
    )
