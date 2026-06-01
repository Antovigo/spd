"""Single-GPU bl-ceiling + block-checkpointing gain for the GPT2-XL masked target forward.

The LW pool's per-rank memory is dominated by the full GPT2-XL forward (clean target_fwd +
masked faithfulness/stoch recon) at the per-rank batch `bl_lw`. Block activation checkpointing
of that forward is the lever to raise `bl_lw` (→ fewer LW ranks → faster). This probe builds the
vendored `ComponentGPT2` at GPT2-XL scale (48 layers, d 1600, q/k decomposed, C=1024), runs a
masked full-model forward → backward at a sweep of `bl`, with and without block checkpointing,
and reports peak GPU memory and the OOM boundary for each.

Random weights (memory profile is identical to real HF weights; no download). bf16 autocast,
matching production. All q/k sites masked (routing="all") — a conservative upper bound on LW
activation memory vs the per-site streaming recon, so the checkpointing GAIN it reports is
conservative.

Run: srun --gres=gpu:1 python scripts/probe_lw_target_bl_ceiling.py
"""

import gc
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.components import make_components  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
DEV = "cuda"
SITES = [f"h.{layer}.attn.{p}_proj" for layer in range(N_LAYERS) for p in ("q", "k")]
BLS = [1, 2, 4, 8, 16, 24, 32, 48]


def build(ckpt: bool) -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    model = GPT2Simple(cfg)
    components = make_components(model, {s: C for s in SITES})
    cg = componentize_gpt2(model, components).to(DEV)
    if ckpt:
        cg.enable_activation_checkpointing()
    return cg


def run(bl: int, ckpt: bool) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    try:
        cg = build(ckpt)
        idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
        mask_infos = {
            s: ComponentsMaskInfo(
                component_mask=torch.rand(bl, SEQ, C, device=DEV), routing_mask="all"
            )
            for s in SITES
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = cg(idx, mask_infos)
            loss = logits.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        del cg, idx, mask_infos, logits, loss
        return "OK", peak
    except torch.cuda.OutOfMemoryError:
        return "OOM", torch.cuda.max_memory_allocated() / 1e9
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    n_params = sum(p.numel() for p in build(False).parameters())
    torch.cuda.empty_cache()
    gc.collect()
    print(
        f"GPU {torch.cuda.get_device_name(0)} ~{cap:.0f}GB | GPT2-XL ComponentGPT2 "
        f"{n_params / 1e9:.2f}B trainable+frozen params ({len(SITES)} q/k sites, C={C}), seq {SEQ}"
    )
    print(f"\n{'bl':>4} | {'plain peak':>14} | {'ckpt peak':>14}")
    print("-" * 40)

    def fmt(r: tuple[str, float]) -> str:
        return "OOM" if r[0] == "OOM" else "skip" if r[0] == "skip" else f"{r[1]:.1f}GB"

    plain_oom = ckpt_oom = False
    for bl in BLS:
        rp = ("skip", 0.0) if plain_oom else run(bl, False)
        rc = ("skip", 0.0) if ckpt_oom else run(bl, True)
        plain_oom = plain_oom or rp[0] == "OOM"
        ckpt_oom = ckpt_oom or rc[0] == "OOM"
        print(f"{bl:>4} | {fmt(rp):>14} | {fmt(rc):>14}", flush=True)
    print("\n=> plain bl_lw ceiling = last OK below first OOM | block ckpt extends it")


if __name__ == "__main__":
    main()
