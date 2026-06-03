"""Single-GPU per-rank batch (`bl`) memory-ceiling probe for each 3-pool pool.

For one pool's per-rank work at GPT2-XL Q/K scale, sweeps the per-rank batch `bl`
with and without activation checkpointing and reports peak GPU memory + the OOM
boundary for each. Answers "what `bl` fits per rank, and how much does checkpointing
buy" on ONE GPU, instead of multi-node 3-pool OOM probes.

Pools (`--pool`):
  ci         CI fn (GlobalSharedTransformerCiFn, ~2B, d4096/8 blocks): fwd → lower-leaky
             → split → bwd → Adam. ckpt = checkpoint each transformer block.
  lw-target  LW masked target forward, CONSERVATIVE: all 96 q/k sites masked at once
             (routing="all"), full unsharded V/U, masked fwd → bwd, no optimizer. An
             upper bound on LW activation memory vs the real per-site streaming recon.
  lw-rank    LW REALISTIC: only `--sites-per-block` sites decomposed (owned, sharded V/U);
             streaming per-site recon (lm-head bypass + fused linear-KL vs clean hidden)
             + Adam over owned V/U. The true per-rank LW ceiling.
  ppgd       PPGD per-rank: full V/U replica (96 sites) + per-batch-per-position `sources`,
             masked fwd with mask = ci+(1-ci)*source → bwd → Adam over {V,U,sources}.

Random weights (memory profile == real HF weights; no download). bf16 autocast, matching
production. ckpt = block activation checkpointing (`enable_activation_checkpointing`).

Run: srun --gres=gpu:1 python scripts/probe_bl_ceiling.py --pool lw-rank
"""

import argparse
import gc
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

import torch
import torch.utils.checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from param_decomp.ci_fns import GlobalSharedTransformerCiFn, TargetLayerConfig  # noqa: E402
from param_decomp.ci_sigmoids import SIGMOID_TYPES  # noqa: E402
from param_decomp.components import make_components  # noqa: E402
from param_decomp.fused_linear_kl import fused_linear_kl_div  # noqa: E402
from param_decomp.masks import ComponentsMaskInfo  # noqa: E402
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import (  # noqa: E402
    GPT2Simple,
    GPT2SimpleConfig,
)
from param_decomp_lab.experiments.lm.vendored.gpt2 import (  # noqa: E402
    ComponentGPT2,
    componentize_gpt2,
)

DEV = "cuda"
# GPT2-XL target.
N_LAYERS, D_MODEL, N_HEADS, VOCAB, SEQ, C = 48, 1600, 25, 50257, 1024, 1024
SITES = [f"h.{layer}.attn.{p}_proj" for layer in range(N_LAYERS) for p in ("q", "k")]
# CI fn (the global-shared transformer; bigger than the target).
CI_D_MODEL, CI_N_LAYERS, CI_N_HEADS, CI_MLP = 4096, 8, 32, [16384]


# ── builders ────────────────────────────────────────────────────────────────
class _BlockCkpt(torch.nn.Module):
    """Wrap a CI-fn transformer block so its forward is activation-checkpointed."""

    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=False)
        assert isinstance(out, torch.Tensor)
        return out


def build_ci_fn(ckpt: bool) -> GlobalSharedTransformerCiFn:
    cfgs = {s: TargetLayerConfig(input_dim=D_MODEL, C=C) for s in SITES}
    m = GlobalSharedTransformerCiFn(
        target_model_layer_configs=cfgs,
        d_model=CI_D_MODEL,
        n_layers=CI_N_LAYERS,
        n_heads=CI_N_HEADS,
        max_len=SEQ,
        mlp_hidden_dims=CI_MLP,
    ).to(DEV)
    if ckpt:
        m._blocks = torch.nn.ModuleList([_BlockCkpt(b) for b in m._blocks])
    return m


def build_component_gpt2(ckpt: bool, decomposed_sites: list[str]) -> ComponentGPT2:
    cfg = GPT2SimpleConfig(
        model_type="GPT2Simple",
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_embd=D_MODEL,
        vocab_size=VOCAB,
        block_size=SEQ,
    )
    model = GPT2Simple(cfg)
    cg = componentize_gpt2(model, make_components(model, {s: C for s in decomposed_sites})).to(DEV)
    if ckpt:
        cg.enable_activation_checkpointing()
    return cg


def _rand_mask(bl: int) -> torch.Tensor:
    return torch.rand(bl, SEQ, C, device=DEV)


# ── per-pool step bodies (one masked train step; tensors freed on return) ─────
def step_ci(m: torch.nn.Module, bl: int) -> None:
    assert isinstance(m, GlobalSharedTransformerCiFn)
    lower = SIGMOID_TYPES["lower_leaky_hard"]
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    inputs = {s: torch.randn(bl, SEQ, D_MODEL, device=DEV) for s in SITES}
    seeds = [torch.randn(bl, SEQ, C, device=DEV) for _ in SITES]
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(inputs)
    splits = list(torch.split(lower(out), m.split_sizes, dim=-1))
    torch.autograd.backward(splits, seeds)
    opt.step()


def step_lw_target(m: torch.nn.Module, bl: int) -> None:
    idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
    mask_infos = {
        s: ComponentsMaskInfo(component_mask=_rand_mask(bl), routing_mask="all") for s in SITES
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = m(idx, mask_infos)
        loss = logits.float().pow(2).mean()
    loss.backward()


def make_step_lw_rank(owned: list[str]) -> Callable[[torch.nn.Module, int], None]:
    def step(m: torch.nn.Module, bl: int) -> None:
        assert isinstance(m, ComponentGPT2)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
        with m.bypass_lm_head() as lm_w, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                target_h = m(idx, None)
            for site in owned:
                mi = {site: ComponentsMaskInfo(component_mask=_rand_mask(bl), routing_mask="all")}
                pred_h = m(idx, mi)
                loss, n = fused_linear_kl_div(
                    pred_h.reshape(-1, D_MODEL), target_h.reshape(-1, D_MODEL), lm_w
                )
                (loss / n).backward()
        opt.step()

    return step


def step_ppgd(m: torch.nn.Module, bl: int) -> None:
    idx = torch.randint(0, VOCAB, (bl, SEQ), device=DEV)
    sources = [torch.rand(bl, SEQ, C, device=DEV, requires_grad=True) for _ in SITES]
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad] + sources, lr=1e-3)
    opt.zero_grad(set_to_none=True)
    mask_infos = {}
    for s, src in zip(SITES, sources, strict=True):
        ci = _rand_mask(bl)
        mask_infos[s] = ComponentsMaskInfo(component_mask=ci + (1 - ci) * src, routing_mask="all")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = m(idx, mask_infos)
        loss = logits.float().pow(2).mean()
    loss.backward()
    opt.step()


# ── probe registry + sweep harness ───────────────────────────────────────────
@dataclass(frozen=True)
class Probe:
    bls: list[int]
    build: Callable[[bool], torch.nn.Module]
    step: Callable[[torch.nn.Module, int], None]
    describe: str
    footer: str


def run(probe: Probe, bl: int, ckpt: bool) -> tuple[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    m: torch.nn.Module | None = None
    try:
        m = probe.build(ckpt)
        probe.step(m, bl)
        torch.cuda.synchronize()
        return "OK", torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        return "OOM", torch.cuda.max_memory_allocated() / 1e9
    finally:
        del m
        torch.cuda.empty_cache()
        gc.collect()


def _fmt(r: tuple[str, float]) -> str:
    return r[0] if r[0] in ("OOM", "skip") else f"{r[1]:.1f}GB"


def sweep(probe: Probe) -> None:
    cap = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {torch.cuda.get_device_name(0)} ~{cap:.0f}GB | {probe.describe}")
    print(f"\n{'bl':>6} | {'plain peak':>14} | {'ckpt peak':>14}")
    print("-" * 42)
    plain_oom = ckpt_oom = False
    for bl in probe.bls:
        rp = ("skip", 0.0) if plain_oom else run(probe, bl, False)
        rc = ("skip", 0.0) if ckpt_oom else run(probe, bl, True)
        plain_oom = plain_oom or rp[0] == "OOM"
        ckpt_oom = ckpt_oom or rc[0] == "OOM"
        print(f"{bl:>6} | {_fmt(rp):>14} | {_fmt(rc):>14}", flush=True)
    print(f"\n=> {probe.footer}")


def build_probe(pool: str, sites_per_block: int) -> Probe:
    match pool:
        case "ci":
            n_params = sum(p.numel() for p in build_ci_fn(False).parameters())
            torch.cuda.empty_cache()
            return Probe(
                bls=[4, 8, 16, 24, 32, 48, 64],
                build=build_ci_fn,
                step=step_ci,
                describe=f"CI fn {n_params / 1e9:.2f}B params "
                f"({len(SITES)} sites, d{CI_D_MODEL}, {CI_N_LAYERS} blocks), seq {SEQ}",
                footer="plain bl_ci ceiling = last OK below first OOM | block ckpt extends it",
            )
        case "lw-target":
            n_params = sum(p.numel() for p in build_component_gpt2(False, SITES).parameters())
            torch.cuda.empty_cache()
            return Probe(
                bls=[1, 2, 4, 8, 16, 24, 32, 48],
                build=lambda ckpt: build_component_gpt2(ckpt, SITES),
                step=step_lw_target,
                describe=f"GPT2-XL ComponentGPT2 {n_params / 1e9:.2f}B params "
                f"({len(SITES)} q/k sites masked at once, C={C}), seq {SEQ} [conservative]",
                footer="plain bl_lw ceiling (conservative, all sites) | block ckpt extends it",
            )
        case "lw-rank":
            owned = SITES[:sites_per_block]
            return Probe(
                bls=[16, 32, 48, 64, 96, 128, 192, 256],
                build=lambda ckpt: build_component_gpt2(ckpt, owned),
                step=make_step_lw_rank(owned),
                describe=f"realistic LW rank: {len(owned)} owned sites decomposed (C={C}), "
                f"streaming per-site recon (bypass+fused-KL), seq {SEQ}",
                footer="per-rank bl_lw ceiling (plain vs ckpt) = last OK below first OOM",
            )
        case "ppgd":
            return Probe(
                bls=[4, 8, 16, 24, 32, 48],
                build=lambda ckpt: build_component_gpt2(ckpt, SITES),
                step=step_ppgd,
                describe=f"GPT2-XL PPGD per-rank (full V/U replica {len(SITES)} sites C={C} "
                f"+ sources + Adam), seq {SEQ}",
                footer="plain bl_pp ceiling = last OK below first OOM | block ckpt extends it",
            )
        case _:
            raise ValueError(f"unknown pool {pool!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, choices=["ci", "lw-target", "lw-rank", "ppgd"])
    ap.add_argument(
        "--sites-per-block", type=int, default=2, help="lw-rank: owned sites per LW rank"
    )
    ap.add_argument(
        "--bls", type=int, nargs="+", default=None, help="override the per-rank batch sweep"
    )
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    probe = build_probe(args.pool, args.sites_per_block)
    if args.bls is not None:
        probe = replace(probe, bls=args.bls)
    sweep(probe)


if __name__ == "__main__":
    main()
