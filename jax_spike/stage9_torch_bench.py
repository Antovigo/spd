"""Stage 9 torch baseline: the SAME PD step as stage9_pd_bench.py, in PyTorch, single-pool DDP.

Mirrors the JAX bench exactly so "JAX vs torch" is apples-to-apples:
  vendored ComponentLlama (random V/U) + global CI-fn MLP + faith/imp/stoch/PGD-recon losses
  + persistent-PGD inner loop + two AdamW optimizers. Plain data-parallel: each rank runs the
  full step, grads manually all-reduced (avg) across the world (matches GSPMD's implicit reduce).

Run via remote/torch_job.sbatch (srun, 1 task/GPU, reads SLURM env for NCCL init).
  --compile  torch.compile the masked forward (fair vs XLA jit).
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from param_decomp.ci_sigmoids import lower_leaky_hard_sigmoid
from param_decomp.components import LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import componentize_llama
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

COEFF = dict(faith=1.0, imp=0.3, stoch=1.0, ppgd=1.0)
P_IMP = 0.9
ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


class CIFn(nn.Module):
    def __init__(self, d, hidden, sites, C):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d, hidden) * d**-0.5)
        self.w2 = nn.Parameter(torch.randn(hidden, len(sites) * C) * hidden**-0.5)
        self.sites, self.C = sites, C

    def forward(self, emb):
        flat = F.gelu(emb @ self.w1) @ self.w2
        return {s: lower_leaky_hard_sigmoid(flat[..., i * self.C : (i + 1) * self.C])
                for i, s in enumerate(self.sites)}


def mi(ci, src=None):
    if src is None:
        return {p: ComponentsMaskInfo(ci[p], "all", None) for p in ci}
    return {p: ComponentsMaskInfo(ci[p] * torch.sigmoid(src[p]), "all", None) for p in ci}


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(n_layer=12, n_embd=2048, n_head=16, n_kv_head=8, n_intermediate=8192,
                     vocab=32768, seq=512, per_gpu_batch=8, C=32, ci_hidden=2048,
                     n_warmup=5, steps=12).items():
        ap.add_argument(f"--{k}", type=int, default=v)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = f"cuda:{local}"
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True  # match JAX tensorfloat32
    torch.backends.cudnn.allow_tf32 = True
    is0 = rank == 0

    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama", max_position_embeddings=8192, vocab_size=args.vocab,
        n_layer=args.n_layer, n_head=args.n_head, n_key_value_heads=args.n_kv_head,
        n_embd=args.n_embd, n_intermediate=args.n_intermediate, rope_theta=500000.0,
        rope_scaling=Llama3RopeScaling(), rms_norm_eps=1e-5,
    )
    model = VendoredLlama(cfg)
    sites = [f"layers.{i}.{leaf}" for i in range(cfg.n_layer) for leaf in (ATTN + MLP)]
    comps = {}
    for p in sites:
        lin = model.get_submodule(p)
        d_out, d_in = lin.weight.shape
        comps[p] = LinearComponents(C=args.C, d_in=d_in, d_out=d_out, bias=None)
    cmodel = componentize_llama(model, comps).to(dev)
    ci_fn = CIFn(cfg.n_embd, args.ci_hidden, sites, args.C).to(dev)
    if args.compile:
        cmodel = torch.compile(cmodel)

    vu_params = [p for p in cmodel.parameters() if p.requires_grad]
    opt_vu = torch.optim.AdamW(vu_params, lr=3e-4)
    opt_ci = torch.optim.AdamW(ci_fn.parameters(), lr=1e-4)

    gbatch = args.per_gpu_batch * world
    g = torch.Generator(device="cpu").manual_seed(42)
    idx_full = torch.randint(0, args.vocab, (gbatch, args.seq), generator=g)
    idx = idx_full[rank * args.per_gpu_batch : (rank + 1) * args.per_gpu_batch].to(dev)
    sources = {p: torch.zeros(args.per_gpu_batch, args.seq, args.C, device=dev) for p in sites}

    def step():
        nonlocal sources
        opt_vu.zero_grad(set_to_none=True)
        opt_ci.zero_grad(set_to_none=True)
        with torch.no_grad():
            clean = cmodel(idx)
        emb = cmodel.embed_tokens(idx)
        ci = ci_fn(emb)
        # PGD inner loop (params + ci detached)
        ci_det = {p: ci[p].detach() for p in ci}
        src = {p: sources[p].detach().clone().requires_grad_(True) for p in sites}
        for _ in range(args.n_warmup):
            recon = ((cmodel(idx, mi(ci_det, src)) - clean) ** 2).mean()
            grads = torch.autograd.grad(recon, list(src.values()))
            src = {p: (src[p] + 0.1 * gg).detach().requires_grad_(True)
                   for p, gg in zip(sites, grads, strict=False)}
        refined = {p: src[p].detach() for p in sites}
        sources = refined
        # losses
        u = {p: torch.rand_like(ci[p]) for p in ci}
        stoch_mi = {p: ComponentsMaskInfo(ci[p] + (1 - ci[p]) * u[p], "all", None) for p in ci}
        l_stoch = ((cmodel(idx, stoch_mi) - clean) ** 2).mean()
        l_ppgd = ((cmodel(idx, mi(ci, refined)) - clean) ** 2).mean()
        l_faith = torch.stack([(d**2).mean() for d in cmodel.calc_weight_deltas().values()]).mean()
        l_imp = torch.stack([ci[p].clamp(0, 1).pow(P_IMP).mean() for p in ci]).mean()
        total = (COEFF["faith"] * l_faith + COEFF["imp"] * l_imp
                 + COEFF["stoch"] * l_stoch + COEFF["ppgd"] * l_ppgd)
        total.backward()
        # manual data-parallel grad all-reduce (avg) — matches GSPMD's implicit reduce
        for p in vu_params + list(ci_fn.parameters()):
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt_vu.step()
        opt_ci.step()
        return total.detach()

    for _ in range(3):  # warmup (alloc/autotune/compile)
        total = step()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(args.steps):
        total = step()
    torch.cuda.synchronize()
    dist.barrier()
    dt = (time.time() - t0) / args.steps
    if is0:
        toks = gbatch * args.seq
        print(f"[r0] {world} GPU | gbatch={gbatch} seq={args.seq} | compile={args.compile}")
        print(f"[r0] {dt * 1e3:.1f} ms/step | {toks / dt:,.0f} tok/s | {toks / dt / world:,.0f} tok/s/GPU "
              f"| loss {float(total):.4f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
