"""Standalone 1-GPU memory probe of the gpt2-xl q/k CI fn.

Builds GlobalSharedTransformerCiFn exactly as in the 3-pool CI pool, drives a
representative forward+backward at batch_local=8, seq=1024 with synthetic
pre-weight-act inputs and an Adam step, under bf16 autocast, and attributes HBM
to: params, grad, Adam state, input-projector / block / output-head activations,
CI value tensors (+ their grads), imp-min temps.

Toggles each lever and prints the measured delta:
  (a) retain_graph double-backward ON vs OFF   -> the step_ci.py fix
  (b) CI value tensors fp32 vs bf16
  (c) Adam state fp32 vs bf16

Run on one GPU: srun --gpus=1 --time=0:30:00 ... (no CPU/mem flags, no --partition).
"""

import argparse
import gc

import torch

from param_decomp.ci_fns import GlobalSharedTransformerCiFn, TargetLayerConfig
from param_decomp.ci_sigmoids import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from param_decomp.metrics.importance_minimality import finalize_imp_min, per_component_lp_sums

GB = 1e9
N_SITES = 96
N_LAYERS_TARGET = 48
INPUT_DIM = 1600
C = 1024
D_MODEL = 4096
N_BLOCKS = 8
N_HEADS = 32
MLP = 16384
MAX_LEN = 1024


def build_ci_fn() -> GlobalSharedTransformerCiFn:
    tlc = {
        f"h.{layer}.attn.{proj}": TargetLayerConfig(input_dim=INPUT_DIM, C=C)
        for layer in range(N_LAYERS_TARGET)
        for proj in ("q_proj", "k_proj")
    }
    assert len(tlc) == N_SITES, len(tlc)
    return GlobalSharedTransformerCiFn(
        target_model_layer_configs=tlc,
        d_model=D_MODEL,
        n_layers=N_BLOCKS,
        n_heads=N_HEADS,
        max_len=MAX_LEN,
        mlp_hidden_dims=[MLP],
    ).cuda()


def count_params(m: torch.nn.Module) -> dict[str, int]:
    def n(mod):
        return sum(p.numel() for p in mod.parameters())

    blocks = sum(n(b) for b in m._blocks)
    return {
        "input_projector": n(m._input_projector),
        "output_head": n(m._output_head),
        "blocks": blocks,
        "total": n(m),
    }


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cur_alloc() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / GB


def peak_alloc() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GB


def split_ci(output: torch.Tensor, split_sizes: list[int]):
    pre = output
    lower = lower_leaky_hard_sigmoid(1.05 * pre - 0.05 * torch.rand_like(pre))
    upper = upper_leaky_hard_sigmoid(pre)
    lower_s = torch.split(lower, split_sizes, dim=-1)
    upper_s = torch.split(upper, split_sizes, dim=-1)
    return lower_s, upper_s


def run(bl: int, seq: int, ci_vals_bf16: bool, adam_bf16: bool, retain_graph: bool):
    """One full fwd+bwd+adam cycle; returns a dict of GB measurements."""
    gc.collect()
    reset_peak()
    m = build_ci_fn()
    params = count_params(m)
    after_params = cur_alloc()

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    site_names = m.layer_order
    split_sizes = m.split_sizes

    inputs = {
        s: torch.randn(bl, seq, INPUT_DIM, device="cuda", dtype=torch.float32) for s in site_names
    }
    after_inputs = cur_alloc()

    reset_peak()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = m(inputs)  # [bl, seq, total_c]
    output = output.float()
    lower_s, upper_s = split_ci(output, split_sizes)
    val_dtype = torch.bfloat16 if ci_vals_bf16 else torch.float32
    lower = {s: lower_s[i].to(val_dtype) for i, s in enumerate(site_names)}
    upper = {s: upper_s[i].to(val_dtype) for i, s in enumerate(site_names)}
    peak_fwd = peak_alloc()
    after_fwd = cur_alloc()

    ci_val_bytes = sum(t.numel() * t.element_size() for t in lower.values()) + sum(
        t.numel() * t.element_size() for t in upper.values()
    )
    ci_val_bytes += output.numel() * output.element_size()  # pre_sigmoid (fp32 always)

    per_sums, n_ex = per_component_lp_sums(ci_upper_leaky=upper, pnorm=2.0, eps=1e-12)
    imp = finalize_imp_min(per_component_sums=per_sums, n_examples=n_ex, beta=1.0)
    scaled_imp = 0.1 * imp

    g_seeds = [torch.ones_like(lower_s[i]) for i in range(len(site_names))]
    lower_tensors = [lower_s[i] for i in range(len(site_names))]

    reset_peak()
    if retain_graph:
        torch.autograd.backward(tensors=lower_tensors, grad_tensors=g_seeds, retain_graph=True)
        torch.autograd.backward(tensors=[scaled_imp], grad_tensors=[None])
    else:
        torch.autograd.backward(tensors=[*lower_tensors, scaled_imp], grad_tensors=[*g_seeds, None])
    peak_bwd = peak_alloc()
    after_bwd = cur_alloc()

    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size() for p in m.parameters() if p.grad is not None
    )

    if adam_bf16:
        opt.step()  # initializes fp32 state
        for st in opt.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                st[k] = st[k].to(torch.bfloat16)
    reset_peak()
    opt.step()
    peak_adam = peak_alloc()
    adam_bytes = 0
    for st in opt.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                adam_bytes += st[k].numel() * st[k].element_size()

    param_bytes = sum(p.numel() * p.element_size() for p in m.parameters())

    res = {
        "param_count_total": params["total"],
        "param_count_input_proj": params["input_projector"],
        "param_count_output_head": params["output_head"],
        "param_count_blocks": params["blocks"],
        "params_GB": param_bytes / GB,
        "grad_GB": grad_bytes / GB,
        "adam_GB": adam_bytes / GB,
        "ci_value_tensors_GB": ci_val_bytes / GB,
        "after_params_GB": after_params,
        "after_inputs_GB": after_inputs,
        "peak_during_fwd_GB": peak_fwd,
        "after_fwd_GB": after_fwd,
        "peak_during_bwd_GB": peak_bwd,
        "after_bwd_GB": after_bwd,
        "peak_during_adam_GB": peak_adam,
        "input_cache_GB": sum(t.numel() * t.element_size() for t in inputs.values()) / GB,
    }
    # activation high-water during fwd = peak_fwd - (params+inputs already resident)
    res["fwd_activation_hw_GB"] = peak_fwd - after_inputs
    res["bwd_activation_hw_GB"] = peak_bwd - after_fwd
    del m, opt, inputs, output, lower, upper, lower_s, upper_s, g_seeds
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bl", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()

    print(
        f"device: {torch.cuda.get_device_name()}  total HBM: {torch.cuda.get_device_properties(0).total_memory / GB:.1f} GB"
    )
    print(
        f"config: bl={args.bl} seq={args.seq} sites={N_SITES} C={C} d_model={D_MODEL} blocks={N_BLOCKS} mlp={MLP}\n"
    )

    base = run(args.bl, args.seq, ci_vals_bf16=False, adam_bf16=False, retain_graph=False)
    print("=== BASELINE (fp32 CI vals, fp32 Adam, single fused backward) ===")
    for k, v in base.items():
        if k.startswith("param_count"):
            print(f"  {k:30s} {v:,}")
        else:
            print(f"  {k:30s} {v:8.3f} GB")

    print("\n=== STATIC (computed from param counts) ===")
    static = base["params_GB"] + base["grad_GB"] + base["adam_GB"]
    print(f"  params + grad + adam = {static:.2f} GB")

    print("\n=== LEVER (a): retain_graph double-backward ON vs OFF ===")
    on = run(args.bl, args.seq, ci_vals_bf16=False, adam_bf16=False, retain_graph=True)
    off = base
    print(f"  retain_graph=True  peak_during_bwd = {on['peak_during_bwd_GB']:.3f} GB")
    print(f"  retain_graph=False peak_during_bwd = {off['peak_during_bwd_GB']:.3f} GB")
    print(
        f"  SAVING from single backward         = {on['peak_during_bwd_GB'] - off['peak_during_bwd_GB']:.3f} GB"
    )

    print("\n=== LEVER (b): CI value tensors fp32 vs bf16 ===")
    fp32v = base
    bf16v = run(args.bl, args.seq, ci_vals_bf16=True, adam_bf16=False, retain_graph=False)
    print(
        f"  fp32 CI vals = {fp32v['ci_value_tensors_GB']:.3f} GB | bf16 = {bf16v['ci_value_tensors_GB']:.3f} GB"
    )
    print(
        f"  SAVING (CI vals)            = {fp32v['ci_value_tensors_GB'] - bf16v['ci_value_tensors_GB']:.3f} GB"
    )

    print("\n=== LEVER (c): Adam state fp32 vs bf16 ===")
    print(
        f"  fp32 Adam = {base['adam_GB']:.3f} GB -> bf16 = {base['adam_GB'] / 2:.3f} GB | SAVING {base['adam_GB'] / 2:.3f} GB"
    )

    print("\n=== CI-VALUE-TENSOR RATIOS (key decision metric) ===")
    civ = base["ci_value_tensors_GB"]
    dyn = base["fwd_activation_hw_GB"] + base["input_cache_GB"]
    act_proper = base["fwd_activation_hw_GB"]
    print(f"  CI value tensors                  = {civ:.3f} GB")
    print(f"  total dynamic (act_hw + input cache) = {dyn:.3f} GB")
    print(f"  activations proper (excl input cache)= {act_proper:.3f} GB")
    print(f"  (a) CI vals as % of total dynamic      = {100 * civ / dyn:.1f}%")
    print(f"  (b) CI vals as % of activations proper = {100 * civ / act_proper:.1f}%")


if __name__ == "__main__":
    main()
