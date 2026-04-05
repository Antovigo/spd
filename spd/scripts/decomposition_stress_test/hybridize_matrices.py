"""Test mechanistic faithfulness by hybridizing original and decomposed matrices.

Runs two types of stochastic ablation tests:

1. **Per-matrix hybridization**: Each matrix at each position is randomly set to either original
   weights or active-only decomposed weights (no inactive components, no delta).

2. **Per-component ablation**: Each individual component and the delta are independently
   enabled or ablated (binomial), akin to StochasticRecon in binomial mode.

If the decomposition is mechanistically faithful, both should produce outputs close to the
original model. The output figure superimposes both distributions for comparison.

Usage:
    python -m spd.scripts.decomposition_stress_test.hybridize_matrices <model_path> \
        --n-batches 10 --n-masks 32 --ci-thr 0.01 --device cuda
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.configs import LMTaskConfig
from spd.data import train_loader_and_tokenizer
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import WeightDeltaAndMask, make_mask_infos


def per_pos_kl(
    p: Float[Tensor, "B S V"],
    log_p: Float[Tensor, "B S V"],
    logits: Float[Tensor, "B S V"],
) -> Float[Tensor, "B S"]:
    log_q = F.log_softmax(logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def plot_superimposed(
    ax: plt.Axes,
    matrix_vals: np.ndarray,
    comp_vals: np.ndarray,
    baseline_mean: float,
    title: str,
) -> None:
    all_pos = np.concatenate([matrix_vals[matrix_vals > 0], comp_vals[comp_vals > 0]])
    if len(all_pos) == 0:
        ax.text(0.5, 0.5, "All KL values are 0", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    log_bins = np.geomspace(max(all_pos.min(), 1e-10), all_pos.max(), 80)

    ax.hist(
        matrix_vals[matrix_vals > 0],
        bins=log_bins,
        alpha=0.5,
        color="tab:blue",
        label="per-matrix",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.hist(
        comp_vals[comp_vals > 0],
        bins=log_bins,
        alpha=0.5,
        color="tab:orange",
        label="per-component",
        edgecolor="black",
        linewidth=0.3,
    )

    ax.axvline(baseline_mean, color="green", linestyle="--", label=f"baseline={baseline_mean:.4f}")

    for vals, color_prefix, style in [
        (matrix_vals, "tab:blue", "-"),
        (comp_vals, "tab:orange", ":"),
    ]:
        mean_val = vals.mean()
        p95 = np.percentile(vals, 95)
        ax.axvline(mean_val, color=color_prefix, linestyle=style, alpha=0.8)
        ax.axvline(p95, color=color_prefix, linestyle=style, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test mechanistic faithfulness via random matrix hybridization"
    )
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: config eval_batch_size)"
    )
    parser.add_argument("--ci-thr", type=float, default=0.01, help="CI rounding threshold")
    parser.add_argument("--n-masks", type=int, default=32, help="Number of random hybrid masks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory for plot (default: model dir)"
    )
    args = parser.parse_args()

    model_path = (
        str(Path(args.model_path).expanduser()) if ":" not in args.model_path else args.model_path
    )
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), "Only LM experiments are supported"

    device = torch.device(args.device)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    batch_size = args.batch_size or config.eval_batch_size
    loader, _tokenizer = train_loader_and_tokenizer(config, batch_size)

    module_names = model.target_module_paths

    # Per-matrix accumulators
    mat_all_kls: list[Tensor] = []
    mat_worst_kls: list[Tensor] = []
    # Per-component accumulators
    comp_all_kls: list[Tensor] = []
    comp_worst_kls: list[Tensor] = []
    # Baseline
    all_baseline_kls: list[Tensor] = []

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        B, S = input_ids.shape
        print(f"Batch {i + 1}/{args.n_batches}")

        with torch.no_grad():
            cached = model(input_ids, cache_type="input")
            target_logits = cached.output

            ci = model.calc_causal_importances(cached.cache, sampling="continuous").lower_leaky
            rounded_masks = {k: (v > args.ci_thr).float() for k, v in ci.items()}

            # Baseline: all matrices use active-only decomposition, no delta
            baseline_logits = model(input_ids, mask_infos=make_mask_infos(rounded_masks))

            p = torch.softmax(target_logits, dim=-1)
            log_p = p.clamp(min=1e-12).log()

            all_baseline_kls.append(per_pos_kl(p, log_p, baseline_logits).cpu())

            # --- Per-matrix hybridization ---
            mat_worst = torch.zeros(B, S, device=device)
            for _m in range(args.n_masks):
                routing_masks = {
                    name: torch.randint(0, 2, (B, S), device=device, dtype=torch.bool)
                    for name in module_names
                }
                mask_infos = make_mask_infos(rounded_masks, routing_masks=routing_masks)
                kl = per_pos_kl(p, log_p, model(input_ids, mask_infos=mask_infos))
                mat_all_kls.append(kl.cpu())
                mat_worst = torch.maximum(mat_worst, kl)
            mat_worst_kls.append(mat_worst.cpu())

            # --- Per-component ablation ---
            weight_deltas = model.calc_weight_deltas()
            comp_worst = torch.zeros(B, S, device=device)
            for _m in range(args.n_masks):
                component_masks = {
                    name: torch.randint(
                        0, 2, (B, S, model.module_to_c[name]), device=device
                    ).float()
                    for name in module_names
                }
                wdam: dict[str, WeightDeltaAndMask] = {
                    name: (
                        weight_deltas[name],
                        torch.randint(0, 2, (B, S), device=device).float(),
                    )
                    for name in module_names
                }
                mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=wdam)
                kl = per_pos_kl(p, log_p, model(input_ids, mask_infos=mask_infos))
                comp_all_kls.append(kl.cpu())
                comp_worst = torch.maximum(comp_worst, kl)
            comp_worst_kls.append(comp_worst.cpu())

    mat_vals = torch.cat([kl.reshape(-1) for kl in mat_all_kls]).numpy()
    comp_vals = torch.cat([kl.reshape(-1) for kl in comp_all_kls]).numpy()
    mat_worst_vals = torch.cat([kl.reshape(-1) for kl in mat_worst_kls]).numpy()
    comp_worst_vals = torch.cat([kl.reshape(-1) for kl in comp_worst_kls]).numpy()
    baseline_vals = torch.cat([kl.reshape(-1) for kl in all_baseline_kls]).numpy()
    baseline_mean = float(baseline_vals.mean())

    print(f"\n{'=' * 60}")
    print("Hybridize Matrices Summary")
    print(f"{'=' * 60}")
    print(f"  n_masks={args.n_masks}, ci_thr={args.ci_thr}, n_modules={len(module_names)}")
    print(f"  Baseline (all active-only) mean KL = {baseline_mean:.6f}")
    print(
        f"  Per-matrix KL:    mean={mat_vals.mean():.6f}  median={np.median(mat_vals):.6f}"
        f"  p95={np.percentile(mat_vals, 95):.6f}  max={mat_vals.max():.6f}"
    )
    print(
        f"  Per-component KL: mean={comp_vals.mean():.6f}  median={np.median(comp_vals):.6f}"
        f"  p95={np.percentile(comp_vals, 95):.6f}  max={comp_vals.max():.6f}"
    )

    # --- Figure ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))

    plot_superimposed(
        ax_top,
        mat_vals,
        comp_vals,
        baseline_mean,
        "All KLs (across inputs, positions, masks)",
    )
    plot_superimposed(
        ax_bot,
        mat_worst_vals,
        comp_worst_vals,
        baseline_mean,
        "Worst KL per position (max across masks)",
    )

    fig.suptitle(
        f"Hybridization Test (n_masks={args.n_masks}, ci_thr={args.ci_thr})",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = args.out_dir or run_info.checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybridize_matrices.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
