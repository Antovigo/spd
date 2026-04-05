"""Test mechanistic faithfulness by hybridizing original and decomposed matrices.

Runs multiple types of stochastic ablation tests and plots their KL divergence distributions:

1. **baseline**: Active-only components, no delta. Deterministic.
2. **unmasked**: All components enabled, no delta. Deterministic.
3. **per_matrix**: Each matrix randomly uses original weights or active-only decomposition.
4. **per_matrix_unmasked**: Each matrix randomly uses all components or active-only components.
5. **stoch**: Per-component binomial ablation (active always on), delta stochastic.
6. **stoch_with_delta**: Like stoch, but delta always ON.
7. **stoch_without_delta**: Like stoch, but delta always OFF.

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

COLORS = {
    "baseline": "tab:green",
    "unmasked": "tab:olive",
    "per_matrix": "tab:blue",
    "per_matrix_unmasked": "tab:cyan",
    "stoch": "tab:orange",
    "stoch_with_delta": "tab:red",
    "stoch_without_delta": "tab:purple",
}


def per_pos_kl(
    p: Float[Tensor, "B S V"],
    log_p: Float[Tensor, "B S V"],
    logits: Float[Tensor, "B S V"],
) -> Float[Tensor, "B S"]:
    log_q = F.log_softmax(logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def plot_superimposed(
    ax: plt.Axes,
    series: list[tuple[np.ndarray, str, str]],
    title: str,
    log_scale: bool = True,
) -> None:
    """Plot freqpoly (line histogram) for multiple series on the same axes.

    Args:
        series: List of (values, color, label) tuples.
    """
    all_pos_parts = [vals[vals > 0] for vals, _, _ in series]
    all_pos = (
        np.concatenate(all_pos_parts) if any(len(p) > 0 for p in all_pos_parts) else np.array([])
    )
    if len(all_pos) == 0:
        ax.text(0.5, 0.5, "All KL values are 0", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    if log_scale:
        bins = np.geomspace(max(all_pos.min(), 1e-10), all_pos.max(), 80)
        midpoints = np.sqrt(bins[:-1] * bins[1:])
    else:
        bins = np.linspace(all_pos.min(), all_pos.max(), 80)
        midpoints = (bins[:-1] + bins[1:]) / 2

    for vals, color, label in series:
        counts, _ = np.histogram(vals[vals > 0], bins=bins)
        density = counts / (counts.sum() * np.diff(bins))
        ax.plot(midpoints, density, color=color, label=label, linewidth=1.5, alpha=0.7)

    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel("KL divergence")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def stoch_component_masks(
    rounded_masks: dict[str, Tensor],
    module_to_c: dict[str, int],
    module_names: list[str],
    B: int,
    S: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Active components always on, inactive stochastically ablated."""
    return {
        name: torch.maximum(
            rounded_masks[name],
            torch.randint(0, 2, (B, S, module_to_c[name]), device=device).float(),
        )
        for name in module_names
    }


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
    parser.add_argument(
        "--scale", choices=["log", "linear"], default="log", help="X-axis scale for plots"
    )
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

    # Deterministic accumulators (no worst-KL)
    baseline_kls: list[Tensor] = []
    unmasked_kls: list[Tensor] = []

    # Stochastic accumulators: (all_kls, worst_kls)
    stoch_series_names = [
        "per_matrix",
        "per_matrix_unmasked",
        "stoch",
        "stoch_with_delta",
        "stoch_without_delta",
    ]
    all_kls: dict[str, list[Tensor]] = {name: [] for name in stoch_series_names}
    worst_kls: dict[str, list[Tensor]] = {name: [] for name in stoch_series_names}

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
            ones_masks = {name: torch.ones_like(v) for name, v in ci.items()}

            p = torch.softmax(target_logits, dim=-1)
            log_p = p.clamp(min=1e-12).log()

            # --- Deterministic ---
            baseline_kls.append(
                per_pos_kl(
                    p, log_p, model(input_ids, mask_infos=make_mask_infos(rounded_masks))
                ).cpu()
            )
            unmasked_kls.append(
                per_pos_kl(p, log_p, model(input_ids, mask_infos=make_mask_infos(ones_masks))).cpu()
            )

            # --- Stochastic ---
            weight_deltas = model.calc_weight_deltas()
            worsts = {name: torch.zeros(B, S, device=device) for name in stoch_series_names}

            for _m in range(args.n_masks):
                # per_matrix: routing to original or active-only
                routing = {
                    name: torch.randint(0, 2, (B, S), device=device, dtype=torch.bool)
                    for name in module_names
                }
                kl = per_pos_kl(
                    p,
                    log_p,
                    model(
                        input_ids, mask_infos=make_mask_infos(rounded_masks, routing_masks=routing)
                    ),
                )
                all_kls["per_matrix"].append(kl.cpu())
                worsts["per_matrix"] = torch.maximum(worsts["per_matrix"], kl)

                # per_matrix_unmasked: per-module binary → all components or active-only
                mat_unmasked_masks = {}
                for name in module_names:
                    binary = torch.randint(0, 2, (B, S, 1), device=device).float()
                    mat_unmasked_masks[name] = (
                        binary * ones_masks[name] + (1 - binary) * rounded_masks[name]
                    )
                kl = per_pos_kl(
                    p,
                    log_p,
                    model(input_ids, mask_infos=make_mask_infos(mat_unmasked_masks)),
                )
                all_kls["per_matrix_unmasked"].append(kl.cpu())
                worsts["per_matrix_unmasked"] = torch.maximum(worsts["per_matrix_unmasked"], kl)

                # Shared stochastic component masks for the three stoch variants
                comp_masks = stoch_component_masks(
                    rounded_masks,
                    model.module_to_c,
                    module_names,
                    B,
                    S,
                    device,
                )

                # stoch: delta stochastic
                wdam_stoch: dict[str, WeightDeltaAndMask] = {
                    name: (
                        weight_deltas[name],
                        torch.randint(0, 2, (B, S), device=device).float(),
                    )
                    for name in module_names
                }
                kl = per_pos_kl(
                    p,
                    log_p,
                    model(
                        input_ids,
                        mask_infos=make_mask_infos(comp_masks, weight_deltas_and_masks=wdam_stoch),
                    ),
                )
                all_kls["stoch"].append(kl.cpu())
                worsts["stoch"] = torch.maximum(worsts["stoch"], kl)

                # stoch_with_delta: delta always ON
                wdam_on: dict[str, WeightDeltaAndMask] = {
                    name: (weight_deltas[name], torch.ones(B, S, device=device))
                    for name in module_names
                }
                kl = per_pos_kl(
                    p,
                    log_p,
                    model(
                        input_ids,
                        mask_infos=make_mask_infos(comp_masks, weight_deltas_and_masks=wdam_on),
                    ),
                )
                all_kls["stoch_with_delta"].append(kl.cpu())
                worsts["stoch_with_delta"] = torch.maximum(worsts["stoch_with_delta"], kl)

                # stoch_without_delta: delta always OFF
                kl = per_pos_kl(
                    p,
                    log_p,
                    model(input_ids, mask_infos=make_mask_infos(comp_masks)),
                )
                all_kls["stoch_without_delta"].append(kl.cpu())
                worsts["stoch_without_delta"] = torch.maximum(worsts["stoch_without_delta"], kl)

            for name in stoch_series_names:
                worst_kls[name].append(worsts[name].cpu())

    # Flatten all accumulators
    baseline_vals = torch.cat([kl.reshape(-1) for kl in baseline_kls]).numpy()
    unmasked_vals = torch.cat([kl.reshape(-1) for kl in unmasked_kls]).numpy()
    stoch_vals = {
        name: torch.cat([kl.reshape(-1) for kl in all_kls[name]]).numpy()
        for name in stoch_series_names
    }
    stoch_worst_vals = {
        name: torch.cat([kl.reshape(-1) for kl in worst_kls[name]]).numpy()
        for name in stoch_series_names
    }

    print(f"\n{'=' * 70}")
    print("Hybridize Matrices Summary")
    print(f"{'=' * 70}")
    print(f"  n_masks={args.n_masks}, ci_thr={args.ci_thr}, n_modules={len(module_names)}")
    print(f"  {'baseline':<25s} mean={baseline_vals.mean():.6f}")
    print(f"  {'unmasked':<25s} mean={unmasked_vals.mean():.6f}")
    for name in stoch_series_names:
        v = stoch_vals[name]
        print(
            f"  {name:<25s} mean={v.mean():.6f}  median={np.median(v):.6f}"
            f"  p95={np.percentile(v, 95):.6f}  max={v.max():.6f}"
        )

    # --- Figure ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9))
    log_scale = args.scale == "log"

    # Top: all 7 series
    top_series: list[tuple[np.ndarray, str, str]] = [
        (baseline_vals, COLORS["baseline"], "baseline"),
        (unmasked_vals, COLORS["unmasked"], "unmasked"),
    ]
    for name in stoch_series_names:
        top_series.append((stoch_vals[name], COLORS[name], name))
    plot_superimposed(ax_top, top_series, "All KLs (across inputs, positions, masks)", log_scale)

    # Bottom: worst-KL for stochastic series only
    bot_series = [(stoch_worst_vals[name], COLORS[name], name) for name in stoch_series_names]
    plot_superimposed(ax_bot, bot_series, "Worst KL per position (max across masks)", log_scale)

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
