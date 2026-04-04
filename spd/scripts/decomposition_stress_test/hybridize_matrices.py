"""Test mechanistic faithfulness by hybridizing original and decomposed matrices.

For each batch, computes active components (CI > threshold), then runs n_masks forward passes
where each matrix at each position is randomly set to either original weights or active-only
decomposed weights (no inactive components, no delta). If the decomposition is mechanistically
faithful, these hybrid combinations should produce outputs close to the original model.

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

from spd.configs import LMTaskConfig
from spd.data import train_loader_and_tokenizer
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos


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

    all_hybrid_kls: list[torch.Tensor] = []  # each: (B, S)
    all_baseline_kls: list[torch.Tensor] = []  # each: (B, S)
    all_worst_kls: list[torch.Tensor] = []  # each: (B, S)

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

            log_q_baseline = F.log_softmax(baseline_logits, dim=-1)
            baseline_kl = (p * (log_p - log_q_baseline)).sum(dim=-1)  # (B, S)
            all_baseline_kls.append(baseline_kl.cpu())

            # Track worst KL per position across masks
            worst_kl = torch.zeros(B, S, device=device)

            for _m in range(args.n_masks):
                # Per-module, per-position random routing mask
                routing_masks = {
                    name: torch.randint(0, 2, (B, S), device=device, dtype=torch.bool)
                    for name in module_names
                }

                mask_infos = make_mask_infos(rounded_masks, routing_masks=routing_masks)
                hybrid_logits = model(input_ids, mask_infos=mask_infos)

                log_q = F.log_softmax(hybrid_logits, dim=-1)
                hybrid_kl = (p * (log_p - log_q)).sum(dim=-1)  # (B, S)

                all_hybrid_kls.append(hybrid_kl.cpu())
                worst_kl = torch.maximum(worst_kl, hybrid_kl)

            all_worst_kls.append(worst_kl.cpu())

    hybrid_vals = torch.cat([kl.reshape(-1) for kl in all_hybrid_kls]).numpy()
    baseline_vals = torch.cat([kl.reshape(-1) for kl in all_baseline_kls]).numpy()
    worst_vals = torch.cat([kl.reshape(-1) for kl in all_worst_kls]).numpy()

    baseline_mean = baseline_vals.mean()

    print(f"\n{'=' * 60}")
    print("Hybridize Matrices Summary")
    print(f"{'=' * 60}")
    print(f"  n_masks={args.n_masks}, ci_thr={args.ci_thr}")
    print(f"  n_modules={len(module_names)}")
    print(f"  Baseline (all active-only) mean KL = {baseline_mean:.6f}")
    print(
        f"  Hybrid KL:  mean={hybrid_vals.mean():.6f}  median={np.median(hybrid_vals):.6f}"
        f"  std={hybrid_vals.std():.6f}  max={hybrid_vals.max():.6f}"
    )
    print(
        f"  Worst KL:   mean={worst_vals.mean():.6f}  median={np.median(worst_vals):.6f}"
        f"  p95={np.percentile(worst_vals, 95):.6f}  max={worst_vals.max():.6f}"
    )

    # --- Figure ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))

    for ax, vals, title in [
        (ax_top, hybrid_vals, "All hybrid KLs (across inputs, positions, masks)"),
        (ax_bot, worst_vals, "Worst KL per position (max across masks)"),
    ]:
        vals_pos = vals[vals > 0]
        if len(vals_pos) == 0:
            ax.text(
                0.5, 0.5, "All KL values are 0", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title)
            continue

        log_bins = np.geomspace(max(vals_pos.min(), 1e-10), vals_pos.max(), 80)
        ax.hist(vals_pos, bins=log_bins, edgecolor="black", linewidth=0.3, alpha=0.7)
        ax.axvline(
            baseline_mean, color="green", linestyle="--", label=f"baseline mean={baseline_mean:.4f}"
        )
        mean_val = vals.mean()
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        ax.axvline(mean_val, color="red", linestyle="--", label=f"mean={mean_val:.4f}")
        ax.axvline(p95, color="orange", linestyle="--", label=f"p95={p95:.4f}")
        ax.axvline(p99, color="darkred", linestyle="--", label=f"p99={p99:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("KL divergence")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Matrix Hybridization Test (n_masks={args.n_masks}, ci_thr={args.ci_thr})",
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
