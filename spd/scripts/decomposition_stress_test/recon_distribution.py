"""Plot per-position reconstruction error distribution using rounded CI masks.

Runs N batches of data through an SPD language model with binarized causal importance masks,
computes per-position KL divergence from the original model output, and plots the distribution.
The goal is to identify whether all positions are reconstructed equally well or whether there
are outliers.

Usage:
    python -m spd.scripts.decomposition_stress_test.recon_distribution <model_path> \
        --n-batches 50 --ci-thr 0.01 --device cuda
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
        description="Plot per-position reconstruction error distribution using rounded CI masks"
    )
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument("--n-batches", type=int, default=50, help="Number of batches to process")
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: config eval_batch_size)"
    )
    parser.add_argument("--ci-thr", type=float, default=0.01, help="CI rounding threshold")
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

    all_errors: list[torch.Tensor] = []

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        print(f"Batch {i + 1}/{args.n_batches}")

        with torch.no_grad():
            cached = model(input_ids, cache_type="input")
            target_logits = cached.output

            ci = model.calc_causal_importances(cached.cache, sampling="continuous").lower_leaky

            rounded_masks = {k: (v > args.ci_thr).float() for k, v in ci.items()}
            recon_logits = model(input_ids, mask_infos=make_mask_infos(rounded_masks))

            # Per-position KL, then mean over positions -> per-example
            p = torch.softmax(target_logits, dim=-1)
            p_safe = p.clamp(min=1e-12)
            log_p = p_safe.log()
            log_q = F.log_softmax(recon_logits, dim=-1)
            per_pos_kl = (p * (log_p - log_q)).sum(dim=-1)  # (B, S)

            all_errors.append(per_pos_kl.cpu().reshape(-1))

    vals = torch.cat(all_errors).numpy()
    mean, median, std = vals.mean(), np.median(vals), vals.std()
    p95, p99 = np.percentile(vals, 95), np.percentile(vals, 99)

    print(f"\n{'=' * 50}")
    print(f"Per-position Reconstruction Error Summary (n={len(vals)})")
    print(f"{'=' * 50}")
    print(f"  mean   = {mean:.6f}")
    print(f"  median = {median:.6f}")
    print(f"  std    = {std:.6f}")
    print(f"  min    = {vals.min():.6f}")
    print(f"  max    = {vals.max():.6f}")
    print(f"  p95    = {p95:.6f}")
    print(f"  p99    = {p99:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    log_bins = np.geomspace(max(vals.min(), 1e-10), vals.max(), 80)
    ax.hist(vals, bins=log_bins, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(mean, color="red", linestyle="--", label=f"mean={mean:.4f}")
    ax.axvline(p95, color="orange", linestyle="--", label=f"p95={p95:.4f}")
    ax.axvline(p99, color="darkred", linestyle="--", label=f"p99={p99:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Per-position KL divergence")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-position Reconstruction Error Distribution (n={len(vals)})")
    ax.legend()
    fig.tight_layout()

    out_dir = args.out_dir or run_info.checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recon_distribution.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
