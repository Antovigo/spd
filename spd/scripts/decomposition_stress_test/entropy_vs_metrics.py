"""Plot L0 and reconstruction error as a function of the target model's output entropy.

Runs batches of data through a decomposed SPD model, collecting per-position statistics:
- Entropy of the target model's output distribution
- L0: number of active components (CI > threshold), summed across layers
- KL divergence between rounded-CI-masked and target model outputs

Produces a single figure with two side-by-side scatter plots.

Usage:
    python -m spd.scripts.decomposition_stress_test.entropy_vs_metrics <model_path> \
        --n-batches 20 --ci-thr 0.01 --out-dir ./plots
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
    parser = argparse.ArgumentParser(description="Plot L0 and KL vs target model entropy")
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: config eval_batch_size)"
    )
    parser.add_argument("--ci-thr", type=float, default=0.01, help="CI threshold for L0")
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

    all_entropy: list[torch.Tensor] = []
    all_l0: list[torch.Tensor] = []
    all_kl: list[torch.Tensor] = []

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        print(f"Batch {i + 1}/{args.n_batches}")

        with torch.no_grad():
            cached = model(input_ids, cache_type="input")
            target_logits = cached.output
            pre_weight_acts = cached.cache

            ci = model.calc_causal_importances(pre_weight_acts, sampling="continuous")
            ci_lower = ci.lower_leaky

            # Round CI to binary masks
            rounded_masks = {k: (v > args.ci_thr).float() for k, v in ci_lower.items()}

            # Per-position L0: sum active components across all layers
            l0_per_pos = sum(v.sum(-1) for v in rounded_masks.values())
            assert isinstance(l0_per_pos, torch.Tensor)

            # Rounded-CI-masked forward pass
            masked_logits = model(input_ids, mask_infos=make_mask_infos(rounded_masks))

            # Per-position KL divergence
            p = torch.softmax(target_logits, dim=-1)
            p_safe = p.clamp(min=1e-12)
            log_p = p_safe.log()
            log_q = F.log_softmax(masked_logits, dim=-1)
            per_pos_kl = (p * (log_p - log_q)).sum(dim=-1)

            # Entropy of target distribution
            entropy = -(p * log_p).sum(dim=-1)

        all_entropy.append(entropy.cpu().reshape(-1))
        all_l0.append(l0_per_pos.cpu().reshape(-1))
        all_kl.append(per_pos_kl.cpu().reshape(-1))

    entropy_np = torch.cat(all_entropy).numpy()
    l0_np = torch.cat(all_l0).numpy()
    kl_np = torch.cat(all_kl).numpy()

    print(f"Collected {len(entropy_np)} token positions")

    # Plot
    out_dir = args.out_dir or run_info.checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_l0, ax_kl) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_data, y_label in [
        (ax_l0, l0_np, "L0 (active components)"),
        (ax_kl, kl_np, "KL divergence (nats)"),
    ]:
        ax.scatter(entropy_np, y_data, alpha=0.1, s=4, color="grey", edgecolors="none")

        # Binned mean summary line (log-spaced bins)
        n_bins = 30
        e_min = max(entropy_np.min(), 1e-6)
        bin_edges = np.geomspace(e_min, entropy_np.max(), n_bins + 1)
        bin_idx = np.clip(np.digitize(entropy_np, bin_edges) - 1, 0, n_bins - 1)
        centers, means = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() < 5:
                continue
            centers.append(np.sqrt(bin_edges[b] * bin_edges[b + 1]))
            means.append(y_data[mask].mean())
        ax.plot(centers, means, "o-", color="red", markersize=4, linewidth=1.5)

        ax.set_xlabel("Target model entropy (nats)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs entropy")

    ax_l0.set_xscale("log")
    ax_kl.set_xscale("log")
    ax_kl.set_yscale("log")

    fig.suptitle(f"Decomposition metrics vs target entropy ({args.model_path})", fontsize=11)
    fig.tight_layout()

    out_path = out_dir / "entropy_vs_metrics.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
