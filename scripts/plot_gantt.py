"""Render a per-pool GPU-occupancy timeseries from `extract_gantt_json.py` output.

Reads the JSON (per-pool binned compute/nccl fractions over one representative step) and
draws one stacked lane per pool: compute vs NCCL occupancy across step-relative time. NCCL
time is mostly *waiting* on a peer pool, so a tall gold band = that pool stalling.

Usage: python scripts/plot_gantt.py <gantt_json> [out_png]
"""

import json
import sys
from pathlib import Path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix(".png")
    payload = json.loads(in_path.read_text())
    pools = payload["pools"]
    step_ms = payload["step_ms"]
    nbins = payload["nbins"]
    t = [i * step_ms / nbins for i in range(nbins)]  # bin left edge, ms

    cream, ink, oxblood, hair, gold = "#f6f0e2", "#2b2b2b", "#6e1423", "#cabfa6", "#b9a06b"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
            "figure.facecolor": cream,
            "axes.facecolor": cream,
            "savefig.facecolor": cream,
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.edgecolor": hair,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        len(pools), 1, figsize=(9, 1.5 * len(pools) + 1.2), sharex=True, squeeze=False
    )
    for ax, p in zip(axes[:, 0], pools, strict=True):
        cf = [b[0] for b in p["bins"]]
        nf = [b[1] for b in p["bins"]]
        ax.fill_between(t, 0, nf, step="mid", color=gold, alpha=0.85, linewidth=0)
        ax.fill_between(t, 0, cf, step="mid", color=oxblood, alpha=0.85, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_xlim(0, step_ms)
        ax.set_ylabel(p["pool"], rotation=0, ha="right", va="center", fontsize=12)
        ax.text(
            0.995,
            0.88,
            f"compute {p['compute_ms']:.0f} · nccl {p['nccl_ms']:.0f} · idle {p['idle_ms']:.0f}ms ({p['idle_pct']:.0f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=ink,
        )
    axes[-1, 0].set_xlabel("step-relative time (ms)")
    fig.suptitle(f"Per-pool GPU occupancy across one step ({step_ms:.0f} ms)", fontsize=13, y=0.995)
    fig.text(0.5, 0.945, payload["source"], ha="center", fontsize=9, color="#6b6253")
    fig.legend(
        handles=[
            Patch(color=oxblood, alpha=0.85, label="compute"),
            Patch(color=gold, alpha=0.85, label="NCCL (mostly cross-pool wait)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
