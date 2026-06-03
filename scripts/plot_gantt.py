"""Render a per-pool GPU-occupancy timeseries from `extract_gantt_json.py` output.

One lane per pool: a stacked area of GPU-kernel categories (matmul / attention / reduction /
elementwise / memory / other / nccl) across step-relative time. Compute categories are
~mutually exclusive on the stream, so the stack reads as "what this pool is doing each
moment"; the gap to 1 is idle. A tall nccl band = that pool stalling on a peer.

Usage: python scripts/plot_gantt.py <gantt_json> [out_png]
"""

import json
import sys
from pathlib import Path

CREAM, INK, HAIR, MUTED = "#f6f0e2", "#2b2b2b", "#cabfa6", "#6b6253"
PALETTE = {
    "matmul": "#6e1423",  # oxblood — tensor-core GEMM (the productive work)
    "attention": "#9c5b6b",  # plum — flash-attention
    "reduction": "#b9a06b",  # gold — fused norms / reductions
    "elementwise": "#cdb88a",  # tan — pointwise
    "memory": "#a89a86",  # taupe — copies / memset
    "other": "#d9cdb0",  # pale
    "nccl": "#5f7470",  # slate — comm / cross-pool wait (cool ≠ warm compute)
}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.with_suffix(".png")
    payload = json.loads(in_path.read_text())
    pools = payload["pools"]
    cats = payload["categories"]
    step_ms = payload["step_ms"]
    nbins = payload["nbins"]
    t = [i * step_ms / nbins for i in range(nbins)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": HAIR,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        len(pools), 1, figsize=(9.5, 1.55 * len(pools) + 1.4), sharex=True, squeeze=False
    )
    for ax, p in zip(axes[:, 0], pools, strict=True):
        bottom = [0.0] * nbins
        for ci, c in enumerate(cats):
            series = [b[ci] for b in p["bins"]]
            top = [bottom[i] + series[i] for i in range(nbins)]
            ax.fill_between(
                t, bottom, top, step="mid", color=PALETTE.get(c, "#cccccc"), linewidth=0
            )
            bottom = top
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_xlim(0, step_ms)
        ax.set_ylabel(p["pool"], rotation=0, ha="right", va="center", fontsize=12)
        bd = p["by_category_ms"]
        top3 = sorted(((c, bd[c]) for c in cats), key=lambda kv: -kv[1])[:3]
        label = " · ".join(f"{c} {ms:.0f}" for c, ms in top3 if ms > 0)
        ax.text(
            0.995,
            0.9,
            f"{label}  ·  idle {p['idle_ms']:.0f}ms ({p['idle_pct']:.0f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=INK,
        )
    axes[-1, 0].set_xlabel("step-relative time (ms)")
    fig.suptitle(
        f"Per-pool GPU-kernel occupancy across one step ({step_ms:.0f} ms)", fontsize=13, y=0.995
    )
    fig.text(0.5, 0.95, payload["source"], ha="center", fontsize=9, color=MUTED)
    fig.legend(
        handles=[Patch(color=PALETTE[c], label=c) for c in cats],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        frameon=False,
        fontsize=8.5,
        ncol=len(cats),
        columnspacing=1.1,
        handlelength=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
