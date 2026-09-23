#!/usr/bin/env python
"""Distribution of the pre-RMSNorm logit shift, dual objective vs outputs-only.

    python plot_logit_shift.py <dual dir> <outputs-only dir> <out.png>

Each `<dir>` is a `collect_logit_shift.py` output. The quantity is, per prompt, the
ORIGINAL model's top-1 next token, and that token's logit in the rounded-mask decomposed
model minus its logit in the original model — taken PRE-RMSNORM (the residual leaving the
last block dotted with the unembedding row), so a uniform rescaling of the residual shows up
instead of being normalized away.

Two panels: the pre-norm shift, and the ordinary post-norm shift for reference. A decomposed
model that reproduces the original exactly sits on 0 in both.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DUAL, OUT_ONLY = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(d: Path) -> tuple[dict[str, np.ndarray], dict]:
    table = np.genfromtxt(d / "logit_shift.tsv", delimiter="\t", names=True, encoding="utf-8")
    meta = json.loads((d / "meta.json").read_text())
    return {n: np.asarray(table[n], float) for n in table.dtype.names}, meta


def main() -> None:
    dual_dir, oo_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    runs = []
    for name, d, color in (
        ("Dual objective (-05)", dual_dir, DUAL),
        ("Outputs-only", oo_dir, OUT_ONLY),
    ):
        cols, meta = load(d)
        runs.append((name, cols, meta, color))

    plt.rcParams.update(
        {
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = (
        ("pre", "Pre-RMSNorm logit shift", "decomposed − original (pre-norm)"),
        ("post", "Post-RMSNorm logit shift (reference)", "decomposed − original (post-norm)"),
    )
    for ax, (prefix, title, xlabel) in zip(axes, panels, strict=True):
        shifts = [
            (n, c[f"{prefix}_decomposed"] - c[f"{prefix}_original"], col) for n, c, _, col in runs
        ]
        lo = min(np.percentile(s, 0.5) for _, s, _ in shifts)
        hi = max(np.percentile(s, 99.5) for _, s, _ in shifts)
        bins = np.linspace(lo, hi, 80)
        for name, shift, color in shifts:
            ax.hist(
                shift, bins=bins, histtype="step", lw=1.6, color=color, label=name, density=True
            )
            ax.axvline(np.median(shift), color=color, lw=1, ls=":", zorder=0)
        ax.axvline(0, color=INK_2, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, labelcolor=INK)

    lines = []
    for name, cols, _meta, _ in runs:
        s = cols["pre_decomposed"] - cols["pre_original"]
        q = cols["post_decomposed"] - cols["post_original"]
        lines.append(
            f"{name}: pre-norm median {np.median(s):+.3g}, mean {s.mean():+.3g}, "
            f"5-95% [{np.percentile(s, 5):+.3g}, {np.percentile(s, 95):+.3g}]; "
            f"post-norm median {np.median(q):+.3g}"
        )
    meta0 = runs[0][2]
    fig.suptitle(
        "Logit of the original model's top-1 next token: rounded-mask decomposed vs original",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.005,
        "\n".join(lines)
        + f"\n{meta0['n_prompts']} prompts, step {meta0['step']}, rounded mask = CI > {meta0['threshold']} "
        "(the runs' own `rounded` definition); dotted lines mark each median.",
        fontsize=7.5,
        color=INK_2,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
