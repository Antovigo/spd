#!/usr/bin/env python
"""Decomposed vs original inner activation of every alive component, one panel per matrix kind.

    python plot_inner_scatter_all_alive.py <harvest dir> <out.png> [title suffix]

`<harvest dir>` is a `collect_inner_scatter.py` output (inner_scatter.tsv + meta.json): one
row per component that is ever ON, carrying its inner activation `x · V_c` at one randomly
drawn on-entry in the original model (x) and in the decomposed model (y). Dead components are
absent from the harvest, so nothing is dropped here.

Same layout as `notes/ci_filter/plot_inner_scatter.py` — seven kind panels, points coloured by
layer, identity line, equal aspect — so the filter-alive and all-alive figures read alike.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KINDS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def main() -> None:
    harvest, out = Path(sys.argv[1]), Path(sys.argv[2])
    suffix = sys.argv[3] if len(sys.argv) > 3 else harvest.name
    meta = json.loads((harvest / "meta.json").read_text())
    table = np.genfromtxt(
        harvest / "inner_scatter.tsv", delimiter="\t", names=True, dtype=None, encoding="utf-8"
    )
    kind, layer = table["kind"], table["layer"].astype(int)
    x, y = table["original"].astype(float), table["decomposed"].astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    norm = plt.Normalize(0, meta["n_layer"] - 1)
    points = None
    for ax, k in zip(axes.flat, KINDS, strict=False):
        sel = (kind == k) & np.isfinite(x) & np.isfinite(y)
        if not sel.any():
            ax.set(title=f"{k}  (none alive)")
            continue
        lo = min(x[sel].min(), y[sel].min())
        hi = max(x[sel].max(), y[sel].max())
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1, zorder=1)
        order = np.argsort(layer[sel])
        points = ax.scatter(
            x[sel][order],
            y[sel][order],
            c=layer[sel][order],
            cmap="viridis",
            norm=norm,
            s=14,
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )
        ax.set(xlim=lims, ylim=lims, aspect="equal", title=f"{k}  (n={int(sel.sum())})")
        ax.set_xlabel("original model")
        ax.set_ylabel("decomposed model")
        ax.grid(alpha=0.25, lw=0.5)
    axes.flat[-1].set_visible(False)
    if points is not None:
        fig.colorbar(points, ax=axes, label="layer", shrink=0.6)
    fig.suptitle(
        f"Inner activation x·V_c at one random (prompt, position) where the component is on — {suffix}, "
        f"ALL {meta['n_alive']} alive of {meta['n_components']} components (no CI filter), "
        f"{meta['n_prompts']} prompts, step {meta['step']}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
