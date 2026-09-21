"""Token-coloured text figures from the non-target probe, for `report_comp_classification.md`.

    python notes/ci_filter/make_probe_figures.py [--probe <.../nontarget_probe>] [--rows 3]

For each probed component, renders its highest-max-KL sequences token by token, each token shaded
by the KL its subtraction causes at that position, plus one figure comparing the components'
KL distributions over all of the probed text."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

DEFAULT_PROBE = Path(
    "~/out/pod-backup/p-ba5a0c05/analysis/ablations/step_40000/nontarget_probe"
).expanduser()
FIGURES = Path(__file__).parent / "figures"
PER_LINE = 16
"""Tokens per rendered line."""


def _label(key: str) -> str:
    site, component = key.split("|")
    parts = site.split(".")
    return f"L{parts[1]} {'.'.join(parts[2:])} c{component}"


def text_figure(key: str, kl: np.ndarray, rows: list[dict[str, Any]], n_rows: int) -> None:
    """One component: `n_rows` sequences, each token's cell shaded by its KL."""
    n_rows = min(n_rows, kl.shape[0])
    lines_per_row = -(-kl.shape[1] // PER_LINE)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 1.05 * lines_per_row * n_rows), squeeze=False)
    norm = LogNorm(vmin=max(float(kl[:n_rows].min()), 1e-4), vmax=float(kl[:n_rows].max()))
    cmap = plt.get_cmap("magma_r")
    for index in range(n_rows):
        ax = axes[index][0]
        values = kl[index]
        tokens = rows[index]["tokens"]
        grid = np.full((lines_per_row, PER_LINE), np.nan)
        for position, value in enumerate(values):
            grid[position // PER_LINE, position % PER_LINE] = value
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        for position, token in enumerate(tokens[: values.size]):
            line, column = divmod(position, PER_LINE)
            shade = cmap(norm(values[position]))
            text = token.replace("\n", "\\n").replace("$", r"\$")[:10]
            ax.text(
                column,
                line,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if sum(shade[:3]) < 1.5 else "black",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"row {rows[index]['row']}", fontsize=7)
        ax.set_title(f"max KL {rows[index]['max_kl']:.3f}", fontsize=8, loc="left")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), label="KL", pad=0.01
    )
    fig.suptitle(f"{_label(key)} subtracted from the model: per-token KL", fontsize=11)
    out = FIGURES / f"probe_{key.replace('|', '_c').replace('.', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def distribution_figure(probe: Path) -> None:
    summary = list(csv.DictReader((probe / "summary.tsv").open(), delimiter="\t"))
    summary.sort(key=lambda row: -float(row["median_kl"]))
    labels = [f"L{r['layer']} {r['kind']} c{r['component']}" for r in summary]
    y = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(summary) + 2.5))
    axes[0].barh(y, [float(r["median_kl"]) for r in summary], color="tab:blue", label="median")
    axes[0].barh(y, [float(r["p99_kl"]) for r in summary], color="tab:blue", alpha=0.3, label="p99")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KL when subtracted (log)")
    axes[0].legend(fontsize=8)
    axes[1].barh(y, [100 * float(r["top1_flip_frac"]) for r in summary], color="tab:red")
    axes[1].set_xlabel("positions whose argmax changes (%)")
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
    fig.suptitle("Candidate arithmetic-interference components on general text", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "probe_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    ap.add_argument("--rows", type=int, default=3, help="sequences drawn per component")
    ap.add_argument("--components", type=int, default=4, help="components drawn, most active first")
    args = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)
    distribution_figure(args.probe)
    with np.load(args.probe / "heatmap.npz") as saved:
        heat = {key: saved[key] for key in saved.files}
    tokens = json.loads((args.probe / "heatmap_tokens.json").read_text())
    ranked = sorted(heat, key=lambda key: -float(np.max(heat[key])))
    for key in ranked[: args.components]:
        text_figure(key, heat[key], tokens[key], args.rows)


if __name__ == "__main__":
    main()
