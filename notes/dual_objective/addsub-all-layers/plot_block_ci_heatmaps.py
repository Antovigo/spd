#!/usr/bin/env python
"""Per-block CI heatmaps, block x training step, the hidden-recon run (-05) beside its
outputs-only twin on ONE shared log colour scale. Two figures, because the two quantities
live at different cadences:

    python plot_block_ci_heatmaps.py     # writes plots/block_alive_l0_heatmap{,_linear}.png
                                         #        plots/block_total_ci_heatmap{,_linear}.png

1. ALIVE COMPONENTS (L0), fast-eval cadence (every 500 steps). `eval/l0/0.0_<site>` is the
   per-token count of components with CI > 0 on the target stream, output head; a block's
   value is the sum over its seven sites. This is the only per-site CI quantity logged at
   fast-eval cadence — it COUNTS components, it does not sum their CI.
2. TOTAL CI, slow-eval cadence (every 4000 steps). From the AB grids: each site's
   output-head `mean_ci` (prompt-mean CI per component at the answer position, over every
   a + b = prompt with a, b in 1..100), summed over the block's components. This is the
   sum of CI values per prompt — but only on the addsub `+` grid, and only when a grid was
   written.

Each figure comes in a LOG and a LINEAR colour scale, both shared by the two runs and
spanning the full range, so nothing clips. The log scale runs from the smallest positive to
the largest value; it cannot place zero, so zero cells are drawn in neutral gray and
counted in the footnote. The linear scale runs from 0 to the largest value, so colour stays
proportional to magnitude and zero needs no special case. Steps a run has not reached yet
are left blank.

GRID DECODING IS COMPUTE: run it through SLURM, not on a login node.
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
BACKUP = Path.home() / "out" / "pod-backup"
N_BLOCKS = 32
SITES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

SURFACE, INK, INK_2, ZERO = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3"
RUNS = (("Hidden-recon run (-05)", "p-ba5a0c05"), ("Outputs-only run", "p-ba5a0c0f"))


def l0_by_block(metrics: Path) -> tuple[list[int], np.ndarray]:
    rows: dict[int, dict[str, Any]] = {}
    for line in metrics.open():
        d = json.loads(line)
        if "step" in d:
            rows.setdefault(d["step"], {}).update(d)
    key = "eval/l0/0.0_layers.{b}.{s}"
    steps = sorted(s for s, r in rows.items() if key.format(b=0, s=SITES[0]) in r)
    grid = np.array(
        [
            [sum(rows[s][key.format(b=b, s=x)] for x in SITES) for s in steps]
            for b in range(N_BLOCKS)
        ]
    )
    return steps, grid


def total_ci_by_block(grid_dir: Path) -> tuple[list[int], np.ndarray]:
    steps, cols = [], []
    for path in sorted(grid_dir.glob("step_*.js"), key=lambda p: int(p.stem.split("_")[1])):
        text = path.read_text()
        doc = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        per_block = np.zeros(N_BLOCKS)
        for mod in doc["modules"]:
            block = int(mod["name"].split(".")[1])
            mean_ci = np.frombuffer(base64.b64decode(mod["mean_ci"]), np.float32)
            per_block[block] += float(mean_ci.sum())  # one recorded position: (1, C)
        steps.append(int(path.stem.split("_")[1]))
        cols.append(per_block)
        print(f"  {grid_dir.parent.name} step {steps[-1]}: total {per_block.sum():.1f}")
    return steps, np.stack(cols, axis=1)


def edges(steps: list[int]) -> np.ndarray:
    s = np.asarray(steps, float)
    gap = np.diff(s).min() if len(s) > 1 else 500.0
    mids = (s[:-1] + s[1:]) / 2
    return np.concatenate([[s[0] - gap / 2], mids, [s[-1] + gap / 2]])


def render(
    data: list[tuple[str, list[int], np.ndarray]],
    label: str,
    title: str,
    cadence: str,
    out: Path,
    log: bool,
) -> None:
    top = float(max(g.max() for _, _, g in data))
    cmap = plt.get_cmap("viridis").copy()
    n_zero = sum(int((g <= 0).sum()) for _, _, g in data)
    if log:
        positive = np.concatenate([g[g > 0] for _, _, g in data])
        norm = LogNorm(vmin=float(positive.min()), vmax=top)
        cmap.set_bad(ZERO)
    else:
        norm = Normalize(vmin=0.0, vmax=top)

    plt.rcParams.update({"font.size": 9, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for ax, (name, steps, grid) in zip(axes, data, strict=True):
        mesh = ax.pcolormesh(
            edges(steps),
            np.arange(N_BLOCKS + 1) - 0.5,
            np.ma.masked_less_equal(grid, 0) if log else grid,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
        ax.set_xlim(0, 40_000)
        ax.set_ylim(-0.5, N_BLOCKS - 0.5)
        ax.set_yticks(range(0, N_BLOCKS, 4))
        ax.set_title(name, loc="left", color=INK, fontsize=11, fontweight="semibold")
        ax.set_xlabel(f"Training step ({cadence})", color=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        right = float(edges(steps)[-1])
        if steps[-1] < 40_000:
            ax.text(
                right + 700,
                N_BLOCKS / 2,
                "not reached yet",
                rotation=90,
                va="center",
                color=INK_2,
                fontsize=8,
            )
    axes[0].set_ylabel("Block", color=INK_2)
    # Its own axes, placed after the margins are set: a colorbar that steals space from
    # `axes` is undone by the later subplots_adjust and ends up over the right panel.
    fig.subplots_adjust(left=0.06, right=0.88, top=0.88, bottom=0.12, wspace=0.06)
    cbar = fig.colorbar(mesh, cax=fig.add_axes((0.9, 0.12, 0.013, 0.76)))
    scale = "log" if log else "linear"
    cbar.set_label(f"{label} ({scale} scale, shared by both runs)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(title, x=0.06, ha="left", color=INK, fontsize=12, fontweight="semibold")
    if log:
        note = (
            f"Log colour scale spanning the full positive range across both runs: "
            f"{norm.vmin:.3g} to {norm.vmax:.3g}, no clipping."
        )
        note += (
            f" {n_zero} zero cell(s), which a log scale cannot place, are drawn gray."
            if n_zero
            else " No zero cells."
        )
    else:
        note = (
            f"Linear colour scale from 0 to the largest value across both runs ({norm.vmax:.3g}), "
            "no clipping."
        )
    fig.text(0.06, 0.012, note, fontsize=7.5, color=INK_2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (range {norm.vmin:.3g}..{norm.vmax:.3g}, {n_zero} zero cells)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backup", type=Path, default=BACKUP)
    ap.add_argument("-o", "--out-dir", type=Path, default=HERE / "plots")
    a = ap.parse_args()

    l0 = [(name, *l0_by_block(a.backup / rid / "metrics.jsonl")) for name, rid in RUNS]
    total_ci = [(name, *total_ci_by_block(a.backup / rid / "ab_grids")) for name, rid in RUNS]
    for log, suffix in ((True, ""), (False, "_linear")):
        render(
            l0,
            "Alive components per block (L0, CI > 0)",
            "Alive components per block, target stream, output head",
            "fast eval, every 500",
            a.out_dir / f"block_alive_l0_heatmap{suffix}.png",
            log,
        )
        render(
            total_ci,
            "Total CI per block (sum of mean CI)",
            "Total CI per block on the addsub grid, output head",
            "slow eval, every 4000",
            a.out_dir / f"block_total_ci_heatmap{suffix}.png",
            log,
        )


if __name__ == "__main__":
    main()
