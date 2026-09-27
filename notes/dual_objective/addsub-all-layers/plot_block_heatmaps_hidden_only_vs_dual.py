#!/usr/bin/env python
"""Per-block heatmaps, block x training step, for the three arms of the hidden-only experiment
on ONE shared colour scale per figure.

    python plot_block_heatmaps_hidden_only_vs_dual.py                   # L0 only (the default)
    python plot_block_heatmaps_hidden_only_vs_dual.py --with-total-ci   # and the CI mass

The default writes `block_alive_l0_heatmap{,_linear}.png` from the metrics files alone. The
total-CI figures need every AB-grid snapshot decoded, so they are behind `--with-total-ci`.

WHAT "IMP-MIN PER BLOCK" IS AND IS NOT. Importance-minimality is logged as ONE scalar per pass,
never per site, so there is no per-block imp-min series to draw. Its stand-in is TOTAL CI per
block: imp-min is a penalty on CI, so the CI a block still carries is what the penalty has and
has not removed. The two figures here are therefore the count (L0) and the mass (total CI) of
the same thing, at the two cadences they are available at:

1. ALIVE COMPONENTS (L0), fast-eval cadence (every 500 steps). `eval/hidden_ci/l0/0.0_<site>` is
   the per-token count of components with CI > 0 on the target stream; a block's value is the sum
   over its seven sites. It COUNTS components, it does not sum their CI.
2. TOTAL CI, slow-eval cadence (every 4000 steps). From the AB grids: each site's hidden-head
   prompt-mean CI per component at the answer position, over every `a + b =` prompt with a, b in
   1..100, summed over the block's components. Stored for EVERY component, not only the saved
   ones, so the sums are comparable across runs whose saved sets differ.

BOTH READ THE HIDDEN HEAD, unlike the outputs-only version of this figure. hidden-only's output
head is frozen at `zero_init_readout` (CI 0.5 for every subcomponent), which would paint every
block fully alive at every step in that column and set the shared colour scale from an artefact.
The hidden head is the one head all three runs train.

Each figure comes in a LOG and a LINEAR colour scale, both shared by all three runs and spanning
the full range, so nothing clips. The log scale cannot place zero, so zero cells are drawn in
neutral gray and counted in the footnote. Steps a run never reached are left blank.

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
# (panel title, run id, ab_grids dir name). hidden-only reads its hidden-CI rewrite: the raw
# snapshots are 6.7 GB each only because its frozen output head saved every subcomponent.
RUNS = (
    ("Dual objective (-05)", "p-ba5a0c05", "ab_grids"),
    ("Hidden-only", "p-ba5a0c0d", "ab_grids_hidden"),
    ("Hidden-only → dual at 28k", "p-ba5a0c0e", "ab_grids"),
)


def l0_by_block(metrics: Path) -> tuple[list[int], np.ndarray]:
    rows: dict[int, dict[str, Any]] = {}
    for line in metrics.open():
        d = json.loads(line)
        if "step" in d:
            rows.setdefault(d["step"], {}).update(d)
    key = "eval/hidden_ci/l0/0.0_layers.{b}.{s}"
    steps = sorted(s for s, r in rows.items() if key.format(b=0, s=SITES[0]) in r)
    grid = np.array(
        [
            [sum(rows[s][key.format(b=b, s=x)] for x in SITES) for s in steps]
            for b in range(N_BLOCKS)
        ]
    )
    return steps, grid


def total_ci_by_block(grid_dir: Path) -> tuple[list[int], np.ndarray]:
    """Hidden-head mean CI summed per block. `mean_ci_hidden` in a raw dual snapshot; in a
    snapshot rewritten by `filter_ab_grids_hidden.py` the hidden values sit in the historical
    `mean_ci` slot (declared by `source_role`) and still cover every component."""
    steps, cols = [], []
    for path in sorted(grid_dir.glob("step_*.js"), key=lambda p: int(p.stem.split("_")[1])):
        text = path.read_text()
        doc = json.loads(text[text.index("(") + 1 : text.rindex(")")])
        hidden_slot = doc.get("source_role") == "hidden"
        per_block = np.zeros(N_BLOCKS)
        for mod in doc["modules"]:
            block = int(mod["name"].split(".")[1])
            raw = mod["mean_ci"] if hidden_slot else mod.get("mean_ci_hidden")
            if raw is None:
                raise SystemExit(f"{path.name}: no hidden-head mean CI (single-role snapshot?)")
            per_block[block] += float(np.frombuffer(base64.b64decode(raw), np.float32).sum())
        steps.append(int(path.stem.split("_")[1]))
        cols.append(per_block)
        print(f"  {grid_dir.parent.name} step {steps[-1]}: total {per_block.sum():.1f}", flush=True)
    return steps, np.stack(cols, axis=1)


def edges(steps: list[int]) -> np.ndarray:
    s = np.asarray(steps, float)
    gap = np.diff(s).min() if len(s) > 1 else 4000.0
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
    fig, axes = plt.subplots(1, len(data), figsize=(17, 6.2), sharey=True)
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
        if steps[0] > 0:
            ax.text(
                steps[0] / 2, N_BLOCKS / 2, "resumed at 28k",
                rotation=90, va="center", ha="center", color=INK_2, fontsize=8,
            )  # fmt: skip
        if steps[-1] < 40_000:
            ax.text(
                float(edges(steps)[-1]) + 700, N_BLOCKS / 2, "run stopped",
                rotation=90, va="center", color=INK_2, fontsize=8,
            )  # fmt: skip
    axes[0].set_ylabel("Block", color=INK_2)
    # Its own axes, placed after the margins are set: a colorbar that steals space from `axes`
    # is undone by the later subplots_adjust and ends up over the right panel.
    fig.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.12, wspace=0.06)
    cbar = fig.colorbar(mesh, cax=fig.add_axes((0.92, 0.12, 0.011, 0.76)))
    scale = "log" if log else "linear"
    cbar.set_label(f"{label} ({scale} scale, shared by all three runs)", color=INK_2)
    cbar.outline.set_visible(False)
    fig.suptitle(title, x=0.05, ha="left", color=INK, fontsize=12, fontweight="semibold")
    if log:
        note = (
            f"Log colour scale spanning the full positive range across all three runs: "
            f"{norm.vmin:.3g} to {norm.vmax:.3g}, no clipping."
        )
        note += (
            f" {n_zero} zero cell(s), which a log scale cannot place, are drawn gray."
            if n_zero
            else " No zero cells."
        )
    else:
        note = (
            f"Linear colour scale from 0 to the largest value across all three runs "
            f"({norm.vmax:.3g}), no clipping."
        )
    note += (
        "\nHidden head throughout: hidden-only's output head is frozen at CI 0.5, which would "
        "paint its whole panel alive and set the shared scale from an artefact."
    )
    fig.text(0.05, 0.012, note, fontsize=7.5, color=INK_2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (range {norm.vmin:.3g}..{norm.vmax:.3g}, {n_zero} zero cells)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backup", type=Path, default=BACKUP)
    ap.add_argument("-o", "--out-dir", type=Path, default=HERE / "plots" / "hidden_only_vs_dual")
    ap.add_argument(
        "--with-total-ci",
        action="store_true",
        help="also draw the total-CI figures, which decode every AB-grid snapshot (minutes of "
        "CPU and GB of RAM). Off by default: L0 answers the same question from metrics alone.",
    )
    a = ap.parse_args()

    l0 = [(name, *l0_by_block(a.backup / rid / "metrics.jsonl")) for name, rid, _ in RUNS]
    for log, suffix in ((True, ""), (False, "_linear")):
        render(
            l0,
            "Alive components per block (L0, CI > 0)",
            "Alive components per block, target stream, hidden head",
            "fast eval, every 500",
            a.out_dir / f"block_alive_l0_heatmap{suffix}.png",
            log,
        )
    if not a.with_total_ci:
        return
    total_ci = [(name, *total_ci_by_block(a.backup / rid / grids)) for name, rid, grids in RUNS]
    for log, suffix in ((True, ""), (False, "_linear")):
        render(
            total_ci,
            "Total CI per block (sum of mean CI)",
            "Total CI per block on the addsub grid, hidden head "
            "(the per-block stand-in for imp-min, which is logged per pass only)",
            "slow eval, every 4000",
            a.out_dir / f"block_total_ci_heatmap{suffix}.png",
            log,
        )


if __name__ == "__main__":
    main()
