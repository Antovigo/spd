#!/usr/bin/env python
"""Inner-activation AB grids of the most active layer-18 `up_proj` components, the
hidden-recon run (-05) on the left of the page and its outputs-only twin on the right.

    python plot_l18_up_proj_inner_grids.py     # reads ~/out/pod-backup/*/ab_grids
                                               # writes plots/l18_up_proj_inner_grids.png

Rows are the slow-eval steps at which a grid was written; each row shows that step's six
most active components side by side. A grid is the component's normalized inner
activation `(x · V_c) / ‖V_c‖` at the answer position, over every `a + b =` prompt with
a, b in 1..100 (a up, b right). "Most active" is the prompt-mean CI of the OUTPUT head at
that position (`mean_ci`), the one CI both runs carry, ranked among the components the
grid saved (only those have grids). Rankings are per step, so a column is a rank, not a
fixed component: the component id sits over each grid.

Inner activations are signed, so each grid uses a diverging scale centred on zero,
symmetric to that grid's own largest magnitude (printed under the id). Magnitudes are not
comparable across grids; the patterns are.

GRID DECODING IS COMPUTE: run it through SLURM, not on a login node — each snapshot is
~100 MB of JSON.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
BACKUP = Path.home() / "out" / "pod-backup"
SITE = "layers.18.mlp.up_proj"
TOP = 6

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Diverging blue <-> red through the reference palette's neutral gray; equal steps per arm
# (the red arm mirrors the blue ramp's lightness around the documented red, #e34948).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#7a1a1a"],
)


def load_grid(path: Path) -> dict[str, Any]:
    text = path.read_text()
    return json.loads(text[text.index("(") + 1 : text.rindex(")")])


def site_grids(doc: dict[str, Any], site: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(saved component ids, output-head mean CI over all C, inner grids `(k, n_a, n_b)`)."""
    (mod,) = [m for m in doc["modules"] if m["name"] == site]
    c, n_a, n_b = mod["C"], doc["n_a"], doc["n_b"]
    mean_ci = np.frombuffer(base64.b64decode(mod["mean_ci"]), np.float32).reshape(-1, c)
    saved = mod["saved"]
    if not saved:
        return [], mean_ci[0], np.zeros((0, n_a, n_b), np.float32)
    inner = np.frombuffer(base64.b64decode(mod["inner"]), np.float16)
    # `[comp, pos, op, a, b]`; one recorded position and one op on these grids.
    inner = inner.reshape(len(saved), -1, 1, n_a, n_b)[:, 0, 0].astype(np.float32)
    return saved, mean_ci[0], inner


def top_components(
    grid_dir: Path, site: str, top: int
) -> dict[int, list[tuple[int, float, np.ndarray]]]:
    """{step: [(component id, mean CI, inner grid)] for the `top` highest-mean-CI saved ones}."""
    out = {}
    for path in sorted(grid_dir.glob("step_*.js"), key=lambda p: int(p.stem.split("_")[1])):
        step = int(path.stem.split("_")[1])
        saved, mean_ci, inner = site_grids(load_grid(path), site)
        order = sorted(range(len(saved)), key=lambda i: -mean_ci[saved[i]])[:top]
        out[step] = [(saved[i], float(mean_ci[saved[i]]), inner[i]) for i in order]
        print(
            f"  {grid_dir.parent.name} step {step}: {len(saved)} saved, top {[saved[i] for i in order]}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dual", type=Path, default=BACKUP / "p-ba5a0c05" / "ab_grids")
    ap.add_argument("--outputs-only", type=Path, default=BACKUP / "p-ba5a0c0f" / "ab_grids")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument(
        "-o", "--out", type=Path, default=HERE / "plots" / "l18_up_proj_inner_grids.png"
    )
    a = ap.parse_args()

    runs = [
        ("Hidden-recon run (-05)", top_components(a.dual, a.site, a.top)),
        ("Outputs-only run", top_components(a.outputs_only, a.site, a.top)),
    ]
    steps = sorted(set(runs[0][1]) | set(runs[1][1]))

    plt.rcParams.update({"font.size": 7, "figure.facecolor": SURFACE})
    n_cols = 2 * a.top + 1  # a spacer column between the two runs
    fig = plt.figure(figsize=(1.1 * n_cols + 0.8, 1.3 * len(steps) + 1.2))
    gs = fig.add_gridspec(
        len(steps),
        n_cols,
        width_ratios=[1] * a.top + [0.35] + [1] * a.top,
        left=0.06,
        right=0.995,
        top=1 - 0.95 / fig.get_figheight(),
        bottom=0.55 / fig.get_figheight(),
        hspace=0.55,
        wspace=0.12,
    )
    for half, (title, per_step) in enumerate(runs):
        col0 = half * (a.top + 1)
        for row, step in enumerate(steps):
            comps = per_step.get(step)
            for rank in range(a.top):
                ax = fig.add_subplot(gs[row, col0 + rank])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#d9d8d4")
                if comps is None or rank >= len(comps):
                    ax.set_facecolor(SURFACE)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if rank == 0:
                        what = "not reached yet" if comps is None else "no components saved"
                        ax.text(0.02, 0.5, what, transform=ax.transAxes, color=INK_2, va="center")
                    elif comps is not None and rank == len(comps):
                        ax.text(
                            0.02,
                            0.5,
                            f"only {len(comps)} saved",
                            transform=ax.transAxes,
                            color=INK_2,
                            va="center",
                        )
                    continue
                cid, mci, grid = comps[rank]
                lim = float(np.abs(grid).max()) or 1.0
                ax.imshow(
                    grid,
                    origin="lower",
                    cmap=DIVERGING,
                    vmin=-lim,
                    vmax=lim,
                    interpolation="nearest",
                    aspect="equal",
                )
                ax.set_title(f"c{cid}  CI {mci:.2f}\n±{lim:.3g}", fontsize=6.5, color=INK, pad=2)
            if half == 0:
                fig.text(
                    0.005,
                    (gs[row, 0].get_position(fig).y0 + gs[row, 0].get_position(fig).y1) / 2,
                    f"step\n{step:,}",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
        x0 = gs[0, col0].get_position(fig).x0
        fig.text(
            x0,
            1 - 0.35 / fig.get_figheight(),
            title,
            fontsize=11,
            fontweight="semibold",
            color=INK,
            va="top",
        )
    fig.suptitle(
        f"{a.site}: inner activations of the {a.top} most active components (by output-head mean CI)",
        x=0.06,
        y=1 - 0.05 / fig.get_figheight(),
        ha="left",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    fig.text(
        0.06,
        0.12 / fig.get_figheight(),
        "Each grid: inner activation (x·V)/‖V‖ at the answer position over a + b = prompts, a (1-100) up, "
        "b (1-100) right. Diverging scale centred on 0 (blue < 0 < red), symmetric to each grid's own ±max; "
        "ranks are per step, so a column is a rank, not a fixed component.",
        fontsize=7,
        color=INK_2,
        va="bottom",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
