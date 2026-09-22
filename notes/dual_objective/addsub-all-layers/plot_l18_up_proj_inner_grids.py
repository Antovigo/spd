#!/usr/bin/env python
"""Inner-activation AB grids of the most active layer-18 `up_proj` components, the
hidden-recon run (-05) on the left of the page and its outputs-only twin on the right.

    python plot_l18_up_proj_inner_grids.py     # reads ~/out/pod-backup/*/ab_grids
                                               # writes plots/l18_up_proj_inner_grids.png

Rows are the slow-eval steps at which a grid was written; each row shows that step's ten
most active components side by side, nothing else drawn on them. A grid is the component's normalized inner
activation `(x · V_c) / ‖V_c‖` at the answer position, over every `a + b =` prompt with
a, b in 1..100 (a up, b right). "Most active" is the prompt-mean CI of the OUTPUT head at
that position (`mean_ci`), the one CI both runs carry, ranked among the components the
grid saved (only those have grids). Rankings are per step, so a column is a rank, not a
fixed component. A row with fewer than ten saved components leaves the rest blank.

Inner activations are signed, so each grid uses a diverging scale centred on zero,
symmetric to that grid's own largest magnitude. Magnitudes are not comparable across
grids; the patterns are.

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
TOP = 10

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

    plt.rcParams.update({"font.size": 8, "figure.facecolor": SURFACE})
    # Placed in inches, so the gap between maps is a fixed few pixels at any page size.
    cell, gap, spacer = 0.72, 0.03, 0.35  # 0.03 in = 4.5 px at 150 dpi
    left, right, head, foot = 0.72, 0.1, 0.55, 0.3
    half_w = a.top * cell + (a.top - 1) * gap
    width = left + 2 * half_w + spacer + right
    height = head + len(steps) * cell + (len(steps) - 1) * gap + foot
    fig = plt.figure(figsize=(width, height))

    def box(x_in: float, y_top_in: float) -> tuple[float, float, float, float]:
        return (x_in / width, 1 - (y_top_in + cell) / height, cell / width, cell / height)

    for half, (title, per_step) in enumerate(runs):
        x0 = left + half * (half_w + spacer)
        fig.text(
            x0 / width,
            1 - 0.12 / height,
            title,
            fontsize=11,
            fontweight="semibold",
            color=INK,
            va="top",
        )
        for row, step in enumerate(steps):
            y = head + row * (cell + gap)
            comps = per_step.get(step)
            if comps is None:
                fig.text(
                    (x0 + 0.05) / width,
                    1 - (y + cell / 2) / height,
                    "not reached yet",
                    color=INK_2,
                    va="center",
                )
                continue
            for rank, (_, _, grid) in enumerate(comps):
                ax = fig.add_axes(box(x0 + rank * (cell + gap), y))
                lim = float(np.abs(grid).max()) or 1.0
                ax.imshow(
                    grid,
                    origin="lower",
                    cmap=DIVERGING,
                    vmin=-lim,
                    vmax=lim,
                    interpolation="nearest",
                    aspect="auto",
                )
                ax.set_axis_off()
            if half == 0:
                fig.text(
                    0.08 / width,
                    1 - (y + cell / 2) / height,
                    f"step\n{step:,}",
                    ha="left",
                    va="center",
                    color=INK,
                )
    fig.text(
        left / width,
        0.1 / height,
        f"{a.site}: top {a.top} components per step by output-head mean CI. Each map: inner "
        "activation (x·V)/‖V‖ at the answer position, a (1-100) up, b (1-100) right; diverging "
        "scale centred on 0 (blue < 0 < red), each map scaled to its own ±max.",
        fontsize=7,
        color=INK_2,
        va="bottom",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
