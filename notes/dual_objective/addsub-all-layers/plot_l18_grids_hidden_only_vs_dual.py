#!/usr/bin/env python
"""Inner-activation AB grids of the most active layer-18 `up_proj` components, for the three
arms of the hidden-only experiment side by side on the page.

    python plot_l18_grids_hidden_only_vs_dual.py   # reads ~/out/pod-backup/*/ab_grids
                                                   # writes plots/hidden_only_vs_dual/

Columns are the three runs: the dual objective (-05, p-ba5a0c05), hidden-only (p-ba5a0c0d,
stopped at 36900) and the hybrid that resumes the dual objective at step 28000 (p-ba5a0c0e).
Rows are the slow-eval steps at which any run wrote a grid; each cell shows that run's ten most
active components at that step, nothing else drawn on them. A run that had not reached a step,
or had already stopped, is marked rather than left ambiguous.

RANKING IS BY THE HIDDEN HEAD's prompt-mean CI (`mean_ci_hidden`), not the output head's. This
differs from the outputs-only montage on purpose: hidden-only's OUTPUT head is frozen at
`zero_init_readout`, CI 0.5 for every subcomponent, so ranking by it would return an arbitrary
ten. The hidden head is the one head all three runs train, so it is the only ranking that means
the same thing in every column. The top ten are chosen per run per step, then laid out in
COMPONENT-INDEX order, so a component that stays in the top ten keeps roughly its place.

A grid is the component's normalized inner activation `(x · V_c) / ‖V_c‖` at the answer
position, over every `a + b =` prompt with a, b in 1..100 (a up, b right). Inner activations are
signed, so each grid uses a diverging scale centred on zero, symmetric to that grid's own
largest magnitude. Magnitudes are not comparable across grids; the patterns are.

CANDIDATE POOLS DIFFER SLIGHTLY between columns and it does not matter here. Only components
the snapshot SAVED have grids, and the floor is a max over the two heads: for -05 and the hybrid
that is ~1,100 components, while hidden-only's frozen output head admits all 124,928. The top
ten by hidden CI sit far above the 0.05 floor in every column, so nothing that would rank is
missing from either pool.

GRID DECODING IS COMPUTE: run it through SLURM, not on a login node. hidden-only is read from its
`ab_grids_hidden/` rewrite rather than its raw snapshots -- see `--hidden-only`.
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
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#7a1a1a"],
)


def load_grid(path: Path) -> dict[str, Any]:
    text = path.read_text()
    return json.loads(text[text.index("(") + 1 : text.rindex(")")])


def site_grids(doc: dict[str, Any], site: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(saved component ids, HIDDEN-head mean CI over all C, inner grids `(k, n_a, n_b)`).

    `mean_ci_hidden` is the hidden head's vector in a dual snapshot. A snapshot rewritten by
    `filter_ab_grids_hidden.py` carries the hidden values in the historical `mean_ci` slot and
    declares `source_role: "hidden"`, so both layouts are accepted."""
    (mod,) = [m for m in doc["modules"] if m["name"] == site]
    c, n_a, n_b = mod["C"], doc["n_a"], doc["n_b"]
    if "mean_ci_hidden" in mod:
        raw = mod["mean_ci_hidden"]
    elif doc.get("source_role") == "hidden":
        raw = mod["mean_ci"]
    else:
        raise SystemExit(f"{site}: no hidden-head mean CI in this snapshot (single-role?)")
    mean_ci = np.frombuffer(base64.b64decode(raw), np.float32).reshape(-1, c)
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
    """{step: [(component id, hidden mean CI, inner grid)] for the `top` highest-CI saved ones}."""
    out = {}
    for path in sorted(grid_dir.glob("step_*.js"), key=lambda p: int(p.stem.split("_")[1])):
        step = int(path.stem.split("_")[1])
        saved, mean_ci, inner = site_grids(load_grid(path), site)
        chosen = sorted(range(len(saved)), key=lambda i: -mean_ci[saved[i]])[:top]
        order = sorted(chosen, key=lambda i: saved[i])  # select by CI, display by index
        out[step] = [(saved[i], float(mean_ci[saved[i]]), inner[i]) for i in order]
        print(
            f"  {grid_dir.parent.name} step {step}: {len(saved)} saved, "
            f"top {[saved[i] for i in order]}",
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dual", type=Path, default=BACKUP / "p-ba5a0c05" / "ab_grids")
    # The HIDDEN-CI rewrite, not the raw snapshots: hidden-only's raw grids are 6.7 GB each
    # because its frozen output head saved all 124,928 subcomponents, and those extra columns
    # carry no information. `filter_ab_grids_hidden.py` drops them and re-applies the floor to
    # the hidden head, which is the head this montage ranks by anyway. The top ten sit far above
    # the 0.05 floor, so the ranking is unchanged -- only ~60x cheaper to decode.
    ap.add_argument("--hidden-only", type=Path, default=BACKUP / "p-ba5a0c0d" / "ab_grids_hidden")
    ap.add_argument("--hybrid", type=Path, default=BACKUP / "p-ba5a0c0e" / "ab_grids")
    ap.add_argument("--site", default=SITE)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=HERE / "plots" / "hidden_only_vs_dual" / "l18_up_proj_inner_grids.png",
    )
    a = ap.parse_args()

    runs = [
        ("Dual objective (-05)", top_components(a.dual, a.site, a.top), None),
        ("Hidden-only", top_components(a.hidden_only, a.site, a.top), None),
        ("Hidden-only → dual at 28k", top_components(a.hybrid, a.site, a.top), 28_000),
    ]
    steps = sorted({s for _, per_step, _ in runs for s in per_step})

    plt.rcParams.update({"font.size": 8, "figure.facecolor": SURFACE})
    # Placed in inches, so the gap between maps is a fixed few pixels at any page size.
    cell, gap, spacer = 0.72, 0.03, 0.35  # 0.03 in = 4.5 px at 150 dpi
    left, right, head, foot = 0.72, 0.1, 0.55, 0.42
    col_w = a.top * cell + (a.top - 1) * gap
    width = left + len(runs) * col_w + (len(runs) - 1) * spacer + right
    height = head + len(steps) * cell + (len(steps) - 1) * gap + foot
    fig = plt.figure(figsize=(width, height))

    def box(x_in: float, y_top_in: float) -> tuple[float, float, float, float]:
        return (x_in / width, 1 - (y_top_in + cell) / height, cell / width, cell / height)

    for col, (title, per_step, start) in enumerate(runs):
        x0 = left + col * (col_w + spacer)
        fig.text(
            x0 / width, 1 - 0.12 / height, title,
            fontsize=11, fontweight="semibold", color=INK, va="top",
        )  # fmt: skip
        last = max(per_step) if per_step else 0
        for row, step in enumerate(steps):
            y = head + row * (cell + gap)
            comps = per_step.get(step)
            if comps is None:
                why = "not started yet" if start is not None and step < start else "run stopped"
                fig.text(
                    (x0 + 0.05) / width, 1 - (y + cell / 2) / height,
                    why if step > last or start is not None else "no grid",
                    color=INK_2, va="center",
                )  # fmt: skip
                continue
            for rank, (_, _, grid) in enumerate(comps):
                ax = fig.add_axes(box(x0 + rank * (cell + gap), y))
                lim = float(np.abs(grid).max()) or 1.0
                ax.imshow(
                    grid, origin="lower", cmap=DIVERGING, vmin=-lim, vmax=lim,
                    interpolation="nearest", aspect="auto",
                )  # fmt: skip
                ax.set_axis_off()
            if col == 0:
                fig.text(
                    0.08 / width, 1 - (y + cell / 2) / height, f"step\n{step:,}",
                    ha="left", va="center", color=INK,
                )  # fmt: skip
    fig.text(
        left / width,
        0.1 / height,
        f"{a.site}: top {a.top} components per run per step by HIDDEN-head mean CI (the one head "
        "all three runs train), shown in component-index order.\nEach map: inner activation "
        "(x·V)/‖V‖ at the answer position, a (1-100) up, b (1-100) right; diverging scale centred "
        "on 0 (blue < 0 < red), each map scaled to its own ±max.\nhidden-only's output head is "
        "frozen at CI 0.5, so ranking by the output head would return an arbitrary ten there.",
        fontsize=7,
        color=INK_2,
        va="bottom",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
