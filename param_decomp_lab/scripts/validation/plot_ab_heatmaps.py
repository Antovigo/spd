"""Plot CI over the (a, b) operand grid for `a<op>b=` prompts, from a per-position JSON.

For each token position, writes one figure whose subplots are laid out as a grid: matrices
down the rows, subcomponents across the columns. Each subplot is an `a`-by-`b` heatmap
(x = a, y = b, both 1..N, equal-scaled) coloured by that subcomponent's lower-leaky CI on
the prompt `a<op>b=` at this position. `--op` selects the arithmetic operator (`+`, `-`,
`*`, or `×`), so subtraction is `--op=-` and the ×-symbol multiplication runs are `--op=×`.
A subcomponent is shown at a position when its MEAN lower-leaky CI across the displayed
prompts exceeds `--ci-thr` (CI below the source JSON's collection threshold, ~0.1, is
absent and counts as 0), so the column set is per-position. `--module-grep` restricts to
modules whose name contains a substring (e.g. `layers.18.mlp` to plot only one layer's
MLP of a full-network run). CPU-only — no model is loaded; reads the
`find_alive_components` per-position JSON.

Usage:
    python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps <per_position_json> \
        [--op=+] [--ci-thr=0.01] [--grep=SUBSTRING] [--module-grep=SUBSTRING] [--output-dir=PATH]

Output: `<run_dir>/analysis/ab_heatmaps_<add|sub|mult>/position_<pos>.png` (one per position).
"""

import json
import re
from pathlib import Path
from typing import Any

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    analysis_dir,
    parse_module_name,
    run_dir_of_dataset,
)

# prompt -> position(str) -> module -> [{"component": int, "ci": float}]
PerPosition = dict[str, dict[str, dict[str, list[dict[str, Any]]]]]
_OP_LABEL = {"+": "add", "-": "sub", "*": "mult", "×": "mult"}
_TILE_IN = 0.5  # square side of each a×b tile, inches
_COL_GAP_IN = 0.12  # horizontal gap between tiles
_ROW_GAP_IN = 0.62  # vertical gap between matrix rows (room for the big facet labels)
# Reserved inches. No right colorbar (it now sits horizontally along the bottom), so the
# right margin is thin and the bottom is tall enough for ticks + axis title + colorbar.
_MARGIN = {"left": 1.05, "right": 0.35, "top": 0.85, "bottom": 1.05}
_CBAR_LEN_IN = 2.2  # horizontal colorbar length, ~matching the old vertical bar
_TITLE_FS, _FACET_FS, _LABEL_FS, _AXIS_FS, _TICK_FS = 11, 9.75, 6.5, 6.3, 4.5


def _parse_ab(
    data: PerPosition, op: str, grep: str | None
) -> tuple[dict[str, tuple[int, int]], int, int]:
    """Map each `a<op>b=` prompt to `(a, b)` (keeping only those containing `grep`); also
    return `(a_max, b_max)`."""
    pattern = re.compile(rf"^(\d+){re.escape(op)}(\d+)=$")
    ab: dict[str, tuple[int, int]] = {}
    for prompt in data:
        if grep is not None and grep not in prompt:
            continue
        match = pattern.match(prompt)
        if match is not None:
            ab[prompt] = (int(match.group(1)), int(match.group(2)))
    assert ab, f"no prompts of the form 'a{op}b=' (grep={grep!r}) found in the JSON"
    a_max = max(a for a, _ in ab.values())
    b_max = max(b for _, b in ab.values())
    return ab, a_max, b_max


def _all_modules(data: PerPosition) -> list[str]:
    """Every decomposed module appearing in the JSON, ordered by `(layer, matrix)`."""
    modules = {m for pp in data.values() for pm in pp.values() for m in pm}
    return sorted(modules, key=parse_module_name)


def _build_grids(
    data: PerPosition,
    ab: dict[str, tuple[int, int]],
    a_max: int,
    b_max: int,
    keep_modules: set[str],
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """pos -> module -> component -> [b_max, a_max] CI grid (filled lazily, 0 where inactive).

    Only `keep_modules` are materialised — skipping the rest keeps memory bounded on
    full-network runs (dozens of matrices) where the caller plots one module subset.
    """
    grids: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for prompt, (a, b) in ab.items():
        for pos, per_module in data[prompt].items():
            per_pos = grids.setdefault(pos, {})
            for module, comps in per_module.items():
                if module not in keep_modules:
                    continue
                per_mod = per_pos.setdefault(module, {})
                for entry in comps:
                    grid = per_mod.get(entry["component"])
                    if grid is None:
                        grid = np.zeros((b_max, a_max))
                        per_mod[entry["component"]] = grid
                    grid[b - 1, a - 1] = entry["ci"]
    return grids


def _plot_position(
    modules: list[str],
    alive: dict[str, list[int]],
    grids_at_pos: dict[str, dict[int, np.ndarray]],
    a_max: int,
    b_max: int,
    out_path: Path,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    title: str,
) -> None:
    """Render one matrix×subcomponent grid of `a×b` heatmaps. Styling (colormap, limits,
    colorbar label, suptitle) is caller-supplied so the same layout serves both the CI and
    inner-activation views."""
    n_rows = len(modules)
    n_cols = max(len(alive[m]) for m in modules)

    # Reserve fixed inch margins so square tiles sit in a tight grid regardless of how many
    # subcomponents a position has; explicit row/col gaps control the breathing room.
    left, right, top, bottom = _MARGIN["left"], _MARGIN["right"], _MARGIN["top"], _MARGIN["bottom"]
    plot_w = n_cols * _TILE_IN + (n_cols - 1) * _COL_GAP_IN
    plot_h = n_rows * _TILE_IN + (n_rows - 1) * _ROW_GAP_IN
    fig_w, fig_h = left + plot_w + right, top + plot_h + bottom
    left_f, right_f = left / fig_w, 1 - right / fig_w
    bottom_f, top_f = bottom / fig_h, 1 - top / fig_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        left=left_f,
        right=right_f,
        top=top_f,
        bottom=bottom_f,
        wspace=_COL_GAP_IN / _TILE_IN,
        hspace=_ROW_GAP_IN / _TILE_IN,
    )

    im = None
    first_axes: list[Any] = []
    for row, module in enumerate(modules):
        for col, component in enumerate(alive[module]):
            ax = fig.add_subplot(gs[row, col])
            grid = grids_at_pos.get(module, {}).get(component)
            if grid is None:
                grid = np.zeros((b_max, a_max))
            im = ax.imshow(
                grid,
                origin="lower",
                extent=(0.5, a_max + 0.5, 0.5, b_max + 0.5),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="none",
            )
            ax.set_title(str(component), fontsize=_LABEL_FS, fontweight="bold", pad=2)
            ax.set_xticks([1, a_max // 2, a_max] if row == n_rows - 1 else [])
            ax.set_yticks([1, b_max // 2, b_max] if col == 0 else [])
            ax.tick_params(labelsize=_TICK_FS)
            if col == 0:
                first_axes.append(ax)

    assert im is not None

    # Facet titles (matrix names) outermost; single axis titles just inside them, next to plots.
    for ax0, module in zip(first_axes, modules, strict=True):
        box = ax0.get_position()
        fig.text(
            (left - 0.66) / fig_w,
            (box.y0 + box.y1) / 2,
            module.rsplit(".", 1)[-1],
            rotation=90,
            ha="center",
            va="center",
            fontsize=_FACET_FS,
            fontweight="bold",
        )
    fig.text(
        (left - 0.30) / fig_w,
        (top_f + bottom_f) / 2,
        "Second operand (b)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=_AXIS_FS,
    )
    fig.text(
        (left_f + right_f) / 2,
        (bottom - 0.30) / fig_h,
        "First operand (a)",
        ha="center",
        va="center",
        fontsize=_AXIS_FS,
    )

    # Horizontal colorbar along the bottom, centred under the grid.
    cbar_w = _CBAR_LEN_IN / fig_w
    cax = fig.add_axes(((left_f + right_f) / 2 - cbar_w / 2, 0.40 / fig_h, cbar_w, 0.09 / fig_h))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=_AXIS_FS - 1)
    cbar.ax.tick_params(labelsize=_TICK_FS)
    fig.suptitle(title, fontsize=_TITLE_FS, fontweight="bold", y=1 - 0.30 / fig_h)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_ab_heatmaps(
    json_path: str,
    op: str = "+",
    ci_thr: float = 0.01,
    grep: str | None = None,
    module_grep: str | None = None,
    output_dir: str | None = None,
    position: int | None = None,
) -> Path:
    """Write one (a, b) CI-heatmap grid PNG per token position. Returns the output folder.

    A subcomponent is shown at a position when its mean CI across the displayed prompts
    (CI absent from the JSON counts as 0) exceeds `ci_thr`. `module_grep` keeps only
    modules whose name contains that substring. `position` restricts output to that single
    token position (e.g. `--position=4` for the `=` answer token of a last-token run).
    """
    assert op in _OP_LABEL, f"--op must be one of {list(_OP_LABEL)}, got {op!r}"
    json_file = Path(json_path).expanduser()
    data: PerPosition = json.loads(json_file.read_text())

    ab, a_max, b_max = _parse_ab(data, op, grep)
    all_modules = [m for m in _all_modules(data) if module_grep is None or module_grep in m]
    assert all_modules, f"no modules match module_grep={module_grep!r}"
    grids = _build_grids(data, ab, a_max, b_max, set(all_modules))
    # Mean CI is taken over the displayed (a, b) prompts only (a full grid has unused cells,
    # e.g. for subtraction); inactive/sub-threshold cells are 0 in the grid.
    displayed = np.zeros((b_max, a_max), dtype=bool)
    for a, b in ab.values():
        displayed[b - 1, a - 1] = True
    if position is not None:
        assert str(position) in grids, (
            f"position {position} not in JSON (have {sorted(grids, key=int)})"
        )
        grids = {str(position): grids[str(position)]}

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir_of_dataset(json_file)) / f"ab_heatmaps_{_OP_LABEL[op]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("position_*.png"):  # don't leave figures from a prior, wider run
        stale.unlink()

    n_written = 0
    for pos in sorted(grids, key=int):
        # Per position, keep subcomponents whose mean CI over the displayed prompts > ci_thr.
        alive = {
            module: sorted(
                comp
                for comp, grid in grids[pos].get(module, {}).items()
                if grid[displayed].mean() > ci_thr
            )
            for module in all_modules
        }
        modules = [module for module in all_modules if alive[module]]
        if not modules:
            continue
        _plot_position(
            modules,
            alive,
            grids[pos],
            a_max,
            b_max,
            out_dir / f"position_{int(pos):02d}.png",
            cmap="RdPu",
            vmin=0.0,
            vmax=1.0,
            cbar_label="causal importance",
            title=f'position {pos}: causal importance over "a{op}b="',
        )
        n_written += 1

    logger.info(f"Wrote {n_written} (a,b) heatmap grids to {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(plot_ab_heatmaps)
