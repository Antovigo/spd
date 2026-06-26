"""Plot inner activations over the (a, b) grid — the inner-activation twin of plot_ab_heatmaps.

Same figure layout as `plot_ab_heatmaps` (matrices down the rows, subcomponents across the
columns, one `a×b` heatmap per subcomponent), but each tile is coloured by the subcomponent's
**previously-collected normalized inner activation** `(x·V_c)/||V_c||` at the last token (read
from `inner_activations_<op>.tsv`) instead of causal importance. Inner activations are signed,
so a diverging colormap (`RdBu_r`, red +, blue −) on a symmetric shared scale replaces CI's
0..1 `RdPu`; every layout dimension (tile size, gaps, fonts, margins, colorbar) is identical
because the rendering is the shared `plot_ab_heatmaps._plot_position`. CPU-only.

There is a single figure (the inner activations are last-token only), written into the *same*
folder as the CI heatmaps so the two views sit side by side.

Usage:
    python -m param_decomp_lab.scripts.validation.plot_ab_inner_heatmaps <inner_activations_tsv> \
        [--output-dir=PATH]

Output: `<run_dir>/analysis/ab_heatmaps_<op>/inner_activations.png`.
"""

import csv
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import analysis_dir, op_symbol, run_dir_of_dataset
from param_decomp_lab.scripts.validation.plot_ab_heatmaps import _plot_position


def _infer_op(tsv_path: Path) -> str:
    stem = tsv_path.stem  # inner_activations_<op>
    assert stem.startswith("inner_activations_"), f"unexpected TSV name: {tsv_path.name}"
    return stem.removeprefix("inner_activations_")


def plot_ab_inner_heatmaps(inner_activations_tsv: str, output_dir: str | None = None) -> Path:
    """Write the inner-activation (a, b) heatmap grid. Returns the output folder."""
    tsv_path = Path(inner_activations_tsv).expanduser()
    assert tsv_path.exists(), f"missing inner-activations TSV: {tsv_path}"
    op = _infer_op(tsv_path)

    # Read once (grid extent isn't known up front), then allocate and fill.
    entries: list[tuple[str, int, int, int, float]] = []  # proj, comp, a, b, inner_act
    a_max = b_max = 0
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            a, b = int(row["a"]), int(row["b"])
            a_max, b_max = max(a_max, a), max(b_max, b)
            entries.append((row["matrix"], int(row["subcomponent"]), a, b, float(row["inner_act"])))
    assert entries, f"no rows in {tsv_path}"

    # proj -> component -> [b_max, a_max] dense inner-activation grid.
    grids: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for proj, comp, a, b, val in entries:
        per_comp = grids[proj]
        if comp not in per_comp:
            per_comp[comp] = np.zeros((b_max, a_max))
        per_comp[comp][b - 1, a - 1] = val

    # Bare proj names sort to down/gate/up — the same order parse_module_name gives the CI
    # plot — and `_plot_position` uses `module.rsplit(".", 1)[-1]` for the row label, so the
    # short names render identically.
    modules = sorted(grids)
    alive = {m: sorted(grids[m]) for m in modules}
    vmax = max(float(np.abs(g).max()) for per in grids.values() for g in per.values())

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir_of_dataset(tsv_path)) / f"ab_heatmaps_{op}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inner_activations.png"
    _plot_position(
        modules,
        alive,
        grids,
        a_max,
        b_max,
        out_path,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_label="inner activation",
        title=f'inner activation over "a{op_symbol(op)}b=" (last token)',
    )
    logger.info(f"wrote inner-activation (a,b) heatmaps to {out_path}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(plot_ab_inner_heatmaps)
