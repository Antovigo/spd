"""The integer-only `(a, b)`-grid applet, written from the collected dataset (no forward pass).

One slice per (operation, position `t`): every integer-only component ever on at `t` (the
dataset's columns at `t`), each as two grids over the operation's `(a, b)` cells, `a` the row
and `b` the column: its binarized original CI (1 where on) and its inner activation in the
integer-only model divided by `||V_c||`. A cell with no prompt (e.g. `b > a` in a pool whose
subtractions keep `a >= b`) is NaN in both, grey in the applet. A component's ranking value
(`mean_ci` in the payload) is its on-fraction over the operation's prompts at `t`.

The page is `ci_filter/grids_app.html` with its data folder as the base URL; slices use the
payload format of `experiments/lm/ab_grid_dataset.py` (split, with thumbnails, when a side
exceeds `THUMBNAIL_MAX_SIDE`) and `ci_filter/grid.py`'s file names."""

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.grid import APPLET, slice_filename, write_slice
from param_decomp.ci_filter.pool import SYMBOLS
from param_decomp.core.log import logger
from param_decomp.experiments.lm.ab_grid_dataset import (
    ABGridSnapshot,
    ab_grid_payload,
    ab_grid_split_payload,
    thumbnail_factor,
)
from param_decomp.experiments.lm.arithmetic_eval import ArithmeticGrid
from param_decomp.integer_only.config import Operation
from param_decomp.integer_only.layout import CI_DIR, INNER_DIR
from param_decomp.integer_only.pool import PromptPool

VIEW_LABELS = {"ci": "original CI > {threshold}", "inner": "inner act (integer-only model)"}
"""The two views' toggle labels (`{threshold}` filled in), keyed by the payload's view names."""


def operation_grid(
    a: np.ndarray, b: np.ndarray, op: Operation
) -> tuple[ArithmeticGrid, np.ndarray]:
    """The bounding `(a, b)` grid of one operation's prompts and each prompt's cell (row-major)."""
    a_values = tuple(range(int(a.min()), int(a.max()) + 1))
    b_values = tuple(range(int(b.min()), int(b.max()) + 1))
    grid = ArithmeticGrid(a_values, b_values, SYMBOLS[op])
    cell = (a - a_values[0]) * grid.n_b + (b - b_values[0])
    assert np.unique(cell).size == cell.size, "two prompts share an (a, b) cell"
    return grid, cell


def scatter(values: np.ndarray, cell: np.ndarray, n_cells: int) -> np.ndarray:
    """`(n_rows, k)` -> `(n_cells, 1, k)` float32, NaN where no prompt sits."""
    out = np.full((n_cells, 1, values.shape[1]), np.nan, np.float32)
    out[cell, 0] = values
    return out


def slice_documents(
    grid: ArithmeticGrid,
    cell: np.ndarray,
    position: int,
    seq_len: int,
    step: int,
    site_columns: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    site_sizes: dict[str, int],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """One slice's index and module documents. `site_columns[site]` = (component ids, the
    operation's binarized CI `(n_rows, k)` bool, its normalized inner activation `(n_rows, k)`)."""
    split = thumbnail_factor(grid) > 1
    n_cells = grid.n_a * grid.n_b
    index: dict[str, object] | None = None
    entries: list[dict[str, object]] = []
    modules: dict[str, dict[str, object]] = {}
    for site in sorted(site_sizes):
        ids, on, inner = site_columns.get(
            site, (np.zeros(0, np.int64), np.zeros((cell.size, 0), bool), np.zeros((cell.size, 0)))
        )
        mean = np.zeros((1, site_sizes[site]), np.float32)
        mean[0, ids] = on.mean(axis=0) if on.shape[0] else 0.0
        snapshot = ABGridSnapshot(
            mean_ci={"output": {site: mean}},
            saved={site: ids},
            ci_columns={"output": {site: scatter(on.astype(np.float32), cell, n_cells)}},
            inner_columns={site: scatter(inner, cell, n_cells)},
        )
        args = (snapshot, grid, (position,), seq_len, step, 0.0)
        if split:
            doc, docs = ab_grid_split_payload(*args)
            modules.update(docs)
        else:
            doc = ab_grid_payload(*args)
        if index is None:
            index = {k: v for k, v in doc.items() if k != "modules"}
        entries.extend(doc["modules"])
    assert index is not None
    return index | {"modules": entries}, modules


def write_manifest(
    grid_dir: Path,
    title: str,
    ops: tuple[Operation, ...],
    position_labels: tuple[str, ...],
    threshold: float,
    view_labels: Mapping[str, str] = VIEW_LABELS,
) -> None:
    grid_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": title,
        "ops": [SYMBOLS[op] for op in ops],
        "positions": list(range(len(position_labels))),
        "position_labels": list(position_labels),
        "files": {
            f"{SYMBOLS[op]}:{t}": slice_filename(op, t)
            for op in ops
            for t in range(len(position_labels))
        },
        "view_labels": {k: v.format(threshold=threshold) for k, v in view_labels.items()},
    }
    (grid_dir / "manifest.js").write_text(f"window.AB_GRID_SLICES = {json.dumps(manifest)};\n")


def applet_page_text(data_dir_name: str) -> str:
    """`grids_app.html` reading its manifest and slices from `<data_dir_name>/`."""
    page = APPLET.read_text()
    anchor = '<meta charset="utf-8">'
    assert page.count(anchor) == 1
    return page.replace(anchor, f'{anchor}\n<base href="{data_dir_name}/">')


def write_operation_slice(
    grid_dir: Path,
    op: Operation,
    position: int,
    seq_len: int,
    step: int,
    grid: ArithmeticGrid,
    cell: np.ndarray,
    site_columns: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    site_sizes: dict[str, int],
) -> int:
    """Write slice `(op, position)`; returns its number of component grids."""
    index, modules = slice_documents(grid, cell, position, seq_len, step, site_columns, site_sizes)
    write_slice(grid_dir, op, SYMBOLS[op], position, index, modules)
    return int(sum(ids.size for ids, _, _ in site_columns.values()))


def write_applet(
    datasets: Path,
    pool: PromptPool,
    sizes: Mapping[str, int],
    step: int,
    threshold: float,
    title: str,
    view_labels: Mapping[str, str],
    applet_data: Path,
    applet_page: Path,
) -> dict[str, object]:
    """The page `applet_page` and its data folder `applet_data` (replaced atomically) from a
    collected dataset folder `datasets` (`collect.collect_dataset`) of `pool`; `sizes` is every
    site's component count."""
    d = datasets
    index = dict(np.load(d / "index.npz"))
    n, ops, labels = pool.n, pool.ops, pool.position_labels
    assert np.array_equal(index["op"], pool.op), "the dataset is not this pool's"
    out = applet_data.with_name(applet_data.name + ".partial")
    if out.exists():
        shutil.rmtree(out)
    t0 = time.time()
    counts: dict[str, int] = {}
    for t in range(len(labels)):
        cols = np.flatnonzero(index["col_position"] == t)
        inner_all = np.load(d / INNER_DIR / f"inner_pos{t}.npy").astype(np.float32)
        inner_all /= np.maximum(index["col_v_norm"][cols], 1e-12)[None, :]
        packed = np.load(d / CI_DIR / f"ci_on_pos{t}.npy")
        on_all = np.unpackbits(packed, axis=0, count=n).astype(bool)
        sites = index["col_site"][cols]
        for k, op in enumerate(ops):
            rows = np.flatnonzero(pool.op == k)
            grid, cell = operation_grid(pool.a[rows], pool.b[rows], op)
            on_op, inner_op = on_all[rows], inner_all[rows]
            site_columns = {}
            for site in np.unique(sites):
                sel = np.flatnonzero(sites == site)
                ids = index["col_component"][cols[sel]]
                site_columns[str(site)] = (ids, on_op[:, sel], inner_op[:, sel])
            counts[f"{op}_pos{t}"] = write_operation_slice(
                out, op, t, len(labels), step, grid, cell, site_columns, dict(sizes)
            )
        logger.info(f"applet: position {t} ({time.time() - t0:.0f}s)")
    write_manifest(out, title, ops, labels, threshold, view_labels)
    if applet_data.exists():
        shutil.rmtree(applet_data)
    out.rename(applet_data)
    applet_page.write_text(applet_page_text(applet_data.name))
    return {"grids_per_slice": counts, "seconds": round(time.time() - t0, 1)}
