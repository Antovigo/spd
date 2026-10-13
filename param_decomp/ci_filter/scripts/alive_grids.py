"""Write `ab_grids/alive/`: the alive-only model's inner-activation grids that `alive.html` shows,
from the filter's activation dataset (`scripts/collect_dataset.py`) — no forward pass.

For every slice file `<op>_pos<t>.js`, `alive/<op>_pos<t>.js` holds, per module, the slice's
saved components in the slice's own order: the ALIVE-ONLY ("decomposed") model's inner
activation divided by ||V_c|| (the full-model grids' normalization), NaN where the component
has no cached column (never on at t, `--per_position` datasets). A split slice (`thumb`) gets
the same pooled thumbnails there and full grids in `alive/<op>_pos<t>/<site>.js`.

    python -m param_decomp.ci_filter.scripts.alive_grids <filter dir>
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from param_decomp.ci_filter.grid import APPLET, alive_page
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.ab_grid_dataset import pool_grid

PREFIX, SUFFIX = "window.registerABGridSlice(", ");"
OP_CODES = {"+": 0, "-": 1, "×": 2}


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def read_index(path: Path) -> dict[str, Any]:
    text = path.read_text()
    assert text.startswith(PREFIX) and text.endswith(SUFFIX), path
    return json.loads(text[len(PREFIX) : -len(SUFFIX)])


class Inner:
    """The dataset's decomposed-model inner activations at one position, by alive column."""

    def __init__(self, dataset: Path, position: int, n_alive: int) -> None:
        per_pos = dataset / "decomposed" / f"inner_pos{position}.npy"
        if per_pos.exists():
            self.mm = np.load(per_pos, mmap_mode="r")
            self.cols: np.ndarray | None = np.load(dataset / "index.npz")[f"pos{position}_cols"]
            self.offset = 0
        else:
            full = np.load(dataset / "decomposed" / "inner.npy", mmap_mode="r")
            self.mm = full.reshape(full.shape[0], -1)
            self.cols = None
            self.offset = position * n_alive

    def block(self, first: int, n: int, alive_cols: np.ndarray) -> np.ndarray:
        """`(n, len(alive_cols))` float32, NaN where a column is not cached."""
        out = np.full((n, alive_cols.size), np.nan, np.float32)
        if self.cols is None:
            where, present = alive_cols + self.offset, np.ones(alive_cols.size, bool)
        else:
            k = np.searchsorted(self.cols, alive_cols)
            k = np.minimum(k, self.cols.size - 1)
            present = self.cols[k] == alive_cols
            where = k
        rows = np.asarray(self.mm[first : first + n])
        out[:, present] = rows[:, where[present]].astype(np.float32)
        return out


def alive_slice(grid_dir: Path, path: Path, dataset: Path, index_npz: dict[str, np.ndarray]) -> int:
    index = read_index(path)
    position, symbol = index["positions"][0], index["ops"][0]
    n_a, n_b = index["n_a"], index["n_b"]
    op = index_npz["op"]
    (rows,) = np.nonzero(op == OP_CODES[symbol])
    first, n = int(rows[0]), n_a * n_b
    assert rows.size == n and rows[-1] == first + n - 1, (symbol, rows.size, n)
    assert index_npz["a"][first] == index["a_min"] and index_npz["b"][first] == index["b_min"]

    col_of = {
        (str(site), int(c)): j
        for j, (site, c) in enumerate(
            zip(index_npz["comp_site"], index_npz["comp_index"], strict=True)
        )
    }
    v_norm = index_npz["comp_v_norm"].astype(np.float32)
    wanted = [
        (entry["name"], c)
        for entry in index["modules"]
        if "ci" in entry or "ci_thumb" in entry
        for c in entry["saved"]
    ]
    alive_cols = np.asarray(sorted({col_of[w] for w in wanted if w in col_of}), np.int64)
    inner = Inner(dataset, position, len(v_norm))
    block = inner.block(first, n, alive_cols) / v_norm[alive_cols]
    at = {int(j): i for i, j in enumerate(alive_cols)}

    thumb = index.get("thumb")
    stem = path.stem
    doc: dict[str, Any] = {"slice": f"{symbol}:{position}", "modules": {}}
    n_written = 0
    for entry in index["modules"]:
        if not entry["saved"] or ("ci" not in entry and "ci_thumb" not in entry):
            continue
        site = entry["name"]
        grids = np.full((len(entry["saved"]), n_a, n_b), np.nan, np.float32)
        for si, c in enumerate(entry["saved"]):
            j = col_of.get((site, c))
            if j is not None:
                grids[si] = block[:, at[j]].reshape(n_a, n_b)
                n_written += 1
        if thumb is None:
            doc["modules"][site] = {"inner": b64(grids.astype(np.float16))}
        else:
            file = f"{stem}/{site}.js"
            module = {"slice": doc["slice"], "name": site, "inner": b64(grids.astype(np.float16))}
            out = grid_dir / "alive" / file
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"window.registerAliveModule({json.dumps(module)});")
            # NaN-aware pooling: a cell block with any cached value averages those it has.
            pooled = pool_grid(grids, thumb["factor"])
            doc["modules"][site] = {"inner": b64(pooled.astype(np.float16)), "file": file}
    (grid_dir / "alive").mkdir(exist_ok=True)
    (grid_dir / "alive" / path.name).write_text(f"window.registerAliveSlice({json.dumps(doc)});")
    return n_written


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("filter_dir", type=Path)
    args = ap.parse_args()
    grid_dir, dataset = args.filter_dir / "ab_grids", args.filter_dir / "dataset"
    index_npz = dict(np.load(dataset / "index.npz"))
    for path in sorted(grid_dir.glob("*_pos*.js")):
        t0 = time.time()
        n = alive_slice(grid_dir, path, dataset, index_npz)
        logger.info(f"{path.name}: {n} component grids ({time.time() - t0:.0f}s)")
    (grid_dir / "index.html").write_bytes(APPLET.read_bytes())
    (grid_dir / "alive.html").write_text(alive_page())


if __name__ == "__main__":
    main()
