"""Rewrite a CI filter's whole-grid `ab_grids/` slices in the split format `grid.write_operation`
now writes for grids wider than `THUMBNAIL_MAX_SIDE`: pooled thumbnails in `<op>_pos<t>.js`,
full grids in `<op>_pos<t>/<site>.js`. No forward pass: the full grids are the slice's own.
Also refreshes `index.html` and writes `alive.html`.

    python -m param_decomp.ci_filter.scripts.split_grids <filter dir>/ab_grids
"""

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from param_decomp.ci_filter.grid import APPLET, alive_page, write_slice
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.ab_grid_dataset import (
    THUMBNAIL_MAX_SIDE,
    encode_ci_u8,
    pool_grid,
)

PREFIX, SUFFIX = "window.registerABGridSlice(", ");"


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def split_slice(path: Path) -> bool:
    text = path.read_text()
    assert text.startswith(PREFIX) and text.endswith(SUFFIX), path
    index = json.loads(text[len(PREFIX) : -len(SUFFIX)])
    if "thumb" in index:
        return False
    n_a, n_b = index["n_a"], index["n_b"]
    factor = -(-max(n_a, n_b) // THUMBNAIL_MAX_SIDE)
    if factor == 1:
        return False
    operation, position = path.stem.rsplit("_pos", 1)
    modules: dict[str, dict[str, Any]] = {}
    for entry in index["modules"]:
        if "ci" not in entry:
            continue
        k = len(entry["saved"])
        shape = (k, 1, 1, n_a, n_b)
        ci = np.frombuffer(base64.b64decode(entry["ci"]), np.uint8).reshape(shape)
        inner = np.frombuffer(base64.b64decode(entry["inner"]), np.float16).reshape(shape)
        assert "ci_hidden" not in entry, "single-role filter grids only"
        modules[entry["name"]] = {
            "step": index["step"],
            "name": entry["name"],
            "ci": entry.pop("ci"),
            "inner": entry.pop("inner"),
        }
        entry["ci_thumb"] = b64(encode_ci_u8(pool_grid(ci.astype(np.float32) / 255.0, factor)))
        entry["inner_thumb"] = b64(pool_grid(inner, factor).astype(np.float16))
    index["thumb"] = {"factor": factor, "n_a": -(-n_a // factor), "n_b": -(-n_b // factor)}
    write_slice(path.parent, operation, index["ops"][0], int(position), index, modules)
    return True


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grid_dir", type=Path)
    args = ap.parse_args()
    for path in sorted(args.grid_dir.glob("*_pos*.js")):
        t0 = time.time()
        done = split_slice(path)
        logger.info(f"{path.name}: {'split' if done else 'left as is'} ({time.time() - t0:.0f}s)")
    (args.grid_dir / "index.html").write_bytes(APPLET.read_bytes())
    (args.grid_dir / "alive.html").write_text(alive_page())


if __name__ == "__main__":
    main()
