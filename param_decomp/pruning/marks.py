"""Mark the components a pruning run removed in a CI filter's ab-grid applet.

    python -m param_decomp.pruning.marks --prune_dir <run>/analysis/pruning/step_<s>/<name>
        --grids_dir <run>/<filter>-<s>/ab_grids

Writes `<grids_dir>/pruned.js` (`window.PRUNED = {source, sites: {site: {component: {block}}}}`)
from the newest `sets/S_<j>.npz` (a run still in progress is marked as far as it got) and
refreshes `index.html` / `alive.html` from the current applet, which draws a red ✕ in a white square in the top-right
corner of every pruned component's tile. A component counts as pruned when it is in `S₀` and off
in that set; `block` is the block (blockwise mode) or round (global mode) that last removed it."""

import argparse
import json
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.grid import APPLET, alive_page
from param_decomp.pruning.model import slots_of


def pruned_payload(prune_dir: Path, alive: dict[str, np.ndarray]) -> dict[str, object]:
    sets = sorted((prune_dir / "sets").glob("S_*.npz"))
    assert sets, f"no sets/S_*.npz under {prune_dir}"
    saved = np.load(sets[-1])
    slots = slots_of(alive)
    on, removed = saved["on"], saved["removed"]
    assert on.size == slots.n, (on.size, slots.n)
    sites: dict[str, dict[str, dict[str, int]]] = {}
    for a in np.nonzero(~on)[0]:
        sites.setdefault(slots.sites[a], {})[str(int(slots.index[a]))] = {"block": int(removed[a])}
    return {
        "source": f"{prune_dir.name}/{sets[-1].name}",
        "n_pruned": int((~on).sum()),
        "sites": sites,
    }


def alive_set(prune_dir: Path) -> dict[str, np.ndarray]:
    """`S₀` as `{site: (C,) bool}`: the filter's alive set the pruning run started from."""
    config = (prune_dir / "config.yaml").read_text()
    filter_id = next(
        line.split(":", 1)[1].strip()
        for line in config.splitlines()
        if line.startswith("filter_id:")
    )
    step = prune_dir.parent.name.removeprefix("step_")
    run_dir = prune_dir.parents[3]
    filt = run_dir / f"{filter_id}-{step}" / "alive"
    ids = json.loads((filt / "alive.json").read_text())["components"]
    with np.load(filt / "max_ci.npz") as max_ci:
        sizes = {site: int(max_ci[site].size) for site in max_ci.files}
    out = {}
    for site, size in sizes.items():
        keep = np.zeros(size, bool)
        keep[np.asarray(ids.get(site, []), np.int64)] = True
        out[site] = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune_dir", type=Path, required=True)
    ap.add_argument("--grids_dir", type=Path, required=True)
    args = ap.parse_args()
    payload = pruned_payload(args.prune_dir, alive_set(args.prune_dir))
    (args.grids_dir / "pruned.js").write_text(f"window.PRUNED = {json.dumps(payload)};\n")
    (args.grids_dir / "index.html").write_bytes(APPLET.read_bytes())
    (args.grids_dir / "alive.html").write_text(alive_page())
    print(f"{payload['n_pruned']} pruned components ({payload['source']}) -> {args.grids_dir}")


if __name__ == "__main__":
    main()
