"""Mark the components a learned mask removed in a CI filter's ab-grid applet.

    python -m param_decomp.pruning.marks --prune_dir <run>/analysis/pruning/step_<s>/<name>
        --mask <prune_dir>/learn/<learn_name>/lam_<λ> --grids_dir <run>/<filter>-<s>/ab_grids

Writes `<grids_dir>/pruned.js` (`window.PRUNED = {source, n_pruned, temperature_r2_cut, sites:
{site: {component: record}}}`) from the mask's `state.npz` (a component is pruned when its logit
`ℓ ≤ 0`; a run still in progress is marked as far as it got) and refreshes `index.html` /
`alive.html` from the current applet, which draws a red ✕ in a white square in the top-right
corner of every pruned component's tile."""

import argparse
import json
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.grid import APPLET, alive_page
from param_decomp.pruning.model import slots_of

TEMPERATURE_R2 = 0.5
"""Temperature-like above this KL-weighted temperature R² (the entropy-neuron cut of
`notes/rmsnorm/entropy-components.md`)."""


def pruned_payload(
    prune_dir: Path, mask_dir: Path, alive: dict[str, np.ndarray]
) -> dict[str, object]:
    """Per `S₀` component: whether the learned mask in `mask_dir` pruned it, and its screen on
    `S₀` — KL from the alive-only model when ablated, `ΔCE` (full vocabulary and among
    integers, with SE), `P(integer)` before and after the ablation — plus its temperature fit
    (`temperature.npz`) when that stage ran."""
    slots = slots_of(alive)
    on = np.load(mask_dir / "state.npz")["logit"] > 0
    assert on.size == slots.n, (on.size, slots.n)
    table = np.genfromtxt(
        prune_dir / "components.tsv", delimiter="\t", names=True, dtype=None, encoding=None
    )
    assert table.shape == (slots.n,), table.shape
    temp_path = prune_dir / "temperature.npz"
    temp = dict(np.load(temp_path)) if temp_path.exists() else None
    if temp is not None:
        assert (temp["comps"] == np.arange(slots.n)).all()
    sites: dict[str, dict[str, dict[str, object]]] = {}
    for a in range(slots.n):
        p_after = float(table["without_p_int"][a])
        info: dict[str, object] = {
            "pruned": bool(not on[a]),
            "kl": float(table["without_kl_base"][a]),
            "d_ce": float(table["d_ce"][a]),
            "se_ce": float(table["se_ce"][a]),
            "d_ce_int": float(table["d_ce_int"][a]),
            "p_int_before": p_after - float(table["d_p_int"][a]),
            "p_int_after": p_after,
        }
        if temp is not None:
            info.update(temp_r2=float(temp["r2"][a]), beta=float(temp["beta"][a]))
        sites.setdefault(slots.sites[a], {})[str(int(slots.index[a]))] = info
    return {
        "source": str(mask_dir.relative_to(prune_dir.parent)),
        "n_pruned": int((~on).sum()),
        "temperature_r2_cut": TEMPERATURE_R2,
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
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--grids_dir", type=Path, required=True)
    args = ap.parse_args()
    payload = pruned_payload(args.prune_dir, args.mask, alive_set(args.prune_dir))
    (args.grids_dir / "pruned.js").write_text(f"window.PRUNED = {json.dumps(payload)};\n")
    (args.grids_dir / "index.html").write_bytes(APPLET.read_bytes())
    (args.grids_dir / "alive.html").write_text(alive_page())
    print(f"{payload['n_pruned']} pruned components ({payload['source']}) -> {args.grids_dir}")


if __name__ == "__main__":
    main()
