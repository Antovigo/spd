"""Mark the components learned masks removed in a CI filter's ab-grid applet.

    python -m param_decomp.pruning.marks --prune_dir <run>/analysis/pruning/step_<s>/<name>
        --grids_dir <run>/<filter>-<s>/ab_grids [--mask <mask_dir> ...] [--default <mask_dir>]

Each mask dir holds a learned mask's `state.npz` (`<prune_dir>/learn[/<learn_name>]/lam_<λ>`, λ
the L0 penalty weight); without `--mask`, every `state.npz` under `<prune_dir>/learn` is used. The blockwise run's newest `sets/S_<j>.npz`, when present, is
always added (its entry also carries `blocks`: the block that removed each component).
A component is pruned when its logit `ℓ ≤ 0`; a run still in progress is marked as far as it got.

Writes `<grids_dir>/pruned.js` (`window.PRUNED = {temperature_r2_cut, default, masks: [{name,
label, n_pruned, pruned: {site: [component]}, [blocks: {site: [block]}]}], sites: {site: {component: screen record}}}`) and
refreshes `index.html` / `alive.html` from the current applet, whose "pruning mask" menu picks
the mask that draws a red ✕ in a white square in the top-right corner of pruned tiles."""

import argparse
import json
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.grid import APPLET, alive_page
from param_decomp.pruning.model import Slots, slots_of

TEMPERATURE_R2 = 0.5
"""Temperature-like above this KL-weighted temperature R² (the entropy-neuron cut of
`notes/rmsnorm/entropy-components.md`)."""


def mask_entry(prune_dir: Path, mask_dir: Path, slots: Slots) -> dict[str, object]:
    """One learned mask: its name (path under `<prune_dir>/learn`), a menu label (λ, sweep,
    size, accuracy from `final.json` when the run finished) and its pruned components."""
    on = np.load(mask_dir / "state.npz")["logit"] > 0
    assert on.size == slots.n, (on.size, slots.n)
    name = str(mask_dir.relative_to(prune_dir / "learn"))
    sweep, lam = (str(Path(name).parent), mask_dir.name.removeprefix("lam_"))
    label = f"λ={lam}" + ("" if sweep == "." else f" ({sweep})") + f": {int(on.sum())} kept"
    final = mask_dir / "final.json"
    if final.exists():
        label += f", acc {100 * json.loads(final.read_text())['all']['correct']:.1f}%"
    else:
        label += ", unfinished"
    pruned: dict[str, list[int]] = {}
    for a in np.flatnonzero(~on):
        pruned.setdefault(slots.sites[a], []).append(int(slots.index[a]))
    return {"name": name, "label": label, "n_pruned": int((~on).sum()), "pruned": pruned}


def blockwise_entry(prune_dir: Path, slots: Slots) -> dict[str, object] | None:
    """The blockwise run's newest set `sets/S_<j>.npz` (`on`, and `removed`: the block that last
    removed each component), labelled with its size and its accuracy on `Q` from
    `learn/baseline.json` when scored there; None without sets."""
    sets = sorted((prune_dir / "sets").glob("S_*.npz"))
    if not sets:
        return None
    saved = np.load(sets[-1])
    on, removed = saved["on"], saved["removed"]
    assert on.size == slots.n, (on.size, slots.n)
    label = f"blockwise ({sets[-1].stem}): {int(on.sum())} kept"
    baseline = prune_dir / "learn" / "baseline.json"
    scored = json.loads(baseline.read_text()).get(sets[-1].stem) if baseline.exists() else None
    if scored is not None:
        label += f", acc {100 * scored['all']['correct']:.1f}%"
    pruned: dict[str, list[int]] = {}
    blocks: dict[str, list[int]] = {}
    for a in np.flatnonzero(~on):
        pruned.setdefault(slots.sites[a], []).append(int(slots.index[a]))
        blocks.setdefault(slots.sites[a], []).append(int(removed[a]))
    return {
        "name": f"blockwise/{sets[-1].stem}",
        "label": label,
        "n_pruned": int((~on).sum()),
        "pruned": pruned,
        "blocks": blocks,
    }


def pruned_payload(
    prune_dir: Path, mask_dirs: list[Path], default: Path | None, alive: dict[str, np.ndarray]
) -> dict[str, object]:
    """Every mask in `mask_dirs` (`mask_entry`), and per `S₀` component its screen on `S₀` — KL
    from the alive-only model when ablated, `ΔCE` (full vocabulary and among integers, with
    SE), `P(integer)` before and after the ablation — plus its temperature fit
    (`temperature.npz`) when that stage ran."""
    slots = slots_of(alive)
    masks = [mask_entry(prune_dir, d, slots) for d in mask_dirs]
    blockwise = blockwise_entry(prune_dir, slots)
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
        "temperature_r2_cut": TEMPERATURE_R2,
        "default": mask_dirs.index(default) if default is not None else 0,
        "masks": masks + ([blockwise] if blockwise is not None else []),
        "sites": sites,
    }


def lam_of(mask_dir: Path) -> float:
    return float(mask_dir.name.removeprefix("lam_"))


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
    ap.add_argument("--mask", type=Path, nargs="*", default=None)
    ap.add_argument("--default", type=Path, default=None)
    args = ap.parse_args()
    mask_dirs = args.mask or sorted(
        (p.parent for p in (args.prune_dir / "learn").rglob("state.npz")),
        key=lambda d: (str(d.parent), lam_of(d)),
    )
    assert mask_dirs, f"no learned masks under {args.prune_dir / 'learn'}"
    default = args.default.resolve() if args.default else None
    mask_dirs = [d.resolve() for d in mask_dirs]
    payload = pruned_payload(
        args.prune_dir.resolve(), mask_dirs, default, alive_set(args.prune_dir)
    )
    (args.grids_dir / "pruned.js").write_text(f"window.PRUNED = {json.dumps(payload)};\n")
    (args.grids_dir / "index.html").write_bytes(APPLET.read_bytes())
    (args.grids_dir / "alive.html").write_text(alive_page())
    for m in payload["masks"]:  # pyright: ignore[reportGeneralTypeIssues]
        print(m["label"])
    print(f"{len(payload['masks'])} masks -> {args.grids_dir}")  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    main()
