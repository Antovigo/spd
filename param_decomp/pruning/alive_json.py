"""Write a learned mask as an alive list in the CI filter's `alive/alive.json` format.

    python -m param_decomp.pruning.alive_json --prune_dir <run>/analysis/pruning/step_<s>/<name>
        --mask <prune_dir>/learn/<learn_name>/lam_<λ> --out <file.json>

`{"components": {site: [component ids]}, "source": ...}`: every site of `S₀`, keeping the `S₀`
components whose logit `ℓ > 0` in the mask's `state.npz`, so tools that read an alive set (e.g. the
alive-only extraction) can run the pruned model instead."""

import argparse
import json
from pathlib import Path

import numpy as np

from param_decomp.pruning.marks import alive_set
from param_decomp.pruning.model import slots_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prune_dir", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    alive = alive_set(args.prune_dir)
    slots = slots_of(alive)
    on = np.load(args.mask / "state.npz")["logit"] > 0
    assert on.size == slots.n, (on.size, slots.n)
    comps: dict[str, list[int]] = {site: [] for site in alive}
    for a in np.nonzero(on)[0]:
        comps[slots.sites[a]].append(int(slots.index[a]))
    for site in comps:
        comps[site].sort()
        assert set(comps[site]) <= set(np.nonzero(alive[site])[0].tolist()), site
    doc = {"components": comps, "source": str(args.mask), "n": int(on.sum())}
    args.out.write_text(json.dumps(doc))
    print(f"{doc['n']} components -> {args.out}")


if __name__ == "__main__":
    main()
