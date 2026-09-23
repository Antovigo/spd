"""One row per alive component: where it is on, and what its write coefficient is a function of.

    python -m param_decomp.arith_repr.autointerp.catalogue --dataset <filter>/dataset --tuning tuning.npz
        --out catalogue.parquet

`label_<op>_p<p>`: the most parsimonious feature (fewest groups) whose adjusted R^2 is within 90 %
of the best feature's, with that R^2 (`r2_...`); `dft_<line>_<op>_p<p>`: energy on each DFT line
and `k_<line>_...` its peak frequency. `main_pos` is the position with the highest on-rate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.tuning import features

OPS = ("add", "sub")
LINES = ("a-only", "b-only", "sum", "diff")


def n_groups() -> dict[str, int]:
    a = np.repeat(np.arange(1, 101), 100)
    b = np.tile(np.arange(1, 101), 100)
    return {k: int(np.unique(v).size) for k, v in features(a, b).items()}


def best_label(
    r2: np.ndarray, names: list[str], sizes: dict[str, int], frac: float = 0.9
) -> tuple[str, float]:
    best = float(r2.max())
    order = sorted(range(len(names)), key=lambda i: (sizes[names[i]], -r2[i]))
    for i in order:
        if r2[i] >= frac * best:
            return names[i], float(r2[i])
    raise AssertionError


def build(dataset: Path, tuning: Path) -> pd.DataFrame:
    index = np.load(dataset / "index.npz")
    t = np.load(tuning)
    meta = json.loads(tuning.with_suffix(".json").read_text())
    names, lin_names = meta["features"], meta["linear"]
    sizes = n_groups()
    # linear features count as 2 "groups" (a line is simpler than any 2+-class partition but binary)
    all_names = names + [f"lin({n})" for n in lin_names]
    sizes |= {f"lin({n})": 2 for n in lin_names}
    df = pd.DataFrame(
        {
            "site": index["comp_site"],
            "layer": index["comp_layer"],
            "kind": [k.split(".")[-1].replace("_proj", "") for k in index["comp_kind"]],
            "cidx": index["comp_index"],
        }
    )
    for p in range(5):
        for o, op in enumerate(OPS):
            df[f"on_{op}_p{p}"] = t["ci_on"][p, o]
            df[f"wstd_{op}_p{p}"] = t["w_std"][p, o]
            df[f"wmean_{op}_p{p}"] = t["w_mean"][p, o]
        df[f"op_r2_p{p}"] = t["op_r2"][p]
        df[f"addsub_corr_p{p}"] = t["add_sub_corr"][p]
    on = np.stack([t["ci_on"][p].mean(0) for p in range(5)], 1)
    df["main_pos"] = on.argmax(1)
    for p in range(1, 5):
        for o, op in enumerate(OPS):
            r2 = np.concatenate([t["r2"][p, o], t["lin"][p, o]], 0)  # (F, A)
            labs = [best_label(r2[:, c], all_names, sizes) for c in range(r2.shape[1])]
            df[f"label_{op}_p{p}"] = [li for li, _ in labs]
            df[f"r2_{op}_p{p}"] = [v for _, v in labs]
            for line in LINES:
                e = t[f"dft_{line}"][p, o]  # (A, 51)
                df[f"dft_{line}_{op}_p{p}"] = e.sum(1)
                df[f"k_{line}_{op}_p{p}"] = e.argmax(1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    df = build(args.dataset, args.tuning)
    df.to_parquet(args.out)
    print(df.shape, "saved", args.out)


if __name__ == "__main__":
    main()
