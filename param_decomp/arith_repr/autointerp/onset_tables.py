"""On-set of every component at every position where it is on, in words.

    python -m param_decomp.arith_repr.autointerp.onset_tables --run <run_dir>

For each component, position p in 1..4 and operation with an on-rate (CI > 0.01) of at least
0.5 % at p, the ON indicator over that operation's 10,000 prompts is explained by the class means
of a fixed list of functions of (a, b) visible at p (adjusted R^2, `tuning.group_r2`). `label` =
the function with the fewest classes whose R^2 is within 90 % of the best (`always` when the
on-rate is > 0.95, `unexplained` when the best R^2 < 0.5), and `desc` spells the on-set out:
a value set / period / arc along a, b or res (`onsets.describe`), or the on cells of the
units-digit (tens-digit) plane for the digit-pair labels. Writes `onsets_all.parquet`."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.onsets import describe_at
from param_decomp.arith_repr.autointerp.tuning import group_r2

FILTER = "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
MIN_ON = 0.005


def feature_set(a: np.ndarray, b: np.ndarray, o: int, pos: int) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {"a": a, "a//10": a // 10}
    for tau in (2, 5, 10, 20, 25, 50):
        f[f"a%{tau}"] = a % tau
    if pos < 3:
        return f
    res = a + b if o == 0 else a - b
    f |= {"b": b, "b//10": b // 10, "res": res, "res//10": res // 10, "res%100": res % 100}
    for tau in (2, 5, 10, 20, 25, 50):
        f[f"b%{tau}"] = b % tau
        f[f"res%{tau}"] = res % tau
    f |= {
        "units(a,b)": (a % 10) * 10 + b % 10,
        "tens(a,b)": (a // 10) * 11 + b // 10,
        "carry" if o == 0 else "borrow": (a % 10 + b % 10 >= 10) if o == 0 else (a % 10 < b % 10),
        "cmp(a,b)": np.sign(a - b),
        "res>=100": res >= 100,
    }
    return f


def base_quantity(label: str) -> str | None:
    for q in ("res", "a", "b"):
        if label == q or label.startswith((q + "%", q + "//")):
            return q
    return None


def plane(on: np.ndarray, x: np.ndarray, y: np.ndarray, xn: str, yn: str, n: int) -> str:
    """Rows of the x-digit / y-digit plane with their on y-digits (rate > 0.5), identical rows merged."""
    g = np.zeros((n, n))
    c = np.zeros((n, n))
    np.add.at(g, (x, y), on)
    np.add.at(c, (x, y), 1)
    rate = g / np.maximum(c, 1)
    rows: dict[str, list[int]] = {}
    for i in range(n):
        ys = np.flatnonzero(rate[i] > 0.5)
        if ys.size:
            rows.setdefault(",".join(map(str, ys)), []).append(i)
    if not rows:
        return "no cell above 0.5"
    parts = [f"{xn} in {{{','.join(map(str, xs))}}} -> {yn} in {{{ys}}}" for ys, xs in rows.items()]
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    ds = args.run / FILTER
    ix = np.load(ds / "index.npz")
    a_all, b_all, op_all = ix["a"], ix["b"], ix["op"]
    ci = np.load(ds / "original" / "ci.npy", mmap_mode="r")
    rows = []
    for pos in range(1, 5):
        on_all = np.asarray(ci[:, pos, :], np.float32) > 0.01
        for o in (0, 1):
            m = op_all == o
            a, b = a_all[m], b_all[m]
            on = on_all[m].astype(np.float32)
            rate = on.mean(0)
            cols = np.flatnonzero(rate >= MIN_ON)
            feats = feature_set(a, b, o, pos)
            names = sorted(feats, key=lambda k: np.unique(feats[k]).size)
            x = on[:, cols] - rate[cols]
            ok = x.std(0) > 0
            R2 = np.zeros((len(names), cols.size), np.float32)
            for i, k in enumerate(names):
                R2[i, ok] = group_r2(x[:, ok], feats[k])
            res = a + b if o == 0 else a - b
            qv = {"a": a, "b": b, "res": res}
            profiles = {}
            for q, v in qv.items():
                vals, inv = np.unique(v, return_inverse=True)
                onehot = np.zeros((v.size, vals.size), np.float32)
                onehot[np.arange(v.size), inv] = 1.0
                profiles[q] = (vals, (onehot.T @ on[:, cols]) / onehot.sum(0)[:, None])
            for j, c in enumerate(cols):
                best = float(R2[:, j].max())
                if rate[c] > 0.95:
                    label, r2 = "always", best
                elif best < 0.5:
                    label, r2 = "unexplained", best
                else:
                    i = next(i for i in range(len(names)) if R2[i, j] >= 0.9 * best)
                    label, r2 = names[i], float(R2[i, j])
                q = base_quantity(label)
                if q is not None:
                    vals, prof = profiles[q]
                    if "%" in label:
                        tau = int(label.split("%")[1])
                    else:
                        tau = 100 if label == "res%100" else 0
                    desc = describe_at(vals, prof[:, j], q, tau)
                    if label.endswith("//10"):
                        desc = "(tens) " + desc
                elif label == "units(a,b)":
                    desc = plane(on[:, c], a % 10, b % 10, "a%10", "b%10", 10)
                elif label == "tens(a,b)":
                    desc = plane(on[:, c], a // 10, b // 10, "a//10", "b//10", 11)
                elif label in ("carry", "borrow", "cmp(a,b)", "res>=100"):
                    g = feats[label]
                    desc = ", ".join(
                        f"{label}={int(v)}: {on[g == v, c].mean():.2f}" for v in np.unique(g)
                    )
                else:
                    desc = ""
                rows.append((int(c), pos, ("add", "sub")[o], float(rate[c]), label, r2, desc))
            print("pos", pos, "op", o, "components", cols.size, flush=True)
    t = pd.DataFrame(rows, columns=["col", "pos", "op", "on_rate", "label", "r2", "desc"])
    t.to_parquet(args.run / "analysis/arith_repr/autointerp/onsets_all.parquet")
    print(t.groupby(["pos", "op"]).size())


if __name__ == "__main__":
    main()
