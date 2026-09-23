"""Two tables about the `=` position.

    python -m param_decomp.arith_repr.autointerp.pos4_tables --run <run_dir>

* `onset_r2_pos4.parquet`: for every component whose main position is `=`, the group R^2 of its
  ON indicator (CI > 0.01) against functions of (a, b) per op (`add:<f>`, `sub:<f>`; res = a+b on
  add, a-b on sub), and `lab_<op>` = the coarsest function within 90 % of the best R^2
  (`unexplained` if the best is < 0.5).
* `delta_profiles.npz`: for every residual writer (o/down) of layer >= 14, its mean direct logit
  contribution at `=` (write coefficient x `dla`) on the token `res + Delta`, Delta in -200..200,
  over the prompts with res >= 0 (`add`, `sub`), with `dla` centred over the number tokens
  0..200 (`<op>_base`, the mean over those tokens, is then 0 and kept for compatibility)."""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.tuning import group_r2

FILTER = "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
ORDER = ["sign", "res>=100", "carry/borrow", "res%2", "res%5", "a%10", "b%10", "res%10", "a//10", "b//10",
         "res//10", "res%20", "res%50", "a", "b", "res%100", "units(a,b)", "tens(a,b)", "res"]  # fmt: skip


def onset_table(ds: Path, df: pd.DataFrame) -> pd.DataFrame:
    ix = np.load(ds / "index.npz")
    a, b, op = ix["a"], ix["b"], ix["op"]
    cols = np.asarray(df.index[df.main_pos == 4].values)
    ci = np.load(ds / "original" / "ci.npy", mmap_mode="r")
    on = (np.asarray(ci[:, 4, :], np.float32)[:, cols] > 0.01).astype(np.float32)
    t = pd.DataFrame(index=cols)
    for o, name in ((0, "add"), (1, "sub")):
        m = op == o
        A, B = a[m], b[m]
        res = A + B if o == 0 else A - B
        feats = {
            "a": A, "b": B, "res": res, "res%100": res % 100, "res%10": res % 10, "res//10": res // 10,
            "res%50": res % 50, "res%20": res % 20, "res%5": res % 5, "res%2": res % 2,
            "units(a,b)": (A % 10) * 10 + B % 10, "tens(a,b)": (A // 10) * 11 + B // 10,
            "carry/borrow": (A % 10 + B % 10 >= 10) if o == 0 else (A % 10 < B % 10),
            "res>=100": res >= 100, "sign": np.sign(res), "a%10": A % 10, "b%10": B % 10,
            "a//10": A // 10, "b//10": B // 10,
        }  # fmt: skip
        x = on[m] - on[m].mean(0)
        ok = x.std(0) > 0
        for k, v in feats.items():
            r = np.zeros(len(cols))
            r[ok] = group_r2(x[:, ok], v)
            t[f"{name}:{k}"] = r
        R2 = np.stack([np.asarray(t[f"{name}:{k}"].values) for k in ORDER], 1)
        best = R2.max(1)
        labs = []
        for i in range(len(cols)):
            if best[i] < 0.5:
                labs.append("unexplained")
                continue
            labs.append(next(k for j, k in enumerate(ORDER) if R2[i, j] >= 0.9 * best[i]))
        t[f"lab_{name}"] = labs
    t["layer"] = df.layer[cols].values
    t["kind"] = df.kind[cols].values
    return t


def delta_profiles(ds: Path, df: pd.DataFrame, dla: np.ndarray) -> dict[str, np.ndarray]:
    ix = np.load(ds / "index.npz")
    a, b, op = ix["a"], ix["b"], ix["op"]
    writers = np.asarray(df.index[df.kind.isin(["down", "o"]) & (df.layer >= 14)].values)
    ci = np.load(ds / "original" / "ci.npy", mmap_mode="r")
    inner = np.load(ds / "original" / "inner.npy", mmap_mode="r")
    W = np.asarray(inner[:, 4, :])[:, writers] * (
        np.asarray(ci[:, 4, :], np.float32)[:, writers] > 0.01
    )
    dla_w = dla[writers][:, :201].astype(np.float32)
    dla_w = dla_w - dla_w.mean(
        1, keepdims=True
    )  # a logit shift common to all number tokens decides nothing
    deltas = np.arange(-200, 201)
    out: dict[str, np.ndarray] = {"writers": writers, "deltas": deltas}
    for o, name in ((0, "add"), (1, "sub")):
        res_all = a + b if o == 0 else a - b
        m = (op == o) & (res_all >= 0)
        res, Wm = res_all[m], W[m]
        P = np.full((len(writers), len(deltas)), np.nan, np.float32)
        for j, dd in enumerate(deltas):
            t = res + dd
            ok = (t >= 0) & (t <= 200)
            if ok.sum() >= 50:
                P[:, j] = (Wm[ok] * dla_w[:, t[ok]].T).mean(0)
        out[name] = P
        out[name + "_base"] = (Wm * dla_w.mean(1)[None]).mean(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    ai = args.run / "analysis/arith_repr/autointerp"
    ds = args.run / FILTER
    df = pd.read_parquet(ai / "catalogue.parquet")
    onset_table(ds, df).to_parquet(ai / "onset_r2_pos4.parquet")
    np.savez(
        ai / "delta_profiles.npz",
        **cast(dict[str, Any], delta_profiles(ds, df, np.load(ai / "wiring.npz")["dla"])),
    )
    print("saved")


if __name__ == "__main__":
    main()
