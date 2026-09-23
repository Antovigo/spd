"""Per-variable profiles of every active component, the inputs of the mechanism clustering.

    python -m param_decomp.arith_repr.autointerp.mech_prep --dataset <filter>/dataset
        --uv uv_alive.npz --wiring wiring.npz --out <dir>

A variable is a categorical function of the prompt visible at the position (`variables`): a
residue q mod tau (q in a, b, res; tau | 100), a value, a tens digit, a digit pair, a flag. For
every position p in 1..4, op, and component on at >= 0.5 % of that op's prompts:

* `prof_<p>_<o>.npz`: `cols` (component columns), `var_names`, per variable `w.<name>` (n_cols,
  n_classes) the class means of the gated write coefficient w = inner * [CI > 0.01] minus its
  overall mean (the component's contribution to the arrangement of that variable, in units of
  its U), `i.<name>` the same for the ungated inner (what a reader reads), `n.<name>` the class
  counts, `vals.<name>` the class values; `r2_w`, `r2_i` (n_vars, n_cols) adjusted R^2 of w / inner
  on each variable; `var_w`, `var_i`, `mean_w`, `on_rate` (n_cols,).
* `u_gram.npy` (n_writer, n_writer) float32: U_c . U_c' over wiring.npz `writers` (o/down).
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from param_decomp.arith_repr.autointerp.tuning import group_r2

THR = 0.01
MIN_ON = 0.005
TAUS = (2, 4, 5, 10, 20, 25, 50)


def variables(a: np.ndarray, b: np.ndarray, o: int, pos: int) -> dict[str, np.ndarray]:
    """Categorical variables visible at `pos`, as integer class labels per prompt."""
    v: dict[str, np.ndarray] = {}
    qs = {"a": a} if pos < 3 else {"a": a, "b": b}
    if pos == 4:
        qs["res"] = a + b if o == 0 else a - b
    for q, x in qs.items():
        for tau in TAUS:
            v[f"{q}%{tau}"] = x % tau
        v[f"{q}%100"] = x % 100
        v[f"{q}//10"] = x // 10
        if q == "res":
            v["res"] = x
    if pos >= 3:
        v["units(a,b)"] = (a % 10) * 10 + b % 10
        v["tens(a,b)"] = (a // 10) * 11 + b // 10
    if pos == 4:
        v["carry" if o == 0 else "borrow"] = (
            (a % 10 + b % 10 >= 10) if o == 0 else (a % 10 < b % 10)
        ).astype(int)
        v["cmp(a,b)"] = np.sign(a - b)
        v["res>=100" if o == 0 else "res<0"] = ((a + b >= 100) if o == 0 else (a < b)).astype(int)
    return v


def class_means(x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    oh = np.zeros((vals.size, g.size), np.float32)
    oh[inv, np.arange(g.size)] = 1.0 / cnt[inv]
    return vals, cnt, (oh @ x).T  # (n_cols, n_classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--wiring", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ix = np.load(args.dataset / "index.npz")
    a_all, b_all, op_all = ix["a"], ix["b"], ix["op"]
    site, cidx = ix["comp_site"], ix["comp_index"]
    writers = np.load(args.wiring)["writers"]
    uv = np.load(args.uv)
    U = np.zeros((writers.size, 4096), np.float32)
    for s in np.unique(site[writers]):
        ids = {int(i): j for j, i in enumerate(uv[s + ".ids"])}
        for n, c in enumerate(writers):
            if site[c] == s:
                U[n] = uv[s + ".U"][ids[int(cidx[c])]]
    np.save(args.out / "u_gram.npy", U @ U.T)
    print("gram saved", flush=True)
    ci_mm = np.load(args.dataset / "original" / "ci.npy", mmap_mode="r")
    inner_mm = np.load(args.dataset / "original" / "inner.npy", mmap_mode="r")
    for pos in range(1, 5):
        on_all = np.asarray(ci_mm[:, pos, :], np.float32) > THR
        inner_all = np.asarray(inner_mm[:, pos, :], np.float32)
        for o in (0, 1):
            m = op_all == o
            on = on_all[m]
            rate = on.mean(0)
            cols = np.flatnonzero(rate >= MIN_ON)
            inner = inner_all[m][:, cols]
            w = inner * on[:, cols]
            w -= w.mean(0)
            mean_w = (inner * on[:, cols]).mean(0)
            inner -= inner.mean(0)
            var = variables(a_all[m], b_all[m], o, pos)
            names = list(var)
            out: dict[str, Any] = {
                "cols": cols,
                "var_names": np.array(names),
                "on_rate": rate[cols],
                "var_w": (w**2).mean(0),
                "var_i": (inner**2).mean(0),
                "mean_w": mean_w,
            }
            r2w = np.zeros((len(names), cols.size), np.float32)
            r2i = np.zeros((len(names), cols.size), np.float32)
            okw, oki = w.std(0) > 0, inner.std(0) > 0
            for k, name in enumerate(names):
                g = var[name]
                r2w[k, okw] = group_r2(w[:, okw], g)
                r2i[k, oki] = group_r2(inner[:, oki], g)
                vals, cnt, mw = class_means(w, g)
                _, _, mi = class_means(inner, g)
                out[f"vals.{name}"], out[f"n.{name}"] = vals, cnt
                out[f"w.{name}"], out[f"i.{name}"] = mw.astype(np.float32), mi.astype(np.float32)
            out["r2_w"], out["r2_i"] = r2w, r2i
            np.savez(args.out / f"prof_{pos}_{o}.npz", **out)
            print("pos", pos, "op", o, "cols", cols.size, flush=True)


if __name__ == "__main__":
    main()
