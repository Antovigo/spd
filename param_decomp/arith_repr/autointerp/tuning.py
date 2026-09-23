"""Tuning of every alive component: what function of (a, b, op) its write coefficient is.

    python -m param_decomp.arith_repr.autointerp.tuning --dataset <filter>/dataset --out tuning.npz

For each position p (1..4) and operation (add, sub), the component's write coefficient on a
prompt is `w = inner * [CI > thr]` (what the rounded-CI decomposed model multiplies U by). Over
the full 100 x 100 (a, b) grid of one operation it is described two ways:

* group-mean R^2 (`r2`, adjusted for the number of groups) against the fixed feature list
  `FEATURES` (a, b, a+b, a-b, each mod tau, digits, carry/borrow, sign), and the linear R^2 of
  a, b, a+b, a-b (`lin`);
* the 2-D DFT of the grid on the (a mod 100, b mod 100) torus, as energy fractions on four
  lines, folded to |k| in 1..50: `a-only` (k_b = 0), `b-only` (k_a = 0), `sum` (k_a = k_b,
  functions of a+b mod 100), `diff` (k_a = -k_b, functions of a-b mod 100). k = 50 lies on both
  diagonals (parity of a+b = parity of a-b) and is booked on `sum`.

Pooled over both operations: the between-op share of the variance (`op_r2`) and the
correlation of the add and sub grids (`add_sub_corr`). CI statistics (mean CI, fraction of
prompts with CI > thr) are kept for every position including <BOS>."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

THR = 0.01
TAUS = (2, 4, 5, 10, 20, 25, 50, 100)


def features(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    s, d = a + b, a - b
    out: dict[str, np.ndarray] = {"a": a, "b": b, "a+b": s, "a-b": d}
    for tau in TAUS:
        out[f"a%{tau}"] = a % tau
        out[f"b%{tau}"] = b % tau
        out[f"(a+b)%{tau}"] = s % tau
        out[f"(a-b)%{tau}"] = d % tau
    out |= {
        "a//10": a // 10,
        "b//10": b // 10,
        "(a+b)//10": s // 10,
        "(a-b)//10": d // 10,
        "carry": (a % 10 + b % 10 >= 10).astype(int),
        "borrow": (a % 10 < b % 10).astype(int),
        "sign(a-b)": np.sign(d),
        "a+b>=100": (s >= 100).astype(int),
        "a+b>100": (s > 100).astype(int),
        "units(a,b)": (a % 10) * 10 + b % 10,
        "tens(a,b)": (a // 10) * 11 + b // 10,
    }
    return out


def group_r2(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Adjusted R^2 of the group means of `g` for every column of `x` (n, A), x centred."""
    _, inv = np.unique(g, return_inverse=True)
    n_g = int(inv.max()) + 1
    onehot = np.zeros((x.shape[0], n_g), np.float32)
    onehot[np.arange(x.shape[0]), inv] = 1.0
    counts = onehot.sum(0)
    sums = onehot.T @ x
    ss_between = (sums**2 / counts[:, None]).sum(0)
    ss_total = (x**2).sum(0)
    r2 = ss_between / np.maximum(ss_total, 1e-30)
    n = x.shape[0]
    return 1.0 - (1.0 - r2) * (n - 1) / max(n - n_g, 1)


def linear_r2(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    fc = (f - f.mean()).astype(np.float32)
    proj = fc @ x / (fc @ fc)
    return proj**2 * (fc @ fc) / np.maximum((x**2).sum(0), 1e-30)


def dft_lines(
    x: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 1024
) -> dict[str, np.ndarray]:
    """Energy fractions of the centred grid on the four DFT lines, folded to k = 1..50."""
    n_comp = x.shape[1]
    grid_idx = (a % 100) * 100 + (b % 100)
    out = {k: np.zeros((n_comp, 51), np.float32) for k in ("a-only", "b-only", "sum", "diff")}
    ks = np.arange(100)
    for lo in range(0, n_comp, chunk):
        xs = x[:, lo : lo + chunk]
        grid = np.zeros((10000, xs.shape[1]), np.float32)
        grid[grid_idx] = xs
        grid = grid.reshape(100, 100, -1)
        power = np.abs(np.fft.fft2(grid, axes=(0, 1))) ** 2
        total = power.sum((0, 1)) - power[0, 0]
        total = np.maximum(total, 1e-30)
        lines = {
            "a-only": power[ks, 0],
            "b-only": power[0, ks],
            "sum": power[ks, ks],
            "diff": power[ks, (-ks) % 100],
        }
        for name, line in lines.items():
            folded = np.zeros((51, xs.shape[1]), np.float32)
            for k in range(1, 100):
                folded[min(k, 100 - k)] += line[k]
            if name == "diff":
                folded[50] = 0.0
            out[name][lo : lo + chunk] = (folded / total).T
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = np.load(args.dataset / "index.npz")
    a_all, b_all, op_all = index["a"], index["b"], index["op"]
    ci_mm = np.load(args.dataset / "original" / "ci.npy", mmap_mode="r")
    inner_mm = np.load(args.dataset / "original" / "inner.npy", mmap_mode="r")
    n_comp = ci_mm.shape[2]
    feats_by_op = {o: features(a_all[op_all == o], b_all[op_all == o]) for o in (0, 1)}
    feat_names = list(feats_by_op[0])
    lin_names = ["a", "b", "a+b", "a-b"]
    res: dict[str, np.ndarray] = {
        "ci_mean": np.zeros((5, 2, n_comp), np.float32),
        "ci_on": np.zeros((5, 2, n_comp), np.float32),
        "w_mean": np.zeros((5, 2, n_comp), np.float32),
        "w_std": np.zeros((5, 2, n_comp), np.float32),
        "r2": np.zeros((5, 2, len(feat_names), n_comp), np.float32),
        "lin": np.zeros((5, 2, len(lin_names), n_comp), np.float32),
        "op_r2": np.zeros((5, n_comp), np.float32),
        "add_sub_corr": np.zeros((5, n_comp), np.float32),
    }
    for line in ("a-only", "b-only", "sum", "diff"):
        res[f"dft_{line}"] = np.zeros((5, 2, n_comp, 51), np.float32)
    for p in range(5):
        ci = np.asarray(ci_mm[:, p, :], np.float32)
        inner = np.asarray(inner_mm[:, p, :], np.float32)
        w = inner * (ci > THR)
        del inner
        pooled = w - w.mean(0)
        per_op = {}
        for o in (0, 1):
            m = op_all == o
            res["ci_mean"][p, o] = ci[m].mean(0)
            res["ci_on"][p, o] = (ci[m] > THR).mean(0)
            res["w_mean"][p, o] = w[m].mean(0)
            res["w_std"][p, o] = w[m].std(0)
            per_op[o] = w[m] - w[m].mean(0)
        res["op_r2"][p] = 1.0 - sum((per_op[o] ** 2).sum(0) for o in (0, 1)) / np.maximum(
            (pooled**2).sum(0), 1e-30
        )
        num = (per_op[0] * per_op[1]).sum(0)
        den = np.sqrt((per_op[0] ** 2).sum(0) * (per_op[1] ** 2).sum(0))
        res["add_sub_corr"][p] = num / np.maximum(den, 1e-30)
        del ci, w, pooled
        if p == 0:
            continue
        for o in (0, 1):
            x = per_op[o]
            f = feats_by_op[o]
            for i, name in enumerate(feat_names):
                res["r2"][p, o, i] = group_r2(x, f[name])
            for i, name in enumerate(lin_names):
                res["lin"][p, o, i] = linear_r2(x, f[name])
            m = op_all == o
            lines = dft_lines(x, a_all[m], b_all[m])
            for line, arr in lines.items():
                res[f"dft_{line}"][p, o] = arr
            print(f"pos {p} op {o} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **cast(dict[str, Any], res))
    args.out.with_suffix(".json").write_text(
        json.dumps({"features": feat_names, "linear": lin_names, "threshold": THR}, indent=1)
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
