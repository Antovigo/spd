"""Tuning curves of every alive component along a, b and the result, per position and op.

    python -m param_decomp.arith_repr.autointerp.profiles --dataset <filter>/dataset --out profiles.npz

`<sig>_<q>` (5, 2, A, n_values) float32, sig in {w (gated write coefficient), on (CI > thr),
inner (ungated)}, q in {a (1..100), b (1..100), res (a+b on add: 2..200 at index v-2; a-b on
sub: -99..99 at index v+99)}: the mean over the prompts with that value. `inner_add_r2`
(5, 2, A): the additive (a main effect + b main effect) share of the ungated inner activation's
variance on the full grid — the part a linear read of a residual carrying separate a and b codes
can produce — and `inner_res_r2`, the share explained by the result value."""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

THR = 0.01


def group_means(x: np.ndarray, g: np.ndarray, lo: int, n: int) -> np.ndarray:
    onehot = np.zeros((x.shape[0], n), np.float32)
    onehot[np.arange(x.shape[0]), g - lo] = 1.0
    counts = np.maximum(onehot.sum(0), 1)
    return (onehot.T @ x / counts[:, None]).T  # (A, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = np.load(args.dataset / "index.npz")
    a, b, op = index["a"], index["b"], index["op"]
    ci_mm = np.load(args.dataset / "original" / "ci.npy", mmap_mode="r")
    inner_mm = np.load(args.dataset / "original" / "inner.npy", mmap_mode="r")
    A = ci_mm.shape[2]
    out: dict[str, np.ndarray] = {}
    for sig in ("w", "on", "inner"):
        out[f"{sig}_a"] = np.zeros((5, 2, A, 100), np.float32)
        out[f"{sig}_b"] = np.zeros((5, 2, A, 100), np.float32)
        out[f"{sig}_res"] = np.zeros((5, 2, A, 199), np.float32)
    out["inner_add_r2"] = np.zeros((5, 2, A), np.float32)
    out["inner_res_r2"] = np.zeros((5, 2, A), np.float32)
    for p in range(5):
        ci = np.asarray(ci_mm[:, p, :], np.float32)
        inner = np.asarray(inner_mm[:, p, :], np.float32)
        on = (ci > THR).astype(np.float32)
        del ci
        sigs = {"w": inner * on, "on": on, "inner": inner}
        for o in (0, 1):
            m = op == o
            res = a[m] + b[m] if o == 0 else a[m] - b[m]
            res_lo = 2 if o == 0 else -99
            for name, x in sigs.items():
                xm = x[m]
                out[f"{name}_a"][p, o] = group_means(xm, a[m], 1, 100)
                out[f"{name}_b"][p, o] = group_means(xm, b[m], 1, 100)
                out[f"{name}_res"][p, o] = group_means(xm, res, res_lo, 199)
            xi = sigs["inner"][m]
            xc = xi - xi.mean(0)
            tot = np.maximum((xc**2).sum(0), 1e-30)
            ma = out["inner_a"][p, o] - xi.mean(0)[:, None]
            mb = out["inner_b"][p, o] - xi.mean(0)[:, None]
            out["inner_add_r2"][p, o] = 100 * ((ma**2).sum(1) + (mb**2).sum(1)) / tot
            vals, cnt = np.unique(res, return_counts=True)
            counts = np.zeros(199)
            counts[vals - res_lo] = cnt
            mr = out["inner_res"][p, o] - xi.mean(0)[:, None]
            out["inner_res_r2"][p, o] = ((mr**2) * counts[None]).sum(1) / tot
        print("pos", p, flush=True)
    np.savez(args.out, **cast(dict[str, Any], out))
    print("saved", args.out)


if __name__ == "__main__":
    main()
