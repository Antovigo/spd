"""Class means of the raw residual stream over a, b and the result, at every residual point.

    python -m param_decomp.arith_repr.autointerp.class_means --dataset <filter>/dataset --out <dir>

The arrangement of a group representation (the residues of q mod tau, q in {a, b, res}) at a
point is the set of class means of the residual over those residues; every such arrangement is
an average of the per-value class means stored here, so this is computed once.

* `class_means.npy` (65 points, n_rows, d_model) float16: row r = the mean raw residual over the
  prompts of (position, op, quantity, value) `rows[r]` minus the mean over all prompts of that
  (position, op) — centred, so the arrangement's centroid is 0. Points follow `resid_labels`:
  0 = embed, 2l + 1 = after block l's attention add, 2l + 2 = after its MLP add.
* `class_means_index.npz`: `pos`, `op`, `q` (0 a, 1 b, 2 res), `value`, `count` per row;
  `total_var` (65, 5, 2): mean squared distance of a prompt's residual from the (position, op)
  mean — the denominator for "what fraction of the stream's variance is this code"."""

import argparse
from pathlib import Path

import numpy as np

ROWS_Q = {1: (0,), 2: (0,), 3: (0, 1), 4: (0, 1, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ix = np.load(args.dataset / "index.npz")
    a, b, op = ix["a"], ix["b"], ix["op"]
    resid = np.load(args.dataset / "original" / "resid.npy", mmap_mode="r")
    n_pt, _, _, d = resid.shape
    rows: list[tuple[int, int, int, int, int]] = []
    onehots: dict[tuple[int, int], np.ndarray] = {}
    for p, qs in ROWS_Q.items():
        for o in (0, 1):
            m = op == o
            quantities = (a[m], b[m], a[m] + b[m] if o == 0 else a[m] - b[m])
            blocks = []
            for qi in qs:
                vals, inv, cnt = np.unique(quantities[qi], return_inverse=True, return_counts=True)
                oh = np.zeros((vals.size, m.sum()), np.float32)
                oh[inv, np.arange(m.sum())] = 1.0 / cnt[inv]
                blocks.append(oh)
                rows += [(p, o, qi, int(v), int(c)) for v, c in zip(vals, cnt, strict=True)]
            onehots[p, o] = np.concatenate(blocks)
    out = np.lib.format.open_memmap(
        args.out / "class_means.npy", mode="w+", dtype=np.float16, shape=(n_pt, len(rows), d)
    )
    total_var = np.zeros((n_pt, 5, 2), np.float32)
    for pt in range(n_pt):
        x_all = np.asarray(resid[pt, :, 1:5, :], np.float32)
        r0 = 0
        for p in ROWS_Q:
            x = x_all[:, p - 1]
            for o in (0, 1):
                xm = x[op == o]
                xc = xm - xm.mean(0)
                total_var[pt, p, o] = float((xc**2).sum(1).mean())
                cm = onehots[p, o] @ xc
                out[pt, r0 : r0 + cm.shape[0]] = cm.astype(np.float16)
                r0 += cm.shape[0]
        print("point", pt, flush=True)
    out.flush()
    r = np.array(rows)
    np.savez(
        args.out / "class_means_index.npz",
        pos=r[:, 0],
        op=r[:, 1],
        q=r[:, 2],
        value=r[:, 3],
        count=r[:, 4],
        total_var=total_var,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
