"""Does the operation reflect an operand's Fourier codes? (basis-free, from the residual stream)

    python -m param_decomp.arith_repr.autointerp.op_phase --resid <original-resid dir> --out op_phase.npz

For every read point (attn_in.l, mlp_in.l), position 3 and 4, quantity q in {a, b} and operation,
the class means of the post-norm read input over q's 100 values give, per frequency k, the
complex vector `z_op(k) = sum_v mean(v) exp(-2 pi i k v / 100)` in R^4096 (+ i R^4096). If the
sub code is the add code with q -> -q, `z_sub(k) = conj(z_add(k))`; if it is the same code,
`z_sub(k) = z_add(k)`. Stored: `same`, `reflect` = |<z_sub, z_add>| and |<z_sub, conj z_add>|
over the norms (L, 2 streams, 2 positions, 2 quantities, 51 k), and `power` (..., 2 ops, 51 k) =
|z(k)|^2 / total class-mean variance."""

import argparse
from pathlib import Path

import numpy as np

from param_decomp.arith_repr.resid import load_read_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="dataset index.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ix = np.load(args.labels)
    q_vals = {"a": ix["a"], "b": ix["b"]}
    op = ix["op"]
    K = 51
    ks = np.arange(K)
    phase = np.exp(-2j * np.pi * np.outer(ks, np.arange(1, 101)) / 100)  # (K, 100)
    same = np.zeros((32, 2, 2, 2, K))
    refl = np.zeros_like(same)
    power = np.zeros((32, 2, 2, 2, 2, K))
    for li in range(32):
        for s_i, stream in enumerate(("attn_in", "mlp_in")):
            for p_i, pos in enumerate((3, 4)):
                x = load_read_input(args.resid, f"{stream}.{li}", pos)
                for q_i, q in enumerate(("a", "b")):
                    z = {}
                    for o in (0, 1):
                        m = op == o
                        onehot = np.zeros((m.sum(), 100), np.float32)
                        onehot[np.arange(m.sum()), q_vals[q][m] - 1] = 1
                        means = onehot.T @ x[m] / onehot.sum(0)[:, None]  # (100, d)
                        means -= means.mean(0)
                        z[o] = phase @ means  # (K, d) complex
                        tot = (means**2).sum() * 100
                        power[li, s_i, p_i, q_i, o] = (np.abs(z[o]) ** 2).sum(1) / tot
                    na = np.linalg.norm(z[0], axis=1)
                    ns = np.linalg.norm(z[1], axis=1)
                    den = np.maximum(na * ns, 1e-30)
                    same[li, s_i, p_i, q_i] = np.abs((np.conj(z[1]) * z[0]).sum(1)) / den
                    refl[li, s_i, p_i, q_i] = np.abs((z[1] * z[0]).sum(1)) / den
        print(
            "layer",
            li,
            "mlp b@= k=10 same/refl",
            same[li, 1, 1, 1, 10],
            refl[li, 1, 1, 1, 10],
            flush=True,
        )
    np.savez(args.out, same=same, reflect=refl, power=power)


if __name__ == "__main__":
    main()
