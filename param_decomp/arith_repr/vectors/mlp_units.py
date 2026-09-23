"""The neurons behind every alive down component: line spectra of silu(gate), up and their product.

    python -m param_decomp.arith_repr.vectors.mlp_units <layer>   # -> OUT/mlp/L<layer>.npz

A down component's V is neuron-sparse (median participation ratio 5.6), so its inner is a short
weighted sum of neuron outputs `act_n = silu(g_n) * u_n`. For the union of the top-8 neurons (by
|V_d|) of the layer's alive down components, at positions 1, 3, 4 (layers 0-5) or 4 only, per op:
line spectra (4 lines, 50 k) and means of the gate pre-activation `g`, `s = silu(g)`, `u` and
`act`. `g` and `u` are linear reads of the stream (sums of gate / up components), `s` and `act` are not. The product of two grid
functions has line coefficient at (k, k) (the sum line)

    act_sum(k) ~ s_a(k) u_b(k) + s_b(k) u_a(k) + s_0 u_sum(k) + u_0 s_sum(k) + ...

— the first two terms are NEW sum-line content made by multiplying an a-code by a b-code, the
last two carry a sum code already present in the inputs; the diff line has the same with u_b ->
conj(u_b). The analysis (`mlp_analysis`) compares these terms with the measured act spectrum."""

import sys
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.vectors.common import DATASET, OUT, comp_table, line_dft, load_uv


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def spectra(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (100, 100, n) -> (4, 50, n) line coefficients of the centred x, and its mean (n,)."""
    m = x.mean((0, 1))
    return line_dft(x - m), m


def main() -> None:
    layer = int(sys.argv[1])
    comps = comp_table()
    ds = np.flatnonzero((comps["layer"] == layer) & (comps["kind"] == "down"))
    V, _ = load_uv(comps, ds)
    Vd = np.stack(V, 1)  # (14336, n_down)
    top = np.argsort(-np.abs(Vd), axis=0)[:8]
    neurons = np.unique(top)
    positions = (1, 3, 4) if layer <= 5 else (4,)
    g_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")
    out: dict[str, np.ndarray] = {"neurons": neurons, "down_cols": ds, "Vd": Vd[neurons]}
    for p in positions:
        g = np.asarray(g_mm[layer, :, p][:, neurons], np.float32)
        u = np.asarray(u_mm[layer, :, p][:, neurons], np.float32)
        s = silu(g)
        for o in range(2):
            sl = slice(o * 10000, (o + 1) * 10000)
            for nm, x in (("g", g[sl]), ("s", s[sl]), ("u", u[sl]), ("act", s[sl] * u[sl])):
                F, m = spectra(x.reshape(100, 100, -1))
                out[f"{nm}_p{p}_o{o}"] = np.moveaxis(F, -1, 0)  # (n, 4, 50)
                out[f"{nm}_mean_p{p}_o{o}"] = m
                out[f"{nm}_var_p{p}_o{o}"] = x.var(0)
        print(layer, p, neurons.size, flush=True)
    (OUT / "mlp").mkdir(exist_ok=True)
    np.savez(OUT / f"mlp/L{layer}.npz", **cast(dict[str, Any], out))


if __name__ == "__main__":
    main()
