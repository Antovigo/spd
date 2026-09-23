"""Every neuron of one MLP at `=`: op-conditional means and b-line spectra of g, silu(g), up, act.

    python -m param_decomp.arith_repr.vectors.mirror_neurons 15   # -> OUT/mirror/L15_neurons.npz

For all 14,336 neurons (not only those behind alive down components), per op: mean of g, s =
silu(g), u, act = s u; b-line coefficients (50 k) of each (a is averaged out: the (0, k) entry of the
grid DFT); a-line coefficients of act. Plus the layer's `W_down` (4096, 14336), so that the MLP's
output code is exactly `W_down @ act_b(k)`."""

import sys
from typing import Any, cast

import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.vectors.common import DATASET, OUT


def b_line(x: np.ndarray) -> np.ndarray:
    """x (10000, n) on the (a, b) grid -> (50, n) b-line coefficients (phase referred to b, not b-1)."""
    xb = x.reshape(100, 100, -1).mean(0)  # (b, n)
    xb = xb - xb.mean(0)
    f = np.fft.fft(xb, axis=0)[1:51] / 100
    k = np.arange(1, 51)[:, None]
    return (f * np.exp(-2j * np.pi * k / 100)).astype(np.complex64)


def a_line(x: np.ndarray) -> np.ndarray:
    return b_line(x.reshape(100, 100, -1).transpose(1, 0, 2).reshape(10000, -1))


def main() -> None:
    layer = int(sys.argv[1])
    g_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")
    g_all = np.asarray(g_mm[layer, :, 4], np.float32)
    u_all = np.asarray(u_mm[layer, :, 4], np.float32)
    out: dict[str, np.ndarray] = {}
    for o in range(2):
        sl = slice(o * 10000, (o + 1) * 10000)
        g, u = g_all[sl], u_all[sl]
        s = g / (1.0 + np.exp(-g))
        act = s * u
        for nm, x in (("g", g), ("s", s), ("u", u), ("act", act)):
            out[f"{nm}_mean_o{o}"] = x.mean(0)
            out[f"{nm}_b_o{o}"] = b_line(x)
            out[f"{nm}_var_o{o}"] = x.var(0)
        out[f"act_a_o{o}"] = a_line(act)
        print("op", o, flush=True)
    ml_dtypes.bfloat16  # imported to register bfloat16 for the safetensors read  # noqa: B018
    out["W_down"] = Weights().get(f"model.layers.{layer}.mlp.down_proj.weight")
    (OUT / "mirror").mkdir(exist_ok=True)
    np.savez(OUT / f"mirror/L{layer}_neurons.npz", **cast(dict[str, Any], out))


if __name__ == "__main__":
    main()
