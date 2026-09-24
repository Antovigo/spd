"""Exact bookkeeping of how one MLP at `=` writes each harmonic of the result code.

    python -m param_decomp.arith_repr.vectors.mlp_periods <layer> [model|comp|comp_mean]
        # -> OUT/periods/L<layer>.npz (model) or L<layer>_<mode>.npz

Sources of the MLP (all positions `=`, no CI gating anywhere):
* `model`: the original model — measured gate / up pre-activations of all 14,336 neurons, and
  the model's W_down.
* `comp`: components only — g_n = sum_c U_c[n] h_c over the alive gate components (h_c = x . V_c,
  the measured inner), u likewise over the alive up components, and W_down replaced by
  sum_d U_d V_d^T over the alive down components.
* `comp_mean`: as `comp`, plus each neuron's missing constant (true mean minus rebuilt mean), so
  silu works at the model's operating point; tells whether a difference comes from the offset.

On the full (a, b) grid (a complete Z_100 x Z_100) the product act = s * u (s = silu(g)) has 2-D DFT
equal to the circular convolution of the DFTs of s and u, so the result-line coefficient of every
neuron splits exactly into (res = sum line (k, k) on add, diff line (k, -k) on sub):

    act(res k) =  s(0,0) u(res k)                       P_u   result code already in up
               + u(0,0) s(res k)                        -> split below (silu of the gate)
               + s(k,0) u(0,±k) + s(0,±k) u(k,0)        X_gu  a-code (x) b-code, gate x up
               + sum_{p != 0,k} s(res p) u(res k-p)     M_gu  result x result, gate x up (harmonic mixing)
               + rest                                   O_gu  anything else (off-line x off-line)

and, with a per-neuron quadratic fit s ~ c0 + c1 g + c2 g^2 (g^2's DFT is exact),

    u(0,0) s(res k) = u00 (c1 + 2 c2 g00) g(res k)                     S_lin  result code already in the gate
                    + u00 c2 2 g(k,0) g(0,±k)                           S_x    a x b inside silu (curvature)
                    + u00 c2 sum_{p != 0,k} g(res p) g(res k-p)          S_m    result x result inside silu
                    + rest                                              S_o    higher order / off-line

The layer's result write at harmonic k is T(k) = W_down act(res k) (4096; exact, all 14,336
neurons). Every term is credited by its projection on T(k): share = Re <W_down term, T(k)> / |T(k)|^2
(shares sum to 1 over the terms). Stored per op: `T` (50, 4096), `share` (50, 9 terms),
`mix_gu` / `mix_s` (50, 100): share of each partner harmonic p in M_gu / S_m, `neuron` (14336, 50, 9)
per-neuron shares, and per-neuron line coefficients `lines` (14336, 8, 50): g_a, g_b, u_a, u_b,
g_res, u_res, s_a, s_b (b at +k on add, -k on sub)."""

import sys
from typing import Any, cast

import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.vectors.common import DATASET, OUT, comp_table, load_uv

TERMS = ("P_u", "S_lin", "X_gu", "S_x", "M_gu", "S_m", "S_o", "O_gu", "total")
CHUNK = 1024


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def fft2(x: np.ndarray) -> np.ndarray:
    return (np.fft.fft2(x.reshape(100, 100, -1), axes=(0, 1)) / 1e4).astype(np.complex64)


def quad_fit(g: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column least squares s ~ c0 + c1 g + c2 g^2 -> (c1, c2)."""
    X = np.stack([np.ones_like(g), g, g * g], -1)  # (N, n, 3)
    XtX = np.einsum("Nni,Nnj->nij", X, X)
    Xty = np.einsum("Nni,Nn->ni", X, s)
    c = np.linalg.solve(XtX + 1e-6 * np.eye(3), Xty[..., None])[..., 0]
    return c[:, 1], c[:, 2]


def rebuilt(layer: int, kind: str, sl: slice) -> np.ndarray:
    """(10000, 14336) pre-activation of every neuron from the alive `kind` components only."""
    comps = comp_table()
    cols = np.flatnonzero((comps["layer"] == layer) & (comps["kind"] == kind))
    _, U = load_uv(comps, cols)
    inner = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    h = np.asarray(inner[sl, 4][:, cols], np.float32)
    return h @ np.stack(U)


def main() -> None:
    layer = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "model"
    assert ml_dtypes.bfloat16  # imported so numpy knows bfloat16 (safetensors read)
    g_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")
    if mode == "model":
        Wd = Weights().get(f"model.layers.{layer}.mlp.down_proj.weight")  # (4096, 14336)
    else:
        comps = comp_table()
        dcols = np.flatnonzero((comps["layer"] == layer) & (comps["kind"] == "down"))
        Vd, Ud = load_uv(comps, dcols)
        Wd = np.stack(Ud, 1) @ np.stack(Vd)  # (4096, 14336) = sum_d U_d V_d^T
    n_neur = Wd.shape[1]
    k = np.arange(100)
    out: dict[str, np.ndarray] = {}
    for o in range(2):
        sl = slice(o * 10000, (o + 1) * 10000)
        g_all = np.asarray(g_mm[layer, sl, 4], np.float32)
        u_all = np.asarray(u_mm[layer, sl, 4], np.float32)
        if mode != "model":
            g_true, u_true = g_all, u_all
            g_all, u_all = rebuilt(layer, "gate", sl), rebuilt(layer, "up", sl)
            if mode == "comp_mean":
                g_all += g_true.mean(0) - g_all.mean(0)
                u_all += u_true.mean(0) - u_all.mean(0)
            del g_true, u_true
        kb = k if o == 0 else (-k) % 100  # b index of the result line
        chunks = [slice(i, min(i + CHUNK, n_neur)) for i in range(0, n_neur, CHUNK)]
        # pass 1: the layer's result write T(k)
        A = np.zeros((n_neur, 50), np.complex64)
        for ch in chunks:
            act = silu(g_all[:, ch]) * u_all[:, ch]
            F = fft2(act)
            A[ch] = F[k[1:51], kb[1:51]].T
        T = (Wd @ A).T  # (50, 4096)
        T2 = (np.abs(T) ** 2).sum(1)
        D = (Wd.T @ np.conj(T).T).astype(np.complex64)  # (n, 50): w_n . conj T(k)
        # pass 2: the terms
        neuron = np.zeros((n_neur, 50, len(TERMS)), np.float32)
        mix_gu = np.zeros((50, 100), np.float64)
        mix_s = np.zeros((50, 100), np.float64)
        lines = np.zeros((n_neur, 8, 50), np.complex64)
        for ch in chunks:
            g = g_all[:, ch]
            s = silu(g)
            u = u_all[:, ch]
            c1, c2 = quad_fit(g, s)
            G, S, U, G2 = fft2(g), fft2(s), fft2(u), fft2(g * g)
            del G2  # g^2 enters only through the decomposition below
            ks = k[1:51]
            g_res = G[k, kb].T  # (n, 100) result line of g, all harmonics
            s_res = S[k, kb].T
            u_res = U[k, kb].T
            s00, u00, g00 = S[0, 0], U[0, 0], G[0, 0]
            a_idx, b_idx = (ks, np.zeros_like(ks)), (np.zeros_like(ks), kb[ks])
            s_a, s_b = S[a_idx].T, S[b_idx].T  # (n, 50)
            u_a, u_b = U[a_idx].T, U[b_idx].T
            g_a, g_b = G[a_idx].T, G[b_idx].T
            act_res = fft2(s * u)[ks, kb[ks]].T  # (n, 50)
            P_u = s00[:, None] * u_res[:, ks]
            X_gu = s_a * u_b + s_b * u_a
            # mixing: p over Z_100 \ {0, k}
            mix_terms = s_res[:, None, :] * u_res[:, (ks[:, None] - k[None, :]) % 100]  # (n,50,100)
            gmix = g_res[:, None, :] * g_res[:, (ks[:, None] - k[None, :]) % 100]
            excl = (k[None, :] == 0) | (k[None, :] == ks[:, None])  # (50, 100)
            mix_terms[:, excl] = 0
            gmix[:, excl] = 0
            M_gu = mix_terms.sum(2)
            S_all = u00[:, None] * s_res[:, ks]
            S_lin = (u00 * (c1 + 2 * c2 * g00))[:, None] * g_res[:, ks]
            S_x = (u00 * c2 * 2)[:, None] * g_a * g_b
            S_m = (u00 * c2)[:, None] * gmix.sum(2)
            S_o = S_all - S_lin - S_x - S_m
            O_gu = act_res - P_u - S_all - X_gu - M_gu
            d = D[ch]  # (n, 50)
            for t, X in enumerate((P_u, S_lin, X_gu, S_x, M_gu, S_m, S_o, O_gu, act_res)):
                neuron[ch, :, t] = (X * d).real / T2[None, :]
            mix_gu += ((mix_terms * d[:, :, None]).real.sum(0)) / T2[:, None]
            mix_s += (((u00 * c2)[:, None, None] * gmix * d[:, :, None]).real.sum(0)) / T2[:, None]
            lines[ch] = np.stack([g_a, g_b, u_a, u_b, g_res[:, ks], u_res[:, ks], s_a, s_b], 1)
            print(layer, o, ch.start, flush=True)
        out[f"T_o{o}"] = T.astype(np.complex64)
        out[f"share_o{o}"] = neuron.sum(0)
        out[f"mix_gu_o{o}"] = mix_gu.astype(np.float32)
        out[f"mix_s_o{o}"] = mix_s.astype(np.float32)
        out[f"neuron_o{o}"] = neuron
        out[f"lines_o{o}"] = lines
    out["terms"] = np.array(TERMS)
    (OUT / "periods").mkdir(exist_ok=True)
    name = f"L{layer}" if mode == "model" else f"L{layer}_{mode}"
    np.savez(OUT / f"periods/{name}.npz", **cast(dict[str, Any], out))


if __name__ == "__main__":
    main()
