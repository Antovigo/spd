"""Attention routing read off the q and k components: which (q, k) pairs make a head at `=` look where.

    python -m param_decomp.arith_repr.vectors.qk   # -> OUT/qk.npz

For query head h of layer l, destination `=` (position 4) and source position s, the alive part of
the attention logit is the sum over (q comp i, k comp j) of

    mean(h_i at 4) * mean(h_j at s) * <rope_4(U_i[h]), rope_s(U_j[kv(h)])> / sqrt(128)

(mean inner activations per op; RoPE applied in the HF rotate-half layout). Stored per layer:
`pair_<l>` (n_q, n_k, 32 heads, 5 sources, 2 ops), `ids_<l>` (q cols then k cols), and
`hmean` (A, 5 pos, 2 op) the mean inner of every alive component at every position."""

from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, llama3_inv_freq
from param_decomp.arith_repr.vectors.common import (
    DATASET,
    HD,
    N_HEAD,
    N_KV,
    OUT,
    comp_table,
    load_uv,
)


def rope(x: np.ndarray, pos: int, inv_freq: np.ndarray) -> np.ndarray:
    """HF rotate-half RoPE on the last axis (head dim 128)."""
    ang = pos * inv_freq  # (64,)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :64], x[..., 64:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def main() -> None:
    comps = comp_table()
    mm = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    A = mm.shape[2]
    hmean = np.zeros((A, 5, 2), np.float32)
    for o in range(2):
        hmean[:, :, o] = np.asarray(mm[o * 10000 : (o + 1) * 10000], np.float32).mean(0).T
    inv_freq = llama3_inv_freq(Weights().config)
    out: dict[str, np.ndarray] = {"hmean": hmean}
    for li in range(32):
        qs = np.flatnonzero((comps["layer"] == li) & (comps["kind"] == "q"))
        ks = np.flatnonzero((comps["layer"] == li) & (comps["kind"] == "k"))
        if not (qs.size and ks.size):
            continue
        _, Uq = load_uv(comps, qs)
        _, Uk = load_uv(comps, ks)
        Q = np.stack(Uq).reshape(-1, N_HEAD, HD)
        Kv = np.stack(Uk).reshape(-1, N_KV, HD)
        Kq = np.repeat(Kv, N_HEAD // N_KV, axis=1)  # (n_k, 32, hd)
        Qr = rope(Q, 4, inv_freq)
        pair = np.zeros((qs.size, ks.size, N_HEAD, 5, 2), np.float32)
        for s in range(5):
            dots = np.einsum("ihd,jhd->ijh", Qr, rope(Kq, s, inv_freq)) / np.sqrt(HD)
            for o in range(2):
                pair[:, :, :, s, o] = (
                    dots * hmean[qs, 4, o][:, None, None] * hmean[ks, s, o][None, :, None]
                )
        out[f"pair_{li}"] = pair
        out[f"ids_{li}"] = np.concatenate([qs, ks])
        print(li, qs.size, ks.size, flush=True)
    np.savez(OUT / "qk.npz", **cast(dict[str, Any], out))


if __name__ == "__main__":
    main()
