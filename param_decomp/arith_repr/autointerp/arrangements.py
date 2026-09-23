"""How the residues of a, b and the result are arranged in the raw residual, point by point.

    python -m param_decomp.arith_repr.autointerp.arrangements --run <run_dir>

For every group G = (position, op, quantity q, modulus tau | 100) and residual point t (0 = embed,
2l + 1 = after block l's attention, 2l + 2 = after its MLP) the arrangement is the set of tau
class means mu_t(v) = mean raw residual over the prompts with q = v mod tau (centred), built
from `mech/class_means.npy`. Written to `mech/`:

* `arrangements.parquet`, one row per (p, o, q, tau, point): `share` (between-residue variance /
  the stream's variance at that position), shape descriptors of the class-mean Gram (`pr`
  participation ratio, `circulant` = share of the Gram that is invariant under shifting every
  residue by the same amount, `flatness` of its spectrum, `ordered` = corr(distance, cyclic
  distance of residues), `shape`), and the step from the previous point: `cka_prev` (is it the
  same geometry, whatever its directions), `cos_prev` (Frobenius cos of the class means: same
  geometry in the same place), `in_span_prev` (share of the new arrangement inside the old one's span —
  1 = it stayed in place, 0 = moved to new directions).
  q = `res_int` (at `=` only): the result's class means minus what the separate a and b codes
  already give them (E[E[x | a] + E[x | b] | res]) — the part only an interaction of a and b
  can write.
* `arr_fourier.npz`: per (p, o, q) key `<p>_<o>_<q>`: `power` (65, 51) |Z_t(k)|^2 / stream
  variance for k = 0..50 on the mod-100 circle (k and -k folded), `keep` (65, 51) |alpha|^2 |Z_t|^2
  / |Z_t+1|^2 with alpha = <Z_t(k), Z_t+1(k)> / |Z_t(k)|^2 (share of the new coefficient that is
  the old one rescaled / rotated in its own plane), `phase` (65, 51) arg alpha (rotation of the
  circle = shift of every residue), `ecc` (65, 51) sigma_2 / sigma_1 of [Re Z, Im Z] (1 = round
  circle, 0 = a line); `pcs_<tau>` (65, tau, 3) the class means on their top 3 principal axes.
"""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.mechanisms import AI, center, geometry

TAUS = (2, 4, 5, 10, 20, 25, 50, 100)
Q = ("a", "b", "res", "res_int")


def residue_means(rows: np.ndarray, vals: np.ndarray, cnt: np.ndarray, tau: int) -> np.ndarray:
    r = vals % tau
    out = np.zeros((tau, rows.shape[1]), np.float64)
    n = np.zeros(tau)
    np.add.at(out, r, rows * cnt[:, None])
    np.add.at(n, r, cnt)
    return out / np.maximum(n, 1)[:, None]


def interaction_free(X_all: np.ndarray, ix: Any, o: int, res_vals: np.ndarray) -> np.ndarray:
    """E[E[x | a] + E[x | b] | res] at `=`: the part of the result's class means that the
    separate a and b codes already carry (full 100 x 100 grid per op)."""
    out = np.zeros((res_vals.size, X_all.shape[1]))
    ai = np.arange(1, 101)
    for q in (0, 1):
        sel = (ix["pos"] == 4) & (ix["op"] == o) & (ix["q"] == q)
        mu = X_all[sel]  # rows for values 1..100
        for r_i, r in enumerate(res_vals):
            # the other operand given res = r and this one = v: a = r - b (add), a = r + b (sub)
            other = (r - ai) if o == 0 else ((ai - r) if q == 0 else (r + ai))
            ok = (other >= 1) & (other <= 100)
            if ok.any():
                out[r_i] += mu[ok].mean(0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    mech = args.run / AI / "mech"
    cm = np.load(mech / "class_means.npy", mmap_mode="r")
    ix = np.load(mech / "class_means_index.npz")
    total_var = ix["total_var"]
    groups = sorted(
        {(int(p), int(o), int(q)) for p, o, q in zip(ix["pos"], ix["op"], ix["q"], strict=True)}
    )
    groups += [(4, 0, 3), (4, 1, 3)]
    n_pt = cm.shape[0]
    ks = np.arange(51)
    fourier: dict[str, np.ndarray] = {}
    rows_out = []
    prev: dict[tuple[int, int, int, int], np.ndarray] = {}
    prevZ: dict[tuple[int, int, int], np.ndarray] = {}
    for g in groups:
        key = "{}_{}_{}".format(*g)
        for name in ("power", "keep", "phase", "ecc"):
            fourier[f"{key}.{name}"] = np.zeros((n_pt, 51), np.float32)
        for tau in (10, 100):
            fourier[f"{key}.pcs_{tau}"] = np.zeros((n_pt, tau, 3), np.float32)
    for t in range(n_pt):
        X_all = np.asarray(cm[t], np.float64)
        for g in groups:
            p, o, q = g
            sel = (ix["pos"] == p) & (ix["op"] == o) & (ix["q"] == min(q, 2))
            rows, vals, cnt = X_all[sel], ix["value"][sel], ix["count"][sel]
            if q == 3:
                rows = rows - interaction_free(X_all, ix, o, vals)
            tv = float(total_var[t, p, o])
            key = "{}_{}_{}".format(*g)
            mu100 = residue_means(rows, vals, cnt, 100)
            ang = 2 * np.pi * np.outer(ks, np.arange(100)) / 100
            Z = (np.exp(-1j * ang) @ mu100) / 100  # (51, d)
            mult = np.where((ks == 0) | (ks == 50), 1.0, 2.0)
            fourier[f"{key}.power"][t] = mult * (np.abs(Z) ** 2).sum(1) / tv
            for k in ks[1:]:
                s = np.linalg.svd(np.stack([Z[k].real, Z[k].imag]), compute_uv=False)
                fourier[f"{key}.ecc"][t, k] = s[1] / max(s[0], 1e-30)
            if g in prevZ:
                Zp = prevZ[g]
                inner = (Zp.conj() * Z).sum(1)
                npv = np.maximum((np.abs(Zp) ** 2).sum(1), 1e-30)
                nz = np.maximum((np.abs(Z) ** 2).sum(1), 1e-30)
                alpha = inner / npv
                fourier[f"{key}.keep"][t] = np.abs(alpha) ** 2 * npv / nz
                fourier[f"{key}.phase"][t] = np.angle(alpha)
            prevZ[g] = Z
            for tau in TAUS:
                mu = residue_means(rows, vals, cnt, tau)
                n = np.bincount(vals % tau, weights=cnt, minlength=tau)
                w = n / n.sum()
                share = float((w * (mu**2).sum(1)).sum() / tv)
                K = center(mu @ mu.T)
                geo = geometry(K, tau)
                row: dict[str, Any] = {
                    "p": p,
                    "o": o,
                    "q": Q[q],
                    "tau": tau,
                    "point": t,
                    "share": share,
                } | geo
                if (p, o, q, tau) in prev:
                    mp = prev[p, o, q, tau]
                    Kx = center(mp @ mp.T)
                    row["cka_prev"] = float(
                        np.trace(Kx @ K) / max(np.sqrt((Kx**2).sum() * (K**2).sum()), 1e-30)
                    )
                    row["cos_prev"] = float(
                        (w * (mp * mu).sum(1)).sum()
                        / max(
                            np.sqrt((w * (mp**2).sum(1)).sum() * (w * (mu**2).sum(1)).sum()), 1e-30
                        )
                    )
                    u, s_, _ = np.linalg.svd(mp.T, full_matrices=False)
                    B = u[:, s_ > 1e-3 * s_.max()] if s_.max() > 0 else u[:, :0]
                    inside = ((mu @ B) ** 2).sum()
                    row["in_span_prev"] = float(inside / max((mu**2).sum(), 1e-30))
                prev[p, o, q, tau] = mu
                rows_out.append(row)
                if tau in (10, 100):
                    muc = mu - mu.mean(0)
                    u, s_, _ = np.linalg.svd(muc, full_matrices=False)
                    fourier[f"{key}.pcs_{tau}"][t] = (u[:, :3] * s_[:3]).astype(np.float32)
        print("point", t, flush=True)
    pd.DataFrame(rows_out).to_parquet(mech / "arrangements.parquet")
    np.savez(mech / "arr_fourier.npz", **cast(dict[str, Any], fourier))
    print("saved")


if __name__ == "__main__":
    main()
