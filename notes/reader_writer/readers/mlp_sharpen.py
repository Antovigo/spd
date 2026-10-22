"""How do the MLPs of blocks 11..16 sharpen the a mod 10 clouds at position 1? (alive-only model, track_build.py stream)

Notation (a = integer operand 1..400, r = a mod 10; nothing is centred except where a spread around a residue's own
mean is the quantity measured):
  X        residual stream entering block b's MLP (mid_b); Y = MLP(X); X' = X + Y the stream leaving the block
  D, D'    the 9-dim period-10 subspaces of X and X' (track_analyze.basis); z = D^T X
  Delta    = MLP(X) - MLP(X - Pi_D X): what the MLP writes because its input contains D
  Y0       = MLP(X - Pi_D X): what it writes without D (made from other features of a)
  Lz       the least-squares linear map of z onto Delta (no intercept): the linear part of Delta
  Dnl      = Delta - Lz: the non-linear part
Clouds are measured in D' coordinates: for a field F, c_r = mean over the 40 a with a mod 10 = r of D'^T F(a),
s_r = RMS distance of D'^T F(a) to c_r, sep(r, r') = ||c_r - c_r'|| / sqrt((s_r^2 + s_r'^2)/2).
Variants of the outgoing stream: X + Y (actual), X + Delta + Y0 (same), X + Lz + Y0 (non-linear part removed),
X + Delta (Y0 removed), X + Y0 (Delta removed), X (no MLP).
"Prototype share" of a field F: share of sum_a ||D'^T F(a)||^2 reproduced by its residue means (uncentred).
Alignment of a field F with the input's within-class deviation e(a) = D'^T X(a) - c_r(X): beta = <F_dev, e> / <e, e>
summed over a, F_dev the field's deviation from its own residue means (beta = -1: F cancels the input's spread).

    python mlp_sharpen.py
"""

import json

import numpy as np
from mod10_L0 import A
from track_analyze import basis
from track_build import OUT, load_blocks

R = A % 10
BLOCKS = (0, 5, 11, 12, 13, 14, 15, 16)


def stats(Z: np.ndarray) -> dict:
    c = np.stack([Z[r == R].mean(0) for r in range(10)])
    s = np.asarray([np.sqrt(((Z[r == R] - c[r]) ** 2).sum(1).mean()) for r in range(10)])
    sep = [
        np.linalg.norm(c[r] - c[q]) / np.sqrt((s[r] ** 2 + s[q] ** 2) / 2)
        for r in range(10)
        for q in range(r + 1, 10)
    ]
    return {"sep_min": float(min(sep)), "sep_median": float(np.median(sep)), "spread": float(np.sqrt((s**2).mean())),
            "radius": float(np.linalg.norm(c, axis=1).mean())}  # fmt: skip


def dev(Z: np.ndarray) -> np.ndarray:
    c = np.stack([Z[r == R].mean(0) for r in range(10)])
    return Z - c[R]


def main() -> None:
    st = np.load(OUT / "stream.npz")
    blocks, _, _ = load_blocks()
    res = {}
    for b in BLOCKS:
        blk = blocks[b]
        X = st[f"mid_{b}"].astype(np.float64)
        _, D = basis(X)
        Y = blk.mlp(X)
        Y0 = blk.mlp(X - (X @ D) @ D.T)
        delta = Y - Y0
        z = X @ D
        L, *_ = np.linalg.lstsq(z, delta, rcond=None)
        Lz = z @ L
        _, D2 = basis(X + Y)
        proj = {
            k: v @ D2
            for k, v in {
                "X": X,
                "Y": Y,
                "Delta": delta,
                "Y0": Y0,
                "Lz": Lz,
                "Dnl": delta - Lz,
            }.items()
        }
        e = dev(proj["X"])
        out = {
            "variants": {
                "actual X+Y": stats(proj["X"] + proj["Y"]),
                "X + Lz + Y0 (no non-linear part)": stats(proj["X"] + proj["Lz"] + proj["Y0"]),
                "X + Delta (no Y0)": stats(proj["X"] + proj["Delta"]),
                "X + Y0 (no Delta)": stats(proj["X"] + proj["Y0"]),
                "X (no MLP)": stats(proj["X"]),
            },
            "alone": {k: stats(proj[k]) for k in ("Y", "Delta", "Lz", "Dnl", "Y0")},
            "prototype_share": {k: float(1 - (dev(proj[k]) ** 2).sum() / (proj[k] ** 2).sum()) for k in ("X", "Y", "Delta", "Lz", "Dnl", "Y0")},
            "beta_vs_input_spread": {k: float((dev(proj[k]) * e).sum() / (e * e).sum()) for k in ("Y", "Delta", "Lz", "Dnl", "Y0")},
            "energy_in_Dprime": {k: float((proj[k] ** 2).sum()) for k in ("X", "Y", "Delta", "Lz", "Dnl", "Y0")},
        }  # fmt: skip
        res[b] = out
        v = out["variants"]
        print(f"block {b}: " + " | ".join(f"{k}: sep {s['sep_min']:.2f}/{s['sep_median']:.2f} spread {s['spread']:.2f} rad {s['radius']:.2f}"
                                          for k, s in v.items()), flush=True)  # fmt: skip
        print("   proto", {k: round(x, 2) for k, x in out["prototype_share"].items()},
              "beta", {k: round(x, 2) for k, x in out["beta_vs_input_spread"].items()}, flush=True)  # fmt: skip
    (OUT / "sharpen.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
