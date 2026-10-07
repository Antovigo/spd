"""Exploration runs of V2 (printed tables); the report figures come from v2_report.py."""

import sys

import numpy as np
from v2 import (
    A,
    Q,
    classes,
    fourier,
    reader_dirs,
    residual_svd,
    span_basis,
    spline,
    stagewise,
    stream,
    variance_table,
)


def sites() -> dict:
    out = {"embedding (l=0, all 4096 dims)": (stream(0), None)}
    for name, (blk, pt, lpos) in {
        "site 1: L0 attention input": (0, "attn", 0),
        "site 2: L0 MLP input": (0, "mlp", 1),
    }.items():
        Vt, names, Ain = reader_dirs(blk, pt)
        X = stream(lpos)
        Qb = span_basis(Vt)
        Z, W = X @ Qb, Qb.T @ Vt.T
        rho = np.sqrt((X**2).mean(1) + 1e-5)
        err = np.abs(Z @ W - Ain * rho[:, None]).max() / np.abs(Ain * rho[:, None]).max()
        print(
            f"{name}: n={len(names)} k={Qb.shape[1]} identity check (raw read = Z W) rel err {err:.1e}"
        )
        out[name] = (Z, W)
    return out


def H_v2b() -> list[Q]:
    tens = np.where((A >= 10) & (A <= 99), A // 10, 0)
    return [
        Q("magnitude curve (1 dir)", "curve", spline(), 1),
        Q("smooth in log a, 2 more dirs", "curve", spline(), 2),
        Q("one digit [a <= 9]", "fixed", (A <= 9).astype(float)[:, None]),
        Q("a mod 10: circle (period 10)", "fixed", fourier(10, [1])),
        Q("a mod 10: harmonics 2-5", "fixed", fourier(10, [2, 3, 4, 5])),
        Q("a mod 3", "fixed", classes(A % 3)),
        Q("tens digit (10..99)", "fixed", classes(tens)),
        Q("place code: smooth in a (25 dims)", "fixed", spline(25, log=False)),
        Q("one-hot remainder", "fixed", np.eye(100)),
    ]


def H_v2a() -> list[Q]:
    return [
        Q("magnitude curve (1 dir)", "curve", spline(), 1),
        Q("smooth in a, 2 more dirs", "curve", spline(), 2),
        Q("digit count", "fixed", classes(np.digitize(A, [10, 100]))),
        Q("a mod 10: circle (period 10)", "fixed", fourier(10, [1])),
        Q("a mod 10: harmonics 2-5", "fixed", fourier(10, [2, 3, 4, 5])),
        Q("a mod 3", "fixed", classes(A % 3)),
        Q("tens digit", "fixed", classes(A // 10)),
        Q("one-hot remainder", "fixed", np.eye(100)),
    ]


if __name__ == "__main__":
    H = globals()[sys.argv[1]]()
    for name, (Z, W) in sites().items():
        print("==", name)
        for r in variance_table(Z, W, H):
            print(
                f"  {r['quantity']:<36} d={r['dims']:3d} incr={r['incr']:.3f} chance={r['chance']:.3f} excess={r['excess']:+.3f} "
                f"cum={r['cum']:.3f} cv={r['cv_cum']:.3f} readers>10%={r['readers_10pct']}"
            )
        mu, parts, _ = stagewise(Z, H[:-1])
        E = Z - mu - sum(parts)
        U, s, null = residual_svd(E)
        print(f"  residual before one-hot: top sv {np.round(s[:5], 3)} null95 {null:.3f}")
