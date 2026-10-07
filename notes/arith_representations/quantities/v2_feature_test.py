"""Test one candidate feature on the residual of a hypothesis set at every site: the share of the
residual's variance (reader metric) along the feature's part orthogonal to the set's structured
span, against the same share for 500 random permutations of the feature over a (each projected
the same way). p = fraction of permutations with a share at least as large.

    python v2_feature_test.py <hypothesis set> <feature name>
"""

import json
import sys
from pathlib import Path

import numpy as np
import v2_sets
from v2 import A, reader_dirs, span_basis, stagewise, stream
from v2_layers import site_list

FEATURES = {
    "two_adic": lambda: np.minimum(v2_sets.v2_adic(A), 4).astype(float),
    "repdigit": lambda: ((A % 11 == 0) & (A <= 99)).astype(float),  # 11, 22, ..., 99
}


def share(g: np.ndarray, E: np.ndarray, tot: float) -> float:
    """Share of the residual E's variance along the (unit-normalised) function g of a."""
    g = g / np.linalg.norm(g)
    return float(((g @ E) ** 2).sum() / tot)


def main() -> None:
    H = getattr(v2_sets, sys.argv[1])()[:-1]
    f = FEATURES[sys.argv[2]]()
    B = np.hstack([np.ones((100, 1))] + [q.Phi for q in H])
    Qb, _ = np.linalg.qr(B)
    P = np.eye(100) - Qb @ Qb.T
    rng = np.random.default_rng(0)
    perms = [P @ f[rng.permutation(100)] for _ in range(500)]
    fp = P @ f
    out = []
    for b, pt, lpos in site_list():
        Vt, names, _ = reader_dirs(b, pt)
        if len(names) == 0:
            continue
        Qr = span_basis(Vt)
        Z, W = stream(lpos) @ Qr, Qr.T @ Vt.T
        mu, parts, _ = stagewise(Z, H)
        E = (Z - mu - sum(parts)) @ W
        tot = (E**2).sum()

        s = share(fp, E, tot)
        null = np.array([share(g, E, tot) for g in perms])
        out.append({"site": f"L{b}.{pt}", "share": s, "null_median": float(np.median(null)),
                    "p": float((null >= s).mean())})  # fmt: skip
        print(
            f"L{b}.{pt}: share {s:.3f} (null median {np.median(null):.3f}) p = {(null >= s).mean():.3f}",
            flush=True,
        )
    Path(f"feature_test_{sys.argv[1]}_{sys.argv[2]}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
