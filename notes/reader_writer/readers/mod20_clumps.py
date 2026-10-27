"""Dimension of the four quarter clumps after block 12 (position 1; alive-only and original models). Nothing is centred.

Notation (a = integer operand 1..400; s = a mod 20; Q = floor(s / 5), integer 0..3; 100 operands per quarter):
  X(a)      residual stream at a boundary; P its frequency-1 plane of a mod 20 (mod20_track.plane); D10 its 9-dim a mod 10
            subspace (track_analyze.basis)
  mu_s      mean of X over the 20 operands with a mod 20 = s; O_s = (mu_s - mu_{s+10}) / 2 for s = 0..9 (the odd part)
  nu_Q      mean of X over the 100 operands of quarter Q; G = [nu_1 - nu_0, nu_2 - nu_0, nu_3 - nu_0] (4096 x 3)
  odd_out   share of sum_s ||O_s||^2 outside P, and the same with the 400 values of a randomly relabelled (100 draws, seed 0,
            P refitted each time): its median and 99th percentile
  G sv      singular values of G (the affine extent of the four clump centres); G_in_P, G_in_D10: share of ||G||_F^2 in P and
            in D10; G_rest: the share in neither (outside span(P, D10))
Writes <run>/analysis/reader_writer/track_mod20/clumps.json.

    python mod20_clumps.py
"""

import json

import numpy as np
from mod20_track import OUT, S20, means, plane
from track_analyze import basis
from track_build import OUT as AOUT
from track_build_orig import OUT as OOUT

Q = S20 // 5


def odd_out(X: np.ndarray, lab: np.ndarray) -> float:
    mu = means(X, lab, 20)
    Od = (mu[:10] - mu[10:]) / 2
    p = np.exp(-2j * np.pi * np.arange(20) / 20) @ mu
    P = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)[0]
    return float(((Od - (Od @ P) @ P.T) ** 2).sum() / (Od**2).sum())


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    for name, src in (("alive", AOUT), ("orig", OOUT)):
        st = np.load(src / "stream.npz")
        for b in (12, 13, 14, 15, 17):
            X = st[f"pre_{b}"].astype(np.float64)
            P, D10 = plane(X), basis(X)[1]
            null = [odd_out(X, rng.permutation(S20)) for _ in range(100)]
            nu = means(X, Q, 4)
            G = (nu[1:] - nu[0]).T
            sv = np.linalg.svd(G, compute_uv=False)
            B = np.linalg.qr(np.concatenate([P, D10], 1))[0]
            tot = (G**2).sum()
            mu = means(X, S20, 20)
            Od = (mu[:10] - mu[10:]) / 2
            osv = np.linalg.svd(Od, compute_uv=False)
            r = {
                "odd_out": odd_out(X, S20), "odd_out_null_median": float(np.median(null)),
                "odd_out_null_p99": float(np.quantile(null, 0.99)),
                "odd_sv_rel": (osv / osv[0]).tolist(),
                "G_sv": sv.tolist(), "G_sv_rel": (sv / sv[0]).tolist(),
                "G_in_P": float(((P.T @ G) ** 2).sum() / tot), "G_in_D10": float(((D10.T @ G) ** 2).sum() / tot),
                "G_rest": float(1 - ((B.T @ G) ** 2).sum() / tot),
            }  # fmt: skip
            out[f"{name}_pre_{b}"] = r
            print(
                name,
                b,
                {
                    k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v[:5]])
                    for k, v in r.items()
                },
                flush=True,
            )
    (OUT / "clumps.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
