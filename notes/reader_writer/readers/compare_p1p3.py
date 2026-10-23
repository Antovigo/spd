"""The mod-10 representations at the token a (position 1) and the token b (position 3), blocks 0..16: RMS compensation
and the effect of context (a, op) copied into position 3. Alive-only model; streams of track_build.py (position 1,
a = 1..400) and track_build_p3.py (position 3: op x ten a x b = 1..400). Nothing is centred.

Notation (per block l, at the MLP input boundary; r = a mod 10 at position 1, r = b mod 10 at position 3):
  X1, X3      residual stream entering block l's MLP at positions 1, 3;  rho = sqrt(mean of X^2 over the 4096 coordinates + eps)
  x = gamma * X / rho   the MLP input (post-attention RMSNorm, gamma its weight)
  A_k(F)      = ||Psi_k(F)||, the frequency-k Fourier amplitude of a field's residue means (k = 1..5)
  R_res_k     = A_k(X3) / A_k(X1);  R_norm_k = A_k(x3) / A_k(x1);  rho ratio = mean rho(X3) / mean rho(X1)
  compensated = 1 - log R_norm / log R_res (share of the residual-stream shrinking, in log scale, undone by the norm)
  C           context subspace at position 3: the frequency 1..5 Fourier components of the means over a mod 10 (9 dims)
              and the difference of the two operation means (1 dim), with b's period-10 subspace D projected out
  Delta_ctx   = MLP(x) - MLP(x_C), x_C = gamma * (X3 - Pi_C X3) / rho(X3): the MLP's response to the context content
              (norm held fixed); its part in D' (b's subspace of the stream leaving the block) is split into a
              per-context offset, a part periodic in b, and the rest
  readers     mod-10 readers: r2 of t_c = x . V_c explained by the input projected onto D (uncentred) >= 0.5;
              ctx share = sum (t_c(x) - t_c(x_C))^2 / sum t_c(x)^2
Writes <run>/analysis/reader_writer/compare_p1p3/compare.json.

    python compare_p1p3.py
"""

import json

import numpy as np
from track_build import OUT as O1
from track_build import load_blocks, rmsnorm
from track_build_p3 import OUT as O3

OUT = O1.parent / "compare_p1p3"
EPS = 1e-5


def psi(F: np.ndarray, lab: np.ndarray) -> np.ndarray:
    mu = np.stack([F[lab == r].mean(0) for r in range(10)])
    return np.exp(-2j * np.pi * np.outer(np.arange(1, 6), np.arange(10)) / 10) @ mu


def span(ps: np.ndarray) -> np.ndarray:
    return np.linalg.qr(
        np.concatenate([np.stack([p.real, p.imag], 1) for p in ps[:4]] + [ps[4].real[:, None]], 1)
    )[0]


def plane(p: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def r2_readers(x: np.ndarray, D: np.ndarray, V: np.ndarray) -> np.ndarray:
    f = x @ V
    g = (x @ D) @ (D.T @ V)
    return 1 - ((f - g) ** 2).sum(0) / (f**2).sum(0)


def clouds(z: np.ndarray, lab: np.ndarray) -> dict:
    c = np.stack([z[lab == r].mean(0) for r in range(10)])
    s = np.asarray([np.sqrt(((z[lab == r] - c[r]) ** 2).sum(1).mean()) for r in range(10)])
    sep = [
        np.linalg.norm(c[r] - c[q]) / np.sqrt((s[r] ** 2 + s[q] ** 2) / 2)
        for r in range(10)
        for q in range(r + 1, 10)
    ]
    return {"sep_min": float(min(sep)), "sep_median": float(np.median(sep))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s1, s3 = np.load(O1 / "stream.npz"), np.load(O3 / "stream.npz")
    lab1 = np.arange(1, 401) % 10
    lab3, la3, op3, ctx3 = s3["b"] % 10, s3["a"] % 10, s3["op"], s3["ctx"]
    mech = np.isin(s3["a"], (7, 171))
    blocks, _, _ = load_blocks()
    res = {}
    for blk in blocks:
        b = blk.b
        X1, X3 = s1[f"mid_{b}"].astype(np.float64), s3[f"mid_{b}"].astype(np.float64)
        x1, x3 = rmsnorm(X1, blk.g2), rmsnorm(X3, blk.g2)
        rho1 = np.sqrt((X1**2).mean(1) + EPS)
        rho3 = np.sqrt((X3**2).mean(1) + EPS)
        P1, P3 = psi(X1, lab1), psi(X3, lab3)
        p1, p3 = psi(x1, lab1), psi(x3, lab3)
        Rres = np.abs(np.linalg.norm(P3, axis=1) / np.linalg.norm(P1, axis=1))
        Rnorm = np.abs(np.linalg.norm(p3, axis=1) / np.linalg.norm(p1, axis=1))
        comp = [
            float(1 - np.log(n) / np.log(r)) if abs(np.log(r)) > 0.05 else None
            for r, n in zip(Rres, Rnorm, strict=True)
        ]
        cosres = [
            float(np.mean(np.linalg.svd(plane(P1[k]).T @ plane(P3[k]), compute_uv=False)))
            for k in range(5)
        ]
        # readers: period-10 response at both positions
        V = np.concatenate([blk.m["V_gate"], blk.m["V_up"]], 1)
        names = [f"{k[0]}{int(i)}" for k, i in zip(blk.kind, blk.index, strict=True) if k != "down"]
        D1, D3 = span(p1), span(p3)
        r1, r3 = r2_readers(x1, D1, V), r2_readers(x3, D3, V)
        mod = np.nonzero((r1 >= 0.5) | (r3 >= 0.5))[0]
        t1, t3 = x1 @ V, x3 @ V
        ratio_reader = np.sqrt(
            (np.abs(psi(t3[:, mod], lab3)) ** 2).sum(0)
            / (np.abs(psi(t1[:, mod], lab1)) ** 2).sum(0)
        )
        # context subspace and its effect
        pa = psi(X3, la3)
        dop = X3[op3 == 0].mean(0) - X3[op3 == 1].mean(0)
        Craw = np.concatenate(
            [np.stack([p.real, p.imag], 1) for p in pa[:4]] + [pa[4].real[:, None], dop[:, None]], 1
        )
        Dres = span(P3)
        Craw = Craw - Dres @ (Dres.T @ Craw)
        C = np.linalg.qr(Craw)[0]
        XC = X3 - (X3 @ C) @ C.T
        xC = blk.g2 * XC / rho3[:, None]
        tC = xC @ V
        ctx_share = ((t3 - tC) ** 2).sum(0) / (t3**2).sum(0)
        Y = blk.mlp_from_t(t3[:, : blk.m["V_gate"].shape[1]], t3[:, blk.m["V_gate"].shape[1] :])[0]
        YC = blk.mlp_from_t(tC[:, : blk.m["V_gate"].shape[1]], tC[:, blk.m["V_gate"].shape[1] :])[0]
        dctx = Y - YC
        Xout = X3 + Y
        Dout = span(psi(Xout, lab3))
        z = dctx @ Dout
        ctx_mean = np.stack([z[ctx3 == k].mean(0) for k in np.unique(ctx3)])[
            np.searchsorted(np.unique(ctx3), ctx3)
        ]
        bmu = np.stack([(z - ctx_mean)[lab3 == r].mean(0) for r in range(10)])[lab3]
        tot = (z**2).sum()
        split = {"offset": float((ctx_mean**2).sum() / tot), "b_periodic": float((bmu**2).sum() / tot),
                 "rest": float(((z - ctx_mean - bmu) ** 2).sum() / tot)}  # fmt: skip
        ydout = Y @ Dout
        out = {
            "R_res": Rres.tolist(), "R_norm": Rnorm.tolist(), "rho_ratio": float(rho3.mean() / rho1.mean()),
            "compensated": comp, "cos_planes_res": cosres,
            "D_share_p1": float(((X1 @ span(P1)) ** 2).sum() / (X1**2).sum()),
            "D_share_p3": float(((X3 @ Dres) ** 2).sum() / (X3**2).sum()),
            "n_mod_readers": int(len(mod)), "mod_readers": [names[i] for i in mod],
            "reader_amp_ratio": ratio_reader.tolist(), "reader_amp_ratio_median": float(np.median(ratio_reader)) if len(mod) else None,
            "C_share_of_X3": float(((X3 @ C) ** 2).sum() / (X3**2).sum()),
            "ctx_share_mod_readers": {names[i]: float(ctx_share[i]) for i in mod},
            "ctx_share_all_median": float(np.median(ctx_share)),
            "top_ctx_readers": [[names[i], float(ctx_share[i]), float(r3[i])] for i in np.argsort(-ctx_share)[:6]],
            "delta_ctx_in_Dout_over_mlp_in_Dout": float(np.sqrt(tot / (ydout**2).sum())),
            "delta_ctx_split_in_Dout": split,
            "clouds_actual": clouds(Xout @ Dout, lab3), "clouds_no_ctx_response": clouds((X3 + YC) @ Dout, lab3),
        }  # fmt: skip
        # per-reader contributions to Delta_ctx in D' (1,600 prompts)
        G = blk.m["V_gate"].shape[1]
        hg = t3[mech, :G] @ blk.m["U_gate"]
        hu = t3[mech, G:] @ blk.m["U_up"]
        sg = hg / (1 + np.exp(-hg))
        o = sg * hu
        Vd, Ud = blk.m["V_down"], blk.m["U_down"]
        contrib = []
        for j in range(len(names)):
            d = t3[mech, j] - tC[mech, j]
            if j < G:
                h2 = hg - np.outer(d, blk.m["U_gate"][j])
                dc = ((o - (h2 / (1 + np.exp(-h2))) * hu) @ Vd) @ Ud
            else:
                dc = ((sg * np.outer(d, blk.m["U_up"][j - G])) @ Vd) @ Ud
            contrib.append(float(((dc @ Dout) ** 2).sum()))
        contrib = np.asarray(contrib)
        oc = np.argsort(-contrib)[:6]
        out["top_ctx_writers_into_D"] = [
            [names[i], float(contrib[i] / contrib.sum()), float(r3[i]), float(ctx_share[i])]
            for i in oc
        ]
        res[b] = out
        print(f"block {b}: R_res {np.round(Rres, 2)} rho {out['rho_ratio']:.2f} R_norm {np.round(Rnorm, 2)} "
              f"comp {[None if c is None else round(c, 2) for c in comp]} cos {np.round(cosres, 2)} | mod readers {len(mod)} "
              f"amp ratio med {out['reader_amp_ratio_median']} | ctx share mod med "
              f"{np.median(list(out['ctx_share_mod_readers'].values())) if len(mod) else None} | dctx/mlp in D' "
              f"{out['delta_ctx_in_Dout_over_mlp_in_Dout']:.2f} split {({k: round(v, 2) for k, v in split.items()})} | "
              f"sep {out['clouds_actual']['sep_min']:.2f} vs no-ctx {out['clouds_no_ctx_response']['sep_min']:.2f} | "
              f"top writers {[(n, round(s, 2), round(r, 2)) for n, s, r, _ in out['top_ctx_writers_into_D'][:3]]}", flush=True)  # fmt: skip
        (OUT / "compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
