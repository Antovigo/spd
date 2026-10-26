"""Where the spread in magnitude of the copied a mod 10 at "=" comes from, in the first Fourier plane (block 16,
set "a", 2,000 prompts; alive-only and original models).

E' is the stream at "=" (position 4) right after block 16's attention write: E' = E + sum_j O_j, with E the stream
entering block 16 and O_j the attention write from source position j (0..4). P_1 is the k = 1 Fourier plane
(4096 x 2, orthonormal) of E''s residue means, residue r = a mod 10. For a prompt n of residue r, z = P_1^T E'(n) is
its point in the plane, u_r the unit vector of the residue-r mean, and rho = z . u_r its radial coordinate (the
magnitude along its residue's direction). The radial spread is sum_r sum_{n in r} (rho - mean_r rho)^2.
rho splits exactly into parts (E, O_0, O_1 from head 21, O_1 from the other heads, O_2, O_3, O_4); a part's share is
the within-residue covariance of its rho with the total rho over the within-residue variance of rho.
Head 21's part is rho_21 = p_21 * c, with p_21 its attention weight on position 1 and c = rho of its value write at
weight 1, which depends on a only. Its spread splits to first order into the weight (p_21 - mean) * mean c, the value
p_21 mean * (c - mean c), and the product of the two deviations.

    python p1_magnitude.py
"""

import glob
import json
from pathlib import Path

import numpy as np
from copy_analyze import attention_parts, copied
from copy_build import OUT
from copy_figs import SUB, planes
from mod10_L0 import HUB
from smear_a import orig_parts
from track_build import load_blocks

L, S, H = 16, 1, 21


def within(y, lab):
    d = y.astype(np.float64).copy()
    for r in range(10):
        d[lab == r] -= y[lab == r].mean()
    return d


def r2_within(y, lab, g):
    """Share of the within-residue variance of y explained by the group label g (nested in the residue)."""
    d = within(y, lab)
    m = np.zeros_like(d)
    for u in np.unique(g):
        m[g == u] = d[g == u].mean()
    return float(1 - ((d - m) ** 2).sum() / (d**2).sum())


def main():
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    st = np.load(OUT / "stream_a.npz")
    sub = SUB["a"]
    a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
    lab = a % 10
    blk = load_blocks()[0][L]
    out = {}
    for name in ("alive-only", "original"):
        if name == "alive-only":
            X = st[f"X{L}"][sub].astype(np.float64)
            parts = attention_parts(blk, X)
            p21 = parts["p"][:, H, S]
            Osrc = np.stack([copied(blk, parts, j) for j in range(5)], 1)
            O21 = copied(blk, parts, S, heads=np.array([H]))
        else:
            X = np.load(OUT / "orig_cache_a.npz")["Xor"].astype(np.float64)
            p, Osrc, O21 = orig_parts(snap, wmap, cfg, X)
            p21 = p[:, H, S]
        E = X[:, 4]
        Ep = E + Osrc.sum(1)
        # the copy alone (the write from position 1, as in copy_planes.png) in its own k = 1 plane
        Pc = planes(Osrc[:, S], lab)[0][0]
        muc = np.stack([(Osrc[lab == q, S] @ Pc).mean(0) for q in range(10)])
        uc = (muc / np.linalg.norm(muc, axis=1, keepdims=True))[lab]
        zc = Osrc[:, S] @ Pc
        rc = (zc * uc).sum(1)
        tc = (zc * np.stack([-uc[:, 1], uc[:, 0]], 1)).sum(1)
        dc_ = within(rc, lab)
        r21 = ((O21 @ Pc) * uc).sum(1)
        cc = r21 / p21
        pmc, cmc = np.zeros_like(p21), np.zeros_like(cc)
        for q in range(10):
            pmc[lab == q], cmc[lab == q] = p21[lab == q].mean(), cc[lab == q].mean()
        dpc, dcc, d21c = within(p21, lab), within(cc, lab), within(r21, lab)
        copy = {
            "radial_share_of_within_spread": float(
                (dc_**2).sum() / ((dc_**2).sum() + (within(tc, lab) ** 2).sum())
            ),
            "rho_within_cv_by_residue": [
                float(rc[lab == q].std() / rc[lab == q].mean()) for q in range(10)
            ],
            "share_head21": float((d21c * dc_).sum() / (dc_ * dc_).sum()),
            "share_other_heads": float((within(rc - r21, lab) * dc_).sum() / (dc_ * dc_).sum()),
            "head21_weight_p21": float(((dpc * cmc) * d21c).sum() / (d21c * d21c).sum()),
            "head21_value_c": float(((pmc * dcc) * d21c).sum() / (d21c * d21c).sum()),
            "head21_product": float((within(dpc * dcc, lab) * d21c).sum() / (d21c * d21c).sum()),
            "within_r2_rho": {
                "exact a": r2_within(rc, lab, a),
                "op": r2_within(rc, lab, op),
                "b and op": r2_within(rc, lab, b * 2 + op),
            },  # fmt: skip
            "c_within_r2_exact_a": r2_within(cc, lab, a),
            "corr_within_c_a": float(np.corrcoef(dcc, within(a.astype(float), lab))[0, 1]),
            "c_mean_by_a_quintile": [
                float(cc[(a > 80 * i) & (a <= 80 * (i + 1))].mean()) for i in range(5)
            ],
            "within_cv_c": float(dcc.std() / np.abs(cc).mean()),
            "within_cv_p21": float(dpc.std() / p21.mean()),
        }
        P1 = planes(Ep, lab)[0][0]
        mu = np.stack([(Ep[lab == r] @ P1).mean(0) for r in range(10)])
        u = (mu / np.linalg.norm(mu, axis=1, keepdims=True))[lab]  # (n, 2)
        perp = np.stack([-u[:, 1], u[:, 0]], 1)

        def rho(V, P1=P1, u=u):
            return ((V @ P1) * u).sum(1)

        z = Ep @ P1
        rho_t, tau_t = (z * u).sum(1), (z * perp).sum(1)
        pieces = {"E (before the write)": rho(E), "O_0 <BOS>": rho(Osrc[:, 0]), "O_1 head 21": rho(O21),
                  "O_1 other heads": rho(Osrc[:, 1] - O21), "O_2 op": rho(Osrc[:, 2]), "O_3 b": rho(Osrc[:, 3]),
                  "O_4 =": rho(Osrc[:, 4])}  # fmt: skip
        d = within(rho_t, lab)
        r = {
            "copy_alone": copy,
            "rho_mean_by_residue": [float(rho_t[lab == q].mean()) for q in range(10)],
            "rho_within_sd": float(d.std()),
            "rho_within_cv_by_residue": [
                float(rho_t[lab == q].std() / rho_t[lab == q].mean()) for q in range(10)
            ],
            "radial_share_of_within_spread": float(
                (d**2).sum() / ((d**2).sum() + (within(tau_t, lab) ** 2).sum())
            ),
            "share_by_part": {
                k: float((within(v, lab) * d).sum() / (d * d).sum()) for k, v in pieces.items()
            },
            "mean_rho_by_part": {k: float(v.mean()) for k, v in pieces.items()},
            "within_r2_rho": {
                "exact a": r2_within(rho_t, lab, a),
                "op": r2_within(rho_t, lab, op),
                "b and op": r2_within(rho_t, lab, b * 2 + op),
            },  # fmt: skip
        }
        c = pieces["O_1 head 21"] / p21
        dp, dc = within(p21, lab), within(c, lab)
        pm, cm = np.zeros_like(p21), np.zeros_like(c)
        for q in range(10):
            pm[lab == q], cm[lab == q] = p21[lab == q].mean(), c[lab == q].mean()
        d21 = within(pieces["O_1 head 21"], lab)
        r["head21_split"] = {
            "weight p_21": float(((dp * cm) * d21).sum() / (d21 * d21).sum()),
            "value c (a only)": float(((pm * dc) * d21).sum() / (d21 * d21).sum()),
            "product of deviations": float((within(dp * dc, lab) * d21).sum() / (d21 * d21).sum()),
            "within_sd_p21": float(dp.std()),
            "within_cv_c": float(dc.std() / np.abs(c).mean()),
            "within_cv_p21": float(dp.std() / p21.mean()),
        }
        r["c_within_r2_exact_a"] = r2_within(c, lab, a)
        r["corr_within_c_a"] = float(np.corrcoef(dc, within(a.astype(float), lab))[0, 1])
        r["c_mean_by_a_quintile"] = [
            float(c[(a > 80 * i) & (a <= 80 * (i + 1))].mean()) for i in range(5)
        ]
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    (OUT / "p1_magnitude.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
