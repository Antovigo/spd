"""Which block-0 writers (down_proj components) the 11 mod-10 readers feed, at position 1 (token a).

Alive-only model of the -08 ceiling filter (as on the "Mod 10 representations" page). Nothing is
centred: every term is measured from the activation-space zero.

Notation:
  a           integer operand 1..400 (position 1 depends on a alone)
  t_c(a)      raw inner activation of component c (x . V_c), from variance/alive_only/L0/inner.npy
  y_c,n(a)    = t_c(a) U_c,n, reader c's write to neuron n (n = 0..14335); gauge-free
  h_g,n, h_u,n  gate / up pre-activation of neuron n = sum over ALL alive gate / up readers of y_c,n
  o_n         = SiLU(h_g,n) h_u,n; t_w(a) = sum_n V_w,n o_n for writer w (V_w its read vector)
Exact zero-reference split of t_w (o_n is a product, zero when either factor is zero, so the
two-player Shapley value gives each factor o_n / 2; inside a factor the straight line from zero is
exact because h_u is linear and SiLU(h) = sigma(h) h, sigma the logistic function):
  M_w,c(a) = 1/2 sum_n V_w,n SiLU(h_g,n) y_c,n          for an up reader c
  M_w,c(a) = 1/2 sum_n V_w,n sigma(h_g,n) h_u,n y_c,n   for a gate reader c
  sum_c M_w,c(a) = t_w(a) exactly.
Reported per (reader c, writer w):
  S_w,c   = sqrt(mean_a M_w,c(a)^2) ||U_w||, RMS size of c's contribution in the residual stream
  phi_w,c = sum_a M_w,c t_w / sum_a t_w^2, uncentred share of writer w's second moment (sums to 1 over c)
  frac_w  = mean_a t_w^2 ||U_w||^2 / sum over writers of the same: writer w's share of block-0 output

    python mod10_writers.py --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mod10_L0 import SRC, A, N

MOD10 = {"u794": 0, "u31": 0, "g43": 1, "g69": 2, "u60": 3, "g62": 4, "u52": 5, "u54": 6,
         "g58": 7, "g59": 8, "g47": 9}  # fmt: skip
TOP = 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    comps = np.load(SRC / "comps.npz")
    kind, idx = comps["kind"], comps["index"]
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :], np.float64)  # (400, K)
    col = {k: np.nonzero(kind == k)[0] for k in ("gate", "up", "down")}
    Ug, Uu = comps["U_gate"].astype(np.float64), comps["U_up"].astype(np.float64)
    Vd, Ud = comps["V_down"].astype(np.float64), comps["U_down"].astype(np.float64)
    tg, tu, td = T[:, col["gate"]], T[:, col["up"]], T[:, col["down"]]
    hg, hu = tg @ Ug, tu @ Uu  # (400, 14336)
    sig = 1 / (1 + np.exp(-hg))
    silu = hg * sig
    t_rec = (silu * hu) @ Vd  # (400, D)
    rel = np.abs(t_rec - td).max(0) / np.abs(td).max(0)
    unorm = np.linalg.norm(Ud, axis=1)
    m2 = (t_rec**2).mean(0) * unorm**2
    frac = m2 / m2.sum()
    names_down = [f"d{int(i)}" for i in idx[col["down"]]]
    res: dict = {"check_rel_max": float(rel.max()), "readers": {}}
    total = np.zeros_like(t_rec)
    terms = {}
    for k, U, coef in (("up", Uu, silu), ("gate", Ug, sig * hu)):
        for j in range(U.shape[0]):
            name = f"{k[0]}{int(idx[col[k]][j])}"
            M = 0.5 * ((coef * U[j]) @ Vd) * tu[:, j : j + 1] if k == "up" else \
                0.5 * ((coef * U[j]) @ Vd) * tg[:, j : j + 1]  # fmt: skip
            total += M
            if name in MOD10:
                terms[name] = M
    res["split_rel_max"] = float((np.abs(total - t_rec).max(0) / np.abs(t_rec).max(0)).max())
    for name, M in terms.items():
        S = np.sqrt((M**2).mean(0)) * unorm
        phi = (M * t_rec).sum(0) / (t_rec**2).sum(0)
        order = np.argsort(-S)[:TOP]
        res["readers"][name] = {
            "residue": MOD10[name],
            "S_total": float(np.sqrt((S**2).sum())),
            "top": [
                {
                    "writer": names_down[w],
                    "S": float(S[w]),
                    "S_frac": float(S[w] ** 2 / (S**2).sum()),
                    "phi": float(phi[w]),
                    "writer_frac": float(frac[w]),
                }  # fmt: skip
                for w in order
            ],
        }
    (args.out / "mod10_writers.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps({"check_rel_max": res["check_rel_max"], "split_rel_max": res["split_rel_max"]})
    )
    for name, r in sorted(res["readers"].items(), key=lambda kv: kv[1]["residue"]):
        print(name, r["residue"], " ".join(
            f"{e['writer']}(S²frac {e['S_frac']:.2f}, phi {e['phi']:+.2f}, wfrac {e['writer_frac']:.3f})"
            for e in r["top"][:4]))  # fmt: skip


if __name__ == "__main__":
    main()
