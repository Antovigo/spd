"""The block-15 readers g19, g29 (gate) and u32 (up) at "=" rewritten through block 15's attention: first in terms of
the activations of block 15's o components at "=", then, for the part read from the token b, in terms of the activations
of block 15's v components at position 3, with the attention pattern split into a constant, a part that depends on b
alone, and the rest.

Input: mod50_eq.py's chain.npz and proj.npz, block 15's attention components (variance/alive_only_attn/L15) and their
stored activations. Nothing is centred: zero is the zero of the activation space.

Notation (prompt x = `<BOS> a op b =`, a, b integers 1..400, op in {+, -}; query position 4, "="):
  c              a reader (L15.gate.19, L15.gate.29, L15.up.32); r_c = gamma * V_c / ||V_c|| as in mod50_eq.py
  rho(x)         the RMS of mid_15 at "=" (the stream block 15's MLP reads); f_c(x) = r_c . mid_15(x) / rho(x)
  h              a query head, 0..31; kv(h) = floor(h / 4) its key/value head, 0..7; d in 0..127 a head coordinate
  s              a source (key) position, 0..4 (BOS, a, op, b, "=")
  p_hs(x)        head h's attention weight from "=" to source s
  o              an alive o component of block 15: read vector V_o (rows h*128 + d), write vector U_o;
                 t_o(x) its activation at "=", t_o^s(x) the part read from source s (sum over s = t_o);
                 w_co = r_c . U_o
  k              an alive v component of block 15: read vector V_v,k, write vector U_v,k in R^1024 (rows kv*128 + d);
                 tau_k(x) = V_v,k . RMSNorm(pre_15 at position 3): its activation at position 3 (the token b)
  B_ohk          = sum_d V_o[h*128 + d] U_v,k[kv(h)*128 + d]: what v component k feeds o component o through head h
  A_chk          = sum_o w_co B_ohk
Exact identities:
  f_c            = (r_c . pre_15 at "=") / rho + sum_o w_co t_o / rho
  t_o^3          = sum_h p_h3 sum_k tau_k B_ohk
  E_c (the source-b term of f_c) = sum_o w_co t_o^3 / rho = sum_h sum_k p_h3 tau_k A_chk / rho
Pattern split (per op): p_h3(x) = pbar_h + beta_h(b) + gamma_h(x), pbar_h the mean of p_h3 over the op's 160,000
  prompts, beta_h(b) the mean over its 400 prompts with that b minus pbar_h (the part that depends on b alone), gamma_h
  the rest (what depends on a). E_c = E_c^const + E_c^b + E_c^rest, each term with p_h3 replaced by its part.
Per-component terms: e_co = w_co t_o / rho (o component o in f_c); E_ck = tau_k sum_h p_h3 A_chk / rho (v component k in
  E_c, sum over k = E_c); T_ok = tau_k sum_h p_h3 B_ohk (v component k in t_o^3, sum over k = t_o^3);
  E_ch = p_h3 sum_k tau_k A_chk / rho (head h in E_c).
Measures (per op, sums over its prompts, uncentred):
  share of y in z   = sum_x y z / sum_x z^2 (shares of the terms of an exact sum add up to 1)
  main components   the largest |share| first, until their |shares| add up to >= 90% of the sum of all |shares|
  class mean        the mean over the prompts with b mod 50 = r, r = 0..49 (3,200 prompts per class and op)
  attention weight statistics per head: pbar_h, rms over prompts of p_h3 - pbar_h, rms over b of beta_h, rms over prompts
  of gamma_h

Writes <run>/analysis/reader_writer/mod50_eq/{chain.json, applet_data.json, fig/chain_*.png}.

    python mod50_chain.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(os.environ.get("RUN_DIR", "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08"))
SRC_ATTN = RUN / "analysis/reader_writer/variance/alive_only_attn/L15"
OUT = RUN / "analysis/reader_writer/mod50_eq"
READERS = ("L15.gate.19", "L15.gate.29", "L15.up.32")
OPS = ("+", "-")
H, KV, DH = 32, 8, 128
N = 320000
CHUNK = 20000
COVER = 0.9
N_APPLET = 6000


def cover(share: np.ndarray) -> list[int]:
    order = np.argsort(-np.abs(share))
    cum = np.cumsum(np.abs(share[order])) / np.abs(share).sum()
    return [int(i) for i in order[: int(np.searchsorted(cum, COVER)) + 1]]


def main() -> None:
    (OUT / "fig").mkdir(parents=True, exist_ok=True)
    ch = np.load(OUT / "chain.npz")
    pr = np.load(OUT / "proj.npz")
    op, a, b = pr["op"].astype(int), pr["a"].astype(int), pr["b"].astype(int)
    p3 = ch["p"][:, :, 3].astype(np.float64)  # (N, H)
    tau = ch["tau3"].astype(np.float64)  # (N, K)
    o3 = ch["o3"].astype(np.float64)  # (N, n_o)
    rho, f, wo = ch["rho"], ch["f"].astype(np.float64), ch["wo"]  # wo (n_o, NR)
    z = np.load(SRC_ATTN / "comps.npz")
    kind, index = z["kind"], z["index"]
    oid, vid = index[kind == "o"], index[kind == "v"]
    to = np.asarray(
        np.load(SRC_ATTN / "inner.npy", mmap_mode="r")[:, 3][:, kind == "o"], np.float64
    )
    Vo = z["V_o"].astype(np.float64).reshape(H, DH, -1)  # (H, d, n_o)
    Uv = z["U_v"].astype(np.float64).reshape(-1, KV, DH)  # (K, kv, d)
    kvh = np.arange(H) // (H // KV)
    B = np.einsum("hdo,khd->ohk", Vo, Uv[:, kvh])  # (n_o, H, K)
    A = np.einsum("oc,ohk->chk", wo, B)  # (NR, H, K)
    res: dict[str, Any] = {
        "definitions": __doc__,
        "checks": {},
        "readers": {},
        "heads": {},
        "o": {},
    }

    # exact identities
    E = np.stack([((p3 @ A[j]) * tau).sum(1) for j in range(len(READERS))], 1) / rho[:, None]
    res["checks"]["E_vs_o3_rel_max"] = float(
        np.abs(E - o3 @ wo / rho[:, None]).max() / np.abs(E).max()
    )
    res["checks"]["E_vs_proj_rel_max"] = float(np.abs(E * rho[:, None] - pr["src15"][:, 3]).max()
                                               / np.abs(E * rho[:, None]).max())  # fmt: skip
    t3 = np.stack([((p3[:20000] @ B[oi]) * tau[:20000]).sum(1) for oi in range(B.shape[0])], 1)
    res["checks"]["t3_vs_o3_rel_max"] = float(
        np.abs(t3 - o3[:20000]).max() / np.abs(o3[:20000]).max()
    )
    pre = f - to @ wo / rho[:, None]  # (r_c . pre_15) / rho
    print("checks", res["checks"], flush=True)

    cls = b % 50
    for o in range(2):
        m = op == o
        pm, bm = p3[m], b[m]
        pbar = pm.mean(0)
        bmean = np.stack([pm[bm == v].mean(0) for v in range(1, 401)])  # (400, H)
        beta = bmean[bm - 1] - pbar
        gam = pm - pbar - beta
        parts = {"const": np.broadcast_to(pbar, pm.shape), "b": beta, "rest": gam}
        Ep = {k: np.stack([((v @ A[j]) * tau[m]).sum(1) for j in range(len(READERS))], 1) / rho[m, None]
              for k, v in parts.items()}  # fmt: skip
        fm, Em, prem = f[m], E[m], pre[m]
        res["heads"][OPS[o]] = {
            "pbar": pbar.tolist(),
            "rms_dev": np.sqrt(((pm - pbar) ** 2).mean(0)).tolist(),
            "rms_beta": np.sqrt(((bmean - pbar) ** 2).mean(0)).tolist(),
            "rms_gamma": np.sqrt((gam**2).mean(0)).tolist(),
            "classmean_p": np.stack(
                [pm[cls[m] == r].mean(0) for r in range(50)]
            ).T.tolist(),  # (H, 50)
        }
        for j, c in enumerate(READERS):
            fc = fm[:, j]
            f2 = float(fc @ fc)

            def share(y: np.ndarray) -> np.ndarray:
                return (y.T @ fc) / f2 if y.ndim == 2 else float(y @ fc / f2)  # noqa: B023

            def cm(y: np.ndarray) -> list[float]:
                return [float(y[cls[m] == r].mean()) for r in range(50)]  # noqa: B023

            # o components (total activation at "=") in f_c
            eo = to[m] * wo[:, j] / rho[m, None]
            sh_o = share(eo)
            main_o = cover(sh_o)
            # source-3 part of each main o component's activation
            o_rows = []
            for oi in main_o:
                t, tt = o3[m, oi], to[m, oi]
                T = tau[m] * (pm @ B[oi])  # (n, K)
                sh_v = (T.T @ t) / float(t @ t)
                mv = cover(sh_v)
                Tc = tau[m] * (pbar @ B[oi])
                o_rows.append({
                    "o": f"L15.o.{oid[oi]}", "w": float(wo[oi, j]), "share_in_f": float(sh_o[oi]),
                    "share_of_t_from_b": float(t @ tt / (tt @ tt)), "mean_t": float(tt.mean()),
                    "rms_t": float(np.sqrt((tt**2).mean())),
                    "r2_const_pattern": float(1 - ((t - Tc.sum(1)) ** 2).sum() / (t @ t)),
                    "main_v": [{"v": f"L15.v.{vid[k]}", "share_in_t3": float(sh_v[k]),
                                "W_const": float(pbar @ B[oi, :, k])} for k in mv],
                })  # fmt: skip
            # v components in E_c and in f_c
            Ek = tau[m] * (pm @ A[j]) / rho[m, None]  # (n, K)
            sh_k = share(Ek)
            Eh = pm * (tau[m] @ A[j].T) / rho[m, None]  # (n, H)
            sh_h = share(Eh)
            Ec = Em[:, j]
            terms = {"f": fc, "E (block-15 attention from b)": Ec, "E const pattern": Ep["const"][:, j],
                     "E b-part of pattern": Ep["b"][:, j], "E rest of pattern": Ep["rest"][:, j],
                     "f - E (everything else)": fc - Ec, "pre_15 term": prem[:, j]}  # fmt: skip
            res["readers"].setdefault(c, {})[OPS[o]] = {
                "share": {k: share(v) for k, v in terms.items()},
                "classmean": {k: cm(v) for k, v in terms.items()},
                "rms_f": float(np.sqrt(f2 / m.sum())),
                "pre_share": share(prem[:, j]),
                "o_total_share": float(sh_o.sum()),
                "main_o": o_rows,
                "main_v": [
                    {
                        "v": f"L15.v.{vid[k]}",
                        "share_in_f": float(sh_k[k]),
                        "share_in_E": float(Ek[:, k] @ Ec / (Ec @ Ec)),
                    }
                    for k in cover(sh_k)
                ],  # fmt: skip
                "main_heads": [{"head": int(h), "share_in_f": float(sh_h[h])} for h in cover(sh_h)],
            }
            r = res["readers"][c][OPS[o]]
            print(c, OPS[o], {k: round(v, 3) for k, v in r["share"].items()})
            print("   main o:", [(x["o"], round(x["share_in_f"], 3), round(x["share_of_t_from_b"], 2),
                                  round(x["r2_const_pattern"], 3)) for x in o_rows])  # fmt: skip
            print("   main v:", [(x["v"], round(x["share_in_f"], 3)) for x in r["main_v"]])
            print(
                "   heads:",
                [(x["head"], round(x["share_in_f"], 3)) for x in r["main_heads"]],
                flush=True,
            )
    (OUT / "chain.json").write_text(json.dumps(res, indent=1))

    # applet data: f for the three readers on N_APPLET random prompts per op
    rng = np.random.default_rng(0)
    pts = []
    for o in range(2):
        idx = np.sort(rng.permutation(np.flatnonzero(op == o))[:N_APPLET])
        pts += [[o, int(a[i]), int(b[i]), *[round(float(v), 4) for v in f[i]]] for i in idx]
    (OUT / "applet_data.json").write_text(json.dumps({"readers": list(READERS), "fields": ["op", "a", "b", *READERS],
                                                      "points": pts}))  # fmt: skip
    figures(res)


def figures(res: dict[str, Any]) -> None:
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    r50 = np.arange(50)
    style = {"f": ("k", "-", 1.6), "E (block-15 attention from b)": ("#2a5db0", "-", 1.2),
             "E const pattern": ("#2a5db0", "--", 1.0), "E b-part of pattern": ("#d9730d", "-", 1.0),
             "E rest of pattern": ("#7a7a7a", ":", 1.0), "f - E (everything else)": ("#2e8b57", "-", 1.0)}  # fmt: skip
    for o, sym in enumerate(OPS):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
        for ax, c in zip(axs, READERS, strict=True):
            cmv = res["readers"][c][sym]["classmean"]
            for k, (col, ls, lw) in style.items():
                ax.plot(r50, cmv[k], color=col, ls=ls, lw=lw, label=k)
            ax.axhline(0, color="0.8", lw=0.5)
            ax.set_xlabel("b mod 50")
            ax.set_ylabel("class mean")
            ax.set_title(f"{c}, prompts a {sym} b", fontsize=8)
        axs[0].legend(frameon=False, fontsize=6)
        fig.savefig(OUT / f"fig/chain_classmeans_{'add' if o == 0 else 'sub'}.png")
        plt.close(fig)

        heads = sorted({x["head"] for c in READERS for x in res["readers"][c][sym]["main_heads"]})
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        cmp = np.asarray(res["heads"][sym]["classmean_p"])
        for h in heads:
            ax.plot(r50, cmp[h], lw=1, label=f"head {h}")
        ax.set_xlabel("b mod 50")
        ax.set_ylabel("class mean of p_h3 (weight from = to b)")
        ax.set_ylim(0, None)
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.set_title(f"Attention weight on b of the main heads, prompts a {sym} b", fontsize=8)
        fig.savefig(OUT / f"fig/chain_pattern_{'add' if o == 0 else 'sub'}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
