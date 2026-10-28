"""Where the block-15 v components that feed g19, g29 and u32 get their activations at position 3 (the token b).

Model and data as in mod50_eq.py (alive-only model rebuilt from its stored activations, all 320,000 prompts
`<BOS> a op b =`, a, b integers 1..400, op in {+, -}); the main v components and the mean attention pattern come from
mod50_chain.py's chain.json. Nothing is centred: zero is the zero of the activation space.

Notation (position 3 throughout; x a prompt):
  X3(x)        pre_15 at position 3: the residual stream entering block 15 at the token b;
               sigma(x) = sqrt(mean_j X3_j(x)^2 + 1e-5), j in 1..4096 the coordinates (block 15's input RMSNorm)
  gamma1       (4096,) the gain of block 15's input RMSNorm
  targets      (1) every main v component k of chain.json (any reader, any op): d_k = gamma1 * V_k / ||V_k||, so
               f_k = d_k . X3 / sigma = tau_k / ||V_k|| (tau_k its activation);
               (2) for each reader c and op, the combination of v activations that the constant attention pattern
               carries to c: S_c = sum_k W_ck tau_k with W_ck = sum_h pbar_h A_chk (chain.json's notation; pbar_h the
               op's mean weight of head h on b), i.e. d = gamma1 * sum_k W_ck V_k and f = d . X3 / sigma
               (E_c^const = S_c / rho at "=")
  writers w    the embedding of the token b; every alive attention o component and every alive MLP down component of
               blocks 0..14 at position 3 (t_w its activation there, U_w its write vector); X3 = emb + sum_w t_w U_w
  e_w(x)       = t_w(x) (d . U_w) / sigma(x): writer w's term in the target; sum_w e_w = f exactly
  Sums over x run over the prompts of one op (160,000 each), uncentred:
  share_w      = sum_x e_w f / sum_x f^2 (sums to 1 over the writers); own_w = sum_x e_w^2 / sum_x f^2
  groups       the embedding, and each block's attention ("L<l>.attn") and MLP ("L<l>.mlp"); r2_alone(g) =
               1 - sum_x (f - e_g)^2 / sum_x f^2 with e_g the group's summed term
  class means  means over the 3,200 prompts of one op with b mod 50 = r, r = 0..49, of f and of the cumulative sum of
               the group terms in stream order (emb, L0.attn, L0.mlp, ..., L14.mlp; the last one is f)

Writes <run>/analysis/reader_writer/mod50_eq/{pos3.json, fig/pos3_*.png}.

    python mod50_pos3.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(os.environ.get("RUN_DIR", "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08"))
SRC = RUN / "analysis/reader_writer/variance/alive_only"
SRC_ATTN = RUN / "analysis/reader_writer/variance/alive_only_attn"
OUT = RUN / "analysis/reader_writer/mod50_eq"
RB = 15
READERS = ("L15.gate.19", "L15.gate.29", "L15.up.32")
OPS = ("+", "-")
H, KV, DH = 32, 8, 128
N = 320000
CHUNK = 8000
EPS = 1e-5
TOP = 30


def comps(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: (z[k].astype(np.float64) if k[0] in "UV" else z[k]) for k in z.files}


def main() -> None:
    W = np.load(OUT / "inputs.npz")
    g1, e_num = W["g1"], W["e_num"]
    chn = json.loads((OUT / "chain.json").read_text())
    ch = np.load(OUT / "chain.npz")
    mlp = {blk: comps(SRC / f"L{blk}" / "comps.npz") for blk in range(RB)}
    att = {blk: comps(SRC_ATTN / f"L{blk}" / "comps.npz") for blk in range(RB + 1)}
    inner = {blk: np.load(SRC / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB)}
    inner_a = {
        blk: np.load(SRC_ATTN / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB + 1)
    }
    a15 = att[RB]
    vid = [int(i) for i in a15["index"][a15["kind"] == "v"]]
    Vv = a15["V_v"]  # (4096, K)
    main_v = sorted({int(x["v"].split(".")[-1]) for c in READERS for op in OPS
                     for x in chn["readers"][c][op]["main_v"]})  # fmt: skip
    cols = [vid.index(k) for k in main_v]

    # constant-pattern weights W_ck per op (mod50_chain.py's A_chk)
    z = np.load(SRC_ATTN / f"L{RB}" / "comps.npz")
    Vo = z["V_o"].astype(np.float64).reshape(H, DH, -1)
    Uv = z["U_v"].astype(np.float64).reshape(-1, KV, DH)
    B = np.einsum("hdo,khd->ohk", Vo, Uv[:, np.arange(H) // (H // KV)])
    A = np.einsum("oc,ohk->chk", ch["wo"], B)  # (NR, H, K)
    names_t, D = [], []
    for k, j in zip(main_v, cols, strict=True):
        names_t.append(f"L15.v.{k}")
        D.append(g1 * Vv[:, j] / np.linalg.norm(Vv[:, j]))
    for sym in OPS:
        pbar = np.asarray(chn["heads"][sym]["pbar"])
        for c_i, c in enumerate(READERS):
            Wck = pbar @ A[c_i]  # (K,)
            names_t.append(f"S[{c}]{sym}")
            D.append(g1 * (Vv @ Wck))
    D = np.stack(D, 1)  # (4096, T)
    nt = D.shape[1]
    t_op = np.asarray(
        [None if n.startswith("L15.v") else OPS.index(n[-1]) for n in names_t], dtype=object
    )

    wnames, wgroups, UD = ["emb"], ["emb"], []
    for blk in range(RB):
        for kind, site in (("o", att[blk]), ("down", mlp[blk])):
            ids = site["index"][site["kind"] == kind]
            wnames += [f"L{blk}.{kind}.{i}" for i in ids]
            wgroups += [f"L{blk}.{'attn' if kind == 'o' else 'mlp'}"] * len(ids)
            UD.append(site[f"U_{kind}"] @ D)
    UD = np.concatenate(UD)  # (n_w - 1, T)
    gnames = list(dict.fromkeys(wgroups))
    garr = np.asarray(wgroups[1:])
    gidx = {g: np.flatnonzero(garr == g) for g in gnames[1:]}
    nw, ng = len(wnames), len(gnames)
    print("targets", nt, "writers", nw, flush=True)

    acc = {o: {"tf": np.zeros((nw - 1, nt)), "t2": np.zeros((nw - 1, nt)), "t1": np.zeros((nw - 1, nt)),
               "embf": np.zeros(nt), "emb2": np.zeros(nt), "emb1": np.zeros(nt), "Gg": np.zeros((nt, ng, ng)),
               "gf": np.zeros((ng, nt)), "f2": np.zeros(nt), "f1": np.zeros(nt), "n": 0,
               "cm": np.zeros((50, ng, nt)), "cmf": np.zeros((50, nt)), "cn": np.zeros(50)} for o in range(2)}  # fmt: skip
    checks = {"tau_rel_max": 0.0, "sigma_min": np.inf, "sigma_max": 0.0}
    rows = np.arange(N)
    op_of, b_of = rows // 160000, rows % 400 + 1
    vsel = a15["kind"] == "v"
    t0 = time.time()
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, N))
        op, b = op_of[sl], b_of[sl]
        X = e_num[b - 1].copy()
        tw = []
        for blk in range(RB):
            ta = np.asarray(inner_a[blk][sl, 2], np.float64)[:, att[blk]["kind"] == "o"]
            td = np.asarray(inner[blk][sl, 2], np.float64)[:, mlp[blk]["kind"] == "down"]
            X += ta @ att[blk]["U_o"] + td @ mlp[blk]["U_down"]
            tw += [ta, td]
        T = np.concatenate(tw, 1)  # (n, n_w - 1)
        sig = np.sqrt((X**2).mean(-1) + EPS)
        f = (X @ D) / sig[:, None]  # (n, T)
        tau = (g1 * X / sig[:, None]) @ Vv[:, cols]
        ref = np.asarray(inner_a[RB][sl, 2], np.float64)[:, vsel][:, cols]
        checks["tau_rel_max"] = max(
            checks["tau_rel_max"], float(np.abs(tau - ref).max() / np.abs(ref).max())
        )
        checks["sigma_min"] = min(checks["sigma_min"], float(sig.min()))
        checks["sigma_max"] = max(checks["sigma_max"], float(sig.max()))
        Ts = T / sig[:, None]
        eemb = (e_num[b - 1] @ D) / sig[:, None]  # (n, T)
        eg = np.stack(
            [eemb] + [(Ts[:, ix] * 1.0) @ UD[ix] for ix in gidx.values()], 1
        )  # (n, ng, T)
        assert np.abs(eg.sum(1) - f).max() <= 1e-8 * np.abs(f).max()
        for o in range(2):
            m = op == o
            Ac = acc[o]
            fm, Tm, egm = f[m], Ts[m], eg[m]
            Ac["tf"] += Tm.T @ fm
            Ac["t2"] += (Tm**2).T @ np.ones_like(fm)
            Ac["t1"] += Tm.sum(0)[:, None]
            Ac["embf"] += (egm[:, 0] * fm).sum(0)
            Ac["emb2"] += (egm[:, 0] ** 2).sum(0)
            Ac["emb1"] += egm[:, 0].sum(0)
            Ac["Gg"] += np.einsum("ngt,nht->tgh", egm, egm)
            Ac["gf"] += np.einsum("ngt,nt->gt", egm, fm)
            Ac["f2"] += (fm**2).sum(0)
            Ac["f1"] += fm.sum(0)
            Ac["n"] += int(m.sum())
            r = b[m] % 50
            np.add.at(Ac["cm"], r, egm)
            np.add.at(Ac["cmf"], r, fm)
            Ac["cn"] += np.bincount(r, minlength=50)
        if i0 % (5 * CHUNK) == 0:
            print(i0, f"{time.time() - t0:.0f}s", checks, flush=True)

    res: dict[str, Any] = {"definitions": __doc__, "checks": checks, "main_v": [f"L15.v.{k}" for k in main_v],
                           "groups": gnames, "targets": {}}  # fmt: skip
    for j, tn in enumerate(names_t):
        res["targets"][tn] = {}
        for o in range(2):
            if t_op[j] is not None and t_op[j] != o:
                continue
            Ac = acc[o]
            f2 = Ac["f2"][j]
            share = np.concatenate([[Ac["embf"][j]], UD[:, j] * Ac["tf"][:, j]]) / f2
            own = np.concatenate([[Ac["emb2"][j]], UD[:, j] ** 2 * Ac["t2"][:, j]]) / f2
            mean = np.concatenate([[Ac["emb1"][j]], UD[:, j] * Ac["t1"][:, j]]) / Ac["n"]
            G = Ac["Gg"][j]
            gshare = Ac["gf"][:, j] / f2
            r2g = 1 - (f2 - 2 * Ac["gf"][:, j] + np.diag(G)) / f2
            cm = Ac["cm"][:, :, j] / Ac["cn"][:, None]
            order = np.argsort(-np.abs(share))
            res["targets"][tn][OPS[o]] = {
                "mean_f": float(Ac["f1"][j] / Ac["n"]), "rms_f": float(np.sqrt(f2 / Ac["n"])),
                "sum_shares": float(share.sum()),
                "groups": {g: {"share": float(gshare[i]), "r2_alone": float(r2g[i])} for i, g in enumerate(gnames)},
                "top": [{"writer": wnames[w], "group": wgroups[w], "share": float(share[w]), "own": float(own[w]),
                         "mean": float(mean[w])} for w in order[:TOP]],
                "classmean_f": (Ac["cmf"][:, j] / Ac["cn"]).tolist(),
                "classmean_cum": np.cumsum(cm, 1).T.tolist(),  # (ng, 50): after each group in stream order
            }  # fmt: skip
            e = res["targets"][tn][OPS[o]]
            big = sorted(e["groups"].items(), key=lambda kv: -abs(kv[1]["share"]))[:10]
            print(
                tn,
                OPS[o],
                "mean",
                round(e["mean_f"], 3),
                "rms",
                round(e["rms_f"], 3),
                "sum",
                round(e["sum_shares"], 3),
            )
            print("   groups:", [(g, round(v["share"], 3)) for g, v in big])
            print(
                "   top:", [(t["writer"], round(t["share"], 3)) for t in e["top"][:12]], flush=True
            )
    (OUT / "pos3.json").write_text(json.dumps(res, indent=1))
    figures(res)


def figures(res: dict[str, Any]) -> None:
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    gn = res["groups"]
    cmap = plt.get_cmap("viridis")
    for o, sym in enumerate(OPS):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for col, c in enumerate(READERS):
            e = res["targets"][f"S[{c}]{sym}"][sym]
            cum = np.asarray(e["classmean_cum"])
            ax = axs[0, col]
            for i in range(len(gn)):
                blk = -1 if gn[i] == "emb" else int(gn[i].split(".")[0][1:])
                if gn[i].endswith("attn"):
                    continue
                ax.plot(range(50), cum[i], color=cmap((blk + 1) / 15), lw=1.6 if i == len(gn) - 1 else 0.8,
                        label=gn[i] if blk in (-1, 4, 9, 12, 13, 14) else None)  # fmt: skip
            ax.axhline(0, color="0.8", lw=0.5)
            ax.set_xlabel("b mod 50")
            ax.set_ylabel("class mean of the partial sum")
            ax.set_title(
                f"S for {c}, prompts a {sym} b: sum of the terms up to each block's MLP", fontsize=8
            )
            ax.legend(frameon=False, fontsize=6)
            ax = axs[1, col]
            v = [e["groups"][g]["share"] for g in gn]
            ax.bar(range(len(gn)), v, color=["#2a5db0" if x > 0 else "#c0392b" for x in v])
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xticks(range(len(gn)), gn, rotation=80, fontsize=5)
            ax.set_ylabel("share of S")
        fig.savefig(OUT / f"fig/pos3_{'add' if o == 0 else 'sub'}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
