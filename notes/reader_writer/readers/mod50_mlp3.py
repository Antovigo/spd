"""How the MLPs of blocks 0..14 at position 3 (the token b) build the copied signal S_c of the block-15 readers g19, g29
and u32: is each MLP's contribution a linear function of its input, and does it need a b mod 50 input?

Model and data as in mod50_pos3.py (alive-only model rebuilt from its stored activations; all 320,000 prompts
`<BOS> a op b =`, a, b integers 1..400, op in {+, -}). Nothing is centred: zero is the zero of the activation space.

Notation (position 3 throughout; x a prompt; l in 0..14 a block):
  mid_l(x)     the stream after block l's attention (the MLP's input before its RMSNorm)
  z_l(x)       = RMSNorm(mid_l) with block l's post-attention gain: what the MLP reads
  MLP_l(z)     the alive-only MLP: g = z . V_gate, u = z . V_up (its gate and up reader activations),
               t_down = (silu(g U_gate) * (u U_up)) . V_down, output t_down U_down
  sigma(x)     the RMS of pre_15 at position 3; d_c (mod50_pos3.py) the direction with S_c = d_c . pre_15 / sigma,
               one per reader c and op (the op's mean attention pattern)
  M_cl(x)      = d_c . MLP_l(z_l(x)) / sigma(x): block l's MLP term in S_c (sigma held at its value in the model)
  in_cl(x)     = d_c . mid_l(x) / sigma(x): the stream entering block l's MLP, along the same direction
  class mean   over the 3,200 prompts of one op with b mod 50 = r, r = 0..49; range = largest - smallest
Linearity: M_cl is regressed, without intercept, on the MLP's reader activations F = (g, u) (MLP_l depends on z_l only
  through them), on the prompts with even row index (fit half); r2_lin = 1 - sum (M - F w)^2 / sum M^2 on the odd half
  (uncentred). cm_lin: the class means of the fitted F w (all prompts).
Interventions: for a set Q of exact periods tau (divisors of 100), the part of z_l that is a function of b with those
  periods: P_Q(b) = the frequencies k with tau(k) = 100 / gcd(k, 100) in Q of the class means of z_l over b mod 100
  (per op). The MLP is rerun on z_l - P_Q(b); M'_cl its term; reported: the class means of M'_cl and the ratio of their
  range to that of M_cl. Q in {50}, {2}, {25}, {2, 25}, {2, 5, 10, 25, 50} (every period of a function of b mod 50).

Writes <run>/analysis/reader_writer/mod50_eq/{mlp3.json, fig/mlp3_*.png}.

    python mod50_mlp3.py prepare   # on the cluster, light: OUT/gains.npz (post-attention gains of blocks 0..14)
    python mod50_mlp3.py           # the analysis
"""

from __future__ import annotations

import json
import os
import sys
import time
from math import gcd
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
CHUNK = 2000
EPS = 1e-5
K = np.arange(100)
TAU = np.asarray([100 // gcd(int(k), 100) for k in K])
SETS = {"50": (50,), "2": (2,), "25": (25,), "2+25": (2, 25), "all|50": (2, 5, 10, 25, 50)}
KEY_BLOCKS = (3, 4, 11, 12, 13, 14)


def comps(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: (z[k].astype(np.float64) if k[0] in "UV" else z[k]) for k in z.files}


def prepare() -> None:
    import glob

    from mod10_L0 import HUB, bf16_tensor

    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    g = {}
    for blk in range(RB):
        name = f"model.layers.{blk}.post_attention_layernorm.weight"
        g[f"g2_{blk}"] = bf16_tensor(snap + wmap[name], name).astype(np.float64)
    np.savez(OUT / "gains.npz", **g)
    print("wrote", OUT / "gains.npz")


def silu(h: np.ndarray) -> np.ndarray:
    return h / (1 + np.exp(-h))


def directions(g1: np.ndarray) -> np.ndarray:
    """d_c per (op, reader): column o * 3 + c, as in mod50_pos3.py."""
    chn = json.loads((OUT / "chain.json").read_text())
    ch = np.load(OUT / "chain.npz")
    z = np.load(SRC_ATTN / f"L{RB}" / "comps.npz")
    Vv = z["V_v"].astype(np.float64)
    Vo = z["V_o"].astype(np.float64).reshape(H, DH, -1)
    Uv = z["U_v"].astype(np.float64).reshape(-1, KV, DH)
    B = np.einsum("hdo,khd->ohk", Vo, Uv[:, np.arange(H) // (H // KV)])
    A = np.einsum("oc,ohk->chk", ch["wo"], B)
    cols = []
    for sym in OPS:
        pbar = np.asarray(chn["heads"][sym]["pbar"])
        cols += [g1 * (Vv @ (pbar @ A[ci])) for ci in range(len(READERS))]
    return np.stack(cols, 1)  # (4096, 6)


def main() -> None:
    W = np.load(OUT / "inputs.npz")
    G2 = np.load(OUT / "gains.npz")
    g1, e_num = W["g1"], W["e_num"]
    D = directions(g1)
    nd = D.shape[1]
    mlp = {blk: comps(SRC / f"L{blk}" / "comps.npz") for blk in range(RB)}
    att = {blk: comps(SRC_ATTN / f"L{blk}" / "comps.npz") for blk in range(RB)}
    inner = {blk: np.load(SRC / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB)}
    inner_a = {blk: np.load(SRC_ATTN / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB)}
    g2 = [G2[f"g2_{blk}"] for blk in range(RB)]
    UD = [mlp[blk]["U_down"] @ D for blk in range(RB)]  # (n_down, 6)
    rows = np.arange(N)
    op_of, b_of = rows // 160000, rows % 400 + 1

    def stream(sl: slice) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
        b = b_of[sl]
        X = e_num[b - 1].copy()
        mids, tds = [], []
        for blk in range(RB):
            ta = np.asarray(inner_a[blk][sl, 2], np.float64)[:, att[blk]["kind"] == "o"]
            X += ta @ att[blk]["U_o"]
            mids.append(X.copy())
            td = np.asarray(inner[blk][sl, 2], np.float64)[:, mlp[blk]["kind"] == "down"]
            tds.append(td)
            X += td @ mlp[blk]["U_down"]
        return mids, np.sqrt((X**2).mean(-1) + EPS), tds

    def rms(x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return g * x / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)

    def run_mlp(blk: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = mlp[blk]
        tg, tu = z @ m["V_gate"], z @ m["V_up"]
        td = (silu(tg @ m["U_gate"]) * (tu @ m["U_up"])) @ m["V_down"]
        return td, tg, tu

    # pass 1: class means of z_l over b mod 100, per op
    t0 = time.time()
    cmz = np.zeros((RB, 2, 100, 4096))
    cnt = np.zeros((2, 100))
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, N))
        op, r100 = op_of[sl], b_of[sl] % 100
        mids, _, _ = stream(sl)
        for o in range(2):
            m = op == o
            oh = np.eye(100)[r100[m]]
            cnt[o] += oh.sum(0)
            for blk in range(RB):
                cmz[blk, o] += oh.T @ rms(mids[blk][m], g2[blk])
        if i0 % (10 * CHUNK) == 0:
            print("pass 1", i0, f"{time.time() - t0:.0f}s", flush=True)
    cmz /= cnt[None, :, :, None]
    proj = {}
    for name, taus in SETS.items():
        mask = np.isin(TAU, taus)
        Mf = np.fft.fft(cmz, axis=2)
        proj[name] = np.fft.ifft(Mf * mask[None, None, :, None], axis=2).real  # (RB, 2, 100, 4096)
    del Mf

    # pass 2
    nf = {
        blk: int((mlp[blk]["kind"] == "gate").sum() + (mlp[blk]["kind"] == "up").sum())
        for blk in range(RB)
    }
    acc = {(blk, o): {"FtF": [np.zeros((nf[blk], nf[blk])) for _ in range(2)],
                      "FtM": [np.zeros((nf[blk], nd)) for _ in range(2)], "M2": [np.zeros(nd) for _ in range(2)],
                      "cmF": np.zeros((50, nf[blk])), "cmM": np.zeros((50, nd)), "cmIn": np.zeros((50, nd)),
                      "cmI": {q: np.zeros((50, nd)) for q in SETS}, "n50": np.zeros(50)}
           for blk in range(RB) for o in range(2)}  # fmt: skip
    check = 0.0
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, N))
        op, b = op_of[sl], b_of[sl]
        half = rows[sl] % 2
        mids, sig, tds = stream(sl)
        for blk in range(RB):
            z = rms(mids[blk], g2[blk])
            td, tg, tu = run_mlp(blk, z)
            check = max(check, float(np.abs(td - tds[blk]).max() / np.abs(tds[blk]).max()))
            M = td @ UD[blk] / sig[:, None]
            In = mids[blk] @ D / sig[:, None]
            F = np.concatenate([tg, tu], 1)
            for o in range(2):
                m = op == o
                A = acc[(blk, o)]
                oh = np.eye(50)[b[m] % 50]
                A["cmF"] += oh.T @ F[m]
                A["cmM"] += oh.T @ M[m]
                A["cmIn"] += oh.T @ In[m]
                A["n50"] += oh.sum(0)
                for hv in range(2):
                    mm = m & (half == hv)
                    A["FtF"][hv] += F[mm].T @ F[mm]
                    A["FtM"][hv] += F[mm].T @ M[mm]
                    A["M2"][hv] += (M[mm] ** 2).sum(0)
                r100 = b[m] % 100
                for q in SETS:
                    tdq, _, _ = run_mlp(blk, z[m] - proj[q][blk, o][r100])
                    A["cmI"][q] += oh.T @ (tdq @ UD[blk] / sig[m, None])
        if i0 % (5 * CHUNK) == 0:
            print(
                "pass 2", i0, f"{time.time() - t0:.0f}s", "mlp check", round(check, 4), flush=True
            )

    res: dict[str, Any] = {"definitions": __doc__, "mlp_rel_max_vs_stored": check, "blocks": {}}
    for blk in range(RB):
        res["blocks"][blk] = {}
        for o, sym in enumerate(OPS):
            A = acc[(blk, o)]
            FtF, FtM = A["FtF"][0], A["FtM"][0]
            lam = 1e-9 * np.trace(FtF) / FtF.shape[0]
            w = np.linalg.solve(FtF + lam * np.eye(FtF.shape[0]), FtM)  # (nf, 6)
            G1, b1, y2 = A["FtF"][1], A["FtM"][1], A["M2"][1]
            n50 = A["n50"][:, None]
            for ci, c in enumerate(READERS):
                j = o * len(READERS) + ci
                wj = w[:, j]
                r2 = float(1 - (y2[j] - 2 * wj @ b1[:, j] + wj @ G1 @ wj) / y2[j])
                cmM = A["cmM"][:, j] / n50[:, 0]
                cmIn = A["cmIn"][:, j] / n50[:, 0]
                cmL = (A["cmF"] / n50) @ wj
                rng = float(cmM.max() - cmM.min())
                ent = {"r2_lin": r2, "range_M": rng, "range_in": float(cmIn.max() - cmIn.min()),
                       "range_lin": float(cmL.max() - cmL.min()), "cm_M": cmM.tolist(), "cm_in": cmIn.tolist(),
                       "cm_lin": cmL.tolist(), "interventions": {}}  # fmt: skip
                for q in SETS:
                    cq = A["cmI"][q][:, j] / n50[:, 0]
                    ent["interventions"][q] = {
                        "range_ratio": float((cq.max() - cq.min()) / rng),
                        "cm": cq.tolist(),
                    }
                res["blocks"][blk].setdefault(c, {})[sym] = ent
    (OUT / "mlp3.json").write_text(json.dumps(res, indent=1))
    for blk in range(RB):
        for c in READERS:
            e = res["blocks"][blk][c]["+"]
            print(blk, c, f"range_M {e['range_M']:.3f} range_in {e['range_in']:.3f} r2_lin {e['r2_lin']:.3f} "
                  f"range_lin {e['range_lin']:.3f}", {q: round(v["range_ratio"], 2) for q, v in e["interventions"].items()})  # fmt: skip
    figures(res)


def figures(res: dict[str, Any]) -> None:
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    colr = {"L15.gate.19": "#2a5db0", "L15.gate.29": "#d9730d", "L15.up.32": "#2e8b57"}
    for o, sym in enumerate(OPS):
        tag = "add" if o == 0 else "sub"
        fig, axs = plt.subplots(3, 5, figsize=(15, 9), layout="constrained")
        for blk, ax in zip(range(RB), axs.flat, strict=True):
            for c in READERS:
                e = res["blocks"][blk][c][sym]
                ax.plot(
                    e["cm_in"],
                    e["cm_M"],
                    "-o",
                    ms=1.8,
                    lw=0.8,
                    color=colr[c],
                    label=c.split(".", 1)[1],
                )
                ax.plot(e["cm_in"], e["cm_lin"], "--", lw=0.8, color=colr[c])
            ax.set_title(f"block {blk} MLP (r² lin: " + ", ".join(f"{res['blocks'][blk][c][sym]['r2_lin']:.2f}"
                         for c in READERS) + ")", fontsize=7)  # fmt: skip
            ax.set_xlabel("class mean of in (stream entering the MLP, along d_c)")
            ax.set_ylabel("class mean of M (MLP term)")
            ax.axhline(0, color="0.85", lw=0.5)
            ax.axvline(0, color="0.85", lw=0.5)
        axs.flat[0].legend(frameon=False, fontsize=6)
        fig.suptitle(f"Prompts a {sym} b: each MLP's term in S_c against its input along d_c, one point per b mod 50 "
                     "class (solid: the MLP; dashed: the linear fit)")  # fmt: skip
        fig.savefig(OUT / f"fig/mlp3_sharpen_{tag}.png")
        plt.close(fig)

        fig, axs = plt.subplots(3, len(KEY_BLOCKS), figsize=(16, 8), layout="constrained")
        qcol = {
            "50": "#c0392b",
            "2": "#8e44ad",
            "25": "#d9730d",
            "2+25": "#7f8c8d",
            "all|50": "#16a085",
        }
        for ri, c in enumerate(READERS):
            for ki, blk in enumerate(KEY_BLOCKS):
                ax = axs[ri, ki]
                e = res["blocks"][blk][c][sym]
                ax.plot(range(50), e["cm_M"], color="k", lw=1.4, label="MLP term")
                for q, v in e["interventions"].items():
                    ax.plot(
                        range(50),
                        v["cm"],
                        color=qcol[q],
                        lw=0.9,
                        label=f"input without periods {q}",
                    )
                ax.axhline(0, color="0.85", lw=0.5)
                ax.set_title(f"{c}, block {blk}", fontsize=7)
                ax.set_xlabel("b mod 50")
        axs[0, 0].legend(frameon=False, fontsize=5.5)
        fig.suptitle(
            f"Prompts a {sym} b: class means of the MLP term in S_c, and with parts of the input removed"
        )
        fig.savefig(OUT / f"fig/mlp3_interv_{tag}.png")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare()
    else:
        main()
