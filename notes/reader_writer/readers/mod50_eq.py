"""Where the inputs of the block-15 readers g19, g29 (gate) and u32 (up) come from at the token "=" (position 4).

Model: the alive-only model of variance/alive_only (every alive component of the ceiling filter addsub-08-last-pos on,
every other component and the weight delta off). Its stored activations (float32, every alive component, positions
1..4, all 320,000 prompts `<BOS> a op b =`, a, b integers 1..400, op in {+, -}) rebuild the residual stream:
MLP sites from variance/alive_only/L<l>/inner.npy, attention sites from variance/alive_only_attn/L<l>/inner.npy; the
BOS position (0), the same on every prompt, from track_mod10/stream.npz (bos_pre_15). Rows: op * 160000 + (a - 1) * 400
+ (b - 1). Nothing is centred: zero is the zero of the activation space.

Notation (position 4 unless stated; x a prompt):
  c            a reader: block-15 gate component 19 or 29, or up component 32; V_c in R^4096 its read vector
  gamma        (4096,) the gain of block 15's post-attention RMSNorm
  r_c          = gamma * V_c / ||V_c||: the read direction with the gain folded in
  X(x)         mid_15(x) in R^4096: the stream after block 15's attention, read by the MLP;
               rho(x) = sqrt(mean_j X_j(x)^2 + 1e-5), j in 1..4096 the coordinates
  f_c(x)       = r_c . X(x) / rho(x): the reader's inner activation divided by ||V_c||
  writers w    the embedding of "=" (one term, "emb", the same on every prompt); every alive attention o component and
               every alive MLP down component of blocks 0..14; and, for block 15's attention, every alive o component
               split by the source position s in 0..4 it reads from (0 BOS, 1 a, 2 op, 3 b, 4 "="), id
               "L15.o.<i>@s<s>": t_w(x) is the o component's activation computed from the heads' outputs restricted to
               source s (pattern weight times value), so the five add up to the o component's activation.
               X = emb + sum_w t_w U_w exactly (U_w the write vector)
  e_w(x)       = t_w(x) (r_c . U_w) / rho(x): writer w's term; sum_w e_w = f_c exactly (rho is the reader's own
               per-prompt RMS, shared by every term)
  Sums over x run over the prompts of one op (160,000 each), uncentred:
  share_w      = sum_x e_w f_c / sum_x f_c^2: writer w's signed share of the reader's activation; sums to 1 over the
               writers. A negative share opposes the activation
  own_w        = sum_x e_w^2 / sum_x f_c^2: the size of the term next to the activation
  mean_w       = mean_x e_w (mean_f the same for f_c): the constant part of the term
  r2(S)        = 1 - sum_x (f_c - sum_{w in S} e_w)^2 / sum_x f_c^2 for a set S of writers (uncentred; computed from the
               Gram matrix of the terms), for groups (a sublayer, a source position) and for the k writers of largest
               |share|, k = 1..60
Checks (stored in the output): the rebuilt block-15 o activations (sum over s) and f_c against the stored ones.

Plots (raw stream, no division by rho): the coordinates x . r_19 and x . r_29 of the stream at position 4, coloured
by b mod 50, on 20,000 random prompts per op: mid_l (before block l's MLP) for l = 0..15 and pre_15 (before block 15's
attention); and block 15's writes (attention, by source position, and MLP: pre_16 - mid_15).

Writes <run>/analysis/reader_writer/mod50_eq/{attribution.json, proj.npz, chain.npz, fig/*.png}. chain.npz holds, per
prompt, what mod50_chain.py needs to rewrite block 15's attention term: p (N, 32, 5) the attention pattern at "=" (query
head, source position), tau3 (N, n_v) the activations of block 15's alive v components at position 3 (computed from
pre_15 at position 3, checked against the stored ones), o3 (N, n_o) the o activations from source 3, rho (N,), f (N, 3),
and R (4096, 3), wo (n_o, 3) = U_o . r_c.

    python mod50_eq.py prepare            # on the cluster: OUT/inputs.npz (embeddings, gains, config, BOS stream)
    python mod50_eq.py [compute|plot|all]  # anywhere with numpy + matplotlib; RUN_DIR env overrides the run path
"""

from __future__ import annotations

import json
import os
import sys
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
BOS_STREAM = RUN / "analysis/reader_writer/track_mod10/stream.npz"
OUT = RUN / "analysis/reader_writer/mod50_eq"
RB = 15
READERS = (("gate", 19), ("gate", 29), ("up", 32))
NR = len(READERS)
OPS = ("+", "-")
N = 320000
CHUNK = 4000
EPS = 1e-5
TOPK = 60
SRCNAME = ("BOS", "a", "op", "b", "=")
INPUTS = OUT / "inputs.npz"


def rope(x: np.ndarray, pos: int, cfg: dict) -> np.ndarray:
    """HF Llama rotary embedding (rotate_half layout) with llama3 frequency scaling; x (..., 128).
    Copy of mlp0_mod10.rope, so that this script runs with numpy and matplotlib alone."""
    d = x.shape[-1]
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, d, 2) / d)
    rs = cfg["rope_scaling"]
    lo_w = rs["original_max_position_embeddings"] / rs["low_freq_factor"]
    hi_w = rs["original_max_position_embeddings"] / rs["high_freq_factor"]
    wl = 2 * np.pi / inv
    smooth = (rs["original_max_position_embeddings"] / wl - rs["low_freq_factor"]) / (
        rs["high_freq_factor"] - rs["low_freq_factor"]
    )
    scaled = np.where(wl > lo_w, inv / rs["factor"], inv)
    mid = (wl <= lo_w) & (wl >= hi_w)
    scaled = np.where(mid, (1 - smooth) * inv / rs["factor"] + smooth * inv, scaled)
    ang = pos * np.concatenate([scaled, scaled])
    half = np.concatenate([-x[..., d // 2 :], x[..., : d // 2]], -1)
    return x * np.cos(ang) + half * np.sin(ang)


def rmsnorm(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return g * x / np.sqrt((x**2).mean(-1, keepdims=True) + EPS)


def prepare() -> None:
    """On the cluster (light: a few rows of the embedding): the token embeddings, block 15's norm gains, the model
    config and the BOS stream entering block 15, so that compute() needs no weights or tokenizer."""
    import glob

    from mlp0_mod10 import BOS
    from mod10_L0 import HUB, bf16_tensor
    from transformers import AutoTokenizer

    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]

    def wt(name: str, rows: np.ndarray | None = None) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name, rows).astype(np.float64)

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    num = [tok.encode(str(v), add_special_tokens=False) for v in range(1, 401)]
    assert all(len(t) == 1 for t in num)
    num_ids = np.asarray([t[0] for t in num])
    op_ids = []
    for sym in OPS:
        for a, b in ((1, 1), (37, 400), (400, 205)):
            t = tok.encode(f"{a}{sym}{b}=", add_special_tokens=True)
            assert t[:2] == [BOS, num_ids[a - 1]] and t[3] == num_ids[b - 1] and len(t) == 5, t
        op_ids.append(t[2])
        eq_id = t[4]
    emb = "model.embed_tokens.weight"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(INPUTS, e_num=wt(emb, num_ids), e_op=wt(emb, np.asarray(op_ids)),
             e_eq=wt(emb, np.asarray([eq_id]))[0], g1=wt(f"model.layers.{RB}.input_layernorm.weight"),
             g2=wt(f"model.layers.{RB}.post_attention_layernorm.weight"),
             bos=np.load(BOS_STREAM)[f"bos_pre_{RB}"].astype(np.float64), cfg=json.dumps(cfg),
             token_ids=np.asarray([BOS, *op_ids, eq_id]))  # fmt: skip
    print("wrote", INPUTS, "op ids", op_ids, "= id", eq_id)


def comps(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: (z[k].astype(np.float64) if k[0] in "UV" else z[k]) for k in z.files}


def compute() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    W = np.load(INPUTS)
    cfg = json.loads(str(W["cfg"]))
    mlp = {blk: comps(SRC / f"L{blk}" / "comps.npz") for blk in range(RB + 1)}
    att = {blk: comps(SRC_ATTN / f"L{blk}" / "comps.npz") for blk in range(RB + 1)}
    inner = {blk: np.load(SRC / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB + 1)}
    inner_a = {
        blk: np.load(SRC_ATTN / f"L{blk}" / "inner.npy", mmap_mode="r") for blk in range(RB + 1)
    }
    g1, g2, e_num, e_op, e_eq = W["g1"], W["g2"], W["e_num"], W["e_op"], W["e_eq"]
    m15, a15 = mlp[RB], att[RB]
    gsel = [int(np.flatnonzero((m15["kind"] == kd) & (m15["index"] == c))[0]) for kd, c in READERS]
    Vc = np.stack([m15[f"V_{kd}"][:, int(np.flatnonzero(m15["index"][m15["kind"] == kd] == c)[0])]
                   for kd, c in READERS], 1)  # fmt: skip
    vn = np.linalg.norm(Vc, axis=0)
    R = g2[:, None] * Vc / vn  # (4096, NR)
    bos = W["bos"]

    # writer list
    names, groups = ["emb"], ["emb"]
    for blk in range(RB):
        for kind, site in (("o", att[blk]), ("down", mlp[blk])):
            ids = site["index"][site["kind"] == kind]
            names += [f"L{blk}.{kind}.{i}" for i in ids]
            groups += [f"L{blk}.{'attn' if kind == 'o' else 'mlp'}"] * len(ids)
    o15 = a15["index"][a15["kind"] == "o"]
    for s in range(5):
        names += [f"L{RB}.o.{i}@s{s}" for i in o15]
        groups += [f"L{RB}.attn.{SRCNAME[s]}"] * len(o15)
    nw = len(names)
    print("writers", nw, flush=True)

    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], 128
    rep = np.arange(H) // (H // KV)
    acc = {(o, j): {"ef": np.zeros(nw), "G": np.zeros((nw, nw)), "e": np.zeros(nw), "f2": 0.0,
                    "f": 0.0, "n": 0} for o in range(2) for j in range(NR)}  # fmt: skip
    proj = {"pre": np.zeros((N, RB + 2, NR), np.float32), "mid": np.zeros((N, RB + 1, NR), np.float32),
            "src15": np.zeros((N, 5, NR), np.float32), "f": np.zeros((N, NR), np.float32)}  # fmt: skip
    vsel = a15["kind"] == "v"
    chain = {"p": np.zeros((N, H, 5), np.float32), "tau3": np.zeros((N, int(vsel.sum())), np.float32),
             "o3": np.zeros((N, int((a15["kind"] == "o").sum())), np.float32), "rho": np.zeros(N)}  # fmt: skip
    checks = {"o15_rel_max": 0.0, "tau3_rel_max": 0.0, "f_rel_max": 0.0, "f_sq_err": np.zeros(NR),
              "f_sq": np.zeros(NR)}  # fmt: skip
    rows = np.arange(N)
    op_of, a_of, b_of = rows // 160000, (rows % 160000) // 400 + 1, rows % 400 + 1
    t0 = time.time()
    for i0 in range(0, N, CHUNK):
        sl = slice(i0, min(i0 + CHUNK, N))
        n = sl.stop - sl.start
        op, a, b = op_of[sl], a_of[sl], b_of[sl]
        # stream at positions 1..4 entering block 0, then the writes of blocks 0..14
        X = np.stack([e_num[a - 1], e_op[op], e_num[b - 1], np.broadcast_to(e_eq, (n, 4096))], 1)
        terms = [np.broadcast_to(e_eq @ R, (n, NR))[:, None]]
        cum = np.broadcast_to(e_eq @ R, (n, NR)).copy()
        for blk in range(RB):
            proj["pre"][sl, blk] = cum
            ta = np.asarray(inner_a[blk][sl], np.float64)[
                :, :, att[blk]["kind"] == "o"
            ]  # (n, 4, n_o)
            X += ta @ att[blk]["U_o"]
            e = ta[:, 3, :, None] * (att[blk]["U_o"] @ R)[None]
            terms.append(e)
            cum += e.sum(1)
            proj["mid"][sl, blk] = cum
            td = np.asarray(inner[blk][sl], np.float64)[:, :, mlp[blk]["kind"] == "down"]
            X += td @ mlp[blk]["U_down"]
            e = td[:, 3, :, None] * (mlp[blk]["U_down"] @ R)[None]
            terms.append(e)
            cum += e.sum(1)
        proj["pre"][sl, RB] = cum
        # block 15 attention at position 4, split by source position
        P = np.concatenate([np.broadcast_to(bos, (n, 1, 4096)), X], 1)  # (n, 5, 4096): pre_15
        xn = rmsnorm(P, g1)
        pr = lambda x, k: (x @ a15[f"V_{k}"]) @ a15[f"U_{k}"]  # noqa: E731
        q = rope(pr(xn[:, 4], "q").reshape(n, H, dh), 4, cfg)
        k = np.stack([rope(pr(xn[:, s], "k").reshape(n, KV, dh), s, cfg) for s in range(5)], 1)[
            :, :, rep
        ]
        v = np.stack([pr(xn[:, s], "v").reshape(n, KV, dh) for s in range(5)], 1)[:, :, rep]
        sc = np.einsum("nhd,nshd->nhs", q, k) / np.sqrt(dh)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o_src = np.stack(
            [(p[:, :, s, None] * v[:, s]).reshape(n, H * dh) @ a15["V_o"] for s in range(5)], 1
        )
        ref = np.asarray(inner_a[RB][sl, 3], np.float64)[:, a15["kind"] == "o"]
        checks["o15_rel_max"] = max(checks["o15_rel_max"],
                                    float(np.abs(o_src.sum(1) - ref).max() / np.abs(ref).max()))  # fmt: skip
        tau3 = xn[:, 3] @ a15["V_v"]
        tref = np.asarray(inner_a[RB][sl, 2], np.float64)[:, vsel]
        checks["tau3_rel_max"] = max(
            checks["tau3_rel_max"], float(np.abs(tau3 - tref).max() / np.abs(tref).max())
        )
        chain["p"][sl], chain["tau3"][sl], chain["o3"][sl] = p, tau3, o_src[:, 3]
        mid = X[:, 3] + o_src.sum(1) @ a15["U_o"]
        for s in range(5):
            e = o_src[:, s, :, None] * (a15["U_o"] @ R)[None]
            terms.append(e)
            proj["src15"][sl, s] = e.sum(1)
            cum += e.sum(1)
        proj["mid"][sl, RB] = cum
        assert np.allclose(cum, mid @ R, rtol=1e-6, atol=1e-6 * np.abs(cum).max())
        td = np.asarray(inner[RB][sl, 3], np.float64)[:, m15["kind"] == "down"]
        proj["pre"][sl, RB + 1] = cum + td @ (m15["U_down"] @ R)
        rho = np.sqrt((mid**2).mean(-1) + EPS)
        chain["rho"][sl] = rho
        f = (mid @ R) / rho[:, None]
        fref = np.asarray(inner[RB][sl, 3], np.float64)[:, gsel] / vn
        checks["f_rel_max"] = max(
            checks["f_rel_max"], float(np.abs(f - fref).max() / np.abs(fref).max())
        )
        checks["f_sq_err"] += ((f - fref) ** 2).sum(0)
        checks["f_sq"] += (fref**2).sum(0)
        proj["f"][sl] = f
        E = np.concatenate(terms, 1) / rho[:, None, None]  # (n, n_w, 2)
        assert E.shape[1] == nw
        assert np.abs(E.sum(1) - f).max() <= 1e-8 * np.abs(f).max() + 1e-12
        for o in range(2):
            m = op == o
            if not m.any():
                continue
            for j in range(NR):
                Ej, fj = E[m, :, j], f[m, j]
                A = acc[(o, j)]
                A["ef"] += Ej.T @ fj
                A["G"] += Ej.T @ Ej
                A["e"] += Ej.sum(0)
                A["f2"] += float(fj @ fj)
                A["f"] += float(fj.sum())
                A["n"] += int(m.sum())
        if i0 % (10 * CHUNK) == 0:
            print(i0, f"{time.time() - t0:.0f}s", {k: (v if np.isscalar(v) else v.tolist()) for k, v in checks.items()},
                  flush=True)  # fmt: skip

    res: dict[str, Any] = {"definitions": __doc__, "n_writers": nw,
                           "checks": {"o15_rel_max": checks["o15_rel_max"], "tau3_rel_max": checks["tau3_rel_max"],
                                      "f_rel_max": checks["f_rel_max"],
                                      "f_r2_vs_stored": (1 - checks["f_sq_err"] / checks["f_sq"]).tolist()},
                           "readers": {}}  # fmt: skip
    gnames = sorted(set(groups), key=lambda g: (g != "emb", groups.index(g)))
    garr = np.asarray(groups)
    for j, (kd, c) in enumerate(READERS):
        cid = f"L{RB}.{kd}.{c}"
        res["readers"][cid] = {}
        for o in range(2):
            A = acc[(o, j)]
            f2, G = A["f2"], A["G"]
            share, own, mean = A["ef"] / f2, np.diag(G) / f2, A["e"] / A["n"]

            def r2(sel: np.ndarray) -> float:
                s = np.zeros(nw)
                s[sel] = 1
                return float(1 - (f2 - 2 * s @ A["ef"] + s @ G @ s) / f2)  # noqa: B023

            order = np.argsort(-np.abs(share))
            grp = {g: {"share": float(share[garr == g].sum()), "r2_alone": r2(np.flatnonzero(garr == g)),
                       "mean": float(mean[garr == g].sum())} for g in gnames}  # fmt: skip
            res["readers"][cid][OPS[o]] = {
                "n": A["n"],
                "mean_f": A["f"] / A["n"],
                "rms_f": float(np.sqrt(f2 / A["n"])),
                "sum_shares": float(share.sum()),
                "groups": grp,
                "top": [
                    {
                        "writer": names[w],
                        "group": groups[w],
                        "share": float(share[w]),
                        "own": float(own[w]),
                        "mean": float(mean[w]),
                    }
                    for w in order[:TOPK]
                ],  # fmt: skip
                "r2_topk": [r2(order[:k]) for k in range(1, TOPK + 1)],
            }
            top = res["readers"][cid][OPS[o]]
            print(cid, OPS[o], "mean_f", round(top["mean_f"], 4), "rms_f", round(top["rms_f"], 4),
                  "sum", round(top["sum_shares"], 4))  # fmt: skip
            big = sorted(grp.items(), key=lambda kv: -abs(kv[1]["share"]))[:12]
            print("  groups:", [(g, round(v["share"], 3), round(v["r2_alone"], 3)) for g, v in big])
            print("  top:", [(t["writer"], round(t["share"], 3)) for t in top["top"][:15]])
            print(
                "  r2 top k=1,5,10,20,60:",
                [round(top["r2_topk"][k - 1], 3) for k in (1, 5, 10, 20, 60)],
            )
    print(json.dumps(res["checks"]))
    (OUT / "attribution.json").write_text(json.dumps(res, indent=1))
    np.savez(
        OUT / "proj.npz",
        op=op_of.astype(np.int8),
        a=a_of.astype(np.int16),
        b=b_of.astype(np.int16),
        **proj,
    )
    np.savez(OUT / "chain.npz", f=proj["f"], R=R, wo=a15["U_o"] @ R, **chain)


def scatter(ax, P: np.ndarray, col: np.ndarray, title: str, lim=None):  # noqa: ANN001, ANN201
    sc = ax.scatter(P[:, 0], P[:, 1], c=col, cmap="viridis", vmin=0, vmax=49, s=1.5, lw=0, alpha=0.6,
                    rasterized=True)  # fmt: skip
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.axvline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x · r_19", fontsize=7)
    ax.set_ylabel("x · r_29", fontsize=7)
    if lim is not None:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
    return sc


def plot() -> None:
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    (OUT / "fig").mkdir(parents=True, exist_ok=True)
    z = np.load(OUT / "proj.npz")
    rng = np.random.default_rng(0)
    for o, sym in enumerate(OPS):
        idx = rng.permutation(np.flatnonzero(z["op"] == o))[:20000]
        col = z["b"][idx] % 50
        mid, pre, src = z["mid"][idx], z["pre"][idx], z["src15"][idx]
        panels = [(mid[:, blk], f"block {blk}: before the MLP (mid_{blk})") for blk in range(RB)]
        panels += [(pre[:, RB], f"block {RB}: before the attention (pre_{RB})"),
                   (mid[:, RB], f"block {RB}: before the MLP (mid_{RB})")]  # fmt: skip
        fig, axs = plt.subplots(4, 5, figsize=(15, 11.5), layout="constrained")
        for ax, (P, t) in zip(axs.flat, panels, strict=False):
            sc = scatter(ax, P, col, t)
        for ax in axs.flat[len(panels) :]:
            ax.axis("off")
        fig.colorbar(sc, ax=axs, fraction=0.015).set_label("b mod 50")
        fig.suptitle(f'Stream at "=" on the read directions of L15 g19 / g29, prompts a {sym} b '
                     "(20,000 random prompts, raw: not divided by the RMS)")  # fmt: skip
        fig.savefig(OUT / f"fig/stream_{'add' if o == 0 else 'sub'}.png")
        plt.close(fig)

        writes = [(mid[:, RB] - pre[:, RB], "block 15 attention write (mid_15 − pre_15)")]
        writes += [
            (src[:, s], f"block 15 attention write from source {SRCNAME[s]}") for s in range(5)
        ]
        writes += [(pre[:, RB + 1] - mid[:, RB], "block 15 MLP write (pre_16 − mid_15)")]
        allp = np.concatenate([w for w, _ in writes] + [mid[:, RB]])
        pad = 0.05 * (allp.max(0) - allp.min(0))
        lim = [(allp[:, 0].min() - pad[0], allp[:, 0].max() + pad[0]),
               (allp[:, 1].min() - pad[1], allp[:, 1].max() + pad[1])]  # fmt: skip
        fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
        scatter(axs.flat[0], mid[:, RB], col, "for reference: mid_15", lim)
        for ax, (P, t) in zip(axs.flat[1:], writes, strict=True):
            sc = scatter(ax, P, col, t, lim)
        fig.colorbar(sc, ax=axs, fraction=0.015).set_label("b mod 50")
        fig.suptitle(f'What block 15 writes at "=" along the read directions of g19 / g29, prompts a {sym} b '
                     "(same axes and limits in every panel)")  # fmt: skip
        fig.savefig(OUT / f"fig/writes15_{'add' if o == 0 else 'sub'}.png")
        plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "prepare":
        prepare()
    if what in ("compute", "all"):
        compute()
    if what in ("plot", "all"):
        plot()
