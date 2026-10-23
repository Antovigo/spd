"""Section 11 of the notebook page "Mod 10 representations": attention patterns of the copying heads (original and
alive-only models), the copying components' vectors in the Fourier planes, and the geometry of the copies at "=".

Prompts: 2,000 of each of copy_build.py's two sets (every 4th of set "a", a seeded random draw of set "b"): set "a" (all a, ten b, both ops) for block 16,
which copies a mod 10 from position 1; set "b" (ten a, all b, both ops) for block 15, which copies b mod 10 from
position 3. The alive-only streams come from copy_build.py; the original model's are computed here, in float32
from the bf16 weights, through block 16. Nothing is centred except where a spread around a class mean is the
quantity measured (Cohen's d). Notation (residue r = the copied operand mod 10):
  p_h(j)       attention weight of head h (0..31) at "=" (position 4) on position j (0..4)
  x_s          the attention input (input RMSNorm) at the source position s; P_k^src its Fourier planes (k = 1..5)
  E, E'        the stream at "=" entering the block and after the attention write; P_k^eq the Fourier planes of E'
  V_c, U_o     read vector of value component c (in x_s space), write vector of output component o (residual space)
  cos(r, r')   cosine between residue means; d(r, r') = |c_r - c_r'| / pooled s.d. of the two clouds projected on
               the unit vector from c_r' to c_r (c_r the residue mean)
Writes <run>/analysis/reader_writer/copy_mod10/figs.json and notebook/fig/copy_{patterns,vectors,geometry}.png.

    python copy_figs.py --out <notebook dir>
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy_analyze import attention_parts, means, psi
from copy_build import OUT
from mlp0_mod10 import rope
from mod10_L0 import HUB, bf16_tensor
from track_build import load_blocks, rmsnorm

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
CM = plt.get_cmap("tab10")
COPY = {"a": (16, 1), "b": (15, 3)}
POS = ["<BOS>", "a", "op", "b", "="]
# 2,000 prompts per set: every 4th of set "a" (all ten b residues), a seeded random draw of set "b" (every 4th of set "b"
# would keep only odd b, since it is ordered by b within each (op, a))
SUB = {
    "a": slice(0, None, 4),
    "b": np.sort(np.random.default_rng(0).choice(8000, 2000, replace=False)),
}


def original_stream(
    tokens_X: np.ndarray, snap: str, wmap: dict, cfg: dict, upto: int
) -> dict[int, np.ndarray]:
    """Original Llama forward on embedded prompts (n, 5, 4096); returns the stream entering blocks 15 and 16."""

    def w(name: str) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name).astype(np.float32)

    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], 128
    rep = np.arange(H) // (H // KV)
    X = tokens_X.astype(np.float32)
    n, T, d = X.shape
    keep = {}
    for b in range(upto + 1):
        if b in (15, 16):
            keep[b] = X.copy()
        pre = f"model.layers.{b}."
        x = rmsnorm(X, w(pre + "input_layernorm.weight")).astype(np.float32)
        q = x @ w(pre + "self_attn.q_proj.weight").T
        k = x @ w(pre + "self_attn.k_proj.weight").T
        v = x @ w(pre + "self_attn.v_proj.weight").T
        q = np.stack([rope(q[:, t].reshape(n, H, dh), t, cfg) for t in range(T)], 1)
        k = np.stack([rope(k[:, t].reshape(n, KV, dh), t, cfg) for t in range(T)], 1)[:, :, rep]
        v = v.reshape(n, T, KV, dh)[:, :, rep]
        s = np.einsum("nthd,nshd->nhts", q, k) / np.sqrt(dh)
        s = np.where(np.tril(np.ones((T, T), bool))[None, None], s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o = (
            np.einsum("nhts,nshd->nthd", p, v).reshape(n, T, H * dh)
            @ w(pre + "self_attn.o_proj.weight").T
        )
        X = X + o
        x = rmsnorm(X, w(pre + "post_attention_layernorm.weight")).astype(np.float32)
        g = x @ w(pre + "mlp.gate_proj.weight").T
        u = x @ w(pre + "mlp.up_proj.weight").T
        X = X + ((g / (1 + np.exp(-g))) * u) @ w(pre + "mlp.down_proj.weight").T
        print("original block", b, flush=True)
    return keep


def orig_attention(
    snap: str, wmap: dict, cfg: dict, L: int, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pattern at "=" (n, 32, 5) and the attention output at "=" (n, 4096) of the original block L."""

    def w(name: str) -> np.ndarray:
        return bf16_tensor(snap + wmap[name], name).astype(np.float64)

    H, KV, dh = cfg["num_attention_heads"], cfg["num_key_value_heads"], 128
    rep = np.arange(H) // (H // KV)
    pre = f"model.layers.{L}."
    n, T, _ = X.shape
    x = rmsnorm(X, w(pre + "input_layernorm.weight"))
    q = rope((x[:, 4] @ w(pre + "self_attn.q_proj.weight").T).reshape(n, H, dh), 4, cfg)
    k = np.stack(
        [
            rope((x[:, t] @ w(pre + "self_attn.k_proj.weight").T).reshape(n, KV, dh), t, cfg)
            for t in range(T)
        ],
        1,
    )[:, :, rep]
    v = (x @ w(pre + "self_attn.v_proj.weight").T).reshape(n, T, KV, dh)[:, :, rep]
    s = np.einsum("nhd,nthd->nht", q, k) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = np.einsum("nht,nthd->nhd", p, v).reshape(n, H * dh) @ w(pre + "self_attn.o_proj.weight").T
    return p, out


def planes(Z: np.ndarray, lab: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Fourier planes P_k (k = 1..5) of Z's residue means and their joint orthonormal basis D; built from the 2 x 2 Gram
    matrix of (Re Psi_k, Im Psi_k) instead of an SVD of the 4096 x 2 matrix."""
    P = []
    for p in psi(Z, lab):
        M = np.stack([p.real, p.imag], 1)
        lam_, vec = np.linalg.eigh(M.T @ M)
        keep = lam_ > 1e-12 * lam_.max()
        P.append((M @ vec[:, keep]) / np.sqrt(lam_[keep]))
    Q, _ = np.linalg.qr(np.concatenate(P, 1))
    return P, Q


def cohen(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    c = means(Z, lab)
    d = np.zeros((10, 10))
    for r in range(10):
        for q in range(10):
            if r != q:
                u = (c[r] - c[q]) / np.linalg.norm(c[r] - c[q])
                a, b = Z[lab == r] @ u, Z[lab == q] @ u
                d[r, q] = abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    return d


def cosmat(M: np.ndarray) -> np.ndarray:
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    return Mn @ Mn.T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks, _, _ = load_blocks()
    alive = json.loads(
        (
            Path(str(OUT).replace("analysis/reader_writer/copy_mod10", "addsub-08-last-pos-40000"))
            / "alive/alive.json"
        ).read_text()
    )["components"]
    cres = json.loads((OUT / "copy.json").read_text())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    res: dict = {}
    data: dict = {}
    for w_, (L, _s) in COPY.items():
        st = np.load(OUT / f"stream_{w_}.npz")
        sub = SUB[w_]
        a, b, op = st["a"][sub], st["b"][sub], st["op"][sub]
        lab = (st[w_][sub] % 10).astype(int)
        print(w_, "prompts per residue", np.bincount(lab, minlength=10).tolist(), flush=True)
        Xal = st[f"X{L}"][sub].astype(np.float64)
        toks = np.asarray(
            [
                tok.encode(f"{x}{'+-'[o]}{y}=", add_special_tokens=True)
                for x, y, o in zip(a, b, op, strict=True)
            ]
        )
        uniq, inv = np.unique(toks.ravel(), return_inverse=True)
        E = bf16_tensor(
            snap + wmap["model.embed_tokens.weight"], "model.embed_tokens.weight", uniq
        )[inv].reshape(*toks.shape, -1)
        cache = OUT / f"orig_cache_{w_}.npz"
        if cache.exists():
            Xor = np.load(cache)["Xor"].astype(np.float64)
        else:
            Xor = original_stream(E, snap, wmap, cfg, L)[L].astype(np.float64)
            np.savez(cache, Xor=Xor.astype(np.float32))
        blk = blocks[L]
        parts = attention_parts(blk, Xal)
        p_al = parts["p"]
        # alive-only attention output at "=": full write through all heads
        from track_build_p3 import attn_seq

        out_al = attn_seq(blk, Xal)[:, 4]
        p_or, out_or = orig_attention(snap, wmap, cfg, L, Xor)
        data[w_] = {"lab": lab, "p_al": p_al, "p_or": p_or, "Xal": Xal, "Xor": Xor, "out_al": out_al, "out_or": out_or,
                    "parts": parts, "blk": blk}  # fmt: skip
        res[w_] = {
            "pattern_alive": p_al.mean(0).tolist(),
            "pattern_original": p_or.mean(0).tolist(),
        }
        print(w_, "patterns done", flush=True)

    # ---- figure 1: attention patterns at "=" for the active heads
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), gridspec_kw={"wspace": 0.35})
    for i, (w_, (L, s)) in enumerate(COPY.items()):
        d = data[w_]
        mo, ma = d["p_or"].mean(0), d["p_al"].mean(0)
        heads = sorted(
            {h for h in range(32) if max(mo[h, s], ma[h, s]) >= 0.15} | {21 if w_ == "a" else 13}
        )
        res[w_]["active_heads"] = heads
        for j, (M, name) in enumerate(((mo, "original"), (ma, "alive-only"))):
            ax = axes[2 * i + j]
            ax.imshow(M[heads], cmap="Greys", vmin=0, vmax=1, aspect="auto")
            for y, h in enumerate(heads):
                for x in range(5):
                    ax.text(x, y, f"{M[h, x]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[h, x] > 0.5 else "black")  # fmt: skip
            ax.set_yticks(range(len(heads)))
            ax.set_yticklabels([f"head {h} (KV {h // 4})" for h in heads], fontsize=7)
            ax.set_xticks(range(5))
            ax.set_xticklabels(POS, fontsize=7)
            ax.axvline(s, color="#c0392b", lw=1)
            ax.set_title(f'block {L}, query "=", {name}\n(source: position {s})', fontsize=9)
    fig.savefig(args.out / "fig" / "copy_patterns.png", bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: v read vectors in the source planes; o write vectors in the "=" planes after the write
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    for i, (w_, (L, s)) in enumerate(COPY.items()):
        d = data[w_]
        blk, lab = d["blk"], d["lab"]
        xs = d["parts"]["x"][:, s]
        Ps, _ = planes(xs, lab)
        E_before = d["Xal"][:, 4]
        E_after = E_before + d["out_al"]
        Pe, De = planes(E_after, lab)
        top_v = [x["id"] for x in cres[w_]["vcomps"]["top90"]]
        top_o = [x["id"] for x in cres[w_]["ocomps"]["top90"]]
        ids_v = list(np.asarray(alive[f"layers.{L}.self_attn.v_proj"]))
        ids_o = list(np.asarray(alive[f"layers.{L}.self_attn.o_proj"]))
        Vv = blk.a["V_v"][:, [ids_v.index(c) for c in top_v]]
        Uo = blk.a["U_o"][[ids_o.index(c) for c in top_o]].T  # (4096, n_o)
        before_in_after = []
        for k in range(5):
            for row, (vecs, P, field, _ttl) in enumerate(((Vv, Ps[k], xs, "value read vectors V_c in the source planes"),
                                                          (Uo, Pe[k], E_after, "output write vectors U_o in the \"=\" planes"))):  # fmt: skip
                ax = axes[2 * i + row, k]
                mu = means(field, lab) @ P
                vn = (P.T @ vecs) / np.linalg.norm(vecs, axis=0)
                if P.shape[1] == 1:
                    mu = np.concatenate([mu, np.zeros_like(mu)], 1)
                    vn = np.concatenate([vn, np.zeros_like(vn)], 0)
                scale = np.abs(mu).max() / max(np.abs(vn).max(), 1e-12)
                for r in range(10):
                    ax.plot(*mu[r], "o", color=CM(r), ms=7)
                    ax.text(
                        *mu[r],
                        str(r),
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                    )
                for j in range(vn.shape[1]):
                    ax.annotate(
                        "",
                        xy=vn[:, j] * scale,
                        xytext=(0, 0),
                        arrowprops={"arrowstyle": "->", "color": "#555", "lw": 0.7},
                    )
                ax.plot([0], [0], "k+", ms=8)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                share = float(
                    (
                        np.linalg.norm(P.T @ vecs, axis=0) ** 2 / np.linalg.norm(vecs, axis=0) ** 2
                    ).mean()
                )
                ax.set_title(
                    f"{w_}: {'V_c' if row == 0 else 'U_o'}, k = {k + 1}; mean share in plane {share:.3f}\narrow scale ×{scale:.2g}",
                    fontsize=7,
                )
            # did the "=" planes hold the information before the write?
            pb, pa = psi(E_before, lab)[k], psi(E_after, lab)[k]
            nb = np.sqrt(
                np.linalg.norm(Pe[k].T @ pb.real) ** 2 + np.linalg.norm(Pe[k].T @ pb.imag) ** 2
            )
            na = np.sqrt(
                np.linalg.norm(Pe[k].T @ pa.real) ** 2 + np.linalg.norm(Pe[k].T @ pa.imag) ** 2
            )
            before_in_after.append(float(nb / na))
        from copy_analyze import loo_acc

        res[w_]["eq_before_over_after_in_eq_planes"] = before_in_after
        res[w_]["loo_acc_eq_before_in_after_D"] = loo_acc(E_before @ De, lab)
        res[w_]["loo_acc_eq_after_in_after_D"] = loo_acc(E_after @ De, lab)
        axes[2 * i, 0].set_ylabel(f"{w_} (block {L}): source position {s}", fontsize=8)
        axes[2 * i + 1, 0].set_ylabel(f'{w_}: "=" after the attention write', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "copy_vectors.png", dpi=100)
    plt.close(fig)

    # ---- figure 3: geometry of the copy at "=" (after the attention write), both models
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for i, (w_, (L, _s)) in enumerate(COPY.items()):
        d = data[w_]
        lab = d["lab"]
        for j, (name, Eb, out) in enumerate(
            (("alive-only", d["Xal"][:, 4], d["out_al"]), ("original", d["Xor"][:, 4], d["out_or"]))
        ):
            Ea = Eb + out
            _, Da = planes(Ea, lab)
            mu = means(Ea, lab)
            C_full, C_D = cosmat(mu), cosmat(mu @ Da)
            dd = cohen(Ea, lab)
            res[w_][f"{name}_cos_full_offdiag_range"] = [
                float(C_full[~np.eye(10, dtype=bool)].min()),
                float(C_full[~np.eye(10, dtype=bool)].max()),
            ]
            res[w_][f"{name}_cos_D_offdiag_range"] = [
                float(C_D[~np.eye(10, dtype=bool)].min()),
                float(C_D[~np.eye(10, dtype=bool)].max()),
            ]
            res[w_][f"{name}_cohen_min_median"] = [
                float(dd[~np.eye(10, dtype=bool)].min()),
                float(np.median(dd[~np.eye(10, dtype=bool)])),
            ]
            row = axes[2 * i + j]
            for ax, M, ttl, kw in ((row[0], C_full, "cos(μ_r, μ_r′), full stream", {"vmin": min(0.9, C_full.min()), "vmax": 1, "cmap": "viridis"}),
                                   (row[1], C_D, "cos inside D (period-10 part)", {"vmin": -1, "vmax": 1, "cmap": "RdBu_r"}),
                                   (row[2], dd, "Cohen's d along the centroid axis", {"vmin": 0, "vmax": max(1.0, dd.max()), "cmap": "magma"})):  # fmt: skip
                Mshow = M.copy()
                np.fill_diagonal(Mshow, np.nan)
                im = ax.imshow(Mshow, **kw)
                for y in range(10):
                    for x in range(10):
                        if x != y:
                            ax.text(
                                x,
                                y,
                                f"{M[y, x]:.2f}" if ax is not row[2] else f"{M[y, x]:.1f}",
                                ha="center",
                                va="center",
                                fontsize=5,
                            )
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.set_title(f"{w_} copy, block {L}, {name}: {ttl}", fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "copy_geometry.png", dpi=100)
    plt.close(fig)
    (OUT / "figs.json").write_text(json.dumps(res, indent=1))
    for w_ in COPY:
        r = res[w_]
        print(w_, "heads", r["active_heads"], "| before/after in eq planes", np.round(r["eq_before_over_after_in_eq_planes"], 3),
              "| loo acc before", r["loo_acc_eq_before_in_after_D"], "after", r["loo_acc_eq_after_in_after_D"],
              "| cos full", np.round(r["alive-only_cos_full_offdiag_range"], 3), np.round(r["original_cos_full_offdiag_range"], 3),
              "| cos D", np.round(r["alive-only_cos_D_offdiag_range"], 2), np.round(r["original_cos_D_offdiag_range"], 2),
              "| d", np.round(r["alive-only_cohen_min_median"], 2), np.round(r["original_cohen_min_median"], 2))  # fmt: skip


if __name__ == "__main__":
    main()
