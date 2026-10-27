"""How attention copies the a mod 20 / b mod 20 circle to the last position "=" (block 16 copies a from position 1,
block 15 copies b from position 3), in the alive-only model (components) and the original model (neurons).

Prompts: copy_build.py's sets. Set "a" (every a = 1..400, ten b, both ops; 8,000 prompts) for block 16; set "b" (ten a,
every b = 1..400, both ops) for block 15. The original model uses copy_figs.py's cached streams (2,000 prompts per set).
Nothing is centred except the within-class spread of Cohen's d. Notation, for one copy (block L, source position u,
copied operand o = a or b, residue s = o mod 20, r = o mod 10):
  x_u        attention input (input RMSNorm output) at the source position u
  O          the attention output at "=" that comes from position u (all heads; copy_analyze.copied); O^h its part
             through head h (0..31); O^c its part through value component c
  E, E'      the stream at "=" entering block L, and after the attention write (E plus the attention output from all
             positions)
  Psi_j(Z)   sum_{s=0..19} m_s(Z) e^{-2 pi i j s / 20}, m_s(Z) the mean of Z over the prompts with o mod 20 = s; j = 1..10
             (odd j tell s from s + 10; j = 2k is the a mod 10 frequency k)
  null99_j   99th percentile of ||Psi_j|| over 100 relabellings of the prompts (seed 0)
  gain_j     ||Psi_j(O)|| / ||Psi_j(x_u)||: how much of frequency j the copy carries, relative to the source
  plane P(Z) span{Re Psi_1(Z), Im Psi_1(Z)};  o_s(Z) = P^T (m_s - m_{s+10}) / 2 (the circle's odd part, s = 0..9)
  map r2     uncentred r2 of o_s(O) ~= o_s(x_u) M with one 2 x 2 matrix M (is the copy one linear map of the circle?)
  roundness  smaller over larger singular value of the 10 x 2 matrix of odd parts o_s (1 = round ring)
  alpha      the quantization index of the mod-20 page (1: evenly spaced ring, 0: four points)
  acc_h      held-out accuracy of h = floor(o/10) mod 2 given r in P (plane and class means fitted on o <= 200, tested on
             o > 200, and the reverse); d_anti: median Cohen's d of residues s, s + 10 in P
  head / vcomp / ocomp shares: ||Psi_1(O^h)||^2 / sum_h ||Psi_1(O^h)||^2 (period-20), next to the same with
             sum_{k=1..5} ||Psi_{2k}||^2 (period-10)
Writes <run>/analysis/reader_writer/track_mod20/copy.json and <out>/mod20_copy_{alive,orig}.png.

    python mod20_copy.py --out notebook_mod20/fig
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy_analyze import attention_parts, copied
from copy_build import OUT as COPY
from copy_figs import SUB
from mlp0_mod10 import rope
from mod10_L0 import HUB, bf16_tensor
from mod20_track import OUT
from track_build import load_blocks, rmsnorm

WHICH = {"a": (16, 1), "b": (15, 3)}
QCM = [plt.get_cmap(c) for c in ("Blues", "Oranges", "Greens", "Purples")]
ODD = (1, 3, 5, 7, 9)


def m20(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.stack([Z[lab == s].mean(0) for s in range(20)])


def psi(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * np.outer(np.arange(1, 11), np.arange(20)) / 20) @ m20(
        Z, lab
    )  # (10, d)


def plane(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    p = psi(Z, lab)[0]
    return np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)[0]


def odd(Z: np.ndarray, lab: np.ndarray, P: np.ndarray) -> np.ndarray:
    m = m20(Z, lab) @ P
    return (m[:10] - m[10:]) / 2


def alpha(o: np.ndarray) -> float:
    oc = np.concatenate([o[:, 0] + 1j * o[:, 1], -(o[:, 0] + 1j * o[:, 1])])
    th = np.unwrap(np.angle(oc * np.conj(oc[0])))
    th = th * np.sign(th[-1] - th[0])
    s = np.arange(20)
    best = (0.0, np.inf)
    for al in np.linspace(0, 1, 101):
        pred = 2 * np.pi / 20 * (al * s + (1 - al) * 5 * (s // 5))
        c = np.angle(np.exp(1j * (th - pred)).mean())
        e = (np.angle(np.exp(1j * (th - pred - c))) ** 2).sum()
        if e < best[1]:
            best = (float(al), float(e))
    return best[0]


def acc_h(Z: np.ndarray, lab: np.ndarray, oper: np.ndarray) -> float:
    half = oper <= 200
    ok = 0
    for tr, te in ((half, ~half), (~half, half)):
        P = plane(Z[tr], lab[tr])
        cs = m20(Z[tr] @ P, lab[tr])
        ze = Z[te] @ P
        r = lab[te] % 10
        d0 = ((ze - cs[r]) ** 2).sum(1)
        d1 = ((ze - cs[r + 10]) ** 2).sum(1)
        ok += int(((d1 < d0) == (lab[te] >= 10)).sum())
    return ok / len(lab)


def d_anti(Z: np.ndarray, lab: np.ndarray, P: np.ndarray) -> float:
    z = Z @ P
    out = []
    for s in range(10):
        a, b = z[lab == s], z[lab == s + 10]
        u = a.mean(0) - b.mean(0)
        n = np.linalg.norm(u)
        u = u / n
        out.append(n / np.sqrt(((a @ u).var(ddof=1) + (b @ u).var(ddof=1)) / 2))
    return float(np.median(out))


def describe(
    Z: np.ndarray, lab: np.ndarray, oper: np.ndarray, rng: np.random.Generator, null: bool
) -> dict:
    ps = psi(Z, lab)
    P = plane(Z, lab)
    o = odd(Z, lab, P)
    sv = np.linalg.svd(o, compute_uv=False)
    out = {"amp": np.linalg.norm(ps, axis=1).tolist(), "alpha": alpha(o), "acc_h": acc_h(Z, lab, oper),
           "d_anti": d_anti(Z, lab, P), "roundness": float(sv[1] / sv[0]), "odd": o.tolist(), "_P": P, "_odd": o}  # fmt: skip
    if null:
        nl = [np.linalg.norm(psi(Z, rng.permutation(lab)), axis=1) for _ in range(100)]
        out["null99"] = np.quantile(np.asarray(nl), 0.99, axis=0).tolist()
    return out


def shares(parts_out, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """parts_out: an iterable of fields (n, 4096), consumed one at a time."""
    e20, e10 = [], []
    for Oc in parts_out:
        p = psi(Oc, lab)
        e20.append(float(np.abs(p[0]) @ np.abs(p[0])))
        e10.append(float(sum(np.abs(p[k - 1]) @ np.abs(p[k - 1]) for k in (2, 4, 6, 8, 10))))
    e20, e10 = np.asarray(e20), np.asarray(e10)
    return e20 / e20.sum(), e10 / e10.sum()


def orig_parts(snap: str, wmap: dict, cfg: dict, L: int, X: np.ndarray):
    """Original block L on streams X (n, 5, 4096): input-norm output x, pattern p (n, 32, 5) at "=", values v
    (n, 5, 32, 128) per head, and the output projection reshaped to (4096, 32, 128)."""

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
    return x, p, v, w(pre + "self_attn.o_proj.weight").reshape(4096, H, dh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    blocks, _, _ = load_blocks()
    alive = json.loads((Path(str(COPY).replace("analysis/reader_writer/copy_mod10", "addsub-08-last-pos-40000")) / "alive/alive.json").read_text())["components"]  # fmt: skip
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    res: dict = {}
    figs: dict = {}
    for which, (L, u) in WHICH.items():
        st = np.load(COPY / f"stream_{which}.npz")
        blk = blocks[L]
        blk.a_ids = {k: np.asarray(alive[f"layers.{L}.self_attn.{k}_proj"]) for k in "qkvo"}
        # components model, all 8,000 prompts
        X = st[f"X{L}"].astype(np.float64)
        oper = st[which].astype(int)
        lab = oper % 20
        parts = attention_parts(blk, X)
        print(which, "attention parts done", flush=True)
        Ocp = copied(blk, parts, u)
        att = sum(copied(blk, parts, t) for t in range(5))
        xs = parts["x"][:, u]
        r = {"weight_on_source": float(parts["p"][:, :, u].mean())}
        for name, Z, nl in (
            ("source", xs, True),
            ("copy", Ocp, True),
            ("eq_before", X[:, 4], True),
            ("eq_after", X[:, 4] + att, True),
        ):
            r[name] = describe(Z, lab, oper, rng, nl)
        r["gain"] = (np.asarray(r["copy"]["amp"]) / np.asarray(r["source"]["amp"])).tolist()
        os_, oc_ = r["source"]["_odd"], r["copy"]["_odd"]
        M, *_ = np.linalg.lstsq(os_, oc_, rcond=None)
        r["map_r2"] = float(1 - ((oc_ - os_ @ M) ** 2).sum() / (oc_**2).sum())
        r["cos_copy_eq_after_planes"] = np.linalg.svd(
            r["copy"]["_P"].T @ r["eq_after"]["_P"], compute_uv=False
        ).tolist()
        h20, h10 = shares((copied(blk, parts, u, heads=np.asarray([h])) for h in range(32)), lab)
        r["heads"] = {"p20": h20.tolist(), "p10": h10.tolist()}
        Cv = blk.a["V_v"].shape[1]
        v20, v10 = shares((copied(blk, parts, u, vcomps=np.asarray([c])) for c in range(Cv)), lab)
        ov = np.argsort(-v20)[:8]
        r["vcomps"] = [
            {"id": int(blk.a_ids["v"][c]), "p20": float(v20[c]), "p10": float(v10[c])} for c in ov
        ]
        r["vcomps_p10_top"] = [
            {"id": int(blk.a_ids["v"][c]), "p20": float(v20[c]), "p10": float(v10[c])}
            for c in np.argsort(-v10)[:8]
        ]
        res[f"alive_{which}"] = r
        figs[("alive", which)] = (r, lab)
        print("alive", which, f"gain j1 {r['gain'][0]:.2f}", "gains mod10", np.round(np.asarray(r["gain"])[[1, 3, 5, 7, 9]], 2),
              f"map r2 {r['map_r2']:.3f}", f"alpha src {r['source']['alpha']:.2f} copy {r['copy']['alpha']:.2f} eq_after {r['eq_after']['alpha']:.2f}",
              "acc_h", [round(r[k]["acc_h"], 2) for k in ("source", "copy", "eq_before", "eq_after")], flush=True)  # fmt: skip
        # original model, cached streams (2,000 prompts)
        sub = SUB[which]
        Xo = np.load(COPY / f"orig_cache_{which}.npz")["Xor"].astype(np.float64)
        opo = st[which][sub].astype(int)
        lo = opo % 20
        x, p, v, Wo = orig_parts(snap, wmap, cfg, L, Xo)
        Oh = np.einsum("nhd,ohd->nho", p[:, :, u, None] * v[:, u], Wo).astype(
            np.float32
        )  # (n, 32, 4096)
        Oo = Oh.sum(1).astype(np.float64)
        atto = np.einsum("nht,nthd->nhd", p, v).reshape(len(Xo), -1) @ Wo.reshape(4096, -1).T
        ro = {"weight_on_source": float(p[:, :, u].mean())}
        for name, Z in (
            ("source", x[:, u]),
            ("copy", Oo),
            ("eq_before", Xo[:, 4]),
            ("eq_after", Xo[:, 4] + atto),
        ):
            ro[name] = describe(Z, lo, opo, rng, True)
        ro["gain"] = (np.asarray(ro["copy"]["amp"]) / np.asarray(ro["source"]["amp"])).tolist()
        os_, oc_ = ro["source"]["_odd"], ro["copy"]["_odd"]
        M, *_ = np.linalg.lstsq(os_, oc_, rcond=None)
        ro["map_r2"] = float(1 - ((oc_ - os_ @ M) ** 2).sum() / (oc_**2).sum())
        h20, h10 = shares((Oh[:, h].astype(np.float64) for h in range(32)), lo)
        ro["heads"] = {"p20": h20.tolist(), "p10": h10.tolist()}
        res[f"orig_{which}"] = ro
        figs[("orig", which)] = (ro, lo)
        print("orig", which, f"gain j1 {ro['gain'][0]:.2f}", "gains mod10", np.round(np.asarray(ro["gain"])[[1, 3, 5, 7, 9]], 2),
              f"map r2 {ro['map_r2']:.3f}", f"alpha src {ro['source']['alpha']:.2f} copy {ro['copy']['alpha']:.2f} eq_after {ro['eq_after']['alpha']:.2f}",
              "acc_h", [round(ro[k]["acc_h"], 2) for k in ("source", "copy", "eq_before", "eq_after")], flush=True)  # fmt: skip
        del Oh, v
    for model in ("alive", "orig"):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
        for row, which in enumerate(WHICH):
            r, lab = figs[(model, which)]
            for col, (name, title) in enumerate((("source", "source position"), ("copy", 'copy written at "="'),
                                                 ("eq_after", '"=" after the attention'))):  # fmt: skip
                ax = axes[row, col]
                o = r[name]["_odd"]
                full = np.concatenate([o, -o])
                ang = np.arctan2(full[0, 1], full[0, 0])
                rot = np.asarray([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
                full = full @ rot.T
                for s in range(20):
                    ax.scatter(*full[s], s=60, color=QCM[s // 5](0.45 + 0.12 * (s % 5)), zorder=3)
                    ax.text(*(full[s] * 1.13), str(s), fontsize=7, ha="center", va="center")
                ax.plot(*np.vstack([full, full[:1]]).T, color="0.75", lw=0.8, zorder=1)
                ax.plot([0], [0], "k+")
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                L, u = WHICH[which]
                ax.set_title(
                    f"{which} (block {L}): {title}\nα {r[name]['alpha']:.2f}, h|r acc {r[name]['acc_h']:.2f}, d(s,s+10) {r[name]['d_anti']:.1f}",
                    fontsize=8,
                )
            ax = axes[row, 3]
            x = np.arange(32)
            ax.bar(x - 0.2, r["heads"]["p20"], 0.4, label="period 20 (j = 1)", color="#b5442d")
            ax.bar(
                x + 0.2,
                r["heads"]["p10"],
                0.4,
                label="period 10 (j = 2, 4, …, 10)",
                color="#1f4e9c",
            )
            ax.set_xlabel("head")
            ax.set_ylabel("share of the copy's energy")
            ax.set_title(f"{which} (block {WHICH[which][0]}): heads", fontsize=8)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out / f"mod20_copy_{model}.png", dpi=100)
        plt.close(fig)
    clean = {k: {kk: ({k3: v3 for k3, v3 in vv.items() if not k3.startswith("_")} if isinstance(vv, dict) else vv) for kk, vv in v.items()}
             for k, v in res.items()}  # fmt: skip
    (OUT / "copy.json").write_text(json.dumps(clean, indent=1))


if __name__ == "__main__":
    main()
