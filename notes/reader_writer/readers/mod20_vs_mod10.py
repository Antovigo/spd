"""Overlap of the mod-20 plane P with the mod-10 subspace D10 at the MLP input of blocks 0..16 (position 1, token a).

x = RMSNorm(mid_b) * g2_b for a = 1..400 (as mod20_harmonics). mu_s (s = a mod 20, 0..19) and nu_r (r = a mod 10, 0..9)
are the residue means of x. P = span{Re Psi_1, Im Psi_1}, Psi_1 = sum_s mu_s exp(-2 pi i s/20) (2 dims). D10 = span of
Re, Im of Phi_k = sum_r nu_r exp(-2 pi i k r/10), k = 1..5 (9 dims: Im Phi_5 = 0). Neither contains the mean of x. Overlap:
the principal cosines between P and D10 (singular values of P^T D10), against the same for a random 9-dim subspace.
Projections: z = P^T x and z_perp = P^T (x - D10 D10^T x). Nothing is centred.

Writes <run>/analysis/reader_writer/track_mod20/vs_mod10.json and notebook_mod20/fig/mod20_vs_mod10_{overlap,alive,orig}.png.

    python mod20_vs_mod10.py
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import HUB, A, bf16_tensor
from mod20_harmonics import FIG, OUT, STREAM, TITLE
from track_build import EPS, LAST

S20, R10 = A % 20, A % 10
SHOW = (0, 4, 8, 12, 13, 16)


def means(X: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    return np.stack([X[lab == q].mean(0) for q in range(n)])


def span(vs: list[np.ndarray]) -> np.ndarray:
    u, s, _ = np.linalg.svd(np.stack(vs, 1), full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def subspaces(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.exp(-2j * np.pi * np.arange(20) / 20) @ means(x, S20, 20)
    nu = means(x, R10, 10)
    phis = [np.exp(-2j * np.pi * k * np.arange(10) / 10) @ nu for k in range(1, 6)]
    P = span([p.real, p.imag])
    if (means(x, S20, 20)[0] - means(x, S20, 20)[10]) @ P[:, 0] < 0:
        P[:, 0] *= -1
    return P, span([v for f in phis for v in (f.real, f.imag)])


def mlp_inputs(path: Path, g2: list[np.ndarray]) -> list[np.ndarray]:
    st = np.load(path)
    out = []
    for b in range(LAST + 1):
        h = st[f"mid_{b}"].astype(np.float64)
        out.append(h / np.sqrt((h**2).mean(1, keepdims=True) + EPS) * g2[b])
    return out


def clouds(xs: list[np.ndarray], mdl: str, path: Path) -> None:
    fig, axes = plt.subplots(2, len(SHOW), figsize=(3.0 * len(SHOW), 6.6))
    cmap = plt.get_cmap("twilight")
    for c, b in enumerate(SHOW):
        P, D = subspaces(xs[b])
        z = xs[b] @ P
        zp = (xs[b] - (xs[b] @ D) @ D.T) @ P
        lim = 1.08 * max(np.abs(z).max(), np.abs(zp).max())
        for r, (zz, name) in enumerate(((z, "x"), (zp, "x − Π_D10 x"))):
            ax = axes[r, c]
            ax.scatter(zz[:, 0], zz[:, 1], c=S20, cmap=cmap, vmin=0, vmax=20, s=7, lw=0)
            for s, m in enumerate(means(zz, S20, 20)):
                ax.text(m[0], m[1], str(s), fontsize=6.5, ha="center", va="center", color="#111")
            ax.axhline(0, color="#bbb", lw=0.5)
            ax.axvline(0, color="#bbb", lw=0.5)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"block {b} MLP input", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"P^T ({name})", fontsize=9)
    fig.suptitle(f"{TITLE[mdl]}: a at position 1 in the mod-20 plane P; colour and labels: a mod 20",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    g2 = []
    for b in range(LAST + 1):
        name = f"model.layers.{b}.post_attention_layernorm.weight"
        g2.append(bf16_tensor(snap + wmap[name], name, None))
    rng = np.random.default_rng(0)
    null = [np.linalg.svd(np.linalg.qr(rng.normal(size=(4096, 2)))[0].T
                          @ np.linalg.qr(rng.normal(size=(4096, 9)))[0], compute_uv=False)[0]
            for _ in range(200)]  # fmt: skip
    res = {"null_cos1_99": float(np.quantile(null, 0.99))}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for mdl, ls in zip(STREAM, ("-", "--"), strict=True):
        xs = mlp_inputs(STREAM[mdl], g2)
        cos, keep = [], []
        for x in xs:
            P, D = subspaces(x)
            cos.append(np.linalg.svd(P.T @ D, compute_uv=False).tolist())
            z = x @ P
            zp = (x - (x @ D) @ D.T) @ P
            ring = lambda zz: means(zz, S20, 20) - means(zz, S20, 20).mean(0)  # noqa: E731
            keep.append(float((ring(zp) ** 2).sum() / (ring(z) ** 2).sum()))
        res[mdl] = {"cos": cos, "ring_kept": keep}
        print(mdl, np.round(np.asarray(cos), 3).tolist(), np.round(keep, 3).tolist(), flush=True)
        c = np.asarray(cos)
        ax.plot(
            range(LAST + 1), c[:, 0], ls, color="#2a6fdb", label=f"{TITLE[mdl]}: largest cosine"
        )
        ax.plot(range(LAST + 1), c[:, 1], ls, color="#e07b39", label=f"{TITLE[mdl]}: second cosine")
        clouds(xs, mdl, FIG / f"mod20_vs_mod10_{mdl}.png")
    ax.axhline(
        res["null_cos1_99"], color="#888", lw=0.8, ls=":", label="random 9-dim subspace, 99th pct"
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(range(LAST + 1))
    ax.set_xlabel("block whose MLP input is used")
    ax.set_ylabel("principal cosine, P vs D10")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "mod20_vs_mod10_overlap.png", dpi=150)
    plt.close(fig)
    (OUT / "vs_mod10.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
