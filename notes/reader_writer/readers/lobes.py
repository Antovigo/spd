"""Why do mod-10 readers respond negatively at their residue +-3? Components and neurons, block 0, position 1.

Nothing is centred. Notation (also on the page):
  r0           a reader's main residue; Delta = r - r0 mod 10, shown in -4..5
  p(r)         = s m(r) / max |m|, the sign-aligned residue profile; m(r) = mu_r . w / ||w|| exactly
  G(r, r')     = Pi_D mu_r . Pi_D mu_r' (the Gram of the residue means inside D)
  model        p(r) = alpha delta(r, r0) + beta G(r0, r) / G(r0, r0) + gamma, least squares over the 10 residues
               (dual part alpha, matched-filter part beta, constant gamma)

    python lobes.py --out <page dir>
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import HUB, SRC, A, N, bf16_tensor, embedding_inputs, kept_readers
from neuron_page import fourier_planes, orth

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
NEURONS = [
    "g#13630",
    "g#8969",
    "g#10825",
    "g#8559",
    "g#7854",
    "g#5708",
    "g#11614",
    "g#5095",
    "u#10825",
    "g#13010",
    "u#8559",
    "u#13010",
    "u#13630",
    "u#5095",
    "u#5708",
    "u#8969",
    "g#6837",
    "u#2147",
    "u#5524",
    "g#5524",
    "u#7854",
    "u#1900",
    "g#1179",
    "g#291",
]
MIXED = {"g#6837", "u#2147", "u#1900", "g#1179", "g#291"}
COMPONENTS = ["g43", "g47", "g58", "g59", "g62", "g69", "u31", "u52", "u54", "u60", "u794"]
SHIFT = np.array([0, 1, 2, 3, 4, 5, -4, -3, -2, -1])  # Delta in -4..5 for (r - r0) mod 10


def gram_D(mu: np.ndarray, D: np.ndarray) -> np.ndarray:
    md = mu @ D
    return md @ md.T


def spectrum(mu: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Share of sum_r ||Pi_D mu_r||^2 at each frequency k = 1..5."""
    md = mu @ D
    rr = np.arange(10)
    e = np.array(
        [
            np.linalg.norm(np.exp(-2j * np.pi * k * rr / 10) @ md) ** 2 * (1 if k == 5 else 2)
            for k in range(1, 6)
        ]
    )
    return e / e.sum()


def circ_from_spectrum(sh: np.ndarray) -> np.ndarray:
    """Normalised circulant Gram value at Delta = 0..9 implied by frequency shares sh (k = 1..5)."""
    d = np.arange(10)
    return np.array(
        [sum(sh[k - 1] * np.cos(2 * np.pi * k * dd / 10) for k in range(1, 6)) for dd in d]
    )


def fit(p: np.ndarray, r0: int, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([np.eye(10)[r0], G[r0] / G[r0, r0], np.ones(10)], 1)
    coef, *_ = np.linalg.lstsq(X, p, rcond=None)
    return coef, X @ coef


def analyse(label: str, mu: np.ndarray, D: np.ndarray, Wrows: np.ndarray, names: list[str]) -> dict:
    """Wrows: (n, 4096) read vectors (any norm)."""
    G = gram_D(mu, D)
    out: dict = {"readers": {}}
    for nm, w in zip(names, Wrows, strict=True):
        m = mu @ (w / np.linalg.norm(w))
        r0 = int(np.argmax(np.abs(m)))
        p = np.sign(m[r0]) * m / np.abs(m).max()
        coef, pred = fit(p, r0, G)
        off = np.arange(10) != r0
        r2_off = 1 - ((p - pred)[off] ** 2).sum() / (p[off] ** 2).sum()
        coef0, pred0 = (
            np.linalg.lstsq(np.stack([np.eye(10)[r0], np.ones(10)], 1), p, rcond=None)[0],
            None,
        )
        pred0 = np.stack([np.eye(10)[r0], np.ones(10)], 1) @ coef0
        r2_off_dual = 1 - ((p - pred0)[off] ** 2).sum() / (p[off] ** 2).sum()
        idx3 = [(r0 + 3) % 10, (r0 - 3) % 10]
        out["readers"][nm] = {
            "r0": r0,
            "p": p.tolist(),
            "alpha": float(coef[0]),
            "beta": float(coef[1]),
            "gamma": float(coef[2]),
            "r2_off_model": float(r2_off),
            "r2_off_dual_const": float(r2_off_dual),
            "lobe3": float(p[idx3].mean()),
            "lobe3_pred": float(pred[idx3].mean()),
            "pred": pred.tolist(),
        }
    n = np.sqrt(np.diag(G))
    C = G / np.outer(n, n)
    d = (np.arange(10)[:, None] - np.arange(10)[None, :]) % 10
    out["cos_by_delta"] = {
        int(dd): [float(C[(d == dd)].mean()), float(C[(d == dd)].min()), float(C[(d == dd)].max())]
        for dd in range(1, 10)
    }
    out["spectrum"] = spectrum(mu, D).tolist()
    out["circ_pred"] = circ_from_spectrum(np.array(out["spectrum"])).tolist()
    out["label"] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neuron_data", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "fig").mkdir(parents=True, exist_ok=True)
    rr = A % 10
    # neurons: original model
    x = np.load(args.neuron_data / "l0_input.npz")["x"].astype(np.float64)
    mu_n = np.stack([x[rr == r].mean(axis=0) for r in range(10)])
    Dn = orth(np.concatenate(list(fourier_planes(mu_n).values()), 1))
    (snap,) = glob.glob(HUB)
    idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    rows = []
    for nm in NEURONS:
        k, i = nm.split("#")
        key = f"model.layers.0.mlp.{ {'g': 'gate', 'u': 'up'}[k] }_proj.weight"
        rows.append(bf16_tensor(snap + idx[key], key, np.asarray([int(i)]))[0])
    neu = analyse("neurons", mu_n, Dn, np.stack(rows).astype(np.float64), NEURONS)
    # components: alive-only model
    cols, cnames, V = kept_readers()
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T
    mu_c = np.stack([xv[rr == r].mean(axis=0) for r in range(10)])
    Dc = orth(np.concatenate(list(fourier_planes(mu_c).values()), 1))
    com = analyse(
        "components", mu_c, Dc, np.stack([V[:, cnames.index(c)] for c in COMPONENTS]), COMPONENTS
    )
    # embedding geometry
    z = embedding_inputs()
    mu_e = np.stack([z[rr == r].mean(axis=0) for r in range(10)])
    De = orth(np.concatenate(list(fourier_planes(mu_e).values()), 1))
    emb = {"spectrum": spectrum(mu_e, De).tolist()}
    Ge = gram_D(mu_e, De)
    ne = np.sqrt(np.diag(Ge))
    Ce = Ge / np.outer(ne, ne)
    d = (np.arange(10)[:, None] - np.arange(10)[None, :]) % 10
    emb["cos_by_delta"] = {
        int(dd): [
            float(Ce[(d == dd)].mean()),
            float(Ce[(d == dd)].min()),
            float(Ce[(d == dd)].max()),
        ]
        for dd in range(1, 10)
    }

    # figure 1: geometry of the residue means inside D, and the spectrum behind it
    fig, axs = plt.subplots(1, 3, figsize=(17, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    dd = np.arange(1, 10)
    dsh = np.array([SHIFT[x_] for x_ in dd])
    order = np.argsort(dsh)
    for lab, res_, col, mk in (
        ("components (alive-only x_V)", com, "#2a5db0", "o"),
        ("neurons (original x)", neu, "#b5442d", "s"),
        ("embedding z", emb, "#6b8e23", "^"),
    ):
        mean = np.array([res_["cos_by_delta"][int(x_)][0] for x_ in dd])
        lo = np.array([res_["cos_by_delta"][int(x_)][1] for x_ in dd])
        hi = np.array([res_["cos_by_delta"][int(x_)][2] for x_ in dd])
        axs[0].errorbar(
            dsh[order] + {"o": -0.12, "s": 0, "^": 0.12}[mk],
            mean[order],
            yerr=[mean[order] - lo[order], hi[order] - mean[order]],
            fmt=mk + "-",
            color=col,
            ms=5,
            lw=1,
            capsize=2,
            label=lab,
        )
    axs[0].axhline(
        -1 / 9, color="#888", ls="--", lw=0.8, label="−1/9 (equal energy at every frequency)"
    )
    axs[0].axhline(0, color="#ccc", lw=0.6)
    axs[0].set_xticks(dsh[order])
    axs[0].set_xlabel("Δ = r′ − r (mod 10)")
    axs[0].set_ylabel("cos(Π_D μ_r, Π_D μ_r′)")
    axs[0].set_title("residue means inside D: mean (min–max) over pairs at each Δ")
    axs[0].legend(frameon=False, fontsize=7)
    wbar = 0.27
    for t, (lab, res_, col) in enumerate(
        (("components", com, "#2a5db0"), ("neurons", neu, "#b5442d"), ("embedding", emb, "#6b8e23"))
    ):
        axs[1].bar(np.arange(1, 6) + (t - 1) * wbar, res_["spectrum"], wbar, color=col, label=lab)
    axs[1].plot([0.6, 4.4], [2 / 9, 2 / 9], color="#888", ls="--", lw=0.8)
    axs[1].plot([4.6, 5.4], [1 / 9, 1 / 9], color="#888", ls="--", lw=0.8)
    ref = np.array([2, 2, 2, 2, 1]) / 9
    cos3 = np.cos(2 * np.pi * 3 * np.arange(1, 6) / 10)
    contrib = (np.array(neu["spectrum"]) - ref) * cos3
    neu["delta3_contrib"] = contrib.tolist()
    axs[2].bar(np.arange(1, 6), contrib, color=["#b5442d" if c < 0 else "#2a5db0" for c in contrib])
    axs[2].axhline(0, color="#888", lw=0.6)
    axs[2].set_xlabel("frequency k")
    axs[2].set_ylabel("(share_k − equal share_k) · cos(2π·3k/10)")
    axs[2].set_title(f"why Δ = ±3: sum = {contrib.sum():+.3f} on top of −1/9")
    axs[1].set_xlabel("frequency k (k = 1..4: plane, counted twice; k = 5: line)")
    axs[1].set_ylabel("share of Σ_r ‖Π_D μ_r‖²")
    axs[1].set_title("Fourier energy of the residue means (dashed: equal energy per dimension)")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "lobes_geometry.png")
    plt.close(fig)

    # figure 2: profiles by Delta, data vs model, for both reader types
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
    for ax, res_, ttl in (
        (axs[0], com, "components (11)"),
        (
            axs[1],
            {"readers": {k: v for k, v in neu["readers"].items() if k not in MIXED}},
            "clean neurons (19)",
        ),
    ):
        for v in res_["readers"].values():
            p, pr = np.array(v["p"]), np.array(v["pred"])
            sh = SHIFT[(np.arange(10) - v["r0"]) % 10]
            o = np.argsort(sh)
            ax.plot(sh[o], p[o], "-", color="#999", lw=0.7)
            ax.plot(sh[o], pr[o], ":", color="#2a5db0", lw=0.9)
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_ylim(-0.7, 0.35)
        ax.set_xticks(range(-4, 6))
        ax.set_xlabel("Δ = r − r₀")
        ax.set_ylabel("p(r) (main residue at 1, off the top)")
        ax.set_title(f"{ttl}: data (grey) and α δ + β G/G₀ + γ (blue dotted)")
    allr = {
        **{("c", k): v for k, v in com["readers"].items()},
        **{("n", k): v for k, v in neu["readers"].items()},
    }
    for (kind, nm), v in allr.items():
        col = "#2a5db0" if kind == "c" else ("#e39a7d" if nm in MIXED else "#b5442d")
        axs[2].scatter(v["lobe3_pred"], v["lobe3"], color=col, s=22, edgecolor="k", lw=0.4)
        axs[2].annotate(
            nm, (v["lobe3_pred"], v["lobe3"]), fontsize=6, xytext=(2, 2), textcoords="offset points"
        )
    lim = [-0.8, 0.2]
    axs[2].plot(lim, lim, color="#888", lw=0.7)
    axs[2].set_xlim(lim)
    axs[2].set_ylim(lim)
    axs[2].set_xlabel("model at r₀ ± 3")
    axs[2].set_ylabel("data: mean of p(r₀ + 3), p(r₀ − 3)")
    axs[2].set_title("blue: components; red: clean neurons; pink: mixed")
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "lobes_fit.png")
    plt.close(fig)

    summary = {"components": com, "neurons": neu, "embedding": emb}
    (args.out / "lobes.json").write_text(json.dumps(summary, indent=1))
    for lab, res_ in (("components", com), ("neurons", neu)):
        print(
            lab,
            "spectrum",
            np.round(res_["spectrum"], 3),
            "cos by delta",
            {k: round(v[0], 3) for k, v in res_["cos_by_delta"].items()},
        )
        print("  circ pred", np.round(res_["circ_pred"], 3))
        for nm, v in res_["readers"].items():
            print(
                f"  {nm:8s} r0={v['r0']} a={v['alpha']:.2f} b={v['beta']:.2f} c={v['gamma']:.2f} r2off={v['r2_off_model']:.2f} "
                f"(dual+const {v['r2_off_dual_const']:.2f}) lobe3={v['lobe3']:.2f} pred={v['lobe3_pred']:.2f}"
            )
    print(
        "embedding spectrum",
        np.round(emb["spectrum"], 3),
        {k: round(v[0], 3) for k, v in emb["cos_by_delta"].items()},
    )


if __name__ == "__main__":
    main()
