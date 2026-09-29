"""Figures for notes/rmsnorm/entropy-components.md (CPU; needs the lm_head for the W_U spectrum).

    python -m param_decomp.arith_repr.rmsnorm.figures <out_dir>

Reads `<run>/analysis/rmsnorm/temperature/{deep_addsub,dose}.npz`, `entropy/*.npz`, `claims/claims.npz`."""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, snapshot
from param_decomp.arith_repr.rmsnorm.temperature import CANDIDATES, CONTROLS, TEMP
from param_decomp.arith_repr.vectors.common import RESID

_ = ml_dtypes.bfloat16  # importing ml_dtypes registers bfloat16 for the safetensors reads
ENT = TEMP / "entropy"
CL = TEMP / "claims"
ENT6 = [1209, 2398, 2564, 3191, 5966, 6696]
# reference palette (dataviz skill): categorical slots 1-3, muted ink for context
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, INK, INK2, GRID, BASE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SHORT = {n: n.replace("L31.", "").replace("L30.", "L30 ") for n in CANDIDATES + CONTROLS}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "legend.frameon": False, "lines.linewidth": 2,
    "font.family": "sans-serif",
})  # fmt: skip


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print("wrote", name, flush=True)


def fig_activation(out: Path) -> None:
    from tokenizers import Tokenizer

    c = np.load(ENT / "components_fineweb.npz")
    a = np.load(TEMP / "deep_addsub.npz")
    tok = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    t = c["tokens"]
    dec = {int(i): tok.decode([int(i)]).strip() for i in np.unique(t)}
    nxt = np.concatenate([t[:, 1:], np.full((len(t), 1), -1)], 1)
    isnum = np.vectorize(
        lambda i: i >= 0 and dec.get(int(i), "").replace(",", "").replace(".", "").isdigit()
    )(nxt)
    m = np.zeros(t.shape, bool)
    m[:, 1:-1] = True
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    inn = c["L31.gate.c14__inner"][m]
    eq = a["L31.gate.c14__inner"][:, -1]
    bins = np.linspace(-10, 50, 121)
    ax[0].hist(inn, bins=bins, color=BLUE, label="fineweb positions")
    ax[0].hist(eq, bins=bins, color=ORANGE, label="addsub '=' (20k prompts)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("inner activation of gate c14  (x · V)")
    ax[0].set_ylabel("positions")
    ax[0].set_title("c14 varies on general text, is pinned high on '='")
    ax[0].legend(loc="upper center")
    for nm in CANDIDATES:
        v = c[f"{nm}__inner"][m]
        qs = np.percentile(v, [0, 50, 80, 90, 95, 98, 99, 99.5, 99.9, 100])
        mid, p = [], []
        for lo, hi in zip(qs[:-1], qs[1:], strict=False):
            s = (v >= lo) & (v <= hi)
            mid.append(np.median(v[s]))
            p.append(isnum[m][s].mean())
        col = BLUE if nm == "L31.gate.c14" else ("#86b6ef" if "gate" in nm else "#1c5cab")
        ax[1].plot(mid, p, color=col, marker="o", ms=3, lw=1.5)
        ax[1].annotate(
            SHORT[nm],
            (mid[-1], p[-1]),
            fontsize=7,
            color=INK2,
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax[1].axhline(isnum[m].mean(), color=MUTED, lw=1)
    ax[1].text(15, isnum[m].mean() + 0.02, "base rate", color=MUTED, fontsize=8)
    ax[1].set_xlabel("inner activation (median of quantile bin)")
    ax[1].set_ylabel("P(next token is a number)")
    ax[1].set_title("what drives them: a number comes next")
    save(fig, out, "fig1_activation.png")


def fig_dose(out: Path) -> None:
    z = np.load(TEMP / "dose.npz")
    al = z["alphas"]
    alpha = np.concatenate([al[:4], [1.0], al[4:]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for nm in CONTROLS + CANDIDATES:
        cand = nm in CANDIDATES
        h_a = [z[f"{nm}__a{a}__H_a"].mean() for a in al]
        Hb = z[f"{nm}__a{al[0]}__H_b"].mean()
        h = np.concatenate([h_a[:4], [Hb], h_a[4:]])
        te = [
            1 - z[f"{nm}__a{a}__kl_resid"].mean() / max(z[f"{nm}__a{a}__kl_total"].mean(), 1e-12)
            for a in al
        ]
        col, lw = (BLUE, 2) if cand else (MUTED, 1)
        ax[0].plot(alpha, h, color=col, lw=lw, marker="o", ms=3)
        ax[1].plot(al, te, color=col, lw=lw, marker="o", ms=3)
        if nm in ("L31.gate.c14", "L31.up.c731", "L31.v.c28"):
            ax[0].annotate(
                SHORT[nm],
                (alpha[-1], h[-1]),
                fontsize=7,
                color=INK2,
                xytext=(3, 0),
                textcoords="offset points",
            )
            ax[1].annotate(
                SHORT[nm],
                (al[-1], te[-1]),
                fontsize=7,
                color=INK2,
                xytext=(3, 0),
                textcoords="offset points",
            )
    ax[0].set_xlabel("activation scale α (1 = the model)")
    ax[0].set_ylabel("output entropy at '=' (nats)")
    ax[0].set_title("scaling the component turns a temperature dial")
    ax[1].set_xlabel("activation scale α")
    ax[1].set_ylabel("share of KL explained by a temperature change")
    ax[1].set_ylim(-0.02, 1)
    ax[1].set_title("…and at every dose the change is mostly temperature")
    ax[0].plot([], [], color=BLUE, label="candidates (6)")
    ax[0].plot([], [], color=MUTED, lw=1, label="controls (6)")
    ax[0].legend(loc="upper left")
    save(fig, out, "fig2_dose.png")


def fig_addsub(out: Path) -> None:
    a = np.load(TEMP / "deep_addsub.npz")
    names = CANDIDATES + CONTROLS
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for i, nm in enumerate(names):
        g = lambda k, abl="zero": a[f"{nm}__{abl}__{k}"]  # noqa: E731, B023
        kl = g("kl_total").mean()
        te = 1 - g("kl_resid").sum() / g("kl_total").sum()
        med = 1 - g("kl_direct").sum() / g("kl_total").sum()
        col = BLUE if nm in CANDIDATES else MUTED
        ax[0].plot([kl], [y[i]], "o", color=col, ms=6)
        ax[0].plot([g("kl_total", "mean").mean()], [y[i]], "o", mfc="none", mec=col, ms=6)
        ax[1].plot([te], [y[i]], "o", color=col, ms=6)
        ax[2].plot([med], [y[i]], "o", color=col, ms=6)
    ax[0].set_xscale("log")
    ax[0].set_yticks(y, [SHORT[n] for n in names])
    ax[0].set_xlabel("KL to the model (log)")
    ax[0].set_title("ablation effect at '='")
    ax[0].plot([], [], "o", color=INK2, label="zero ablation")
    ax[0].plot([], [], "o", mfc="none", mec=INK2, label="mean ablation")
    ax[0].legend(loc="lower left", fontsize=7)
    ax[1].set_xlim(-0.1, 1)
    ax[1].set_xlabel("share explained by temperature")
    ax[1].set_title("temperature purity")
    ax[2].set_xlim(-0.2, 1)
    ax[2].set_xlabel("1 − DE/TE (final-norm mediated)")
    ax[2].set_title("goes through the final norm")
    save(fig, out, "fig3_addsub.png")


def fig_bins(out: Path) -> None:
    c = np.load(ENT / "components_fineweb.npz")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    labels = ["<p50", "p50–90", "p90–99", "p99–99.9", ">p99.9"]
    for nm in CANDIDATES:
        inn = c[f"{nm}__inner"][:, 1:].ravel()
        g = lambda k: c[f"{nm}__mean__{k}"][:, 1:].ravel()  # noqa: E731, B023
        kl, vd, vr, kd = g("kl_total"), g("var_d"), g("var_resid"), g("kl_direct")
        qs = np.percentile(inn, [0, 50, 90, 99, 99.9, 100])
        K, R, D = [], [], []
        for lo, hi in zip(qs[:-1], qs[1:], strict=False):
            s = (inn >= lo) & (inn <= hi)
            K.append(kl[s].mean())
            R.append(1 - (vr[s] * kl[s]).sum() / max((vd[s] * kl[s]).sum(), 1e-12))
            D.append(1 - kd[s].sum() / max(kl[s].sum(), 1e-12))
        col = "#86b6ef" if "gate" in nm else "#1c5cab"
        for j, v in enumerate((K, R, D)):
            ax[j].plot(range(5), v, color=col, marker="o", ms=3, lw=1.5)
        ax[0].annotate(
            SHORT[nm], (4, K[-1]), fontsize=7, color=INK2, xytext=(3, 0), textcoords="offset points"
        )
    ax[0].set_yscale("log")
    ax[0].set_ylabel("KL of mean ablation")
    ax[0].set_title("effect grows with activation")
    ax[1].set_ylabel("temperature R² (KL-weighted)")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("…and becomes more temperature-like")
    ax[2].set_ylabel("1 − DE/TE")
    ax[2].set_ylim(-0.8, 1)
    ax[2].axhline(0, color=BASE, lw=1)
    ax[2].set_title("…and more final-norm mediated")
    for x in ax:
        x.set_xticks(range(5), labels, fontsize=7)
        x.set_xlabel("activation quantile bin (fineweb)")
    ax[1].plot([], [], color="#86b6ef", label="gate c14/c238/c288")
    ax[1].plot([], [], color="#1c5cab", label="up c534/c36/c50")
    ax[1].legend(loc="lower right", fontsize=7)
    save(fig, out, "fig4_fineweb_bins.png")


def unembed_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = Weights()
    g = np.load(RESID / "norms.npz")["final"]
    Weff = w.get("lm_head.weight") * g[None]
    ev, evec = np.linalg.eigh((Weff.T @ Weff).astype(np.float64))
    u = Weff.sum(0)
    return ev, evec, u / np.linalg.norm(u)


def fig_where(out: Path, ev: np.ndarray, evec: np.ndarray, u: np.ndarray) -> None:
    c = np.load(ENT / "components_fineweb.npz")
    ce = np.load(ENT / "components_eq.npz")
    fl = c["flagged"][c["dump_mask"]]
    B40, B512 = evec[:, :40], evec[:, 40:512]
    u1 = u - evec[:, :512] @ (evec[:, :512].T @ u)
    u1 /= np.linalg.norm(u1)

    def parts(X: np.ndarray, base: np.ndarray) -> np.ndarray:
        e = (X * X).sum(1)
        n40 = ((X @ B40) ** 2).sum(1)
        n512 = ((X @ B512) ** 2).sum(1)
        un = (X @ u1) ** 2
        xh = base / np.linalg.norm(base, axis=1, keepdims=True)
        x1 = xh - (xh @ evec[:, :512]) @ evec[:, :512].T - (xh @ u1)[:, None] * u1
        x1 /= np.linalg.norm(x1, axis=1, keepdims=True)
        par = (X * x1).sum(1) ** 2
        rest = e - n40 - n512 - un - par
        return np.stack([n40, n512, un, par, rest], 1).sum(0) / e.sum()

    rows = {}
    rng = np.random.default_rng(0)
    R = rng.standard_normal((2000, 4096))
    xs, xe = c["x_base_dump"][fl], ce["x_base_eq"]
    rows["random direction"] = parts(R, xe[:2000])
    rows["final stream x itself ('=')"] = parts(xe, xe)
    for nm in [
        "L31.gate.c14",
        "L31.gate.c238",
        "L31.up.c534",
        "L31.up.c36",
        "L31.up.c731",
        "L31.down.c18",
    ]:
        rows[f"{SHORT[nm]} ('=')"] = parts(ce[f"{nm}__dx_eq"], xe)
        rows[f"{SHORT[nm]} (fineweb)"] = parts(c[f"{nm}__dx_dump"][fl], xs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    s = np.sqrt(np.maximum(ev, 0))
    ax[0].plot(np.arange(1, len(s) + 1), s, color=BLUE)
    ax[0].axvspan(1, 512, color="#cde2fb", lw=0)
    ax[0].axvline(40, color=MUTED, lw=1)
    ax[0].text(45, s.max() * 0.8, "k = 40", color=INK2, fontsize=8)
    ax[0].text(60, s.max() * 0.6, "lowest-gain eighth\n(512 directions)", color=INK2, fontsize=8)
    top_cos = abs(float(evec[:, -1] @ u))
    print(
        "cos(uniform-logit direction, top singular direction) =",
        round(top_cos, 3),
        "top/median gain",
        s[-1] / np.median(s),
    )
    ax[0].annotate(
        f"top direction ≈ uniform-logit\ndirection (|cos| = {top_cos:.2f})",
        (len(s), s[-1]),
        xytext=(-150, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("singular direction of W_U·diag(g) (ascending)")
    ax[0].set_ylabel("singular value (logit gain)")
    ax[0].set_title("no sharp null space, a soft one")
    names = list(rows)[::-1]
    cols = [BLUE, "#86b6ef", ORANGE, AQUA, "#e1e0d9"]
    labs = [
        "bottom-40 directions",
        "next 472 low-gain directions",
        "uniform-logit direction",
        "along the stream x",
        "everything else",
    ]
    left = np.zeros(len(names))
    for j in range(5):
        vals = np.array([rows[n][j] for n in names])
        ax[1].barh(
            range(len(names)),
            vals,
            left=left,
            color=cols[j],
            label=labs[j],
            height=0.7,
            edgecolor="#fcfcfb",
            linewidth=1,
        )
        left += vals
    ax[1].set_yticks(range(len(names)), names, fontsize=7)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("share of write energy |dx|²")
    ax[1].set_title("where the writes go")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=7)
    save(fig, out, "fig5_where.png")


def fig_split(out: Path) -> None:
    z = np.load(CL / "claims.npz")
    names = CANDIDATES + CONTROLS
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey=True)
    for r, tag in enumerate(("eq", "fw")):
        for i, nm in enumerate(names):
            p = f"{nm}__{tag}__"
            full = float(z[p + "full__kl_total"])
            for part, col in (("k512_S0", BLUE), ("k512_S1", ORANGE)):
                kl = float(z[p + part + "__kl_total"])
                r2 = 1 - float(z[p + part + "__var_resid"]) / max(
                    float(z[p + part + "__var_d"]), 1e-12
                )
                ax[r, 0].plot([kl / full], [y[i]], "o", color=col, ms=5)
                ax[r, 1].plot([r2], [y[i]], "o", color=col, ms=5)
            r2f = 1 - float(z[p + "full__var_resid"]) / max(float(z[p + "full__var_d"]), 1e-12)
            ax[r, 1].plot([r2f], [y[i]], "|", color=INK, ms=10)
        where = "addsub '='" if tag == "eq" else "fineweb (flagged)"
        ax[r, 0].set_title(f"{where}: KL(remove part) / KL(remove all)")
        ax[r, 1].set_title(f"{where}: temperature R² of each part")
        ax[r, 0].set_xscale("log")
        ax[r, 1].set_xlim(-0.05, 1)
        ax[r, 0].set_yticks(y, [SHORT[n] for n in names], fontsize=7)
    ax[0, 0].plot([], [], "o", color=BLUE, label="token-neutral part S0")
    ax[0, 0].plot([], [], "o", color=ORANGE, label="complement S1")
    ax[0, 1].plot([], [], "|", color=INK, ms=10, label="whole write")
    ax[0, 0].legend(loc="lower left", fontsize=7)
    ax[0, 1].legend(loc="lower left", fontsize=7)
    save(fig, out, "fig6_split.png")


def fig_profile(out: Path) -> None:
    """The NON-temperature residual of each component's effect (claims.residual): which tokens it moves."""
    z = np.load(CL / "residual.npz")
    names = CANDIDATES + CONTROLS
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for tag, col, off in (("eq", BLUE, 0.15), ("fw", ORANGE, -0.15)):
        for i, nm in enumerate(names):
            p = f"{nm}__{tag}__r_"
            ax[0].plot([float(z[p + "num_off"])], [y[i] + off], "o", color=col, ms=5)
            ax[1].plot([float(z[p + "corr_freq"])], [y[i] + off], "o", color=col, ms=5)
            ax[2].plot([float(z[p + "r2_numfreq"])], [y[i] + off], "o", color=col, ms=5)
    ax[0].set_yticks(y, [SHORT[n] for n in names], fontsize=7)
    for a in ax[:2]:
        a.axvline(0, color=BASE, lw=1)
    ax[0].set_xlabel("number tokens − other tokens (sd units)")
    ax[0].set_title("residual pushes numbers (sign varies)")
    ax[1].set_xlabel("corr with log unigram frequency")
    ax[1].set_title("…and frequent tokens, weakly")
    ax[2].set_xlabel("R² of [is-number, log-frequency]")
    ax[2].set_xlim(0, 1)
    ax[2].set_title("…but those explain little of it")
    ax[0].plot([], [], "o", color=BLUE, label="addsub '='")
    ax[0].plot([], [], "o", color=ORANGE, label="fineweb (flagged)")
    ax[0].legend(loc="lower right", fontsize=7)
    save(fig, out, "fig7_residual.png")


def neuron_scan() -> dict[str, np.ndarray]:
    d: dict[str, list[np.ndarray]] = {}
    for f in sorted(glob.glob(str(ENT / "neurons_*of5.npz"))):
        z = np.load(f)
        for k in z.files:
            d.setdefault(k, []).append(z[k])
    dd = {k: np.concatenate(v) for k, v in d.items()}
    o = np.argsort(dd["ids"])
    return {k: v[o] for k, v in dd.items()}


def fig_neurons(out: Path) -> None:
    s = neuron_scan()
    z = np.load(CL / "claims.npz")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, tag in enumerate(("fw", "eq")):
        TE, DE = s[f"{tag}__kl_total"], s[f"{tag}__kl_direct"]
        med = 1 - DE / np.maximum(TE, 1e-12)
        ax[j].scatter(TE, med, s=3, color=MUTED, alpha=0.4, lw=0)
        ax[j].scatter(TE[ENT6], med[ENT6], s=24, color=BLUE, edgecolor="#fcfcfb", lw=1)
        for n in ENT6:
            ax[j].annotate(
                str(n),
                (TE[n], med[n]),
                fontsize=7,
                color=INK2,
                xytext=(3, 2),
                textcoords="offset points",
            )
        ax[j].set_xscale("log")
        ax[j].set_xlim(1e-7, None)
        ax[j].set_ylim(-1, 1.05)
        ax[j].set_xlabel("total effect TE (KL of ablation)")
        ax[j].set_ylabel("1 − DE/TE (final-norm mediated)")
        ax[j].set_title("fineweb, mean ablation" if tag == "fw" else "addsub '=', zero ablation")
    nrm, null = z["neuron_norm"], z["neuron_null40"]
    ax[2].scatter(nrm, null, s=3, color=MUTED, alpha=0.4, lw=0)
    ax[2].scatter(nrm[ENT6], null[ENT6], s=24, color=BLUE, edgecolor="#fcfcfb", lw=1)
    ax[2].set_xlabel("output-weight norm ||W_down[:, n]||")
    ax[2].set_ylabel("share in bottom-40 directions")
    ax[2].set_title("entropy neurons: low norm, null-heavy")
    fig.suptitle(
        "All 14,336 layer-31 neurons of Llama-3.1-8B; the six causal entropy neurons in blue",
        fontsize=9,
        color=INK2,
    )
    save(fig, out, "fig8_entropy_neurons.png")


def fig_same(out: Path) -> None:
    c = np.load(ENT / "components_fineweb.npz")
    ce = np.load(ENT / "components_eq.npz")
    s = neuron_scan()
    w = Weights()
    Wd = w.get("model.layers.31.mlp.down_proj.weight")
    fl = c["flagged"][c["dump_mask"]]
    hm = s["h_mean_fw"]
    QE = np.linalg.qr(Wd[:, ENT6])[0]
    names = CANDIDATES + CONTROLS
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(names))[::-1]
    rng = np.random.default_rng(0)
    QR = [np.linalg.qr(Wd[:, rng.choice(Wd.shape[1], 6, replace=False)])[0] for _ in range(5)]
    for tag, H, off in (("fw", c["h_base_dump"][fl], -0.18), ("eq", ce["h_base_eq"], 0.18)):
        wE = (H[:, ENT6] - hm[ENT6]) @ Wd[:, ENT6].T
        col = BLUE if tag == "eq" else ORANGE
        for i, nm in enumerate(names):
            X = c[f"{nm}__dx_dump"][fl] if tag == "fw" else ce[f"{nm}__dx_eq"]
            cos = (X * wE).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(wE, axis=1) + 1e-12)
            q = np.percentile(cos, [25, 50, 75])
            ax[0].plot([q[0], q[2]], [y[i] + off] * 2, color=col, lw=2)
            ax[0].plot([q[1]], [y[i] + off], "o", color=col, ms=4)
            e = (X * X).sum(1)
            ax[1].plot([(((X @ QE) ** 2).sum(1) / e).mean()], [y[i] + off], "o", color=col, ms=5)
            ax[1].plot(
                [np.mean([(((X @ Q) ** 2).sum(1) / e).mean() for Q in QR])],
                [y[i] + off],
                "x",
                color=MUTED,
                ms=5,
            )
    ax[0].axvline(0, color=BASE, lw=1)
    ax[0].set_yticks(y, [SHORT[n] for n in names], fontsize=7)
    ax[0].set_xlabel("cos(component write, entropy neurons' write)  median & IQR")
    ax[0].set_title("same direction on the same inputs")
    ax[1].set_xlabel("share of component write in the 6-dim entropy-neuron span")
    ax[1].set_title("…captured by six neurons' output weights")
    ax[0].plot([], [], "o-", color=BLUE, label="addsub '='")
    ax[0].plot([], [], "o-", color=ORANGE, label="fineweb (flagged)")
    ax[1].plot([], [], "x", color=MUTED, label="6 random neurons")
    ax[0].legend(loc="upper left", fontsize=7)
    ax[0].set_xlim(-0.85, 0.8)
    ax[1].legend(loc="lower right", fontsize=7)
    save(fig, out, "fig9_same_direction.png")


def fig_connect(out: Path) -> None:
    z = np.load(CL / "claims.npz")
    cls = {k: z[f"class_{k}"] for k in ("entropy_like", "number", "freq")}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for nm, col in (("L31.gate.c14", BLUE), ("L31.up.c534", ORANGE)):
        u_vec = z[f"{nm}__U"]
        nom = np.sort(u_vec**2 / (u_vec**2).sum())[::-1]
        s = z[f"{nm}__eq__share"]
        eff = np.sort(np.abs(s))[::-1] / np.abs(s).sum()
        x = np.arange(1, len(nom) + 1)
        ax[0].plot(x, np.cumsum(eff), color=col, label=f"{SHORT[nm]}: effective write")
        ax[0].plot(
            x, np.cumsum(nom), color=col, lw=1, alpha=0.45, label=f"{SHORT[nm]}: nominal |U|²"
        )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("number of neurons (sorted)")
    ax[0].set_ylabel("cumulative share")
    ax[0].set_title("few neurons do the work (addsub '=')")
    ax[0].legend(loc="lower right", fontsize=7)
    nm = "L31.gate.c14"
    u_vec = z[f"{nm}__U"]
    nom = u_vec**2 / (u_vec**2).sum()
    s = np.abs(z[f"{nm}__eq__share"]) + 1e-7
    ax[1].scatter(nom, s, s=3, color=MUTED, alpha=0.35, lw=0, label="other neurons")
    for k, col, lab in (
        ("number", ORANGE, "number neurons"),
        ("freq", AQUA, "frequency neurons"),
        ("entropy_like", BLUE, "entropy neurons"),
    ):
        mk = cls[k]
        ax[1].scatter(
            nom[mk], s[mk], s=16 if k == "entropy_like" else 8, color=col, lw=0, label=lab
        )
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1e-9, None)
    ax[1].set_xlabel("nominal share |U_c[n]|² / |U_c|²")
    ax[1].set_ylabel("|effective share of the write|")
    ax[1].set_title("gate c14: weight ≠ use")
    ax[1].legend(loc="lower right", fontsize=7, markerscale=1.5)
    sens = z["eq__state_abs_dsilu_g_u"]
    ax[2].scatter(sens * np.sqrt(nom), s, s=3, color=MUTED, alpha=0.35, lw=0)
    mk = cls["entropy_like"]
    ax[2].scatter((sens * np.sqrt(nom))[mk], s[mk], s=16, color=BLUE, lw=0)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlim(1e-7, None)
    ax[2].set_xlabel("|U_c[n]| × gate sensitivity |silu'(g_n) u_n| at '='")
    ax[2].set_ylabel("|effective share of the write|")
    ax[2].set_title("gate state decides which channels are live")
    save(fig, out, "fig10_connectivity.png")


def fig_classes(out: Path) -> None:
    z = np.load(CL / "claims.npz")
    classes = ["entropy_like", "number", "freq", "rest"]
    labs = [
        "entropy neurons (7)",
        "number neurons (144)",
        "frequency neurons (124)",
        "all other (14,061)",
    ]
    cols = [BLUE, ORANGE, AQUA, MUTED]
    comps = CANDIDATES
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(comps))[::-1]
    for tag, mk in (("eq", "o"),):
        for i, nm in enumerate(comps):
            p = f"{nm}__{tag}__"
            full = float(z[p + "full__kl_total"])
            for j, cn in enumerate(classes):
                q = p + f"class_{cn}"
                off = (j - 1.5) * 0.15
                kl = float(z[q + "__kl_total"]) / full
                r2 = 1 - float(z[q + "__var_resid"]) / max(float(z[q + "__var_d"]), 1e-12)
                ax[0].plot([kl], [y[i] + off], mk, color=cols[j], ms=5)
                ax[1].plot([r2], [y[i] + off], mk, color=cols[j], ms=5)
                ax[2].plot([float(z[q + "_prof__num_off"])], [y[i] + off], mk, color=cols[j], ms=5)
    ax[0].set_xscale("log")
    ax[0].set_yticks(y, [SHORT[n] for n in comps], fontsize=8)
    ax[0].set_xlabel("KL of the class-routed part / KL of the whole")
    ax[0].set_title("how much each class carries (addsub '=')")
    ax[1].set_xlim(-0.05, 1)
    ax[1].set_xlabel("temperature R² of that part")
    ax[1].set_title("which part is temperature")
    ax[2].axvline(0, color=BASE, lw=1)
    ax[2].set_xlabel("number-token offset of its direct logit change (sd)")
    ax[2].set_title("which part pushes numbers")
    for j in range(4):
        ax[2].plot([], [], "o", color=cols[j], label=labs[j])
    ax[2].legend(loc="lower right", fontsize=7)
    save(fig, out, "fig11_classes.png")


def fig_coupling(out: Path) -> None:
    from tokenizers import Tokenizer

    z = np.load(CL / "claims.npz")
    c = np.load(ENT / "components_fineweb.npz")
    tok = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    t = c["tokens"][:, 1:]
    dec = {int(i): tok.decode([int(i)]).strip() for i in np.unique(c["tokens"])}
    nxt = np.concatenate([c["tokens"][:, 2:], np.full((len(t), 1), -1)], 1)
    isnum = np.vectorize(
        lambda i: i >= 0 and dec.get(int(i), "").replace(",", "").replace(".", "").isdigit()
    )(nxt).ravel()
    feats = {}
    for k, lab in (
        ("ent6", "entropy neurons"),
        ("number", "number neurons"),
        ("freq", "frequency neurons"),
    ):
        act = z[f"acts_{k}"].astype(np.float32).reshape(-1, z[f"acts_{k}"].shape[-1])
        act = (act - act.mean(0)) / (act.std(0) + 1e-6)
        feats[lab] = act.mean(1)
    feats["c14 activation"] = c["L31.gate.c14__inner"][:, 1:].ravel()
    feats["next token is a number"] = isnum.astype(float)
    names = list(feats)
    M = np.corrcoef(np.stack([feats[n] for n in names]))
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{M[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=INK if abs(M[i, j]) < 0.6 else "#ffffff",
            )
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(
        "Pearson correlation over 32k fineweb positions\n(class = mean z-scored activation)"
    )
    save(fig, out, "fig12_coupling.png")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else None

    def want(n: str) -> bool:
        return which is None or n in which

    if want("1"):
        fig_activation(out)
    if want("2"):
        fig_dose(out)
    if want("3"):
        fig_addsub(out)
    if want("4"):
        fig_bins(out)
    if want("5"):
        fig_where(out, *unembed_basis())
    if want("6"):
        fig_split(out)
    if want("7"):
        fig_profile(out)
    if want("8"):
        fig_neurons(out)
    if want("9"):
        fig_same(out)
    if want("10"):
        fig_connect(out)
    if want("11"):
        fig_classes(out)
    if want("12"):
        fig_coupling(out)
