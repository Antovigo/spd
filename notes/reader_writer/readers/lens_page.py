"""Section 9 of the notebook page "Mod 10 representations": the SiLU lens of block 13's MLP (position 1, alive-only).

Notation (a = integer operand 1..400, r = a mod 10; nothing is centred):
  x(a)       block-13 MLP input (post-attention RMSNorm of the stream, R^4096)
  mu_r       mean of x over the 40 a with a mod 10 = r; mu*_r the dual vectors (rows of the pseudo-inverse of
             [mu_0 ... mu_9]), so mu*_r . mu_r' = 1 if r = r' and 0 otherwise
  p(a)       = mu*_r0 . x(a): the input along the representation of the neuron's residue r0
  h_g, h_u   the neuron's gate and up pre-activations; o = SiLU(h_g) h_u its output
  on / off   a mod 10 = r0 / a mod 10 != r0;  d' = |mean_on - mean_off| / sqrt((var_on + var_off) / 2)
Also reads track_mod10/sharpen.json (mlp_sharpen.py) for the block-13 linear / non-linear split. Writes
notebook/fig/mod10_lens.png and the section between <!-- lens:begin --> and <!-- lens:end --> of the mod10 entry.

    python lens_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import A
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold
from track_build import OUT, load_blocks, rmsnorm

R = A % 10
NEURONS = {1653: "g9", 10984: "g14", 9765: "g11", 8597: "g10"}
BEGIN, END = "<!-- lens:begin -->", "<!-- lens:end -->"
ON, OFF = "#c0392b", "#2a5db0"


def dprime(on: np.ndarray, off: np.ndarray) -> float:
    return float(abs(on.mean() - off.mean()) / np.sqrt((on.var() + off.var()) / 2))


def ridge_lda(Z: np.ndarray, on: np.ndarray, lam: float) -> np.ndarray:
    """w = (S_w + lam * tr(S_w)/d * I)^-1 (m_on - m_off), S_w the pooled within-class scatter, computed in sample
    space (Woodbury) so the 4096 x 4096 matrix is never formed."""
    m1, m0 = Z[on].mean(0), Z[~on].mean(0)
    C = np.concatenate([Z[on] - m1, Z[~on] - m0]) / np.sqrt(len(Z))
    d = C.shape[1]
    reg = lam * (C**2).sum() / d
    dm = m1 - m0
    K = C @ C.T + reg * np.eye(len(C))
    return (dm - C.T @ np.linalg.solve(K, C @ dm)) / reg


def heldout_lda(Z: np.ndarray, on: np.ndarray) -> dict:
    """Optimal linear discriminant of on vs off (ridge-regularised Fisher LDA; the ridge strength is chosen by an
    inner 4-fold cross-validation on the training prompts), evaluated by 5-fold cross-validation. Each held-out fold is
    projected on the direction w trained on the other folds, scaled so that the training gap between the on and off
    means is 1 and shifted so that the training off mean is 0 (one affine map per fold, fixed on the training prompts,
    so that the five folds share one axis; the raw w . Z off means are kept in off_mean_by_fold)."""
    lams = (1e-3, 1e-2, 1e-1, 1.0)
    score = np.zeros(len(Z))
    dps, aucs, offs, chosen = [], [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, on):
        inner = []
        for lam in lams:
            v = []
            for itr, ite in StratifiedKFold(4, shuffle=True, random_state=1).split(Z[tr], on[tr]):
                w = ridge_lda(Z[tr][itr], on[tr][itr], lam)
                sv = Z[tr][ite] @ w
                v.append(dprime(sv[on[tr][ite]], sv[~on[tr][ite]]))
            inner.append(np.mean(v))
        lam = lams[int(np.argmax(inner))]
        chosen.append(lam)
        w = ridge_lda(Z[tr], on[tr], lam)
        w = w / ((Z[tr][on[tr]] @ w).mean() - (Z[tr][~on[tr]] @ w).mean())
        off_tr = (Z[tr][~on[tr]] @ w).mean()
        sc = Z[te] @ w - off_tr  # this fold's training off mean at 0, training on mean at 1
        score[te] = sc
        a, b = sc[on[te]], sc[~on[te]]
        dps.append(dprime(a, b))
        aucs.append(float((a[:, None] > b[None, :]).mean()))
        offs.append(float(off_tr))
    return {"score": score, "dprime": float(np.mean(dps)), "auc": float(np.mean(aucs)), "off_mean_by_fold": offs,
            "ridge_chosen": chosen}  # fmt: skip


def density(ax, on: np.ndarray, off: np.ndarray, xlabel: str) -> None:  # noqa: ANN001
    lo, hi = min(on.min(), off.min()), max(on.max(), off.max())
    pad = 0.15 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)
    for v, c, lab in ((off, OFF, "a mod 10 ≠ r₀"), (on, ON, "a mod 10 = r₀")):
        ax.fill_between(grid, gaussian_kde(v)(grid), color=c, alpha=0.35, lw=0)
        ax.plot(
            grid,
            gaussian_kde(v)(grid),
            color=c,
            lw=1.2,
            label=f"{lab}: {v.mean():+.2f} ± {v.std():.2f}",
        )
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, loc="upper center")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    st = np.load(OUT / "stream.npz")
    blk = load_blocks()[0][13]
    X = st["mid_13"].astype(np.float64)
    x = rmsnorm(X, blk.g2)
    tg, tu = blk.readers(X)
    mlp_out = blk.mlp(X)
    hg, hu = tg @ blk.m["U_gate"], tu @ blk.m["U_up"]
    mu = np.stack([x[r == R].mean(0) for r in range(10)])
    dual = np.linalg.pinv(mu.T)  # (10, 4096): dual[r] . mu[r'] = delta
    gid = list(blk.index[blk.kind == "gate"])
    fig, axes = plt.subplots(len(NEURONS), 4, figsize=(17, 3.3 * len(NEURONS)))
    rows, stats = [], {}
    for i, (n, reader) in enumerate(NEURONS.items()):
        r0 = int(np.argmax([hg[r == R, n].mean() for r in range(10)]))
        on, off = r0 == R, r0 != R
        p = x @ dual[r0]
        g, u = hg[:, n], hu[:, n]
        sg = g / (1 + np.exp(-g))
        o = sg * u
        j = gid.index(int(reader[1:]))
        share = float(np.abs(tg[:, j] * blk.m["U_gate"][j, n]).mean() / np.abs(g).mean())
        ax = axes[i]
        before = heldout_lda(X, on)
        after = heldout_lda(X + mlp_out, on)
        density(
            ax[0],
            before["score"][on],
            before["score"][off],
            "held-out score w · X (stream entering the MLP)",
        )
        ax[0].set_title(
            f"{n} (r₀ = {r0}) A: before MLP, d′ = {before['dprime']:.1f}, AUC = {before['auc']:.3f}",
            fontsize=9,
        )
        lim = np.abs(u).max() * 1.15
        ax[1].plot([-lim, lim], [-lim, lim], color="#999", lw=1)
        ax[1].scatter(u[off], u[off], s=6, color=OFF, alpha=0.5, lw=0)
        ax[1].scatter(u[on], u[on], s=10, color=ON, alpha=0.8, lw=0)
        ax[1].set_xlabel("up pre-activation h_u")
        ax[1].set_ylabel("up arm output (= h_u)")
        ax[1].set_title(
            f"B: up arm, on {u[on].mean():+.2f} ± {u[on].std():.2f}, off {u[off].mean():+.2f} ± {u[off].std():.2f}",
            fontsize=9,
        )
        hh = np.linspace(min(g.min(), -3) - 0.2, max(g.max(), 2.5) + 0.2, 400)
        ax[2].plot(hh, hh / (1 + np.exp(-hh)), color="#999", lw=1)
        ax[2].axhline(0, color="#ddd", lw=0.8)
        ax[2].scatter(g[off], sg[off], s=6, color=OFF, alpha=0.5, lw=0)
        ax[2].scatter(g[on], sg[on], s=10, color=ON, alpha=0.8, lw=0)
        ax[2].set_xlabel("gate pre-activation h_g")
        ax[2].set_ylabel("SiLU(h_g)")
        ax[2].set_title(
            f"C: gate arm, h_g spread on {g[on].std():.2f} → SiLU {sg[on].std():.2f}; off {g[off].std():.2f} → {sg[off].std():.2f}",
            fontsize=9,
        )
        density(
            ax[3],
            after["score"][on],
            after["score"][off],
            "held-out score w · X′ (stream leaving the MLP)",
        )
        ax[3].set_title(
            f"D: after MLP, d′ = {after['dprime']:.1f}, AUC = {after['auc']:.3f}", fontsize=9
        )
        stats[n] = {"reader": reader, "share": share, "r0": r0, "p": [p[on].mean(), p[on].std(), p[off].mean(), p[off].std(), dprime(p[on], p[off])],
                    "hg": [g[on].mean(), g[on].std(), g[off].mean(), g[off].std(), dprime(g[on], g[off])], "hu": [u[on].mean(), u[on].std(), u[off].mean(), u[off].std()],
                    "silu": [sg[on].mean(), sg[on].std(), sg[off].mean(), sg[off].std()],
                    "o": [o[on].mean(), o[on].std(), o[off].mean(), o[off].std(), dprime(o[on], o[off])],
                    "lda_before": {k: v for k, v in before.items() if k != "score"},
                    "lda_after": {k: v for k, v in after.items() if k != "score"}}  # fmt: skip
        s = stats[n]
        rows.append(f"<tr><td>{n}</td><td>{reader} ({share:.2f})</td><td>{r0}</td>"
                    f"<td>{s['p'][0]:.2f} ± {s['p'][1]:.2f} / {s['p'][2]:.2f} ± {s['p'][3]:.2f} (d′ {s['p'][4]:.1f})</td>"
                    f"<td>{s['hu'][0]:+.2f} ± {s['hu'][1]:.2f} / {s['hu'][2]:+.2f} ± {s['hu'][3]:.2f}</td>"
                    f"<td>{s['hg'][0]:+.2f} ± {s['hg'][1]:.2f} / {s['hg'][2]:+.2f} ± {s['hg'][3]:.2f}</td>"
                    f"<td>{s['silu'][0]:+.2f} ± {s['silu'][1]:.2f} / {s['silu'][2]:+.2f} ± {s['silu'][3]:.2f}</td>"
                    f"<td>{s['o'][0]:+.2f} ± {s['o'][1]:.2f} / {s['o'][2]:+.2f} ± {s['o'][3]:.2f} (d′ {s['o'][4]:.1f})</td></tr>")  # fmt: skip
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "mod10_lens.png", dpi=100)
    plt.close(fig)
    (OUT / "lens.json").write_text(json.dumps(stats, indent=1, default=float))
    sh = json.loads((OUT / "sharpen.json").read_text())["13"]["variants"]
    dp = [stats[n]["p"][4] for n in NEURONS]
    do = [stats[n]["o"][4] for n in NEURONS]
    off_in = [stats[n]["hg"][3] for n in NEURONS]
    off_out = [stats[n]["silu"][3] for n in NEURONS]
    dg = [stats[n]["hg"][4] for n in NEURONS]
    lb = [stats[n]["lda_before"]["dprime"] for n in NEURONS]
    la = [stats[n]["lda_after"]["dprime"] for n in NEURONS]
    gain = [b / a for a, b in zip(lb, la, strict=True)]
    soff = [stats[n]["silu"][2] for n in NEURONS]
    html = f"""{BEGIN}
<h3>9. Block 13's MLP: a SiLU lens</h3>
<p>Block 13's MLP is the largest single sharpening step (section 8). Without it, the closest-pair separation of the clouds
leaving the block would be {sh["X (no MLP)"]["sep_min"]:.2f}. With only the part of its period-10 write that is linear in the
input's period-10 coordinates, it would be {sh["X + Lz + Y0 (no non-linear part)"]["sep_min"]:.2f}. With the full MLP it is
{sh["actual X+Y"]["sep_min"]:.2f} (<code>mlp_sharpen.py</code>). A purely linear path cannot do better, because it writes each
prompt's position with the same signal-to-noise ratio as it reads it. The non-linear part comes from neurons like the four below.
Script: <code>lens_page.py</code>.</p>
<ul>
<li><b>Notation.</b> <i>x</i>(<i>a</i>) is block 13's MLP input. μ<sub><i>r</i></sub> is its mean over the 40 values of <i>a</i> with <i>a</i> mod 10 = <i>r</i>,
and μ*<sub><i>r</i></sub> the dual vectors (μ*<sub><i>r</i></sub> · μ<sub><i>r′</i></sub> = 1 if <i>r</i> = <i>r′</i>, 0 otherwise).
<i>p</i>(<i>a</i>) = μ*<sub><i>r</i>₀</sub> · <i>x</i>(<i>a</i>) is the input along the representation of the neuron's residue <i>r</i>₀. A neuron <i>n</i> has
gate and up pre-activations <i>h</i><sub>g</sub> and <i>h</i><sub>u</sub> and output <i>o</i> = SiLU(<i>h</i><sub>g</sub>) <i>h</i><sub>u</sub>.
"On" means <i>a</i> mod 10 = <i>r</i>₀ (40 prompts) and "off" the other 360. d′ = |mean<sub>on</sub> − mean<sub>off</sub>| /
√((var<sub>on</sub> + var<sub>off</sub>)/2). Values below are mean ± standard deviation over the on / off prompts.</li>
</ul>
<div class="tw"><table><tr><th>neuron</th><th>main gate reader (share of |h<sub>g</sub>|)</th><th>r₀</th><th>p on / off</th>
<th>h<sub>u</sub> on / off</th><th>h<sub>g</sub> on / off</th><th>SiLU(h<sub>g</sub>) on / off</th><th>o on / off</th></tr>{"".join(rows)}</table></div>
<p><b>The mechanism, in SwiGLU terms.</b></p>
<ul>
<li><b>The up arm is a constant.</b> For each neuron, <i>h</i><sub>u</sub> is about the same on and off (panel B), so the neuron
acts as SiLU(<i>h</i><sub>g</sub>) times a fixed number.</li>
<li><b>The gate is far up on its residue and far down elsewhere.</b> On, <i>h</i><sub>g</sub> is +1.8 to +2.1, where SiLU is
nearly linear: SiLU(<i>h</i><sub>g</sub>) is +1.5 to +1.8, and the output is large, with the sign of <i>h</i><sub>u</sub>. Off,
<i>h</i><sub>g</sub> is around −1.4, near SiLU's minimum (<i>h</i> ≈ −1.28, SiLU ≈ −0.28), where the curve is flat (panel C).</li>
<li><b>The flat bottom absorbs the off-residue noise.</b> Off, the gate pre-activation spreads by {min(off_in):.2f}–{max(off_in):.2f},
but SiLU(<i>h</i><sub>g</sub>) spreads by only {min(off_out):.2f}–{max(off_out):.2f}. The nine other residues all land on nearly the
same small value. On, the spread passes through about unchanged. So the output is not near zero off: it is a small,
almost constant value, SiLU(<i>h</i><sub>g</sub>) ≈ {min(soff):.2f} to {max(soff):.2f} times <i>h</i><sub>u</sub>, with the opposite sign
to <i>h</i><sub>u</sub>.</li>
<li><b>What the neuron does to its own signal.</b> Along the dual vector μ*<sub><i>r</i>₀</sub>, the MLP input separates on from off with
d′ = {min(dp):.1f}–{max(dp):.1f}. The gate pre-activation separates them better, d′ = {min(dg):.1f}–{max(dg):.1f}, because the gate
readers do not read along μ*<sub><i>r</i>₀</sub> alone. The neuron's output separates them with d′ = {min(do):.1f}–{max(do):.1f}.</li>
<li><b>What the whole MLP does to the stream (panels A and D).</b> A neuron's output is one direction; the question is whether the stream
is better separated after the MLP. Panels A and D project the residual stream entering and leaving block 13's MLP
(<i>X</i> and <i>X′</i> = <i>X</i> + MLP(<i>X</i>)) on its optimal linear discriminant of on against off: a ridge-regularised Fisher
discriminant in the full 4096 dimensions, trained and scored by 5-fold cross-validation, with the ridge strength chosen inside
each training set. Before the MLP the pools are already perfectly linearly separable on held-out prompts (AUC 1.000, d′
{min(lb):.1f}–{max(lb):.1f}). After the MLP, d′ is {min(la):.1f}–{max(la):.1f}: {min(gain):.2f}–{max(gain):.2f} times larger
(AUC 1.000). So the MLP does not create the separation; it enlarges an existing one, modestly by the optimal linear
measure. What changes more is where the separation sits. Before, it needs a tuned direction in 4096 dimensions: along the dual vector the pools overlap (d′
{min(dp):.1f}–{max(dp):.1f}). Off-residue prompts are also tighter after the MLP: their spread on the held-out discriminant falls
from 0.04–0.07 to 0.02–0.03 of the on/off gap. After, each lensing neuron has written a near
noise-free one-hot signal along its own write direction, and the clouds in the period-10 subspace separate (closest pair 1.79 →
3.06, section 8).</li>
</ul>
<figure><img src="notebook/fig/mod10_lens.png"><figcaption>Four block-13 neurons, one per row (residues 7, 3, 5, 8). (A) Held-out score of
the residual stream entering the MLP on its optimal linear discriminant of on (red) against off (blue). (B) Up arm: <i>h</i><sub>u</sub>
against itself, on the line <i>y</i> = <i>x</i>. (C) Gate arm: SiLU(<i>h</i><sub>g</sub>) against <i>h</i><sub>g</sub>, on the SiLU curve.
(D) The same as (A) for the stream leaving the MLP. In (A) and (D) the discriminant of each of the five folds is scaled and
shifted so that its training off and on means sit at 0 and 1, and its held-out prompts are pooled. Titles give the mean
held-out d′ and the AUC over the folds; legends give mean ± standard deviation.</figcaption></figure>
{END}"""
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        anchor = "<!-- track:end -->"
        assert anchor in s
        s = s.replace(anchor, anchor + "\n" + html, 1)
    p.write_text(s)
    for n in NEURONS:
        print(
            n,
            {k: np.round(v, 2).tolist() if isinstance(v, list) else v for k, v in stats[n].items()},
        )


if __name__ == "__main__":
    main()
