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
from track_build import OUT, load_blocks, rmsnorm

R = A % 10
NEURONS = {1653: "g9", 10984: "g14", 9765: "g11", 8597: "g10"}
BEGIN, END = "<!-- lens:begin -->", "<!-- lens:end -->"
ON, OFF = "#c0392b", "#2a5db0"


def dprime(on: np.ndarray, off: np.ndarray) -> float:
    return float(abs(on.mean() - off.mean()) / np.sqrt((on.var() + off.var()) / 2))


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
        density(ax[0], p[on], p[off], f"p(a) = μ*_{r0} · x(a)")
        ax[0].set_title(
            f"neuron {n} (r₀ = {r0}) — A: input along μ*_{r0}, d′ = {dprime(p[on], p[off]):.1f}",
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
        density(ax[3], o[on], o[off], "neuron output o = SiLU(h_g) · h_u")
        ax[3].set_title(f"D: neuron output, d′ = {dprime(o[on], o[off]):.1f}", fontsize=9)
        stats[n] = {"reader": reader, "share": share, "r0": r0, "p": [p[on].mean(), p[on].std(), p[off].mean(), p[off].std(), dprime(p[on], p[off])],
                    "hg": [g[on].mean(), g[on].std(), g[off].mean(), g[off].std(), dprime(g[on], g[off])], "hu": [u[on].mean(), u[on].std(), u[off].mean(), u[off].std()],
                    "silu": [sg[on].mean(), sg[on].std(), sg[off].mean(), sg[off].std()],
                    "o": [o[on].mean(), o[on].std(), o[off].mean(), o[off].std(), dprime(o[on], o[off])]}  # fmt: skip
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
<li><b>Result.</b> Along the dual vector, the input separates on from off with d′ = {min(dp):.1f}–{max(dp):.1f} (panel A). The gate
pre-activation already separates them better, d′ = {min(dg):.1f}–{max(dg):.1f}, because the gate readers do not read along
μ*<sub><i>r</i>₀</sub> alone (section 8). The neuron's output separates them with d′ = {min(do):.1f}–{max(do):.1f} (panel D). Each such neuron writes, through its writers, an almost noise-free one-hot
signal for its residue. Adding one per residue moves the residue centres apart while adding little spread.</li>
</ul>
<figure><img src="notebook/fig/mod10_lens.png"><figcaption>Four block-13 neurons, one per row (residues 7, 3, 5, 8). (A) Density of
<i>p</i>(<i>a</i>) = μ*<sub><i>r</i>₀</sub> · <i>x</i>(<i>a</i>), on (red) and off (blue). (B) Up arm: <i>h</i><sub>u</sub> against itself, on
the line <i>y</i> = <i>x</i>. (C) Gate arm: SiLU(<i>h</i><sub>g</sub>) against <i>h</i><sub>g</sub>, on the SiLU curve. (D) Density of the
neuron's output <i>o</i> = SiLU(<i>h</i><sub>g</sub>) <i>h</i><sub>u</sub>. Legends give mean ± standard deviation.</figcaption></figure>
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
