"""Section 10 of the notebook page "Mod 10 representations": copying a and b to "=" (blocks 16 and 15).

Reads copy_mod10/{copy.json, copy_geo.npz, stream_*.npz} (copy_analyze.py, copy_build.py); writes
notebook/fig/copy_planes.png, notebook/fig/copy_parts.png and the section between <!-- copy:begin --> and
<!-- copy:end --> of the mod10 entry. Nothing is centred.

    python copy_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy_build import OUT

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
CM = plt.get_cmap("tab10")
BEGIN, END = "<!-- copy:begin -->", "<!-- copy:end -->"
HEAD = {"a": 21, "b": 13}
BLOCK = {"a": 16, "b": 15}
SRCPOS = {"a": 1, "b": 3}


def scatter(ax, q: np.ndarray, lab: np.ndarray, title: str) -> None:  # noqa: ANN001
    c0 = q[lab == 0].mean(0)
    ang = -np.arctan2(c0[1], c0[0])
    q = q @ np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]).T
    for r in range(10):
        ax.scatter(q[lab == r, 0], q[lab == r, 1], s=1, color=CM(r), alpha=0.3, lw=0)
        ax.text(*q[lab == r].mean(0), str(r), fontsize=8, ha="center", va="center", weight="bold")
    ax.plot([0], [0], "k+", ms=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = json.loads((OUT / "copy.json").read_text())
    g = np.load(OUT / "copy_geo.npz")
    labs = {w: (np.load(OUT / f"stream_{w}.npz")[w] % 10).astype(int) for w in ("a", "b")}

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for i, w in enumerate(("a", "b")):
        lab = labs[w]
        Ps, Po, Pn = (
            g[f"{w}_Ps"],
            g[f"{w}_Po"],
            g[f"{w}_Pn"],
        )  # columns: planes k = 1..5 concatenated (2,2,2,2,1)
        xs, Ocp, X4n = g[f"{w}_xs"], g[f"{w}_O"], g[f"{w}_X4n"]
        name = "a mod 10, block 16" if w == "a" else "b mod 10, block 15"
        scatter(axes[i, 0], xs @ Ps[:, 0:2], lab, f"{name}: source (position {SRCPOS[w]}), k = 1")
        scatter(axes[i, 1], Ocp @ Po[:, 0:2], lab, 'copy at "=" (from the source), k = 1')
        scatter(axes[i, 2], xs @ Ps[:, 4:6], lab, "source, k = 3")
        scatter(axes[i, 3], Ocp @ Po[:, 4:6], lab, "copy, k = 3")
        scatter(axes[i, 4], X4n @ Pn[:, 0:2], lab, '"=" stream leaving the block, k = 1')
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "copy_planes.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    for w, c in (("a", "#2a5db0"), ("b", "#c0392b")):
        e = np.asarray(res[w]["head_period10_energy"])
        ax[0].bar(
            np.arange(32) + (-0.2 if w == "a" else 0.2),
            e / e.sum(),
            0.4,
            color=c,
            label=f"{w} (block {BLOCK[w]})",
        )
        ax[3].bar(
            np.arange(6) + (-0.2 if w == "a" else 0.2),
            res[w]["transfer"]["gain"],
            0.4,
            color=c,
            label=w,
        )
    ax[0].set_xlabel("head")
    ax[0].set_title("(a) share of the copied period-10 energy by head")
    ax[0].legend(frameon=False, fontsize=7)
    ax[3].set_xlabel("frequency k (0: direction shared by all residues)")
    ax[3].set_title("(d) gain source → copy per frequency")
    ax[3].legend(frameon=False, fontsize=7)
    for j, w in ((1, "a"), (2, "b")):
        top = res[w]["vcomps"]["top90"]
        M = np.asarray([d["profile"] for d in top])
        M = M / np.abs(M).max(1, keepdims=True)
        ax[j].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax[j].set_yticks(range(len(top)))
        ax[j].set_yticklabels([f"v{d['id']} ({d['share']:.2f})" for d in top], fontsize=6)
        ax[j].set_xticks(range(10))
        ax[j].set_xlabel(f"{w} mod 10")
        ax[j].set_title(
            f"({'b' if j == 1 else 'c'}) value components copying {w}: residue profile", fontsize=9
        )
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "copy_parts.png", dpi=100)
    plt.close(fig)

    def f(x: float, d: int = 2) -> str:
        return f"{x:.{d}f}"

    rows = ""
    for w in ("a", "b"):
        r = res[w]
        ra = r["attention_weight_on_source"]
        e = np.asarray(r["head_period10_energy"])
        vt, ot = r["vcomps"]["top90"], r["ocomps"]["top90"]
        rows += (f"<tr><td>{w} mod 10</td><td>{BLOCK[w]}</td><td>{SRCPOS[w]}</td><td>{HEAD[w]} (KV {HEAD[w] // 4})</td>"
                 f"<td>{f(e[HEAD[w]] / e.sum())}</td><td>{f(ra['by_head'][HEAD[w]])}</td>"
                 f"<td>{r['source']['acc']:.3f} / {r['source']['ovr']}</td><td>{r['copy']['acc']:.3f} / {r['copy']['ovr']}</td>"
                 f"<td>{' / '.join(f(x) for x in r['transfer']['gain'])}</td><td>{f(r['copy']['offdiag'], 3)}</td>"
                 f"<td>{len(vt)} of {r['vcomps']['n_alive']}</td><td>{len(ot)} of {r['ocomps']['n_alive']}</td></tr>")  # fmt: skip
    ab = res["a_vs_b"]
    va = res["a"]["vcomps"]["top90"]
    vb = res["b"]["vcomps"]["top90"]
    v114 = next(d for d in va if d["id"] == 114)
    v115 = next(d for d in va if d["id"] == 115)
    html = f"""{BEGIN}
<h3>10. Copying a and b to "="</h3>
<p>In the decomposition (the AB grids, with CI masks), the value components of block 15 that are active at the token
<i>b</i> (position 3) are periodic in <i>b</i>, and so are block 15's output components at "=" (position 4). The value components of block 16
that are active at the token <i>a</i> (position 1) are periodic in <i>a</i>, and so are block 16's output components at "=". So block 15 copies <i>b</i> mod 10
and block 16 copies <i>a</i> mod 10. Block 18 makes a smaller second copy of <i>a</i>. This section analyses the two main copies in the alive-only model
(<code>copy_build.py</code>, <code>copy_analyze.py</code>, <code>copy_page.py</code>). Each uses 8,000 prompts. For the copy of <i>a</i>: every
<i>a</i> = 1 … 400, ten values of <i>b</i> (one per residue) and both operations. For <i>b</i>: ten values of <i>a</i>, every <i>b</i> and both operations.
The rebuild matches the stored MLP activations to within 1.4% at every block.</p>
<ul>
<li><b>Notation.</b> <i>X<sub>j</sub></i> is the stream at position <i>j</i> entering the block, and <i>x<sub>j</sub></i> = RMSNorm(<i>X<sub>j</sub></i>) is the attention's
input. A value component <i>c</i> has inner activation <i>t<sub>c</sub></i>(<i>j</i>) = <i>x<sub>j</sub></i> · <i>V<sub>c</sub></i> and writes <i>t<sub>c</sub></i> <i>U<sub>c</sub></i>
into the eight key-value (KV) heads of 128 dimensions each. Query head <i>h</i> (0 … 31) reads KV head ⌊<i>h</i>/4⌋, with attention weight
<i>p<sub>h</sub></i> at "=" on the source position <i>s</i>. The <b>copy</b> <i>O<sub>s</sub></i> is the part of the attention output at "=" that comes from position
<i>s</i>: Σ<sub><i>h</i></sub> <i>p<sub>h</sub></i> × (head <i>h</i>'s value at <i>s</i>), passed through the output projection.
Residue means, Fourier components Ψ<sub><i>k</i></sub>, planes <i>P<sub>k</sub></i> and the 9-dimensional period-10 subspace <i>D</i> are as before,
with <i>r</i> the copied operand mod 10.</li>
<li><b>Gain per frequency.</b> <i>g<sub>k</sub></i> = √(λ<sub><i>k</i></sub>(copy)/λ<sub><i>k</i></sub>(source)), where λ<sub><i>k</i></sub> is the frequency-<i>k</i>
diagonal entry of the residue means' Gram matrix in the Fourier basis (<i>k</i> = 0 is the direction shared by all residues).
"offdiag" is the share of the frequency 1–5 Gram energy off that diagonal: 0 means the residues sit equally spaced on circles.
"Accuracy / separable" gives the leave-one-out nearest-mean accuracy, and the number of residues linearly separable from the
other nine, in the field's own <i>D</i>.</li>
</ul>
<div class="tw"><table><tr><th>copy</th><th>block</th><th>source position</th><th>main head</th><th>share of copied period-10 energy</th>
<th>its weight on the source</th><th>source accuracy / separable</th><th>copy accuracy / separable</th><th>gain g<sub>k</sub>, k = 0 … 5</th>
<th>offdiag of copy</th><th>value comps for 90%</th><th>output comps for 90%</th></tr>{rows}</table></div>
<p><b>One head does each copy.</b> Head 21 carries 98% of the copied period-10 energy of <i>a</i>, with weight 0.90 on position 1. Head 13
carries 97% of <i>b</i>'s, with weight 0.89 on position 3. Averaged over all heads, "=" puts 0.11 of its weight on <i>a</i> in block 16 and
0.15 on <i>b</i> in block 15. The main head's weight on the source varies between prompts (<i>a</i>: 0.90 ± 0.14, minimum
0.25; <i>b</i>: 0.89 ± 0.23, minimum 0.09), and it scales the whole copied vector. Rescaling each prompt's residue mean reproduces
28% (<i>a</i>) and 47% (<i>b</i>) of the copy's within-class spread in its <i>D</i>, and the fitted scale correlates with the head weight
(correlation 0.83 for both). This is why the copied clouds stretch along lines through the activation zero in the figure below.</p>
<p><b>Value components spread over several KV heads; the copy uses one.</b> Most value components that carry the copy of <i>a</i> have
their largest write in KV head 5 (read by head 21; two have it in KV head 0), and no single KV head holds more than 17–40% of a
component's write energy. For <i>b</i>, every carrying component's largest write is in KV head 3 (read by head 13), with 32–51% of the energy. The rest goes to KV heads that other query heads read with little weight on the
source. Output components read several heads (5–10 heads for 90% of their read energy). The 24 (<i>a</i>) and 17 (<i>b</i>) output
components that carry 90% of the copy each read head 21 (<i>a</i>) or head 13 (<i>b</i>) most, with 66–84% of their read energy.</p>
<p><b>All ten residues are copied, but not with equal gain.</b> The copy of <i>a</i> keeps every residue apart (accuracy 1.000, 10 of 10
separable) and stays circulant (offdiag {f(res["a"]["copy"]["offdiag"], 3)}). Its frequencies are scaled unequally: <i>k</i> = 1, 2, 5 by 0.24–0.27, and
<i>k</i> = 3, 4 by 0.10–0.11, against 0.07 for the shared direction. So the copy is the same structure up to a linear map that rotates the planes
and shrinks the full-residue plane <i>k</i> = 3 and the mod-5 plane <i>k</i> = 4 about 2.5 times more than the others. It is not a single
rotation times a scale. The copy of <i>b</i> has the same pattern (<i>k</i> = 3, 4 scaled by 0.08–0.09, others by 0.25–0.29). It keeps less:
accuracy 0.974 and 5 of 10 residues separable, against 0.998 and 9 at the source.
The copy of <i>b</i> also carries <i>a</i> mod 10 (Fourier amplitude 2.69 at <i>k</i> = 1, against 4.98 for <i>b</i>), because position 3 holds <i>a</i> mod 10
(see the position-3 page). The copy of <i>a</i> carries little <i>b</i> (0.46 against 6.25).</p>
<p><b>The copy lands in new directions.</b> The planes of the copy at "=" are almost orthogonal to the source's (mean principal cosines
≤ 0.08). The stream at "=" after the block holds the copied planes (cosines 0.83–0.99 for <i>a</i>; 0.29–0.91 for <i>b</i>). Before the block,
"=" carried little of either operand (amplitude ≤ 0.78).</p>
<p><b>How a value component can be mod 2 or a three-residue bump.</b> A value component's activation at the source is a linear read
<i>V<sub>c</sub></i> · <i>x<sub>s</sub></i> of the circles. Its residue profile is therefore a sum of sinusoids, one per frequency, with any amplitude and phase.
v114 reads mostly the <i>k</i> = 5 line and the shared direction (frequency shares 0.52 at <i>k</i> = 0, 0.41 at <i>k</i> = 5): a mod-2 pattern on an offset.
v115 reads <i>k</i> = 1 and 2 (0.43, 0.22) plus the offset (0.29), which sums to a bump over a few consecutive residues. Together the
{len(va)} value components carrying 90% of the copy of <i>a</i> span all nine period-10 dimensions: their profiles' frequency 1–5 parts have rank 9,
with singular values {f(res["a"]["vcomps"]["top_profiles_sv"][0])} down to {f(res["a"]["vcomps"]["top_profiles_sv"][8])}. The same holds for <i>b</i>
({len(vb)} components, rank 9, smallest singular value {f(res["b"]["vcomps"]["top_profiles_sv"][8])}). No single value component carries the
representation. The set does.</p>
<p><b><i>a</i> and <i>b</i>: the same source structure, different destinations.</b> At their sources, the planes of <i>a</i> (position 1, entering block 16)
and of <i>b</i> (position 3, entering block 15) are nearly the same directions: principal cosines 0.88–0.96 per plane, 0.84–0.99 for the
nine directions of <i>D</i>. At "=", the two copies land in almost orthogonal places (cosines between the copies' <i>D</i> subspaces
{f(ab["cos_D_copy"][0])} down to {f(ab["cos_D_copy"][8])}). The value components that copy <i>a</i> and <i>b</i> are different components in different blocks, and their read
vectors are only partly aligned: for each component copying <i>a</i>, the largest |cosine| with a component copying <i>b</i> is
{f(min(ab["vcomp_read_cos_max_per_a"]))}–{f(max(ab["vcomp_read_cos_max_per_a"]))}.</p>
<figure><img src="notebook/fig/copy_planes.png"><figcaption>Top: <i>a</i> mod 10, block 16. Bottom: <i>b</i> mod 10, block 15. From left: the source
position's attention input in its own planes <i>k</i> = 1 and <i>k</i> = 3; the copy <i>O<sub>s</sub></i> at "=" in its own planes <i>k</i> = 1 and 3;
the stream at "=" leaving the block in its plane <i>k</i> = 1. Axes are turned so that residue 0's mean is on the positive horizontal axis,
and the cross is the activation zero. Dots are the 8,000 prompts, coloured by residue.</figcaption></figure>
<figure><img src="notebook/fig/copy_parts.png"><figcaption>(a) Share of each head in the copied period-10 energy (<i>a</i>: block 16;
<i>b</i>: block 15). (b, c) Residue profiles (mean inner activation per residue, scaled to the largest |value|) of the value
components carrying 90% of each copy, labelled with their share. (d) Gain per frequency from the source to the copy.</figcaption></figure>
{END}"""
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        anchor = "<!-- lens:end -->"
        assert anchor in s
        s = s.replace(anchor, anchor + "\n" + html, 1)
    p.write_text(s)
    print("v114", v114["freq_share"], "v115", v115["freq_share"])


if __name__ == "__main__":
    main()
