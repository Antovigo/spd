"""Notebook page "Geometry of the mod-10 representation across blocks 0-16" (entry id "mod10_geometry").

Figures come from ../repr_repo/plot_gram.py (copied into notebook/fig/ as geom_*.png) and, for section 5, from the
θ_means values of ../repr_repo/plot_angular.py and the hull areas of ../repr_repo/plot_hull.py (geom_extent.png); the numbers on the page are recomputed here from the same functions. Nothing is centred except in
the explicitly centred view (section 3), which says so.

Notation (also defined on the page):
  a          integer operand 1..400 at position 1 (the token a); r = a mod 10 in 0..9
  x_V(a)     block MLP input projected onto the span of the block's kept readers' read vectors (plot_gram.block_xv)
  mu_r       mean of x_V(a) over the 40 operands with a mod 10 = r; G[r, r'] = mu_r . mu_r'
  c, d_r     c = mean over r of mu_r, d_r = mu_r - c (an algebraic split of G, used only to read it)
  s_r        = c . d_r

    python geometry_page.py --out notebook
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "repr_repo"))
from plot_gram import BLOCKS, block_xv, cohen, residue_means  # noqa: E402
from repo import ROOT  # noqa: E402

RR = HERE.parent / "repr_repo"
FIGS = {
    "gram_mod10_p1.png": "geom_gram.png",
    "gram_cos_shared_mod10_p1.png": "geom_cos.png",
    "gram_centred_shared_mod10_p1.png": "geom_centred.png",
    "cohen_shared_log_mod10_p1.png": "geom_cohen.png",
}


def stats() -> list[dict]:
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    out = []
    eye = np.eye(10, dtype=bool)
    for b in BLOCKS:
        xv = block_xv(b, readers)
        mu = residue_means(xv)
        G = mu @ mu.T
        n = np.linalg.norm(mu, axis=1)
        cos = G / np.outer(n, n)
        c = mu.mean(0)
        d = mu - c
        s = d @ c
        D = d @ d.T
        nd = np.linalg.norm(d, axis=1)
        E = (np.abs(np.fft.fft(mu, axis=0)) ** 2).sum(1)
        coh = cohen(xv)
        out.append({
            "block": b, "n_readers": sum(r["block"] == b for r in readers),
            "c2": float(c @ c), "d2": float(np.mean(np.diag(D))),
            "diag_excess": float(np.mean(np.diag(G)) - G[~eye].mean()),
            "kilt": float(np.std(s[:, None] + s[None])),
            "min_cos": float(cos[~eye].min()), "max_cos": float(cos[~eye].max()),
            "k5": float(E[5] / E[1:].sum()), "cos_d": float((D / np.outer(nd, nd))[~eye].mean()),
            "d_median": float(np.median(coh[~eye])), "d_min": float(coh[~eye].min()),
        })  # fmt: skip
    return out


def page(st: list[dict], ang: dict, hull: dict) -> str:
    rows1 = "".join(
        f"<tr><td>{x['block']}</td><td>{x['n_readers']}</td><td>{x['c2']:.0f}</td><td>{x['d2']:.2f}</td>"
        f"<td>{x['diag_excess']:.2f}</td><td>{x['kilt']:.2f}</td><td>{x['min_cos']:.3f}</td><td>{x['k5']:.2f}</td>"
        f"<td>{x['cos_d']:+.3f}</td></tr>"
        for x in st
    )
    rows2 = "".join(
        f"<tr><td>{x['block']}</td><td>{x['d_median']:.1f}</td><td>{x['d_min']:.1f}</td></tr>"
        for x in st
    )
    rows3 = "".join(
        f"<tr><td>{b}</td><td>{ang['full'][i]['theta_means']:.1f}</td><td>{hull['full'][i]['cap_deg']:.2f}</td>"
        f"<td>{hull['full'][i]['log10_fraction']:.1f}</td><td>{ang['span'][i]['theta_means']:.1f}</td>"
        f"<td>{hull['span'][i]['cap_deg']:.2f}</td><td>{hull['span'][i]['log10_fraction']:.1f}</td></tr>"
        for i, b in enumerate(BLOCKS)
    )
    gap = ((1, 31), (2, 16), (3, 6.7), (4, 2.3), (5, 0.6), (6, 0.13))
    rows4 = "".join(f"<tr><td>{d}</td><td>{p}%</td><td>{p * 0.8:.1f}</td></tr>" for d, p in gap)
    return f"""
<p><b>Summary.</b> This page follows the ten residue means of <i>a</i> mod 10 through the MLP inputs of blocks
0–16 at the token <i>a</i>, with five views: their Gram matrix, their cosines, their cosines after removing their
mean, the Cohen's d between residue clouds, and how much of the sphere the ten mean directions span. The ten means always
point in nearly the same direction (pairwise cosines ≥ 0.65 at block 0 and ≥ 0.82 from block 1 on). Block 0's Gram
matrix is circulant. A parity checkerboard grows through blocks 3–5 and peaks at block 5. In blocks 6–10 the
residue-specific part of the means shrinks below 1% of their energy and the residue clouds overlap (median Cohen's d
0.4–1.0). The diagonal of the Gram matrix then disappears, leaving a "kilt" of row and column stripes. From block 11
the residue-specific parts grow back and take the one-hot-like shape again (blocks 12–14), and the clouds become
separable (median d 9.6 and 14.5 at blocks 13 and 14). The ten mean directions close up on the sphere from block 0 to
blocks 8–11 (in the full MLP input, the spherical hull they span shrinks from the area of a 7.3° cap to that of a
1.3° cap) and open again over blocks 12–15 (3.7° at block 15).</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000, prompts <code>&lt;BOS&gt; a op b =</code> with integer operands <i>a</i>,
<i>b</i> ∈ {{1, …, 400}} and op ∈ {{+, −}}. The model is the alive-only model (every component of the ceiling filter
<code>addsub-08-last-pos</code> on, everything else and the weight delta off). Everything is at position 1, the token
<i>a</i>, where the stream depends on <i>a</i> alone. <b>Nothing is centred</b> except in section 3, which removes the
mean of the ten residue means on purpose and says so. Scripts: <code>notes/reader_writer/repr_repo/plot_gram.py</code>,
<code>plot_angular.py</code>, <code>plot_hull.py</code> and <code>notes/reader_writer/readers/geometry_page.py</code> (branch
experiment/arith_representations).</p>
<ul>
<li><b>Residue.</b> <i>r</i> = <i>a</i> mod 10 ∈ {{0, …, 9}}; each residue has 40 operands.</li>
<li><b>Block input.</b> <i>x<sub>V</sub></i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is block <i>l</i>'s MLP input (after the
post-attention RMSNorm) projected onto the span of the block's kept readers' read vectors: what those readers see.
The number of kept readers per block is in table 1. Section 5 also uses the full MLP input
<i>x</i>(<i>a</i>) ∈ ℝ<sup>4096</sup>, without the projection.</li>
<li><b>Residue means.</b> μ<sub><i>r</i></sub> ∈ ℝ<sup>4096</sup> is the mean of <i>x<sub>V</sub></i>(<i>a</i>) over the
operands with <i>a</i> mod 10 = <i>r</i>. <i>G</i>[<i>r</i>, <i>r</i>′] = μ<sub><i>r</i></sub> · μ<sub><i>r</i>′</sub> is
their 10 × 10 Gram matrix.</li>
<li><b>Circulant.</b> <i>G</i> is circulant when <i>G</i>[<i>r</i>, <i>r</i>′] depends only on the lag
(<i>r</i>′ − <i>r</i>) mod 10, which is what ten means equally spaced on circles give. "offdiag" in the figure
titles is the share of |<i>F G F</i><sup>H</sup>|² that lies off the diagonal within Fourier frequencies 1–5, with
<i>F</i> the unitary 10-point DFT over <i>r</i>. It is 0 for a circulant <i>G</i>, and "shuffled" is its median over
200 random relabellings of the residues.</li>
</ul>

<h3>1. Gram matrices</h3>
<figure><img src="notebook/fig/geom_gram.png" alt="Gram matrices of the ten residue means, blocks 0-16"><figcaption>Figure 1.
<i>G</i>[<i>r</i>, <i>r</i>′] for blocks 0–16 (row <i>r</i>, column <i>r</i>′), each panel on its own colour
range.</figcaption></figure>
<p>All entries are large and positive: the ten means point into the same region of activation space. To read the
patterns, split each mean into the part all ten share and the rest: <i>c</i> = mean over <i>r</i> of
μ<sub><i>r</i></sub>, and δ<sub><i>r</i></sub> = μ<sub><i>r</i></sub> − <i>c</i>. This is an algebraic split of
<i>G</i> only; the plots stay uncentred. Then</p>
<p><i>G</i>[<i>r</i>, <i>r</i>′] = ‖<i>c</i>‖² + <i>s<sub>r</sub></i> + <i>s<sub>r′</sub></i> +
δ<sub><i>r</i></sub> · δ<sub><i>r</i>′</sub>, with <i>s<sub>r</sub></i> = <i>c</i> · δ<sub><i>r</i></sub>.</p>
<ul>
<li>‖<i>c</i>‖² is a constant offset.</li>
<li><i>s<sub>r</sub></i> + <i>s<sub>r′</sub></i> makes stripes: if residue <i>r</i> is longer or shorter along
<i>c</i>, its whole row and column shift together. This is the "kilt" pattern.</li>
<li>δ<sub><i>r</i></sub> · δ<sub><i>r</i>′</sub> carries the diagonal, which stands out by about 10/9 of the mean
‖δ<sub><i>r</i></sub>‖².</li>
</ul>
<p>A diagonal is visible only when the residue-specific part ‖δ<sub><i>r</i></sub>‖² is large next to the spread
of <i>s<sub>r</sub></i> + <i>s<sub>r′</sub></i>, because each panel spends its colour range on the largest variation.
Table 1 gives the terms. "Diagonal excess" is the mean diagonal entry minus the mean off-diagonal entry, and
"stripe spread" is the standard deviation over (<i>r</i>, <i>r</i>′) of <i>s<sub>r</sub></i> + <i>s<sub>r′</sub></i>.
"Parity share" is the Fourier energy of the means at frequency 5 in <i>r</i> (the (−1)<sup><i>r</i></sup> pattern)
divided by their energy at frequencies 1–5. "cos δ" is the mean cosine between δ<sub><i>r</i></sub> and
δ<sub><i>r</i>′</sub> over the 45 pairs; ten equally spaced, one-hot-like directions give −1/9 ≈ −0.111.</p>
<table><tr><th>block</th><th>readers</th><th>‖<i>c</i>‖²</th><th>mean ‖δ<sub><i>r</i></sub>‖²</th><th>diagonal excess</th>
<th>stripe spread</th><th>min cos(μ<sub><i>r</i></sub>, μ<sub><i>r</i>′</sub>)</th><th>parity share</th><th>cos δ</th></tr>{rows1}</table>
<ul>
<li><b>Block 0</b> is circulant (offdiag 0.05 against 0.11 shuffled), with one-hot-like δ<sub><i>r</i></sub>
(cos δ −0.110).</li>
<li><b>Blocks 3–5.</b> The parity share rises from 0.16 at block 3 to 0.24 at block 4 and 0.47 at block 5, which
draws a checkerboard. The step into block 4 is the write of the block-3 down component <code>d189</code> (see the
mod 2 notebook); the further rise at block 5 comes from block 4 and is not traced here.</li>
<li><b>Blocks 6–10.</b> Mean ‖δ<sub><i>r</i></sub>‖² falls to 0.36–1.15 while the stripe spread is 1.8–2.7, so the
stripes take the colour range and the diagonal disappears. The δ<sub><i>r</i></sub> also lose the one-hot shape
(cos δ −0.06 to 0.00). Residue 0 is the shortest along <i>c</i> from block 3 on, and residue 1 is also short at
blocks 8, 10 and 11, hence the dark row and column at <i>r</i> = 0 (and 1).</li>
<li><b>Blocks 11–14.</b> Mean ‖δ<sub><i>r</i></sub>‖² grows from 2.0 to 3.4, 7.2 and 11.0, and cos δ returns to
−0.10 to −0.11: the diagonal comes back.</li>
<li><b>Blocks 15–16.</b> At block 15 the δ<sub><i>r</i></sub> keep their size (7.2) but not the shape (cos δ −0.06;
residue 1 dominates the diagonal). At block 16 they shrink to 2.6 and cos δ is −0.01.</li>
</ul>

<h3>2. Cosines between the residue means</h3>
<figure><img src="notebook/fig/geom_cos.png" alt="Cosines between the ten residue means, blocks 0-16, one colour scale"><figcaption>Figure 2.
cos(μ<sub><i>r</i></sub>, μ<sub><i>r</i>′</sub>) on one colour scale for all blocks; the diagonal (1 by definition) is
grey. The titles repeat the circulant score of <i>G</i>.</figcaption></figure>
<p>Normalising removes the length differences along <i>c</i>. Block 0 is the only block well below 0.9
(0.65–0.79). Blocks 1 and 6–11 are nearly uniform at 0.97–1.00, and blocks 13–15 spread out again (minimum 0.91, 0.87
and 0.82). Row and column <i>r</i> = 0 stay the lowest at most blocks even after normalising, so residue 0 differs from
the others in direction as well as in length.</p>

<h3>3. Cosines after removing the mean of the residue means</h3>
<figure><img src="notebook/fig/geom_centred.png" alt="Cosines between the centred residue means, blocks 0-16, one colour scale"><figcaption>Figure 3.
cos(δ<sub><i>r</i></sub>, δ<sub><i>r</i>′</sub>), δ<sub><i>r</i></sub> = μ<sub><i>r</i></sub> − <i>c</i>, on one colour
scale symmetric about 0; the diagonal is grey. <b>This view is centred on purpose</b>, to show the relative orientation
of the residues once their shared direction is removed.</figcaption></figure>
<ul>
<li><b>Block 0:</b> small negative values everywhere, close to the −1/9 of equally spaced directions.</li>
<li><b>Blocks 3–5 and 9:</b> a checkerboard, strongest at blocks 4–5 (about ±0.75). Residues of the same parity point
the same way.</li>
<li><b>Blocks 6–10:</b> residues 3, 4 and 6–9 form a mutually positive group; residues 0 and 5 point away from it.</li>
<li><b>Blocks 12–14:</b> positive next to the diagonal and negative far from it. Neighbouring residues <i>r</i> and
<i>r</i> ± 1 point the same way.</li>
</ul>

<h3>4. Separability of the residue clouds</h3>
<p>For two residues <i>r</i> and <i>r</i>′, project both clouds {{<i>x<sub>V</sub></i>(<i>a</i>): <i>a</i> mod 10 =
<i>r</i>}} and {{… = <i>r</i>′}} (40 operands each) onto the unit vector <i>u</i> from μ<sub><i>r</i>′</sub> to
μ<sub><i>r</i></sub>. Cohen's d is ‖μ<sub><i>r</i></sub> − μ<sub><i>r</i>′</sub>‖ divided by the pooled within-cloud
standard deviation of the projections. The activations are not centred; the standard deviation is a spread around
each cloud's own mean.</p>
<figure><img src="notebook/fig/geom_cohen.png" alt="Cohen's d between residue clouds, blocks 0-16, log colour scale"><figcaption>Figure 4.
Cohen's d between residue clouds <i>r</i> and <i>r</i>′ along the axis joining their centroids, on one logarithmic
colour scale for all blocks; the diagonal is grey. The titles give the median and the minimum over the 45 pairs.</figcaption></figure>
<p>A reference scale: for two Gaussian clouds of equal spread cut at the midpoint of their centroids, the
misclassified fraction is Φ(−<i>d</i>/2), with Φ the standard normal cumulative distribution function.</p>
<table><tr><th><i>d</i></th><th>misclassified</th><th>expected errors among 80 points</th></tr>{rows4}</table>
<p>Below about 2 the clouds overlap heavily; above about 4–5 they separate with few errors. The centroid axis is not
the optimal discriminant and the clouds are not Gaussian, so these numbers are a lower bound on separability.</p>
<table><tr><th>block</th><th>median <i>d</i></th><th>min <i>d</i></th></tr>{rows2}</table>
<p>Blocks 0, 3, 13 and 14 separate every pair (minimum <i>d</i> ≥ 4.8). Blocks 1, 6, 7, 8 and 10 have median
<i>d</i> 0.4–1.3: in the span their readers see, most residue pairs overlap.</p>

<h3>5. How much of the sphere the ten mean directions span</h3>
<p>Read from the origin, only the directions of the residue means matter here: <i>b<sub>r</sub></i> =
μ<sub><i>r</i></sub> / ‖μ<sub><i>r</i></sub>‖, <i>r</i> = 0..9. Two measures:</p>
<ul>
<li><b>θ<sub>means</sub></b>: the mean angle (degrees) over the 45 pairs (<i>b<sub>r</sub></i>, <i>b<sub>r′</sub></i>).</li>
<li><b>Hull area.</b> The ten directions span a 10-dimensional subspace. The smallest convex set on the sphere that
contains them is the spherical simplex they span: every positive combination Σ<sub><i>r</i></sub> <i>w<sub>r</sub></i>
<i>b<sub>r</sub></i> (<i>w<sub>r</sub></i> ≥ 0), rescaled to length 1. It is a piece of the 9-dimensional unit
sphere S<sup>9</sup> of that subspace. With <i>B</i> = [<i>b</i><sub>0</sub> … <i>b</i><sub>9</sub>] (4096 × 10) and
<i>C</i> = <i>B</i><sup>T</sup><i>B</i> the cosine matrix, its area is Ω = √det <i>C</i> · ∫ ‖<i>B</i>λ‖<sup>−10</sup>
dλ, the integral running over the weights λ ∈ ℝ<sup>10</sup>, λ ≥ 0, Σλ = 1 (Lebesgue measure on λ<sub>0</sub>..λ<sub>8</sub>).
This follows from the volume of the cone cut by the unit ball, which is Ω / 10. The integral is estimated by Monte
Carlo over 200,000 uniform draws of λ (relative error below 0.01%); the method returns exactly 2<sup>−10</sup> for ten
orthogonal directions. The area is reported as a fraction of the area of S<sup>9</sup> and as ρ, the half-angle of
the round cap on S<sup>9</sup> that has the same area. Area on S<sup>9</sup> grows roughly as ρ<sup>9</sup>, so a
2-fold change in ρ is about a 500-fold change in area.</li>
</ul>
<p>"Full" uses the residue means of the whole MLP input <i>x</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> (float64 rebuild of the
alive-only model); "span" uses those of <i>x<sub>V</sub></i>(<i>a</i>). Both measures see only the ten means, not
the spread of single operands around them.</p>
<figure><img src="notebook/fig/geom_extent.png" alt="Angular extent of the ten residue-mean directions across blocks"><figcaption>Figure 5.
Left: θ<sub>means</sub>. Middle: log<sub>10</sub> of the hull area as a fraction of S<sup>9</sup>. Right: ρ. Black:
full MLP input; blue: readers' span.</figcaption></figure>
<table><tr><th>block</th><th>θ<sub>means</sub> (full)</th><th>ρ (full)</th><th>log<sub>10</sub> fraction (full)</th>
<th>θ<sub>means</sub> (span)</th><th>ρ (span)</th><th>log<sub>10</sub> fraction (span)</th></tr>{rows3}</table>
<p>In the full MLP input the ten directions close up steadily from block 0 (θ<sub>means</sub> 41°, ρ 7.3°) to
blocks 8–11 (θ<sub>means</sub> 10–11°, ρ 1.27–1.33°), about 7 orders of magnitude in hull area, and open again
over blocks 12–15 (θ<sub>means</sub> 23°, ρ 3.7° at block 15). The narrowest blocks coincide with the stripes of
section 1 and the overlapping clouds of section 4, and the reopening with the diagonal coming back. In the readers'
span the same shape has deeper dips at blocks 6–10 (ρ 0.27–0.54°) and at block 16 (ρ 0.96°).</p>

<h3>Caveat: the readers' span changes from block to block</h3>
<p>Sections 1–4 work in <i>x<sub>V</sub></i>, the input projected onto the span of each block's kept readers. Blocks
6–10 have the fewest kept readers (33–53, against 61–153 elsewhere), so part of the shrinking of δ<sub><i>r</i></sub>
and of the overlap of the clouds there may come from a smaller span rather than from the residual stream. Section 5
shows the narrowing of the ten mean directions in the full MLP input as well, but sections 1–4 have not been repeated in the full input.</p>
"""


def extent_figure(ang: dict, hull: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bl = list(BLOCKS)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    panels = ((ang, "theta_means", "θ_means: mean angle between the\nten residue-mean directions (deg)"),
              (hull, "log10_fraction", "hull area: log10 (fraction of S⁹)"),
              (hull, "cap_deg", "ρ: half-angle of the S⁹ cap\nwith the hull's area (deg)"))  # fmt: skip
    for ax, (src, k, title) in zip(axs, panels, strict=True):
        for sp, c, name in (
            ("full", "#1d1d1b", "full MLP input"),
            ("span", "#2a5db0", "readers' span x_V"),
        ):
            ax.plot(bl, [r[k] for r in src[sp]], marker="o", ms=3, color=c, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("block")
        ax.set_xticks(bl)
        ax.tick_params(labelsize=7)
        if k != "log10_fraction":
            ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for src, dst in FIGS.items():
        shutil.copy(RR / src, args.out / "fig" / dst)
    ang = json.loads((RR / "angular_mod10_p1.json").read_text())
    hull = json.loads((RR / "hull_mod10_p1.json").read_text())
    extent_figure(ang, hull, args.out / "fig" / "geom_extent.png")
    html = page(stats(), ang, hull)
    p = args.out / "entries.js"
    s = p.read_text()
    entry = (
        '{\nid: "mod10_geometry",\ntitle: "Mod 10 representations: geometry across blocks 0-16",\n'
        'date: "2026-10-23",\nhtml: `' + html + "`\n},\n"
    )
    if 'id: "mod10_geometry"' in s:
        i = s.index('{\nid: "mod10_geometry"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)


if __name__ == "__main__":
    main()
