"""Section 11 of the notebook page "Mod 10 representations": attention patterns of the copying heads, the copying
components' vectors in the Fourier planes, and the geometry of the copies at "=" (figures from copy_figs.py).

Writes the section between <!-- copyfig:begin --> and <!-- copyfig:end -->, after section 10 of the mod10 entry.

    python copyfig_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from copy_build import OUT

BEGIN, END = "<!-- copyfig:begin -->", "<!-- copyfig:end -->"


def rng(v: list[float], d: int = 2) -> str:
    return f"{min(v):.{d}f}–{max(v):.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = json.loads((OUT / "figs.json").read_text())
    src = {"a": 1, "b": 3}
    w = {}
    for k, s in src.items():
        pa, po = np.asarray(r[k]["pattern_alive"]), np.asarray(r[k]["pattern_original"])
        w[k] = {h: (po[h, s], pa[h, s], po[h, 0]) for h in r[k]["active_heads"]}
    other_a = [v for h, v in w["a"].items() if h != 21]
    other_b = [v for h, v in w["b"].items() if h != 13]
    ba, bb = (
        r["a"]["eq_before_over_after_in_eq_planes"],
        r["b"]["eq_before_over_after_in_eq_planes"],
    )
    html = f"""{BEGIN}
<h3>11. The copying heads' patterns, the copying vectors, and the copied geometry</h3>
<p>This section draws the attention patterns at "=" in the original model and in the alive-only model. It also plots
the copying components' vectors in the Fourier planes, and the structure of the copied representation at "=". It uses
2,000 prompts of each set of section 10: every 4th prompt of the <i>a</i> set, and a seeded random draw from the <i>b</i> set
(181–219 prompts per <i>b</i> residue). For the original model, positions 0–4 were recomputed through block 16, in float32
from the bf16 weights (<code>copy_figs.py</code>, <code>copyfig_page.py</code>).</p>
<ul>
<li><b>Notation.</b> <i>p<sub>h</sub></i>(<i>j</i>) is the attention weight of query head <i>h</i> (0 … 31) at "=" on position <i>j</i>
(0 … 4: &lt;BOS&gt;, <i>a</i>, op, <i>b</i>, =). A head is shown when its weight on the source is ≥ 0.15 in either model.
<i>x<sub>s</sub></i> is the attention input at the source position. <i>E</i> is the stream at "=" entering the block, and
<i>E′</i> = <i>E</i> + the attention output is the stream right after the attention write. <i>P<sub>k</sub></i> are the Fourier planes of
a field's residue means, for <i>k</i> = 1 … 5. <i>V<sub>c</sub></i> is the read vector of value component <i>c</i>, in <i>x<sub>s</sub></i> space,
and <i>U<sub>o</sub></i> is the write vector of output component <i>o</i>, in the residual stream.</li>
<li><b>Pairwise measures.</b> cos(<i>r</i>, <i>r′</i>) is the cosine between the residue means μ<sub><i>r</i></sub> and μ<sub><i>r′</i></sub> of <i>E′</i>, in
the full stream and after projection onto <i>E′</i>'s period-10 subspace <i>D</i>. Cohen's d(<i>r</i>, <i>r′</i>) is the distance between the
two residue means divided by the pooled standard deviation of the two clouds projected on the unit vector between the means.
This is the only quantity here measured around class means.</li>
</ul>
<p><b>Attention patterns.</b> The main copying head is the same in both models and attends to the source in both, more
sharply in the alive-only model. Head 21 (block 16) puts {w["a"][21][0]:.2f} of its weight on <i>a</i> in the original model and
{w["a"][21][1]:.2f} in the alive-only model. Head 13 (block 15) puts {w["b"][13][0]:.2f} and {w["b"][13][1]:.2f} on <i>b</i>.
In the original model, the other heads put {rng([v[0] for v in other_a])} (block 16) and {rng([v[0] for v in other_b])} (block 15)
on the source, and {rng([v[2] for v in other_a + other_b])} on &lt;BOS&gt;. The alive-only model raises their weight on the source
to {rng([v[1] for v in other_a])} and {rng([v[1] for v in other_b])}. The largest increases are head 3 (block 16: 0.17 → 0.42; block 15:
0.31 → 0.63), head 8 (0.08 → 0.28) and head 31 (block 15: 0.20 → 0.36).</p>
<figure><img src="notebook/fig/copy_patterns.png"><figcaption>Mean attention weight at "=" on each position, for the heads with weight ≥ 0.15 on the
source in either model. Left pair: block 16 (source: <i>a</i>, position 1), original and alive-only. Right pair: block 15 (source: <i>b</i>,
position 3). The red line marks the source.</figcaption></figure>
<p><b>The copying vectors in the Fourier planes.</b> The top row of each pair shows the read vectors <i>V<sub>c</sub></i> of the value components that
carry 90% of the copy (section 10), projected onto the source's planes. The bottom row shows the write vectors <i>U<sub>o</sub></i> of the output components
that carry 90% of the copy, projected onto the planes of <i>E′</i>. The arrows point in many directions within each plane: no single
component aligns with one residue. Each vector has only a small share of its norm² in any one plane (titles; 0.03–0.12 for
<i>V<sub>c</sub></i>, 0.02–0.22 for <i>U<sub>o</sub></i>).</p>
<figure><img src="notebook/fig/copy_vectors.png"><figcaption>Rows 1–2: the copy of <i>a</i> (block 16); rows 3–4: the copy of <i>b</i> (block 15). Rows 1 and
3: residue means of the source's attention input in its planes <i>k</i> = 1 … 5 (dots, labelled by residue), and the value read vectors
<i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖ (arrows). Rows 2 and 4: residue means of <i>E′</i> in its planes, and the output write vectors
<i>U<sub>o</sub></i> / ‖<i>U<sub>o</sub></i>‖ (arrows). Arrows are scaled by the factor in each title; the cross is the activation zero.</figcaption></figure>
<p><b>The "=" planes were empty before the write.</b> Project the residue means of <i>E</i>, before the write, onto the planes of <i>E′</i>.
For the copy of <i>a</i>, their frequency-<i>k</i> amplitude is {ba[0]:.3f}, {ba[1]:.3f}, {ba[2]:.3f}, {ba[3]:.3f} and {ba[4]:.2f} times
<i>E′</i>'s, for <i>k</i> = 1 … 5. The exception is the parity line: the stream at "=" already carries some <i>a</i> mod 2 there before block 16. For the copy of
<i>b</i>, it is {bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.2f}, {bb[3]:.2f} and {bb[4]:.3f}. Decoding the residue from <i>E</i> in <i>E′</i>'s period-10 subspace gives a
leave-one-out accuracy of {r["a"]["loo_acc_eq_before_in_after_D"]:.3f} (<i>a</i>) and {r["b"]["loo_acc_eq_before_in_after_D"]:.3f} (<i>b</i>), against
chance 0.1. After the write it gives {r["a"]["loo_acc_eq_after_in_after_D"]:.3f} and {r["b"]["loo_acc_eq_after_in_after_D"]:.3f}.</p>
<p><b>The structure after the copy.</b></p>
<ul>
<li><b>Full stream.</b> In <i>E′</i> the ten residue means point almost the same way. The pairwise cosines are
{rng(r["a"]["alive-only_cos_full_offdiag_range"], 3)} (<i>a</i>) and {rng(r["b"]["alive-only_cos_full_offdiag_range"], 3)} (<i>b</i>) in the
alive-only model, and {rng(r["a"]["original_cos_full_offdiag_range"], 3)} and {rng(r["b"]["original_cos_full_offdiag_range"], 3)} in the original.</li>
<li><b>Inside <i>D</i>, <i>a</i>.</b> The cosines show a parity checkerboard: even residues point the same way (up to about 0.8),
odd residues less so. The range is {rng(r["a"]["alive-only_cos_D_offdiag_range"])} alive-only and {rng(r["a"]["original_cos_D_offdiag_range"])} original.
This matches the copy's frequency gains (section 10): the parity line <i>k</i> = 5 is copied at 0.27, and the full-residue plane
<i>k</i> = 3 and the mod-5 plane <i>k</i> = 4 at only 0.10–0.11.</li>
<li><b>Inside <i>D</i>, <i>b</i>.</b> All ten means point the same way inside <i>D</i>: cosines {rng(r["b"]["alive-only_cos_D_offdiag_range"])} alive-only and
{rng(r["b"]["original_cos_D_offdiag_range"])} original.</li>
<li><b>Separation.</b> For <i>a</i>, Cohen's d has minimum {r["a"]["alive-only_cohen_min_median"][0]:.2f} and median
{r["a"]["alive-only_cohen_min_median"][1]:.2f} alive-only, and {r["a"]["original_cohen_min_median"][0]:.2f} and
{r["a"]["original_cohen_min_median"][1]:.2f} original. For <i>b</i> it is {r["b"]["alive-only_cohen_min_median"][0]:.2f} and
{r["b"]["alive-only_cohen_min_median"][1]:.2f}, against {r["b"]["original_cohen_min_median"][0]:.2f} and
{r["b"]["original_cohen_min_median"][1]:.2f}.</li>
</ul>
<figure><img src="notebook/fig/copy_geometry.png"><figcaption>The stream at "=" right after the attention write of block 16 (<i>a</i>, rows 1–2) and
block 15 (<i>b</i>, rows 3–4), alive-only and original. Left: cosines between the ten raw residue means. Middle: the same after projection onto
the stream's period-10 subspace <i>D</i>. Right: Cohen's d between residue clouds along the axis joining their means.</figcaption></figure>
{END}"""
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        anchor = "<!-- copy:end -->"
        assert anchor in s
        s = s.replace(anchor, anchor + "\n" + html, 1)
    p.write_text(s)


if __name__ == "__main__":
    main()
