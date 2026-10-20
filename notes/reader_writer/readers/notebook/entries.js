// Notebook pages for the MLP readers applet (index.html, "Notebook" tab) and the hosted notebook.
// Figures live in notebook/fig/ next to this file; links are relative to index.html.
window.NOTEBOOK = [
{
id: "mod10",
title: "Mod 10 representations",
date: "2026-10-20",
html: `
<p><b>Summary.</b> At the token <i>a</i>, the block-0 MLP input contains a 9-dimensional subspace <i>D</i> that encodes
<i>a</i> mod 10 as five nearly orthogonal Fourier components. These are four planes and one line, and the residues
sit on circles around the zero of the activation space. <i>D</i> is the token embedding's own mod-10 structure.
Eleven of the 153 kept block-0 readers are explained by <i>D</i> (r² 0.62–0.98). Ten of them respond to a single
residue, and every residue 0–9 has at least one such reader. Each reader's read vector lies almost entirely in
<i>D</i>, and in every plane it points at its residue. No reader reads the parity line or the mod-5 planes on its
own with any strength.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {1, …, 400} and op ∈ {+, −}. The model is the alive-only model: every component of the
ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off.
<b>Nothing on this page is centred or detrended.</b> The network has no biases, so zero is always the zero of the
activation space. Scripts: <code>notes/reader_writer/readers/mod10_page.py</code> and
<code>notes/reader_writer/repr_repo/seed_l0_mod10.py</code> (branch experiment/arith_representations).</p>
<ul>
<li><b>Position and residue.</b> Everything is measured at position 1, the token <i>a</i>. Causal attention means the
block-0 MLP input there depends on <i>a</i> alone. <i>r</i> = <i>a</i> mod 10 ∈ {0, …, 9}.</li>
<li><b>Readers.</b> A reader <i>c</i> is a block-0 gate component <code>g&lt;id&gt;</code> or up component
<code>u&lt;id&gt;</code> kept by the learned mask "kl_int, sampled mask, λ=5e-05" (153 readers).
<i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup> is its read vector.</li>
<li><b>Input.</b> <i>x<sub>V</sub></i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the block-0 MLP input (after the post-attention RMSNorm),
projected orthogonally onto span{<i>V<sub>c</sub></i>}. Every reader sees this projection exactly.</li>
<li><b>Inner activation.</b> <i>f<sub>c</sub></i>(<i>a</i>) = <i>x<sub>V</sub></i>(<i>a</i>) · <i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖.</li>
<li><b>Residue means.</b> μ<sub><i>r</i></sub> is the mean of <i>x<sub>V</sub></i>(<i>a</i>) over the 40 values of <i>a</i> with
<i>a</i> mod 10 = <i>r</i>, and <i>m<sub>c</sub></i>(<i>r</i>) is the mean of <i>f<sub>c</sub></i> over the same values.</li>
<li><b>Fourier components.</b> <i>i</i> is the imaginary unit and <i>k</i> ∈ {1, …, 5} an integer frequency.
Ψ<sub><i>k</i></sub> = Σ<sub><i>r</i></sub> μ<sub><i>r</i></sub> exp(−2π<i>i</i><i>kr</i>/10) ∈ ℂ<sup>4096</sup>.
<i>P<sub>k</sub></i> is the orthonormal basis of span{Re Ψ<sub><i>k</i></sub>, Im Ψ<sub><i>k</i></sub>}: a plane for
<i>k</i> = 1 … 4 and a line for <i>k</i> = 5. These are the representations <code>L0.p1.*.k1</code> … <code>.k5</code>
of the representation repository. <i>D</i> is the span of all five, 9 dimensions.</li>
<li><b>Projections and r².</b> Π<sub><i>X</i></sub> is the orthogonal projection onto a span <i>X</i>. The reader's
activation with its input restricted to <i>X</i> is
<i>g<sub>c</sub><sup>X</sup></i>(<i>a</i>) = Π<sub><i>X</i></sub><i>x<sub>V</sub></i>(<i>a</i>) · <i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖, and
r²<sub><i>X</i></sub>(<i>c</i>) = 1 − Σ<sub><i>a</i></sub>(<i>f<sub>c</sub></i> − <i>g<sub>c</sub><sup>X</sup></i>)² / Σ<sub><i>a</i></sub><i>f<sub>c</sub></i>²,
summed over <i>a</i> = 1 … 400.</li>
<li><b>Sign convention.</b> A component's sign is a gauge choice: (<i>V<sub>c</sub></i>, <i>U<sub>c</sub></i>) and
(−<i>V<sub>c</sub></i>, −<i>U<sub>c</sub></i>) give the same weights. For each mod-10 reader, <i>s<sub>c</sub></i> ∈ {+1, −1} is
the sign that makes its response at its main residue positive. Profiles and read directions below are shown
multiplied by <i>s<sub>c</sub></i>; only u54 has <i>s<sub>c</sub></i> = −1.</li>
</ul>

<h3>1. The representation: circles around zero</h3>
<p>The frequency of a plane determines what it can encode. Residue <i>r</i> sits at angle 2π<i>kr</i>/10 in the
plane of frequency <i>k</i>:</p>
<ul>
<li><i>k</i> = 1 and <i>k</i> = 3 each carry the full residue, <i>a</i> mod 10.</li>
<li><i>k</i> = 2 and <i>k</i> = 4 carry only <i>a</i> mod 5, so residues <i>r</i> and <i>r</i> + 5 coincide.</li>
<li><i>k</i> = 5 carries only <i>a</i> mod 2.</li>
</ul>
<figure><img src="notebook/fig/mod10_planes.png"><figcaption>Coordinates <i>P<sub>k</sub></i><sup>T</sup><i>x<sub>V</sub></i>(<i>a</i>) for
every <i>a</i> (small dots) and <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> (large dots, labelled by <i>r</i>). The
black cross is the zero of the activation space. For <i>k</i> = 5, the line coordinate is plotted against
<i>a</i>.</figcaption></figure>
<p>The circles are centred close to zero. The table gives ‖centre‖, the norm of the mean of the ten
<i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub>, and the radius, their mean distance from that centre:</p>
<table><tr><th>k</th><th>encodes</th><th>‖centre‖</th><th>radius</th></tr>
<tr><td>1</td><td>a mod 10</td><td>0.105</td><td>0.875</td></tr>
<tr><td>2</td><td>a mod 5</td><td>0.084</td><td>0.906</td></tr>
<tr><td>3</td><td>a mod 10</td><td>0.020</td><td>0.608</td></tr>
<tr><td>4</td><td>a mod 5</td><td>0.050</td><td>0.706</td></tr>
<tr><td>5</td><td>a mod 2</td><td>0.032</td><td>0.658</td></tr></table>
<p><i>D</i> holds 10.3% of ‖<i>x<sub>V</sub></i>(<i>a</i>)‖² on average over <i>a</i> (minimum 4.2%, maximum 18.9%).</p>
<p><b>The planes are nearly orthogonal.</b> The largest principal-angle cosine between two different planes is
0.27, between <i>k</i> = 2 and <i>k</i> = 4, both of which are mod 5. The next is 0.17 (<i>k</i> = 3, 4). Every other
pair is ≤ 0.08.</p>
<figure><img src="notebook/fig/mod10_planes_cos.png"><figcaption>Largest principal-angle cosine between span <i>P<sub>k</sub></i> and
span <i>P<sub>k′</sub></i>.</figcaption></figure>
<p><b>The residue directions.</b> Let <i>M</i> = [μ<sub>0</sub> … μ<sub>9</sub>]. Its first singular direction
carries 76% of ‖<i>M</i>‖². This is the part all ten residues share, and it lies outside <i>D</i>. As a result, the
raw residue means are far from orthogonal: cos(μ<sub><i>r</i></sub>, μ<sub><i>r′</i></sub>) is 0.65–0.79 for every pair
<i>r</i> ≠ <i>r′</i>. Inside <i>D</i>, cos(Π<sub><i>D</i></sub>μ<sub><i>r</i></sub>, Π<sub><i>D</i></sub>μ<sub><i>r′</i></sub>) ranges from −0.36 to
+0.09. Pairs whose residues differ by ±3 (mod 10) give −0.36 to −0.24 (mean −0.30), and all other pairs give −0.24
to +0.09. The 11 readers of section 2 need all of <i>M</i>: their mean r² over the top-<i>n</i> singular directions
reaches 0.96 only at <i>n</i> = 10, where <i>n</i> is an integer from 0 to 10.</p>
<figure><img src="notebook/fig/mod10_mu.png"><figcaption>From left: share of ‖<i>M</i>‖² per singular direction <i>j</i>;
mean r² of the 11 readers against <i>n</i>; raw cosines between residue means; cosines after projecting onto
<i>D</i>.</figcaption></figure>
<p><b>Origin: the token embedding.</b> Let <i>e</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> be Llama-3.1-8B's embedding row for the
single token of <i>a</i>, and γ ∈ ℝ<sup>4096</sup> the weight of block 0's post-attention RMSNorm. Define
<i>z</i>(<i>a</i>) = γ ⊙ <i>e</i>(<i>a</i>) / rms(<i>e</i>(<i>a</i>)), where ⊙ is the elementwise product and rms is the
root-mean-square over the 4096 coordinates. This is the MLP input the embedding alone would give. Let
ε<sub><i>r</i></sub> be its residue means, and <i>D</i><sub>ε</sub> the span of their five Fourier components, built
exactly like <i>D</i>. Then:</p>
<ul>
<li>The nine principal-angle cosines between <i>D</i> and the projection of <i>D</i><sub>ε</sub> onto
span{<i>V<sub>c</sub></i>} are ≥ 0.9998. Without that projection they are 0.89–0.94: on average 84% of
<i>D</i><sub>ε</sub> lies in span{<i>V<sub>c</sub></i>}.</li>
<li>For each of the 11 readers, the cosine over <i>r</i> between <i>V<sub>c</sub></i> · Π<sub><i>D</i></sub>μ<sub><i>r</i></sub> and
<i>V<sub>c</sub></i> · Π<sub><i>D</i>ε</sub>ε<sub><i>r</i></sub> is ≥ 0.9986. The ratio of their norms is 0.951–0.960.</li>
</ul>
<figure><img src="notebook/fig/mod10_embedding.png"><figcaption>Left: principal-angle cosines between <i>D</i> and the projected
<i>D</i><sub>ε</sub>. Right: per-reader cosine between the observed and the embedding-predicted residue
profiles.</figcaption></figure>

<h3>2. The readers it explains</h3>
<p>A reader counts as a mod-10 reader when r²<sub><i>D</i></sub>(<i>c</i>) ≥ 0.5. As a control, the largest r² any
reader gets from any of 20 random 9-dimensional subspaces of span{<i>V<sub>c</sub></i>} is 0.31.</p>
<figure><img src="notebook/fig/mod10_r2_sorted.png"><figcaption>r²<sub><i>D</i></sub> of the 153 readers, sorted (bars), and the
random-subspace control (dots).</figcaption></figure>
<p><b>Result: 11 readers.</b> In order of r²<sub><i>D</i></sub>: g59 0.98, g69 0.96, g58 0.96, u60 0.95, g47 0.92,
u54 0.91, g62 0.89, u52 0.88, g43 0.84, u794 0.74, u31 0.62. The next readers are g152 (0.40), u41 (0.25) and g175
(0.19).</p>
<p><b>Each reader picks one residue.</b> Outside its residue, each reader's profile is close to zero: the median of
<i>m<sub>c</sub></i>(<i>r</i>) over <i>r</i> is within ±0.02 of max |<i>m<sub>c</sub></i>|. The table lists each reader's
main residue and its next-largest values of <i>s<sub>c</sub></i> <i>m<sub>c</sub></i>(<i>r</i>), as fractions of
max<sub><i>r</i></sub> |<i>m<sub>c</sub></i>(<i>r</i>)|:</p>
<table><tr><th>reader</th><th>main residue</th><th>next</th></tr>
<tr><td>u794</td><td>0</td><td>all |·| ≤ 0.01</td></tr>
<tr><td>u31</td><td>0</td><td>5 (+0.26), 3 (−0.18)</td></tr>
<tr><td>g43</td><td>1</td><td>0 (+0.05), 8 (−0.04)</td></tr>
<tr><td>g69</td><td>2</td><td>5 (−0.61)</td></tr>
<tr><td>u60</td><td>3</td><td>6 (−0.29)</td></tr>
<tr><td>g62</td><td>4</td><td>all |·| ≤ 0.02</td></tr>
<tr><td>u52</td><td>5</td><td>3 (+0.04)</td></tr>
<tr><td>u54</td><td>6</td><td>all |·| ≤ 0.01</td></tr>
<tr><td>g58</td><td>7</td><td>4 (−0.19)</td></tr>
<tr><td>g59</td><td>8</td><td>5 (−0.42)</td></tr>
<tr><td>g47</td><td>9</td><td>6 (−0.32), 8 (+0.12)</td></tr></table>
<p>Every residue has a reader. Every second value of magnitude ≥ 0.18 is negative and at the main residue ± 3.
The one exception is u31, which has +0.26 at residue 5. This matches the −0.30 cosine between Π<sub><i>D</i></sub>μ<sub><i>r</i></sub> and Π<sub><i>D</i></sub>μ<sub><i>r</i>±3</sub> in section 1.</p>
<figure><img src="notebook/fig/mod10_profiles.png"><figcaption><i>s<sub>c</sub></i> <i>m<sub>c</sub></i>(<i>r</i>) / max<sub><i>r</i></sub>
|<i>m<sub>c</sub></i>(<i>r</i>)|, one row per reader.</figcaption></figure>
<figure><img src="notebook/fig/mod10_overlay.png"><figcaption><i>f<sub>c</sub></i>(<i>a</i>) for <i>a</i> = 1 … 100 (grey) and
<i>g<sub>c</sub><sup>D</sup></i>(<i>a</i>) (blue), in the component's own sign.</figcaption></figure>

<h3>3. How much each reader reads from each plane</h3>
<figure><img src="notebook/fig/mod10_per_plane.png"><figcaption>Left: ‖<i>P<sub>k</sub></i><sup>T</sup><i>V<sub>c</sub></i>‖² / ‖<i>V<sub>c</sub></i>‖², the
share of the read vector in each plane <i>k</i>. Right: r²<sub><i>P<sub>k</sub></i></sub>(<i>c</i>) for each plane alone (stacked
bars) and r²<sub><i>D</i></sub>(<i>c</i>) (black ticks). The planes are not exactly orthogonal, so the stacked values
overlap and can add up to more than 1. Readers are ordered by main residue.</figcaption></figure>
<ul>
<li><b>Read vectors sit in <i>D</i>.</b> For ten readers, 86–97% of ‖<i>V<sub>c</sub></i>‖² lies in <i>D</i>; for u31 it
is 38%.</li>
<li><b>All five frequencies, no single plane.</b> Every reader reads from all five. No plane alone gives any reader
more than r² = 0.39, and the two full-residue planes (<i>k</i> = 1, 3) together give 0.18–0.49.</li>
<li><b>Each read vector points at its residue.</b> After applying <i>s<sub>c</sub></i>, the cosine between
<i>s<sub>c</sub></i> <i>P<sub>k</sub></i><sup>T</sup><i>V<sub>c</sub></i> and <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> at the reader's main
residue is 0.79–1.00 in each plane <i>k</i> = 1 … 4, and the sign matches on the line <i>k</i> = 5, for all 11
readers.</li>
</ul>
<figure><img src="notebook/fig/mod10_planes_arrows.png"><figcaption>Residue means <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> (dots, labelled
by <i>r</i>) and the 11 read directions <i>s<sub>c</sub></i> <i>P<sub>k</sub></i><sup>T</sup><i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖ (arrows
from the origin, coloured by the reader's main residue). Each panel's arrow scale is given in its title.</figcaption></figure>
<figure><img src="notebook/fig/mod10_reader_vectors.png"><figcaption>Left: ‖Π<sub><i>D</i></sub><i>V<sub>c</sub></i>‖² / ‖<i>V<sub>c</sub></i>‖². Right: cosines
between the sign-aligned projected read vectors <i>s<sub>c</sub></i> Π<sub><i>D</i></sub><i>V<sub>c</sub></i>.</figcaption></figure>
<p><b>No mod-2 or mod-5 readers.</b> A reader responds to a plane when that plane alone gives r² ≥ 0.05; this is the
representation repository's threshold. Six readers respond only to the lower-period planes (<i>k</i> = 2, 4, 5), and
their total r² is 0.06–0.12: u122 (0.12, mod 5), g158 (0.08), u129 (0.08, parity), g175 (0.07), g87 (0.07) and
g75 (0.06).</p>

<h3>4. What D contributes to the other readers</h3>
<p>For each of the 142 other readers, the change is ‖<i>g<sub>c</sub><sup>D</sup></i>‖ / ‖<i>f<sub>c</sub></i>‖, with norms over
<i>a</i> = 1 … 400, and the residual is ρ<sub><i>c</i></sub> = <i>f<sub>c</sub></i> − <i>g<sub>c</sub><sup>D</sup></i>. The change has median
0.14, 90th percentile 0.26 and maximum 0.67 (g152). The next largest are u41 (0.59) and g175 (0.46).</p>
<figure><img src="notebook/fig/mod10_others_change.png"><figcaption>Change for the 142 other readers, sorted (the first 40 are
labelled).</figcaption></figure>
<figure><img src="notebook/fig/mod10_others_residual.png"><figcaption>The 24 other readers with the largest change, in order (left
column first). Grey: <i>f<sub>c</sub></i>. Blue: ρ<sub><i>c</i></sub>. The horizontal axis is <i>a</i>.</figcaption></figure>
`
},
];
