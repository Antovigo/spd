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
<i>D</i>, and in every plane it points at its residue. The read vectors match the dual basis of the residue means
(cosine 0.85–1.00) rather than the means themselves (0.46–0.60), mostly because they leave out the direction all
ten means share. No reader reads the parity line or the mod-5 planes on its
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
<h3>4. The readers read with the dual basis of the residue means</h3>
<p>The ten residue means μ<sub>0</sub> … μ<sub>9</sub> are linearly independent but far from orthogonal: they span
a 10-dimensional space <i>M</i>, which is <i>D</i> plus one shared direction. A one-hot reader therefore cannot use
μ<sub><i>r</i>₀</sub> itself as its read vector. The vector that responds 1 at <i>r</i>₀ and 0 at every other residue
is the dual vector μ*<sub><i>r</i>₀</sub>, defined as follows.</p>
<ul>
<li><b>Dual basis.</b> μ*<sub><i>r</i></sub> ∈ <i>M</i> are the ten vectors with μ*<sub><i>r</i></sub> · μ<sub><i>r′</i></sub> = 1 if
<i>r</i> = <i>r′</i> and 0 otherwise. They are the rows of the pseudo-inverse of [μ<sub>0</sub> … μ<sub>9</sub>].</li>
<li><b>Notation.</b> Π<sub><i>M</i></sub> is the orthogonal projection onto <i>M</i>, and <i>r</i>₀ is the reader's main residue.
<i>u</i> ∈ ℝ<sup>4096</sup> is the unit vector along the part of the mean of the ten μ<sub><i>r</i></sub> that lies
outside <i>D</i>; this is the shared direction.</li>
</ul>
<figure><img src="notebook/fig/mod10_dual.png"><figcaption>From left: cosines of each reader's
sign-aligned read vector Π<sub><i>M</i></sub><i>s<sub>c</sub></i><i>V<sub>c</sub></i> with every residue mean
μ<sub><i>r</i></sub>; the same with every dual vector μ*<sub><i>r</i></sub>; both cosines at the reader's own residue;
and each reader's largest side lobe, the most negative <i>s<sub>c</sub></i><i>m<sub>c</sub></i>(<i>r</i>) / max |<i>m<sub>c</sub></i>|
over <i>r</i> ≠ <i>r</i>₀, against its dual cosine.</figcaption></figure>
<ul>
<li><b>The read vectors are the dual vectors.</b> The cosine of Π<sub><i>M</i></sub><i>s<sub>c</sub></i><i>V<sub>c</sub></i> with
μ*<sub><i>r</i>₀</sub> is 0.85–1.00, and it is ≥ 0.98 for six readers: g43, g58, g62, u52, u54 and u794. The cosine
with μ<sub><i>r</i>₀</sub> is 0.46–0.60. Off the main residue, the cosines with the dual vectors are small, as the
second panel shows. Note that ‖Π<sub><i>M</i></sub><i>V<sub>c</sub></i>‖² / ‖<i>V<sub>c</sub></i>‖² is 0.87–0.97 for ten readers
and 0.38 for u31.</li>
<li><b>The side lobes are departures from the dual.</b> The six readers at cosine ≥ 0.98 have side lobes no larger
than 0.19 in magnitude. Of the other five, all but u31 (−0.18) have the larger lobes: g69 at cosine 0.85 has −0.61,
g59 at 0.93 has −0.42, g47 at 0.92 has −0.32, and u60 at 0.96 has −0.29.</li>
</ul>
<figure><img src="notebook/fig/mod10_dual_split.png"><figcaption>Left: cosine with the shared direction <i>u</i> of
<i>s<sub>c</sub></i><i>V<sub>c</sub></i>, of μ<sub><i>r</i>₀</sub> and of μ*<sub><i>r</i>₀</sub>. Right: inside <i>D</i>, the cosine of
Π<sub><i>D</i></sub><i>s<sub>c</sub></i><i>V<sub>c</sub></i> with Π<sub><i>D</i></sub>μ<sub><i>r</i>₀</sub> and with Π<sub><i>D</i></sub>μ*<sub><i>r</i>₀</sub>.</figcaption></figure>
<p><b>Most of the difference is the shared direction.</b></p>
<ul>
<li>On average, <i>u</i> carries 76% of ‖μ<sub><i>r</i></sub>‖² but only 0.4% of ‖μ*<sub><i>r</i></sub>‖². The cosine with
<i>u</i> is 0.84–0.89 for μ<sub><i>r</i>₀</sub>, 0.04–0.10 for μ*<sub><i>r</i>₀</sub>, and 0.02–0.10 for
<i>s<sub>c</sub></i><i>V<sub>c</sub></i>.</li>
<li>Inside <i>D</i>, the read vectors are close to both: the cosine is 0.92–0.98 with Π<sub><i>D</i></sub>μ<sub><i>r</i>₀</sub>
and 0.85–1.00 with Π<sub><i>D</i></sub>μ*<sub><i>r</i>₀</sub>. The dual is closer for six readers (g43, g58, g62, u52,
u54, u794), and the residue mean is closer for g47, g59, g69 and u31. u60 is equally close to both (0.96).</li>
</ul>

<p><b>No mod-2 or mod-5 readers.</b> A reader responds to a plane when that plane alone gives r² ≥ 0.05; this is the
representation repository's threshold. Six readers respond only to the lower-period planes (<i>k</i> = 2, 4, 5), and
their total r² is 0.06–0.12: u122 (0.12, mod 5), g158 (0.08), u129 (0.08, parity), g175 (0.07), g87 (0.07) and
g75 (0.06).</p>

<h3>5. What D contributes to the other readers</h3>
<p>For each of the 142 other readers, the change is ‖<i>g<sub>c</sub><sup>D</sup></i>‖ / ‖<i>f<sub>c</sub></i>‖, with norms over
<i>a</i> = 1 … 400, and the residual is ρ<sub><i>c</i></sub> = <i>f<sub>c</sub></i> − <i>g<sub>c</sub><sup>D</sup></i>. The change has median
0.14, 90th percentile 0.26 and maximum 0.67 (g152). The next largest are u41 (0.59) and g175 (0.46).</p>
<figure><img src="notebook/fig/mod10_others_change.png"><figcaption>Change for the 142 other readers, sorted (the first 40 are
labelled).</figcaption></figure>
<figure><img src="notebook/fig/mod10_others_residual.png"><figcaption>The 24 other readers with the largest change, in order (left
column first). Grey: <i>f<sub>c</sub></i>. Blue: ρ<sub><i>c</i></sub>. The horizontal axis is <i>a</i>.</figcaption></figure>

<h3>6. Which writers the mod-10 readers feed</h3>
<p>The writers are the block-0 <code>down_proj</code> components, written <code>d&lt;id&gt;</code>. Each writer <i>w</i> has an
inner activation <i>t<sub>w</sub></i> (its read vector dotted with the 14,336 neuron outputs) and a write vector
<i>U<sub>w</sub></i> ∈ ℝ<sup>4096</sup>. The connections come from the exact variance split of
<code>notes/reader_writer/plan_variance.md</code>, run on the model with only the λ=5e-05 mask's components on. Its
block-0 readers are exactly the 153 readers of this page. That split splits <i>t<sub>w</sub></i> exactly into one
term per reader, and the variance of <i>t<sub>w</sub></i> over the prompts into one share per term. Unlike the
rest of this page, these shares are variances, so they are taken around the mean over the prompts. The data are
stored in the representation repository, in the <code>writers</code> field of each reader's position 1.
u31 is left out of this section.</p>
<ul>
<li><b>φ<sub><i>w</i>,<i>c</i></sub></b> is the share of Var(<i>t<sub>w</sub></i>) credited to reader <i>c</i>. Summed over all readers it is 1.</li>
<li><b>Weight.</b> α<sub><i>w</i>,<i>c</i></sub> = φ<sub><i>w</i>,<i>c</i></sub> Var(<i>t<sub>w</sub></i>) ‖<i>U<sub>w</sub></i>‖² is the share times the variance
that the writer puts into the residual stream.</li>
<li><b>Reader fraction.</b> |α<sub><i>w</i>,<i>c</i></sub>| / Σ<sub><i>w′</i></sub> |α<sub><i>w′</i>,<i>c</i></sub>|, summed over the 115
writers <i>w′</i>, is how reader <i>c</i>'s weight divides among the writers. For each reader, the kept writers are
the largest, until their reader fractions add up to at least 0.9.</li>
</ul>
<p><b>Each reader feeds mainly the writer with its own id.</b> u794 is the exception: it feeds d41.</p>
<div class="tw"><table><tr><th>reader</th><th>r</th><th>main writer</th><th>reader fraction</th><th>φ</th><th>other writers ≥ 0.05</th></tr>
<tr><td>u794</td><td>0</td><td>d41</td><td>0.76</td><td>+0.54</td><td>–</td></tr>
<tr><td>g43</td><td>1</td><td>d43</td><td>0.87</td><td>+0.84</td><td>–</td></tr>
<tr><td>g69</td><td>2</td><td>d69</td><td>0.72</td><td>+0.97</td><td>d52 (0.18)</td></tr>
<tr><td>u60</td><td>3</td><td>d60</td><td>0.81</td><td>+0.90</td><td>–</td></tr>
<tr><td>g62</td><td>4</td><td>d62</td><td>0.85</td><td>+0.86</td><td>–</td></tr>
<tr><td>u52</td><td>5</td><td>d52</td><td>0.63</td><td>+0.37</td><td>d54 (0.08), d43 (0.05)</td></tr>
<tr><td>u54</td><td>6</td><td>d54</td><td>0.67</td><td>+0.62</td><td>d52 (0.05), d60 (0.05)</td></tr>
<tr><td>g58</td><td>7</td><td>d58</td><td>0.81</td><td>+0.90</td><td>–</td></tr>
<tr><td>g59</td><td>8</td><td>d59</td><td>0.81</td><td>+0.97</td><td>d52 (0.08)</td></tr>
<tr><td>g47</td><td>9</td><td>d47</td><td>0.85</td><td>+0.94</td><td>d54 (0.06)</td></tr></table></div>
<p>For the gate readers g43, g47, g58, g59 and g62, φ of the main writer is 0.84–0.97: the reader alone carries
almost all of that writer's variance. For the up readers u52 and u54 it is 0.37 and 0.62. d52 and d54 are also
kept writers of seven other readers each, with reader fractions 0.01–0.18.</p>
<figure><img src="notebook/fig/mod10_writers_grids.png"><figcaption>Inner activations on the prompts
<code>a + b =</code> at position 1 (the token <i>a</i>), in the alive-only model, for <i>a</i>, <i>b</i> = 1 … 400. In each
panel <i>b</i> runs horizontally and <i>a</i> vertically, starting from the bottom. Position 1 depends on <i>a</i> alone, so
every panel is horizontal stripes. One row per reader, ordered by main residue <i>r</i>. The first panel (blue frame) is the
reader's <i>s<sub>c</sub></i> <i>t<sub>c</sub></i>, where <i>t<sub>c</sub></i> = <i>x</i> · <i>V<sub>c</sub></i> is its raw inner
activation and <i>x</i> the block-0 MLP input. Next come its kept writers' <i>t<sub>w</sub></i>, in raw gauge, titled with
the reader fraction and φ. The colour scale is symmetric about zero in each panel (red positive, blue negative),
with the limit in the title.</figcaption></figure>

<h3>7. What the block-0 MLP does to the representation</h3>
<p>Only the MLP's component activations are stored, so block 0 of the alive-only model (attention and MLP, weight delta
off) was rebuilt in float64 from the checkpoint. It matches the stored gate and up activations to 0.4% and the writer
activations to 1%. Scripts: <code>mlp0_mod10.py</code>, <code>mlp0_linear.py</code>, <code>mlp0_linear_fig.py</code> (same folder).</p>
<ul>
<li><b>Fields.</b> <i>r</i><sub>mid</sub>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream entering the MLP, <i>m</i>(<i>a</i>) the MLP
output, and <i>r</i><sub>out</sub> = <i>r</i><sub>mid</sub> + <i>m</i>. The MLP input is <i>x</i> = γ ⊙ <i>r</i><sub>mid</sub> / ρ, where
γ ∈ ℝ<sup>4096</sup> is the post-attention RMSNorm weight and ρ(<i>a</i>) &gt; 0 the root-mean-square of <i>r</i><sub>mid</sub>(<i>a</i>).</li>
<li><b>Fourier components.</b> Ψ<sub><i>k</i></sub><sup><i>X</i></sup> is defined as in the setup, from the residue means of a field <i>X</i>.
<i>D<sub>x</sub></i> is the 9-dimensional span of the frequency 1–5 planes of <i>x</i>, and Π<sub><i>D<sub>x</sub></i></sub> the
orthogonal projection onto it.</li>
<li><b>Neurons.</b> <i>n</i> ∈ {0, …, 14335} indexes the neurons, and <i>U</i><sub>g,<i>c</i>,<i>n</i></sub> and <i>U</i><sub>u,<i>c</i>,<i>n</i></sub>
are gate or up reader <i>c</i>'s writes to neuron <i>n</i>. ω<sub><i>n</i></sub> ∈ ℝ<sup>4096</sup> is neuron <i>n</i>'s write vector,
summed over the alive writers.</li>
<li><b>C</b> is the set of the 10 period-10 readers of section 2, with u31 left out. In C, every residue has exactly one reader.</li>
</ul>
<p><b>Attention adds no period-10 structure.</b> At frequencies 1–5, block-0 attention's output has Fourier amplitudes
0.001–0.004, against 0.39–0.61 for the embedding. The representation entering the MLP is the embedding's.</p>
<p><b>The MLP does not erase it.</b> Write the MLP output's frequency-<i>k</i> component as β<sub><i>k</i></sub> Ψ<sub><i>k</i></sub><sup>mid</sup>
plus a part outside the input plane, with β<sub><i>k</i></sub> a complex number. Re β<sub><i>k</i></sub> is positive at every frequency:
0.68, 0.71, 0.09, 0.32 and 0.82 for <i>k</i> = 1 … 5. The MLP adds along the original direction. After the block, the component
inside the original plane is 1.09–1.82 times the input's. Most of the MLP's frequency-<i>k</i> output is written to new
directions: only 1–24% of ‖Ψ<sub><i>k</i></sub><sup><i>m</i></sup>‖² lies in the input's plane.</p>
<p><b>The SwiGLU in closed form.</b> With <i>t<sub>c</sub></i>(<i>a</i>) = <i>x</i>(<i>a</i>) · <i>V<sub>c</sub></i> for every reader <i>c</i>:</p>
<p style="padding-left:2em">
<i>h</i><sub>g,<i>n</i></sub> = Σ<sub><i>c</i></sub> <i>t<sub>c</sub></i> <i>U</i><sub>g,<i>c</i>,<i>n</i></sub>, &nbsp;
<i>h</i><sub>u,<i>n</i></sub> = Σ<sub><i>c</i></sub> <i>t<sub>c</sub></i> <i>U</i><sub>u,<i>c</i>,<i>n</i></sub>, &nbsp;
<i>o<sub>n</sub></i> = SiLU(<i>h</i><sub>g,<i>n</i></sub>) <i>h</i><sub>u,<i>n</i></sub>, &nbsp;
<i>m</i> = Σ<sub><i>n</i></sub> <i>o<sub>n</sub></i> ω<sub><i>n</i></sub>.</p>
<p>Let <i>m</i><sup>0</sup>(<i>a</i>) be the output with <i>t<sub>c</sub></i> set to 0 for every <i>c</i> in C (all other readers
unchanged), and <i>m</i><sup><i>c</i>→0</sup>(<i>a</i>) the output with only <i>t<sub>c</sub></i> set to 0. The <b>period-10 path</b> is
Δ<sub>C</sub>(<i>a</i>) = <i>m</i>(<i>a</i>) − <i>m</i><sup>0</sup>(<i>a</i>), and Δ<sub><i>c</i></sub>(<i>a</i>) = <i>m</i>(<i>a</i>) − <i>m</i><sup><i>c</i>→0</sup>(<i>a</i>).
Σ<sub><i>a</i></sub>‖Δ<sub>C</sub>‖² is 3.5% of Σ<sub><i>a</i></sub>‖<i>m</i>‖².</p>
<p><b>The gates are not in a linear regime.</b> The components are not neuron-aligned. Each spreads over about 5,700
neurons, which together hold 90% of ‖<i>U<sub>c</sub></i>‖². On the data, the gate pre-activations of the neurons they
reach span −2.07 to +1.56. On that range, SiLU is neither off nor linear (panel a), and even its quadratic
approximation <i>h</i>/2 + <i>h</i>²/4 misses it by up to 0.27.</p>
<p><b>The readers are one-hot, and that makes the path linear.</b> Each <i>t<sub>c</sub></i> puts 71–99.5% of
Σ<sub><i>a</i></sub> <i>t<sub>c</sub></i>² on its own residue. The lowest is g69 at 0.71, because of its lobe at <i>r</i> = 5.
Δ<sub>C</sub> is a function of the ten <i>t<sub>c</sub></i> that is zero when they are all zero. If each <i>t<sub>c</sub></i> took
only the values 0 and one fixed value, and only one reader were non-zero at a time, any such function would be exactly
Σ<sub><i>c</i></sub> κ<sub><i>c</i></sub> <i>t<sub>c</sub></i>, whatever the nonlinearity in between. Here κ<sub><i>c</i></sub> ∈ ℝ<sup>4096</sup> is
the secant, the least-squares slope of Δ<sub><i>c</i></sub>(<i>a</i>) on <i>t<sub>c</sub></i>(<i>a</i>). Hence</p>
<p style="padding-left:2em">Δ<sub>C</sub>(<i>a</i>) ≈ Σ<sub><i>c</i>∈C</sub> κ<sub><i>c</i></sub> <i>V<sub>c</sub></i><sup>T</sup> <i>x</i>(<i>a</i>) = <i>W</i> <i>x</i>(<i>a</i>),
&nbsp; <i>W</i> = Σ<sub><i>c</i>∈C</sub> κ<sub><i>c</i></sub> <i>V<sub>c</sub></i><sup>T</sup> (a 4096 × 4096 matrix of rank 10; singular values
0.019–0.177).</p>
<p>How well each step holds, as uncentred r² = 1 − Σ<sub><i>a</i></sub>‖<i>Y</i> − <i>Ŷ</i>‖² / Σ<sub><i>a</i></sub>‖<i>Y</i>‖²:</p>
<div class="tw"><table><tr><th>step</th><th>r²</th></tr>
<tr><td>each reader: Δ<sub><i>c</i></sub> ≈ κ<sub><i>c</i></sub> <i>t<sub>c</sub></i></td><td>0.85–0.97 (g69 0.85, u794 0.88)</td></tr>
<tr><td>additivity: Δ<sub>C</sub> ≈ Σ<sub><i>c</i></sub> Δ<sub><i>c</i></sub></td><td>0.985</td></tr>
<tr><td>one linear map: Δ<sub>C</sub> ≈ <i>W x</i></td><td>0.943</td></tr>
<tr><td>the same map reading only <i>D<sub>x</sub></i>: Δ<sub>C</sub> ≈ <i>W</i> Π<sub><i>D<sub>x</sub></i></sub><i>x</i></td><td>0.886</td></tr></table></div>
<p>71–84% of each ‖<i>V<sub>c</sub></i>‖² lies in <i>D<sub>x</sub></i>. Here <i>D<sub>x</sub></i> is measured in the full MLP input,
whereas section 3's 86–97% was measured inside span{<i>V<sub>c</sub></i>}. At the level of the period-10 content, <i>W x</i>
reproduces Δ<sub>C</sub>'s Fourier components with amplitude ratio 0.90–1.04 and overlap
Re⟨Ψ<sub><i>k</i></sub><sup>Δ</sup>, Ψ<sub><i>k</i></sub><sup><i>Wx</i></sup>⟩ / ‖Ψ<sub><i>k</i></sub><sup>Δ</sup>‖² of 0.89–1.04. The lowest values
are <i>k</i> = 3 (0.90) and <i>k</i> = 4 (0.94).</p>
<p><b>With the residual connection:</b> <i>r</i><sub>out</sub>(<i>a</i>) = <i>r</i><sub>mid</sub>(<i>a</i>) + <i>m</i><sup>0</sup>(<i>a</i>) +
<i>W</i> diag(γ) <i>r</i><sub>mid</sub>(<i>a</i>) / ρ(<i>a</i>) + ε(<i>a</i>). Here diag(γ) is the diagonal matrix with γ on its diagonal,
and ε(<i>a</i>) is the remainder (r² above). The path is linear in <i>r</i><sub>mid</sub> except for the RMSNorm's per-prompt
factor 1/ρ(<i>a</i>), where ρ ranges from 0.70 to 1.10 times its mean over <i>a</i>. So the MLP's period-10 path rewrites the
representation through a single rank-10 linear map. It reaches 94% of the path's energy, and the remainder comes from
imperfect one-hotness.</p>
<figure><img src="notebook/fig/mod10_mlp_linear.png"><figcaption>(a) SiLU over the range of gate pre-activations met on
the data (shaded), with ReLU and the tangent at zero for comparison. (b) For each reader of C: share of
Σ<sub><i>a</i></sub> <i>t<sub>c</sub></i>² on its own residue (bars) and r² of the secant fit Δ<sub><i>c</i></sub> ≈ κ<sub><i>c</i></sub> <i>t<sub>c</sub></i>
(dots). (c) Fourier amplitudes ‖Ψ<sub><i>k</i></sub>‖ of Δ<sub>C</sub> and of <i>W x</i>. (d) ‖Δ<sub>C</sub>(<i>a</i>)‖ and the
per-prompt error ‖Δ<sub>C</sub>(<i>a</i>) − <i>W x</i>(<i>a</i>)‖, for <i>a</i> = 1 … 400.</figcaption></figure>
<!-- track:begin -->
<h3>8. The representation through blocks 0–16</h3>
<p>Everything in this section is at position 1 (the token <i>a</i>), in the alive-only model. Position 1 attends
only to &lt;BOS&gt; and to itself, so the model was rebuilt in float64 from the checkpoint for those two
positions through block 16. Every block's alive attention and MLP components were on, and the weight delta was off.
The rebuild matches the stored MLP reader and writer activations of every block to within 1.8% (max relative
error). Scripts: <code>track_build.py</code>, <code>track_analyze.py</code>, <code>track_page.py</code>.</p>
<ul>
<li><b>Boundaries and sublayers.</b> <i>X</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream at one boundary: entering block
<i>b</i>, or after its attention. A sublayer <i>S</i> is one block's attention or MLP, including its RMSNorm. It adds
<i>S</i>(<i>X</i>) to the stream.</li>
<li><b>Fourier planes.</b> Ψ<sub><i>k</i></sub>, <i>P<sub>k</sub></i> and roundness are as in sections 1 and 7, computed from
<i>X</i>'s residue means. <i>D<sup>X</sup></i> is the 9-dimensional span of <i>X</i>'s five planes, Π<sub><i>D</i></sub> the orthogonal
projection onto it, and <i>z</i>(<i>a</i>) ∈ ℝ<sup>9</sup> the coordinates of <i>X</i>(<i>a</i>) in an orthonormal basis of
<i>D<sup>X</sup></i>.</li>
<li><b>What a sublayer does with the representation.</b> Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>)(<i>a</i>).
<i>S</i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>) is what the sublayer writes when its input has no period-10 structure. Its own
period-10 content is therefore made from other features of <i>a</i>.
<b>Share from D<sup>X</sup></b> is Re⟨Ψ<sub><i>k</i></sub>(<i>S</i>(<i>X</i>)), Ψ<sub><i>k</i></sub>(Δ)⟩ / ‖Ψ<sub><i>k</i></sub>(<i>S</i>(<i>X</i>))‖², for
<i>k</i> = 1…5. It is 1 when the sublayer's period-10 output is entirely made from <i>D<sup>X</sup></i>, and 0 when none of it is.</li>
<li><b>Linearity tests.</b> r²<sub>lin</sub> is the uncentred r² of Δ ≈ <i>L z</i>, with one 4096 × 9 matrix <i>L</i> fitted by least squares
with no intercept, over the 400 prompts. For an MLP, the secant test generalises section 7. For each reader <i>c</i>,
dt<sub><i>c</i></sub> = <i>t<sub>c</sub></i>(<i>X</i>) − <i>t<sub>c</sub></i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>). Δ<sub><i>c</i></sub> is the change in
the MLP's output when only reader <i>c</i>'s activation is changed that way, and κ<sub><i>c</i></sub> ∈ ℝ<sup>4096</sup> is the
least-squares secant of Δ<sub><i>c</i></sub> on dt<sub><i>c</i></sub>. "Secant r²" is the r² of Δ ≈ Σ<sub><i>c</i></sub> κ<sub><i>c</i></sub>
dt<sub><i>c</i></sub>, and "additivity r²" is the r² of Δ ≈ Σ<sub><i>c</i></sub> Δ<sub><i>c</i></sub>. For attention, the value
path keeps the attention pattern of <i>X</i> and takes the values from <i>X</i> − Π<sub><i>D</i></sub><i>X</i>.</li>
</ul>
<p><b>Attention does not act on the representation at position 1.</b> In every block, the attention output's
frequency-<i>k</i> amplitude is at most 0.06 times the input's. Averaged over heads and prompts, position 1 puts
93.4–98.2% of its attention weight on &lt;BOS&gt;.</p>
<p><b>Residue clouds.</b> At each boundary, <i>c<sub>r</sub></i> ∈ ℝ<sup>9</sup> is the mean of <i>z</i>(<i>a</i>) over the 40
values of <i>a</i> with <i>a</i> mod 10 = <i>r</i> (its norm is the residue's radius), and <i>s<sub>r</sub></i> is the root-mean-square
distance of those <i>z</i>(<i>a</i>) to <i>c<sub>r</sub></i> (the cloud's spread). The separation of two residues <i>r</i>, <i>r′</i> is
‖<i>c<sub>r</sub></i> − <i>c<sub>r′</sub></i>‖ / √((<i>s<sub>r</sub></i>² + <i>s<sub>r′</sub></i>²)/2). "Accuracy" is the
leave-one-out nearest-mean classification accuracy of <i>r</i> from <i>z</i>(<i>a</i>); chance is 0.1. Per residue, an MLP's
write is the mean of its output over the residue's prompts projected onto <i>D</i> of the stream leaving it, divided by
the residue's input radius ‖Π<sub><i>D</i></sub>μ<sub><i>r</i></sub>‖.</p>
<p><b>Summary of blocks 0–16.</b></p>
<ul>
<li><b>The MLPs' period-10 output comes from the representation itself.</b> In every block, 0.87–1.05 of each MLP's
frequency-<i>k</i> output comes from <i>D<sup>X</sup></i> (panel e). Little period-10 content is made from other features of
<i>a</i>.</li>
<li><b>Blocks 0–11 rewrite it linearly; blocks 12–16 do not.</b> In blocks 0–11, the MLP's action on
<i>D<sup>X</sup></i> is a sum of independent, linear reader paths (secant r² 0.91–0.99, additivity 0.94–1.00). In blocks
12–16, the secant r² falls to 0.51–0.77, and in blocks 12–14 additivity falls to 0.75–0.83: readers interact (panel f).</li>
<li><b>The clouds first blur, then sharpen.</b> From the embedding to the stream entering block 11, the clouds' spread
grows faster than their radius, and the closest-pair separation falls from 5.93 to 0.95 (accuracy 0.897). The MLPs of
blocks 11–14 (12–14 non-additive) tighten the clouds and push them apart: the closest pair reaches 4.12 and accuracy 1.000
(panels g–i). Blocks 15–16 widen the clouds again (closest pair 3.52).</li>
<li><b>The planes drift.</b> Each MLP turns the planes a little (cosines 0.62–1.00 per block). By the stream leaving
block 16, the mean principal cosine with the embedding's planes is 0.20–0.34 (panel c).</li>
<li><b>Some MLPs move a few residues more than others.</b> Residue 0 in blocks 2 and 3; residue 1 in blocks 7, 8, 11 and 16;
residues 1 and 9 in block 12; residues 3, 7 and 8 in block 13. In block 14, residues 1, 2, 4, 6 and 9 get more than twice
the write of 0, 3, 5, 7 and 8. Residue 0's cloud is the widest or second widest at every boundary (1.11–1.35 times the
mean spread), together with residue 1 up to block 14 and residue 5 after it.</li>
</ul>
<figure><img src="notebook/fig/mod10_track_clouds.png"><figcaption>Residue clouds in <i>D<sup>X</sup></i> at the residual
stream entering each block. (g) Closest-pair and median separation. (h) Mean radius ‖<i>c<sub>r</sub></i>‖ and mean spread
<i>s<sub>r</sub></i>. (i) Leave-one-out nearest-mean accuracy of the residue.</figcaption></figure>
<p><b>The blur and the sharpening, seen in one plane.</b> The figure below projects the stream onto its own
frequency-1 plane <i>P</i><sub>1</sub>, the plane that carries <i>a</i> mod 10 at one turn per ten residues. In that plane, the
closest-pair separation falls from 2.58 at the embedding to 0.19 entering block 11, and the median from 7.23 to 1.88.
Entering blocks 13–15 they are back to 1.00–1.86 and 5.71–6.96. From block 4 to block 12, the residue means' ring no longer
surrounds the activation zero. From block 6 to block 12 it is also squeezed into a flattened curve, and the clouds overlap. Entering blocks
13–16, they sit on a circle around zero again, with tight, separated clouds.</p>
<figure><img src="notebook/fig/mod10_track_k1.png"><figcaption>Residual stream at position 1, entering the block named
in each title (the last panel leaves block 16), projected onto that boundary's frequency-1 plane <i>P</i><sub>1</sub>. Within the
plane, the axes are turned so that residue 0's mean lies on the positive horizontal axis. Dots are the 400 values of
<i>a</i>, coloured by <i>r</i> = <i>a</i> mod 10, and labels are the residue means. The cross is the activation zero.
Each title gives the closest-pair and median separation in the plane. Script: <code>track_k1_fig.py</code>.</figcaption></figure>
<figure><img src="notebook/fig/mod10_track_planes.png"><figcaption>Residual stream at every boundary. (a) ‖Ψ<sub><i>k</i></sub>‖.
(b) Mean principal cosine between <i>P<sub>k</sub></i> and the same frequency's plane at the previous boundary. (c) The
same against the embedding's plane.</figcaption></figure>
<figure><img src="notebook/fig/mod10_track_sublayers.png"><figcaption>Per sublayer. (d) Period-10 output relative to
the input, per frequency (solid: MLP, dotted: attention). (e) Share of the MLP's period-10 output that comes from
<i>D<sup>X</sup></i>. (f) Linearity of Δ: the linear fit, the reader-secant reduction, and additivity over readers.</figcaption></figure>

<h4>Block 0</h4>
<p>The MLP is where the representation grows: frequency-k amplitude goes up 1.4–1.9 times, and the planes turn (mean cosine 0.62–0.78 with the embedding's). The step is linear (secant r² 0.93; section 7). Residue 0 is moved least: its write is 1.07 times its radius, against 1.74–2.07 for the other residues. The clouds widen faster than they move apart (spread 0.035 → 0.091, radius 0.157 → 0.381), so the closest pair (1, 2) drops from 5.93 to 4.83.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.01 / 0.00 / 0.00 / 0.01 / 0.00</td><td>0.91–1.06</td>
<td>0.007</td><td>0.51</td>
<td>value path r² 0.30; weight on itself 0.055</td></tr>
<tr><td>MLP</td><td>1.86 / 1.80 / 1.37 / 1.43 / 1.68</td><td>0.94–0.99</td>
<td>1.703</td><td>0.90</td>
<td>secant r² 0.93; additivity r² 0.94</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.70 / 0.72 / 0.62 / 0.67 / 0.78. Roundness (k = 1…4) entering the block:
0.98 / 0.91 / 0.91 / 0.72; leaving it: 0.93 / 0.91 / 0.86 / 0.57.
Largest attention heads in the value path: head 2 (0.85), head 1 (0.04), head 10 (0.04).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (3 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g47</td><td>0.15</td><td>9</td><td>0.78</td><td>0.91</td></tr><tr><td>g69</td><td>0.14</td><td>2</td><td>0.60</td><td>0.84</td></tr><tr><td>g43</td><td>0.12</td><td>1</td><td>0.82</td><td>0.93</td></tr><tr><td>g59</td><td>0.11</td><td>8</td><td>0.78</td><td>0.92</td></tr><tr><td>u60</td><td>0.10</td><td>3</td><td>0.82</td><td>0.94</td></tr><tr><td>g58</td><td>0.10</td><td>7</td><td>0.91</td><td>0.95</td></tr><tr><td>g62</td><td>0.08</td><td>4</td><td>0.86</td><td>0.95</td></tr><tr><td>u794</td><td>0.08</td><td>0</td><td>0.70</td><td>0.83</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>1.07</td><td>1.89</td><td>1.88</td><td>1.74</td><td>1.85</td><td>1.82</td><td>2.07</td><td>1.92</td><td>1.85</td><td>1.89</td></tr>
<tr><td>radius after / before</td><td>1.82</td><td>2.54</td><td>2.55</td><td>2.34</td><td>2.49</td><td>2.41</td><td>2.69</td><td>2.51</td><td>2.45</td><td>2.54</td></tr>
<tr><td>spread after / mean spread</td><td>1.32</td><td>1.07</td><td>0.97</td><td>0.94</td><td>0.95</td><td>1.13</td><td>0.89</td><td>0.84</td><td>0.94</td><td>0.95</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 5.93 (1, 2) →
4.83 (1, 2); median sep 6.60 → 5.49;
mean spread 0.035 → 0.091; mean radius 0.157 →
0.381; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b0.png"><figcaption>Block 0: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 1</h4>
<p>The MLP writes little period-10 content (gain 0.04–0.10), and what it writes is linear in the readers (secant r² 0.91, additivity 1.00). The linear fit on z alone is lower (0.54) because Δ here is small (0.13 of the input's D norm). The planes do not move (cosine ≥ 0.99). Spread grows, and the smallest separation falls 4.83 → 3.59.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.02 / 0.03 / 0.03 / 0.04 / 0.02</td><td>0.92–1.02</td>
<td>0.033</td><td>0.59</td>
<td>value path r² 0.11; weight on itself 0.042</td></tr>
<tr><td>MLP</td><td>0.10 / 0.09 / 0.08 / 0.09 / 0.04</td><td>0.92–1.05</td>
<td>0.132</td><td>0.54</td>
<td>secant r² 0.91; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.99 / 1.00 / 1.00 / 1.00 / 1.00. Roundness (k = 1…4) entering the block:
0.93 / 0.91 / 0.86 / 0.57; leaving it: 0.91 / 0.90 / 0.86 / 0.57.
Largest attention heads in the value path: head 26 (0.55), head 25 (0.16), head 18 (0.12).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (4 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g38</td><td>0.32</td><td>8</td><td>0.22</td><td>0.90</td></tr><tr><td>u223</td><td>0.17</td><td>8</td><td>0.17</td><td>0.86</td></tr><tr><td>u38</td><td>0.16</td><td>0</td><td>0.27</td><td>0.90</td></tr><tr><td>g24</td><td>0.12</td><td>0</td><td>0.56</td><td>0.91</td></tr><tr><td>u13</td><td>0.07</td><td>0</td><td>0.49</td><td>0.86</td></tr><tr><td>g116</td><td>0.04</td><td>4</td><td>0.24</td><td>0.93</td></tr><tr><td>u234</td><td>0.04</td><td>1</td><td>0.65</td><td>0.90</td></tr><tr><td>g42</td><td>0.02</td><td>6</td><td>0.17</td><td>0.93</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.24</td><td>0.20</td><td>0.23</td><td>0.24</td><td>0.24</td><td>0.23</td><td>0.21</td><td>0.22</td><td>0.24</td><td>0.22</td></tr>
<tr><td>radius after / before</td><td>0.89</td><td>0.99</td><td>1.02</td><td>1.15</td><td>1.26</td><td>1.22</td><td>1.16</td><td>1.12</td><td>1.12</td><td>1.17</td></tr>
<tr><td>spread after / mean spread</td><td>1.21</td><td>1.13</td><td>1.05</td><td>0.96</td><td>0.96</td><td>1.06</td><td>0.89</td><td>0.84</td><td>0.94</td><td>0.96</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 4.83 (1, 2) →
3.59 (1, 2); median sep 5.49 → 4.39;
mean spread 0.092 → 0.116; mean radius 0.380 →
0.423; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b1.png"><figcaption>Block 1: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 2</h4>
<p>Linear (secant r² 0.94). The MLP writes mostly to residues 0 and 1 (0.83 and 0.66 of their radius, against 0.19–0.30 for the others), and roundness drops (k = 2: 0.90 → 0.77; k = 4: 0.57 → 0.43).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.02 / 0.06 / 0.02 / 0.02 / 0.00</td><td>0.35–1.44</td>
<td>0.072</td><td>0.86</td>
<td>value path r² -0.16; weight on itself 0.018</td></tr>
<tr><td>MLP</td><td>0.32 / 0.45 / 0.42 / 0.64 / 0.29</td><td>0.92–0.98</td>
<td>0.512</td><td>0.93</td>
<td>secant r² 0.94; additivity r² 0.99</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.96 / 0.94 / 0.93 / 0.92 / 0.96. Roundness (k = 1…4) entering the block:
0.91 / 0.90 / 0.86 / 0.57; leaving it: 0.83 / 0.77 / 0.85 / 0.43.
Largest attention heads in the value path: head 23 (0.73), head 5 (0.10), head 6 (0.06).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>u70</td><td>0.79</td><td>3</td><td>0.12</td><td>0.96</td></tr><tr><td>u411</td><td>0.06</td><td>1</td><td>0.24</td><td>0.95</td></tr><tr><td>g255</td><td>0.06</td><td>1</td><td>0.37</td><td>0.93</td></tr><tr><td>u313</td><td>0.04</td><td>5</td><td>0.65</td><td>0.91</td></tr><tr><td>u275</td><td>0.03</td><td>8</td><td>0.24</td><td>0.97</td></tr><tr><td>g335</td><td>0.02</td><td>5</td><td>0.15</td><td>0.81</td></tr><tr><td>g229</td><td>0.01</td><td>6</td><td>0.26</td><td>0.63</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.83</td><td>0.66</td><td>0.30</td><td>0.25</td><td>0.20</td><td>0.22</td><td>0.21</td><td>0.23</td><td>0.19</td><td>0.21</td></tr>
<tr><td>radius after / before</td><td>1.44</td><td>1.41</td><td>1.13</td><td>1.15</td><td>1.06</td><td>1.09</td><td>1.05</td><td>1.09</td><td>1.03</td><td>1.05</td></tr>
<tr><td>spread after / mean spread</td><td>1.35</td><td>1.34</td><td>1.04</td><td>0.91</td><td>0.87</td><td>1.11</td><td>0.82</td><td>0.79</td><td>0.85</td><td>0.91</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 3.50 (1, 2) →
2.61 (2, 3); median sep 4.29 → 3.35;
mean spread 0.119 → 0.172; mean radius 0.447 →
0.505; leave-one-out accuracy 1.000 → 0.998.</p>
<figure><img src="notebook/fig/mod10_track_b2.png"><figcaption>Block 2: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 3</h4>
<p>Linear (secant r² 0.97). The largest write is to residue 0 (1.23 of its radius), then 2 and 5 (0.90, 0.98); the others get 0.57–0.85. Radius grows 0.52 → 0.78, and the closest pair rises slightly (2.58 → 2.73).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.03 / 0.03 / 0.05 / 0.04 / 0.03</td><td>0.90–1.10</td>
<td>0.067</td><td>0.82</td>
<td>value path r² 0.15; weight on itself 0.034</td></tr>
<tr><td>MLP</td><td>0.90 / 0.77 / 0.48 / 0.55 / 0.90</td><td>0.98–1.00</td>
<td>0.760</td><td>0.96</td>
<td>secant r² 0.97; additivity r² 0.98</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.88 / 0.89 / 0.92 / 0.92 / 0.89. Roundness (k = 1…4) entering the block:
0.83 / 0.77 / 0.85 / 0.43; leaving it: 0.89 / 0.90 / 0.84 / 0.37.
Largest attention heads in the value path: head 5 (0.44), head 7 (0.15), head 28 (0.11).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g145</td><td>0.37</td><td>8</td><td>0.12</td><td>0.96</td></tr><tr><td>g61</td><td>0.19</td><td>4</td><td>0.34</td><td>0.95</td></tr><tr><td>g67</td><td>0.16</td><td>5</td><td>0.43</td><td>0.99</td></tr><tr><td>g30</td><td>0.16</td><td>6</td><td>0.30</td><td>0.96</td></tr><tr><td>g177</td><td>0.12</td><td>8</td><td>0.30</td><td>0.96</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>1.23</td><td>0.85</td><td>0.90</td><td>0.76</td><td>0.80</td><td>0.98</td><td>0.78</td><td>0.68</td><td>0.57</td><td>0.58</td></tr>
<tr><td>radius after / before</td><td>2.24</td><td>1.48</td><td>1.66</td><td>1.42</td><td>1.52</td><td>1.83</td><td>1.37</td><td>1.34</td><td>1.20</td><td>1.24</td></tr>
<tr><td>spread after / mean spread</td><td>1.23</td><td>1.29</td><td>1.02</td><td>0.94</td><td>0.94</td><td>1.12</td><td>0.85</td><td>0.83</td><td>0.89</td><td>0.89</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 2.58 (2, 3) →
2.73 (3, 4); median sep 3.33 → 3.55;
mean spread 0.173 → 0.250; mean radius 0.518 →
0.778; leave-one-out accuracy 0.998 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b3.png"><figcaption>Block 3: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 4</h4>
<p>Linear (0.97). Writes are spread over all residues (0.33–0.57 of their radius). Spread grows faster than radius (0.25 → 0.35 against 0.78 → 1.05), and the closest pair falls 2.72 → 2.22; accuracy stays 1.000.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.01 / 0.01 / 0.02 / 0.02 / 0.01</td><td>0.97–1.09</td>
<td>0.030</td><td>0.85</td>
<td>value path r² -0.42; weight on itself 0.041</td></tr>
<tr><td>MLP</td><td>0.48 / 0.35 / 0.32 / 0.44 / 0.60</td><td>0.97–1.00</td>
<td>0.430</td><td>0.96</td>
<td>secant r² 0.97; additivity r² 0.99</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.95 / 0.96 / 0.95 / 0.94 / 0.93. Roundness (k = 1…4) entering the block:
0.89 / 0.90 / 0.84 / 0.37; leaving it: 0.80 / 0.82 / 0.91 / 0.35.
Largest attention heads in the value path: head 26 (0.58), head 9 (0.14), head 3 (0.10).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g91</td><td>0.38</td><td>0</td><td>0.18</td><td>0.96</td></tr><tr><td>u147</td><td>0.22</td><td>5</td><td>0.12</td><td>0.99</td></tr><tr><td>u52</td><td>0.16</td><td>0</td><td>0.46</td><td>0.96</td></tr><tr><td>u94</td><td>0.16</td><td>9</td><td>0.19</td><td>0.96</td></tr><tr><td>u48</td><td>0.08</td><td>0</td><td>0.28</td><td>0.96</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.47</td><td>0.33</td><td>0.45</td><td>0.37</td><td>0.42</td><td>0.38</td><td>0.57</td><td>0.52</td><td>0.57</td><td>0.48</td></tr>
<tr><td>radius after / before</td><td>1.38</td><td>0.99</td><td>1.36</td><td>1.27</td><td>1.35</td><td>1.32</td><td>1.56</td><td>1.45</td><td>1.52</td><td>1.37</td></tr>
<tr><td>spread after / mean spread</td><td>1.29</td><td>1.21</td><td>1.02</td><td>0.94</td><td>0.98</td><td>1.07</td><td>0.85</td><td>0.83</td><td>0.93</td><td>0.88</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 2.72 (3, 4) →
2.22 (2, 4); median sep 3.53 → 3.10;
mean spread 0.251 → 0.346; mean radius 0.778 →
1.054; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b4.png"><figcaption>Block 4: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 5</h4>
<p>Linear (0.96). Uniform writes (0.36–0.49). Separation keeps falling (closest pair 2.22 → 1.70), and leave-one-out accuracy drops below 1 for the first time (0.985).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.01 / 0.01 / 0.01 / 0.01</td><td>0.90–1.01</td>
<td>0.008</td><td>0.84</td>
<td>value path r² 0.40; weight on itself 0.032</td></tr>
<tr><td>MLP</td><td>0.24 / 0.38 / 0.47 / 0.58 / 0.26</td><td>0.98–1.00</td>
<td>0.434</td><td>0.96</td>
<td>secant r² 0.96; additivity r² 0.98</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.97 / 0.94 / 0.91 / 0.90 / 0.97. Roundness (k = 1…4) entering the block:
0.80 / 0.82 / 0.91 / 0.35; leaving it: 0.79 / 0.77 / 0.92 / 0.32.
Largest attention heads in the value path: head 22 (0.73), head 13 (0.13), head 21 (0.03).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>u339</td><td>0.43</td><td>8</td><td>0.18</td><td>0.98</td></tr><tr><td>u40</td><td>0.28</td><td>0</td><td>0.55</td><td>0.98</td></tr><tr><td>g22</td><td>0.21</td><td>9</td><td>0.15</td><td>0.94</td></tr><tr><td>g4</td><td>0.05</td><td>4</td><td>0.16</td><td>0.91</td></tr><tr><td>g138</td><td>0.03</td><td>8</td><td>0.17</td><td>0.84</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.40</td><td>0.45</td><td>0.43</td><td>0.43</td><td>0.42</td><td>0.36</td><td>0.41</td><td>0.44</td><td>0.47</td><td>0.49</td></tr>
<tr><td>radius after / before</td><td>1.24</td><td>1.27</td><td>1.23</td><td>1.21</td><td>1.25</td><td>1.24</td><td>1.27</td><td>1.26</td><td>1.28</td><td>1.32</td></tr>
<tr><td>spread after / mean spread</td><td>1.26</td><td>1.22</td><td>1.00</td><td>0.93</td><td>0.97</td><td>1.08</td><td>0.85</td><td>0.84</td><td>0.94</td><td>0.91</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 2.22 (2, 4) →
1.70 (2, 4); median sep 3.10 → 2.48;
mean spread 0.346 → 0.469; mean radius 1.063 →
1.336; leave-one-out accuracy 1.000 → 0.985.</p>
<figure><img src="notebook/fig/mod10_track_b5.png"><figcaption>Block 5: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 6</h4>
<p>Linear (0.99). Uniform writes (0.28–0.40); closest pair 1.70 → 1.36, accuracy 0.955.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.01 / 0.01 / 0.00 / 0.01</td><td>0.95–1.14</td>
<td>0.008</td><td>0.93</td>
<td>value path r² 0.08; weight on itself 0.036</td></tr>
<tr><td>MLP</td><td>0.19 / 0.30 / 0.38 / 0.42 / 0.17</td><td>0.96–1.00</td>
<td>0.433</td><td>0.98</td>
<td>secant r² 0.99; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.98 / 0.96 / 0.94 / 0.91 / 0.99. Roundness (k = 1…4) entering the block:
0.79 / 0.77 / 0.92 / 0.32; leaving it: 0.79 / 0.76 / 0.94 / 0.34.
Largest attention heads in the value path: head 8 (0.25), head 11 (0.13), head 7 (0.12).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>u375</td><td>0.65</td><td>9</td><td>0.14</td><td>0.99</td></tr><tr><td>u26</td><td>0.17</td><td>0</td><td>0.33</td><td>0.99</td></tr><tr><td>u185</td><td>0.11</td><td>6</td><td>0.17</td><td>0.99</td></tr><tr><td>u567</td><td>0.08</td><td>9</td><td>0.18</td><td>0.99</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.28</td><td>0.37</td><td>0.36</td><td>0.36</td><td>0.34</td><td>0.31</td><td>0.33</td><td>0.36</td><td>0.35</td><td>0.40</td></tr>
<tr><td>radius after / before</td><td>1.02</td><td>1.03</td><td>1.09</td><td>1.13</td><td>1.11</td><td>1.06</td><td>1.10</td><td>1.12</td><td>1.10</td><td>1.15</td></tr>
<tr><td>spread after / mean spread</td><td>1.20</td><td>1.21</td><td>0.99</td><td>0.97</td><td>0.95</td><td>1.02</td><td>0.85</td><td>0.89</td><td>0.97</td><td>0.95</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.70 (2, 4) →
1.36 (2, 4); median sep 2.48 → 2.00;
mean spread 0.469 → 0.597; mean radius 1.343 →
1.468; leave-one-out accuracy 0.985 → 0.955.</p>
<figure><img src="notebook/fig/mod10_track_b6.png"><figcaption>Block 6: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 7</h4>
<p>Linear (0.99). Residue 1 receives the largest write (0.97 of its radius, against 0.61–0.73); closest pair 1.36 → 1.12, accuracy 0.917.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.01 / 0.01 / 0.01 / 0.00 / 0.00</td><td>0.99–1.10</td>
<td>0.008</td><td>0.96</td>
<td>value path r² 0.46; weight on itself 0.032</td></tr>
<tr><td>MLP</td><td>0.17 / 0.13 / 0.29 / 0.25 / 0.11</td><td>0.87–0.97</td>
<td>0.386</td><td>0.98</td>
<td>secant r² 0.99; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.99 / 0.99 / 0.96 / 0.94 / 0.99. Roundness (k = 1…4) entering the block:
0.79 / 0.76 / 0.94 / 0.34; leaving it: 0.79 / 0.75 / 0.91 / 0.37.
Largest attention heads in the value path: head 0 (0.86), head 5 (0.05), head 2 (0.02).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>u286</td><td>0.44</td><td>9</td><td>0.14</td><td>1.00</td></tr><tr><td>u828</td><td>0.34</td><td>4</td><td>0.13</td><td>1.00</td></tr><tr><td>u198</td><td>0.10</td><td>8</td><td>0.15</td><td>1.00</td></tr><tr><td>g0</td><td>0.06</td><td>0</td><td>0.27</td><td>0.96</td></tr><tr><td>u171</td><td>0.06</td><td>8</td><td>0.17</td><td>1.00</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.73</td><td>0.97</td><td>0.71</td><td>0.69</td><td>0.61</td><td>0.66</td><td>0.64</td><td>0.69</td><td>0.67</td><td>0.69</td></tr>
<tr><td>radius after / before</td><td>1.40</td><td>1.20</td><td>1.27</td><td>1.15</td><td>1.13</td><td>1.25</td><td>1.23</td><td>1.27</td><td>1.22</td><td>1.17</td></tr>
<tr><td>spread after / mean spread</td><td>1.15</td><td>1.21</td><td>0.97</td><td>1.00</td><td>0.94</td><td>0.99</td><td>0.87</td><td>0.91</td><td>1.00</td><td>0.97</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.36 (2, 4) →
1.12 (2, 4); median sep 2.00 → 1.62;
mean spread 0.597 → 0.756; mean radius 1.478 →
1.816; leave-one-out accuracy 0.955 → 0.917.</p>
<figure><img src="notebook/fig/mod10_track_b7.png"><figcaption>Block 7: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 8</h4>
<p>Linear (0.98). Residue 1 again receives the largest write (0.48, against 0.30–0.37); closest pair 1.11 → 1.04, accuracy 0.907.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.01 / 0.01 / 0.02 / 0.03 / 0.00</td><td>0.91–1.29</td>
<td>0.019</td><td>0.91</td>
<td>value path r² 0.71; weight on itself 0.039</td></tr>
<tr><td>MLP</td><td>0.14 / 0.13 / 0.26 / 0.19 / 0.08</td><td>0.91–0.98</td>
<td>0.337</td><td>0.99</td>
<td>secant r² 0.98; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.99 / 0.99 / 0.97 / 0.96 / 1.00. Roundness (k = 1…4) entering the block:
0.79 / 0.75 / 0.91 / 0.37; leaving it: 0.79 / 0.75 / 0.89 / 0.38.
Largest attention heads in the value path: head 23 (0.87), head 19 (0.07), head 16 (0.04).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g109</td><td>0.80</td><td>9</td><td>0.13</td><td>0.98</td></tr><tr><td>g628</td><td>0.15</td><td>5</td><td>0.12</td><td>0.94</td></tr><tr><td>u799</td><td>0.06</td><td>5</td><td>0.13</td><td>0.97</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.31</td><td>0.48</td><td>0.33</td><td>0.37</td><td>0.32</td><td>0.31</td><td>0.30</td><td>0.32</td><td>0.33</td><td>0.34</td></tr>
<tr><td>radius after / before</td><td>1.20</td><td>1.17</td><td>1.19</td><td>1.21</td><td>1.16</td><td>1.16</td><td>1.14</td><td>1.14</td><td>1.17</td><td>1.14</td></tr>
<tr><td>spread after / mean spread</td><td>1.14</td><td>1.20</td><td>0.96</td><td>1.00</td><td>0.95</td><td>0.97</td><td>0.88</td><td>0.92</td><td>1.01</td><td>0.97</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.11 (2, 4) →
1.04 (2, 4); median sep 1.62 → 1.52;
mean spread 0.758 → 0.811; mean radius 1.777 →
2.077; leave-one-out accuracy 0.917 → 0.907.</p>
<figure><img src="notebook/fig/mod10_track_b8.png"><figcaption>Block 8: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 9</h4>
<p>Linear (0.99). Small, uniform writes (0.14–0.20); closest pair 1.04 → 1.01, accuracy 0.912.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.00 / 0.00 / 0.00 / 0.00</td><td>1.00–1.12</td>
<td>0.003</td><td>0.96</td>
<td>value path r² 0.96; weight on itself 0.039</td></tr>
<tr><td>MLP</td><td>0.13 / 0.18 / 0.28 / 0.32 / 0.26</td><td>0.98–1.00</td>
<td>0.299</td><td>0.99</td>
<td>secant r² 0.99; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.99 / 0.99 / 0.96 / 0.95 / 0.97. Roundness (k = 1…4) entering the block:
0.79 / 0.75 / 0.89 / 0.38; leaving it: 0.79 / 0.75 / 0.90 / 0.37.
Largest attention heads in the value path: head 11 (0.43), head 5 (0.21), head 15 (0.07).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g61</td><td>0.52</td><td>4</td><td>0.14</td><td>0.99</td></tr><tr><td>u33</td><td>0.31</td><td>9</td><td>0.17</td><td>1.00</td></tr><tr><td>u462</td><td>0.11</td><td>6</td><td>0.13</td><td>0.97</td></tr><tr><td>g14</td><td>0.04</td><td>0</td><td>0.55</td><td>0.95</td></tr><tr><td>g1</td><td>0.03</td><td>8</td><td>0.16</td><td>0.93</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.18</td><td>0.20</td><td>0.15</td><td>0.16</td><td>0.15</td><td>0.14</td><td>0.14</td><td>0.14</td><td>0.16</td><td>0.14</td></tr>
<tr><td>radius after / before</td><td>1.04</td><td>0.85</td><td>0.95</td><td>0.97</td><td>0.93</td><td>0.94</td><td>0.94</td><td>0.94</td><td>0.98</td><td>0.90</td></tr>
<tr><td>spread after / mean spread</td><td>1.14</td><td>1.21</td><td>0.96</td><td>1.01</td><td>0.94</td><td>0.98</td><td>0.88</td><td>0.92</td><td>1.01</td><td>0.96</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.04 (2, 4) →
1.01 (2, 4); median sep 1.52 → 1.50;
mean spread 0.811 → 0.840; mean radius 2.016 →
1.908; leave-one-out accuracy 0.907 → 0.912.</p>
<figure><img src="notebook/fig/mod10_track_b9.png"><figcaption>Block 9: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 10</h4>
<p>Linear (0.98). The smallest writes of all blocks (0.07–0.15), radius ratios 0.98–1.00. The closest pair reaches its lowest value (1.01 → 0.95), with accuracy 0.897.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.00 / 0.00 / 0.00 / 0.00</td><td>0.95–1.13</td>
<td>0.003</td><td>0.93</td>
<td>value path r² 0.96; weight on itself 0.028</td></tr>
<tr><td>MLP</td><td>0.14 / 0.14 / 0.25 / 0.19 / 0.09</td><td>0.98</td>
<td>0.236</td><td>0.98</td>
<td>secant r² 0.98; additivity r² 1.00</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.99 / 0.99 / 0.97 / 0.96 / 1.00. Roundness (k = 1…4) entering the block:
0.79 / 0.75 / 0.90 / 0.37; leaving it: 0.79 / 0.75 / 0.89 / 0.38.
Largest attention heads in the value path: head 10 (0.78), head 30 (0.07), head 9 (0.05).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g195</td><td>0.76</td><td>4</td><td>0.13</td><td>0.99</td></tr><tr><td>g35</td><td>0.10</td><td>0</td><td>0.17</td><td>0.96</td></tr><tr><td>u165</td><td>0.07</td><td>7</td><td>0.12</td><td>0.90</td></tr><tr><td>g20</td><td>0.07</td><td>4</td><td>0.14</td><td>0.98</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.07</td><td>0.15</td><td>0.10</td><td>0.10</td><td>0.11</td><td>0.09</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.12</td></tr>
<tr><td>radius after / before</td><td>1.00</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.99</td><td>1.00</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td></tr>
<tr><td>spread after / mean spread</td><td>1.12</td><td>1.20</td><td>0.96</td><td>1.01</td><td>0.94</td><td>0.97</td><td>0.88</td><td>0.93</td><td>1.02</td><td>0.96</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.01 (2, 4) →
0.95 (2, 4); median sep 1.50 → 1.42;
mean spread 0.840 → 0.891; mean radius 1.829 →
1.805; leave-one-out accuracy 0.912 → 0.897.</p>
<figure><img src="notebook/fig/mod10_track_b10.png"><figcaption>Block 10: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 11</h4>
<p>Still linear (secant 0.94, additivity 0.99), but the first block since block 0 with a large write (gain up to 0.81). Residue 1 is moved most (0.68 of its radius, against 0.19–0.38). Separation starts to recover: median 1.42 → 1.82, accuracy 0.897 → 0.945.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.00 / 0.00 / 0.00 / 0.00</td><td>0.68–1.86</td>
<td>0.002</td><td>0.79</td>
<td>value path r² -0.08; weight on itself 0.025</td></tr>
<tr><td>MLP</td><td>0.81 / 0.59 / 0.61 / 0.44 / 0.33</td><td>0.99–1.01</td>
<td>0.344</td><td>0.95</td>
<td>secant r² 0.94; additivity r² 0.99</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.85 / 0.89 / 0.88 / 0.92 / 0.95. Roundness (k = 1…4) entering the block:
0.79 / 0.75 / 0.89 / 0.38; leaving it: 0.71 / 0.72 / 0.76 / 0.36.
Largest attention heads in the value path: head 4 (0.41), head 16 (0.13), head 19 (0.07).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (2 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g18</td><td>0.28</td><td>1</td><td>0.33</td><td>0.95</td></tr><tr><td>g32</td><td>0.24</td><td>5</td><td>0.46</td><td>0.95</td></tr><tr><td>g43</td><td>0.19</td><td>0</td><td>0.57</td><td>0.91</td></tr><tr><td>u86</td><td>0.10</td><td>9</td><td>0.15</td><td>0.95</td></tr><tr><td>u47</td><td>0.06</td><td>8</td><td>0.52</td><td>0.90</td></tr><tr><td>u395</td><td>0.04</td><td>4</td><td>0.16</td><td>0.87</td></tr><tr><td>g3</td><td>0.02</td><td>5</td><td>0.16</td><td>0.89</td></tr><tr><td>g837</td><td>0.02</td><td>7</td><td>0.46</td><td>0.95</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.19</td><td>0.68</td><td>0.38</td><td>0.31</td><td>0.29</td><td>0.34</td><td>0.29</td><td>0.30</td><td>0.27</td><td>0.29</td></tr>
<tr><td>radius after / before</td><td>1.13</td><td>1.67</td><td>1.32</td><td>1.28</td><td>1.25</td><td>1.25</td><td>1.23</td><td>1.26</td><td>1.20</td><td>1.27</td></tr>
<tr><td>spread after / mean spread</td><td>1.11</td><td>1.20</td><td>0.97</td><td>1.01</td><td>0.96</td><td>0.99</td><td>0.88</td><td>0.93</td><td>1.00</td><td>0.96</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 0.95 (2, 4) →
1.05 (7, 9); median sep 1.42 → 1.82;
mean spread 0.891 → 0.904; mean radius 1.818 →
2.297; leave-one-out accuracy 0.897 → 0.945.</p>
<figure><img src="notebook/fig/mod10_track_b11.png"><figcaption>Block 11: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 12</h4>
<p>The first block where the MLP's period-10 action is no longer a sum of independent linear reader paths: secant r² 0.76 and additivity 0.81, while the linear fit on z is still 0.92. The k = 3 plane turns most (cosine 0.70) and becomes round (0.76 → 0.98). The clouds tighten (spread 0.90 → 0.83) and separate (median 1.82 → 2.70; accuracy 0.988). Residues 1 and 9 are moved most (0.49, 0.45, against 0.19–0.27).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.01 / 0.00 / 0.01 / 0.01</td><td>0.91–1.17</td>
<td>0.008</td><td>0.88</td>
<td>value path r² 0.77; weight on itself 0.060</td></tr>
<tr><td>MLP</td><td>0.67 / 0.62 / 1.17 / 0.72 / 0.45</td><td>0.97–1.01</td>
<td>0.573</td><td>0.92</td>
<td>secant r² 0.76; additivity r² 0.81</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.88 / 0.88 / 0.70 / 0.80 / 0.92. Roundness (k = 1…4) entering the block:
0.71 / 0.72 / 0.76 / 0.36; leaving it: 0.76 / 0.77 / 0.98 / 0.39.
Largest attention heads in the value path: head 12 (0.74), head 20 (0.14), head 3 (0.06).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (1 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g10</td><td>0.24</td><td>2</td><td>0.23</td><td>0.73</td></tr><tr><td>g20</td><td>0.17</td><td>8</td><td>0.21</td><td>0.80</td></tr><tr><td>g33</td><td>0.16</td><td>6</td><td>0.36</td><td>0.81</td></tr><tr><td>g37</td><td>0.15</td><td>3</td><td>0.35</td><td>0.82</td></tr><tr><td>g26</td><td>0.08</td><td>0</td><td>0.39</td><td>0.65</td></tr><tr><td>g381</td><td>0.07</td><td>9</td><td>0.14</td><td>0.98</td></tr><tr><td>u26</td><td>0.05</td><td>0</td><td>0.62</td><td>0.57</td></tr><tr><td>g11</td><td>0.05</td><td>7</td><td>0.29</td><td>0.74</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.24</td><td>0.49</td><td>0.19</td><td>0.27</td><td>0.22</td><td>0.21</td><td>0.27</td><td>0.24</td><td>0.21</td><td>0.45</td></tr>
<tr><td>radius after / before</td><td>0.94</td><td>0.90</td><td>0.87</td><td>0.87</td><td>0.88</td><td>0.89</td><td>0.94</td><td>0.87</td><td>0.82</td><td>1.00</td></tr>
<tr><td>spread after / mean spread</td><td>1.14</td><td>1.18</td><td>0.98</td><td>1.03</td><td>0.97</td><td>1.01</td><td>0.87</td><td>0.90</td><td>0.96</td><td>0.96</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.05 (7, 9) →
1.45 (2, 3); median sep 1.82 → 2.70;
mean spread 0.903 → 0.829; mean radius 2.335 →
2.096; leave-one-out accuracy 0.945 → 0.988.</p>
<figure><img src="notebook/fig/mod10_track_b12.png"><figcaption>Block 12: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 13</h4>
<p>The most non-linear block: linear fit 0.70, secant 0.51, additivity 0.75; its main readers are less one-hot (0.12–0.59) and individually non-linear (secant r² 0.34–0.54). Planes k = 3, 4 turn (cosine 0.66, 0.65), and k = 4 becomes rounder (0.39 → 0.70). The clouds tighten (spread 0.83 → 0.67) and separate sharply: closest pair 1.45 → 3.06, median 2.70 → 4.21, accuracy 1.000. Residues 3, 7 and 8 are moved most (0.59, 0.65, 0.52, against 0.21–0.45).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.00 / 0.00 / 0.00 / 0.00</td><td>0.71–1.09</td>
<td>0.003</td><td>0.92</td>
<td>value path r² 0.98; weight on itself 0.041</td></tr>
<tr><td>MLP</td><td>0.52 / 0.63 / 1.22 / 0.96 / 0.49</td><td>1.00</td>
<td>0.537</td><td>0.70</td>
<td>secant r² 0.51; additivity r² 0.75</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.92 / 0.89 / 0.66 / 0.65 / 0.92. Roundness (k = 1…4) entering the block:
0.76 / 0.77 / 0.98 / 0.39; leaving it: 0.87 / 0.81 / 0.90 / 0.70.
Largest attention heads in the value path: head 28 (0.80), head 7 (0.07), head 19 (0.06).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (2 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g9</td><td>0.19</td><td>7</td><td>0.59</td><td>0.53</td></tr><tr><td>g11</td><td>0.16</td><td>5</td><td>0.53</td><td>0.43</td></tr><tr><td>g14</td><td>0.13</td><td>3</td><td>0.41</td><td>0.39</td></tr><tr><td>g10</td><td>0.12</td><td>8</td><td>0.27</td><td>0.34</td></tr><tr><td>g114</td><td>0.11</td><td>8</td><td>0.12</td><td>0.54</td></tr><tr><td>g46</td><td>0.09</td><td>0</td><td>0.47</td><td>0.59</td></tr><tr><td>u52</td><td>0.08</td><td>0</td><td>0.49</td><td>0.59</td></tr><tr><td>g53</td><td>0.07</td><td>7</td><td>0.39</td><td>0.34</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.21</td><td>0.29</td><td>0.32</td><td>0.59</td><td>0.40</td><td>0.45</td><td>0.22</td><td>0.65</td><td>0.52</td><td>0.31</td></tr>
<tr><td>radius after / before</td><td>1.04</td><td>0.97</td><td>1.06</td><td>1.18</td><td>1.12</td><td>1.14</td><td>0.99</td><td>1.19</td><td>1.19</td><td>1.07</td></tr>
<tr><td>spread after / mean spread</td><td>1.18</td><td>1.14</td><td>1.03</td><td>1.00</td><td>0.98</td><td>1.07</td><td>0.83</td><td>0.87</td><td>0.95</td><td>0.95</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 1.45 (2, 3) →
3.06 (2, 3); median sep 2.70 → 4.21;
mean spread 0.829 → 0.667; mean radius 2.057 →
2.250; leave-one-out accuracy 0.988 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b13.png"><figcaption>Block 13: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 14</h4>
<p>Non-linear (linear fit 0.73, secant 0.64, additivity 0.83). The writes split the residues into two groups: 1, 2, 4, 6 and 9 receive 0.38–0.43 of their radius, and 0, 3, 5, 7 and 8 receive 0.17. Separation increases further (closest pair 3.06 → 4.12, median 4.22 → 5.09).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.00 / 0.00 / 0.00 / 0.00</td><td>0.98–1.14</td>
<td>0.003</td><td>0.66</td>
<td>value path r² -0.68; weight on itself 0.040</td></tr>
<tr><td>MLP</td><td>0.42 / 0.48 / 0.68 / 0.65 / 0.53</td><td>0.99–1.00</td>
<td>0.568</td><td>0.73</td>
<td>secant r² 0.64; additivity r² 0.83</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.92 / 0.92 / 0.83 / 0.86 / 0.91. Roundness (k = 1…4) entering the block:
0.87 / 0.81 / 0.90 / 0.70; leaving it: 0.89 / 0.92 / 0.91 / 0.66.
Largest attention heads in the value path: head 26 (0.50), head 8 (0.22), head 18 (0.06).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (2 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g98</td><td>0.19</td><td>8</td><td>0.26</td><td>0.63</td></tr><tr><td>g23</td><td>0.18</td><td>2</td><td>0.62</td><td>0.45</td></tr><tr><td>g24</td><td>0.13</td><td>4</td><td>0.89</td><td>0.84</td></tr><tr><td>g40</td><td>0.12</td><td>9</td><td>0.63</td><td>0.69</td></tr><tr><td>u83</td><td>0.11</td><td>8</td><td>0.12</td><td>0.44</td></tr><tr><td>u63</td><td>0.10</td><td>6</td><td>0.60</td><td>0.68</td></tr><tr><td>g63</td><td>0.06</td><td>6</td><td>0.79</td><td>0.72</td></tr><tr><td>u38</td><td>0.05</td><td>0</td><td>0.48</td><td>0.83</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.17</td><td>0.43</td><td>0.40</td><td>0.17</td><td>0.42</td><td>0.17</td><td>0.38</td><td>0.17</td><td>0.17</td><td>0.39</td></tr>
<tr><td>radius after / before</td><td>0.97</td><td>1.25</td><td>1.12</td><td>1.03</td><td>1.18</td><td>1.03</td><td>1.19</td><td>1.06</td><td>1.04</td><td>1.19</td></tr>
<tr><td>spread after / mean spread</td><td>1.28</td><td>1.09</td><td>1.01</td><td>0.98</td><td>0.97</td><td>1.12</td><td>0.84</td><td>0.84</td><td>0.95</td><td>0.91</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 3.06 (2, 3) →
4.12 (5, 6); median sep 4.22 → 5.09;
mean spread 0.667 → 0.642; mean radius 2.244 →
2.469; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b14.png"><figcaption>Block 14: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 15</h4>
<p>Additive over readers again (0.97) but not linear per reader (secant 0.75, linear fit 0.74). Writes are small (0.08–0.20). Spread grows (0.64 → 0.71), and the closest pair falls slightly (4.12 → 3.89).</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.00 / 0.01 / 0.00 / 0.00 / 0.01</td><td>0.98–1.03</td>
<td>0.006</td><td>0.90</td>
<td>value path r² 0.28; weight on itself 0.048</td></tr>
<tr><td>MLP</td><td>0.34 / 0.32 / 0.28 / 0.35 / 0.27</td><td>0.94–0.99</td>
<td>0.395</td><td>0.74</td>
<td>secant r² 0.75; additivity r² 0.97</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.95 / 0.95 / 0.96 / 0.96 / 0.97. Roundness (k = 1…4) entering the block:
0.89 / 0.92 / 0.91 / 0.66; leaving it: 0.90 / 0.86 / 0.90 / 0.62.
Largest attention heads in the value path: head 6 (0.32), head 0 (0.16), head 13 (0.09).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (4 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g67</td><td>0.40</td><td>7</td><td>0.14</td><td>0.81</td></tr><tr><td>u111</td><td>0.16</td><td>0</td><td>0.25</td><td>0.88</td></tr><tr><td>g268</td><td>0.09</td><td>1</td><td>0.22</td><td>0.59</td></tr><tr><td>u399</td><td>0.06</td><td>1</td><td>0.23</td><td>0.75</td></tr><tr><td>u176</td><td>0.06</td><td>9</td><td>0.68</td><td>0.65</td></tr><tr><td>g5</td><td>0.04</td><td>0</td><td>0.29</td><td>0.87</td></tr><tr><td>u324</td><td>0.04</td><td>0</td><td>0.36</td><td>0.49</td></tr><tr><td>g230</td><td>0.04</td><td>9</td><td>0.85</td><td>0.72</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.20</td><td>0.08</td><td>0.12</td><td>0.18</td><td>0.17</td><td>0.17</td><td>0.17</td><td>0.17</td><td>0.19</td><td>0.16</td></tr>
<tr><td>radius after / before</td><td>1.08</td><td>0.97</td><td>1.05</td><td>1.12</td><td>1.13</td><td>1.07</td><td>1.08</td><td>1.07</td><td>1.12</td><td>1.08</td></tr>
<tr><td>spread after / mean spread</td><td>1.25</td><td>1.02</td><td>1.00</td><td>1.01</td><td>1.01</td><td>1.14</td><td>0.89</td><td>0.83</td><td>0.96</td><td>0.89</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 4.12 (5, 6) →
3.89 (2, 3); median sep 5.09 → 4.72;
mean spread 0.642 → 0.707; mean radius 2.465 →
2.658; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b15.png"><figcaption>Block 15: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>
<h4>Block 16</h4>
<p>Additive (0.95), secant 0.77, linear fit 0.79. Residue 1 is moved most (0.47, against 0.11–0.29). Spread grows (0.71 → 0.88) faster than radius (2.69 → 3.21), and the closest pair falls 3.88 → 3.52.</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>0.04 / 0.04 / 0.03 / 0.02 / 0.04</td><td>0.99–1.00</td>
<td>0.039</td><td>0.95</td>
<td>value path r² 0.86; weight on itself 0.066</td></tr>
<tr><td>MLP</td><td>0.32 / 0.40 / 0.38 / 0.50 / 0.36</td><td>0.95–0.99</td>
<td>0.398</td><td>0.79</td>
<td>secant r² 0.77; additivity r² 0.95</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
1.00 / 1.00 / 1.00 / 1.00 / 1.00; across the MLP:
0.95 / 0.94 / 0.95 / 0.91 / 0.95. Roundness (k = 1…4) entering the block:
0.90 / 0.86 / 0.90 / 0.62; leaving it: 0.89 / 0.86 / 0.92 / 0.69.
Largest attention heads in the value path: head 22 (0.47), head 30 (0.44), head 5 (0.02).</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖² (5 more readers not listed):</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr><tr><td>g51</td><td>0.30</td><td>1</td><td>0.18</td><td>0.85</td></tr><tr><td>g48</td><td>0.16</td><td>8</td><td>0.65</td><td>0.80</td></tr><tr><td>g348</td><td>0.16</td><td>0</td><td>0.25</td><td>0.81</td></tr><tr><td>u134</td><td>0.09</td><td>7</td><td>0.89</td><td>0.91</td></tr><tr><td>u413</td><td>0.06</td><td>3</td><td>0.76</td><td>0.87</td></tr><tr><td>u207</td><td>0.05</td><td>1</td><td>0.14</td><td>0.48</td></tr><tr><td>g62</td><td>0.04</td><td>4</td><td>0.20</td><td>0.83</td></tr><tr><td>u666</td><td>0.04</td><td>0</td><td>0.26</td><td>0.54</td></tr></table></div>
<div class="tw"><table><tr><th>MLP, per residue r</th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>write / input radius</td><td>0.11</td><td>0.47</td><td>0.17</td><td>0.21</td><td>0.16</td><td>0.21</td><td>0.27</td><td>0.27</td><td>0.29</td><td>0.16</td></tr>
<tr><td>radius after / before</td><td>0.96</td><td>1.43</td><td>1.16</td><td>1.17</td><td>1.14</td><td>1.19</td><td>1.25</td><td>1.23</td><td>1.25</td><td>1.17</td></tr>
<tr><td>spread after / mean spread</td><td>1.26</td><td>1.06</td><td>1.02</td><td>0.98</td><td>1.01</td><td>1.07</td><td>0.89</td><td>0.83</td><td>1.01</td><td>0.88</td></tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep 3.88 (2, 3) →
3.52 (2, 3); median sep 4.70 → 4.31;
mean spread 0.711 → 0.877; mean radius 2.691 →
3.205; leave-one-out accuracy 1.000 → 1.000.</p>
<figure><img src="notebook/fig/mod10_track_b16.png"><figcaption>Block 16: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure><!-- track:end -->
`
},
{
id: "mod10_p3",
title: "Mod 10 representations at b (position 3)",
date: "2026-10-22",
html: `
<p><b>Summary.</b> At the token <i>b</i> (position 3), block 0 holds the same mod-10 representation as at the token
<i>a</i>: the same five Fourier planes (principal cosines ≥ 0.9996), read by the same 11 readers, which feed the
same writers. Two things differ. (1) Its amplitude at the MLP input is 0.87–0.87 times
position 1's, because block-0 attention at position 3 adds a larger vector to the stream and the RMSNorm then divides
by a larger norm. (2) From block 0 on, attention brings the operation and <i>a</i> into position 3. Over blocks 2–10, attention and
the MLPs turn them into an offset of the whole <i>b</i> circle inside its own subspace, one offset per context (operation, <i>a</i>).
Pooled over contexts, the <i>b</i> clouds then overlap: by block 11 no residue is linearly separable from the others
(position 1: all ten at every block). Within a single context, at least 8 of the 10 still are, and all 10
up to block 7 and from block 13. The MLPs of blocks 12–14
shrink the operation offset and enlarge the circle, and separability comes back (9 of 10 from block 14).</p>

<h3>Setup</h3>
<p>Run p-ba8b0c08, step 40000, alive-only model, prompts <code>&lt;BOS&gt; a op b =</code> (tokenized without spaces, as
in the pool). The token <i>b</i> is the same token as the number <i>b</i> at position 1, so the two positions start from the
same embedding rows. Position 3 attends to &lt;BOS&gt;, <i>a</i>, the operation and itself, so its stream depends on
(<i>a</i>, op, <i>b</i>). The residue is <i>r</i> = <i>b</i> mod 10. Block-0 numbers use the stored activations of all 320,000 prompts
(<code>mod10_p3_L0.py</code>). The tracking through block 16 uses a float64 rebuild of positions 0–3 (<code>track_build_p3.py</code>;
max relative error against the stored activations 2.3%) on 8,000 prompts: the operands <i>a</i> = 7, 48, 89, 130, 171, 212,
253, 294, 335, 376 (one per residue of <i>a</i>), every <i>b</i> = 1 … 400 and both operations; a <b>context</b> is one
(operation, <i>a</i>) pair, 20 in all, and each residue class pools 800 prompts over the 20 contexts. All quantities are
defined as in the "Mod 10 representations" page (sections 1–8), with <i>a</i> mod 10 replaced by <i>b</i> mod 10. The
per-reader mechanism of each MLP runs on 1,600 of the prompts (<i>a</i> = 7 and 171). Nothing is centred.
Scripts: <code>track_p3.py</code>, <code>p3_page.py</code>.</p>

<h3>1. Block 0: the same representation, smaller</h3>
<ul>
<li><b>Same planes.</b> In the readers' span, the five planes of <i>b</i> mod 10 at position 3 and of <i>a</i> mod 10 at position 1
have principal cosines 0.9996–1.0000. Circulant statistic: 0.045 (position 1: 0.047).</li>
<li><b>Same readers.</b> The readers with r²<sub><i>D</i></sub> ≥ 0.5 are the same 11 (g69, g59, u60, g47, g58, u54, g62, u52, g43,
u794, u31), with r² 0.53–0.98.
The position-1 subspace explains them equally well
(0.50–0.98).
Their activation hardly depends on the context: the part that varies with (<i>a</i>, op) at fixed <i>b</i> holds
0.04% or less of Σ <i>f</i>² (u31: 0.56%).</li>
<li><b>Same writers.</b> Each reader's main writer (variance split at position 3) is the same as at position 1, with
reader fractions 0.62–0.88: g43 → d43, g47 → d47, g58 → d58, g59 → d59, g62 → d62, g69 → d69, u52 → d52, u54 → d54,
u60 → d60, u794 → d41, u31 → d41.</li>
<li><b>Smaller amplitude, from the RMSNorm.</b> ‖Ψ<sub><i>k</i></sub>‖ in the readers' span is
0.87–0.87 times position 1's. Block-0 attention adds a vector of mean norm 0.47 at position 3
against 0.22 at position 1, so the stream entering the MLP has mean norm 0.80 against 0.70. The same embedding structure is
therefore divided by a 1.16 times larger norm. Block-0 MLP's period-10 gain is correspondingly lower: 1.08–1.51 against
1.37–1.86.</li>
</ul>

<h3>2. Through blocks 0–16: what is the same</h3>
<ul>
<li>Attention never acts on the <i>b</i> representation itself (period-10 gain ≤ 0.07). Position 3 puts 79–93% of its attention
on &lt;BOS&gt;, 1–8% on <i>a</i>, 3–11% on the operation and 1–8% on itself.</li>
<li>The MLPs rewrite the representation as at position 1: their gains follow the same pattern (block 0, then blocks 3–4,
11–14), they are near-additive over readers (≥ 0.92 in blocks 0–12), and blocks 12–14 are the least linear (secant r²
0.80, 0.53, 0.67; additivity 0.92, 0.80, 0.81), as at position 1 (0.76, 0.51, 0.64).</li>
<li>The planes drift from the embedding's at the same pace (mean principal cosine leaving block 16: 0.23–0.37, against
0.20–0.34).</li>
</ul>

<h3>3. What differs: the context moves the <i>b</i> circle</h3>
<p><b><i>a</i> and the operation arrive at position 3.</b> The Fourier amplitude of the means over <i>a</i> mod 10 at position 3 is
0 in the embedding and grows from block 0's attention on (0.2 after block 0's attention, 1.2 after block 1's, 2.6 entering
block 3, 7.2 entering block 11, 9.0 leaving block 16),
larger than <i>b</i>'s own. At most 6% of it lies in <i>b</i>'s subspace <i>D</i>. The difference between the + and − means grows from 0.3
(after block-0 attention) to 4.9 entering block 11.</p>
<p><b>Inside <i>D</i>, the context translates the circle.</b> For each context <i>k</i>, let <i>C<sub>k,r</sub></i> be the mean of the
<i>D</i> coordinates over that context's prompts with <i>b</i> mod 10 = <i>r</i>. From block 2 to block 16, one offset vector
per context reproduces 0.84–0.99 of the variation of <i>C<sub>k,r</sub></i> across contexts. Each context
therefore has its own copy of the circle, shifted as a whole. Most of the shift is between the two operations
(panel d: the offset between the + and − circles reaches 1.10 entering block 11, against a circle radius of 0.89). The
shift builds up over blocks 2–10. Block-2 attention takes it from 0.04 to 0.16; the attention of blocks 4–9 adds 0.03–0.07
each; the MLPs add 0.02–0.07 each, except block 7's, which adds 0.36. The MLPs' period-10
output at position 3 is less tied to <i>D</i> than at position 1: in blocks 2–10, 0.77–1.00 of it comes from <i>D</i>
(position 1: 0.87–1.00).</p>
<p><b>Pooled clouds overlap; each context stays separable.</b> With the 20 contexts pooled, the closest-pair separation falls
to 0.73 entering block 11 (position 1: 0.95) and leave-one-out accuracy to 0.764 (position 1: 0.897). Linear separability
in <i>D</i>, which holds at every block at position 1, is lost: 9 residues separable entering block 5, 5 entering block 7, 0
entering block 11. Within a single context, the residues stay far more separable: the worst of the 20 contexts has all ten separable up to
block 7 and from block 13, and 8 or 9 in between (panel c, green).</p>
<p><b>Blocks 12–14 undo the shift.</b> The operation offset falls from 1.09 to 0.57 entering block 14 while the circle's
radius grows from 0.96 to 1.44. Separation returns to 2.52–3.07 entering blocks 14–15, and 9 of 10 residues are separable
again from block 14.</p>
<figure><img src="notebook/fig/p3_overview.png"><figcaption>(a) Fourier amplitudes ‖Ψ<sub><i>k</i></sub>‖ for <i>k</i> = 1, 5 of the operand's
residue means at position 1 (solid) and position 3 (dashed), and of <i>a</i> mod 10 at position 3 (dotted). (b) Closest-pair
separation of the residue clouds in <i>D</i>. (c) Number of residues linearly separable (affine) from the other nine in <i>D</i>:
position 1, position 3 with the 20 contexts pooled, and the worst single context at position 3. (d) Position 3: mean radius of
the <i>b</i> circle in <i>D</i> (averaged over contexts), norm of the offset between the + and − circles, and mean offset of an
<i>a</i>-context from its operation's average.</figcaption></figure>
<figure><img src="notebook/fig/p3_k1.png"><figcaption>Position 3, projected onto the frequency-1 plane of <i>b</i> mod 10 at each
boundary (axes turned so that residue 0's mean is on the positive horizontal axis; the cross is the activation zero). Top: all
8,000 prompts coloured by <i>b</i> mod 10, with the residue means. Bottom: the same points split by operation (blue +, red −), with
each operation's residue means.</figcaption></figure>

<h3>4. Block by block</h3>
<p>Each cell gives position 1 / position 3 where both are listed. MLP gain is the range over <i>k</i> = 1 … 5 of the MLP's period-10
output relative to its input. Separation and accuracy are measured on the stream leaving the block. "Separable" counts the
residues linearly separable (affine) from the other nine in <i>D</i> at position 3 with contexts pooled, and in brackets the
worst single context. "Op offset / radius" is position 3's offset between the + and − circles and the circle's radius in
<i>D</i>, leaving the block. The last column is position 3's mean attention weight on &lt;BOS&gt; / <i>a</i> / op / itself in the block.</p>
<div class="tw"><table><tr><th>block</th><th>MLP gain p1</th><th>MLP gain p3</th><th>secant r²</th><th>additivity r²</th>
<th>closest-pair sep</th><th>accuracy</th><th>separable p3 (worst context)</th><th>op offset / radius p3</th>
<th>attention weights p3</th></tr><tr><td>0</td><td>1.37–1.86</td><td>1.08–1.51</td><td>0.93 / 0.94</td><td>0.94 / 0.95</td><td>4.83 / 4.80</td><td>1.000 / 1.000</td><td>10 (10)</td><td>0.01 / 0.33</td><td>0.79 / 0.06 / 0.11 / 0.04</td></tr><tr><td>1</td><td>0.04–0.10</td><td>0.04–0.12</td><td>0.91 / 0.68</td><td>1.00 / 1.00</td><td>3.59 / 3.58</td><td>1.000 / 1.000</td><td>10 (10)</td><td>0.04 / 0.32</td><td>0.82 / 0.07 / 0.08 / 0.02</td></tr><tr><td>2</td><td>0.29–0.64</td><td>0.14–0.45</td><td>0.94 / 0.77</td><td>0.99 / 0.99</td><td>2.61 / 2.37</td><td>0.998 / 0.998</td><td>10 (10)</td><td>0.18 / 0.40</td><td>0.93 / 0.01 / 0.04 / 0.01</td></tr><tr><td>3</td><td>0.48–0.90</td><td>0.31–0.77</td><td>0.97 / 0.93</td><td>0.98 / 0.97</td><td>2.73 / 2.47</td><td>1.000 / 1.000</td><td>10 (10)</td><td>0.22 / 0.56</td><td>0.90 / 0.04 / 0.03 / 0.02</td></tr><tr><td>4</td><td>0.32–0.60</td><td>0.27–0.62</td><td>0.97 / 0.93</td><td>0.99 / 1.00</td><td>2.22 / 2.09</td><td>1.000 / 1.000</td><td>9 (10)</td><td>0.33 / 0.69</td><td>0.89 / 0.03 / 0.04 / 0.04</td></tr><tr><td>5</td><td>0.24–0.58</td><td>0.11–0.41</td><td>0.96 / 0.70</td><td>0.98 / 0.94</td><td>1.70 / 1.70</td><td>0.985 / 0.998</td><td>8 (10)</td><td>0.43 / 0.74</td><td>0.86 / 0.05 / 0.06 / 0.03</td></tr><tr><td>6</td><td>0.17–0.42</td><td>0.11–0.31</td><td>0.99 / 0.84</td><td>1.00 / 1.00</td><td>1.36 / 1.38</td><td>0.955 / 0.978</td><td>5 (10)</td><td>0.51 / 0.71</td><td>0.86 / 0.03 / 0.07 / 0.03</td></tr><tr><td>7</td><td>0.11–0.29</td><td>0.08–0.16</td><td>0.99 / 0.86</td><td>1.00 / 1.00</td><td>1.12 / 0.94</td><td>0.917 / 0.883</td><td>5 (9)</td><td>0.94 / 0.91</td><td>0.82 / 0.07 / 0.06 / 0.04</td></tr><tr><td>8</td><td>0.08–0.26</td><td>0.07–0.17</td><td>0.98 / 0.89</td><td>1.00 / 1.00</td><td>1.04 / 0.86</td><td>0.907 / 0.845</td><td>3 (8)</td><td>1.00 / 0.84</td><td>0.82 / 0.04 / 0.07 / 0.07</td></tr><tr><td>9</td><td>0.13–0.32</td><td>0.08–0.22</td><td>0.99 / 0.91</td><td>1.00 / 1.00</td><td>1.01 / 0.76</td><td>0.912 / 0.786</td><td>2 (8)</td><td>1.10 / 1.04</td><td>0.91 / 0.02 / 0.03 / 0.05</td></tr><tr><td>10</td><td>0.09–0.25</td><td>0.08–0.16</td><td>0.98 / 0.88</td><td>1.00 / 1.00</td><td>0.95 / 0.73</td><td>0.897 / 0.764</td><td>0 (8)</td><td>1.10 / 0.89</td><td>0.90 / 0.02 / 0.03 / 0.05</td></tr><tr><td>11</td><td>0.33–0.81</td><td>0.29–0.77</td><td>0.94 / 0.93</td><td>0.99 / 0.98</td><td>1.05 / 0.88</td><td>0.945 / 0.881</td><td>4 (9)</td><td>1.09 / 0.96</td><td>0.90 / 0.04 / 0.03 / 0.03</td></tr><tr><td>12</td><td>0.45–1.17</td><td>0.50–1.46</td><td>0.76 / 0.80</td><td>0.81 / 0.92</td><td>1.45 / 1.33</td><td>0.988 / 0.993</td><td>7 (10)</td><td>0.76 / 1.16</td><td>0.81 / 0.05 / 0.05 / 0.08</td></tr><tr><td>13</td><td>0.49–1.22</td><td>0.50–1.29</td><td>0.51 / 0.53</td><td>0.75 / 0.80</td><td>3.06 / 2.52</td><td>1.000 / 0.999</td><td>9 (10)</td><td>0.57 / 1.44</td><td>0.88 / 0.04 / 0.03 / 0.06</td></tr><tr><td>14</td><td>0.42–0.68</td><td>0.39–0.62</td><td>0.64 / 0.67</td><td>0.83 / 0.81</td><td>4.12 / 3.07</td><td>1.000 / 0.999</td><td>9 (10)</td><td>0.54 / 1.60</td><td>0.85 / 0.04 / 0.07 / 0.03</td></tr><tr><td>15</td><td>0.27–0.35</td><td>0.22–0.29</td><td>0.75 / 0.74</td><td>0.97 / 0.97</td><td>3.89 / 2.69</td><td>1.000 / 0.997</td><td>9 (10)</td><td>0.60 / 1.72</td><td>0.84 / 0.03 / 0.07 / 0.06</td></tr><tr><td>16</td><td>0.32–0.50</td><td>0.24–0.38</td><td>0.77 / 0.84</td><td>0.95 / 0.97</td><td>3.52 / 2.24</td><td>1.000 / 0.994</td><td>9 (10)</td><td>0.95 / 1.91</td><td>0.79 / 0.08 / 0.07 / 0.06</td></tr></table></div>

<h3>5. Repository</h3>
<p>The representation repository now holds position 3 for blocks 0–11: the five block-0 planes <code>L0.p3.b_mod_10.k1</code>, …,
the residue-mean spans <code>L1.p3.b_mod_10.means</code> … <code>L11.p3.b_mod_10.means</code> with their circulant statistics, every
kept reader's position-3 response, the block-0 mod-10 readers' position-3 writers, and the transitions
<code>L0.p3.b_mod_10</code> … <code>L11.p3.b_mod_10</code> with this page's per-block measurements.</p>
`
},
];
