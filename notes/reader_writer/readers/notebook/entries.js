// Notebook entries for the MLP readers applet (index.html, "Notebook" tab), oldest first.
// Figures live in notebook/fig/ next to this file; links are relative to index.html.
window.NOTEBOOK = [
{
id: "l0-mod10",
title: "Block-0 readers of a mod 10",
date: "2026-10-20",
html: `
<p><b>Short version.</b> Eleven of the 153 kept block-0 readers respond to <i>a</i> mod 10. Ten of them pick one
residue each, and together they cover all ten residues (residue 0 has two readers). The ten residue directions
fill a full 9-dimensional subspace: nothing lower-dimensional explains these readers (95% of their periodic
variance needs all 9 directions). The directions are not mutually orthogonal. Their Gram matrix depends only on the
residue difference, and its most negative entries are at a difference of ±3. In Fourier terms, all five frequencies
of period 10 are present, each on its own nearly orthogonal plane. Inside the space the readers can see, this
subspace is the token embedding's mod-10 structure: the principal-angle cosines are ≥ 0.9998. Removing it from the
other 142 readers changes them by a median of 15% (at most 70%), and leaves no mod-10 periodicity behind except in
one reader.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code> with integer operands
<i>a</i>, <i>b</i> ∈ {1, …, 400} and op ∈ {+, −}. The model is the <b>alive-only</b> model: every component of the
ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off.
Activations come from <code>notes/reader_writer/variance/extract_alive_only.py</code>, and the analysis script is
<code>notes/reader_writer/readers/mod10_L0.py</code> (branch experiment/arith_representations).</p>
<ul>
<li><b>Position.</b> Everything here is at token position 1, the token <i>a</i>. Causal attention means the block-0 MLP
input there depends on <i>a</i> only (measured: the maximum deviation across 8 random (<i>b</i>, op) pairs is 0).</li>
<li><b>Readers.</b> A reader <i>c</i> is a gate component (<code>g&lt;id&gt;</code>) or an up component (<code>u&lt;id&gt;</code>)
of block 0. This entry uses only the readers kept by the learned mask "kl_int, sampled mask, λ=5e-05", which leaves
K = 153 of them.</li>
<li><b>Vectors and activations.</b> <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup> is reader <i>c</i>'s read vector, and
<i>V</i> = [<i>V<sub>c</sub></i>] is the 4096 × K matrix. <i>x</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the block-0 MLP input
(after the post-attention RMSNorm). The plotted inner activation is
<i>f<sub>c</sub></i>(<i>a</i>) = <i>x</i>(<i>a</i>) · <i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖, a real number.</li>
<li><b>Projection onto span(V).</b> <i>x<sub>V</sub></i>(<i>a</i>) = <i>V</i>(<i>V</i><sup>T</sup><i>V</i>)<sup>+</sup><i>V</i><sup>T</sup><i>x</i>(<i>a</i>)
is the orthogonal projection of <i>x</i>(<i>a</i>) onto span(<i>V</i>), where <sup>+</sup> is the pseudo-inverse. It can
be computed from the stored activations, and every reader sees it exactly: <i>V<sub>c</sub></i> · <i>x<sub>V</sub></i> = <i>V<sub>c</sub></i> · <i>x</i>.
"v-space" in this entry means span(<i>V</i>).</li>
<li><b>Trend and periodic part.</b> The trend of a function of <i>a</i> is its moving average over the window
<i>a</i>−5 … <i>a</i>+4. Each such window contains every residue once, so the trend has no period-10 part. The
oscillating part is <i>p<sub>c</sub></i>(<i>a</i>) = <i>f<sub>c</sub></i>(<i>a</i>) − trend, taken on
<i>a</i> ∈ 𝒜 = {6, …, 395} (39 full periods).</li>
<li><b>Residue profile.</b> <i>r</i> = <i>a</i> mod 10 ∈ {0, …, 9}. <i>m<sub>c</sub></i>(<i>r</i>) is the mean of <i>p<sub>c</sub></i>
over the <i>a</i> ∈ 𝒜 with <i>a</i> mod 10 = <i>r</i>.</li>
<li><b>Periodic share.</b> <i>S</i><sub>10</sub>(<i>c</i>) = Σ<sub><i>a</i>∈𝒜</sub> <i>m<sub>c</sub></i>(<i>a</i> mod 10)² / Σ<sub><i>a</i>∈𝒜</sub> (<i>f<sub>c</sub></i>(<i>a</i>) − mean <i>f<sub>c</sub></i>)²
∈ [0, 1] is the share of <i>f<sub>c</sub></i>'s variance that repeats with period 10. <i>S<sub>q</sub></i> is the same
quantity for an integer modulus <i>q</i>, with a width-<i>q</i> trend.</li>
</ul>

<h3>1. Which readers respond to a mod 10</h3>
<p>A reader counts as a mod-10 reader when <i>S</i><sub>10</sub>(<i>c</i>) ≥ τ, with threshold τ = 0.5. As a control,
the largest <i>S<sub>q</sub></i> for <i>q</i> ∈ {7, 11, 13} over all 153 readers is 0.05, 0.09 and 0.08. For
<i>q</i> ∈ {9, 12} the maximum is 0.87 and 0.86, but this comes from one reader, g124, whose
<i>S</i><sub>10</sub> = 0.03.</p>
<figure><img src="notebook/fig/l0_s10_sorted.png"><figcaption>Readers sorted by <i>S</i><sub>10</sub> (bars). Dots: the
largest <i>S<sub>q</sub></i> over the control moduli <i>q</i> ∈ {7, 9, 11, 12, 13}. Dashed line: τ.</figcaption></figure>
<p><b>Result: 11 readers.</b> Ten readers have <i>S</i><sub>10</sub> between 0.97 and 0.98: g43, g47, g58, g59, g62, g69,
u52, u54, u60, u794. u31 is at 0.62. The next readers are below τ: g152 (0.45), u122 (0.42), u41 (0.33), and then
g175 (0.21).</p>
<figure><img src="notebook/fig/l0_examples.png"><figcaption><i>f<sub>c</sub></i>(<i>a</i>) for <i>a</i> = 1 … 400 (blue) and its
trend (grey), for u54, u60 and two other top readers.</figcaption></figure>
<p><b>Each reader picks one residue.</b> The table lists, for each reader, the largest |<i>m<sub>c</sub></i>(<i>r</i>)| / max<sub><i>r</i></sub> |<i>m<sub>c</sub></i>(<i>r</i>)|
and its residue <i>r</i>, then the second largest:</p>
<table><tr><th>reader</th><th>main residue (value)</th><th>second (value)</th></tr>
<tr><td>u794</td><td>0 (+1.00)</td><td>3, 7 (−0.12)</td></tr>
<tr><td>u31</td><td>0 (+1.00)</td><td>3 (−0.29), 7 (−0.24)</td></tr>
<tr><td>g43</td><td>1 (+1.00)</td><td>8 (−0.17)</td></tr>
<tr><td>g69</td><td>2 (+1.00)</td><td>5 (−0.69)</td></tr>
<tr><td>u60</td><td>3 (+1.00)</td><td>6 (−0.39)</td></tr>
<tr><td>g62</td><td>4 (+1.00)</td><td>7 (−0.13)</td></tr>
<tr><td>u52</td><td>5 (+1.00)</td><td>8 (−0.14)</td></tr>
<tr><td>u54</td><td>6 (−1.00)</td><td>9, 2 (+0.12)</td></tr>
<tr><td>g58</td><td>7 (+1.00)</td><td>4 (−0.29)</td></tr>
<tr><td>g59</td><td>8 (+1.00)</td><td>5 (−0.50)</td></tr>
<tr><td>g47</td><td>9 (+1.00)</td><td>6 (−0.43)</td></tr></table>
<p>Every residue 0 … 9 has a reader. u54 is the only one whose main response is negative: it dips at residue 6. When a
reader has a large second residue, that residue is always its main residue ± 3. Examples are u60 (3 and 6), g69
(2 and 5) and g59 (8 and 5). Section 2 shows that this follows from the geometry of the directions.</p>
<figure><img src="notebook/fig/l0_mod10_profiles.png"><figcaption><i>m<sub>c</sub></i>(<i>r</i>) / max<sub><i>r</i></sub> |<i>m<sub>c</sub></i>(<i>r</i>)| for the
11 readers (rows) and the residues <i>r</i> (columns).</figcaption></figure>

<h3>2. The directions they read</h3>
<p><b>Residue means.</b> For each residue <i>r</i>, μ<sub><i>r</i></sub> ∈ ℝ<sup>4096</sup> is the mean of <i>x<sub>V</sub></i>(<i>a</i>) − trend over the
<i>a</i> ∈ 𝒜 with <i>a</i> mod 10 = <i>r</i>. μ̄ is the mean of the ten μ<sub><i>r</i></sub>, and
<i>M</i> = [μ<sub>0</sub> … μ<sub>9</sub>] is the 4096 × 10 matrix. Every reader's profile is a read-out of these vectors:
<i>m<sub>c</sub></i>(<i>r</i>) = <i>V<sub>c</sub></i> · μ<sub><i>r</i></sub> / ‖<i>V<sub>c</sub></i>‖. μ̄ is close to zero:
‖μ̄‖ is 0.008 times the mean ‖μ<sub><i>r</i></sub>‖.</p>
<p><b>No subspace smaller than 9 dimensions.</b> The centred matrix <i>M</i> − μ̄ has rank at most 9, and here it has
9 non-zero singular values σ<sub><i>j</i></sub>, where <i>j</i> = 1 … 9 indexes them. They run from 2.30 down to 1.28,
and the smallest is 0.56 times the largest. Let <i>D<sub>n</sub></i> be the span of the top-<i>n</i> left singular
vectors, for an integer <i>n</i> from 0 to 9. Projecting <i>x<sub>V</sub></i>(<i>a</i>) − mean onto <i>D<sub>n</sub></i>
explains the following share of the 11 readers' variance:</p>
<table><tr><th>n</th><th>1</th><th>3</th><th>5</th><th>7</th><th>8</th><th>9</th></tr>
<tr><td>periodic part <i>p<sub>c</sub></i></td><td>0.19</td><td>0.44</td><td>0.69</td><td>0.85</td><td>0.92</td><td>0.98</td></tr>
<tr><td>whole <i>f<sub>c</sub></i></td><td>0.19</td><td>0.44</td><td>0.68</td><td>0.83</td><td>0.90</td><td>0.95</td></tr></table>
<p>The smallest <i>n</i> that reaches 95% of the periodic variance is <i>n</i>* = 9. Below, <i>D</i> = <i>D</i><sub>9</sub>
denotes this subspace and Π<sub><i>D</i></sub> the orthogonal projection onto it.</p>
<p><b>The residue directions are not orthogonal, and the pattern depends only on the residue difference.</b> The cosines
between centred residue means, cos(μ<sub><i>r</i></sub> − μ̄, μ<sub><i>r′</i></sub> − μ̄) for residues <i>r</i> ≠ <i>r′</i>, average
−0.110. That is the value −1/9 a regular simplex would give, but the entries are not equal. They range from −0.34 to
+0.07. Every pair with <i>r′</i> − <i>r</i> ≡ ±3 (mod 10) lies between −0.34 and −0.24 (mean −0.31), and every other
pair lies between −0.16 and +0.07. This is why a reader tuned to residue <i>r</i> responds negatively at <i>r</i> ± 3
(section 1).</p>
<figure><img src="notebook/fig/l0_mu_svd_cos.png"><figcaption>Left: share of ‖<i>M</i> − μ̄‖² (squared Frobenius norm)
carried by each singular direction <i>j</i>. Middle: variance of the 11 readers explained by <i>D<sub>n</sub></i>.
Right: cosines between centred residue means.</figcaption></figure>
<p><b>Fourier structure.</b> <i>i</i> is the imaginary unit. For an integer frequency <i>k</i> ∈ {1, …, 5}, define
Ψ<sub><i>k</i></sub> = Σ<sub><i>r</i>=0…9</sub> μ<sub><i>r</i></sub> exp(−2π<i>i</i><i>kr</i>/10) ∈ ℂ<sup>4096</sup>,
with real part Re Ψ<sub><i>k</i></sub> and imaginary part Im Ψ<sub><i>k</i></sub>. Frequency <i>k</i> = 5 is real (it is the
parity of <i>a</i>). Each <i>k</i> = 1 … 4 spans a plane. The share of the energy Σ<sub><i>r</i></sub>‖μ<sub><i>r</i></sub> − μ̄‖²
at each frequency is:</p>
<table><tr><th>k</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th></tr>
<tr><td>energy share</td><td>0.27</td><td>0.28</td><td>0.13</td><td>0.17</td><td>0.15</td></tr></table>
<p>All five frequencies are present, and none dominates. Within each plane, |cos(Re Ψ<sub><i>k</i></sub>, Im Ψ<sub><i>k</i></sub>)| ≤ 0.05
and ‖Re Ψ<sub><i>k</i></sub>‖ / ‖Im Ψ<sub><i>k</i></sub>‖ is between 1.02 and 1.39. So the ten residues sit on a circle
in each plane, visiting it at angle 2π<i>kr</i>/10. Across frequencies the nine vectors
{Re Ψ<sub>1…5</sub>, Im Ψ<sub>1…4</sub>} are nearly orthogonal: |cos| ≤ 0.27, with the largest value between Re Ψ<sub>2</sub>
and Re Ψ<sub>4</sub>. A circulant geometry like this predicts the cosine at residue difference 3 to be
Σ<sub><i>k</i></sub> (energy share)<sub><i>k</i></sub> · cos(2π·3<i>k</i>/10) = −0.30, which matches the measured mean −0.31.</p>
<figure><img src="notebook/fig/l0_fourier.png"><figcaption>Left: energy share per frequency <i>k</i>. Middle four: the
centred μ<sub><i>r</i></sub> projected onto the plane of Ψ<sub><i>k</i></sub> (orthonormalised Re, Im axes), labelled by
<i>r</i>. Right: |cos| between the nine real Fourier vectors.</figcaption></figure>
<p><b>The readers' vectors sit inside D and are close to orthogonal there.</b> For the ten one-residue readers,
‖Π<sub><i>D</i></sub><i>V<sub>c</sub></i>‖² / ‖<i>V<sub>c</sub></i>‖² is between 0.86 and 0.97. For u31 it is 0.38. Within
<i>D</i>, the cosines between projected read vectors are at most 0.42 in absolute value. The exception is the two
residue-0 readers, u31 and u794, whose cosine is 0.89.</p>
<figure><img src="notebook/fig/l0_reader_vectors.png"><figcaption>Left: share of each reader's ‖<i>V<sub>c</sub></i>‖² inside
<i>D</i> (its main residue in brackets). Right: cosines between the projected read vectors Π<sub><i>D</i></sub><i>V<sub>c</sub></i>.</figcaption></figure>
<p><b>The subspace comes from the token embedding.</b> Let <i>e</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> be Llama-3.1-8B's embedding
row of the single token for <i>a</i>, and γ ∈ ℝ<sup>4096</sup> the weight of block 0's post-attention RMSNorm. The MLP
input would be <i>z</i>(<i>a</i>) = γ ⊙ <i>e</i>(<i>a</i>) / rms(<i>e</i>(<i>a</i>)) if attention added nothing, where ⊙ is
the elementwise product and rms is the root-mean-square over the 4096 coordinates. Define ε<sub><i>r</i></sub> as the
residue mean of <i>z</i>(<i>a</i>) − trend, exactly as μ<sub><i>r</i></sub> is defined, and ε̄ as their mean. Let
Π<sub><i>V</i></sub> be the orthogonal projection onto span(<i>V</i>). Then:</p>
<ul>
<li>The principal-angle cosines between span{μ<sub><i>r</i></sub> − μ̄} and span{Π<sub><i>V</i></sub>(ε<sub><i>r</i></sub> − ε̄)}
are ≥ 0.9998 for all nine angles. Without the projection Π<sub><i>V</i></sub> they are 0.89 … 0.94. span(<i>V</i>) holds
83% of Σ<sub><i>r</i></sub>‖ε<sub><i>r</i></sub> − ε̄‖², so part of the embedding's mod-10 structure is invisible to the
kept readers.</li>
<li>For each of the 11 readers, the correlation over <i>r</i> between <i>m<sub>c</sub></i>(<i>r</i>) and
<i>V<sub>c</sub></i> · Π<sub><i>V</i></sub>ε<sub><i>r</i></sub> / ‖<i>V<sub>c</sub></i>‖ is ≥ 0.999998. The ratio of their norms is
0.951 … 0.954, a single factor shared by all 11 readers.</li>
</ul>
<figure><img src="notebook/fig/l0_embedding.png"><figcaption>Left: principal-angle cosines between the data subspace and
the projected embedding subspace (the 10th pairs the null directions of two rank-9 matrices and carries no information). Right: per-reader correlation
between the observed and embedding-predicted residue profiles.</figcaption></figure>

<h3>3. The other readers after removing D</h3>
<p>For each of the 142 kept block-0 readers outside the 11 above, the part along <i>D</i> is
<i>w<sub>c</sub></i>(<i>a</i>) = <i>V<sub>c</sub></i> · Π<sub><i>D</i></sub>(<i>x<sub>V</sub></i>(<i>a</i>) − mean<sub><i>a</i></sub> <i>x<sub>V</sub></i>) / ‖<i>V<sub>c</sub></i>‖.
The residual is ρ<sub><i>c</i></sub>(<i>a</i>) = <i>f<sub>c</sub></i>(<i>a</i>) − mean <i>f<sub>c</sub></i> − <i>w<sub>c</sub></i>(<i>a</i>), and the
change is ‖<i>w<sub>c</sub></i>‖ / ‖<i>f<sub>c</sub></i> − mean <i>f<sub>c</sub></i>‖, with norms taken over <i>a</i> = 1 … 400.
Note that <i>w<sub>c</sub></i> includes any non-periodic content of <i>x</i> that lies in <i>D</i>.</p>
<p><b>Result.</b> The change has median 0.15, 90th percentile 0.28 and maximum 0.70. The largest changes are g152 (0.70),
u122 (0.64) and u41 (0.59), which are the three readers just below τ in section 1. After the subtraction,
<i>S</i><sub>10</sub>(ρ<sub><i>c</i></sub>) ≤ 0.021 for 141 of the 142 readers. The exception is g140, at 0.157 (it was
0.164 before the subtraction, with change 0.095).</p>
<figure><img src="notebook/fig/l0_others_change.png"><figcaption>Change for the 142 other readers, sorted (the first 40
are labelled).</figcaption></figure>
<figure><img src="notebook/fig/l0_others_residual.png"><figcaption>The 24 other readers with the largest change, in
order (left column first). Grey: <i>f<sub>c</sub></i> − mean <i>f<sub>c</sub></i>. Blue: residual ρ<sub><i>c</i></sub>.
The horizontal axis is <i>a</i>.</figcaption></figure>
`
},
{
id: "l0-mod10-planes",
title: "Block-0 a mod 10: the Fourier planes",
date: "2026-10-20",
html: `
<p>This entry follows up on "Block-0 readers of a mod 10" and reuses its notation: <i>a</i>, <i>r</i>, <i>c</i>,
<i>V<sub>c</sub></i>, <i>x<sub>V</sub></i>(<i>a</i>), <i>f<sub>c</sub></i>, μ<sub><i>r</i></sub>, Ψ<sub><i>k</i></sub>, <i>D</i>
and Π. The 11 mod-10 readers are g43, g47, g58, g59, g62, g69, u31, u52, u54, u60 and u794. Script:
<code>notes/reader_writer/readers/mod10_planes.py</code>.</p>
<p><b>Planes and terminology.</b> <i>k</i> ∈ {1, …, 5} is the integer frequency of a period-10 function of <i>a</i>.
<i>k</i> = 1 is the fundamental and <i>k</i> = 2 … 5 are its harmonics, so "harmonic" is the correct word. <i>P<sub>k</sub></i>
is an orthonormal basis of span{Re Ψ<sub><i>k</i></sub>, Im Ψ<sub><i>k</i></sub>}: a plane for <i>k</i> = 1 … 4 and a line
for <i>k</i> = 5. Together the five spans give exactly <i>D</i> (9 dimensions). The smallest principal-angle cosine
between span{<i>P</i><sub>1…5</sub>} and span{μ<sub><i>r</i></sub> − μ̄} is 1.0000.</p>
<p><b>Not every plane carries the full residue.</b> In the plane of frequency <i>k</i>, residue <i>r</i> sits at angle
2π<i>kr</i>/10, so a plane can only tell apart residues that differ modulo 10/gcd(<i>k</i>, 10):</p>
<ul>
<li><i>k</i> = 1 and <i>k</i> = 3: each plane alone identifies all of <i>a</i> mod 10.</li>
<li><i>k</i> = 2 and <i>k</i> = 4: each identifies only <i>a</i> mod 5, so <i>r</i> and <i>r</i> + 5 coincide. The Ψ<sub>2</sub>
panel of the Fourier figure in the previous entry shows these pairs on top of each other.</li>
<li><i>k</i> = 5: identifies only <i>a</i> mod 2.</li>
</ul>
<p>So the full residue is encoded twice, in the planes <i>k</i> = 1 and <i>k</i> = 3. The readers do not use that
redundancy, though: each reader draws on all five frequencies (see the variance table below).</p>
<p><b>The planes are close to orthogonal.</b> For each pair of frequencies (<i>k</i>, <i>k′</i>), the figure shows the
largest principal-angle cosine between span <i>P<sub>k</sub></i> and span <i>P<sub>k′</sub></i>. Off the diagonal, the
values are ≤ 0.08 except for (2, 4) at 0.27 and (3, 4) at 0.17.</p>
<figure><img src="notebook/fig/l0_planes_cos.png"><figcaption>Largest principal-angle cosine between the frequency-<i>k</i>
and frequency-<i>k′</i> spans.</figcaption></figure>
<p><b>Projecting the input onto the planes.</b> Replace each reader's input <i>x<sub>V</sub></i>(<i>a</i>) by its projection
onto <i>D</i>, or onto a subset of the planes, and compare the resulting activation with <i>f<sub>c</sub></i>. There are
two versions.</p>
<p><b>Centred version.</b> Here the replaced activation is <i>V<sub>c</sub></i> · Π<sub><i>X</i></sub>(<i>x<sub>V</sub></i>(<i>a</i>) − mean<sub><i>a</i></sub> <i>x<sub>V</sub></i>) / ‖<i>V<sub>c</sub></i>‖ + mean <i>f<sub>c</sub></i>,
where Π<sub><i>X</i></sub> is the orthogonal projection onto the span of the planes in subset <i>X</i>. The table gives the
share of Var<sub><i>a</i></sub> <i>f<sub>c</sub></i> it reproduces, over <i>a</i> = 1 … 400:</p>
<table><tr><th>X</th><th>g43</th><th>g47</th><th>g58</th><th>g59</th><th>g62</th><th>g69</th><th>u31</th><th>u52</th><th>u54</th><th>u60</th><th>u794</th></tr>
<tr><td>all (<i>D</i>)</td><td>0.99</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.70</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.95</td></tr>
<tr><td><i>k</i>=1</td><td>0.27</td><td>0.26</td><td>0.22</td><td>0.24</td><td>0.22</td><td>0.22</td><td>0.11</td><td>0.19</td><td>0.22</td><td>0.24</td><td>0.27</td></tr>
<tr><td><i>k</i>=2</td><td>0.21</td><td>0.36</td><td>0.25</td><td>0.28</td><td>0.17</td><td>0.36</td><td>0.43</td><td>0.27</td><td>0.18</td><td>0.23</td><td>0.36</td></tr>
<tr><td><i>k</i>=3</td><td>0.26</td><td>0.14</td><td>0.08</td><td>0.07</td><td>0.19</td><td>0.02</td><td>0.11</td><td>0.07</td><td>0.19</td><td>0.10</td><td>0.37</td></tr>
<tr><td><i>k</i>=4</td><td>0.15</td><td>0.15</td><td>0.14</td><td>0.14</td><td>0.13</td><td>0.11</td><td>0.40</td><td>0.24</td><td>0.13</td><td>0.10</td><td>0.42</td></tr>
<tr><td><i>k</i>=5</td><td>0.11</td><td>0.17</td><td>0.15</td><td>0.16</td><td>0.09</td><td>0.17</td><td>0.10</td><td>0.09</td><td>0.12</td><td>0.17</td><td>0.15</td></tr>
<tr><td><i>k</i>=1,3</td><td>0.50</td><td>0.42</td><td>0.35</td><td>0.33</td><td>0.40</td><td>0.28</td><td>0.21</td><td>0.23</td><td>0.38</td><td>0.34</td><td>0.60</td></tr></table>
<p>With all of <i>D</i>, ten readers keep 95–99% of their variance, and u31 keeps 70%. No single plane gives more than
0.43, and the two planes that each carry the full residue (<i>k</i> = 1, 3) give 0.21–0.60 together. Each reader's single peak at
one residue is a sum over all five harmonics.</p>
<p><b>Raw version.</b> Here the replaced activation is <i>V<sub>c</sub></i> · Π<sub><i>D</i></sub> <i>x<sub>V</sub></i>(<i>a</i>) / ‖<i>V<sub>c</sub></i>‖:
everything outside <i>D</i> is discarded, including the mean input. The mean input mean<sub><i>a</i></sub> <i>x<sub>V</sub></i>
has only 0.19% of its squared norm in <i>D</i>. As a result, each reader's activation shifts by a constant of −0.50 to
+0.29 standard deviations of <i>f<sub>c</sub></i>, while its correlation with <i>f<sub>c</sub></i> over <i>a</i> stays at
0.98–0.99 (u31: 0.84).</p>
<figure><img src="notebook/fig/l0_planes_ev.png"><figcaption>Centred version: share of Var <i>f<sub>c</sub></i>
reproduced, per reader and plane subset <i>X</i>.</figcaption></figure>
<figure><img src="notebook/fig/l0_planes_overlay.png"><figcaption><i>f<sub>c</sub></i>(<i>a</i>) for <i>a</i> = 1 … 100 (grey), the
centred version (blue) and the raw version (red, dashed).</figcaption></figure>
`
},
];
