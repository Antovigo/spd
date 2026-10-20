// Page of the hosted "Mod 10 Neurons" notebook. Figures live in notebook/fig/ (published path).
window.NOTEBOOK = [
{
id: "mod10-neurons",
title: "Mod 10 representations in the neuron basis",
date: "2026-10-20",
html: `
<p><b>Summary.</b> The same 9-dimensional mod-10 subspace appears in the original model's block-0 MLP input. Its
five Fourier components (four planes and one line) are again circles around zero, and they again come from the
token embedding. Of the 28,672 block-0 neurons (14,336 gate rows and 14,336 up rows), 24 are explained by this
subspace with r² ≥ 0.5 (0.52–0.93). They belong to 15 neuron indices; for 9 of these indices, both the gate row and
the up row pass, and the two always pick the same residue. Every residue 0–9 has at least one neuron. Compared with
the decomposition's 11 mod-10 components, these neurons are less clean:</p>
<ul>
<li>they put only 3–32% of their weight norm² in the subspace, against 86–97% for ten of the components;</li>
<li>their off-residue responses are larger;</li>
<li>three of them mix residues, and one of these, u#1900, reads <i>a</i> mod 5.</li>
</ul>

<h3>Setup and notation</h3>
<p>The model is the original Llama-3.1-8B, with no decomposition. Its block-0 MLP input at position 1 (the token
<i>a</i> of <code>&lt;BOS&gt; a op b =</code>) depends only on <i>a</i>, so I ran 400 two-token forwards
(<code>&lt;BOS&gt;</code>, <i>a</i>) for <i>a</i> = 1 … 400. Each forward goes through the embedding, block-0
attention and the post-attention RMSNorm, in float32 from the bf16 weights. Check: projected onto the kept
components' read vectors, these inputs match the alive-only model's stored activations to 2.4% relative error,
with a median per-component correlation of 0.9998. <b>Nothing is centred or detrended.</b> Scripts:
<code>notes/reader_writer/readers/{neurons_l0_input,neuron_page}.py</code> (branch experiment/arith_representations).</p>
<ul>
<li><b>Operand and residue.</b> <i>a</i> ∈ {1, …, 400} is an integer, and <i>r</i> = <i>a</i> mod 10 ∈ {0, …, 9}.</li>
<li><b>Input.</b> <i>x</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the original model's block-0 MLP input at position 1.</li>
<li><b>Neurons.</b> A neuron reader <i>n</i> is a row <i>w<sub>n</sub></i> ∈ ℝ<sup>4096</sup> of <code>gate_proj</code>, written
<code>g#<i>i</i></code>, or of <code>up_proj</code>, written <code>u#<i>i</i></code>, for a neuron index <i>i</i> ∈ {0, …, 14335}.
Its inner activation is <i>f<sub>n</sub></i>(<i>a</i>) = <i>x</i>(<i>a</i>) · <i>w<sub>n</sub></i> / ‖<i>w<sub>n</sub></i>‖.</li>
<li><b>Residue means.</b> μ<sub><i>r</i></sub> is the mean of <i>x</i>(<i>a</i>) over the 40 values of <i>a</i> with
<i>a</i> mod 10 = <i>r</i>, and <i>m<sub>n</sub></i>(<i>r</i>) is the mean of <i>f<sub>n</sub></i> over the same values.</li>
<li><b>Fourier components.</b> <i>i</i> in exponents is the imaginary unit and <i>k</i> ∈ {1, …, 5} an integer frequency.
Ψ<sub><i>k</i></sub> = Σ<sub><i>r</i></sub> μ<sub><i>r</i></sub> exp(−2π<i>i</i><i>kr</i>/10). <i>P<sub>k</sub></i> is the
orthonormal basis of span{Re Ψ<sub><i>k</i></sub>, Im Ψ<sub><i>k</i></sub>}: a plane for <i>k</i> = 1 … 4, a line for
<i>k</i> = 5. <i>D</i> is the span of all five, 9 dimensions.</li>
<li><b>Projections and r².</b> Π<sub><i>X</i></sub> is the orthogonal projection onto a span <i>X</i>.
<i>g<sub>n</sub><sup>X</sup></i>(<i>a</i>) = Π<sub><i>X</i></sub><i>x</i>(<i>a</i>) · <i>w<sub>n</sub></i> / ‖<i>w<sub>n</sub></i>‖, and
r²<sub><i>X</i></sub>(<i>n</i>) = 1 − Σ<sub><i>a</i></sub>(<i>f<sub>n</sub></i> − <i>g<sub>n</sub><sup>X</sup></i>)² / Σ<sub><i>a</i></sub><i>f<sub>n</sub></i>², summed over
<i>a</i> = 1 … 400.</li>
<li><b>Display sign.</b> <i>s<sub>n</sub></i> ∈ {+1, −1} makes <i>s<sub>n</sub></i> <i>m<sub>n</sub></i> positive at the neuron's main
residue (the <i>r</i> with the largest |<i>m<sub>n</sub></i>(<i>r</i>)|). An up row's sign can be traded against the
matching <code>down_proj</code> column, but a gate row's sign is not arbitrary because SiLU is not odd. In the
event, every selected gate row already has <i>s<sub>n</sub></i> = +1; 6 up rows are flipped for display.</li>
</ul>

<h3>1. The representation</h3>
<figure><img src="notebook/fig/n_planes.png"><figcaption>Coordinates <i>P<sub>k</sub></i><sup>T</sup><i>x</i>(<i>a</i>) for every
<i>a</i> (small dots) and <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> (large dots, labelled by <i>r</i>). The black cross is
the zero of the activation space. For <i>k</i> = 5, the line coordinate is plotted against <i>a</i>.</figcaption></figure>
<p>As in the decomposition, <i>k</i> = 1 and 3 carry the full residue, <i>k</i> = 2 and 4 carry <i>a</i> mod 5, and
<i>k</i> = 5 carries <i>a</i> mod 2. The circles are centred close to zero. Here ‖centre‖ is the norm of the mean of
the ten <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub>, and the radius is their mean distance from that centre:</p>
<table><tr><th>k</th><th>‖centre‖</th><th>radius</th></tr>
<tr><td>1</td><td>0.103</td><td>0.956</td></tr><tr><td>2</td><td>0.078</td><td>0.979</td></tr>
<tr><td>3</td><td>0.050</td><td>0.647</td></tr><tr><td>4</td><td>0.049</td><td>0.755</td></tr>
<tr><td>5</td><td>0.015</td><td>0.711</td></tr></table>
<ul>
<li><i>D</i> holds 5.6% of ‖<i>x</i>(<i>a</i>)‖² on average over <i>a</i> (2.8–12.5%).</li>
<li>The planes are nearly orthogonal. The largest principal-angle cosine between two different planes is 0.28
(<i>k</i> = 2, 4), then 0.17 (<i>k</i> = 3, 4); every other pair is ≤ 0.09.</li>
<li><b>Same subspace as the decomposition's.</b> The nine principal-angle cosines between this <i>D</i> and the
decomposition's <i>D</i> are 0.89–0.94. The decomposition's <i>D</i> lies inside the span of the kept components'
read vectors, which holds about 84% of the embedding's mod-10 subspace.</li>
<li><b>Origin: the embedding.</b> Let <i>z</i>(<i>a</i>) = γ ⊙ <i>e</i>(<i>a</i>) / rms(<i>e</i>(<i>a</i>)), where
<i>e</i>(<i>a</i>) is the token's embedding row, γ the post-attention RMSNorm weight, ⊙ the elementwise product and
rms the root-mean-square over 4096 coordinates. Let <i>D</i><sub>ε</sub> be the span of the five Fourier components of
the residue means of <i>z</i>, built like <i>D</i>. The nine principal-angle cosines between <i>D</i> and
<i>D</i><sub>ε</sub> are 0.993–0.999.</li>
</ul>

<h3>2. The neurons it explains</h3>
<figure><img src="notebook/fig/n_screen.png"><figcaption>Left: r²<sub><i>D</i></sub> of the top 400 neurons, sorted, against
the same quantity for 5 random 9-dimensional subspaces of ℝ<sup>4096</sup> (largest over the draws). Right:
histogram of r²<sub><i>D</i></sub> over all 28,672 neurons.</figcaption></figure>
<p><b>Selection.</b> A neuron counts as a mod-10 neuron when r²<sub><i>D</i></sub> ≥ 0.5. That gives 24 neurons: 13
gate rows and 11 up rows. Across all neurons, the median r²<sub><i>D</i></sub> is 0.040 and the 99.9th percentile
is 0.38. The random-subspace control reaches at most 0.12. The next neurons below the threshold are u#11614
(0.49), u#5727 (0.46) and g#5727 (0.43).</p>
<p><b>Indices and residues.</b> The 24 neurons come from 15 neuron indices. For 9 indices, the gate and the up row
both pass, always with the same main residue: 0 (5524), 2 (5708), 3 (13010), 4 (8559), 5 (5095), 6 (7854),
7 (13630), 8 (8969) and 9 (10825). Every residue is covered: residues 0 and 5 by four neurons each, residue 3 by three, and every
other residue by one or two.</p>
<figure><img src="notebook/fig/n_profiles.png"><figcaption><i>s<sub>n</sub></i> <i>m<sub>n</sub></i>(<i>r</i>) /
max<sub><i>r</i></sub> |<i>m<sub>n</sub></i>(<i>r</i>)| for the 24 neurons (r²<sub><i>D</i></sub> in brackets), grouped by main
residue.</figcaption></figure>
<figure><img src="notebook/fig/n_traces.png"><figcaption>The 24 neurons, grouped by main residue <i>r</i> (left column
first): <i>f<sub>n</sub></i>(<i>a</i>) for <i>a</i> = 1 … 400 in the row's own sign (grey) and <i>g<sub>n</sub><sup>D</sup></i>(<i>a</i>)
(blue).</figcaption></figure>
<p><b>Cleanliness.</b> The neurons are less clean than the components. The second-largest |<i>s<sub>n</sub></i>
<i>m<sub>n</sub></i>(<i>r</i>)| / max is 0.29 at the median and 0.54 at the 90th percentile. For the components, 6 of 11
stay below 0.2 and the largest is 0.61. Most neurons' negative side lobes sit at the main residue ± 3, as for the
components (for example, g#13630 at 7 has −0.38 at 4). Three neurons mix residues:</p>
<ul>
<li>g#6837 has +1.0 at 5 and −0.88 at 8.</li>
<li>u#2147 has +1.0 at 3, then −0.55 at 0 and +0.55 at 5.</li>
<li>u#1900 has +1.0 at 5, +0.90 at 0 and −0.93 at 1. This is <i>a</i> mod 5 (residues 0 and 5 positive): it gets
r² 0.56 from the two mod-5 planes alone.</li>
</ul>

<h3>3. How much each neuron reads from each plane</h3>
<figure><img src="notebook/fig/n_per_plane.png"><figcaption>Left: ‖<i>P<sub>k</sub></i><sup>T</sup><i>w<sub>n</sub></i>‖² /
‖<i>w<sub>n</sub></i>‖², the share of the weight row in each plane <i>k</i>. Right: r²<sub><i>P<sub>k</sub></i></sub>(<i>n</i>) for each
plane alone (stacked bars) and r²<sub><i>D</i></sub>(<i>n</i>) (black ticks). The planes are not exactly orthogonal, so the
stacked values overlap and can add up to more than 1.</figcaption></figure>
<ul>
<li><b>Small share of the weight row in <i>D</i>.</b> The 24 neurons put 3–32% of ‖<i>w<sub>n</sub></i>‖² in
<i>D</i> (median 19%). The other 68–97% reads directions outside <i>D</i>.</li>
<li><b>All frequencies.</b> Neurons read from all five frequencies. The largest single-plane r² is 0.50, and the two
full-residue planes together give at most 0.42.</li>
<li><b>Read directions point at the residue.</b> For 21 of the 24 neurons, the cosine between
<i>s<sub>n</sub></i> <i>P<sub>k</sub></i><sup>T</sup><i>w<sub>n</sub></i> and <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> at the main residue is
0.87–1.00 in every plane <i>k</i> = 1 … 4. The three mixed neurons are the exceptions: g#6837 (−0.01 in <i>k</i> = 3),
u#2147 (0.02 in <i>k</i> = 3, 0.64 in <i>k</i> = 4) and u#1900 (0.15 in <i>k</i> = 1).</li>
</ul>
<figure><img src="notebook/fig/n_arrows.png"><figcaption>Residue means <i>P<sub>k</sub></i><sup>T</sup>μ<sub><i>r</i></sub> (dots, labelled by
<i>r</i>) and the 24 read directions <i>s<sub>n</sub></i> <i>P<sub>k</sub></i><sup>T</sup><i>w<sub>n</sub></i> / ‖<i>w<sub>n</sub></i>‖ (arrows from the
origin, coloured by main residue). Each panel's arrow scale is given in its title.</figcaption></figure>
<p><b>Lower-period neurons.</b> A neuron responds to a plane when that plane alone gives r² ≥ 0.05; this is the
representation repository's threshold. 2,785 neurons respond to at least one plane, and 1,495 respond only to the
lower-period planes (<i>k</i> = 2, 4, 5). Unlike the decomposition, one of them is strong: u#1900 (r² 0.56 from
<i>a</i> mod 5). The next are g#5727 (0.29), g#9679 (0.27) and u#9679 (0.23). The strongest parity-only neuron,
u#8062, gets r² 0.17.</p>

<h3>4. Do the neurons read with the dual basis of the residue means?</h3>
<p>This repeats the dual-basis test of the component page. The ten residue means μ<sub>0</sub> … μ<sub>9</sub> span a
10-dimensional space <i>M</i>: <i>D</i> plus one shared direction. A reader that responds 1 at residue
<i>r</i>₀ and 0 at the other residues has to read with the dual vector μ*<sub><i>r</i>₀</sub> rather than with
μ<sub><i>r</i>₀</sub>.</p>
<ul>
<li><b>Dual basis.</b> μ*<sub><i>r</i></sub> ∈ <i>M</i> are the ten vectors with μ*<sub><i>r</i></sub> · μ<sub><i>r′</i></sub> = 1 if <i>r</i> = <i>r′</i> and
0 otherwise. They are the rows of the pseudo-inverse of [μ<sub>0</sub> … μ<sub>9</sub>].</li>
<li><b>Projections and directions.</b> Π<sub><i>M</i></sub> is the orthogonal projection onto <i>M</i>, and <i>r</i>₀ is the neuron's main
residue. <i>u</i> is the unit vector along the part of the mean of the ten μ<sub><i>r</i></sub> that lies outside
<i>D</i>; it carries on average 78% of ‖μ<sub><i>r</i></sub>‖² and 0.3% of ‖μ*<sub><i>r</i></sub>‖².</li>
<li><b>Groups.</b> "Clean" neurons are the 19 whose main-residue activations separate from all other residues. The
five "mixed" neurons are g#6837, u#2147, u#1900, g#1179 and g#291.</li>
</ul>
<figure><img src="notebook/fig/n_dual.png"><figcaption>From left:
<ol>
<li>cosines of each neuron's sign-aligned weight row inside <i>M</i>, Π<sub><i>M</i></sub><i>s<sub>n</sub></i><i>w<sub>n</sub></i>, with every
residue mean μ<sub><i>r</i></sub>;</li>
<li>the same with every dual vector μ*<sub><i>r</i></sub>;</li>
<li>both cosines at the neuron's own residue;</li>
<li>each neuron's most negative side lobe, the minimum of <i>s<sub>n</sub></i><i>m<sub>n</sub></i>(<i>r</i>) / max |<i>m<sub>n</sub></i>| over
<i>r</i> ≠ <i>r</i>₀, against its dual cosine.</li>
</ol>
Neurons are grouped by main residue.</figcaption></figure>
<p><b>The part of each neuron in <i>M</i> is dual-like.</b></p>
<ul>
<li><b>Cosines at the main residue.</b> For the 19 clean neurons, the cosine of Π<sub><i>M</i></sub><i>s<sub>n</sub></i><i>w<sub>n</sub></i>
with μ*<sub><i>r</i>₀</sub> is 0.84–0.99, and it is ≥ 0.95 for ten of them. The cosine with μ<sub><i>r</i>₀</sub> is 0.38–0.61.
The components gave 0.85–1.00 and 0.46–0.60.</li>
<li><b>Mixed neurons.</b> They are further from the dual: g#291 0.90, g#1179 0.74, u#1900 0.71, g#6837 0.64 and
u#2147 0.53.</li>
<li><b>How much of each row this covers.</b> Only a small part of each neuron is in <i>M</i>:
‖Π<sub><i>M</i></sub><i>w<sub>n</sub></i>‖² / ‖<i>w<sub>n</sub></i>‖² is 0.14–0.32 for the clean neurons and 0.03–0.13 for the mixed ones,
against 0.87–0.97 for ten components.</li>
<li><b>Off the main residue.</b> The cosines with the other dual vectors are larger than for the components: the
largest |cosine| at <i>r</i> ≠ <i>r</i>₀ is 0.12–0.38 for the clean neurons.</li>
<li><b>Side lobes.</b> Across the clean neurons, the most negative side lobe (−0.42 to −0.05) grows as the dual
cosine falls; the correlation between the two is 0.53. Among the ten clean neurons with dual cosine ≥ 0.95, the
lobes are −0.30 to −0.05.</li>
</ul>
<figure><img src="notebook/fig/n_dual_split.png"><figcaption>Left: cosine with the shared direction <i>u</i> of
<i>s<sub>n</sub></i><i>w<sub>n</sub></i>, of μ<sub><i>r</i>₀</sub> and of μ*<sub><i>r</i>₀</sub>. Right: inside <i>D</i>, the cosine of
Π<sub><i>D</i></sub><i>s<sub>n</sub></i><i>w<sub>n</sub></i> with Π<sub><i>D</i></sub>μ<sub><i>r</i>₀</sub> and with Π<sub><i>D</i></sub>μ*<sub><i>r</i>₀</sub>.</figcaption></figure>
<p><b>As for the components, the difference is the shared direction.</b></p>
<ul>
<li>Every neuron's weight row is nearly orthogonal to <i>u</i>, with cosine −0.02 to 0.07. Against that, the cosine
with <i>u</i> is 0.85–0.89 for μ<sub><i>r</i>₀</sub> and 0.04–0.10 for μ*<sub><i>r</i>₀</sub>.</li>
<li>Inside <i>D</i>, the clean neurons are close to both: the cosine is 0.81–1.00 with Π<sub><i>D</i></sub>μ<sub><i>r</i>₀</sub> and
0.84–0.99 with Π<sub><i>D</i></sub>μ*<sub><i>r</i>₀</sub>. The residue mean is closer for 13 of them and the dual for 6 (g#7854,
g#13010, u#5708, u#8969, u#10825, u#13630).</li>
</ul>

<h3>5. Neurons against components</h3>
<p>For each of the decomposition's 11 mod-10 components, take the neuron row with the largest |cosine| to its read
vector. Every gate component's best match is a gate row with the same main residue (cosine 0.61–0.70). The same holds for
every up component except u31, whose best match is not a mod-10 neuron:</p>
<table><tr><th>component</th><th>best neuron</th><th>cos</th><th>neuron r²<sub>D</sub></th></tr>
<tr><td>g43 (r=1)</td><td>g#11614</td><td>0.61</td><td>0.84</td></tr>
<tr><td>g69 (r=2)</td><td>g#5708</td><td>0.67</td><td>0.84</td></tr>
<tr><td>g62 (r=4)</td><td>g#8559</td><td>0.65</td><td>0.86</td></tr>
<tr><td>g58 (r=7)</td><td>g#13630</td><td>0.70</td><td>0.93</td></tr>
<tr><td>g59 (r=8)</td><td>g#8969</td><td>0.66</td><td>0.90</td></tr>
<tr><td>g47 (r=9)</td><td>g#10825</td><td>0.67</td><td>0.89</td></tr>
<tr><td>u60 (r=3)</td><td>u#13010</td><td>0.66</td><td>0.78</td></tr>
<tr><td>u52 (r=5)</td><td>u#5095</td><td>0.60</td><td>0.74</td></tr>
<tr><td>u54 (r=6)</td><td>u#7854</td><td>0.59</td><td>0.60</td></tr>
<tr><td>u794 (r=0)</td><td>u#5524</td><td>0.39</td><td>0.63</td></tr>
<tr><td>u31 (r=0)</td><td>u#9757</td><td>0.32</td><td>0.02</td></tr></table>
<p>The cosines are well below 1. This fits the component read vectors lying almost entirely in <i>D</i> while the
neuron rows lie mostly outside it.</p>

<h3>6. What D contributes to the other neurons</h3>
<p>For each of the 28,648 other neurons, the change is ‖<i>g<sub>n</sub><sup>D</sup></i>‖ / ‖<i>f<sub>n</sub></i>‖, with norms over
<i>a</i> = 1 … 400, and the residual is ρ<sub><i>n</i></sub> = <i>f<sub>n</sub></i> − <i>g<sub>n</sub><sup>D</sup></i>. The change has median
0.20, 90th percentile 0.29, 99th percentile 0.37 and maximum 0.74 (u#11614). The next are g#11563 (0.69) and u#5727
(0.67).</p>
<figure><img src="notebook/fig/n_others_change.png"><figcaption>Histogram of the change over the other neurons.</figcaption></figure>
<figure><img src="notebook/fig/n_others_residual.png"><figcaption>The 24 other neurons with the largest change, in order (left
column first). Grey: <i>f<sub>n</sub></i>. Blue: ρ<sub><i>n</i></sub>. The horizontal axis is <i>a</i>.</figcaption></figure>
`
},
];
