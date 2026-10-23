// Notebook pages of the "Mod 20 representations" notebook (index.html). Figures in fig/.
window.NOTEBOOK = [
{
id: "mod20",
title: "Mod 20 representation at a (position 1)",
date: "2026-10-22",
html: `
<p><b>Summary.</b> At the token <i>a</i>, the information that tells <i>a</i> mod 20 apart from <i>a</i> mod 10, that is,
whether the tens digit is even or odd, sits in a single plane of the residual stream: the frequency-1 Fourier plane of
<i>a</i> mod 20. No other frequency that separates <i>s</i> from <i>s</i> + 10 rises above a shuffled-label null at any block
from the embedding to block 16. In that plane the 20 residues lie on a ring around the activation zero, in order. At the
embedding the ring is nearly evenly spaced, with a larger gap where the tens digit changes (between 9 and 10, and
between 19 and 0). The MLP of block 12 turns it into four clumps of five residues, {0–4}, {5–9}, {10–14}, {15–19}: the
ring then carries two bits, the parity of the tens digit (<i>h</i>) and the parity of <i>a</i> rounded to the nearest ten
(<i>g</i>). The readers say the same thing. Up to block 12, the readers that respond to the plane read <i>h</i>; from
block 13 on, most read <i>g</i>. Block 12 holds the step between the two: its MLP multiplies a reading of the ring by a
gate that is open only when the units digit is 5–9, which is the product that converts <i>h</i> into <i>g</i>. In
blocks 0–11 every MLP moves the residue means within the plane by one 2 × 2 linear map, and in the components model it
does so through reader paths that are each linear in their own input. Blocks 12 and 14 are not linear in the plane; they
become close to linear once a separate linear map is allowed for each value of <i>a</i> mod 10. In the components model
the clouds blur in blocks 3–11 (their spread grows faster than the ring), and the MLPs of blocks 11–13 sharpen them
again. The same plane, the same ring-to-quarters change at block 12, the same switch of the readers from <i>h</i> to
<i>g</i> and the same gating are found in the original model's neuron basis, with matching reader profiles; the original
model blurs much less.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands <i>a</i>,
<i>b</i> ∈ {1, …, 400} and op ∈ {+, −}. Everything on this page is at position 1, the token <i>a</i>. Position 1 attends
only to &lt;BOS&gt; and to itself, so its residual stream depends on <i>a</i> alone, and 400 two-token forwards give
every value. Two models are compared, both rebuilt in float64 on the CPU through block 16:</p>
<ul>
<li><b>Components</b>: the alive-only model of section 1.3 of the mod-10 notebook (every alive attention and MLP
component of the ceiling filter <code>addsub-08-last-pos</code> on, the weight delta off; <code>track_build.py</code>). A
<i>reader</i> is a kept gate or up component of a block's MLP; its activation is its read vector applied to the
MLP input.</li>
<li><b>Neurons</b>: the original Llama-3.1-8B, with no decomposition (<code>track_build_orig.py</code>). A reader is one row of
<code>gate_proj</code> ("gate neuron <i>i</i>") or of <code>up_proj</code> ("up neuron <i>i</i>"), <i>i</i> ∈ {0, …, 14335}.
Its residual stream has mean cosine 0.93–0.96 with the components model's at every block.</li>
</ul>
<p>Nothing is centred anywhere on this page: the network has no biases, so zero is the zero of the activation space, and
every plot's origin is that zero.</p>
<p><b>Variables of a.</b> <i>s</i> = <i>a</i> mod 20 (integer 0–19; 20 values of <i>a</i> per residue) and <i>r</i> = <i>a</i> mod 10
(integer 0–9). <i>h</i> = ⌊<i>a</i>/10⌋ mod 2 is the parity of the tens digit: 0 for <i>s</i> ∈ 0–9, 1 for <i>s</i> ∈ 10–19.
<i>g</i> = ⌊(<i>a</i> + 5)/10⌋ mod 2 is the parity of <i>a</i> rounded to the nearest ten (halves rounded up): 0 for <i>s</i> ∈ 0–4
and 15–19, 1 for <i>s</i> ∈ 5–14. <i>Q</i> = ⌊<i>s</i>/5⌋ (integer 0–3) is the quarter of the 20-cycle; the pair (<i>h</i>, <i>g</i>)
is (0, 0), (0, 1), (1, 1), (1, 0) for <i>Q</i> = 0, 1, 2, 3, so <i>Q</i> and (<i>h</i>, <i>g</i>) carry the same information. Note
<i>g</i> = <i>h</i> XOR [<i>r</i> ≥ 5], where [·] is 1 when the condition holds and 0 otherwise. More generally, for an integer
<i>e</i> ∈ 0–9, <i>B<sub>e</sub></i> = ⌊(<i>a</i> + 10 − <i>e</i>)/10⌋ mod 2 is the binary variable that flips at <i>s</i> = <i>e</i>
and <i>s</i> = <i>e</i> + 10 ("edge <i>e</i>"); <i>B</i><sub>0</sub> = <i>h</i> and <i>B</i><sub>5</sub> = <i>g</i>.</p>
<p><b>Stream and Fourier plane.</b> <i>X</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream at one boundary: entering
block <i>ℓ</i> (<i>ℓ</i> = 0 is the embedding; "entering block 17" is the stream leaving block 16) or after block <i>ℓ</i>'s
attention. <i>μ<sub>s</sub></i> ∈ ℝ<sup>4096</sup> is the mean of <i>X</i> over the 20 <i>a</i> with <i>a</i> mod 20 = <i>s</i>. For an integer
frequency <i>j</i> ∈ 1–10, Ψ<sub><i>j</i></sub> = Σ<sub><i>s</i>=0…19</sub> <i>μ<sub>s</sub></i> exp(−2π i <i>j s</i>/20), with i the
imaginary unit; the exact period of frequency <i>j</i> is <i>τ</i> = 20/gcd(<i>j</i>, 20). The odd frequencies <i>j</i> = 1, 3, 5, 7, 9 are
exactly the ones in which residues <i>s</i> and <i>s</i> + 10 differ, so they hold everything <i>a</i> mod 20 adds to
<i>a</i> mod 10 (the even frequencies are the <i>a</i> mod 10 representation of the mod-10 page). null<sub><i>j</i></sub> is the 99th
percentile of ‖Ψ<sub><i>j</i></sub>‖ over 100 random relabellings of the 400 values of <i>a</i>. <i>P</i> (a 4096 × 2 matrix with
orthonormal columns) spans Re Ψ<sub>1</sub> and Im Ψ<sub>1</sub>: the frequency-1 plane of <i>a</i> mod 20.
<i>z</i>(<i>a</i>) = <i>P</i><sup>T</sup><i>X</i>(<i>a</i>) ∈ ℝ<sup>2</sup> and <i>m<sub>s</sub></i> = <i>P</i><sup>T</sup><i>μ<sub>s</sub></i> are
a prompt's and a residue's coordinates in it, and Π<sub><i>P</i></sub> = <i>PP</i><sup>T</sup> is the projection onto it.</p>
<p><b>Ring and cloud measures</b> (used from chapter 1 on; illustrated in chapter 2).
<i>o<sub>s</sub></i> = (<i>m<sub>s</sub></i> − <i>m</i><sub><i>s</i>+10</sub>)/2 ∈ ℝ<sup>2</sup> for <i>s</i> = 0–9, and <i>o</i><sub><i>s</i>+10</sub> = −<i>o<sub>s</sub></i>;
<i>φ<sub>s</sub></i> is its angle, unwrapped along <i>s</i>, and the step from <i>s</i> to <i>s</i> + 1 is <i>φ</i><sub><i>s</i>+1</sub> − <i>φ<sub>s</sub></i>
in degrees (18° on an evenly spaced ring). The quantization index <i>α</i> ∈ [0, 1] minimises the squared angular error of
<i>φ<sub>s</sub></i> ≈ <i>c</i> + (2π/20)(<i>α s</i> + (1 − <i>α</i>)·5⌊<i>s</i>/5⌋) over <i>α</i> and an offset angle <i>c</i>: 1 is the even ring, 0 is
four points, one per quarter. In the plane, <i>c<sub>s</sub></i> is the mean of <i>z</i> over residue <i>s</i> and <i>sp<sub>s</sub></i> the
root-mean-square distance of those <i>z</i> to <i>c<sub>s</sub></i>; the separation of residues <i>s</i>, <i>s′</i> is
‖<i>c<sub>s</sub></i> − <i>c<sub>s′</sub></i>‖/√((<i>sp<sub>s</sub></i>² + <i>sp<sub>s′</sub></i>²)/2) (the "antipodal" separation is that of <i>s</i> and
<i>s</i> + 10), and the same for the four quarters ("closest quarters": the smallest of their pairwise separations). Held-out
accuracies fit the plane and the class means on the <i>a</i> ≤ 200 half and test on the other half, and the reverse: <i>Q</i>
among 4 classes (chance 0.25), <i>s</i> among 20 (0.05), and <i>h</i> given <i>r</i> (choose between residues <i>r</i> and <i>r</i> + 10; 0.5).</p>

<h3>1. One plane carries what a mod 20 adds to a mod 10</h3>
<p>Figure 1 compares the strength of each frequency with what shuffled labels give. Of the five odd frequencies, only
<i>j</i> = 1 is above its null: ‖Ψ<sub>1</sub>‖/null<sub>1</sub> is 1.14–2.19 (components) and
1.48–2.20 (neurons) over the 18 boundaries, while <i>j</i> = 3, 5, 7, 9 stay at
0.40–0.66 (components) and 0.61–0.82 (neurons) of
their nulls from block 1 to block 15. At the embedding they reach 1.01, and leaving block 16
1.00 (components) and 1.08 (neurons). The period-20 information is therefore
in one plane, <i>P</i>.</p>
<p>A held-out linear probe checks that nothing linear is missed (figure 7d). An L2 logistic regression (inverse
regularisation 1) is trained to predict <i>h</i> on the prompts with <i>a</i> ≤ 200 and tested on <i>a</i> &gt; 200, and the
reverse; the plane used is fitted on the training half only. From the two coordinates <i>z</i> alone it reaches
0.67–0.94 accuracy (components) and
0.83–0.91 (neurons). From the whole 4096-dimensional stream it reaches
0.89–0.97 and 0.88–0.94.
From the stream with the plane projected out, <i>X</i> − Π<sub><i>P</i></sub><i>X</i>, it reaches only
0.54–0.67 and
0.57–0.67 (chance is 0.5). In the components model's blocks 5–12 the
whole stream beats the plane alone (for example 0.90 against
0.67 entering block 11), while the stream without the plane stays at
0.54: other directions do not carry <i>h</i> themselves but help the probe
remove the spread of the clouds in the plane (section 5).</p>
<figure><img src="fig/mod20_fourier.png"><figcaption>Figure 1. ‖Ψ<sub><i>j</i></sub>‖/null<sub><i>j</i></sub> for
every frequency <i>j</i> of <i>a</i> mod 20 (rows; <i>τ</i> its exact period) at the residual stream entering each block
(columns). White contour: 1, the 99th percentile of shuffled labels. Odd <i>j</i> (τ = 20 or 4) separate <i>s</i> from
<i>s</i> + 10; only <i>j</i> = 1 is above 1. Even <i>j</i> are the <i>a</i> mod 10, mod 5 and mod 2 representations.</figcaption></figure>

<h3>2. Geometry: a ring of the 20 residues, which turns into four clumps</h3>
<p>Figure 2 shows the 400 prompts in the plane at several blocks. The residues sit on a ring around the activation zero,
in the order 0, 1, …, 19, so that <i>s</i> and <i>s</i> + 10 are on opposite sides. Roundness (minor over major axis of the
ellipse traced by Re Ψ<sub>1</sub>, Im Ψ<sub>1</sub>) is 0.89 at the embedding.</p>
<figure><img src="fig/mod20_planes.png"><figcaption>Figure 2. Residual stream at position 1 projected onto the
frequency-1 plane <i>P</i> of its own boundary (top: components; bottom: neurons). Each dot is one <i>a</i> (400), coloured by
<i>s</i> = <i>a</i> mod 20 in four colour families, one per quarter <i>Q</i> (blue 0–4, orange 5–9, green 10–14, purple 15–19);
labels are the residue means <i>m<sub>s</sub></i>. The axes are turned so that the vector from residue 10's mean to residue 0's
mean points right. The cross is the activation zero.</figcaption></figure>
<p><b>How evenly the residues are spaced.</b> Let <i>o<sub>s</sub></i> = (<i>m<sub>s</sub></i> − <i>m</i><sub><i>s</i>+10</sub>)/2 ∈ ℝ<sup>2</sup>
for <i>s</i> = 0–9 and <i>o</i><sub><i>s</i>+10</sub> = −<i>o<sub>s</sub></i>: half the vector from a residue's antipode to the residue,
which removes whatever <i>s</i> and <i>s</i> + 10 share (it is the frequency-1 part, not a subtraction of the overall mean).
<i>φ<sub>s</sub></i> is its angle, unwrapped along <i>s</i>, and the step from <i>s</i> to <i>s</i> + 1 is
<i>φ</i><sub><i>s</i>+1</sub> − <i>φ<sub>s</sub></i> in degrees. An evenly spaced ring has every step at 18°. The quantization index
<i>α</i> ∈ [0, 1] is the value minimising the squared angular error of <i>φ<sub>s</sub></i> ≈ <i>c</i> + (2π/20)(<i>α s</i> + (1 − <i>α</i>)·5⌊<i>s</i>/5⌋)
over <i>α</i> and an offset angle <i>c</i>: <i>α</i> = 1 is the even ring, <i>α</i> = 0 is four points, one per quarter.</p>
<p>At the embedding, <i>α</i> = 0.86: steps are 14–17° inside each run of ten and
35° from 9 to 10 (and from 19 to 0), with 21° from 4 to 5. The embedding already
marks the change of tens digit. Through blocks 1–11 the components model keeps <i>α</i> at
0.71–0.75 (neurons 0.77–0.79 through block 12) and widens the
9 → 10 step to 57° (neurons 44°). The MLP of block 12 drops <i>α</i> from
0.66 to 0.44 (neurons 0.77 → 0.61), and it stays
there through block 13: entering block 14, <i>α</i> = 0.47 (neurons 0.58), the 4 → 5 step is
60° (neurons 57°), the 9 → 10 step 48°
(37°), and the steps inside a quarter shrink to 3–17°. Blocks 15–16 partly undo it (<i>α</i>
0.57 and 0.68 leaving block 16). Figure 3 shows every step at every block.</p>
<figure><img src="fig/mod20_ring.png"><figcaption>Figure 3. Left and centre: the angular step <i>φ</i><sub><i>s</i>+1</sub> −
<i>φ<sub>s</sub></i> (degrees) from residue <i>s</i> to <i>s</i> + 1 (columns) at the stream entering each block (rows; 17 =
leaving block 16); 18° everywhere would be an evenly spaced ring. The quarter edges are the columns 4→5, 9→10, 14→15, 19→0.
Right: the quantization index <i>α</i>.</figcaption></figure>

<h3>3. What variable is stored: h early, g late</h3>
<p>The plane holds a two-dimensional code, so which binary variable a reader extracts depends on the direction it reads.
The readers therefore show what the network uses. For each reader <i>c</i> (component, or neuron) of each block,
<i>t<sub>c</sub></i>(<i>a</i>) is its activation at the MLP input, <i>m<sub>c</sub></i>(<i>s</i>) its mean over the 20 <i>a</i> with
<i>a</i> mod 20 = <i>s</i>, and <i>o<sub>c</sub></i>(<i>s</i>) = (<i>m<sub>c</sub></i>(<i>s</i>) − <i>m<sub>c</sub></i>(<i>s</i> + 10))/2 for <i>s</i> = 0–9 its
odd part. The <b>odd share</b> is 20 Σ<sub><i>s</i></sub> <i>o<sub>c</sub></i>(<i>s</i>)² / Σ<sub><i>a</i></sub> <i>t<sub>c</sub></i>(<i>a</i>)² (uncentred). The odd
part of a ±1-coded <i>B<sub>e</sub></i> is the step <i>σ<sub>e</sub></i>(<i>s</i>) = −1 for <i>s</i> &lt; <i>e</i>, +1 for <i>s</i> ≥ <i>e</i>;
the <b>step r²<sub><i>e</i></sub></b> is the uncentred r² of <i>o<sub>c</sub></i> ≈ β<i>σ<sub>e</sub></i> with one coefficient β, and the best
edge is the <i>e</i> with the largest one (<code>mod20_readers.py</code>).</p>
<p>Figure 4 (top) counts the readers with odd share ≥ 0.05 and best step r² ≥ 0.8, by best edge. Only edges 0 (<i>h</i>) and
5 (<i>g</i>) are common. Components: readers of <i>h</i> appear in blocks 3, 4 and 12 (5 in block 12), readers
of <i>g</i> from block 13 on (17, 3, 16, 10 in blocks 13–16), and
only 4 readers in all have another edge. Neurons: 69 read <i>h</i> in blocks 0–12
against 14 for <i>g</i>; in blocks 13–16 the counts are 70 for <i>h</i> and
208 for <i>g</i>, 62 and 95 of them in blocks 14 and 15. The neuron basis has more
readers at every block (it is not sparse), but the same switch from <i>h</i> to <i>g</i> at block 13.</p>
<p>The bottom row of figure 4 pairs the largest component of a block with a neuron of the same block, chosen as the
largest contributor to that block's MLP write from the plane (section 4). Their profiles over <i>s</i> match. Block 3:
component u141 and up neuron 13082 are high for <i>s</i> ∈ 0–8 and low for 10–19 (<i>h</i>, with a smooth edge). Block 12:
component g11 and gate neuron 8953 fire for <i>s</i> ∈ 10–14 only, the quarter <i>Q</i> = 2 (<i>h</i> = 1 and <i>r</i> &lt; 5). Blocks 13
and 14: component g58 and gate neuron 10534, component g73 and gate neuron 3498, are steps at <i>s</i> = 5 and 15:
<i>g</i>.</p>
<figure><img src="fig/mod20_readers.png"><figcaption>Figure 4. Top: number of readers per block with odd share ≥
0.05 and best step r² ≥ 0.8, coloured by their best edge (left: kept gate and up components; right: all 28,672 gate and up
neurons). Bottom: mean activation over <i>s</i> of matched readers, each divided by its largest absolute value (solid:
component, dashed: neuron). Grey lines mark the quarter edges.</figcaption></figure>

<h3>4. How the blocks move the representation</h3>
<p><b>Per-sublayer quantities</b> (<code>mod20_track.py</code>). A sublayer <i>S</i> is one block's attention or MLP, including its
RMSNorm; it adds <i>S</i>(<i>X</i>) to the stream. Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>)(<i>a</i>)
is what <i>S</i> writes because its input contains the plane. The <b>gain</b> is ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖/‖Ψ<sub>1</sub>(<i>X</i>)‖,
with Ψ<sub>1</sub>(·) the frequency-1 Fourier vector of a field over <i>a</i>. <b>Share from <i>P</i></b> is
Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(Δ)⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖², the part of the period-20 output made from the plane;
<b>without <i>P</i></b> is ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>))‖/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖, the period-20 output the
sublayer still makes from other features of <i>a</i>. For r²(<i>Y</i>, <i>Ŷ</i>) = 1 − Σ<sub><i>a</i></sub>‖<i>Y</i> − <i>Ŷ</i>‖²/Σ<sub><i>a</i></sub>‖<i>Y</i>‖²
(uncentred): <b>r²<sub>lin</sub></b> is the held-out r² of Δ ≈ <i>Lz</i> with one 4096 × 2 matrix <i>L</i> (fitted on <i>a</i> ≤ 200,
tested on <i>a</i> &gt; 200, and the reverse); <b>r²<sub>gated</sub></b> is the same with one matrix <i>L<sub>r</sub></i> for each <i>r</i> = <i>a</i> mod 10.
For the MLP readers, <i>dt<sub>c</sub></i> = <i>t<sub>c</sub></i>(<i>X</i>) − <i>t<sub>c</sub></i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>) is the change of reader <i>c</i>'s
activation when the plane is removed, Δ<sub><i>c</i></sub> the change of the MLP output when only that reader changes, and
<i>κ<sub>c</sub></i> ∈ ℝ<sup>4096</sup> the least-squares slope of Δ<sub><i>c</i></sub> on <i>dt<sub>c</sub></i>; <i>κ<sub>c,r</sub></i> is the same slope within each
<i>r</i>. <b>Secant</b> is the r² of Δ ≈ Σ<sub><i>c</i></sub> <i>κ<sub>c</sub> dt<sub>c</sub></i> (every reader path linear in its own input),
<b>gated secant</b> the r² of Δ ≈ Σ<sub><i>c</i></sub> <i>κ</i><sub><i>c</i>, <i>a</i> mod 10</sub> <i>dt<sub>c</sub></i> (the slope may depend on the
units digit), and <b>additivity</b> the r² of Δ ≈ Σ<sub><i>c</i></sub> Δ<sub><i>c</i></sub> (no interaction between readers).</p>
<p><b>Attention does not act on it.</b> The attention gain is at most 0.04 (components) and
0.10 (neurons) in every block, as for <i>a</i> mod 10.</p>
<p><b>Block 0's MLP enlarges the ring and turns the plane.</b> Its gain is 1.79 (1.57),
and the plane turns: the mean principal cosine between the plane entering block 1 and the embedding's is
0.67 (0.70). Only
0.55 (0.37) of its period-20 output comes from the plane;
the rest is made from other features of the token. The residue means still move by a near-linear map of the plane: the
odd parts <i>o<sub>s</sub></i> after the block are fitted from those before with one 2 × 2 matrix at r² 1.00
(1.00).</p>
<p><b>Blocks 1–11: small rewrites, linear within the plane.</b> MLP gains are 0.14–0.78
(0.25–0.67). The residue means' odd parts after each MLP are fitted from those
before by one 2 × 2 matrix at r² ≥ 0.996. Per prompt, Δ is fitted
from <i>z</i> by one linear map at held-out r²<sub>lin</sub> 0.71–0.91
(0.45–0.72), and a map per units digit does not do better
(r²<sub>gated</sub> − r²<sub>lin</sub> ≤ 0.01). In the components model the reader paths are linear in their own input
(secant 0.84–0.97, additivity
1.00–1.00), and allowing a slope per units digit adds at most
0.03. In the neuron basis
the secant r² is lower (0.52–0.79) because each neuron's SiLU is curved
over the range the plane moves it, and the per-digit slope again adds little (at most
0.04). The plane drifts
slowly: by the stream entering block 12 the mean principal cosine with the embedding's plane is
0.36 (0.24).</p>
<p><b>Block 12: h times [units digit ≥ 5].</b> The MLP's gain rises to 0.90
(0.61), and 0.89 (0.86) of its
period-20 output comes from the plane. Its write is not linear in the plane, r²<sub>lin</sub> = 0.63
(0.61), but is close to linear for each units digit separately, r²<sub>gated</sub> =
0.90 (0.80). The reader decomposition says the same: secant
0.64 → gated secant 0.92 (neurons
0.57 → 0.81). In the components model three
readers carry 90% of Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²: g11 (the <i>Q</i> = 2 detector of figure 4), u57 and g7. In the neuron
basis gate neuron 8953 and up neuron 5212 carry the two largest shares. Figure 5 (left) shows how u57's write is gated.
Its activation follows the ring: highest for <i>s</i> ∈ 1–4, near zero for 5–9, lowest for 13–19 (the up neuron 5212:
positive for <i>s</i> ∈ 0–9, negative for 11–19). The size of its effect on the output, ‖(silu(<i>h</i><sub>gate</sub>) ⊙ <i>U</i><sub>up,57</sub>)<i>V</i><sub>down</sub><i>U</i><sub>down</sub>‖
(here <i>h</i><sub>gate</sub> ∈ ℝ<sup>14336</sup> is the vector of gate pre-activations, <i>U</i><sub>up,57</sub> component 57's write
into the 14,336 neurons, <i>V</i><sub>down</sub>, <i>U</i><sub>down</sub> the down components, ⊙ the elementwise product), is large when
<i>r</i> ∈ 5–9 and small when <i>r</i> ∈ 0–4. For the neuron, the slope is silu of its gate pre-activation: 0.36–0.87
for <i>r</i> ∈ 5–9 and -0.27–0.14 for <i>r</i> ∈ 0–4. A write proportional to (ring
reading) × [<i>r</i> ≥ 5] is what turns <i>h</i> into <i>g</i>: with ±1 coding, <i>g</i> = <i>h</i>·(1 − 2[<i>r</i> ≥ 5]).</p>
<p><b>Block 13: g is read, and the clouds tighten.</b> Across block 13, <i>α</i> barely moves, while the clouds
separate: held-out quarter accuracy 0.83 → 0.91
(0.82 → 0.90), closest quarters
2.08 → 2.85
(2.63 → 2.71). Block 13's own write is not linear in the plane either:
r²<sub>lin</sub> 0.46 (0.43), and per-digit maps do not help
(0.47, 0.45). Its largest component, g58 (65%
of the reader energy), reads <i>g</i> (figure 4); so does the largest neuron, 10534 (gate and up). In the neuron basis the
additivity of block 13 is -0.17: removing the plane changes both the gate and the up
pre-activation of neuron 10534, and the product of the two changes, which belongs to neither reader alone, is large. Summed
per neuron instead of per reader, the decomposition is exact by construction.</p>
<p><b>Block 14: g read again, gated by a units-digit set.</b> r²<sub>lin</sub> 0.46 →
r²<sub>gated</sub> 0.73 (neurons 0.42 → 0.64),
secant 0.49 → 0.78
(0.18 → 0.41). The largest component is up
reader u33 and the largest gated neuron path up neuron 6838; both activations are <i>g</i>-shaped (figure 5, right), and both
writes are gated by a function of the units digit that is largest for <i>r</i> ∈ {2, 3, 8}, smaller for 9 (component),
and largest for <i>r</i> ∈ {2, 3, 8, 9}, smaller for 5 (neuron).</p>
<figure><img src="fig/mod20_gating.png"><figcaption>Figure 5. The gated up readers of blocks 12 and 14. First and
third panels: per residue <i>s</i>, the mean size of the reader's write per unit of its activation (component: the norm of the
MLP output's derivative in <i>t<sub>c</sub></i>; neuron: silu of the same neuron's gate pre-activation). Second and fourth panels: the
reader's mean activation. Solid: component, dashed: neuron.</figcaption></figure>
<p><b>Blocks 15–16.</b> Gains 0.33–0.42 (0.39–0.42),
r²<sub>gated</sub> − r²<sub>lin</sub> = 0.06–0.08
(0.05–0.07); <i>α</i> rises back to
0.57 (0.68) leaving block 16, and the ring moves away from the activation zero (the
norm of the mean of <i>z</i> grows from 0.48 to 1.08 while the mean
residue radius goes from 1.07 to 1.43).</p>
<figure><img src="fig/mod20_sublayers.png"><figcaption>Figure 6. Per block (solid: components, dashed: neurons).
(a) Period-20 gain of the MLP and of attention. (b) Share of the MLP's period-20 output made from the plane, and the
period-20 output it makes without the plane. (c) Held-out r² of Δ from the plane's coordinates: one linear map, or one
per units digit. (d) Reader decomposition of Δ: secant, gated secant, additivity.</figcaption></figure>

<h3>5. Separation of the clouds</h3>
<p>Per boundary, in the plane: <i>c<sub>s</sub></i> is the mean of <i>z</i> over residue <i>s</i>, <i>sp<sub>s</sub></i> the root-mean-square
distance of those <i>z</i> to <i>c<sub>s</sub></i>, and the separation of residues <i>s</i>, <i>s′</i> is
‖<i>c<sub>s</sub></i> − <i>c<sub>s′</sub></i>‖/√((<i>sp<sub>s</sub></i>² + <i>sp<sub>s′</sub></i>²)/2); the same for the four quarters. η<sub><i>v</i></sub> is the
uncentred share of Σ<sub><i>a</i></sub>‖<i>z</i>(<i>a</i>)‖² explained by the class means of variable <i>v</i>. Accuracies are held out
(plane and class means fitted on one half of the <i>a</i>, tested on the other): <i>s</i> among 20 classes (chance 0.05), <i>Q</i>
among 4 (0.25), and <i>h</i> given <i>r</i> (choose between residues <i>r</i> and <i>r</i> + 10; 0.5).</p>
<p>Neighbouring residues are never well separated in this one plane (held-out accuracy for <i>s</i>
0.13–0.35); the units digit is carried by the mod-10 representation, and the plane
adds <i>h</i>. Antipodal residues are. At the embedding the median separation of <i>s</i> and <i>s</i> + 10 is
5.84. In the components model it falls to 1.45 entering block 11 (closest quarters
0.74; <i>h</i> given <i>r</i> 0.82): the residue radius
grows 0.073 → 0.426 while the spread grows
0.024 → 0.441, and η<sub><i>s</i></sub> drops from
0.90 to 0.43 (entering block 9). The neuron model blurs much less:
3.88 entering block 11, <i>h</i> given <i>r</i> 0.93, η<sub><i>s</i></sub>
≥ 0.81. The MLPs of blocks 11–13 restore it in both (block 11 brings the
components model's antipodal separation to 2.21): entering block 14 the
antipodal separation is 5.61 (6.42), <i>h</i> given <i>r</i>
0.99 (0.99), quarters
0.91 (0.90).</p>
<figure><img src="fig/mod20_clouds.png"><figcaption>Figure 7. Per boundary (solid: components, dashed: neurons).
(a) η<sub><i>s</i></sub> and η<sub><i>Q</i></sub>. (b) Median antipodal separation and closest pair of quarters. (c) Held-out nearest-mean
accuracies. (d) Held-out probe accuracy for <i>h</i> from the whole stream, the plane only, and the stream minus the plane.</figcaption></figure>

<h3>6. Block by block</h3>
<p>Columns per model: MLP gain, share from <i>P</i>, r²<sub>lin</sub>, r²<sub>gated</sub>, secant, gated secant and additivity of
the block's MLP (section 4); then <i>α</i>, held-out quarter accuracy and median antipodal separation of the stream leaving the
block.</p>
<table class='tight'><tr><th rowspan=2>block</th><th colspan=10>components (alive-only model)</th><th class='sep'></th><th colspan=10>neurons (original model)</th></tr><tr><th>gain</th><th>share from P</th><th>r²<sub>lin</sub></th><th>r²<sub>gated</sub></th><th>secant</th><th>gated secant</th><th>additivity</th><th>α after</th><th>acc<sub>Q</sub> after</th><th>sep(s, s+10) after</th><th class='sep'></th><th>gain</th><th>share from P</th><th>r²<sub>lin</sub></th><th>r²<sub>gated</sub></th><th>secant</th><th>gated secant</th><th>additivity</th><th>α after</th><th>acc<sub>Q</sub> after</th><th>sep(s, s+10) after</th></tr><tr><td>0</td><td>1.79</td><td>0.55</td><td>0.71</td><td>0.71</td><td>0.75</td><td>0.77</td><td>1.00</td><td>0.75</td><td>0.84</td><td>5.57</td><td class='sep'></td><td>1.57</td><td>0.37</td><td>0.28</td><td>0.25</td><td>0.34</td><td>0.38</td><td>0.98</td><td>0.78</td><td>0.78</td><td>5.31</td></tr><tr><td>1</td><td>0.14</td><td>0.73</td><td>0.76</td><td>0.76</td><td>0.91</td><td>0.91</td><td>1.00</td><td>0.75</td><td>0.81</td><td>4.51</td><td class='sep'></td><td>0.48</td><td>0.79</td><td>0.45</td><td>0.45</td><td>0.52</td><td>0.56</td><td>0.94</td><td>0.79</td><td>0.79</td><td>5.01</td></tr><tr><td>2</td><td>0.33</td><td>0.71</td><td>0.77</td><td>0.78</td><td>0.88</td><td>0.89</td><td>1.00</td><td>0.75</td><td>0.73</td><td>3.13</td><td class='sep'></td><td>0.52</td><td>0.75</td><td>0.45</td><td>0.44</td><td>0.56</td><td>0.60</td><td>0.99</td><td>0.79</td><td>0.75</td><td>4.42</td></tr><tr><td>3</td><td>0.78</td><td>0.92</td><td>0.91</td><td>0.92</td><td>0.92</td><td>0.95</td><td>1.00</td><td>0.73</td><td>0.70</td><td>3.12</td><td class='sep'></td><td>0.67</td><td>0.79</td><td>0.50</td><td>0.49</td><td>0.61</td><td>0.64</td><td>0.97</td><td>0.79</td><td>0.76</td><td>4.57</td></tr><tr><td>4</td><td>0.22</td><td>0.76</td><td>0.80</td><td>0.80</td><td>0.88</td><td>0.89</td><td>1.00</td><td>0.71</td><td>0.63</td><td>2.42</td><td class='sep'></td><td>0.51</td><td>0.81</td><td>0.54</td><td>0.54</td><td>0.65</td><td>0.68</td><td>0.98</td><td>0.78</td><td>0.76</td><td>4.22</td></tr><tr><td>5</td><td>0.20</td><td>0.66</td><td>0.71</td><td>0.71</td><td>0.84</td><td>0.85</td><td>1.00</td><td>0.71</td><td>0.63</td><td>2.14</td><td class='sep'></td><td>0.39</td><td>0.79</td><td>0.53</td><td>0.52</td><td>0.66</td><td>0.68</td><td>0.99</td><td>0.77</td><td>0.71</td><td>4.04</td></tr><tr><td>6</td><td>0.19</td><td>0.71</td><td>0.86</td><td>0.84</td><td>0.93</td><td>0.94</td><td>1.00</td><td>0.71</td><td>0.58</td><td>1.88</td><td class='sep'></td><td>0.43</td><td>0.83</td><td>0.64</td><td>0.63</td><td>0.74</td><td>0.76</td><td>0.99</td><td>0.79</td><td>0.66</td><td>4.08</td></tr><tr><td>7</td><td>0.17</td><td>0.90</td><td>0.86</td><td>0.80</td><td>0.96</td><td>0.96</td><td>1.00</td><td>0.72</td><td>0.56</td><td>1.62</td><td class='sep'></td><td>0.40</td><td>0.89</td><td>0.71</td><td>0.70</td><td>0.79</td><td>0.81</td><td>0.99</td><td>0.78</td><td>0.66</td><td>3.89</td></tr><tr><td>8</td><td>0.14</td><td>0.80</td><td>0.82</td><td>0.76</td><td>0.97</td><td>0.97</td><td>1.00</td><td>0.73</td><td>0.55</td><td>1.57</td><td class='sep'></td><td>0.28</td><td>0.88</td><td>0.72</td><td>0.71</td><td>0.79</td><td>0.81</td><td>1.00</td><td>0.77</td><td>0.66</td><td>4.01</td></tr><tr><td>9</td><td>0.15</td><td>0.79</td><td>0.86</td><td>0.83</td><td>0.95</td><td>0.95</td><td>1.00</td><td>0.72</td><td>0.55</td><td>1.52</td><td class='sep'></td><td>0.32</td><td>0.92</td><td>0.71</td><td>0.72</td><td>0.77</td><td>0.80</td><td>0.97</td><td>0.77</td><td>0.68</td><td>4.05</td></tr><tr><td>10</td><td>0.14</td><td>0.78</td><td>0.71</td><td>0.69</td><td>0.88</td><td>0.89</td><td>1.00</td><td>0.72</td><td>0.53</td><td>1.45</td><td class='sep'></td><td>0.25</td><td>0.87</td><td>0.72</td><td>0.72</td><td>0.79</td><td>0.81</td><td>0.99</td><td>0.78</td><td>0.65</td><td>3.88</td></tr><tr><td>11</td><td>0.76</td><td>0.66</td><td>0.79</td><td>0.77</td><td>0.86</td><td>0.86</td><td>1.00</td><td>0.66</td><td>0.54</td><td>2.21</td><td class='sep'></td><td>0.53</td><td>0.64</td><td>0.61</td><td>0.61</td><td>0.66</td><td>0.67</td><td>0.97</td><td>0.77</td><td>0.68</td><td>4.21</td></tr><tr><td>12</td><td>0.90</td><td>0.89</td><td>0.63</td><td>0.90</td><td>0.64</td><td>0.92</td><td>0.99</td><td>0.44</td><td>0.83</td><td>4.19</td><td class='sep'></td><td>0.61</td><td>0.86</td><td>0.61</td><td>0.80</td><td>0.57</td><td>0.81</td><td>0.96</td><td>0.61</td><td>0.82</td><td>5.40</td></tr><tr><td>13</td><td>0.53</td><td>0.64</td><td>0.46</td><td>0.47</td><td>0.55</td><td>0.58</td><td>0.97</td><td>0.47</td><td>0.91</td><td>5.61</td><td class='sep'></td><td>0.47</td><td>0.72</td><td>0.43</td><td>0.45</td><td>-0.25</td><td>-0.24</td><td>-0.17</td><td>0.58</td><td>0.90</td><td>6.42</td></tr><tr><td>14</td><td>0.52</td><td>0.79</td><td>0.46</td><td>0.73</td><td>0.49</td><td>0.78</td><td>0.97</td><td>0.52</td><td>0.90</td><td>5.58</td><td class='sep'></td><td>0.47</td><td>0.88</td><td>0.42</td><td>0.64</td><td>0.18</td><td>0.41</td><td>0.49</td><td>0.58</td><td>0.85</td><td>5.76</td></tr><tr><td>15</td><td>0.42</td><td>0.69</td><td>0.56</td><td>0.64</td><td>0.61</td><td>0.70</td><td>0.99</td><td>0.56</td><td>0.91</td><td>5.44</td><td class='sep'></td><td>0.42</td><td>0.81</td><td>0.47</td><td>0.54</td><td>0.53</td><td>0.62</td><td>0.97</td><td>0.65</td><td>0.84</td><td>5.42</td></tr><tr><td>16</td><td>0.33</td><td>0.68</td><td>0.57</td><td>0.63</td><td>0.69</td><td>0.76</td><td>1.00</td><td>0.57</td><td>0.88</td><td>4.69</td><td class='sep'></td><td>0.39</td><td>0.79</td><td>0.45</td><td>0.50</td><td>0.55</td><td>0.63</td><td>0.98</td><td>0.68</td><td>0.80</td><td>4.87</td></tr></table>

<h3>7. Components against neurons</h3>
<ul>
<li>One plane, the frequency-1 plane of <i>a</i> mod 20, holds the period-20 information at every block: both models (figure 1,
probes in figure 7d).</li>
<li>A ring that turns into four quarter clumps in blocks 12–13: both (figures 2–3; <i>α</i> 0.47 vs
0.58 entering block 14).</li>
<li>Readers of <i>h</i> up to block 12, of <i>g</i> from block 13: both (figure 4), with matched reader profiles in blocks 3,
12, 13 and 14.</li>
<li>Block 12 multiplies a ring reading by [<i>r</i> ≥ 5]; block 14 reads <i>g</i> gated by a units-digit set: both (figure 5).</li>
<li>Blocks 1–11 act through reader paths linear in their own input: components only (secant ≥
0.84); single neurons are curved (secant
0.52–0.79).</li>
<li>Blur of the clouds in blocks 3–11: much stronger in the components model.</li>
</ul>
<p>Scripts: <code>track_build_orig.py</code>, <code>mod20_track.py</code>, <code>mod20_readers.py</code>, <code>mod20_page.py</code>;
data in <code>&lt;run&gt;/analysis/reader_writer/track_mod20/</code>.</p>
`
},
];
