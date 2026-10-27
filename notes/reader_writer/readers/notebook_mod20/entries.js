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
<p>Scripts: <code>track_build_orig.py</code>, <code>mod20_track.py</code>, <code>mod20_readers.py</code>,
<code>mod20_b12_fig.py</code>, <code>mod20_geometry.py</code>, <code>mod20_page.py</code>; data in
<code>&lt;run&gt;/analysis/reader_writer/track_mod20/</code>.</p>
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
<p>Each block adds two things to the stream at the token <i>a</i>: its attention output and its MLP output. This section
goes through the blocks and asks four questions about each addition. Does it change the mod-20 plane <i>P</i> at all? Is the
change caused by the plane itself, or made from other information about <i>a</i>? Is it a simple linear map of the plane,
or does it depend on the units digit? And which components do it?</p>

<p><b>How each question is measured</b> (<code>mod20_track.py</code>, <code>mod20_components.py</code>). <i>X</i>(<i>a</i>) is the
stream entering the sublayer and <i>S</i>(<i>X</i>)(<i>a</i>) ∈ ℝ<sup>4096</sup> what the sublayer adds (<i>S</i> includes the block's
RMSNorm). For any quantity <i>f</i>(<i>a</i>) over the operands, Ψ<sub>1</sub>(<i>f</i>) is the frequency-1 Fourier coefficient of its 20
residue means (section 0.2): its period-20 content.</p>
<ul>
<li><b>Does it change the plane?</b> The <b>gain</b> ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖/‖Ψ<sub>1</sub>(<i>X</i>)‖ compares the period-20 signal the
sublayer writes with the period-20 signal already in the stream. A gain of 0.5 means it writes half as much as is there.</li>
<li><b>Is the change caused by the plane?</b> Run the sublayer twice: on <i>X</i>, and on <i>X</i> − Π<sub><i>P</i></sub><i>X</i>, the same stream with
the plane removed. The difference Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>)(<i>a</i>) is what the sublayer
writes because the plane is there. The <b>share from <i>P</i></b>, Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(Δ)⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖², is
the fraction of the sublayer's period-20 output that disappears when the plane is removed.</li>
<li><b>Is it linear in the plane?</b> Try to predict Δ(<i>a</i>) from the two plane coordinates <i>z</i>(<i>a</i>) with one fixed
4096 × 2 matrix, fitted on the operands <i>a</i> ≤ 200 and scored on <i>a</i> &gt; 200 (and the reverse). The score <b>r²<sub>lin</sub></b> is
1 − Σ<sub><i>a</i></sub>‖Δ − prediction‖² / Σ<sub><i>a</i></sub>‖Δ‖² (uncentred). <b>r²<sub>gated</sub></b> allows a different matrix for each units digit
<i>r</i> = <i>a</i> mod 10. When r²<sub>gated</sub> is clearly above r²<sub>lin</sub>, the block does something different to the plane for
different units digits.</li>
<li><b>Which components?</b> <i>Readers</i> (gate and up components): change one reader's activation as the plane's removal would,
keep the others fixed, and measure how much of Δ that produces; a reader's <b>share</b> is its part of the sum over readers of
Σ<sub><i>a</i></sub>‖Δ<sub>reader</sub>‖². Each listed reader is labelled with what it reads, from the scan of section 3.1: <i>h</i>
(tens parity), <i>g</i> (parity of <i>a</i> rounded to the nearest ten), "the ring" (a smooth period-20 profile), or "little period-20
signal" (odd share below 0.05). <i>Writers</i> (down components): writer <i>w</i> adds <i>t<sub>w</sub></i>(<i>a</i>) <i>U<sub>w</sub></i> to the stream, and its
<b>share of the period-20 output</b> is Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(<i>t<sub>w</sub></i>) <i>U<sub>w</sub></i>⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖²; the shares sum to
1 over writers and can be negative (a writer that shrinks the ring).</li>
<li><b>Reader-level linearity</b> (used in the table of section 2.3 and figure 6d). <b>Secant</b>: the r² of Δ when each reader's
contribution is replaced by a fixed multiple of its own activation change; high when every reader path is linear on its own.
<b>Gated secant</b>: the same with one multiple per units digit. <b>Additivity</b>: the r² of Δ by the plain sum of the single-reader
effects; below 1 when readers interact.</li>
</ul>
<p>Numbers in brackets are the original model's (neurons); the component lists are for the components model.</p>

<p><b>Attention leaves the plane alone.</b> In every block the attention's gain is at most 0.04
(0.10). Everything below is done by the MLPs.</p>

<p><b>Block 0: the ring grows and turns.</b> The MLP writes 1.79 times (1.57) the
period-20 signal it receives, and the plane turns: its mean principal cosine with the embedding's plane is
0.67 (0.70) after the block. Only
0.55 (0.37) of that output is caused by the plane: block 0 builds
most of the new ring from other features of the token. The work is spread thin: no reader carries more than
9% of the plane-driven write, and 28 readers carry 90% of it. Components:
readers g70 (9%, little period-20 signal), g76 (9%, little period-20 signal), u77 (7%, little period-20 signal), g55 (7%, little period-20 signal); writers d53 (+6%), d68 (+6%), d70 (+6%).</p>

<p><b>Blocks 1–11: small linear touch-ups.</b> Each MLP writes 0.14–0.78
(0.25–0.67) of the period-20 signal it receives. Its effect is a linear map of the plane:
Δ is predicted from the two plane coordinates at held-out r² 0.71–0.91
(0.45–0.72), and one map per units digit does no better (at most
0.01 higher). In the components model each
reader path is itself linear (secant 0.84–0.97) and the paths add up
(additivity ≥ 0.997); single neurons are more curved (secant
0.52–0.79). The plane drifts slowly: entering block 12, its mean principal
cosine with the embedding's plane is 0.36 (0.24).
The two largest writes are in blocks 3 and 11 (gains 0.78 and 0.76). Block 3's is
almost one reader, u141 (79%), which reads <i>h</i> (figure 4); its components: readers u141 (79%, reads <i>h</i>), g255 (3%, reads the ring), u281 (1%, little period-20 signal), g285 (1%, little period-20 signal); writers d141 (+70%), d201 (+4%), d285 (+3%).
Block 11's: readers g24 (22%, little period-20 signal), g17 (17%, little period-20 signal), g30 (11%, little period-20 signal), u86 (9%, little period-20 signal); writers d24 (+19%), d17 (+16%), d30 (+16%). The components of every other block are in the table below.</p>

<p><b>Block 12: the ring becomes four quarters.</b> This MLP turns the ring into the four clumps of section 2.1. Its gain is
0.90 (0.61), and 0.89
(0.86) of its period-20 output is caused by the plane. The write is not one linear map of the plane
(r²<sub>lin</sub> 0.63, 0.61) but is close to one map per units digit
(r²<sub>gated</sub> 0.90, 0.80). Three components carry 90% of it:</p>
<ul>
<li>u57 (21%) reads the plane (in-plane value 0.66, section 3.2) and feeds neuron 5212's up input. That neuron's gate opens only
for units digits 5–9, so u57's write is multiplied by [<i>r</i> ≥ 5]: figure 5, left, shows the size of its write per unit of
activation, large for <i>r</i> = 5–9 and small for <i>r</i> = 0–4. In the original model, the same neuron's silu(gate) is
0.36–0.87 for <i>r</i> = 5–9 and
−0.27 to 0.14 for <i>r</i> = 0–4. A ring reading times [<i>r</i> ≥ 5] is the
product that turns <i>h</i> into <i>g</i>: in ±1 coding, <i>g</i> = <i>h</i>·(1 − 2[<i>r</i> ≥ 5]).</li>
<li>g11 (69%) fires only for <i>s</i> = 10–14 (figure 4) and drives neuron 8953's gate,
which writes a vector only for that quarter.</li>
<li>g7 (1%).</li>
</ul>
<p>Sections 3.2 and 4.2 draw these readers in the plane. Writers: d11 (+58%), d57 (+19%), d7 (+5%), d16 (+4%).</p>

<p><b>Block 13: <i>g</i> is read and the clouds tighten.</b> The quarter structure stays (<i>α</i> barely moves), and the clouds
separate: held-out quarter accuracy goes from 0.83 to 0.91
(0.82 → 0.90). The write is not linear in the plane
(r²<sub>lin</sub> 0.46, 0.43) and a map per units digit does not help
(0.47, 0.45). One reader dominates: g58
(65%), which reads <i>g</i>; in the original model neuron 10534 (gate and up) does the same.
Removing the plane changes both the gate and the up input of neuron 10534, and the product of the two changes belongs to
neither reader alone, which is why the neuron-basis additivity is -0.17. Components:
readers g58 (65%, reads <i>g</i>), g6 (4%, little period-20 signal), u34 (4%, reads the ring), g21 (2%, little period-20 signal); writers d6 (+13%), d21 (+10%), d34 (+10%).</p>

<p><b>Block 14: <i>g</i> read again, gated by the units digit.</b> r²<sub>lin</sub> 0.46 →
r²<sub>gated</sub> 0.73 (0.42 → 0.64). The
largest reader, u33 (49%), and the largest neuron path, up neuron 6838, both read <i>g</i>
(figure 5, right); their writes are largest for units digits 2, 3 and 8 (component) and 2, 3, 8 and 9 (neuron). Components:
readers u33 (49%, reads <i>g</i>), g73 (19%, reads <i>g</i>), g69 (4%, reads the ring), g67 (3%, little period-20 signal); writers d33 (+30%), d67 (+5%), d69 (+4%).</p>
<figure><img src="fig/mod20_gating.png"><figcaption>Figure 5. The gated up readers of blocks 12 and 14. First and
third panels: per residue <i>s</i>, the mean size of the reader's write per unit of its activation (component: the norm of the
MLP output's derivative in <i>t<sub>c</sub></i>; neuron: silu of the same neuron's gate pre-activation). Second and fourth panels: the
reader's mean activation. Solid: component, dashed: neuron.</figcaption></figure>

<p><b>Blocks 15–16: the quarters loosen and the ring drifts off zero.</b> Gains 0.33–0.42
(0.39–0.42); the per-digit maps add 0.06–0.08
(0.05–0.07) to r². <i>α</i> rises back to
0.57 (0.68) leaving block 16, and the ring moves away from the activation zero (the
distance of the plane coordinates' mean from zero grows from 0.48 to 1.08
while the mean residue radius goes from 1.07 to 1.43). The work is
spread over more readers (23 and 33 for 90%). Block 15:
readers g123 (14%, reads <i>g</i>), g142 (9%, little period-20 signal), u41 (8%, reads the plane), g70 (7%, little period-20 signal); writers d123 (+13%), d142 (+8%), d497 (+6%). Block 16: readers u256 (8%, little period-20 signal), u216 (7%, reads the ring), g795 (6%, little period-20 signal), g149 (6%, little period-20 signal); writers d252 (+8%), d186 (+6%), d344 (+5%).</p>

<p><b>All blocks.</b> Gain, linearity and the main components of every MLP (components model):</p>
<table><tr><th>block</th><th>gain</th><th>r²<sub>lin</sub></th><th>r²<sub>gated</sub></th><th>components (readers: share of the plane-driven write, what they read; writers: share of the period-20 output)</th></tr><tr><td>0</td><td>1.79</td><td>0.71</td><td>0.71</td><td>readers g70 (9%, little period-20 signal), g76 (9%, little period-20 signal), u77 (7%, little period-20 signal), g55 (7%, little period-20 signal); writers d53 (+6%), d68 (+6%), d70 (+6%)</td></tr><tr><td>1</td><td>0.14</td><td>0.76</td><td>0.76</td><td>readers u223 (28%, little period-20 signal), u61 (9%, little period-20 signal), g166 (9%, little period-20 signal), g116 (9%, little period-20 signal); writers d234 (+32%), d109 (+16%), d937 (+12%)</td></tr><tr><td>2</td><td>0.33</td><td>0.77</td><td>0.78</td><td>readers u277 (17%, reads the ring), g335 (13%, little period-20 signal), u67 (13%, little period-20 signal), u77 (11%, little period-20 signal); writers d77 (+8%), d327 (+7%), d18 (+7%)</td></tr><tr><td>3</td><td>0.78</td><td>0.91</td><td>0.92</td><td>readers u141 (79%, reads <i>h</i>), g255 (3%, reads the ring), u281 (1%, little period-20 signal), g285 (1%, little period-20 signal); writers d141 (+70%), d201 (+4%), d285 (+3%)</td></tr><tr><td>4</td><td>0.22</td><td>0.80</td><td>0.80</td><td>readers u52 (24%, little period-20 signal), u94 (13%, little period-20 signal), g27 (9%, little period-20 signal), g101 (7%, little period-20 signal); writers d94 (+11%), d85 (+11%), d87 (+10%)</td></tr><tr><td>5</td><td>0.20</td><td>0.71</td><td>0.71</td><td>readers u40 (14%, little period-20 signal), u33 (12%, little period-20 signal), u307 (11%, little period-20 signal), g138 (6%, little period-20 signal); writers d273 (+13%), d101 (+11%), d33 (+10%)</td></tr><tr><td>6</td><td>0.19</td><td>0.86</td><td>0.84</td><td>readers u185 (21%, little period-20 signal), u545 (10%, little period-20 signal), u125 (9%, little period-20 signal), u27 (8%, little period-20 signal); writers d185 (+20%), d311 (+14%), d576 (+14%)</td></tr><tr><td>7</td><td>0.17</td><td>0.86</td><td>0.80</td><td>readers u198 (17%, little period-20 signal), u286 (16%, little period-20 signal), g0 (12%, little period-20 signal), u828 (10%, little period-20 signal); writers d28 (+33%), d101 (+25%), d432 (+13%)</td></tr><tr><td>8</td><td>0.14</td><td>0.82</td><td>0.76</td><td>readers g44 (30%, little period-20 signal), g109 (25%, little period-20 signal), u81 (11%, little period-20 signal), u660 (9%, little period-20 signal); writers d0 (+28%), d109 (+24%), d44 (+20%)</td></tr><tr><td>9</td><td>0.15</td><td>0.86</td><td>0.83</td><td>readers u33 (47%, little period-20 signal), g692 (10%, little period-20 signal), u115 (10%, little period-20 signal), u131 (8%, little period-20 signal); writers d33 (+30%), d96 (+15%), d875 (+13%)</td></tr><tr><td>10</td><td>0.14</td><td>0.71</td><td>0.69</td><td>readers g195 (36%, little period-20 signal), g7 (13%, little period-20 signal), g573 (11%, little period-20 signal), g15 (10%, little period-20 signal); writers d60 (+26%), d15 (+14%), d840 (+13%)</td></tr><tr><td>11</td><td>0.76</td><td>0.79</td><td>0.77</td><td>readers g24 (22%, little period-20 signal), g17 (17%, little period-20 signal), g30 (11%, little period-20 signal), u86 (9%, little period-20 signal); writers d24 (+19%), d17 (+16%), d30 (+16%)</td></tr><tr><td>12</td><td>0.90</td><td>0.63</td><td>0.90</td><td>readers g11 (69%, reads the ring), u57 (21%, reads <i>h</i>), g7 (1%, little period-20 signal); writers d11 (+58%), d57 (+19%), d7 (+5%)</td></tr><tr><td>13</td><td>0.53</td><td>0.46</td><td>0.47</td><td>readers g58 (65%, reads <i>g</i>), g6 (4%, little period-20 signal), u34 (4%, reads the ring), g21 (2%, little period-20 signal); writers d6 (+13%), d21 (+10%), d34 (+10%)</td></tr><tr><td>14</td><td>0.52</td><td>0.46</td><td>0.73</td><td>readers u33 (49%, reads <i>g</i>), g73 (19%, reads <i>g</i>), g69 (4%, reads the ring), g67 (3%, little period-20 signal); writers d33 (+30%), d67 (+5%), d69 (+4%)</td></tr><tr><td>15</td><td>0.42</td><td>0.56</td><td>0.64</td><td>readers g123 (14%, reads <i>g</i>), g142 (9%, little period-20 signal), u41 (8%, reads the plane), g70 (7%, little period-20 signal); writers d123 (+13%), d142 (+8%), d497 (+6%)</td></tr><tr><td>16</td><td>0.33</td><td>0.57</td><td>0.63</td><td>readers u256 (8%, little period-20 signal), u216 (7%, reads the ring), g795 (6%, little period-20 signal), g149 (6%, little period-20 signal); writers d252 (+8%), d186 (+6%), d344 (+5%)</td></tr></table>
<figure><img src="fig/mod20_sublayers.png"><figcaption>Figure 6. Per block (solid: components, dashed: neurons).
(a) Period-20 gain of the MLP and of attention. (b) Share of the MLP's period-20 output caused by the plane, and the
period-20 output it makes without the plane. (c) Held-out r² of Δ from the plane's coordinates: one linear map, or one
per units digit. (d) Reader-level linearity: secant, gated secant, additivity.</figcaption></figure>

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
<h3>8. Block 12 in the planes: components</h3>
<p>Block 12's MLP turns the ring into quarters (section 2). The components model's MLP has the same 14,336 neurons as
the original (indices 0–14335); only the gate, up and down matrices are replaced by sums of components. In the components
model, two neurons carry the step, and each is fed by a few components (<code>mod20_b12_fig.py</code>). A component reader feeds the 14,336 neurons through its write row
<i>U</i><sub>up,<i>c</i></sub> or <i>U</i><sub>gate,<i>c</i></sub> ∈ ℝ<sup>14336</sup>; a neuron <i>n</i>'s gate pre-activation is
gate<sub><i>n</i></sub> = Σ<sub><i>c</i></sub> <i>t<sub>c</sub></i> <i>U</i><sub>gate,<i>c</i></sub>[<i>n</i>] over gate components, its up
pre-activation up<sub><i>n</i></sub> the same over up components, and its <b>write coefficient</b> is silu(gate<sub><i>n</i></sub>)·up<sub><i>n</i></sub>
(silu(<i>x</i>) = <i>x</i>/(1 + e<sup>−<i>x</i></sup>)). The read vector of a reader in residual-stream coordinates is
<i>ρ</i> = <i>γ</i> ⊙ <i>w</i>, with <i>γ</i> ∈ ℝ<sup>4096</sup> block 12's post-attention RMSNorm gain, ⊙ the elementwise product and
<i>w</i> the reader's read vector, so that its activation is <i>ρ</i> · <i>X</i>/rms(<i>X</i>). Its <b>in-plane</b> value is
‖<i>P</i><sup>T</sup><i>ρ</i>‖/‖<i>ρ</i>‖ (1 when <i>ρ</i> lies in the plane), and the arrows in figure 8 are <i>P</i><sup>T</sup><i>ρ</i>/‖<i>ρ</i>‖ drawn
from the activation zero. <i>P</i><sub>10</sub> is the frequency-1 plane of <i>a</i> mod 10 of the same stream.</p>
<ul>
<li><b>Neuron 5212, the h × [r ≥ 5] product.</b> Its up input is almost only component u57
(<i>U</i><sub>up,57</sub>[5212] = 0.31; u57's next largest write to any neuron is 0.08).
u57 reads the ring (in-plane 0.66): its contribution to up<sub>5212</sub> is
0.11–0.12 for <i>s</i> = 1–3 and −0.69 to −0.64 for <i>s</i> = 15–17. Its gate is set by
components that read the units digit, not the ring (in-plane in <i>P</i>: g37 0.10, g33
0.01; in <i>P</i><sub>10</sub>: 0.55 and
0.48). g37 lowers gate<sub>5212</sub> by −1.31 to −1.02 for <i>r</i> = 2–4, and g33
raises it by 0.45–0.59 for <i>r</i> = 5–7. silu(gate<sub>5212</sub>) is 0.90–1.47
for <i>r</i> = 5–9 and −0.21 to 0.22 for <i>r</i> = 0–4. The write coefficient is
0.35–0.55 for <i>s</i> = 5–9, −0.39 to −0.11 for <i>s</i> = 15–19 and between
-0.10 and 0.04 elsewhere: a sign set by the
ring's <i>h</i> side, nonzero only when <i>r</i> ≥ 5.</li>
<li><b>Neuron 8953, the quarter s = 10–14.</b> Its gate is mostly component g11 (<i>U</i><sub>gate,11</sub>[8953] =
0.36; next largest 0.10), which reads the ring (in-plane 0.69)
and contributes 1.08–1.42 to gate<sub>8953</sub> for <i>s</i> = 10–14. Its up input is nearly constant
(−1.62 to −1.35, mostly component u1, in-plane 0.03). The write coefficient is
−1.34 to −0.87 for <i>s</i> = 10–14 and 0.06–0.43 elsewhere.</li>
</ul>
<figure><img src="fig/mod20_b12_components.png"><figcaption>Figure 8. Components model, block 12's MLP input. (a) The
<i>a</i> mod 20 plane <i>P</i> (residues coloured by quarter, as in figure 2; the vector from residue 10's mean to residue 0's mean
points right) with the read directions of the components that feed neurons 5212 and 8953; the legend gives each one's
in-plane value. (b) The same readers in the <i>a</i> mod 10 frequency-1 plane <i>P</i><sub>10</sub> (blue <i>r</i> = 0–4, orange
<i>r</i> = 5–9; residue 0's mean points right). An arrow gives the sign of the component's activation; g37 enters gate<sub>5212</sub>
with a negative weight. (c) Per residue <i>s</i>: each component's contribution to neuron 5212's gate or up pre-activation, and
the neuron's write coefficient. (d) The same for neuron 8953.</figcaption></figure>

<h3>9. Block 12 in the planes: neurons</h3>
<p>The same two neurons of the original model, with quantities defined as in section 8 (here <i>w</i> is the row of
<code>gate_proj</code> or <code>up_proj</code>). <b>Neuron 5212</b>: the up row reads the ring (in-plane
0.67; up<sub>5212</sub> is 0.29–0.56 for <i>s</i> = 1–8 and −0.60 to −0.25
for <i>s</i> = 13–19). The gate row reads the units digit (in-plane in <i>P</i> 0.26, in
<i>P</i><sub>10</sub> 0.48): silu(gate<sub>5212</sub>) is 0.36–0.87
for <i>r</i> = 5–9 and −0.27 to 0.14 for <i>r</i> = 0–4. The write coefficient is
0.24–0.35 for <i>s</i> = 5–9, −0.44 to −0.09 for <i>s</i> = 15–19, and between
-0.14 and 0.08 elsewhere. <b>Neuron 8953</b>:
the gate row reads the ring (in-plane 0.61) and silu(gate<sub>8953</sub>) is
0.66–1.00 for <i>s</i> = 10–14 against −0.26 to −0.10 elsewhere; the up row
is nearly constant (−1.53 to −0.93, in-plane 0.19). Its write coefficient is
−1.02 to −0.66 for <i>s</i> = 10–14 and 0.14–0.36 elsewhere. The components model
routes the same two neurons through u57 (up 5212), g37 and g33 (gate 5212), g11 (gate 8953) and u1 (up 8953) (section 8).</p>
<figure><img src="fig/mod20_b12_neurons.png"><figcaption>Figure 9. Original model, block 12's MLP input, as figure 8 with the
gate and up rows of neurons 5212 and 8953 as readers. (c), (d): per residue <i>s</i>, the neuron's up pre-activation,
silu of its gate pre-activation, and their product (the write coefficient).</figcaption></figure>
<h3>10. The 20 residue means in the full MLP input: cosines and Cohen's d</h3>
<p>Sections 1 and 2 look only at the plane <i>P</i>. This section looks at the whole residue means, in the same way as the
mod-10 geometry page (<code>mod20_geometry.py</code>). <i>y</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is block <i>ℓ</i>'s MLP input (the
post-attention RMSNorm output) in one of three spaces: <i>x</i>, the full input of the components model; <i>x</i><sub>orig</sub>,
the full input of the original model; <i>x<sub>V</sub></i>, <i>x</i> projected onto the span of the block's kept readers' read
vectors (the space of the mod-10 geometry page). <i>ν<sub>s</sub></i> ∈ ℝ<sup>4096</sup> is the mean of <i>y</i> over the 20 operands with
<i>a</i> mod 20 = <i>s</i> (the residue's direction; nothing is centred). cos[<i>s</i>, <i>s′</i>] = <i>ν<sub>s</sub></i> · <i>ν<sub>s′</sub></i>/(‖<i>ν<sub>s</sub></i>‖ ‖<i>ν<sub>s′</sub></i>‖).
Cohen's d of residues <i>s</i>, <i>s′</i> projects both clouds of 20 operands onto <i>u</i> = (<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>)/‖<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>‖ and
divides ‖<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>‖ by the pooled within-cloud standard deviation of the projections (unbiased variances, averaged
over the two clouds). Pairs are grouped as <b>antipodal</b> (<i>s′</i> = <i>s</i> + 10: same <i>r</i>, other <i>h</i>), <b>lag 1</b>
(<i>s′</i> = <i>s</i> ± 1 mod 20) and <b>other</b>.</p>
<p><b>Cosines.</b> The antipodal pairs are the most similar pairs at every block in <i>x</i> and <i>x</i><sub>orig</sub>: mean
cosine 0.815 at block 0 (lag 1 0.737, other
0.681) and 0.94–0.99 over blocks 1–16. From
block 1 on every cosine is at least 0.76, because all 20 means share one large
direction; figure 10 gives each block its own colour range. The bright diagonal lines at <i>s′</i> = <i>s</i> ± 10 are the
mod-10 structure: residues that share <i>r</i> point almost the same way, and the part that tells <i>s</i> from <i>s</i> + 10 is small.
The original model gives the same picture (0.93–0.99 for antipodal pairs over
blocks 1–16).</p>
<p><b>Cohen's d.</b> The antipodal pairs are also the least separated pairs at every block, in all three spaces. In <i>x</i>,
their median d is 12.8 at block 0 (lag 1 19.8), falls to
1.5 by block 11, and rises to 7.9 at
block 15 (lag 1 11.4). In <i>x</i><sub>orig</sub> the minimum over blocks is
3.0 (block 9). In <i>x<sub>V</sub></i> the
separations are lower at several blocks (antipodal median 1.0 at block 1,
0.7 at block 16, against 9.6 and
7.3 in <i>x</i>): part of the difference between the residue clouds lies outside the span of
the kept readers.</p>
<figure><img src="fig/mod20_geom_cos_x.png"><figcaption>Figure 10. cos[<i>s</i>, <i>s′</i>] between the 20 residue means of the
components model's full MLP input, blocks 0–16 (row <i>s</i>, column <i>s′</i>; diagonal grey; white lines at the quarter
edges). Each panel has its own colour range. Titles: mean cosine of the antipodal and lag-1 pairs.</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x.png"><figcaption>Figure 11. Cohen's d between residue clouds <i>s</i> and <i>s′</i> along the
axis joining their means, components model, full MLP input; one logarithmic colour scale for all blocks. Titles: median d of
the antipodal and lag-1 pairs.</figcaption></figure>
<figure><img src="fig/mod20_geom_cos_x_orig.png"><figcaption>Figure 12. As figure 10, original model.</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x_orig.png"><figcaption>Figure 13. As figure 11, original model.</figcaption></figure>
<figure><img src="fig/mod20_geom_cos_x_V.png"><figcaption>Figure 14. As figure 10, in <i>x<sub>V</sub></i> (components model,
kept readers' span).</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x_V.png"><figcaption>Figure 15. As figure 11, in <i>x<sub>V</sub></i>.</figcaption></figure>
<h3>11. Copying the circle to "="</h3>
<p>The answer is written at the last token, "=". The operands reach it through attention: block 16 copies <i>a</i> from
position 1 and block 15 copies <i>b</i> from position 3 (the mod-10 notebook, part 5). This section asks whether the mod-20
circle is copied too, by which heads and value components, and what it looks like once it arrives (<code>mod20_copy.py</code>).</p>
<p><b>Setup.</b> The prompts are the two sets of the mod-10 copy analysis: for the copy of <i>a</i>, every <i>a</i> = 1…400 with
ten values of <i>b</i> and both operations (8,000 prompts); for the copy of <i>b</i>, ten values of <i>a</i> and every <i>b</i>. The
original model uses 2,000 of each set. In this section <i>s</i> = <i>o</i> mod 20, <i>r</i> = <i>o</i> mod 10 and <i>h</i> = ⌊<i>o</i>/10⌋ mod 2,
where <i>o</i> is the copied operand (<i>a</i> or <i>b</i>). Four places are compared:</p>
<ul>
<li><b>source position</b>: the attention input (after the block's input RMSNorm) at the operand's own position;</li>
<li><b>copy written at "="</b>: the part of the attention output at "=" that comes from the operand's position (all heads);</li>
<li><b>"=" before / after the attention</b>: the stream at "=" entering the block, and the same plus the whole attention output.</li>
</ul>
<p>For a quantity over the prompts, Ψ<sub><i>j</i></sub> is the Fourier coefficient at frequency <i>j</i> of its 20 residue means
(section 0.2), and "‖Ψ<sub>1</sub>‖/null" compares the circle's strength with the 99th percentile over 100 random relabellings of
the prompts. The <b>copy gain</b> at frequency <i>j</i> is ‖Ψ<sub><i>j</i></sub>(copy)‖/‖Ψ<sub><i>j</i></sub>(source)‖. α, the held-out accuracy
of <i>h</i> given <i>r</i> and the antipodal Cohen's d are as in sections 2.1 and 2.2, computed in the plane of the place in
question. <b>Roundness</b> is the smaller over the larger singular value of the ten odd parts <i>o<sub>s</sub></i> (1 for a round ring).
A head's or value component's <b>share</b> is its part of Σ‖Ψ<sub>1</sub>‖² over heads (or components) of the copy, set beside its
part of Σ<sub><i>k</i>=1…5</sub>‖Ψ<sub>2<i>k</i></sub>‖², the copy of the mod-10 representation.</p>

<p><b>The circle is copied, by the same head as the mod-10 representation.</b> The copy of <i>a</i> carries the circle with a
gain of 0.22 (0.21), within the range of the mod-10 frequencies'
gains, 0.10–0.27 (0.14–0.20). For <i>b</i> the
circle's gain is 0.21 (0.19) against 0.08–0.29
(0.13–0.24). The heads: for <i>a</i>, head 21 (99% of the period-20 copy, 98% of the period-10 copy), head 3 (1% of the period-20 copy, 1% of the period-10 copy) (original model:
head 21 (96% of the period-20 copy, 92% of the period-10 copy), head 3 (1% of the period-20 copy, 4% of the period-10 copy)); for <i>b</i>, head 13 (97% of the period-20 copy, 97% of the period-10 copy), head 3 (3% of the period-20 copy, 3% of the period-10 copy) (original model: head 13 (85% of the period-20 copy, 86% of the period-10 copy), head 14 (7% of the period-20 copy, 7% of the period-10 copy)). The same head carries both
periods in the same proportion.</p>
<p><b>But through different value components.</b> In the components model, the value components that carry the copied circle
are not the ones that carry the copied mod-10 representation (each listed with its share of the period-20 copy; then of the
period-10 copy). For <i>a</i>, period 20: v213 (23%; 2%), v18 (15%; 5%), v65 (14%; 0%), v26 (7%; 0%); period 10: v86 (11%; 0%), v39 (10%; 0%), v114 (8%; 0%), v115 (7%; 0%). For <i>b</i>,
period 20: v120 (41%; 1%), v112 (12%; 0%), v153 (12%; 12%), v74 (9%; 1%); period 10: v134 (17%; 0%), v153 (12%; 12%), v83 (10%; 0%), v85 (9%; 0%).</p>
<p><b>The copy is one linear map of the circle, which flattens it.</b> The ten odd parts of the copy are fitted from those at
the source by one 2 × 2 matrix at r² 0.997 and 0.997 (components; neurons
0.996 and 0.995). The map is not a rotation: it keeps one direction of the plane
and shrinks the other. Roundness goes from 0.89 at the source to 0.48 in the copy of
<i>a</i> (neurons 0.90 → 0.74) and from 0.84 to 0.13
in the copy of <i>b</i> (0.82 → 0.37); α falls with it (0.56 →
0.38 and 0.49 → 0.14). The long axis is the rounded-tens parity <i>g</i>, the short one mostly the tens parity <i>h</i>.
In the components model the copied ring carries <i>g</i> on the long axis (82% of the ring's energy, step r² 0.98) and <i>h</i> on the short axis (18% of the ring's energy, step r² 0.91) for <i>a</i>,
and <i>g</i> on the long axis (98% of the ring's energy, step r² 0.98) and a step at <i>s</i> = 1 on the short axis (2% of the ring's energy, step r² 0.88) for <i>b</i>. In the original model: <i>g</i> on the long axis (65% of the ring's energy, step r² 0.97) and <i>h</i> on the short axis (35% of the ring's energy, step r² 0.88) for <i>a</i>; <i>g</i> on the long axis (88% of the ring's energy, step r² 0.99) and <i>h</i> on the short axis (12% of the ring's energy, step r² 0.83)
for <i>b</i>.</p>
<p><b>What arrives at "=".</b> Before the attention, "=" holds no circle of the operand (‖Ψ<sub>1</sub>‖/null
0.38 for <i>a</i>, 0.42 for <i>b</i>; <i>h</i> given <i>r</i> at chance). After
it, the circle is there: <i>h</i> given <i>r</i> is decoded at 0.98 for <i>a</i> and
0.89 for <i>b</i> (neurons 0.90 and
0.67), and the plane at "=" is the copy's plane (principal cosines
0.999, 0.989 for <i>a</i>,
1.000, 0.803 for <i>b</i>).</p>
<p><b>A caveat on the null.</b> At the source position the stream depends on the operand alone, so the 400 prompts of one
residue share 20 identical copies of each value, and the shuffled-label null is weak. Here the odd frequencies <i>j</i> = 3…9
also exceed it (up to 4.5 times for <i>a</i> at the source), unlike in the
400-prompt analysis of section 1.1.</p>
<table><tr><th>model</th><th>copy</th><th>where</th><th>‖Ψ<sub>1</sub>‖/null</th><th>α</th><th>roundness</th><th><i>h</i> given <i>r</i> (held out)</th><th>d(<i>s</i>, <i>s</i>+10)</th></tr><tr><td>components</td><td>a (block 16)</td><td>source position</td><td>10.03</td><td>0.56</td><td>0.89</td><td>1.00</td><td>8.0</td></tr><tr><td>components</td><td>a (block 16)</td><td>copy written at "="</td><td>10.89</td><td>0.38</td><td>0.48</td><td>1.00</td><td>8.0</td></tr><tr><td>components</td><td>a (block 16)</td><td>"=" before the attention</td><td>0.38</td><td>0.44</td><td>0.60</td><td>0.50</td><td>0.1</td></tr><tr><td>components</td><td>a (block 16)</td><td>"=" after the attention</td><td>4.91</td><td>0.38</td><td>0.48</td><td>0.98</td><td>6.4</td></tr><tr><td>components</td><td>b (block 15)</td><td>source position</td><td>5.44</td><td>0.49</td><td>0.84</td><td>1.00</td><td>5.6</td></tr><tr><td>components</td><td>b (block 15)</td><td>copy written at "="</td><td>8.88</td><td>0.14</td><td>0.13</td><td>0.94</td><td>4.3</td></tr><tr><td>components</td><td>b (block 15)</td><td>"=" before the attention</td><td>0.42</td><td>0.64</td><td>0.16</td><td>0.50</td><td>0.1</td></tr><tr><td>components</td><td>b (block 15)</td><td>"=" after the attention</td><td>3.41</td><td>0.18</td><td>0.18</td><td>0.89</td><td>3.1</td></tr><tr><td>neurons</td><td>a (block 16)</td><td>source position</td><td>4.58</td><td>0.65</td><td>0.90</td><td>0.99</td><td>7.4</td></tr><tr><td>neurons</td><td>a (block 16)</td><td>copy written at "="</td><td>5.51</td><td>0.57</td><td>0.74</td><td>0.99</td><td>5.2</td></tr><tr><td>neurons</td><td>a (block 16)</td><td>"=" before the attention</td><td>0.59</td><td>0.60</td><td>0.88</td><td>0.54</td><td>0.5</td></tr><tr><td>neurons</td><td>a (block 16)</td><td>"=" after the attention</td><td>2.06</td><td>0.56</td><td>0.76</td><td>0.90</td><td>4.5</td></tr><tr><td>neurons</td><td>b (block 15)</td><td>source position</td><td>2.82</td><td>0.60</td><td>0.82</td><td>0.88</td><td>3.5</td></tr><tr><td>neurons</td><td>b (block 15)</td><td>copy written at "="</td><td>4.22</td><td>0.36</td><td>0.37</td><td>0.92</td><td>3.5</td></tr><tr><td>neurons</td><td>b (block 15)</td><td>"=" before the attention</td><td>0.67</td><td>1.00</td><td>0.55</td><td>0.51</td><td>0.2</td></tr><tr><td>neurons</td><td>b (block 15)</td><td>"=" after the attention</td><td>1.69</td><td>0.50</td><td>0.50</td><td>0.67</td><td>1.7</td></tr></table>
<figure><img src="fig/mod20_copy_alive.png"><figcaption>Figure 16. Components model. Rows: the copy of <i>a</i> by block 16
and of <i>b</i> by block 15. First three panels: the circle's odd parts <i>o<sub>s</sub></i> and −<i>o<sub>s</sub></i> (<i>s</i> = 0…19, coloured by
quarter as in figure 2) at the source position, in the copy written at "=", and at "=" after the attention, each in its own
plane, turned so that residue 0 points right; the cross is the plane's zero. Titles give α, the held-out accuracy of <i>h</i>
given <i>r</i>, and the median antipodal Cohen's d. Last panel: each head's share of the copy's period-20 energy and of its
period-10 energy.</figcaption></figure>
<figure><img src="fig/mod20_copy_orig.png"><figcaption>Figure 17. As figure 16, original model (2,000 prompts per copy).</figcaption></figure>
`
},
];
