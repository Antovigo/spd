// Notebook pages of the mod 2 report (index.html).
window.NOTEBOOK = [
{
id: "mod2_l3",
title: "Mod 2: block-3 d189 and the parity line",
date: "2026-10-22",
html: `
<p><b>Summary.</b> At the token <i>a</i>, the block-3 down component <code>d189</code> is one of the first
components whose activation is binary in <i>a</i>: about 0.33 for even <i>a</i> and
0.01 for odd <i>a</i>. It is driven mostly by one gate reader, <code>g30</code>, which reads
the parity line of block 3's mod-10 representation. <code>d189</code> writes back into the same mod-10 representation,
along its parity line, not into a separate subspace. Only the block-4 readers that are themselves binary pick the
write up. It is the largest single source of the parity signal of the block-4 reader <code>u147</code>. Block 3's
MLP raises the even/odd separation d′ at the block-4 MLP input from 4.68 to 6.17.
Without <code>d189</code> the rest of that MLP would lower it to 3.31.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {1, …, 400} and op ∈ {+, −}. The model is the alive-only model: every component of the
ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off. It is
rebuilt in float64 at position 1 (the token <i>a</i>), where everything depends on <i>a</i> alone.
<b>Nothing on this page is centred or detrended.</b> The network has no biases, so zero is the zero of the
activation space. Script: <code>notes/reader_writer/readers/mod2_page.py</code> (branch
experiment/arith_representations).</p>
<ul>
<li><b>Residues.</b> <i>r</i> = <i>a</i> mod 10 ∈ {0, …, 9}. "Even" and "odd" mean <i>a</i> mod 2 = 0 and 1
(200 values each).</li>
<li><b>Components.</b> <code>g&lt;id&gt;</code>, <code>u&lt;id&gt;</code> and <code>d&lt;id&gt;</code> are gate,
up and down components of the block named in the text; the ids are the AB-grid ids. <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup>
is a reader's read vector and <i>U<sub>w</sub></i> ∈ ℝ<sup>4096</sup> a down component's write vector.
<i>t<sub>189</sub></i>(<i>a</i>) is the inner activation of <code>d189</code>, so it adds
<i>t<sub>189</sub></i>(<i>a</i>) <i>U<sub>189</sub></i> to the residual stream.</li>
<li><b>Mod-10 representation.</b> <code>L&lt;l&gt;.p1.a_mod_10.means</code> in the representation repository is
the span of the ten residue means μ<sub><i>r</i></sub> ∈ ℝ<sup>4096</sup> of block <i>l</i>'s MLP input
(10 dimensions, including the direction shared by all residues). Its parity line is the direction of
Σ<sub><i>r</i></sub> (−1)<sup><i>r</i></sup> μ<sub><i>r</i></sub>, which is Fourier frequency <i>k</i> = 5 in <i>r</i>.
<i>B</i><sub>4</sub> is an orthonormal basis (4096 × 10) of the block-4 span.</li>
<li><b>Residual states.</b> mid<sub>4</sub>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream after block 4's
attention. "After block 3's MLP" is mid<sub>4</sub>. "Before" is mid<sub>4</sub> minus block 3's whole MLP output.
"Without d189" is mid<sub>4</sub> minus <i>t<sub>189</sub></i> <i>U<sub>189</sub></i>. Block 4's attention is held
fixed in all three. Re-running it on the edited stream changes none of the block-4 reader numbers in section 2 at the third decimal.</li>
<li><b>Block-4 MLP input.</b> <i>x</i>(<i>a</i>) = RMSNorm(state(<i>a</i>)) with block 4's post-attention gain.
This is what block 4's readers read.</li>
<li><b>Separation axis.</b> <i>m<sub>e</sub></i>, <i>m<sub>o</sub></i> ∈ ℝ<sup>4096</sup> are the means of
<i>x</i> over even and odd <i>a</i>. <i>u</i> = (<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>) / ‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖
is a unit vector.</li>
<li><b>Mod-2 amplitude.</b> For a scalar <i>f</i>(<i>a</i>), A5(<i>f</i>) = mean over <i>a</i> of
<i>f</i>(<i>a</i>) (−1)<sup><i>a</i></sup>.</li>
</ul>

<h3>1. What drives d189</h3>
<p>The exact variance split of block 3 credits each share φ of <code>d189</code>'s variance over the prompts to one
reader (φ sums to 1 over readers). It was run on the model with only the λ = 5e-05 mask on. At position 1 one gate
dominates:</p>
<table><tr><th>reader</th><th>φ</th></tr><tr><td><code>g30</code></td><td>+0.820</td></tr><tr><td><code>g145</code></td><td>+0.115</td></tr><tr><td><code>u281</code></td><td>+0.066</td></tr><tr><td><code>g67</code></td><td>+0.062</td></tr><tr><td><code>g83</code></td><td>-0.021</td></tr><tr><td><code>g61</code></td><td>-0.017</td></tr></table>
<p>Both main gates are explained by block 3's mod-10 representation (r² 0.96 for <code>g30</code>, 0.97 for
<code>g145</code>). Inside it they read different directions; the cosine of their read vectors is
0.22. The table gives each gate's pre-activation on the ten residue means,
μ<sub><i>r</i></sub> · <i>V<sub>c</sub></i>, and the share of that 10-vector's energy at each Fourier frequency
<i>k</i> in <i>r</i>:</p>
<table><tr><th></th><th>r=0</th><th>r=1</th><th>r=2</th><th>r=3</th><th>r=4</th><th>r=5</th><th>r=6</th><th>r=7</th><th>r=8</th><th>r=9</th><th>k=0</th><th>k=1</th><th>k=2</th><th>k=3</th><th>k=4</th><th>k=5</th></tr><tr><td><code>g30</code></td><td>-0.01</td><td>+0.83</td><td>-1.80</td><td>-0.00</td><td>-1.93</td><td>+0.07</td><td>-2.31</td><td>-0.08</td><td>-1.50</td><td>+1.20</td><td>0.18</td><td>0.19</td><td>0.06</td><td>0.00</td><td>0.01</td><td>0.55</td></tr><tr><td><code>g145</code></td><td>+4.78</td><td>+0.29</td><td>+0.05</td><td>-0.02</td><td>+0.12</td><td>+2.23</td><td>+0.10</td><td>-0.04</td><td>-0.08</td><td>+0.10</td><td>0.20</td><td>0.05</td><td>0.38</td><td>0.04</td><td>0.30</td><td>0.02</td></tr></table>
<p><code>g30</code> is mostly the parity line (<i>k</i> = 5), with a <i>k</i> = 1 part that makes it positive at
<i>r</i> = 1 and 9. <code>g145</code> is mostly <i>k</i> = 2 and 4. It is high at both <i>r</i> = 0 and <i>r</i> = 5,
so it carries <i>a</i> mod 5 = 0 rather than parity. The resulting activation of <code>d189</code> by residue is
0.35, 0.01, 0.32, -0.00, 0.29, 0.02, 0.32, 0.02, 0.35, -0.01 for <i>r</i> = 0, …, 9 (right panel of figure 2).</p>

<h3>2. Where d189 writes</h3>
<p>Block 4's readers see the write through block 4's norm gain. Of that gained write vector,
67% of the norm lies in the span of block 4's read vectors. Of that part,
94.5% lies in the block-4 mod-10 span, and its cosine with that span's parity line is
0.92. So <code>d189</code> writes back into the mod-10 representation, along its parity line.</p>
<p>The table lists every block-4 reader that the mod-10 representation explains (r² ≥ 0.5). "Removed" is
‖<i>t</i> − <i>t</i><sub>no d189</sub>‖² / ‖<i>t</i>‖² over <i>a</i> = 1..400, where <i>t</i> is the reader's
pre-activation and <i>t</i><sub>no d189</sub> the same without <code>d189</code>'s write. "rms" is the root-mean-square
of <i>t</i>.</p>
<table><tr><th>reader</th><th>mod-10 r²</th><th>rms <i>t</i></th><th>removed</th><th>A5(<i>t</i>)</th><th>A5 without d189</th></tr><tr><td><code>g182</code></td><td>0.82</td><td>0.84</td><td>0.194</td><td>-0.599</td><td>-0.348</td></tr><tr><td><code>u147</code></td><td>0.98</td><td>1.45</td><td>0.133</td><td>+0.976</td><td>+0.618</td></tr><tr><td><code>u854</code></td><td>0.68</td><td>0.42</td><td>0.090</td><td>+0.211</td><td>+0.126</td></tr><tr><td><code>u48</code></td><td>0.85</td><td>0.83</td><td>0.011</td><td>+0.263</td><td>+0.205</td></tr><tr><td><code>g117</code></td><td>0.95</td><td>1.17</td><td>0.002</td><td>-0.053</td><td>-0.089</td></tr><tr><td><code>u52</code></td><td>0.91</td><td>1.35</td><td>0.001</td><td>+0.123</td><td>+0.089</td></tr><tr><td><code>g91</code></td><td>0.92</td><td>2.17</td><td>0.000</td><td>+0.075</td><td>+0.095</td></tr><tr><td><code>g729</code></td><td>0.69</td><td>5.42</td><td>0.000</td><td>-0.025</td><td>-0.060</td></tr><tr><td><code>g622</code></td><td>0.58</td><td>0.70</td><td>0.000</td><td>-0.026</td><td>-0.025</td></tr><tr><td><code>g51</code></td><td>0.91</td><td>7.58</td><td>0.000</td><td>-0.176</td><td>-0.156</td></tr><tr><td><code>u611</code></td><td>1.00</td><td>13.80</td><td>0.000</td><td>+0.026</td><td>-0.002</td></tr><tr><td><code>g37</code></td><td>0.99</td><td>12.98</td><td>0.000</td><td>-0.086</td><td>-0.064</td></tr></table>
<p>The large-amplitude mod-10 readers barely move. The binary readers <code>g182</code>, <code>u147</code> and
<code>u854</code> lose 9–19% of their pre-activation.</p>

<h3>3. Where u147 reads its parity from</h3>
<p><code>u147</code>'s read vector lies 87% (by norm) in the block-4 mod-10 span. Its cosine with
that span's parity line is 0.82. Its mean pre-activation by residue is
1.71, 0.02, 2.00, 0.02, 2.05, 0.01, 2.12, 0.08, 2.04, 0.02 for <i>r</i> = 0, …, 9, and A5 = 0.976. The residual stream
mid<sub>4</sub> is exactly the sum of the token embedding, the attention outputs of blocks 0–4 and every down
component's write in blocks 0–3. Holding block 4's RMS scale fixed, A5 of <code>u147</code> splits over these terms
(the terms sum to A5 exactly). The largest are:</p>
<table><tr><th>term</th><th>share of A5</th></tr><tr><td><code>L3.down.189</code></td><td>+0.362</td></tr><tr><td><code>embed</code></td><td>+0.219</td></tr><tr><td><code>L0.down.60</code></td><td>+0.046</td></tr><tr><td><code>L0.down.69</code></td><td>+0.038</td></tr><tr><td><code>L0.down.58</code></td><td>+0.037</td></tr><tr><td><code>L0.down.47</code></td><td>+0.033</td></tr></table>
<p>All block-0 down components together give +0.346. Deleting <code>d189</code>'s write lowers
A5 from 0.976 to 0.618.</p>

<h3>4. Does d189 sharpen the parity split?</h3>
<p>Three metrics compare the states. All are computed on <i>x</i> projected onto that state's own <i>u</i>:</p>
<ul>
<li>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖, the distance between the class means.</li>
<li>d′ = ‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖ / σ<sub>w</sub>, where σ<sub>w</sub> is the standard
deviation of <i>x</i> · <i>u</i> within each class, pooled over the two classes.</li>
<li>gap = (min over even <i>a</i> of <i>x</i> · <i>u</i> − max over odd <i>a</i> of <i>x</i> · <i>u</i>) /
‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖. It is positive when no even and odd <i>a</i> overlap.</li>
</ul>
<p>σ<sub>w</sub> is a spread around each class mean; the activations themselves are not centred. The metrics are
given in the full 4096-dimensional space and inside the block-4 mod-10 span <i>x</i> <i>B</i><sub>4</sub>. In the span,
"LDA d′" uses the full 10 × 10 within-class covariance Σ<sub>w</sub>:
√((<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>)<sup>T</sup> Σ<sub>w</sub><sup>−1</sup> (<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>)).</p>
<table><tr><th>state</th><th>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖</th><th>d′</th><th>gap</th><th>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖ in <i>B</i><sub>4</sub></th><th>d′ in <i>B</i><sub>4</sub></th><th>gap in <i>B</i><sub>4</sub></th><th>LDA d′ in <i>B</i><sub>4</sub></th></tr><tr><td>before block 3's MLP</td><td>1.82</td><td>4.68</td><td>+0.16</td><td>1.36</td><td>4.23</td><td>+0.03</td><td>6.96</td></tr><tr><td>after, without d189</td><td>1.52</td><td>3.31</td><td>-0.22</td><td>1.09</td><td>3.58</td><td>-0.46</td><td>6.57</td></tr><tr><td>after block 3's MLP</td><td>2.12</td><td>6.17</td><td>+0.39</td><td>1.70</td><td>6.42</td><td>+0.26</td><td>9.03</td></tr></table>
<p>Block 3's MLP moves the class means apart and turns the overlap into a margin. Without <code>d189</code> the same
MLP makes the classes overlap (negative gap).</p>
<figure><img src="fig/mod2_l3_dist.png" alt="Density of x·u for even and odd a, before and after block 3's MLP">
<figcaption>Figure 1. Density over <i>a</i> (Gaussian kernel) of <i>x</i>(<i>a</i>) · <i>u</i>, where <i>u</i> is the axis
of the "after" state. Blue: even <i>a</i>; orange: odd <i>a</i>. Left: before block 3's MLP. Right: after it (solid)
and after it without <code>d189</code>'s write (dashed). One axis for all states, so positions are comparable.</figcaption></figure>
<figure><img src="fig/mod2_l3_dist_own.png" alt="The same densities, each state on its own axis">
<figcaption>Figure 1b. As figure 1, but each state projected on its own axis <i>u</i> (the axis the table's metrics use).</figcaption></figure>
<figure><img src="fig/mod2_l3_residues.png" alt="Projection by residue and d189 activation">
<figcaption>Figure 2. Left: mean ± standard deviation over <i>a</i> of <i>x</i> · <i>u</i> (the "after" axis) for each residue
<i>r</i>, in the three states. Right: <i>t</i><sub>189</sub>(<i>a</i>) against <i>a</i>, coloured by parity.</figcaption></figure>
`
},
];
