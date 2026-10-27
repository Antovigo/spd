// Notebook pages of the mod 50 report.
window.NOTEBOOK = [
{
id: "mod50",
title: "Mod 50: census and the block-15 readers of b",
date: "2026-10-27",
html: `
<p><b>Summary.</b> At the token <i>b</i>, the block-15 gate readers <code>g19</code> and <code>g29</code> read a
pattern of <i>b</i> mod 50. Splitting their input exactly into one term per upstream writer, the exact-period-50 part
of that pattern comes mostly from three MLP writers of blocks 11–12 (<code>L12.down.24</code>,
<code>L11.down.10</code>, <code>L12.down.14</code>), from block-13 writers, and from the embedding of the token
<i>b</i>. Block 15's own attention opposes it: its share is −0.62
for <code>g19</code> and −0.61 for <code>g29</code>.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {1, …, 400} and op ∈ {+, −}. Positions: 1 is the token <i>a</i>, 3 the token <i>b</i>. The
model is the alive-only model: every component of the ceiling filter <code>addsub-08-last-pos</code> kept by the
mask “kl_int, sampled mask, λ = 5e-05” is on, and every other component and the weight delta are off.
<b>Nothing in this page is centred or detrended</b>: the network has no biases, so zero is the zero of the
activation space.</p>
<ul>
<li>A <b>component</b> <i>c</i> has a read vector <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup> (gate, up, q, k, v) or
ℝ<sup>14336</sup> (down) and a write vector <i>U<sub>c</sub></i>. Its id is <code>L&lt;block&gt;.&lt;kind&gt;.&lt;index&gt;</code>;
<code>g19</code> is <code>L15.gate.19</code>. A <b>writer</b> is a component that adds to the residual stream: an
attention <code>o</code> component or an MLP <code>down</code> component; the embedding of the token <i>b</i> counts
as one writer, “emb”.</li>
<li><i>f<sub>c</sub></i>(<i>x</i>): the inner activation of component <i>c</i> on prompt <i>x</i> divided by
‖<i>V<sub>c</sub></i>‖, a real number.</li>
<li><i>k</i> ∈ {0, …, 99}: a frequency of <i>b</i> mod 100 (integer); its exact period is
τ(<i>k</i>) = 100 / gcd(<i>k</i>, 100), a divisor of 100. The <b>exact-period-τ part</b> of a function of <i>b</i> is
its part made of the frequencies <i>k</i> with τ(<i>k</i>) = τ, computed from its means over the classes
<i>b</i> mod 100. Parts of different τ are orthogonal; a function of <i>b</i> mod 50 has parts with τ ∈ {1, 2, 5,
10, 25, 50} (τ = 1 is the constant).</li>
<li><b>share<sub>τ</sub></b>: the energy of the exact-period-τ part over the total energy Σ<sub><i>b</i></sub>
<i>f</i>(<i>b</i>)<sup>2</sup> (uncentred). A component is <b>labelled with period 50</b> when share<sub>50</sub> ≥
0.05, beats the 99th percentile of 1,000 random permutations of <i>b</i>, and beats the 99th percentile of 1,000
draws that rotate the hundreds of <i>b</i> independently; 50 is then in its <code>periods</code> list (definitions in
<code>repr_repo/seed_periodicity.py</code>). <b>smooth_r2</b> is the uncentred share of the energy explained by a
degree-3 polynomial in <i>b</i>; a slow trend in <i>b</i> also puts energy into τ = 50 and 100.</li>
</ul>

<h3>1. Components with period 50 at the operand positions</h3>
<p>Counts of the components of each block labelled with period 50 at position 1 (variable <i>a</i>) and at
position 3 (variable <i>b</i>), from <code>periodicity.json</code>. All kinds (gate, up, down, q, k, v, o) are counted.
“τ = 50 largest” counts those whose share<sub>50</sub> is the largest of their shares with τ &gt; 1. “smooth” counts
those with smooth_r2 &gt; 0.5.</p>
<table><tr><th>block</th><th>p1: labelled 50</th><th>p1: τ = 50 largest</th><th>p1: smooth</th>
<th>p3: labelled 50</th><th>p3: τ = 50 largest</th><th>p3: smooth</th></tr><tr><td>0</td><td>7</td><td>0</td><td>0</td><td>14</td><td>4</td><td>0</td></tr><tr><td>1</td><td>3</td><td>2</td><td>0</td><td>5</td><td>4</td><td>2</td></tr><tr><td>2</td><td>11</td><td>6</td><td>0</td><td>5</td><td>2</td><td>0</td></tr><tr><td>3</td><td>8</td><td>5</td><td>0</td><td>9</td><td>3</td><td>0</td></tr><tr><td>4</td><td>14</td><td>8</td><td>1</td><td>9</td><td>4</td><td>1</td></tr><tr><td>5</td><td>15</td><td>8</td><td>0</td><td>9</td><td>3</td><td>2</td></tr><tr><td>6</td><td>8</td><td>4</td><td>0</td><td>7</td><td>2</td><td>2</td></tr><tr><td>7</td><td>7</td><td>5</td><td>0</td><td>7</td><td>4</td><td>2</td></tr><tr><td>8</td><td>4</td><td>2</td><td>0</td><td>2</td><td>1</td><td>1</td></tr><tr><td>9</td><td>3</td><td>1</td><td>1</td><td>5</td><td>3</td><td>3</td></tr><tr><td>10</td><td>4</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>11</td><td>3</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0</td></tr><tr><td>12</td><td>9</td><td>3</td><td>1</td><td>8</td><td>5</td><td>3</td></tr><tr><td>13</td><td>8</td><td>6</td><td>0</td><td>11</td><td>6</td><td>1</td></tr><tr><td>14</td><td>13</td><td>6</td><td>1</td><td>9</td><td>5</td><td>0</td></tr><tr><td>15</td><td>56</td><td>31</td><td>15</td><td>41</td><td>24</td><td>15</td></tr><tr><td>16</td><td>29</td><td>15</td><td>3</td><td>29</td><td>18</td><td>7</td></tr><tr><td><b>total</b></td><td>202</td><td>104</td><td>22</td><td>171</td><td>89</td><td>39</td></tr></table>
<p>Block 15 has the most labelled components at both positions (56 at position 1,
41 at position 3), followed by block 16.</p>

<h3>2. The block-15 gate readers g19 and g29 and their writers</h3>
<p>Shares of <code>g19</code> at position 3 (<code>periodicity.json</code>, all 320,000 prompts): τ = 1: 0.047, τ = 5: 0.079, τ = 25: 0.045, τ = 50: 0.142, τ = 100: 0.116; its
<code>periods</code> list is [50]. Shares of <code>g29</code>: τ = 1: 0.661, τ = 5: 0.010, τ = 25: 0.033, τ = 50: 0.049, τ = 100: 0.027; its
<code>periods</code> list is empty: its share<sub>50</sub> is under the
0.05 threshold, and most of its energy is a constant.</p>
<p><b>Data.</b> The float64 rebuild of the alive-only model at position 3 (<code>track_build_p3.py</code>): 8,000
prompts, op ∈ {+, −}, ten values of <i>a</i> (7 + 41<i>i</i>, <i>i</i> = 0, …, 9), every <i>b</i>. On it,
<i>F<sub>c</sub></i>(<i>b</i>) is the mean of <i>f<sub>c</sub></i> over the 20 prompts with that <i>b</i>; the
share<sub>50</sub> of <code>g19</code> measured on these 8,000 prompts is
0.144 (against 0.142 on all
prompts).</p>
<figure><img src="fig/mod50_l15_curves.png" alt="F(b) of g19 and g29 against b and against b mod 50">
<figcaption>Left: <i>F<sub>c</sub></i>(<i>b</i>) for <i>b</i> = 1, …, 400 (grey dots), with its constant plus its
<i>b</i> mod 50 part (colour) and its constant plus its exact-period-50 part (black). Right: the same values against
<i>b</i> mod 50, one dot per <i>b</i>; the black line is the mean over the 8 values of <i>b</i> in each class.
Top: <code>g19</code>; bottom: <code>g29</code>.</figcaption></figure>

<p><b>The split.</b> Let <i>X</i>(<i>x</i>) ∈ ℝ<sup>4096</sup> be the residual stream at position 3 after block 15's
attention, γ ∈ ℝ<sup>4096</sup> the gain of block 15's post-attention RMSNorm, and
ρ(<i>x</i>) = (mean<sub><i>j</i></sub> <i>X<sub>j</sub></i>(<i>x</i>)<sup>2</sup> + 10<sup>−5</sup>)<sup>1/2</sup>
its RMS (<i>j</i> ∈ {1, …, 4096} indexes the coordinates). Then
<i>f<sub>c</sub></i>(<i>x</i>) = (γ ⊙ <i>V<sub>c</sub></i>) · <i>X</i>(<i>x</i>) / (ρ(<i>x</i>) ‖<i>V<sub>c</sub></i>‖),
and <i>X</i> is the sum of the embedding of <i>b</i> and of <i>t<sub>w</sub></i>(<i>x</i>) <i>U<sub>w</sub></i> over
every alive <code>o</code> component of blocks 0–15 and every alive <code>down</code> component of blocks 0–14
(<i>t<sub>w</sub></i> the writer's activation at position 3). So
<i>f<sub>c</sub></i> = Σ<sub><i>w</i></sub> <i>e<sub>w</sub></i> exactly, with
<i>e<sub>w</sub></i>(<i>x</i>) = <i>t<sub>w</sub></i>(<i>x</i>) (γ ⊙ <i>V<sub>c</sub></i>) · <i>U<sub>w</sub></i> /
(ρ(<i>x</i>) ‖<i>V<sub>c</sub></i>‖) (the reconstruction error is below 10<sup>−3</sup> of the largest
|<i>f<sub>c</sub></i>|). Writing <i>P</i><sub>50</sub><i>G</i> for the exact-period-50 part of the per-<i>b</i> mean
<i>G</i> of a field, writer <i>w</i>'s <b>share</b> is
⟨<i>P</i><sub>50</sub><i>E<sub>w</sub></i>, <i>P</i><sub>50</sub><i>F<sub>c</sub></i>⟩ /
‖<i>P</i><sub>50</sub><i>F<sub>c</sub></i>‖<sup>2</sup> (inner products over <i>b</i> = 1, …, 400; <i>E<sub>w</sub></i>
the per-<i>b</i> mean of <i>e<sub>w</sub></i>). The shares are signed and sum to 1 over the writers; a negative share
opposes the reader's pattern. <b>own</b> = ‖<i>P</i><sub>50</sub><i>E<sub>w</sub></i>‖<sup>2</sup> /
‖<i>P</i><sub>50</sub><i>F<sub>c</sub></i>‖<sup>2</sup> is the size of the writer's own period-50 part next to the
reader's.</p>
<figure><img src="fig/mod50_l15_shares.png" alt="writer shares of g19 and g29's period-50 part">
<figcaption>Left: the 12 writers with the largest |share| of the reader's exact-period-50 part (blue positive, red
negative). Right: the shares summed over each sublayer (“L<i>l</i>.mlp” all down components of block <i>l</i>,
“L<i>l</i>.attn” all its <code>o</code> components), sublayers with |sum| ≥ 0.02. Top: <code>g19</code>; bottom:
<code>g29</code>.</figcaption></figure>
<table><tr><th>writer</th><th>share in g19</th><th>own</th></tr><tr><td><code>L12.down.24</code></td><td>+0.49</td><td>0.34</td></tr><tr><td><code>L11.down.10</code></td><td>+0.40</td><td>0.20</td></tr><tr><td><code>emb</code></td><td>+0.32</td><td>0.21</td></tr><tr><td><code>L12.down.14</code></td><td>+0.26</td><td>0.26</td></tr><tr><td><code>L12.down.18</code></td><td>−0.20</td><td>0.11</td></tr><tr><td><code>L13.down.21</code></td><td>+0.17</td><td>0.04</td></tr><tr><td><code>L13.down.18</code></td><td>+0.17</td><td>0.06</td></tr><tr><td><code>L13.down.79</code></td><td>−0.13</td><td>0.02</td></tr><tr><td><code>L15.o.226</code></td><td>−0.11</td><td>0.02</td></tr><tr><td><code>L4.down.27</code></td><td>+0.10</td><td>0.01</td></tr></table>
<table><tr><th>writer</th><th>share in g29</th><th>own</th></tr><tr><td><code>L12.down.24</code></td><td>+0.39</td><td>0.17</td></tr><tr><td><code>L13.down.21</code></td><td>+0.28</td><td>0.09</td></tr><tr><td><code>L11.down.10</code></td><td>+0.26</td><td>0.08</td></tr><tr><td><code>emb</code></td><td>+0.22</td><td>0.13</td></tr><tr><td><code>L13.down.18</code></td><td>+0.19</td><td>0.05</td></tr><tr><td><code>L11.down.16</code></td><td>+0.18</td><td>0.04</td></tr><tr><td><code>L13.down.26</code></td><td>−0.16</td><td>0.10</td></tr><tr><td><code>L12.down.7</code></td><td>−0.16</td><td>0.03</td></tr><tr><td><code>L13.down.6</code></td><td>+0.14</td><td>0.05</td></tr><tr><td><code>L15.o.226</code></td><td>−0.12</td><td>0.02</td></tr></table>
<p>Summed by sublayer:</p>
<table><tr><th>sublayer</th><th>g19</th><th>g29</th></tr><tr><td>emb</td><td>+0.32</td><td>+0.22</td></tr><tr><td>L4.mlp</td><td>+0.24</td><td>+0.14</td></tr><tr><td>L8.mlp</td><td>+0.16</td><td>+0.13</td></tr><tr><td>L11.mlp</td><td>+0.39</td><td>+0.43</td></tr><tr><td>L12.mlp</td><td>+0.41</td><td>+0.27</td></tr><tr><td>L13.mlp</td><td>+0.38</td><td>+0.48</td></tr><tr><td>L14.mlp</td><td>−0.25</td><td>−0.12</td></tr><tr><td>L15.attn</td><td>−0.62</td><td>−0.61</td></tr></table>
<p>All MLP and attention sublayers of blocks 0–10 together: +0.37 for <code>g19</code>,
+0.34 for <code>g29</code>. The block-15 attention term with the largest |share| is
<code>L15.o.226</code> for <code>g19</code> and <code>L15.o.226</code> for <code>g29</code>.</p>
<p class="note">Scripts: <code>repr_repo/mod50_l15_writers.py</code> (split; output
<code>&lt;run&gt;/analysis/reader_writer/mod50_p3/l15_writers.json</code>, which also holds the split of the whole
<i>b</i> mod 50 part) and <code>readers/mod50_page.py</code> (this page).</p>
`
},
];
