// Notebook pages of the mod 50 report.
window.NOTEBOOK = [
{
id: "mod50_eq",
title: "Mod 50: the block-15 readers g19 and g29 at =",
date: "2026-10-27",
html: `
<p><b>Summary.</b> At the token “=”, the block-15 gate readers <code>g19</code> and <code>g29</code> carry a
pattern of <i>b</i> mod 50. Their activation there splits exactly into one term per upstream writer. The block-15
attention reading from the token <i>b</i> supplies 0.51–0.64 of it, depending on the reader and the operator.
The other source positions of that attention supply almost nothing. The rest comes from many small writers of blocks
0–14; none of them reaches 0.13. On the read directions of the two readers, the stream at “=” stays in small
clouds up to block 15's attention. After that attention the points spread along a band about 0.5 long, with the
colours of <i>b</i> mod 50 sorted along it, and the source-<i>b</i> write alone draws that band.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {1, …, 400} and op ∈ {+, −}: 320,000 prompts, 160,000 per operator. Positions 0 to 4 are
<code>&lt;BOS&gt;</code>, <i>a</i>, op, <i>b</i> and “=”. The model is the alive-only model: every alive component
of the ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off.
It is rebuilt from its stored activations (float32, every alive component, positions 1–4); position 0 is the same on
every prompt. <b>Nothing in this notebook is centred or detrended</b>: the network has no biases, so zero is the
zero of the activation space.</p>
<ul>
<li>A <b>component</b> has a read vector and a write vector. Its id is
<code>L&lt;block&gt;.&lt;kind&gt;.&lt;index&gt;</code>; <code>g19</code> is <code>L15.gate.19</code>,
<code>g29</code> is <code>L15.gate.29</code>. <i>c</i> denotes one of these two readers, <i>V<sub>c</sub></i> ∈
ℝ<sup>4096</sup> its read vector.</li>
<li>γ ∈ ℝ<sup>4096</sup>: the gain of block 15's post-attention RMSNorm. <b>r<sub>c</sub></b> =
γ ⊙ <i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖ (⊙ the elementwise product): the reader's read direction with
the gain folded in.</li>
<li><i>x</i>: a prompt. <b>pre<sub><i>l</i></sub></b>(<i>x</i>), <b>mid<sub><i>l</i></sub></b>(<i>x</i>) ∈
ℝ<sup>4096</sup>: the residual stream at “=” entering block <i>l</i> and after block <i>l</i>'s attention (the input
of its MLP), <i>l</i> ∈ {0, …, 16} a block index.</li>
<li>ρ(<i>x</i>) = (mean<sub><i>j</i></sub> mid<sub>15,<i>j</i></sub>(<i>x</i>)<sup>2</sup> +
10<sup>−5</sup>)<sup>1/2</sup>, <i>j</i> ∈ {1, …, 4096} the coordinates: the RMS that block 15's RMSNorm divides
by. <b>f<sub>c</sub></b>(<i>x</i>) = r<sub>c</sub> · mid<sub>15</sub>(<i>x</i>) / ρ(<i>x</i>): the reader's inner
activation divided by ‖<i>V<sub>c</sub></i>‖.</li>
<li><b>Writers</b> <i>w</i>: the embedding of “=” (one term, the same on every prompt); every alive attention
<code>o</code> component and every alive MLP <code>down</code> component of blocks 0–14, at “=”; and every alive
<code>o</code> component of block 15 split by the source position <i>s</i> ∈ {0, …, 4} its heads read from, id
<code>L15.o.&lt;index&gt;@s&lt;s&gt;</code>. The split recomputes block 15's attention at “=” and keeps, in every
head, only the pattern weight on source <i>s</i> times the value at <i>s</i>; the five pieces add up to the
component's activation. 2440 writers in all. <i>t<sub>w</sub></i>(<i>x</i>) is the writer's
activation at “=”, <i>U<sub>w</sub></i> its write vector, and mid<sub>15</sub> = Σ<sub><i>w</i></sub>
<i>t<sub>w</sub></i> <i>U<sub>w</sub></i> exactly.</li>
<li><b>e<sub>w</sub></b>(<i>x</i>) = <i>t<sub>w</sub></i>(<i>x</i>) (r<sub>c</sub> · <i>U<sub>w</sub></i>) /
ρ(<i>x</i>): writer <i>w</i>'s term in the reader's activation, so f<sub>c</sub> = Σ<sub><i>w</i></sub>
e<sub>w</sub> exactly. ρ is the reader's own per-prompt RMS, shared by every term.</li>
<li>Sums over <i>x</i> run over the 160,000 prompts of one operator, uncentred.
<b>share<sub><i>w</i></sub></b> = Σ<sub><i>x</i></sub> e<sub>w</sub> f<sub>c</sub> / Σ<sub><i>x</i></sub>
f<sub>c</sub><sup>2</sup>: writer <i>w</i>'s signed share of the activation, constant part included; the shares
add up to 1 over the writers, and a negative share opposes the activation. <b>own<sub><i>w</i></sub></b> =
Σ<sub><i>x</i></sub> e<sub>w</sub><sup>2</sup> / Σ<sub><i>x</i></sub> f<sub>c</sub><sup>2</sup>: the size of the
term next to the activation. <b>mean<sub><i>w</i></sub></b>: the mean of e<sub>w</sub> over the prompts.
<b>r²</b>(<i>S</i>) = 1 − Σ<sub><i>x</i></sub> (f<sub>c</sub> − Σ<sub><i>w</i> ∈ <i>S</i></sub>
e<sub>w</sub>)<sup>2</sup> / Σ<sub><i>x</i></sub> f<sub>c</sub><sup>2</sup> for a set <i>S</i> of writers
(uncentred): how much of the activation the writers of <i>S</i> rebuild on their own.</li>
<li>Periodicity shares (from <code>periodicity.json</code>, addition prompts): share<sub>τ</sub> is the energy of
the part of f<sub>c</sub> that is a function of <i>b</i> mod 100 with exact period τ (τ a divisor of 100, τ = 1 the
constant), as a share of Σ f<sub>c</sub><sup>2</sup>.</li>
</ul>

<h3>1. The readers g19 and g29 at “=” and where their input comes from</h3>
<p>Periodicity shares in <i>b</i> at “=”: <code>g19</code> τ = 1: 0.49, τ = 5: 0.00, τ = 10: 0.01, τ = 25: 0.00, τ = 50: 0.49, τ = 100: 0.00; periods [50]; <code>g29</code>
τ = 1: 0.61, τ = 5: 0.00, τ = 10: 0.00, τ = 25: 0.00, τ = 50: 0.38, τ = 100: 0.00; periods [50].</p>
<p><b>Checks.</b> The rebuilt activations of the block-15 <code>o</code> components (summed over the sources) match
the stored ones to 0.014 of their largest value, and the rebuilt f<sub>c</sub> matches the stored
reader activations with r² = 0.99999. Σ e<sub>w</sub> = f<sub>c</sub> holds on every
prompt to float64 precision.</p>
<div class="tw"><table><tr><th></th><th>g19, a + b</th><th>g19, a - b</th><th>g29, a + b</th><th>g29, a - b</th></tr><tr><td>mean of f<sub>c</sub></td><td>−0.97</td><td>−0.87</td><td>−1.37</td><td>−1.02</td></tr><tr><td>rms of f<sub>c</sub></td><td>1.39</td><td>1.25</td><td>1.75</td><td>1.45</td></tr><tr><td>block-15 attention, source BOS</td><td>+0.00</td><td>+0.01</td><td>−0.02</td><td>−0.02</td></tr><tr><td>block-15 attention, source a</td><td>+0.00</td><td>+0.00</td><td>−0.00</td><td>+0.00</td></tr><tr><td>block-15 attention, source op</td><td>−0.00</td><td>+0.00</td><td>+0.01</td><td>−0.00</td></tr><tr><td>block-15 attention, source b</td><td>+0.64</td><td>+0.58</td><td>+0.51</td><td>+0.63</td></tr><tr><td>block-15 attention, source =</td><td>−0.00</td><td>+0.00</td><td>+0.00</td><td>−0.00</td></tr><tr><td>blocks 0–14, every writer</td><td>+0.34</td><td>+0.39</td><td>+0.49</td><td>+0.39</td></tr><tr><td>embedding of “=”</td><td>+0.01</td><td>+0.02</td><td>+0.00</td><td>+0.00</td></tr><tr><td>r² of block-15 attention from b alone</td><td>0.74</td><td>0.64</td><td>0.61</td><td>0.73</td></tr><tr><td>r² of the 1 writers of largest |share|</td><td>0.35</td><td>0.26</td><td>0.21</td><td>0.22</td></tr><tr><td>r² of the 5 writers of largest |share|</td><td>0.68</td><td>0.59</td><td>0.48</td><td>0.65</td></tr><tr><td>r² of the 10 writers of largest |share|</td><td>0.76</td><td>0.58</td><td>0.73</td><td>0.81</td></tr><tr><td>r² of the 20 writers of largest |share|</td><td>0.78</td><td>0.74</td><td>0.92</td><td>0.92</td></tr><tr><td>r² of the 60 writers of largest |share|</td><td>0.84</td><td>0.86</td><td>0.98</td><td>0.97</td></tr></table></div>
<figure><img src="fig/mod50_eq_groups.png" alt="shares by sublayer for g19 and g29, both operators">
<figcaption>Shares summed by group: the embedding of “=”, the attention and the MLP of each block 0–14, and
block 15's attention by source position (blue positive, red negative). One row per reader and operator.</figcaption>
</figure>
<p>The writers with the largest |share|:</p>
<h4>g19, prompts a + b</h4><table><tr><th>writer</th><th>share</th><th>own</th><th>mean</th></tr><tr><td><code>L15.o.124@s3</code></td><td>+0.22</td><td>0.08</td><td>−0.33</td></tr><tr><td><code>L15.o.80@s3</code></td><td>+0.11</td><td>0.02</td><td>−0.12</td></tr><tr><td><code>L15.o.59@s3</code></td><td>+0.11</td><td>0.03</td><td>−0.17</td></tr><tr><td><code>L15.o.19@s3</code></td><td>+0.10</td><td>0.03</td><td>−0.18</td></tr><tr><td><code>L7.down.1</code></td><td>+0.09</td><td>0.02</td><td>−0.18</td></tr><tr><td><code>L15.o.140@s3</code></td><td>+0.08</td><td>0.01</td><td>−0.07</td></tr><tr><td><code>L12.down.17</code></td><td>+0.07</td><td>0.01</td><td>−0.15</td></tr><tr><td><code>L3.down.5</code></td><td>+0.07</td><td>0.01</td><td>−0.14</td></tr><tr><td><code>L8.down.1</code></td><td>−0.06</td><td>0.01</td><td>+0.12</td></tr><tr><td><code>L15.o.88@s3</code></td><td>−0.05</td><td>0.02</td><td>+0.13</td></tr><tr><td><code>L13.down.13</code></td><td>+0.05</td><td>0.00</td><td>−0.09</td></tr><tr><td><code>L5.o.49</code></td><td>+0.03</td><td>0.00</td><td>−0.07</td></tr><tr><td><code>L13.down.3</code></td><td>−0.03</td><td>0.00</td><td>+0.06</td></tr><tr><td><code>L15.o.152@s3</code></td><td>+0.03</td><td>0.00</td><td>−0.04</td></tr><tr><td><code>L15.o.126@s3</code></td><td>+0.03</td><td>0.00</td><td>−0.05</td></tr></table><h4>g19, prompts a - b</h4><table><tr><th>writer</th><th>share</th><th>own</th><th>mean</th></tr><tr><td><code>L15.o.124@s3</code></td><td>+0.15</td><td>0.05</td><td>−0.18</td></tr><tr><td><code>L12.down.17</code></td><td>+0.12</td><td>0.03</td><td>−0.22</td></tr><tr><td><code>L14.down.82</code></td><td>+0.11</td><td>0.02</td><td>−0.19</td></tr><tr><td><code>L15.o.59@s3</code></td><td>+0.09</td><td>0.03</td><td>−0.12</td></tr><tr><td><code>L15.o.19@s3</code></td><td>+0.09</td><td>0.03</td><td>−0.15</td></tr><tr><td><code>L15.o.80@s3</code></td><td>+0.09</td><td>0.02</td><td>−0.07</td></tr><tr><td><code>L15.o.140@s3</code></td><td>+0.07</td><td>0.01</td><td>−0.05</td></tr><tr><td><code>L7.down.1</code></td><td>+0.07</td><td>0.01</td><td>−0.13</td></tr><tr><td><code>L3.down.5</code></td><td>+0.07</td><td>0.01</td><td>−0.12</td></tr><tr><td><code>L13.down.13</code></td><td>+0.06</td><td>0.01</td><td>−0.11</td></tr><tr><td><code>L13.o.371</code></td><td>−0.06</td><td>0.01</td><td>+0.10</td></tr><tr><td><code>L8.down.17</code></td><td>−0.05</td><td>0.00</td><td>+0.09</td></tr><tr><td><code>L2.down.3</code></td><td>+0.05</td><td>0.00</td><td>−0.09</td></tr><tr><td><code>L3.down.16</code></td><td>−0.05</td><td>0.00</td><td>+0.08</td></tr><tr><td><code>L8.down.1</code></td><td>−0.04</td><td>0.00</td><td>+0.08</td></tr></table><h4>g29, prompts a + b</h4><table><tr><th>writer</th><th>share</th><th>own</th><th>mean</th></tr><tr><td><code>L15.o.226@s3</code></td><td>+0.12</td><td>0.03</td><td>−0.16</td></tr><tr><td><code>L15.o.124@s3</code></td><td>−0.10</td><td>0.04</td><td>+0.28</td></tr><tr><td><code>L15.o.143@s3</code></td><td>+0.10</td><td>0.02</td><td>−0.12</td></tr><tr><td><code>L15.o.51@s3</code></td><td>+0.10</td><td>0.02</td><td>−0.11</td></tr><tr><td><code>L14.down.0</code></td><td>+0.09</td><td>0.01</td><td>−0.20</td></tr><tr><td><code>L8.down.1</code></td><td>+0.08</td><td>0.01</td><td>−0.19</td></tr><tr><td><code>L15.o.88@s3</code></td><td>+0.07</td><td>0.01</td><td>−0.14</td></tr><tr><td><code>L15.o.59@s3</code></td><td>−0.07</td><td>0.03</td><td>+0.18</td></tr><tr><td><code>L15.o.56@s3</code></td><td>+0.06</td><td>0.02</td><td>−0.10</td></tr><tr><td><code>L15.o.74@s3</code></td><td>+0.06</td><td>0.01</td><td>−0.08</td></tr><tr><td><code>L13.down.3</code></td><td>+0.05</td><td>0.00</td><td>−0.12</td></tr><tr><td><code>L15.o.110@s3</code></td><td>+0.05</td><td>0.01</td><td>−0.08</td></tr><tr><td><code>L15.o.19@s3</code></td><td>−0.05</td><td>0.01</td><td>+0.12</td></tr><tr><td><code>L9.down.8</code></td><td>−0.05</td><td>0.00</td><td>+0.11</td></tr><tr><td><code>L13.o.1</code></td><td>+0.05</td><td>0.00</td><td>−0.11</td></tr></table><h4>g29, prompts a - b</h4><table><tr><th>writer</th><th>share</th><th>own</th><th>mean</th></tr><tr><td><code>L15.o.51@s3</code></td><td>+0.12</td><td>0.02</td><td>−0.11</td></tr><tr><td><code>L15.o.143@s3</code></td><td>+0.11</td><td>0.02</td><td>−0.09</td></tr><tr><td><code>L15.o.226@s3</code></td><td>+0.10</td><td>0.02</td><td>−0.08</td></tr><tr><td><code>L15.o.56@s3</code></td><td>+0.07</td><td>0.02</td><td>−0.07</td></tr><tr><td><code>L15.o.158@s3</code></td><td>+0.06</td><td>0.01</td><td>−0.08</td></tr><tr><td><code>L13.o.371</code></td><td>+0.06</td><td>0.01</td><td>−0.12</td></tr><tr><td><code>L8.down.1</code></td><td>+0.06</td><td>0.01</td><td>−0.13</td></tr><tr><td><code>L15.o.88@s3</code></td><td>+0.06</td><td>0.01</td><td>−0.10</td></tr><tr><td><code>L15.o.74@s3</code></td><td>+0.06</td><td>0.01</td><td>−0.06</td></tr><tr><td><code>L6.down.71</code></td><td>−0.05</td><td>0.01</td><td>+0.11</td></tr><tr><td><code>L15.o.19@s3</code></td><td>−0.05</td><td>0.01</td><td>+0.10</td></tr><tr><td><code>L11.down.120</code></td><td>+0.05</td><td>0.01</td><td>−0.11</td></tr><tr><td><code>L15.o.124@s3</code></td><td>−0.05</td><td>0.03</td><td>+0.15</td></tr><tr><td><code>L15.o.59@s3</code></td><td>−0.05</td><td>0.02</td><td>+0.13</td></tr><tr><td><code>L14.down.0</code></td><td>+0.04</td><td>0.00</td><td>−0.09</td></tr></table>

<h3>2. The stream at “=” on the read directions of g19 and g29, blocks 0–15</h3>
<p>Each panel plots, for 20,000 random prompts of one operator, the two coordinates mid<sub><i>l</i></sub> ·
r<sub>19</sub> and mid<sub><i>l</i></sub> · r<sub>29</sub> (raw: not divided by the RMS of the stream), coloured by
<i>b</i> mod 50 on the viridis scale. Blocks 0–14 show mid<sub><i>l</i></sub>; the last two panels show
pre<sub>15</sub> (before block 15's attention) and mid<sub>15</sub> (what block 15's MLP reads, up to its RMS).
Each panel has its own axis limits.</p>
<figure><img src="fig/stream_add.png" alt="stream at = on r19 and r29, blocks 0 to 15, addition">
<figcaption>Prompts <i>a</i> + <i>b</i>.</figcaption></figure>
<figure><img src="fig/stream_sub.png" alt="stream at = on r19 and r29, blocks 0 to 15, subtraction">
<figcaption>Prompts <i>a</i> − <i>b</i>.</figcaption></figure>

<h3>3. What block 15 writes along these directions</h3>
<p>The same coordinates for block 15's writes: its attention write mid<sub>15</sub> − pre<sub>15</sub>, the same
split by source position (BOS, <i>a</i>, op, <i>b</i>, “=”), and its MLP write pre<sub>16</sub> − mid<sub>15</sub>.
The first panel repeats mid<sub>15</sub>. All panels of a figure share the same axis limits.</p>
<figure><img src="fig/writes15_add.png" alt="block 15 writes on r19 and r29, addition">
<figcaption>Prompts <i>a</i> + <i>b</i>.</figcaption></figure>
<figure><img src="fig/writes15_sub.png" alt="block 15 writes on r19 and r29, subtraction">
<figcaption>Prompts <i>a</i> − <i>b</i>.</figcaption></figure>
<p class="note">Scripts: <code>readers/mod50_eq.py</code> (rebuild, split and plots; outputs in
<code>&lt;run&gt;/analysis/reader_writer/mod50_eq/</code>) and <code>readers/mod50_eq_page.py</code> (this page).</p>
`
},
];
