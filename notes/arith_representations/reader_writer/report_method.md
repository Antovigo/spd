# Which readers drive each writer? Exact attributions inside one MLP block of a decomposed model

*Method write-up for the reader → writer analysis of the addsub -08 decomposition, tested on the
MLP of layer 18. This post explains the method; the results are reported separately once the
L18 run finishes.*

## 1. The question

We have taken a large language model apart into small pieces and want to know how the pieces
talk to each other.

The model is **Llama 3.1 8B**, a transformer with 32 layers ("blocks"). Each block has an
attention part and an MLP part. We study it on a narrow task: prompts of the form
`a + b =` and `a - b =`, where the operands `a` and `b` are integers from 1 to 400. That gives
2 × 400 × 400 = 320,000 prompts, each five tokens long: a beginning-of-sequence token, `a`, the
operator, `b`, and `=`. The model's job is to predict the result after `=`.

The pieces come from **parameter decomposition** (the run is called *addsub -08*, internal id
`p-ba8b0c08`, checkpoint step 40000). Parameter decomposition rewrites every weight matrix of
the model as a sum of many rank-one matrices called **components**. A component `c` of a weight
matrix is the outer product of two vectors:

* a **read vector** `V_c`, the direction the component looks for in its input;
* a **write vector** `U_c`, the direction it adds to its output.

When the input to that matrix is a vector `x`, the component computes one number, its **inner
activation** `z_c = x · V_c` (a real number), and contributes `z_c U_c` to the matrix's output.
The decomposition is trained so that, on any given prompt, only a few components are needed:
a learned "causal importance" (CI) function says which ones, and the rest can be switched off
without changing the model's prediction much.

An MLP block of Llama has three weight matrices:

* **gate_proj** and **up_proj** read the (normalised) residual stream `x`, a vector of length
  4096, and each produce a vector of length `d_mlp` = 14336, one entry per *neuron*;
* **down_proj** reads the neurons and writes back into the residual stream.

We call the components of gate_proj and up_proj **readers** (they read from the residual
stream) and the components of down_proj **writers** (they write into it). The question of this
post is:

> For each writer, which readers do we need to know in order to predict the writer's activation?

The prior expectation is that each writer depends on only a handful of readers. If that is true,
a block's computation can be described as a small wiring diagram: these three readers feed this
writer, those two feed that one.

## 2. The decomposed model we analyse

To get clean, discrete answers we analyse a slightly simplified model rather than the original:
the **rounded-CI decomposed model** of the CI filter `addsub-08-last-pos` (a fine-tuned CI
function for this task). On each prompt and each token position, each component is either fully
on or fully off:

* its **mask** `m_c` is 1 if its CI, computed on the original model's activations, exceeds 0.01,
  and 0 otherwise;
* the model runs with only the on components; the leftover weights that the components do not
  explain (the "weight delta") are switched off.

In this model, the MLP block of a given layer computes, at each prompt and token position, exactly
the following. Write

* `s_r = m_r z_r` for the **masked activation** of reader `r` (a real number, 0 when the reader is
  off);
* `G = Σ_{gate readers r} s_r U_r`, the gate pre-activation, a vector of length 14336;
* `H = Σ_{up readers r} s_r U_r`, the up pre-activation, a vector of length 14336;
* `silu(ξ) = ξ / (1 + e^{−ξ})` for a real number `ξ`, applied entry by entry.

The neuron values are `silu(G) ⊙ H`, where `⊙` multiplies two vectors entry by entry. Writer `w`
then has inner activation

$$ y_w = V_w \cdot \big(\mathrm{silu}(G) \odot H\big) = \sum_{n=1}^{14336} V_w[n]\, \mathrm{silu}(G_n)\, H_n , $$

where `n` indexes the neurons, `V_w[n]` is entry `n` of the writer's read vector, and `G_n`, `H_n`
are entry `n` of `G` and `H`. When the writer is on, it adds `y_w U_w` to the residual stream.

This equation is exact. There is no approximation and no hidden input: the writer's activation
is a known, deterministic function of the readers' masked activations. The question "which
readers matter for this writer?" is therefore a question about this one function. All that is
left to decide is how to measure the importance of each input.

At the "=" token of layer 18, 126 gate readers, 158 up readers and 158 writers are on for at
least one prompt. At the "b" token the counts are 92, 80 and 148.

## 3. Why the obvious measures are not enough

**Weights alone.** One could score a reader-writer pair by how much the reader's write vector
overlaps with the writer's read vector, for instance the size of `V_w ⊙ U_r`. This ignores how
large the reader's activation actually is on the task, and it ignores the nonlinearity: a gate
reader matters only when silu is in a range where changing `G` changes the output. We keep this
measure as a baseline (section 6).

**Correlation.** Readers that encode the same quantity correlate with each other, so a reader
can correlate strongly with a writer without the writer using it.

**A single gradient.** The derivative `∂y_w / ∂s_r` at a prompt says how sensitive the writer is
to a small change of the reader there. It is local: silu is flat for very negative inputs and
nearly linear for positive ones, so the derivative at one point can badly misrepresent what the
reader contributes overall.

## 4. Integrated gradients: an exact split of each activation

**Integrated gradients** fixes the last problem. Choose a reference point, here the point where
every reader is off (`s_r = 0` for all `r`). There `G = H = 0`, so every writer's activation is 0.
Now move from the reference to the actual prompt along a straight line: scale every reader's
activation by a factor `λ` that goes from 0 to 1. Accumulate, along the way, how much each
reader's change contributes to the change of `y_w`:

$$ \varphi_r(w) = s_r \int_0^1 \frac{\partial y_w}{\partial s_r}\Big|_{\lambda s}\, d\lambda , $$

where `λ s` means every reader's activation multiplied by `λ`. The key property is
**completeness**: on every prompt,

$$ \sum_r \varphi_r(w) = y_w . $$

So `φ_r(w)` is an exact split of the writer's activation into one share per reader. Nothing is
left over.

For our MLP the integral has a closed form, because along the path the gate pre-activation is
`λG` and the up pre-activation is `λH`. With two scalar functions of a real number `ξ`,

$$ P(\xi) = \int_0^1 \mathrm{silu}(\lambda \xi)\, d\lambda , \qquad Q(\xi) = \int_0^1 \lambda\, \mathrm{silu}'(\lambda \xi)\, d\lambda , $$

(`silu'` is the derivative of silu), the attributions are

* for an **up** reader: `φ_r(w) = s_r Σ_n V_w[n] U_r[n] P(G_n)`;
* for a **gate** reader: `φ_r(w) = s_r Σ_n V_w[n] U_r[n] H_n Q(G_n)`.

Integrating by parts gives `Q(ξ) = (silu(ξ) − P(ξ)) / ξ`, and summing the two lines over all readers
recovers `Σ_n V_w[n] H_n silu(G_n) = y_w`, which is completeness. In the code, `P` and `Q` are
evaluated with 24-point Gauss–Legendre quadrature. On a small random test case the closed form
agreed with a brute-force integral over 2001 path steps to 3 × 10⁻⁶, and completeness holds to
3 × 10⁻⁷ relative error.

**A worked example.** Take one gate reader `g` and one up reader `u` and a single neuron with
`V_w = 1`, `U_g = U_u = 1`. Then `y_w = silu(s_g) s_u`. If `s_g = 3` and `s_u = 2`,
`y_w = 2 silu(3) ≈ 5.71`. The up reader gets `φ_u = 2 P(3)`; the gate reader gets
`φ_g = 2 · 3 · Q(3) = 2 (silu(3) − P(3))`. They add up to `2 silu(3)`. The up reader's share
depends on how open the gate was *on average along the path*, not only at the end.

## 5. From attributions to a score per reader-writer pair

A writer is only written into the stream when it is on, so we only care about prompts where
`m_w = 1`. Over those prompts we average the squared attribution, and normalise per writer:

$$ \text{score}(r, w) = \mathbb{E}\big[\varphi_r(w)^2 \mid m_w = 1\big], \qquad \text{share}(r, w) = \frac{\text{score}(r, w)}{\sum_{r'} \text{score}(r', w)} . $$

Here `𝔼[· | m_w = 1]` is the average over prompts where the writer is on, and later `Var` is the variance over the same prompts. The square treats positive and negative contributions alike, and it rewards readers that
contribute a lot on some prompts even when they cancel on average. Readers are ranked, per
writer, by this score. The averages use a random half of the 320,000 prompts (the **fit half**);
the other half is kept for testing.

**Constant readers.** Many gate and up readers take nearly the same value on every prompt. They
are central to the computation: they set the operating point of the neurons, much like a bias.
Because the reference point of integrated gradients is "everything off", a large constant reader
gets a large attribution and a high rank, so the ranking counts it.

## 6. Checking that the top readers are enough

A ranking is only useful if the top of the list really determines the writer. We test this
directly, writer by writer:

1. Keep the `k` highest-ranked readers (`k` a positive integer) at their true values on each
   prompt.
2. Replace every other reader by its mean `s̄_r` over the fit half's prompts where the writer is
   on. This is what "not knowing" a reader means here: we use the value we would guess without
   looking.
3. Recompute the writer's activation exactly with the formula of section 2, giving a prediction
   `ŷ_w`.
4. Measure the fraction of the writer's variance this explains, on 20,000 held-out prompts
   (restricted to those where the writer is on):

$$ R^2_k = 1 - \frac{\mathbb{E}\big[(y_w - \hat y_w)^2\big]}{\mathrm{Var}(y_w)} . $$

As `k` grows from 1 to the number of readers, `R²_k` rises to 1. The **support** of a writer is
its top `k` readers for the smallest `k` reaching `R²_k ≥ 0.9`; we also record the smallest `k`
for `R²_k ≥ 0.99`. A small support confirms the "few readers per writer" expectation for that
writer; a large one refutes it. When a target is never reached, the writer is flagged.

Note that in this test a reader that is truly constant costs nothing to leave out, since its mean
equals its value. Constant readers that rank high still take a place in the support, so the
support size counts them too.

Two comparisons calibrate the ranking:

* **Weights-only ranking**: score `(r, w)` by the reader's typical size times the overlap of the
  vectors, `rms(s_r) · ‖V_w ⊙ U_r‖`, where `rms` is the root-mean-square over the writer's on
  prompts and `‖·‖` the Euclidean length. If the integrated-gradients ranking reaches a given
  `R²` with far fewer readers, the data-dependent part of the score is doing real work.
* **Greedy subset search**: for the 30 writers with the largest contribution to the block's
  output, build the support one reader at a time, each time adding the reader that most improves
  `R²` (fit on 1,500 prompts, tested on held-out prompts), up to 15 readers. This is close to the
  best possible subset of each size. If the integrated-gradients curve sits close to the greedy
  curve, the cheap ranking is near-optimal.

## 7. A compact predictor for each writer

The test of section 6 also produces a small model of each writer. With support `S` (a set of
readers), the left-out readers' mean contributions collapse into two fixed vectors of length
14336, the **offsets**

$$ O^{\text{gate}}_w = \sum_{\text{gate } r \notin S} \bar s_r U_r , \qquad O^{\text{up}}_w = \sum_{\text{up } r \notin S} \bar s_r U_r , $$

and the writer's activation is predicted from its support readers alone:

$$ \hat y_w = V_w \cdot \Big(\mathrm{silu}\big(O^{\text{gate}}_w + \textstyle\sum_{\text{gate } r \in S} s_r U_r\big) \odot \big(O^{\text{up}}_w + \textstyle\sum_{\text{up } r \in S} s_r U_r\big)\Big) . $$

The analysis stores the support, the offsets and the means for both targets (`R² ≥ 0.9` and
`≥ 0.99`), so any writer's activation can be computed from the values of its own readers.

## 8. The explorer applet

An interactive page shows the result for one block:

* **Writer view.** Pick a writer. The page shows its activation over the full 400 × 400 grid of
  operand pairs, one grid for `a + b` and one for `a − b` (rows `a` = 1 to 400 from top to bottom,
  columns `b` = 1 to 400 from left to right, hatched where the writer is off), and its `R²_k`
  curve for the three rankings. Below, one card per support reader shows that reader's two grids,
  the other writers whose support contains it, and a **pair plot**.
* **Pair plot.** The horizontal axis is one reader's activation `s_r`, swept over its range on
  the writer's on prompts; the vertical axis is the predicted `ŷ_w`. A second support reader `o`
  is held at its 10, 30, 50, 70 and 90 % quantiles, one coloured line each, and every other reader
  is at its mean. A dropdown picks `o` when the support has more than two readers. Parallel lines
  mean the two readers add up; fanning or crossing lines mean they interact, for example a gate
  reader that switches an up reader's effect on or off.
* **Reader view.** Pick a reader to see the writers whose support contains it, each with its
  grids and the same pair plot.
* Hovering a grid cell shows the prompt and its value, marks the same cell in every grid on the
  page, and rings the corresponding point in every pair plot.

## 9. What was computed, and how

* **Extraction** (`extract_layer.py`, one GPU job). The decomposed model is run on all 320,000
  prompts. For the block's gate, up and down components it stores the inner activations `z_c`
  and the masks `m_c` at all five positions, and it captures the true `G`, `H` and neuron values on
  400 prompts so the formula of section 2 can be checked against the model.
* **Attribution** (`attribute.py`, one GPU job), at the "=" and "b" positions separately: it
  computes the integrated-gradients attributions and scores on the fit half (checking
  completeness), the `R²_k` curves on 20,000 held-out prompts for `k` from 1 to every reader, the
  greedy paths, the supports and the offsets. It also reports how well the recomputed `y_w`
  matches the stored activations from the model.
* **Applet data** (`build_app.py`, CPU job): grids at 8-bit resolution per component and
  operation, read and write vectors, and per-writer means, offsets and reader quantiles.

## 10. Limitations

* **The reference point matters.** Integrated gradients from "all readers off" counts constant
  readers as important. A different reference (for example every reader at its mean) would rank
  only readers whose *variation* matters. The `R²_k` test does not depend on this choice, but the
  order in which readers enter the support does.
* **Correlated readers share credit.** When two readers carry the same information, either can
  stand in for the other, so the support may name one where the model uses both. The greedy
  comparison shows how much this costs in `R²`, not which reader is "the real one".
* **Supports depend on the threshold.** `R² ≥ 0.9` and `R² ≥ 0.99` can give very different `k`,
  especially for writers with long tails of small contributions.
* **Writer-on prompts only.** Everything is measured where the writer is on. A reader that only
  decides *whether* a writer fires acts through the CI function, not through `y_w`, and is not
  captured here.
* **One simplified model.** The rounded-CI decomposed model differs from the original (KL
  divergence of the last-position prediction 0.070 on average over the pool). Statements are
  about this model.

Code: `notes/arith_representations/reader_writer/` on branch `experiment/arith_representations`.
