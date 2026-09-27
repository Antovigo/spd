# What the readers read

*A label-free atlas of the quantities Llama-3.1-8B reads while it adds and subtracts, how they are
built, and whether they matter.*

Interactive version: https://claude.ai/artifact/FHwWcPdX6cjPSech1iGR3Z. Code:
`param_decomp/arith_repr/isa/`. Outputs: `<run>/analysis/arith_repr/atlas/`. Decomposition p-ba5a0c05,
step 40000, ceiling filter `addsub-05-filter-last-pos-ceiling`.

## In short

- **Readers read few, clean quantities.** Without using any labels, the components that read the
  residual stream can be summarised by 832 quantities at 256 read points: circles such as b mod 50,
  binary variables such as the operation, and a few larger frames. Named afterwards, they tell a
  readable story: b is copied to the `=` position at layer 15, a at layer 16, and the result is
  assembled from their periodic codes from layer 19 on.
- **Their construction can be traced exactly.** Every quantity is a sum of identified component
  writes, and the writers compute it from earlier quantities: for example, the result's mod-20 code
  is a product of a's and b's mod-20 codes.
- **They are causal, at the component that reads them.** Swapping the quantities of an MLP's input
  from another prompt reproduces 51-92 % of that MLP's response to the other prompt through the middle
  of the network. They do not move the final answer, because later layers read the operands again.
- **What they miss is load-bearing, and it is task information.** The part of the readers the
  quantities do not explain carries result information in a code of up to 17 dimensions, too large for
  the method to find. Removing it at `=` halves the accuracy on additions; taking it from another
  prompt with the same result leaves the accuracy unchanged. It is not a correction for interference.

## 1. The question

Prompts are `<BOS> a op b =` with a and b from 1 to 100 and op + or − (20,000 prompts). A *read
point* is a place where components read the residual stream: the q, k and v components of one
attention layer, or the gate and up components of one MLP, at one of the positions a, op, b or `=`.
Each such component is a *reader*; its inner activation on a prompt is how much it reads.

We want, at every read point, a short list of **quantities** that together explain what the readers
read. A quantity is an unlabeled variable with a value on every prompt, and a shape:

- a **circle**, such as b mod 50: the prompt sits at angle 2π·b/50 on a circle, so b = 3 and b = 53
  are the same point and b = 28 is opposite;
- a **binary variable**, such as the operation (+1 for addition, −1 for subtraction), or b's mod-20
  square wave (+1 when b mod 20 is below 10, −1 otherwise);
- a **simplex**, such as a variable with three values placed at the three corners of a triangle.

For each quantity we want where it is written in the stream, which readers read it, whether the same
quantity is read elsewhere, how it is built, and whether the network uses it. Labels (a, b, op, the
result) are only used afterwards, to name and check what was found.

## 2. The model we analyse

The analysis runs on the **components-only model**: the model rebuilt from the decomposition's
components that are active on each prompt (causal importance above 0.01), without the weight delta
(last-position KL to the original model 0.05). Its readers carry the same codes as the original
model's, with much less background; the original model is kept for comparison in the interactive
version. It answers 92.8 % of the additions and 22 % of the subtractions correctly, so the accuracy
numbers below are on additions.

One fact makes the analysis possible: **readers are linear in the stream.** A reader's inner
activation times the stream's RMS is `h_c = x · (g ⊙ V_c)`, a fixed linear read of the raw stream x
(g is the norm's gain, V_c the reader's input vector).

## 3. Finding quantities without labels

Three observations turn the question into a search:

- **Each reader reads very few quantities.** On the a × b grids, almost every reader responds to one
  period, sometimes in both operands. So a reader's direction points into one quantity's subspace,
  and each reader is a good starting point.
- **Most quantities are spheres.** A binary variable is ±1, a circle has a constant radius, a simplex
  puts its values at the same distance from its centre: inside its own subspace, every prompt has the
  same norm. Random mixtures of quantities do not.
- **Uncorrelated quantities are orthogonal after whitening.** Two periods of b are uncorrelated over
  the prompts, so once the readers are whitened, their subspaces are perpendicular.

**Sphere pursuit** (`atlas.py`, `pursuit.py`). At one read point: multiply the readers by the stream
RMS, drop the constant ones, keep the dimensions that beat shuffled data (parallel analysis) and
whiten them. Start from the strongest reader not yet explained; for k = 1 to 4, fit the k-dimensional
frame, started at the reader's direction, in which the prompts' squared norm varies least,
`J = Var(|z|²) / (2k)` (0 for a circle, a binary variable or a simplex, 1 for Gaussian noise). Keep the
frame if J < 0.5, split it if it holds two independent spheres, remove it, and move to the next
reader. Each quantity comes with its coordinates on every prompt, its **pattern** `E[(x − x̄) z]` (the
direction along which the stream moves with it) and each reader's share of it.

**The held-out check.** At position a the stream depends only on a, so the 20,000 prompts collapse
onto 100 distinct points. A frame fitted to 100 points in 20 dimensions can make them almost equally
long by chance. So frames are fitted on half of the distinct prompts and accepted on the other half:
a real circle stays a circle on values it was not fitted to, a fitted accident does not.

**Does it work?** On synthetic readers with known quantities (`synth.py`), the circles of a and b mod
50 and mod 10 and of (a + b) mod 10 come back as 2-dimensional frames at subspace overlap 0.98-1.00,
and b's square wave, a linear ramp in a and a checkerboard as 1-dimensional ones at 0.99.

Two more steps connect read points. The **same code** appears at two read points when their
quantities have the same dimension and are linear images of each other (all canonical correlations
≥ 0.8). **Names** come last: the label (op, a, b, a + b, a − b or the result) that explains most of a
quantity's variance, with its dominant period.

## 4. What the atlas finds

At the 256 read points of the components-only model, the pursuit finds 832 quantities (709 in the
original model), which form 616 codes; 20 codes are read at two points or more. On average the
quantities explain 45 % of the readers' loading energy (32 % in the original model).

Followed through the network, the codes tell the story of the computation:

1. **The operation** becomes a binary flag at positions op and b in layer 0 and is read at 173 read
   points, at every layer.
2. **The operands are encoded at their own positions:** a mod 100 at positions a and op (L1-L15),
   b mod 100 and b mod 50 at position b (L2-L18).
3. **They are copied to `=`**: b's codes from L15, a's from L16. At L17-L18 the MLP inputs at `=` hold
   the operands' periodic codes: a's parity, a mod 5, 10 and 50, b mod 2, 5, 20 and 100.
4. **The result appears from L19** as codes of the sum and difference (result mod 2, mod 10, mod 20),
   and from about L24 as codes of the result that mix several periods.

The interactive version maps every code across layers and positions, and shows each layer's
quantities with their grids and top readers.

## 5. How the codes are built

In the components-only model, the stream at a position is exactly the token embedding plus the writes
of the active o and down components before it, `x = e + Σ_c h_c m_c U_c` (m_c the mask, U_c the write
vector; checked to within 1 %). Each quantity is a fixed linear read of the stream,
`z = (x − x̄) · F`, so each writer's share of it is exact, and the shares of the embedding and of all
writers add up to 1 (0.999-1.001 over the 832 quantities; `writers.py`).

For each writer with at least 15 % of a quantity, a regression asks what it computes it from, in the
form the component can compute: linear in the quantities for an attention writer (its values are
linear in the source streams), degree 2 for an MLP writer (its neurons are `silu(g) · u`, so products
of two quantities are within reach).

- **Operation.** Head 10 of layer 0 writes the flag at position b (L0.o.c32 and c0, 98 %), linearly
  from the flag at position op (R² 0.999). At `=`, MLP components rewrite it layer after layer, each
  from the flag already there.
- **b.** At position b its codes come from the embedding and the L0 MLP (b mod 100 at L2: 37 % and
  48 %). Head 13 of layer 15 copies them to `=`: 81 % of b mod 50 and 99 % of b's mod-20 square wave at
  the L15 MLP input, linear in b's codes at position b.
- **a.** Head 21 of layer 16 copies a's codes to `=`: 50-97 % of them at the L16-L18 MLP inputs.
- **The result.** L19.down.c6 writes 52 % of the result's mod-20 code at the L20 input; its activation
  is a product of b's and a's mod-20 square waves (R² 0.83 for the pair, 0.32 and 0.08 alone).
  L17.down.c67 computes the result's parity from a's and b's parities (0.89). L18.down.c4 writes 61 %
  of the result's parity, but the quantities at its input predict it poorly (0.16). Later MLPs refresh
  the result codes from themselves (L21.down.c1, 0.90).

## 6. Do the quantities matter?

Being read is not being used. The test is an **interchange**: give a quantity the value it has on
another prompt, and see whether the network then behaves as if it had seen that prompt. The tests run
on an explicit forward pass of the components-only model with hooks on the stream and on the readers
(`components_model.py`; it reproduces the stored activations to within 1 % and the top token on
99.2 % of prompts).

**The swap.** Take a base prompt, 23 + 45, and a source prompt that differs in one input, 23 + 71. At
a read point, move the base stream so that the quantity reads the source's value:
`x' = x + ((x_src − x) F) (Pᵀ F)⁻¹ Pᵀ`, with F the quantity's filter (how it is read) and P its pattern
(where it is written). Moving along the pattern leaves the point's other quantities at their base
values, because uncorrelated quantities have `F₂ᵀ P = 0`.

**The final answer is the wrong place to look.** Swapping a point's quantities, one at a time or all
together, never moves the answer to the source's answer (0-4 % of prompts). Replacing the whole stream
at `=` does not either, until late:

![Answer-level swaps](figures_causal/answer_swap.png)

Until about L24, later attention heads read the operands again at positions a and b, which still hold
the base prompt; after that, the answer sits in stream directions that the unembedding reads and no
later reader does, so no quantity describes them.

**The component that reads the quantities is the right place.** Run each MLP at `=` on its own, with
its masks held at the base prompt's, and ask how much of its response to the source (its output on
the source minus its output on the base) the swap reproduces: 1 means it responds exactly as to the
source's whole input, 0 not at all.

![Local interchange](figures_causal/local_interchange.png)

- All the quantities of an MLP's input reproduce 73-92 % of its response at L14-L17, 51-92 % at
  L18-L26, and less after (44-74 % at L27-L30, 13 % at L31).
- The baseline is a random subspace of the same dimension, moved the same way. It is not a null: its
  pattern favours the stream's largest-variance directions. The quantities beat it at every layer,
  clearly at L15-L17 and L20-L26 (L16, a's quantities: 0.83 against 0.55; L17: 0.86 against 0.41;
  L20, the result's: 0.63 against 0.26), barely at L18-L19 and L27-L30.
- Single codes carry visible shares on their own: b mod 100 at L16 reproduces 0.61 of the MLP's
  response to b (random 0.11), a's parity circle at L17 0.39 of its response to a (random 0.06), result
  mod 20 at L20 0.20-0.24 (random 0.04). The operation flag, one dimension, reproduces 97-100 % of the
  L14-L15 MLPs' response to the operation.
- The orange curve swaps what the quantities do not explain instead. It reproduces little at L15-L17
  and most of the response at L27-L30 (0.85-0.92), where the quantities do worst.

## 7. What the quantities miss

**Splitting each reader.** Take one reader's activation at a read point (times the RMS, minus its
average). Its *explained part* E is the best least-squares combination of the point's quantities: for
a reader that follows b's mod-20 square wave plus a small bump at b = 7, E is the square wave and the
bump is left over. The leftover splits into U, the part inside the dimensions parallel analysis kept
as signal, and T, the tail below them. The three parts are uncorrelated, so their variances add up.
Weighted by variance and averaged over the read points, E holds 70 %, U 28 % and T 0.7 %; at the MLP
inputs at `=`, U grows from 2-11 % at L14-L17 to 30-60 % at L22-L30. T never matters (removing it
everywhere changes the KL by 0.001), so the question is U.

**Is U load-bearing?** Each part is set to its average over the prompts (it then carries no
information about the prompt), or replaced by its value on another prompt, at the readers of a range
of layers at `=`.

![Ablations over layer windows](figures_causal/ablation_windows.png)

- **One layer at a time, nothing breaks**, even with every reader of the layer at its mean (accuracy
  at worst 0.93 → 0.91): the layers back each other up. The same ablations at positions a, op and b,
  over all layers, cost at most 0.02, and U in the attention readers does nothing.
- **Over L14-L31 at `=`, removing U halves the accuracy** (0.93 → 0.52). Removing the quantities
  instead leaves 0.17, removing both leaves 0. Over L26-L31 alone, U matters more than the quantities
  (0.86 against 0.92).
- **What breaks is confidence, not arithmetic.** 95 % of the errors are not numbers, nearly all `?\n`,
  the token this model gives when it does not answer (the clean model gives it on 4 % of additions).
  Among the number tokens, the right answer is still ranked first on 83 % of prompts without U
  (97 % clean), against 45 % without the quantities.
- **U carries the result, not prompt-specific corrections.** Taking U from another prompt with the
  same result keeps the accuracy at 0.91; taking it from a random prompt with the same operation gives
  0.41, worse than removing it. A correction for interference, say for cross-talk between a's and b's
  codes, would be specific to the operands and would break when U comes from 45 + 23 instead of
  23 + 45.

**What U is.** Its dependence on the labels, measured over all 20,000 prompts:

![Unexplained part](figures_causal/unexplained_profile.png)

- *L22-L30: a code of the result.* 69-88 % of U is a function of the result (one average per result
  value), and only 7-21 % is a sum `f(a) + g(b)` of separate functions of the operands. Its per-result
  averages span more and more dimensions, from 6 at L22 to 17 at L30 (participation ratio), and change
  quickly from one result to the next. The quantities' result codes span 4-7 dimensions. Such a code
  can put every result at the same distance from its centre, like a simplex, but it needs more than
  the 4 dimensions the sphere pursuit searches.
- *L16-L18: codes of the operands.* 78-85 % of U is a sum of functions of a and of b, but only 3-7 % is
  linear in them: smooth, non-linear operand codes that are not spheres.

**Whole components.** At `=` over L14-L31, 483 of the 2,484 readers are mostly unexplained (E holds
less than 20 % of their variance), but they hold only 9 % of the variance. Setting them to their
average costs nothing (accuracy 0.94); switching them off costs more (0.68), still far less than
switching off as many random readers (0.07). Their variation is not used, only their constant input.
The load-bearing unexplained signal sits inside readers that also read quantities.

**So:** the part of the inner activations the quantities do not account for is causally important,
and it is part of the computation, not a correction for interference. It overlaps with the
quantities, so over a few layers either can stand in for the other, except late (L26-L31), where the
network relies on it more.

## 8. Limits and open problems

- **Larger codes.** Result codes of more than 4 dimensions (L22-L30) and non-linear, non-spherical
  operand codes (L16-L18) are load-bearing and not in the atlas. Magnitudes such as a linear ramp in a
  are not spheres either; they are only found inside frames with other quantities.
- **Mixed readers.** Readers that mix quantities (plaids, products) are split among several
  quantities rather than explained by one.
- **Linking.** Codes transformed non-linearly between read points, or read together with another
  quantity at one of them, show up as separate codes; most codes are read at a single point.
- **Readouts.** The answer cannot validate a quantity before about L24; the consumer's output can,
  but only for MLPs, whose output depends on one position.
- **Provenance** only sees the quantities found at the writer's input, and "a function of" misses
  relations that oscillate fast when noise blurs neighbouring values.

## Appendix A. Method details

- **Relations at one read point.** `R²(i | j)`: the share of quantity i predicted by averaging it over
  the 20 prompts nearest in quantity j's coordinates (`variables.cond_r2`). It runs one way: at L18,
  `R²(a mod 5 | a mod 10) = 0.84`, 0.23 the other way.
- **Splitting frames.** Two independent spheres together are also a sphere, so a frame is split when
  both parts are spheres and their squared norms are not anti-correlated (the two halves of one circle
  are anti-correlated, −1).
- **Codes.** Coordinates are unit-variance and uncorrelated within each quantity, so canonical
  correlations are the singular values of the cross-covariance. A smaller quantity inside a larger one
  (all its canonical correlations ≥ 0.8) is recorded as contained, not linked.
- **Maps for the causal tests** (`interventions.py maps`): per read point, `F = G W` with G the kept
  readers' `g ⊙ V` and `W = lstsq(Hc, Z)` (reproduces Z to 0.997-1.000), `B = lstsq(Z, Hc)`, the
  top-r principal subspace Q of the readers, and a random filter with pattern `Σ F` for the baseline.
- **Local interchange** (`local`): 500 additions, sources differing in a, in b, in the operation, or
  another addition; the score is `1 − Σ|w − w_src|² / Σ|w_base − w_src|²`, with w_src the MLP's output
  on the source's input under the base masks. Swapping the stream RMS alone moves the output away from
  the source, so the RMS is not a channel.
- **Ablations** (`ablate layers|windows`): 500 additions and 500 subtractions; mean ablation sets a
  part to 0 (its average); resampling takes the part computed from another prompt's clean stream.
- **Label profile** (`labels`): shares of each part's variance explained by the operation, a, b, the
  result (as categories within each operation), the best additive `f_op(a) + g_op(b)`, and a linear
  function of a and b.

## Appendix B. What did not work, and why

- *Log cosh ISA on the original model:* log cosh favours sparse directions, and the original model's
  background makes bumps sparser than circle axes. On the components-only model it recovers clean
  codes (L18 at `=`: 17 of 26 outputs explained by one label at η² ≥ 0.8; `pipeline.py`).
- *Varimax plus grouping by anti-correlated squares:* nested periods of one operand (the square of a
  period-T coordinate has period T/2) chain into large groups.
- *Mutual information for relations:* on near-deterministic data a 3 % shared impurity gives 0.3 nats;
  `R²` measures the size of the dependence instead.
- *Looser linking across read points:* chains unrelated codes through axes many frames share (a's
  magnitude, the op flag).
- *Masked activations* (inner activation times the mask): gated, prompt-selective signals that are no
  longer linear reads of the stream.
- *Interchange read at the answer:* see section 6.

## Appendix C. Reproducing

```
python -m param_decomp.arith_repr.isa.atlas prep|layer <l>|link decomposed
python -m param_decomp.arith_repr.isa.writers masks|run <position>|provenance
python -m param_decomp.arith_repr.isa.interventions maps <l>        # CPU, per layer
python -m param_decomp.arith_repr.isa.interventions swap|ablate layers|ablate windows   # 1 GPU
python -m param_decomp.arith_repr.isa.interventions local|labels <l>                    # CPU
```

The GPU runs compile each new batch shape once (about 5 minutes); prompts run in chunks of 500, and a
persistent JAX compilation cache avoids recompiling across runs.
