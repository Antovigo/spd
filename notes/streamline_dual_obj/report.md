# Giving each decomposed matrix its own clean inputs

Design study, 2026-08-06, against `feature/dual_hidden_acts` at `f417a94c4`. Empirical inputs
are `notes/hidden_dual/report.md` and the `addsub-L18-{09,10,11}` runs.

The question: today the hidden-activation loss judges each decomposed matrix on an input that
earlier decomposed matrices have already damaged. Should it instead give every matrix the
input the real model would have given it? This report works through what changes, what it
costs, what it buys, and what it stops being able to see.

Short answer: **yes, add it — and it is a much bigger change than it looks, because it removes
a forward pass through the model entirely.** But add it alongside the current loss rather than
replacing it, because the difference between the two is itself the measurement we most want.

---

## 1. What the hidden-activation objective is for

A decomposed matrix is split into many small pieces (subcomponents). For each token, a
causal-importance network decides which pieces are needed; the rest are switched off. Two
different questions can be asked about "needed":

- **Needed for the output.** Switch the pieces off, run the whole model, and check the logits
  are unchanged. This is the original objective.
- **Needed for the activations.** Switch the pieces off and check that each decomposed matrix
  still produces the numbers it used to produce.

The second exists because a piece can matter internally while its effect cancels out before
the logits. The output objective is happy to switch such a piece off; the activation objective
is not. That is the whole point of the second causal-importance network.

The error is measured as a fraction, not as raw squared error: for each matrix, the summed
squared difference divided by the summed squared true value. So a matrix with large
activations and a matrix with small ones count equally, and a coefficient tuned on one block
transfers to another.

## 2. How it works today

One training step, in order:

1. **Clean pass.** Run the frozen model once on the batch and cache the input that arrives at
   every decomposed matrix. This pass is needed for the output objective anyway, so it is free
   for our purposes. Call the cached input to matrix `i` its **clean input**.
2. **Importance.** Both causal-importance networks read those clean inputs and produce an
   importance value for every piece of every matrix.
3. **Output reconstruction.** Masked passes through the whole model, comparing logits.
4. **Activation reconstruction.** A *separate* masked pass through the model. At each
   decomposed matrix the frozen matrix is swapped out for the masked pieces. This pass stops
   early — as soon as the last decomposed matrix has been reached, the rest of the model is
   skipped. Each matrix's output from that pass is compared against what the frozen matrix
   would have produced **on the clean input**.

Step 4 has an asymmetry that is easy to miss and matters a lot:

> The **prediction** is computed on a damaged input. The **target** is computed on the clean
> input.

So the error at a matrix mixes two things that have nothing to do with each other:

- the damage this matrix's own switched-off pieces caused, and
- the damage it inherited, because matrices upstream of it were switched off too.

### This is not hypothetical even within a single block

Decomposing only layer 18 gives seven matrices, and they are not independent. Inside one
transformer block:

| matrix | reads | so its input is damaged by |
|---|---|---|
| `q_proj`, `k_proj`, `v_proj` | the block input | nothing |
| `o_proj` | the attention output | q, k, v |
| `gate_proj`, `up_proj` | the stream after attention is added back | q, k, v, o |
| `down_proj` | the gate/up product | q, k, v, o, gate, up |

`q_proj`, `k_proj` and `v_proj` read the block's input, which nothing has touched. But
`o_proj` reads the attention output, which depends on all three of them; and `gate_proj`,
`up_proj`, `down_proj` sit after the attention output has been added back into the residual
stream, so they depend on all four attention matrices as well as on each other.

**Four of the seven matrices already receive damaged inputs today.** Only q, k and v are
judged fairly.

## 3. The proposal

The three-step sketch under discussion:

1. Clean pass, cache activations, compute both sets of importance values.
2. Masked passes for output reconstruction (stochastic and adversarial), compute those losses.
3. For activation reconstruction, hand each matrix its **original** input and compare the
   masked matrix's output against the true output.

Step 3 is the change. Written out for one matrix: instead of comparing
`masked_matrix(damaged input)` against `true_matrix(clean input)`, compare
`masked_matrix(clean input)` against `true_matrix(clean input)`. Same input on both sides.

Three notes on the sketch as stated:

**(a) Step 3 needs no forward pass at all.** This is the part worth dwelling on, and §4 is
about it. Once every matrix reads its own clean input, the matrices stop depending on each
other, and there is nothing left to run *through*. The chain is the only reason a forward pass
was ever needed here.

**(b) The target should be "what this matrix would have produced", not "the input to the next
decomposed matrix".** The sketch says compare `W_masked · x_i` against `x_{i+1}`, which is
right if `x_{i+1}` just means matrix `i`'s true output — and that is exactly what the code
already computes (`clean_site_outputs` builds `W · x_clean + b` from the cached input). It
would be wrong if it meant the cached input of the *next* decomposed matrix: in a transformer
those are different tensors with attention, a residual add and a normalisation sitting in
between, and often different widths (`gate_proj` outputs 14336 numbers, `down_proj` reads
14336, but `q_proj` outputs 4096 and the next matrix along reads the block input, not that).
No change needed here — just worth being explicit, because this distinction is exactly what
`hidden_readout_sites` was invented for (§7).

**(c) Keep one backward pass.** The sketch has the output loss backpropagated in step 2 and
the activation loss in step 3. Today there is a single summed `total_loss.backward()`
(`optimize.py:863`). Two separate backward passes would give the same gradients only if no
optimizer step happens in between, and both share the importance-network computation from step
1, so the first backward would free a graph the second still needs unless it is explicitly
retained. There is no benefit to pay for that; keep the single summed backward.

### An intermediate option worth knowing about

The motivation splits into two separable complaints, and there is a cheap fix for only the
first:

| complaint | fixed by |
|---|---|
| a matrix receives *gradient* for damage it did not cause | detaching each matrix's input inside the current chained pass |
| a matrix's *error number* includes damage it did not cause | only the local formulation |

Detaching the input in the chained pass keeps the forward exactly as it is — drift still flows
downstream and is still measured — but stops gradients from flowing backwards between
matrices, so each matrix's pieces are only trained on their own error. It is a two-line change
and it saves nothing. If the goal were purely "each matrix should get signal about its own
effect", that would be enough. The local formulation is worth more because of §4.

## 4. Why this is much cheaper: the forward pass disappears

Everything the local loss needs is already cached. For one matrix, with `x` its clean input:

```
true output    = W x + b                                    (already built, from the cache)
masked output  = U^T ( mask ⊙ (V x) )  +  δ_mask · (Δ x)  + b
```

`V x` turns the input into one number per piece; the mask switches pieces off; `U^T` projects
back out. `Δ x` is the leftover-weight term. There is no model in any of that — just two
matrix multiplies on a tensor already sitting in memory.

Two further savings fall out:

- **`V x` does not depend on the mask.** Neither does `Δ x`. Compute them once per step and
  every extra mask sample, and every step of an adversarial mask search, costs only the second
  multiply.
- **Nothing has to be kept for the backward pass.** Today, gradient flows from an early matrix
  through the frozen attention and normalisation ops into a later one, so all those
  intermediates must be retained. Locally there is nothing between matrices to retain.

### The arithmetic, for `addsub-L18-11-bigc`

Seven matrices on layer 18, batch 128 x 16 tokens = 2048 tokens.

| work | multiply-adds | approx |
|---|---|---|
| today: one truncated forward, **per mask sample** (embeddings + 19 of 32 blocks) | — | **~17 TFLOP** |
| local: `V x` for all seven matrices, **once per step** | 35.6 M per token | ~0.15 TFLOP |
| local: masked output for all seven matrices, **per mask sample** | 41.4 M per token | ~0.17 TFLOP |

Roughly **a hundred times less arithmetic per mask sample**. Treat it as an order of magnitude
rather than a precise figure — it counts multiply-adds only, and these small multiplies are
more limited by memory bandwidth than by arithmetic. Even so: eight local mask samples cost
about 1.5 TFLOP, still an order of magnitude below *one* sample today.

## 5. What the cheapness is for

The saving is only worth having if it is spent. Three things it buys:

- **More than one mask sample.** `n_mask_samples` is 1 in every current config, and the reason
  is structural: the truncated forward sits *inside* that loop
  (`stochastic_hidden_recon_subset.py:79-88`), so each extra sample costs a whole pass.
  Locally, sixteen samples cost less than one sample does today. A single random mask per step
  is a noisy estimate of "how much does switching pieces off hurt"; more samples is a directly
  better gradient.
- **An adversary during training, not just at evaluation.** The adversarial probe searches for
  the mask that hurts most. At 20 search steps it costs 21 truncated passes per batch today,
  which is why it is evaluation-only and on the slow cadence. Locally those 20 steps are 20
  cheap multiplies. There is a second reason it should work better: locally, a matrix's output
  is a **straight-line function of the mask** with fixed coefficients, so the search is over a
  simple bowl-shaped landscape rather than through a deep non-linear model. The existing PGD
  helper already accepts any mask-consuming objective, so it plugs straight in.
- **Analysis without the 8B model.** The local loss never touches the target model — it reads
  cached activations. So it can be evaluated offline, on CPU, from harvested activations,
  which is how much of the analysis tooling already works.

## 6. What it gives up, and why that is the interesting part

Judging every matrix on clean inputs means the loss can no longer see errors **compounding**.
The failure it becomes blind to is specific and plausible: every matrix reproduces its own
output well when handed the true input, yet when the masked model is actually run end to end
the small errors feed into each other and the internal state drifts a long way from the target
model's.

That is not a hypothetical worry. It is the exact phenomenon the three-block `L18to20`
experiment was set up to study, and it is why the current loss compares a damaged-input
prediction against a clean-input target in the first place.

So: **do not replace, add.** And then the two losses together are worth more than either alone,
because they are the same formula, at the same matrices, against the same targets, differing
only in which input the prediction was computed from. Subtract them and what is left is
exactly the part of each matrix's error that was inherited rather than caused.

```
chained error  −  local error   =   inherited (compounding) error, per matrix
```

That number does not exist today, it costs almost nothing to produce, and it is per-matrix, so
it says where drift is *created* and where it is *amplified*. For the three-block scale-up it
is close to being the headline measurement.

## 7. What has to stay chained regardless

`hidden_readout_sites` — the residual-stream measurement points added on
`feature/hidden_site_targets` — cannot be done locally, and not by accident. A readout point is
not a decomposed matrix's output; it is a place in the stream whose value is a property of the
whole chain. Feed every matrix its clean input and the stream is unchanged by construction, so
a readout's error would be exactly zero and would measure nothing.

This gives a clean division of labour:

| measured at | by |
|---|---|
| the decomposed matrices | the local loss — cheap, many samples, its own adversary |
| the residual stream | the chained loss — the only thing that can see it |

## 8. Predictions, including one that is a free correctness test

**q, k and v must give identical chained and local errors** on a single-block run. Their input
is the block input, which nothing upstream has touched, so the two formulations are handed the
same tensor. Compute both from the same mask draw and assert equality at those three matrices:
if they differ, the implementation is wrong. (On the three-block run this holds only for layer
18's q/k/v; layers 19 and 20 inherit.)

**The gap should concentrate on `o_proj` and `down_proj`.** They are the two matrices furthest
downstream inside the block — `o_proj` behind q/k/v, `down_proj` behind everything. If
compounding matters at all at one block, it should show up there.

**Part of the hidden network's density may be compensation, not necessity.** In
`addsub-L18-10-dual-ppgd` the hidden network's alive counts were pinned at the ceiling of the
components available on four matrices — q 125/128, k 124/128, v 252/256, down 246/256, with o
238/256 just behind. Some of that could be pieces switched on to *correct* inherited error
rather than because the matrix's own output needs them. If so, the local objective should need
fewer pieces at `o_proj` and `down_proj`. Note the prediction cannot extend to q/k/v — their
input is already clean, so whatever is saturating them is genuine local difficulty.

**The gap may be small at one block.** The existing report found per-matrix errors summing
almost exactly to the joint error (0.05184 against 0.05183), which suggests cross-matrix
interaction is weak at L18. If so, chained and local will nearly agree there and the difference
only opens up across blocks — which is itself worth knowing before spending the scale-up budget.

## 9. What stays exactly the same

Worth stating, because it is most of the machinery and it bounds the size of the change:

- the targets (`clean_site_outputs`, already built from the cached clean inputs);
- the relative-error formula, and the rule that numerator and denominator are summed
  separately across the whole pass and across GPUs before dividing;
- matrix selection by pattern (`select_sites`);
- the mask sampling and subset routing;
- the leftover-weight (delta) handling, and the bias, which appears on both sides and cancels.

Two small differences in meaning to be aware of:

- **Positions not routed to components.** The error is measured only where components replaced
  the frozen matrix. The code justifies this by saying the untouched positions ran the frozen
  matrix, so their error is zero — which is exactly true only at the *first* matrix in a chain.
  Further downstream the untouched matrix ran on a drifted input, so today's measurement is
  already quietly discarding some inherited error. Locally the claim becomes exactly true
  everywhere. This matters when reading `chained − local` as the complete compounding term: it
  is a lower bound.
- **The local error has an exactly attainable zero.** Switch every piece on with the leftover
  weight included and the masked matrix reproduces the frozen one exactly, so the loss measures
  purely the cost of switching things off.

## 10. Recommended shape

**Add a local loss carrying most of the activation objective's weight; keep the chained loss.**

- New `StochasticLocalHiddenReconLoss` (`ci_role: hidden`) becomes the workhorse: several mask
  samples, and once it is settled, its own adversarial variant.
- Keep `StochasticHiddenReconSubsetLoss` at a smaller coefficient. Consider narrowing it to
  `site_patterns: ["resid_*"]` — the readout points, where it is the only thing that can see
  anything. This is not the "measure only the residual-stream writes" idea previously ruled out:
  that one discarded attention from the activation objective altogether, whereas here the local
  loss covers every matrix and the chained loss is specialised to what only it can reach.
- Log both from step 0 of the next run even before rebalancing coefficients. `chained − local`
  is worth having immediately.

### Implementation

Small. Everything in `metrics/hidden_acts.py` applies unchanged; the new code is a per-matrix
loop replacing the one call to `model.site_outputs(...)`:

```python
targets = clean_site_outputs(self.model, ctx.pre_weight_acts, self.measured_sites)
for _ in range(ctx.n_mask_samples):
    mask_infos = calc_stochastic_component_mask_info(...)          # unchanged
    outputs = {
        site: self.model.components[site](
            ctx.pre_weight_acts[site],
            mask=mask_infos[site].component_mask,
            weight_delta_and_mask=mask_infos[site].weight_delta_and_mask,
        )
        for site in self.measured_sites
    }
    add_site_errors(batch_errors, site_squared_errors(outputs, targets, mask_infos))
```

Notes:

- Prefer a new metric class over a flag on the existing one. The two differ by an entire
  forward pass, and the intended configuration runs both at once with different
  `site_patterns` — which a single instance cannot express.
- Restrict to linear matrices, as `clean_site_outputs` already asserts. Readout points are not
  valid measurement targets here and should be rejected loudly rather than silently scoring
  zero.
- **Do not hoist `V x` out of the loop in the first version.** `LinearComponents.forward`
  recomputes it per sample, which wastes roughly half the local cost — irrelevant next to
  removing a 17 TFLOP pass, and it keeps the first version to a straightforward reuse of the
  existing forward. Hoisting only starts to matter once mask samples go up or an adversary is
  added, and it is a contained follow-up (a `forward_from_acts` entry point on
  `LinearComponents`).
- The loss never reads `ctx.batch`, only `ctx.pre_weight_acts`. Worth keeping true — it is what
  makes offline analysis possible.

### Sequencing

1. Land the local loss, run it alongside the chained one at a small coefficient, and check the
   q/k/v identity (§8). Read `chained − local` per matrix on the single-block config.
2. Shift the coefficient weight onto the local loss, raise `n_mask_samples`, narrow the chained
   loss to the readout points.
3. Add the local adversary. Then reconsider whether the chained loss needs an adversary at all,
   given it is by then only watching the stream.

## 11. Things that will change in the logs

- **Activation-error numbers will drop and are not comparable to earlier runs.** The local loss
  is a strictly easier objective — it never charges a matrix for inherited damage. Do not
  overlay the new curves on `addsub-L18-{09,10,11}`. (This is the second time this metric has
  changed definition; the first was raw squared error to relative error.)
- **The coefficient will need retuning**, for the same reason.
- **`n_alive` for the hidden network may fall**, if the compensation hypothesis in §8 holds.
  That is a result, not a regression.

---

*The earlier version of this document analysed two unrelated proposals for the causal-importance
networks themselves — sharing a trunk between the two networks, and forcing the activation
importance to be at least the output importance. Both are independent of everything above and
neither is superseded by it; they are preserved in commit `d3716702f`.*
