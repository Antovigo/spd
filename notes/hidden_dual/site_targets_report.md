# Which hidden activations should the second CI net reconstruct?

Status as of 2026-08-02: **phase 1 complete (5 arms), phase 2 launched.** Full generated
tables in `site_targets_tables.md`. Design and chronology in `lab_notebook.md`; the
dual-CI scheme itself is described in `report.md`.

The dual-CI scheme trains a second CI net to reconstruct the *decomposed sites'
activations*. "The decomposed sites" is a choice, not a given — it happens to be every
matrix we decompose, which is an artefact of where components live rather than a claim
about where reconstruction signal is most useful. This series asks which choice is best.

## The arms

Five runs, everything but the measured site set held identical to
`addsub-L18-10-dual-ppgd`:

| arm | hidden objective measured at | rationale |
|---|---|---|
| `addsub-L18-11-baseline` | all 7 decomposed matrices | the status quo |
| `addsub-L18-11-module-out` | `o_proj` + `down_proj` | what the modules *add* to the stream, excluding their internals |
| `addsub-L18-11-resid` | the residual stream, post-attn and post-MLP | what the model actually carries forward |
| `addsub-L18-11-down-only` | `down_proj` alone | floor: the site carrying 97% of the joint KL |
| `addsub-L18-11-mlp-only` | `gate_proj` + `up_proj` + `down_proj` | drops attention entirely, keeps the MLP whole |

15000 steps, gamma annealed over the last 5000, C raised 4x (total 6144), 2 GPUs per arm.

`mlp-only` was added after the baseline landed, and is the sharpest test of that arm's main
finding. The baseline showed the hidden net's surplus over the output net is ~4.4x per
position in attention against ~1.6x in the MLP, while the anomaly census showed almost none
of the attention surplus is visible to the output objective. `mlp-only` differs from
`baseline` in exactly the four attention-internal sites, so the pair prices that surplus
directly: if attention hidden signal is doing real work, dropping it should cost output
quality; if it is inert, it should not.

The five arms bracket attention cleanly — `baseline` has all four attention sites,
`module-out` keeps only `o_proj`, and `mlp-only` / `down-only` have none.

**Every arm declares the same two residual readout sites and differs only in the training
losses' `site_patterns`.** So every arm logs the same eval panel — hidden error at all 7
matrices *and* at both stream points, under both CI nets. Questions like "does training on
`down_proj` alone also repair the stream?" are then read straight off the panel rather than
inferred.

## Reading the residual stream at all

The stream is not any matrix's output, so measuring there needed a new concept:
`pd.hidden_readout_sites`, a `{name: module_path}` map whose module *input* is captured —
clean and masked — and joined to the decomposed sites in `ComponentModel.measurement_sites`.
Existing `site_patterns` then selects it like anything else. In a Llama block the two
capture points are `layers.18.post_attention_layernorm` (post-attention stream) and
`layers.19.input_layernorm` (post-MLP stream).

One semantic difference is load-bearing. A decomposed site's error is measured only at the
positions its routing mask selects, which is sound because an unrouted position ran the
frozen module and its error is identically zero. That argument fails on the stream:
attention mixes positions, so a position routed to nothing still receives error from the
routed positions it attends to. Readout sites are therefore measured at **every position**.

## The residual objective is 2830x smaller, and had to be recalibrated

Measured at step 0, the same CI mask gives:

| measured at | relative error |
|---|---|
| the 7 matrices | 0.942 |
| the residual stream | 0.000333 |

The stream's `Σ tgt²` denominator is dominated by the frozen incoming residual, so the same
physical perturbation reads ~2830x smaller. Left at `coeff: 1.0`, the resid arm's objective
would have sat far below the sparsity penalty it competes with, and the run would have
measured "hidden objective switched off" rather than "the stream is a worse target".

That arm's coefficients are therefore scaled by 2830 (stochastic 1.0 -> 2830, PPGD
0.5 -> 1415), equalising the two objectives at step 0. The reported quantity remains the
true relative error; only the arm's weight changes, so the variable under test stays *which*
activations rather than *how strongly*. Two caveats worth carrying into the analysis:

- The calibration is probe-dependent — 2830 CI-masked, ~2100 stochastic-masked. The 35% gap
  is immaterial against the correction itself, but this is not a constant of nature.
- It equalises at step 0 only. If the two objectives fall at different rates the arms drift
  apart in effective weight; the logged panel makes that visible rather than hidden.

## C was at its ceiling, and is now 4x

The -10 baseline was saturated at step 10000 — the hidden net had `q_proj` and `k_proj` at
exactly 128/128 — which would have clipped the very readout this series is about. Probes:

| C factor | total C | peak / GPU (of 46068 MiB) | headroom |
|---|---|---|---|
| 1x | 1536 | 39002 | 7.0 GB |
| 4x | 6144 | 41546 | 4.4 GB |

Quadrupling the components costs 2.5 GB: C is a weak memory lever here, as the -09 series
also found (the weight-delta tensors dominate and are full-weight-shaped regardless of C).
Stopped at 4x rather than 6x, which extrapolates to ~43.2 GB — the knife-edge that OOM'd
earlier 8B runs on this node's smaller cards, and a mid-run OOM costs 13 h.

## The anomaly census

The (a,b) grids encode both nets' CI per (component, position, op, a, b) cell. In the
applet's subtractive merge, **magenta = output-active but hidden-inactive** — the case the
scheme says should not happen, since a component mattering for the logits must matter for
the activations producing them. Green (hidden-only) is expected and common.

Counting is done offline from `ab_grids/step_*.js`, calling a cell active at CI >= 0.5.

The two counts do **not** have the same robustness, which matters for how hard each can be
leaned on. Sweeping the cut on the reference run at step 10000:

| cut | magenta cells | green cells | anomalous components | output-only components |
|---|---|---|---|---|
| 0.1 | 43920 | 2357110 | 4 | 0 |
| 0.3 | 40284 | 1802205 | 9 | 0 |
| 0.5 | 42415 | 1465776 | 13 | 2 |
| 0.7 | 47676 | 1202344 | 21 | 2 |
| 0.9 | 57910 | 957161 | 28 | 3 |

Magenta *cells* move by 1.4x over a 9x range of threshold — the saturation of CI makes the
count essentially cut-independent, so it is safe to compare across arms. The *component*
count moves 7x over the same range, because "more magenta than green cells" is a near-tie
for many components and the tie breaks differently as the cut moves. Component counts are
therefore reported at a stated cut (0.5) and read as an ordering, never as a magnitude.

### Reference run: anomalies are an MLP phenomenon

From `addsub-L18-10-dual-ppgd` at step 10000 (the pre-anneal state, so directly comparable
to this series' mid-training):

| matrix | saved | magenta cells | green cells | both | anomalous components |
|---|---|---|---|---|---|
| mlp.gate_proj | 68 | 11380 | 117922 | 190829 | 3 |
| mlp.up_proj | 88 | 14095 | 205333 | 193147 | 2 |
| mlp.down_proj | 93 | 16525 | 213517 | 240278 | 8 |
| attn.q_proj | 43 | 52 | 179576 | 47125 | 0 |
| attn.k_proj | 18 | **0** | 107368 | 239 | 0 |
| attn.v_proj | 60 | **0** | 241384 | 321 | 0 |
| attn.o_proj | 178 | 363 | 400676 | 83557 | 0 |
| **total** | 548 | 42415 | 1465776 | 755496 | 13 |

Magenta is 1.9% of active cells and lives almost entirely in the MLP. Attention is
essentially anomaly-free — `k_proj` and `v_proj` are at exactly zero, and their tiny
`both` counts show the output net barely uses them at all while the hidden net uses them
heavily. Of the 13 anomalous components, 2 are magenta with no green cells whatsoever.

So "output-important implies hidden-important" holds almost perfectly in attention and
leaks only in the MLP. This is the baseline the four arms are measured against, and it
already suggests the attention matrices' hidden signal is nearly pure surplus — which is
what the `module-out` and `down-only` arms probe directly.

## Arm 1 of 4: `baseline` (all 7 matrices), final at step 15000

The reference the other three are judged against. C=4x did its job — nothing is near the
ceiling, so every count below is a measurement rather than a clip.

| | output net | hidden net |
|---|---|---|
| alive / 6144 | 1229 (20.0%) | 1906 (31.0%) |
| CI_L0 (mean active per position) | 25.4 | 62.8 |
| hidden recon error, 7 matrices | 0.2625 | 0.0334 |
| hidden recon error, resid stream | 5.838e-05 | 1.983e-05 |

Worst-saturated matrix is `v_proj` at 35.5% (hidden net). Output quality: `kl_ci_masked`
0.003953, `kl_unmasked` 0.001546, `PGDReconLoss` 0.005470.

### The hidden net's extra components are overwhelmingly attention

Absolute counts per locus. `alive` is the number of distinct components reaching CI 0.1
anywhere in the eval pass; `CI_L0` is the mean number active at a single position.

| locus | C | alive output | alive hidden | CI_L0 output | CI_L0 hidden | hid/out L0 |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 201 | 263 | 5.3 | 8.3 | 1.6x |
| mlp.up_proj | 1024 | 240 | 319 | 5.8 | 10.7 | 1.8x |
| mlp.down_proj | 1024 | 264 | 292 | 6.7 | 10.6 | 1.6x |
| attn.q_proj | 512 | 72 | 172 | 1.5 | 6.9 | **4.6x** |
| attn.k_proj | 512 | 51 | 158 | 1.7 | 7.0 | **4.1x** |
| attn.v_proj | 1024 | 175 | 364 | 2.1 | 9.5 | **4.5x** |
| attn.o_proj | 1024 | 226 | 338 | 2.3 | 9.8 | **4.3x** |
| **MLP subtotal** | **3072** | **705** | **874** | **17.8** | **29.6** | 1.7x |
| **attention subtotal** | **3072** | **524** | **1032** | **7.6** | **33.2** | **4.4x** |
| **total** | **6144** | **1229** | **1906** | **25.4** | **62.8** | 2.5x |

The two nets differ by ~1.6x per position in the MLP and ~4.4x in attention. In absolute
terms the hidden net keeps 1032 attention components alive against the output net's 524,
and is active on 33.2 attention components per position against 7.6 — so attention accounts
for 25.6 of the hidden net's 37.4 extra active components per position, from half the
component budget. Whatever the
hidden objective is buying, it is mostly buying it in attention — where, per the anomaly
census below, essentially none of it is visible to the output objective. This is the
quantitative form of the prediction that motivated the series, and it is what the
`module-out` and `down-only` arms test directly: both drop all four attention-internal
sites from the objective, and `module-out` keeps only `o_proj`.

### Anomalies, again MLP-only

1.00% of active cells are magenta (21854 magenta vs 1444261 green vs 720683 both), 6
anomalous components, **0** output-only components. By matrix: gate 5299, up 4093, down
12375, q_proj 4, k_proj 0, v_proj 0, o_proj 83.

The pattern from the reference run survives a 4x change in C and the full anneal: the
output net's activity is a near-subset of the hidden net's in attention, and leaks only in
the MLP — concentrated in `down_proj`, the site the exchange-rate study found carries 97%
of the joint KL.

## Why this baseline is not comparable to `-10-dual-ppgd` in absolute terms

The series baseline ends ~9% worse on `kl_ci_masked` and ~23% worse on `PGDReconLoss` than
`addsub-L18-10-dual-ppgd`'s final. That is a schedule artefact, not a regression. Three
things differ: 15000 steps instead of 20000 (which also makes the cosine LR decay 1.33x
faster), gamma annealed over the last 5000 steps instead of the last 10000 — both anneals
start at step 10000 and end at gamma 0.01, so this one is twice as fast — and C raised 4x.

| metric | -10 @10k | -10 @15k | -10 @20k | -11 @10k | -11 @15k |
|---|---|---|---|---|---|
| kl_ci_masked | 0.004094 | 0.003720 | 0.003633 | **0.003993** | 0.003953 |
| kl_unmasked | 0.002637 | 0.002011 | 0.001755 | **0.002306** | 0.001546 |
| PGDReconLoss | 0.006582 | 0.005232 | 0.004429 | **0.006173** | 0.005469 |
| hidErr_outCI | 0.2423 | 0.2397 | 0.2484 | **0.2345** | 0.2625 |
| hidErr_hidCI | 0.0365 | 0.03377 | 0.03425 | **0.0349** | 0.0334 |
| CI_L0 output | 39.95 | 31.66 | 23.87 | 47.72 | 25.42 |

**At matched step 10000 the new baseline is better on every output and hidden metric**, so
4x C does not cost quality — it helps at matched training. The endpoint gap is the missing
5000 steps: in `-10` those steps alone moved `PGDReconLoss` 0.005232 -> 0.004429 (16%).

Two traps when eyeballing this:

- Raw alive counts are not comparable. 1906 hidden-alive here vs 1461 in `-10` looks worse
  but is 31% of 6144 against 95% of 1536 — the old figure was pinned to a ceiling. The
  unclipped density measure, `CI_L0`, has the new run *sparser* at 15k than `-10` was at 15k
  (25.4 vs 31.66).
- `report.md`'s 0.2238 / 0.0518 come from the CPU exchange-rate probe (256 prompts, fp32,
  different mask), not the eval loop. The logged eval at step 20000 is 0.2484 / 0.03425.

All four arms share this schedule, so the cross-arm comparison — the point of the series —
is unaffected. `-10` is a historical reference here, not a control.

## Selection metric and standings

The arm is chosen by **output-PGD nats against total alive components**, both minimised:
`PGDReconLoss * alive-either`, lower is better. `PGDReconLoss` is the adversarial probe on
the model output; `alive-either` counts components alive under *either* CI net.

They combine as a **product, not a ratio**. Both are costs, so `nats / alive` is
wrong-signed in its denominator — it scores an arm better for keeping *more* components.
This was the metric's first form here and it inverted the ranking, putting the densest
Pareto-optimal arm on top; the corrected ranking is in the phase-1 result section.

The denominator must be a **union**, not a sum — both nets score the same shared
subcomponent pool, so summing the two `NAlive` values double-counts everything both keep.
Neither logged count exposes the overlap, so it is computed from `ab_grids`, which stores
per-component mean CI for both roles over all C: alive means the per-position mean CI
reaches 0.1 under either net. Stricter than the `NAlive` eval metric (mean over the prompt
pool rather than max over examples), so these counts run lower than the logged ones, but
identical across arms — which is what a ranking needs.

### The output net's alive set is a strict subset of the hidden net's

In every arm `alive-either` equals `alive-hidden` **exactly**. Every component
the output objective keeps alive is also kept by the hidden objective — the containment the
scheme predicts, now at component granularity rather than the cell granularity of the
anomaly census. It also means the hidden net's count *is* the decomposition's total cost:
the output objective never pays for a component the hidden objective was not already
keeping.

### Alive-either per locus

| locus | C | baseline | resid |
|---|---|---|---|
| mlp.gate_proj | 1024 | 46 | 57 |
| mlp.up_proj | 1024 | 59 | 76 |
| mlp.down_proj | 1024 | 77 | 130 |
| attn.q_proj | 512 | 33 | 11 |
| attn.k_proj | 512 | 23 | **1** |
| attn.v_proj | 1024 | 40 | **1** |
| attn.o_proj | 1024 | 96 | 91 |
| **total** | **6144** | **374** | **367** |

`resid` all but abandons attention-internal sites — `k_proj` and `v_proj` collapse to a
single alive component each — and reinvests in `down_proj`. It reaches nearly the same
total cost as `baseline` by a very different allocation, and pays 26% more adversarial
output error for it.

## Phase 1 result: all five arms, ranked

Winner: **`mlp-only` — `gate_proj` + `up_proj` + `down_proj`, no attention.**

Both quantities are costs and are therefore combined as a **product**. Dividing nats by
alive components is wrong-signed in the denominator — it credits an arm for keeping *more*
components, which is the opposite of the goal. Pareto status is reported alongside, because
the product fixes one particular exchange rate between the two costs while domination is
exchange-rate-free.

| arm | PGDRecon (nats) | alive either | **nats x alive** | Pareto |
|---|---|---|---|---|
| **mlp-only** | 0.00596 | **298** | **1.7756** | **optimal** |
| down-only | 0.00627 | 315 | 1.9735 | dominated by mlp-only |
| baseline | **0.00547** | 374 | 2.0456 | **optimal** |
| module-out | 0.00602 | 406 | 2.4429 | dominated by baseline, mlp-only |
| resid | 0.00692 | 367 | 2.5394 | dominated by down-only, mlp-only |

The frontier has exactly two points: `mlp-only`, the sparsest, and `baseline`, the most
faithful. `mlp-only` buys **20% fewer components for 9% more adversarial error** and wins
the product by 13%. The three arms between them are strictly dominated — `module-out` and
`resid` are beaten on *both* axes at once.

**Dropping attention from the hidden objective is the single best move available.** It is
also the one the phase-1 mid-analysis argued against: the baseline's attention surplus
(4.4x the output net's per-position activity) is real activity, but it is expensive activity
— removing it costs little output fidelity and saves a fifth of the component budget.

### The module writes beat the residual stream at reconstructing the residual stream

| arm | hidden error at the resid stream (hidden CI) |
|---|---|
| module-out | **1.173e-05** |
| resid | 1.511e-05 |
| baseline | 1.983e-05 |
| down-only | 7.05e-05 |
| mlp-only | 7.234e-05 |

`module-out` reconstructs the stream 29% better than `resid` does, despite `resid` training
on the stream directly and `module-out` never measuring it. Targeting `o_proj` + `down_proj`
— the two writes — is a better-conditioned proxy for the stream than the stream itself,
whose relative error is dominated by the frozen incoming residual in the denominator.

Note also `baseline`, which never measures the stream, is within 2x of `resid`, which trains
on it at a 2830x calibration. Little of the stream's structure is absent from the matrices.

### Anomalies fall when the objective narrows, but not usefully

| arm | magenta cells | % of active | anomalous comps |
|---|---|---|---|
| baseline | 21854 | 1.00% | 6 |
| module-out | 7750 | 0.28% | 2 |
| resid | 6012 | 0.28% | 2 |
| down-only | 3931 | 0.18% | 1 |
| mlp-only | 9370 | 0.54% | 1 |

Every narrower arm has fewer anomalies than `baseline`, but they also reconstruct worse — the
anomaly rate tracks how much the hidden net is doing, not how well. `k_proj` and `v_proj` are
at exactly zero magenta in **all five arms**, the most robust structural fact in the series.
The one informative deviation is `mlp-only`'s `q_proj` (374 magenta against 4 for `baseline`):
with attention absent from the hidden objective, output-only attention activity appears
exactly where the hidden net stopped looking.

### Containment holds everywhere

In all five arms `alive-either` equals `alive-hidden` exactly. The output net's alive set is
a strict subset of the hidden net's, without exception — so the dual scheme never costs a
component beyond what the hidden objective already keeps.

## Phase 2: the winner at 20k under increased pressure

`addsub-L18-12-press3` and `-press10`: the winning locus (`mlp-only` — the three MLP
matrices, no attention), back on the
`-10-dual-ppgd` schedule — 20000 steps, gamma annealed over the last 10000 — with C kept at
4x, since reverting it would reintroduce the alive-count ceiling. The hidden reconstruction
coefficients are scaled 3x and 10x (stochastic 1.0 -> 3.0 / 10.0, PPGD 0.5 -> 1.5 / 5.0);
everything else is byte-identical to the phase-1 baseline config.

The question is whether the hidden objective is currently underweighted. Phase 1 says
*where* the signal is (everywhere, including attention internals); phase 2 asks how hard to
push on it.

## Open, pending the runs

1. Per-matrix active (`CI_L0`) and alive counts for both nets, with the ceiling lifted.
2. Saturation ratio at 4x C — whether alive counts are now genuinely measured.
3. Whether a narrower target degrades the *output* decomposition (the ranking criterion).
4. Whether training on a subset repairs the sites it never measures.
5. Whether the anomaly rate is a property of the scheme or of the target set.
