# Incentivizing mechanistic faithfulness in the decomposition procedure

Ideas for modifying PD/SPD training so that decompositions pass the subspace-filtering
battery (`param_decomp_lab/scripts/validation/subspace_filtering/`), and the trade-offs
involved. Written 2026-07-08, after the targeted 8B L18 results and while the full-data
4-block runs were in flight.

## 1. What the battery measures, and what failed

The battery tests three properties that a *mechanistically faithful* decomposition
should have, beyond functional sufficiency of the circuit:

- **F1 — span sufficiency (information routing).** Projecting the original model's
  activations onto the selected subcomponents' read span (`V_S`) / write span (`U_S`)
  at a site should not change the output — after re-adding the site mean μ ("centered"
  flavor), so that only *prompt-varying* signal is tested.
- **F2 — offset routing.** The raw (uncentered) version additionally requires the
  circuit to take its constant/bias intake from the same activation components as the
  original (in a bias-free model, all effective biases are read from constant
  activation directions).
- **F3 — legality (no spurious mechanism).** The circuit must not read directions the
  original matrix provably annihilates (`V_S ⊄ row(W)`) or write directions it cannot
  produce (`U_S ⊄ col(W)`).

Observed on the targeted 8B L18 run: the MLP fails F1 badly (centered KL 10–70× the
circuit baseline; worst at the neuron-space write interface), attention essentially
passes F1 but fails F2 (q-side raw KL ≈ bias-only KL, centered ≈ 0), and the circuit
fails F3 at the down_proj read (row-space projection of the circuit's own input moves
its output 20×; most of that is constant intake from directions `W_down` cannot read).

## 2. Why the current objective doesn't enforce any of this

Nothing in the loss stack sees the *geometry* of individual subcomponents relative to
`W` or to the activation distribution:

- **Faithfulness** constrains only the sum `Σ U_c V_cᵀ + Δ = W`. Individual components
  can carry arbitrary mass outside `row(W)`/`col(W)` as long as it cancels in the sum
  (this is exactly the F3 failure), and the delta component absorbs whatever remains.
- **Stochastic/masked reconstruction** evaluates the circuit *as a function* on the raw
  activations `x`. A component is free to read any direction of `x` that correlates
  with the needed output on-distribution — including directions `W` never uses (F3) and
  directions other than the ones `W` uses (F1). Masked recon rewards *a* mechanism that
  reproduces the output, not *the* mechanism.
- **Importance minimality** shrinks how many components are active, not where they
  point. If anything it pushes toward efficient re-encodings (few components reading
  dense mixtures) rather than alignment with the original's routing.
- **CI semantics are weight-space.** Gates are trained to predict maskability of
  weight-space terms, so "the circuit at this input" is defined by masking — the
  activation-subspace reading we test was never part of the training signal.

So the failures are not surprising; they are the unconstrained degrees of freedom.

## 3. Proposals

Ordered roughly from cheap-and-safe to expensive-and-strict. They compose; §5 gives a
suggested staging.

### A. Hard legality by parameterization (fixes F3, free at run time)

Parameterize components inside the matrix's own spaces: `V_c = P_row(W) Ṽ_c`,
`U_c = P_col(W)ᵀ Ũ_c` (projectors precomputed once from the frozen `W`), or
equivalently learn components in the coordinates of `W`'s (economy) SVD basis. Since
the components must sum to `W` anyway, mass outside these spaces can only ever cancel —
removing it does not restrict what the sum can represent, only how components may
share/cancel.

- **Cost:** one fixed projection per matrix at init (or per forward if applied on the
  fly — one extra matmul per component update; negligible). No new loss terms, no new
  hyperparameters.
- **Fixes:** F3 exactly, by construction. Partially helps F2 (offsets must at least be
  read from readable directions).
- **Risks:** removes cancellation degrees of freedom that gradient descent may be using
  as slack — optimization could get slightly harder; worth an A/B on a toy. Vacuous for
  square full-rank matrices (most attention matrices), which is fine — F3 is also
  vacuous there.
- **Retrofit variant:** post-hoc project existing checkpoints (`V ← P_row V`,
  `U ← P_col U`), fold the removed mass into Δ, fine-tune briefly. Cheap way to test
  how much F3 matters before touching the training loop.

### B. An explicit, always-on offset component per matrix (targets F2)

The F2 failure is structural: the original reads its effective biases from
high-norm constant activation directions, while sparse CI components have no reason to
reproduce that routing — and forcing every sparse component to also carry offset
routing fights minimality. Instead, add **one designated low-rank "offset component"
per matrix** that is always on (like the delta, but inside the decomposition and
low-rank), initialized/regularized toward carrying `x ↦ W μ`-like routing (reading the
site's constant subspace, writing the mean output). Sparse CI components then only
need to carry prompt-varying signal, which is exactly the centered test.

- **Cost:** +1 component per matrix; trivial.
- **Fixes:** F2 cleanly, and makes F1-centered the honest target for the rest of the
  decomposition. Also likely reduces the "constant intake through illegal directions"
  part of F3.
- **Risks:** definitional — the offset component is distribution-relative (μ is a
  target-distribution object; for full-data decompositions μ is the token-pooled mean).
  It also slightly changes the interpretation story ("every circuit includes the offset
  component"), but that matches how we already read the battery (bias flavor separated
  from centered).

### C. Projection-consistency losses (directly train F1)

Add loss terms that are the differentiable versions of the battery's interventions:

- **C1 — input-side:** replace `x` at a site by `μ + P_S (x − μ)` and require the
  *original* weights (and downstream model) to reproduce the target output (KL or the
  existing recon loss). This is experiment 1A-centered as a training signal.
- **C2 — output-side:** project the original output onto `span(U_S)` (plus mean) and
  require no change (experiment 1B-centered).
- **Differentiable projector:** avoid SVD/QR of discrete active sets. Use the ridge
  least-squares form with **CI-weighted columns**: `A = V · diag(g(x))` (g = the gate
  values), `P_ε = A (AᵀA + εI)⁻¹ Aᵀ`. This is smooth in both `V` and the gates, needs
  no discrete set, and the solve is `n_active × n_active`-ish in effective size — cheap
  precisely when importance minimality is doing its job (synergy: sparsity keeps the
  projector affordable). Batched-padded per position exactly like `_pair_bases`.
- **Sampling for efficiency:** one random (site, flavor) per step, like the existing
  stochastic-source pattern, rather than all sites every step. Expected overhead then
  is one extra partial forward per step (~10–30%), instead of ~2× for all-sites.
- **Site-local surrogate (cheapest, strictest):** skip the downstream forward and
  penalize `‖W (I − P_S)(x − μ)‖²` — the prompt-varying input signal that `W` transmits
  but the span cannot represent. One matmul; no extra forward. Caveat: it enforces
  sufficiency *at the site*, which is stricter than the battery (the battery permits
  discarded signal that downstream nonlinearities dampen). Use as a warm-up or
  regularizer, not as the definition.

- **Fixes:** F1 — the main observed failure.
- **Risks:** two serious ones.
  1. **Rank-inflation gaming.** The trivial way to pass a projection test is to grow
     the span (at rank = d the projector is the identity — we observed exactly this
     degeneracy in the union-scope full-data runs). Any projection loss must be paired
     with pressure on span size: the existing minimality loss on gates, plus,
     if needed, an explicit per-position rank/L0 penalty on the active set. Always
     co-report span-rank stats with the KL (the battery already does).
  2. **Goodhart.** Once we train on the battery's objective, the battery stops being an
     independent validation. Keep held-out variants: different thresholds, per-position
     vs per-prompt scopes, held-out data, and the raw flavor if only centered is
     trained.

### D. Activation-space masking semantics (deeper version of C)

Rather than adding projection losses next to weight-space masking, change what masking
*means*: implement component masking as **projection of the site activation onto the
kept components' span** (input side), or train with both semantics stochastically
(sample weight-masking steps and projection-masking steps). Optionally retrain the CI
function to predict maskability under the projection semantics, so "causally important"
in the released decomposition means exactly what the battery measures.

- **Cost:** moderate implementation surgery in `masks.py` / `metrics/`; per-step cost
  similar to C with all-sites sampling.
- **Fixes:** F1 at the root — the definition of the circuit and the test coincide.
- **Risks:** it is a different decomposition problem; all existing hyperparameters,
  baselines, and intuitions need re-tuning. The projection semantics is also
  weaker at attributing *which* component mattered (projectors are set functions, not
  per-component), so CI training may become noisier. This is the right long-term
  direction if F1 is judged essential, but it is a research project, not a patch.

### E. Basis anchoring at nonlinearity interfaces (cheap partial F1 for MLPs)

The worst F1 failures are at the neuron interface (gate/up writes, down reads), where
the elementwise nonlinearity makes the *neuron basis* the causally meaningful one. Add
an L1/group-sparsity penalty on `U_c` (writes) and `V_c` (reads) *in neuron
coordinates* for the matrices adjacent to a nonlinearity, encouraging each component to
touch few neurons. Aligned writes/reads make the spans coincide with the coordinates
the nonlinearity actually gates, shrinking the span mismatch without any projector
machinery.

- **Cost:** a regularizer; negligible.
- **Fixes:** part of F1 at MLP interfaces; empirically checkable via the battery.
- **Risks:** pushes toward per-neuron decompositions (may sacrifice genuinely
  distributed structure and worsen minimality); tension with superposed features.
  Coefficient needs care.

### F. Monitoring and model selection (no training change)

Add the battery (centered + raw KL at a few sites, plus span-rank stats) as a slow eval
metric — on small models it is cheap (the 4-block battery runs in minutes). Use it for
hyperparameter selection (β, minimality coefficient, C, seeds) and early stopping.
Zero optimization risk; establishes how much faithfulness varies across the existing
hyperparameter landscape before we spend on new losses (it may be that some existing
knobs — e.g. stronger minimality, smaller C — already trade recon for subspace
faithfulness).

## 4. Trade-off summary

| proposal | fixes | training cost | implementation | main risk |
|---|---|---|---|---|
| A row/col parameterization | F3 | ~0 | small | optimization slack removed |
| B offset component | F2 | ~0 | small | distribution-relative definition |
| C projection losses | F1 | +10–30% (sampled) | medium (ridge projector) | rank inflation, Goodhart |
| C-surrogate (site-local) | F1 (stricter) | ~0 | small | over-constrains vs "dampened downstream" allowance |
| D projection-mask semantics | F1 at root | ~+50–100% | large | new problem; CI attribution noisier |
| E neuron-basis anchoring | F1 @ MLP | ~0 | small | forces per-neuron granularity |
| F eval + selection | measurement | eval-only | small | none (but selection-only pressure is weak) |

Strictness dial, weakest → strongest: F < A+B < E < C (centered) < C (raw) <
C-surrogate < D. Note the battery itself defines a natural curriculum: legality (F3)
and offsets (F2) are cheap structural fixes; information routing (F1) is the expensive,
fundamental one.

## 5. Suggested staging

1. **Now, zero-risk:** F (battery as eval metric on small runs) + A as a *post-hoc
   retrofit* on an existing checkpoint, re-running the battery to quantify how much of
   F3/F2 the retrofit alone recovers.
2. **Next training run:** A (parameterization) + B (offset component) from scratch on
   the 4-block Pile model; battery before/after. These are cheap and should not hurt
   recon; if they do, that is itself informative about how load-bearing the illegal
   cancellations were.
3. **If F1 is the goal:** add C1/C2 with the CI-weighted ridge projector, sampled one
   site per step, with span-rank co-monitoring and a held-out battery configuration.
   Tune the projection-loss coefficient against the recon/minimality frontier.
4. **Only if C plateaus:** consider D (projection-mask semantics) as a separate
   research track on toys first.

## 6. Open questions

- Is F1 *achievable* at high sparsity, or is span insufficiency partly forced by
  superposition? If the original genuinely routes k-dimensional signal through a site
  on some prompt, no (< k)-rank span passes; the per-position rank stats from the
  battery give an empirical lower bound on required span sizes — worth extracting
  before choosing coefficients.
- Should μ (offset reference) be per-position rather than pooled for full-data
  decompositions? BOS and early positions have very different statistics; a pooled μ
  makes the offset component carry position-dependent residue.
- Does passing F1 on-distribution transfer off-distribution (e.g. targeted battery
  prompts vs Pile)? The full-data run gives us a first read.
- Interaction with the delta component: with A+B in place, should Δ also be
  span-constrained, or explicitly reserved for "everything the decomposition refuses
  to explain"?
