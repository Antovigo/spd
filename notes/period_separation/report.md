# Period separation — dynamics model & experimental plan

2026-07-21. Rewritten: the 5k probe series (`addsub-L18-06-psep-*`) compressed every
schedule into 5000 steps and is therefore **invalid** as evidence about scheduling
dynamics; those runs are deleted (disk + wandb). What survives as evidence: the
full-length 20k runs (the `addsub-L18-05-*` grid and `addsub-L18-07-10x-hid0.1`), which
suggest — but on a different base recipe and without controls — that impmin dose and
early hidden-acts pressure both matter. This report sets up the clean experiment.

## 1. Objective (lexicographic)

1. **Reconstruction accuracy first.** Maximize target-distribution reconstruction
   (final whole-sequence rounded KL; PGD recon as the adversarial check). This is never
   traded away.
2. **Then separation.** Conditional on near-perfect reconstruction, subcomponents should
   isolate the operand periods (2, 5, 10, 20, 50, 100) — one period per subcomponent.
3. **Then parsimony.** Among decompositions at the same reconstruction level, fewer
   subcomponents is better. Extra subcomponents that *enable* better reconstruction are
   wanted; extra subcomponents at equal reconstruction are bloat.

Operationally: a **recon gate** `rounded KL ≤ τ` (τ set from the best baseline, see §5.1)
defines "near-perfect"; separation and size are only ranked among gate-passing runs.

## 2. The system, and what happens over training

Losses acting on the decomposition (04-hidden reference recipe):

- **Reconstruction forces**: `StochasticReconSubsetLoss` (1.0), `UnmaskedReconLoss`
  (0.5), `PersistentPGDReconLoss` (0.5) on output KL; `StochasticHiddenActsReconLoss`
  (0.001) on MLP-internal activations. The delta component is exact by construction and
  pinned off on target data, so these force the *components* to carry the behavior.
- **Sparsity forces**: `ImportanceMinimalityLoss` — per-datapoint L_p on CI, coeff 5e-5
  with multiplier starting at 2x decaying to 1x, p annealing 2.0 → 0.5 linearly (the new
  `SmoothL0ImportanceMinimalityLoss` replaces p-annealing with γ: φ(c)=c²/(c²+γ²),
  γ 1 → 0.01) — and **CI-scaled weight decay** (0.3): each subcomponent shrinks at a
  rate ∝ (1 − max CI over the batch), a use-it-or-lose-it prune gated by *max*, so rare
  use protects a component.

### Phase model (prediction)

Because the schedules move the balance of these forces, the run should pass through
three qualitatively different phases:

- **P0 — recruitment/imprinting (~0–10%).** CI-fn is uncalibrated, recon losses dominate
  (impmin is small against large recon gradients even with the 2x early multiplier).
  Components grow to capture behavior; many get mid-range CIs. Whatever *alignment*
  pressure exists now (hidden-acts matching period-specific MLP neurons) shapes which
  feature ends up in which component — cheaply, because nothing is committed yet.
- **P1 — crystallization (~10–50%).** CIs binarize (leaky-hard sigmoid saturates), the
  alive set stabilizes, redundant components die (impmin + CI-scaled WD). Merges happen
  here if the sparsity gain beats the recon cost: two features that co-occur on most
  prompts (all mod-p features do — every answer needs every residue) are merge
  candidates. **Which periods share a component is decided by the end of P1** (matches
  the empirical "topology set by ~10k of 20k").
- **P2 — refinement (~50–100%).** The concavity schedule arrives at p < 1 (or γ ≪ 1):
  the penalty differential now acts on *mid* CIs — saturated CI≈1 costs the same
  regardless of p, but ambiguous usage is pushed to 0. Prediction: this phase cannot
  restructure (basin lock-in; cf. the ablation-floor negative result), it can only
  *purify* CI patterns (mixing ↓) and fine-tune recon. Consistent with purity improving
  15k→20k in every baseline.

## 3. Predicted effects, per hyperparameter

Notation: **R** = reconstruction, **N** = number of alive/active components,
**S** = period separation (mixing ↓ = S ↑).

| knob | R | N | S | phase it acts through |
|---|---|---|---|---|
| impmin coeff (dose) ↑ | ↓ monotonic | ↓ monotonic | inverted-U | P1 deaths/merges |
| impmin coeff timing (early vs late, matched dose) | early ≥ late | early ↓ more | early ↑ | early prunes pre-commitment; late must dismantle formed structure |
| concavity anneal (p→0.5 / γ→0.01) present vs absent | ≈ / slight ↓ | reported-N ↓ (binarization) | ↑ | P2 purification of mid CIs |
| concavity arriving early (concave during P0/P1) | ↓ (cliff/instability, L_p) | ↓ | ↓ (winner-take-all merges co-active periods) | P0/P1 |
| SmoothL0 vs L_p (matched dose & schedule) | ≥ (no gradient cliff at c→0) | ≈ | ≥ (γ sets an explicit CI scale; cleaner binarization) | P2 mostly |
| hidden-acts coeff 0.1 early | ≈ (free) | ≈ | ↑↑ (imprints period-specific neuron alignment) | P0 |
| hidden-acts held high to the end | ↓ slight | ≈ | ↓ vs decayed (freezes P1 patterns, blocks P2 purification) | P2 |
| hidden-acts too high (≥0.3) | ↓ | ≈ | ↓ (over-constrains components to span many neurons) | P0/P1 |
| CI-scaled WD ↑ | ≈ (max-gated) | ↓ | ≈ | P1/P2 continuous |
| CI-fn LR ratio ↓ | ≈ | ↓ | ↑ (prior recipe finding) | P0/P1 |
| C (capacity) | ≈ (not binding: ~59/456 used) | — | ≈ | — |

Key predicted *interactions*:

- **impmin dose × hidden-acts**: hidden-acts imprinting should make impmin pruning
  *safer* (components are cleaner, so the survivors reconstruct better) — the two
  compose rather than trade off. (Suggested by hid_sched-5x being the best 20k
  separator.)
- **impmin dose × concavity timing**: strong early dose + late concavity is the
  "concentrate then purify" pattern; strong dose *with* early concavity should be the
  worst cell for S (merges under winner-take-all).
- **Dynamics signature to look for**: if the phase model is right, N(t) should drop
  mostly in P1 and flatline in P2, while S(t) (mixing) improves mostly in P2. Schedule
  variants should move *which* phase does the work, visible in these curves.

## 4. Shortlisted factors

Fixed at 04-hidden values: recon-loss coeffs (1.0/0.5/0.5), CI-scaled WD 0.3, CI-fn/component
LRs, C, beta 0.75, sampling, seed 0 (except replicates), **steps 24000 — every run runs the
full schedule; no compressed probes**.

Varied:

1. **impmin dose**: coeff ∈ {2.5e-5, 5e-5, 1e-4, 2e-4} (04-hidden multiplier shape kept).
2. **impmin family**: L_p (p 2→0.5) vs SmoothL0 (γ 1→0.01) at matched coeff.
3. **impmin coeff timing**: flat vs early-2x vs late-2x at matched ∫coeff·dt.
4. **concavity window**: full (0→100%), late-only (50→100%), none (stay convex).
5. **hidden-acts**: coeff ∈ {0.001, 0.1} × schedule {constant, exp-decay→~0, decay ending
   at 50%}.
6. **seed**: ×3 at the center and final recipes (noise floor: the probe series showed
   seed moves med_band by ~0.015 and active counts by ~15%).

## 5. Experimental design

### 5.1 Metrics (identical pipeline for every run)

- **R**: final `eval/target_recon/rounded` (primary), `eval/loss/PGDReconLoss`
  (adversarial check); full curves from metrics.jsonl (eval every 500).
- **N**: answer-position active count (mean CI > 0.01 over prompts, from the
  per-position JSON) — primary, anchor-free. Secondary: alive-sweep curve
  (`alive_subcomponents_curve.tsv`) read at a **fixed** KL threshold for all runs
  (post-hoc from the curve — no re-runs needed), since the default anchored threshold
  moves with each run's recon and confounds cross-run size comparison.
- **S**: `score_period_separation` — `n50` (primary: mean orbits to 50% power),
  clean fraction (band_purity > 0.5), median band_purity (reported, noise-limited).
  Weight-side confirmation (`collect_inner_activations` → `compute_subcomp_periods`)
  on finalists only.
- **Gate**: τ = rounded KL of the best control + 20% margin, set once stage 1 lands.
- **Dynamics runs** (one designated run per factor level, not per seed): keep all
  checkpoints (5k/10k/15k/20k/24k; `keep_last_n_checkpoints` raised), run the S
  pipeline per checkpoint → S(t), N(t) curves at authentic schedule pace.

### 5.2 Stages (each gates the next; ~2 GPUs × 14h per run, ≤3 concurrent)

- **S0 — family pilot (running).** `addsub-L18-08-smoothl0-b` (= 04-hidden with SmoothL0,
  coeff 1e-4 flat — the 04-hidden peak value held throughout — γ 1→0.01) vs the existing `addsub-L18-04-hidden` control. Decides: is
  SmoothL0 viable at this coeff scale (its loss ≈ active-count, so the scale differs
  from L_p's)? If wildly off, recalibrate coeff before S1.
- **S1 — dose–response Pareto (6 new runs).** coeff × family grid
  {2.5e-5, 1e-4, 2e-4} × {L_p, SmoothL0} (5e-5 cells covered by S0). Hidden-acts fixed
  at 0.001. Output: R-vs-N Pareto curve per family (plot P1), S along the curve (P2).
  Sets τ and each family's **knee coeff** (last dose before recon degrades past the
  gate).
- **S2 — hidden-acts (4 new runs).** At each family's knee: hidden-acts
  {0.1→~0 decay, 0.1 constant} (0.001-constant = the S1 cell). Tests the imprinting
  main effect, the freeze hypothesis (constant vs decayed), and the dose×hid
  interaction.
- **S3 — schedule shapes (5 new runs, winner family+knee+hid from S2).**
  (a) coeff timing at matched dose: flat 7.5e-5 vs late-2x (early-2x = existing shape);
  (b) concavity window: late-only (50→100%), none;
  (c) hidden-acts decay ending at 50% instead of 100%.
  These are the runs whose **checkpoint dynamics** (S(t), N(t)) test the phase model
  directly: does moving a schedule move *when* N drops and *when* S improves?
- **S4 — confirmation (2–3 new runs + analyses).** 3 seeds of the chosen recipe;
  weight-side period analysis; final plots and recipe statement.

Naming: `addsub-L18-08-<family><coeff>[-hid<h><shape>][-<schedule>][-s<seed>]`, e.g.
`addsub-L18-08-sl0c1e-4-hid0.1decay`. Fresh names throughout (the deleted `-06-psep-*`
wandb ids are tombstoned).

### 5.3 Planned plots

1. **P1 Pareto**: rounded KL (y, log) vs active count N (x); families as colors, doses
   connected, seed error bars, gate line. The knee is the headline.
2. **P2 conditional separation**: n50 (y) vs rounded KL (x), gate region shaded — shows
   what separation is *buyable* at near-perfect recon.
3. **P3 training dynamics**: recon and NAlive vs step, one panel per factor, all levels
   overlaid (from metrics.jsonl, no extra compute).
4. **P4 separation dynamics**: n50 and N vs checkpoint step for the S3 schedule
   variants — the direct test that early schedules move P1 (N drops) and late schedules
   move P2 (n50 drops).
5. **P5 interaction**: R and n50 for the dose×hidden-acts cells (small grid heatmap).

### 5.4 Threats to validity, handled

- *Schedule compression* (killed the last attempt): every run is full-length; dynamics
  are read from checkpoints of full runs, never from shortened runs.
- *Seed noise*: replicates at center + final; factor effects only claimed when outside
  the replicate spread.
- *Anchor confound*: cross-run size uses the anchor-free active count + fixed-threshold
  sweep reads.
- *One-factor-at-a-time blindness*: the two interactions we have prior reason to expect
  (dose×hid, dose×concavity-timing) get explicit cells; everything else stays OFAT
  around a fixed center.

## 6. Status

- S0 pilot `addsub-L18-08-smoothl0-b` training (job 5115, 24k steps, ~14h): SmoothL0 coeff 1e-4 flat (the 04-hidden peak value held throughout).
- `addsub-L18-04-hidden` is the L_p control (already trained + analysed).
- S1 launches after the pilot's first eval points confirm the SmoothL0 coeff scale is
  sane (n_alive not collapsing to 0 or staying at C).
