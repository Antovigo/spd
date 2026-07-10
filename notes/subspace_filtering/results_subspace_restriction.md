# Subspace restriction — results log

Running results for the experiments in `plan.md`. Reference:
`addsub-L18-04-hidden` (dense, 24k steps). Battery numbers are **raw**-flavor mean KL
over the active subset, add+sub pooled, unless stated.

## E0 — singular spectra (2026-07-09)

All 7 L18 matrices are numerically full rank (r = min dim at τ = 1e-2, except q/o
losing a handful of directions). Spectra are flat; meaningful truncation starts around
τ = 0.1. Ranks at candidate thresholds:

| module | full | τ=0.1 | τ=0.2 | τ=0.3 |
|---|---|---|---|---|
| gate_proj | 4096 | 2760 | 297 | 17 |
| up_proj | 4096 | 4065 | 3580 | 2334 |
| down_proj | 4096 | 4066 | 3221 | 1797 |
| q_proj | 4096 | 1451 | 366 | 95 |
| k_proj | 1024 | 787 | 411 | 176 |
| v_proj | 1024 | 1022 | 942 | 801 |
| o_proj | 4096 | 2420 | 1279 | 490 |

Consequence: at τ = 0 the parameterization constrains only the wide sides (gate/up
write, down/k/v read) — exactly where the battery's legality tests bite. E3 uses
τ ∈ {0.1, 0.2}. Artifacts: `~/pd_scratch/subspace_restriction/spectra/`.

## E1 — retrofit of the dense checkpoint (2026-07-09)

`retrofit_svd_projection` at τ = 1e-5 → `addsub-L18-04-hidden-Aretro`. Relative mass
removed by the projection, `‖ΔV‖/‖V‖` resp. `‖ΔU‖/‖U‖` on the constrained sides:
**0.84 (down V), 0.81 (gate U), 0.82 (up U), 0.85 (k V), 0.83 (v V)** — i.e. at the
level a random vector would have (√(1 − 4096/14336) ≈ 0.845). Training never removed
the illegal mass present at init. q/o: ~0.02 (full-rank square, no-op).

Battery, raw flavor, mean KL (mlp block):

| experiment | dense ref | retrofit |
|---|---|---|
| circuit_baseline | 0.0117 | **0.5210** |
| circuit_in_row:down | 0.1741 | 0.5212 (= baseline) |
| circuit_out_col:gate | 0.0469 | 0.5210 (= baseline) |
| circuit_out_col:up | 0.0617 | 0.5210 (= baseline) |
| orig_in_span:down | 0.1871 | 0.1265 |
| orig_in_span:gate/up | 0.148 | 0.148 |
| orig_out_span:gate | 0.1058 | 0.2160 |
| orig_out_span:up | 0.3089 | 0.2493 |
| orig_out_span:down | 0.0978 | 0.0978 |

Attention block: every number unchanged (ratio 1.00) — q/o are no-ops and k/v's
illegal read mass is dormant (orthogonal to on-distribution activations).

Conclusions:

1. Legality (F3) becomes structural, as designed: the circuit-side projections are
   exact no-ops after retrofit.
2. **The illegal mass was functionally load-bearing for the MLP circuit**: removing
   it degrades the circuit baseline 44× (0.012 → 0.52 nats). It cancels in the full
   sum but does real work in the masked circuit. Post-hoc projection without
   fine-tuning is not viable; the interesting question is E2 (training under the
   constraint from scratch).
3. F1 (orig-span tests) barely moves — no free win from projection alone.

## TMS-5-2 (tied pentagon target): tied vs untied vs SVD-restricted (2026-07-09)

Target: the original tied TMS-5-2 (`goodfire/spd-pre-Sep-2025/runs/0hsp07o4`,
migrated to `~/pd_scratch/subspace_restriction_tms/targets/tms5-2`; pentagon W, tied,
bias ≈ −0.25). Three decompositions (C=20, impmin 3e-3 pnorm 1.0, 10k steps), runs
`~/out/runs/tms_5-2_tiedtarget_{tieddecomp,untieddecomp,svddecomp}`:

- **tieddecomp** (components tied like the target): perfect — 5 components/matrix,
  |cos to e_i| ≈ 0.995, IdentityCIError 0.
- **untieddecomp** (tied target, untied components): perfect at seed 1 — 5+5
  components, cos ≈ 0.99, IdentityCIError 0. Seed 0 merged features 0+4 in linear2
  (4 error); seed sensitivity, not hyperparameters (coeff 1e-3 and pnorm 2.0 variants
  were worse; sweep runs kept as `tms52tune-*`).
- **svddecomp** (untied + `svd_rank_threshold: 0.0`): **collapses to exactly
  r = 2 components per matrix** with dense CI (IdentityCIError 18); functionally
  excellent (recon 1e-4, faithfulness 1e-3) but per-feature structure is gone, and
  weaker minimality (1e-3, 3e-4) does not recover it. Structural, not a tuning
  issue: the per-feature read vectors e_i do not lie in the 2-dim row(W1), so the
  identity solution is not expressible under the restriction — on a matrix whose
  rank is far below the number of features it transmits (superposition), the
  restriction forbids exactly the mechanism we want. Caveat: this is NOT ruled out
  for 8B L18 by the E0 full-rank finding — TMS-5-2's W1 is also numerically full
  rank (2 = min dim); superposition means features > min-dim, which weight spectra
  can't see. L18's constrained interfaces have TMS-like compression ratios
  (down 4096/14336, k/v 1024/4096, gate/up col 4096/14336 vs TMS 2/5), so whether
  they are in the superposed regime is exactly what E2's CI structure (n_alive,
  heatmap density at those interfaces) will show. Only square q/o at τ=0 are
  provably unaffected.

Battery (raw flavor, mean per-sample MSE, 4096 samples; baseline in row 1):

| experiment | tied | untied | svd |
|---|---|---|---|
| circuit_baseline | 0.00022 | 0.00023 | 0.00006 |
| circuit_in_row:linear1 | 0.00851 | 0.00882 | **0.00006 (= baseline)** |
| circuit_out_col:linear2 | 0.00263 | 0.00258 | 0.00560 |
| orig_in_span:linear1 | 0.00004 | 0.00004 | 0.00058 |
| orig_in_span:linear2 | 0.00001 | 0.00001 | 0.00063 |
| orig_out_span:linear1 | 0.00001 | 0.00002 | 0.00003 |
| orig_out_span:linear2 | 0.00019 | 0.00024 | 0.00528 |

Readings:

1. Both dense decompositions **fail legality on linear1's read side ~40× baseline**
   (same signature as the 8B MLP): the perfect per-feature mechanism itself reads
   e_i ∉ row(W1), i.e. on a rank-deficient matrix the *ground-truth* mechanism is
   "illegal" by the F3 criterion. F3-as-hard-constraint and per-feature mechanisms
   are mutually exclusive under superposition.
2. The SVD run passes `circuit_in_row` exactly (structural) but is *worse* on the
   span tests (`orig_out_span:linear2` 28× the tied run) — its two dense components'
   per-sample spans no longer carry per-feature signal.
3. `circuit_out_col:linear2` is elevated for all three: the frozen output bias is not
   in col(W2); that term is what the bias flavor isolates, not a decomposition defect.

Plots: `beeswarm_tms.png` (boxplot replaced by density-binned beeswarm — discrete
5-input MSE distributions collapse in a boxplot) and
`mse_by_feature_tms[_single_active].png` under each run's `analysis/subspace_filtering/`.

## E2 — addsub-L18-05-svd-tau0 (in flight)

Memory probe (steps=3): peak 44.7 GiB / 46 — fits with the ~1 GB of fp32 Q buffers.
Found + fixed a real bug: DDP's default per-forward buffer broadcast in-place
clobbers Q_in/Q_out saved for backward → `broadcast_buffers=False` (all buffers are
static and rank-identical).

Full run: SLURM job 4158 (4157 died pending on QOSMaxWallDuration — resubmitted at
24 h), 24k steps, 2×L40, exact reference recipe + `svd_rank_threshold: 0.0`.

### Early investigation (step ~1600): why is it ahead?

Vs the reference at matched steps — three separable effects:

1. **Better init.** Step 0: unmasked KL 0.48 vs 1.27, eval PGD-recon 1.02 vs 4.32.
   Random coordinates confined to row/col(W) damage the model far less than ambient
   random init.
2. **Growing lead on sparsity, not just a head start.** Equivalent-quality lead on
   CI-L0: ref needs +500 steps at step 500, +1500 steps by step 1500 (12.9 vs 15.0
   total CI-L0). kl_ci holds a ~constant 500-step lead (0.0160 vs 0.0171).
3. **The dominant effect is targetedness.** Nontarget CI-masked recon 0.026 vs 0.057
   (2.2×), nontarget L0 1.36 vs 2.65, nontarget train ImpMin ~3× lower. Largest
   per-matrix CI-L0 win at the most-restricted read interface (v_proj, rank
   1024/4096): 0.17 vs 0.35 (5× at step 500).

Mechanism (corrected — the global_shared_transformer CI fn reads raw site
activations, not `x·V`, so the gate *inputs* are identical across runs): illegal
mass inflates each component's *functional footprint under masking*. A dense
component's standalone contribution `(x·V_c) U_c` responds to null(W) directions
that only cancel across the full sum; every partial mask (stochastic subsets, PGD,
CI-masked eval) breaks the cancellation, so the trained gates must keep components
more-on, on more inputs — especially off-distribution, where null-space directions
are active. The restriction shrinks footprints to the function-shaped part, letting
gates shut components off cheaply. Fits the PGD gap (adversarial masks exploit
cancellation-breaking) and E1's 44× circuit shift when the illegal mass was deleted
post-hoc. No TMS-collapse signature (CI got sparser, not denser) — consistent with
addsub routing far fewer features through L18 than the ranks allow.

## Soft legality pressure (transfer to the dense parameterization)

`PDConfig.legality_pressure = {rank_threshold, project_init, decay_coeff}`
(commit `2b1c1ca16`): keeps dense V/U, optionally (a) projects the fresh init into
row/col(W), (b) decays only the out-of-space mass by `lr * decay_coeff` per step.
Superposition-compatible: illegal mass survives wherever the losses defend it (the
TMS-5-2 per-feature solution pays a finite, recoverable cost, unlike under the hard
restriction). Attribution pair on the reference recipe:

- `addsub-L18-05-dense-projinit` — init projection only (isolates effect 1).
- `addsub-L18-05-dense-legdecay` — projection + decay 1.0 (adds the training-time
  pressure; cumulative unprotected shrink over 24k steps ≈ e^-4 of illegal mass).

(Superseded by the seed-controlled init study below.)

## Init study — 3 seeds × {dense, legdecay, projinit, svd}, 1000 steps (2026-07-10)

12 runs (`initstudy-L18-<t>-s{0,1,2}`), each replicating the first 1000 steps of the
reference recipe (schedules remapped: LR cosine→linear segment ≤0.1% error, pnorm
endpoint 1.9375 exact, PPGD warmup 600 steps exact, ImpMin coeff ≈ constant 2×
(≤2.1% drift); eval every 100). `legdecay` (`project_init: false`, decay 1.0) has
**bit-identical init to `dense` per seed**; `projinit` isolates the init;
`svd` = hard constraint. Sanity: `svd-s0` matches E2's step-1000 kl_ci within 1.3%.

Step-1000 means (3 seeds), ratio to dense:

| metric | dense | legdecay | projinit | svd |
|---|---|---|---|---|
| kl_ci_masked | 0.0212 | 0.97 | 0.94 | 0.90 |
| CI_L0 total | 16.98 | 0.92 | 0.90 | 0.88 |
| n_alive total | 1168 | 0.76 | 0.73 | 0.69 |
| PGD recon (eval) | 0.0584 | 0.77 | 0.67 | 0.54 |
| nontarget ci-masked | 0.0885 | 0.78 | 0.61 | 0.44 |
| nontarget L0 | 4.35 | 0.74 | 0.59 | 0.41 |

Readings (figure: `~/pd_scratch/subspace_restriction/initstudy_curves.png`):

1. Consistent ordering dense > legdecay > projinit > svd on essentially every
   metric; the nontarget panels separate beyond seed bands.
2. **Init is the dominant identified factor at this horizon**: projinit alone
   captures ~60–70% of svd's nontarget advantage and most of the L0/n_alive gain.
3. **The decay is real but slow-acting**: with identical init, legdecay bends away
   from dense late in the window — by step 1000 the cumulative shrink (Σ lr·coeff ≈
   0.32) has removed only ~27% of unprotected illegal mass (vs ~98% by 24k), so the
   short run *underestimates* its converged effect.
4. Effects are roughly additive → the combined soft treatment (projected init +
   decay) is predicted to approach the hard-svd numbers. Being tested as
   `initstudy-L18-soft-s{0,1,2}`.
5. The residual svd-vs-projinit gap (e.g. nontarget L0 0.41 vs 0.59) is the
   constraint/optimizer-geometry contribution — the part a soft method must win via
   the decay's cumulative action (or not at all).
