# Adversarial (PGD) reconstruction on non-target data — experimental plan

2026-08-19. Goal: get a satisfying **output** PGD recon loss on the non-target stream
while adding as little compute as possible to the decomposition. Hidden-acts recon on
non-target is a helper — it does NOT need adversarial robustness (decision, this plan).

## Problem statement + baseline

The targeted dual-objective runs train non-target reconstruction stochastically only
(SPEC T5/T7: the non-target vocabulary is closed, adversaries are target-pass-only), so
worst-case masks on the broad stream are unconstrained. Measured on addsub-L18-14-4x
(p-bdf13b20, slow eval @ step 16000, fresh 20-step PGD, `source_shape: c`):

| metric | target stream | non-target stream |
|---|---|---|
| eval `loss/PGDReconLoss` (output) | 0.0044 | **0.259** |
| eval `hidden_ci/loss/PGDReconLoss` | 0.0035 | 0.276 (out of scope) |
| train stochastic recon (output), @19800 | — | 0.0019 |
| eval `ce_kl/kl_ci_masked` (output) | — | 0.118 |

So the adversary buys ~60x the target stream's worst case, ~135x the stream's own
stochastic recon. Step time to beat: **3.39 s/step** (2x L40, dp 2, sequential passes).

Success criteria:
- eval `nontarget_data/loss/PGDReconLoss` ≲ 0.05, ideally approaching the CI-masked
  KL (0.118 is the *average* CI-masked KL; the PGD probe landing near or below the
  random-masked KL band would mean the adversary gains ~nothing).
- step-time overhead ≤ ~5% for the from-scratch recipe (a fine-tune may spend more
  per step since it runs few steps).
- no regression on target-stream metrics (target PGD recon, CI-L0, ab-grid story).

## Why the naive fix is slow (cost model)

The broad stream dominates the step (seq 64 x batch 96 vs the pool's 5 tokens x 128).
A canonical PPGD term copied onto the non-target pass costs `n_warmup` extra
fwd+bwd ascents *on the broad stream* plus one extra masked forward — with the
production `n_warmup_steps: 2` that's ~3 extra broad-stream passes, roughly doubling
the step. That is what was observed when this was tried (torch branch
`feature/nontarget_ppgd`, commit 185c3ff9c, which also prototyped `delta_override`:
the non-target adversary attacks components only, delta pinned — we keep that
semantics, it is T4's pinned delta).

The cheap levers, in increasing overhead:

1. **Merged stochastic/persistent loss (S34), already implemented for the plain/target
   vocabulary** (`MergedStochasticSubsetPPGDReconLoss`): per batch element, adversarial
   with probability `adv_fraction` (schedulable), stochastic otherwise — ONE masked
   forward total, the S14' final ascent rides the shared backward. Replacing the
   non-target `StochasticReconSubsetLoss` with this term costs **zero extra forwards**;
   with `n_warmup_steps: 0` the bundle still gets one Adam ascent per step (final
   ascent), gated to the adversarially-assigned samples.
2. **Dedicated non-target PPGD term with `n_warmup_steps: 0`**: one extra masked
   fwd+bwd on the broad stream (~+15–25% step), stronger/cleaner gradient signal.
3. **Warmup ascents on the broad stream** (`n_warmup > 0`): the expensive part; only if
   the free adversary is too stale/weak.

## Constraints discovered (SPEC / code)

- T5/T7 close the non-target recon vocabulary against adversarial + mixed sources; T2
  sizes all persistent sources off *target* geometry. Admitting an adversary on the
  non-target OUTPUT pass is a deliberate SPEC amendment (T5/T7/T2), to be recorded like
  T12 was ("experimental choice this amendment enables"), pending Oli sign-off. The
  non-target HIDDEN vocabulary stays closed (this plan's scope decision).
- S23 (one bundle ↔ one term) survives: the non-target term gets ITS OWN bundle, sized
  `(96, 64, C+1)` for `bsc` or `(1, 1, C+1)` for `c`. Cross-stream *sharing* of a
  bundle (idea (c) below) would additionally amend S23 — deferred unless needed.
- T4 pinned delta: the non-target adversary's source delta channel is ignored; every
  non-target forward keeps `m_Δ = 1`. (= torch `delta_override` semantics.)
- The step machinery is already per-pass: `train.resolve_pass` indexes another pass's
  adversaries into the global state namespace (built for T12's hidden pass), and
  `ascend_adversaries` scopes warmup ascents to one pass's terms + scoring forward. The
  code change is mostly schema/type-narrowing + state sizing, not step surgery.
- Fine-tune init (S33, `resume_provenance`) restores decomposition only; fresh
  adversaries/optimizers/step; structure (site Cs, ci-fn arch) must match the PARENT —
  so fine-tunes of 14-4x use its Cs (o 128, down 456), NOT the new SOTA seat's widened
  Cs (o 256, down 512). Coefficients/schedules MAY change: hold every scheduled
  coefficient constant at its final value (imp-min 5e-05, freq 2.5e-05, equal
  non-target imp-min 5e-05 per the 14-equal decision).
- The EVAL adversary is `source_shape: c` (one shared source per site across the whole
  eval batch). A `c`-shaped training adversary matches that threat model exactly and is
  the natural *shared/recyclable* shape; `bsc` is the production target-pass shape.
  `bsc` on the non-target stream costs ~90 MB/rank of sources+moments at these Cs
  (fine), `c` is negligible.

## Ideas considered

(a) **Fine-tune the finished run** a few k steps with a non-target adversarial term,
    schedules held at final values. Cheapest total compute; directly answers "can the
    existing decomposition be patched to worst-case robustness without retraining".
(b) **Recycle target-pass adversarial sources on non-target batches.** As stated it
    needs shape compatibility: the target bundle is `bsc` over (128, 5) — not
    broadcastable to (96, 64). Only a `c`-shaped bundle transfers across geometries,
    which collapses (b) into (c).
(c) **One shared `c`-shaped bundle for both streams.** Needs the S23 amendment; the
    ascent signal is the sum of both terms' source-grads. Deferred: (1)/(2) above get
    the same compute profile without cross-term coupling, and the worst case of the
    5-token arithmetic pool and of fineweb likely differ — a shared source would chase
    both at once.
(d) **Merged S34 term on the non-target output pass** (mine): zero extra forwards, the
    lowest-overhead adversary possible. First choice.
(e) **Dedicated PPGD term, `n_warmup 0`** (mine): +1 broad masked fwd+bwd. Second
    choice / robustness fallback.
(f) **Ramped `adv_fraction`** (mine): 0 for the first ~half of a from-scratch run, then
    up — the early decomposition isn't worth attacking; may also stabilize training.
(g) Not pursued: fresh-PGD as a training loss on non-target (n_steps fwd+bwd per step —
    the expensive shape by construction); cadenced adversarial steps (breaks the static
    jit graph for marginal savings); sub-batch adversarial slice (S34's per-sample
    assignment is the same idea, cleaner).

## Prioritized experiments

- **E1 — code: admit adversaries on the non-target OUTPUT pass.** Config union
  (`NontargetReconLossMetricConfig` + `PersistentPGDReconLossConfig` +
  `MergedStochasticSubsetPPGDReconLossConfig`), type narrowing in
  `objective.NontargetPass`, non-target-geometry source sizing in state init, per-pass
  ascent wiring (reuse `resolve_pass`), T4 delta pinning over persistent sources,
  metric namespaces, SPEC amendments (T5/T7/T2, "pending sign-off"), tests (extend
  `test_dual_objective.py` pattern: parity of the merged term vs its expectation,
  checkpoint round-trip of the new bundle, step regression). Non-target hidden refuses
  adversaries as before. **Gates everything below.**
- **E2 — fine-tune patch-up of 14-4x (p-bdf13b20 @ 20000), 2000 steps.** The lowest
  hanging fruit. Config = 14-4x's, Cs unchanged, all scheduled coefficients constant at
  final values, `resume_provenance: {parent_run_dir: .../runs/p-bdf13b20, parent_step:
  20000}`, non-target output recon = **merged term** (`adv_fraction 0.5` constant,
  `n_warmup_steps 0`, `bsc`, source LR 0.01 as production) alongside nothing else
  (it subsumes the stochastic term in expectation). Watch: eval non-target PGD every
  500 steps; target metrics for regression. ~2h on 2x L40.
  - E2b (if adversary too stale): same but dedicated PPGD term (e) next to the
    stochastic term; accept the +1 forward for 2k steps.
  - E2c (if `bsc` staleness on streaming data is the problem): `source_shape: c`
    (matches the eval threat model exactly).
- **E3 — from-scratch SOTA arm.** New 20k run from the updated seat
  (addsub-L18-dual-obj: widened o/down) with the merged term on non-target output,
  `adv_fraction` ramp 0 → 0.5 over the first half (idea f), `n_warmup 0`. Expect step
  time ≈ unchanged (verify on a 50-step smoke before committing the slot). Launch
  after E2 reads out, informed by its adv_fraction/source-shape findings.
- **E4 — only if E2/E3 miss the bar**: warmup ascents on the broad stream (n_warmup 1)
  on the fine-tune shape, where the per-step cost multiplies only 2k steps; then the
  shared-bundle idea (c) with its S23 amendment.

## Execution notes

- 14-4x (p-bdf13b20, job 10069) finishes ~now (ETA <15 min at last log); 14-equal
  (p-6f3e11a0, job 10070) ~20 min later. Both free their 2x L40 slots — E2 can launch
  as soon as E1's code is validated (smoke: save/resume at production shape, per repo
  rule).
- Launch procedure: staged configs + sbatch in `~/pd_scratch/dual_obj_jax/`, frozen
  detached worktree under `~/pd_scratch/worktrees/<run-name>` at the commit under test
  (E1 lands on `feature/dual_obj_jax` first), `VENV_PY` on the main checkout's venv.
- Fine-tune runs are one-offs: configs stay in `~/pd_scratch`, never committed
  (CONFIGS.md rule 2). If the merged-term recipe wins at E3, the SOTA seat picks it up
  as a seat edit.

## Status log

- 2026-08-19: plan written. Baselines harvested (table above). E1 started.
- 2026-08-19: E1 LANDED (b9062f642): persistent + merged adversarial recon admitted on
  the non-target OUTPUT pass, delta-pinned, bundles sized per-pass; SPEC T2/T5/T7
  amended (pending sign-off); pinned by `core/tests/test_nontarget_adversary.py`; full
  core+targets suites, 4-sim-device variant, parse gate, and basedpyright all green.
  14-4x finished (rc=0): final eval nontarget PGD 0.270 output / 0.270 hidden vs
  target 0.0043 / 0.0033.
- 2026-08-19: targeted runs gained `resume_provenance` (601c9a756) — the schema had
  deliberately refused it ("semantics undefined"); E2 defines them: targeted parent
  only, compat-checked by parsing the parent's pin under the targeted schema.
- 2026-08-19: E2 deviation from the sketch above: the fine-tune keeps the PARENT's
  final coefficients (nontarget imp-min 1.0e-04, the 2x arm), NOT 14-equal's equal
  choice — one change at a time; the equal question belongs to the from-scratch E3.
  Configs staged in `~/pd_scratch/dual_obj_jax/addsub-L18-15-ntadv-ft{,-SMOKE}.yaml`,
  frozen worktree `~/pd_scratch/worktrees/nt-ppgd-ft` @ 601c9a756. SMOKE (60 steps,
  save/resume shape) submitted as job 10089.
- 2026-08-19: SMOKE GREEN. Job 10089 (p-bbd77143): fine-tune init from p-bdf13b20 @
  20000 clean; the merged non-target term trains; **step time 3.29–3.36 s vs the
  parent's 3.39, peak 39.7 GB/rank unchanged** — the zero-extra-forward claim holds.
  Save/resume leg (p-b94c5896, jobs 10090/10092): scancel exercised the SIGTERM-save
  fast path (saved mid-flight @ 26), resume picked it up, completed rc=0. One hiccup
  worth knowing: an immediate resubmit after scancel hit the GPU-contention guard
  (leg 1's process still held ~47 GB while dying) — wait for release before requeueing.
- 2026-08-19: E2 FULL RUN launched, job 10093 (2000 steps, ~2.2 h). Readout: eval
  nontarget PGD at steps 0 (parent baseline under this probe) / 1000 / 2000 vs the
  0.270 parent final; target-stream regression watch on kl_ci_masked (0.0033) and
  target PGD (0.0043).
