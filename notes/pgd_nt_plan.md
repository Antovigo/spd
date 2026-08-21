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
- 2026-08-20: E2a (merged, bsc) VERDICT — NOT EFFECTIVE as a 2k-step patch. p-bb64cede:
  nontarget PGD 0.270 -> 0.2535 (@250) -> 0.2207 (@1000) -> 0.2307 (@2000): a ~15%
  dent that plateaued and bounced. Target stream fully unregressed (PGD 0.0041,
  kl_ci_masked 0.0031). The free adversary's pressure is real but far too weak at this
  horizon. Two hypotheses now under test side by side:
  - STRENGTH: E2b (job 10098, sibling session): canonical PPGD term, n_warmup 2,
    5.89 s/step (+76%); its train adversary found 3.5x-worse-than-stochastic masks
    immediately.
  - THREAT-MODEL MATCH: E2c (job 10099, p-? see by-name/addsub-L18-15-ntadv-c-ft):
    E2a with the nontarget bundle C-SHAPED — the eval probe is `source_shape: c`, so
    bsc slot-persistence against streaming data may be optimizing the wrong worst case;
    the c bundle accumulates the probe's own dataset-level attack, still at zero extra
    forwards.
  Decision tree: E2b strong + E2c weak -> pay for strength (or recover it cheaply:
  higher adv_fraction / source LR, warmup on the fine-tune only); E2c ~matches E2b ->
  the free adversary suffices once aimed right; both weak -> the fine-tune framing is
  suspect, move the pressure into the from-scratch E3.
- 2026-08-20: E2c CANCELLED by Antoine at ~step 350 (job 10099, p-cdcf88d4; removed
  from disk and W&B). Its only readout: 0.2543 @250 — indistinguishable from the other
  arms at that horizon. The c-shape hypothesis is untested at the 1k/2k horizon, not
  refuted. E2b (job 10098) continues; the 16-line from-scratch merged runs are queued
  separately.
- 2026-08-20: E2b VERDICT (p-579bacee) — the strong adversary buys NOTHING over the
  free one at this horizon: 0.2530 (@250) -> 0.2240 (@1000) -> 0.2303 (@2000), within
  noise of E2a's 0.2535 -> 0.2207 -> 0.2307, at +76% step time. **E2 synthesis: both
  arms plateau at ~0.23 regardless of adversary strength, while E2b's TRAIN adversary
  demonstrably bites (3.5x-worse-than-stochastic masks from step 1) and the train-side
  merged/PPGD losses do fall. So the residual ~0.23 eval worst case is structural to
  the PARENT decomposition — 2000 steps at final LR cannot reshape V/U + CI enough to
  remove it, whatever the training adversary. The fine-tune patch-up framing (idea a)
  is the weak link, not the adversary.** For scale: the plateau (~0.23) sits ~2x the
  zero-masked KL (0.123), i.e. PGD still finds single masks worse than ablating every
  component. Next lever per the decision tree: the pressure belongs in from-scratch
  training (E3 — the 16-line merged runs, queued), where the adversary shapes the
  decomposition while it forms. Open fine-tune variants if wanted later: higher
  component/CI LR (a real re-training, not a patch), or the untested c-shape at full
  horizon.
- 2026-08-20: E2b LAUNCHED (job 10098, run per runs/by-name/addsub-L18-15-ntppgd-ft):
  the STRONG-adversary comparison arm — canonical separate PersistentPGDReconLoss on
  the nontarget output pass (coeff 0.5, n_warmup 2, bsc, next to the stochastic term),
  same parent/steps/constant schedules as E2. Its 60-step smoke measured the cost the
  merged arm avoids: 5.89 s/step vs 3.35 (+76%), peak 41.1 GB/rank (2.5 GB headroom);
  the nontarget PPGD train loss opened at ~3.5x the stochastic term's (0.0067 vs
  0.0019) — the adversary bites immediately. The {ntadv, ntppgd} pair reads out
  adversary strength per unit compute, and whether components trained without
  nontarget PGD are fixable at all.
- 2026-08-20: E2 first in-flight readout (step 250 slow eval): nontarget output PGD
  0.2535 vs 0.270 parent baseline — the free adversary dents it only slightly so far;
  target metrics unregressed (PGD 0.0045 / hidden 0.0035).
- 2026-08-20: smoke cleanup — both ntadv-ft-SMOKE runs (p-bbd77143, p-b94c5896) and
  the ntppgd-ft smoke (p-8506002e) deleted from disk, local wandb dirs, W&B cloud,
  and slurm logs.
- 2026-08-20: E1-BIS LANDED (40a52a516): the MERGED term admitted on the non-target
  HIDDEN pass too (T5/T12 amended, pending sign-off) — it rides the one masked forward
  the stochastic term already ran, so the hidden adversary is compute-free by the same
  argument as the output pass's; the standalone PPGD type stays refused there. State
  keys `nontarget_hidden/<name>`, bundles off broad geometry, delta pinned. Pinned in
  `test_nontarget_adversary.py`; full core+targets (629), 4-sim-device, basedpyright,
  Codex review all green.
- 2026-08-20: E3 SPLIT INTO TWO FROM-SCRATCH ARMS and LAUNCHED (both 20k, SOTA seat
  base, seed 0, frozen worktree `~/pd_scratch/worktrees/L18-16-merged` @ 40a52a516,
  configs in `~/pd_scratch/dual_obj_jax/addsub-L18-16-{ntmerged,allmerged}.yaml`; NO
  smokes — deliberate deviation from the smoke rule, owner's call, to take the freeing
  slots immediately):
  - **addsub-L18-16-ntmerged (job 10104, variant 2 = E3 proper; resubmitted over
    10100 pre-start)**: target stream's loss structure unchanged, but its PPGD bundles
    re-shaped bsc -> sc (owner's call, post-submit): both arms now share sc on the
    target stream, so the {ntmerged, allmerged} pair differs ONLY in merged-vs-separate
    terms — the sc-vs-seat change is common-mode, read against the 14-4x/seat history
    instead. Non-target output AND hidden stochastic terms each -> Merged coeff 1.0,
    `adv_fraction` ramp 0 -> 0.5 over the first 100 steps (idea f, revised from
    "first half"), `n_warmup 0`, `source_shape: c` (E2c's threat-model-match choice).
    Step time should hold at ~3.35 s.
  - **addsub-L18-16-allmerged (job 10105, variant 1; resubmitted over 10101 to keep
    ntmerged first in the FIFO)**: every stochastic/PPGD pair collapsed into the
    merged term. Target output 1.5 / target hidden 3.0 (matching the pairs' totals;
    adv_fraction 0.5 const evens the split to 0.75+0.75 and 1.5+1.5 vs the seat's
    1.0+0.5 / 2.0+1.0 — a deliberate rounding), `n_warmup 2` kept on the target
    terms, `source_shape: sc` on the target stream (sc is a batch-shared, weaker
    per-sample adversary than the seat's bsc; with ntmerged now also sc, that change
    is common to both arms rather than a confound between them). Non-target side
    identical to ntmerged. Expect a mild step-time WIN (the target PPGD's extra
    masked forward folds into the merged one).
  - Possible next step, deliberately deferred: `n_warmup 0` on the target merged
    terms (true zero-extra-forward recipe everywhere; risks a staler target adversary).
  - Launch churn: E2c was cancelled by its owner; jobs 10104/10105 started onto its
    freshly-freed GPUs and died on the contention guard (exit 75 — the release-lag
    hiccup the E2 smoke recorded). Relaunched sequentially (each arm submitted only
    after the previous one is stepping): ntmerged = **job 10106**, allmerged =
    **job 10107**, both RUNNING 2026-08-20.
  - First readings: ntmerged 3.374 s/step @ peak 39.74 GB/rank — step-time parity
    with the seat (3.35-3.39) holds with merged adversaries on BOTH non-target
    passes; its nontarget merged train losses open ~3-4x the stochastic-only
    counterpart's (output 0.0058 / hidden 0.0091 @ step 1600 vs E2a's ~0.002) — the
    free adversary bites from scratch too. allmerged 3.581 s/step @ peak 36.45 GB/rank —
    ~6% over the seat despite one fewer masked forward per pair (folding the
    separate PPGD forwards does save 3.3 GB/rank); suspects: the sc bundles'
    different broadcast, and warmup ascents now scoring the merged plan.
- 2026-08-20 (in-flight finding, jobs 10106/10107): TARGET-stream eval PGD is ~10x
  worse than the 14-line at matched steps (ntmerged 0.328@500 / 0.082@4000 vs 14-4x
  0.035 / 0.0086; 14-equal matches 14-4x, so equal-impmin is exonerated). NOT the
  metric: `recon_eval.py` is byte-identical since the 14-4x commit (2b0018d38), the
  slow_eval diff is figure layout, and the E2 fine-tunes reproduced the parent's
  0.0043 under the post-E1 code. The smoking gun is the sc-shaped TARGET sources
  (the one target-side change common to both arms): ntmerged's persistent adversary
  reports a BETTER loss than 14-4x's (train PPGD 0.0052@4000 vs 0.0070) while fresh
  eval PGD finds 16x worse — one batch-shared source is a single tracked attack
  point (vs bsc's 128 parallel per-sample attacks), so it undercovers and certifies
  robustness the decomposition doesn't have. Stochastic recon / kl_ci_masked stay
  within ~1.5x (widened-C-sized), so the regression is worst-case-specific.
  Implication: bsc's per-sample parallelism is load-bearing for target robustness;
  sc is a bad training shape here even though it strictly contains the c eval space.
- 2026-08-20: SWAP DECISION (owner): let each sc arm reach its step-5000 checkpoint
  (save_every 5000 — first comparison point on disk, plus the scancel SIGTERM-save
  adds a ~51xx snapshot; keep_last 2 retains both), then cancel it and launch its
  -bsc twin: `addsub-L18-16-{ntmerged,allmerged}-bsc` — byte-identical configs
  except target-stream sources sc -> bsc (nontarget stays c). Cost of bsc on the
  pool geometry is nil (~15 MB/bundle, no step-time effect; 14-4x ran 3.39 s with
  it). The sc runs (p-b62de0d9, p-ae980ca6) are kept as the sc datapoints. Swap
  watchers automate ckpt-wait -> scancel -> guard-retry relaunch.
- 2026-08-20: SWAP EXECUTED. sc arms stopped at their step-5000 checkpoints (ntmerged
  p-b62de0d9: ckpts 5000+5132; allmerged p-ae980ca6: ckpts 5000+5128 — the sc
  datapoints for the shape comparison). -bsc twins running: ntmerged-bsc job 10123,
  allmerged-bsc job 10137 (each first submit died on the GPU release-lag contention
  guard; the retry launcher got both through). **sc diagnosis CONFIRMED**: ntmerged-bsc
  eval target PGD @500 = 0.0308, back in the 14-4x band (0.0354) vs the sc arm's
  0.328 — a 10x recovery from the shape change alone; step time 3.383 s (parity) and
  peak 39.84 GB, so bsc costs nothing here, as expected. allmerged-bsc peak 36.65 GB.
- 2026-08-20: 8000-STEP READOUT, bsc twins vs 14-4x. Target side healthy (output PGD
  0.0061 ntmerged-bsc = 14-4x; allmerged-bsc 0.0069, ~13% behind and closing; hidden
  similar; target L0s slightly better than baseline). **Non-target side: the free c
  adversary is NOT delivering** — nt output PGD 0.254 (ntmerged-bsc) / 0.269
  (allmerged-bsc) vs 0.268 baseline, and the trend is non-monotone (0.24 -> 0.19 @4k
  -> 0.25 @8k: the early dent evaporates). Same undercoverage pathology as sc on
  target: the persistent c bundle's own train loss reads ~0.006 while fresh PGD finds
  0.25 — one shared source cannot track the broad stream's worst case; per-sample
  parallelism (bc/bsc, the thing that makes the target adversary honest) is the
  obvious next lever, at ~90 MB/rank for bsc. ALSO: nontarget L0 inflates under the
  weak pressure (output 6.2 -> 13-15, hidden 6.9 -> 38-48) — the decomposition buys
  off the adversary with active components rather than robustness.
- 2026-08-20: NEW ARM launched — addsub-L18-16-allmerged-bsc-w3: allmerged-bsc with
  the two TARGET merged terms at n_warmup 3 (one extra ascent), probing whether the
  merged arm's small target-PGD lag is warmup-limited. Non-target unchanged.
- 2026-08-20: cleanup — the abandoned w3 arm (p-afa6ccd9, job 10154) deleted from
  disk, by-name, local wandb, W&B cloud, and its staged config/sbatch; the
  contention-guard stub slurm logs (10104/10105/10122/10124), the historical smoke
  logs (smoke-abgrid-bindtime-*, tpd-jax-smoke*, jax-dual-01-smoke, seat-smoke), the
  derived -SMOKE configs (regenerable via derive_*_smoke), and the abgrid-smoke
  frozen worktree all removed. Kept: both sc runs (comparison data), the
  L18-16-merged worktree (still backing jobs 10123/10137 until the evalfix cycle).
- 2026-08-20: stub cleanup — the cancelled w3 run (p-afa6ccd9: run dir, by-name link,
  slurm log; no cloud wandb existed) and the guard-stub job logs (10104/10105/10122/
  10124) were removed (partly by a concurrent sweep from the owner's side); this
  session removed the last leftover: p-83b757e5 + its job-10096 log — an E2a requeue
  leg that minted a fresh run id instead of resuming (no ckpts, superseded by
  p-bb64cede). runs/ now holds exactly the nine real runs, each with its by-name
  link. The w3 config/sbatch stay in ~/pd_scratch as the deferred n_warmup-3 recipe.
- 2026-08-20: sc runs REMOVED too (owner): p-b62de0d9 / p-ae980ca6 deleted from disk,
  by-name, local wandb, W&B cloud, slurm logs, and their staged configs. Their key
  numbers survive in this log (sc target PGD 0.328@500 / 0.082@4000; ckpt-5000
  freeze); the {bsc twin, 14-4x} pair carries the shape comparison from here.
- 2026-08-20/21: ROOT CAUSE FOUND — the nontarget PGD eval was attacking the DELTA
  CHANNEL. `masks_from_sources` hands the probe the sources' trailing channel as the
  weight-delta mask, while every non-target TRAINING forward pins delta to 1.0 (T4).
  On the broad stream the delta escape valve carries the reconstruction BY DESIGN
  (CSS-only floor ~0.113 KL), so the probe's dominant move was delta->0 + component
  steering (~0.25-0.27, even beating full-layer ablation's 0.119) — an attack space no
  component-side training can enter. This, not adversary weakness, is why E2a = E2b =
  E2c = both 16-line arms sat flat. The "free adversary too weak" reading above is
  RETRACTED for the output metric (the L0-inflation observation stands).
- 2026-08-21: EVAL FIX LANDED (b05904f3e, SPEC T4 amended, pending sign-off): a
  targeted run's nontarget-stream fresh-PGD probes (both roles) now compose
  delta-pinned; plain runs/target stream unchanged. Both bsc twins cycled onto the fix
  (jobs 10157/10158, resumed from own ckpts @14204/11514; corrected in-loop numbers
  from their next slow evals). Backfill of retained ckpts via
  `~/pd_scratch/dual_obj_jax/pgd_backfill*.{py,sbatch}` (one process per (run, ckpt) —
  job 10155's in-process double-restore wedged on BFC pressure + NCCL stall; ckpts
  preserved under `~/pd_scratch/ckpt_preserve`), logged to `<run>-pgdfix-backfill`
  sibling wandb runs.
- 2026-08-21: CORRECTED-METRIC READOUT (delta-pinned nontarget PGD, output/hidden):
  - 14-4x @20000 (UNTREATED baseline): 0.0168 / 0.0157 (unpinned control 0.222) —
    the component-only worst case was ALREADY ~0.017, inside the <=0.05 bar: the plain
    stochastic nontarget term defends it. The campaign's 0.259 premise was the
    delta-channel artifact end to end.
  - ntmerged-bsc @5000: 0.0172 / 0.0172 (unpinned 0.261).
  - E2a ntadv-ft @1000: 0.0158 / 0.0156 — the free merged fine-tune trims the
    baseline ~6% in 1000 steps; more panels (E2a@2000, E2b, 14-equal, twins) pending.
  - OPEN QUESTION reframed: do the adversarial terms tighten an already-acceptable
    0.017 (and at what L0 cost off-target), rather than "can 0.26 be fixed".
- 2026-08-21: BACKFILL COMPLETE (job 10159, rc=0; all panels in
  `pgd_backfill_results.json` + per-run `<id>-pgdfix-backfill` wandb siblings). The
  corrected (delta-pinned) nontarget PGD picture, output/hidden:
  | run | step | pinned | (unpinned control) |
  | 14-4x untreated (2x nt impmin)  | 20000 | 0.0168 / 0.0157 | 0.222 |
  | 14-equal untreated (equal)      | 20000 | 0.0159 / 0.0159 | ~0.22 |
  | E2a free merged ft              | 2000  | 0.0149 / 0.0150 | 0.249 |
  | E2b strong PPGD ft (+76% step)  | 2000  | 0.0133 / 0.0137 | ~0.25 |
  | ntmerged-bsc from-scratch       | 10000 | 0.0139 / 0.0136 | 0.248 |
  | allmerged-bsc from-scratch      | 10000 | 0.0137 / 0.0140 | 0.265 |
  READINGS: (1) the component-only worst case was ~0.017 even untreated — inside the
  <=0.05 bar; the campaign's 0.259 premise was the delta-channel artifact entirely.
  (2) Fine-tuning DOES tighten it: -11% free / -21% strong in 2k steps — adversary
  strength buys ~2x speed at +76% compute. (3) The from-scratch merged arms reach the
  strong fine-tune's level by 10k steps at zero step-time cost, both variants
  equivalent by 10k. (4) 14-equal slightly tighter than 14-4x — heavier nt imp-min
  correlates with a looser worst case. Success criterion should be restated against
  the corrected metric (e.g. "<= untreated baseline - X%" rather than the artifact-era
  0.05-vs-0.26 framing); the L0-inflation trade (nt out 6->13-15, hid 7->38-48)
  is now the main open cost question for the adversarial terms.
- 2026-08-21: CAMPAIGN CLOSE-OUT — both bsc twins finished 20k (rc=0). Final panels
  vs 14-4x (14-4x's nt-PGD cell is its OLD unpinned in-loop value; its corrected
  number is the backfill's 0.0168):
  | @20000 | 14-4x | ntmerged-bsc | allmerged-bsc |
  | PGD target out / hid        | 0.0043 / 0.0032 | 0.0043 / 0.0037 | 0.0048 / 0.0034 |
  | PGD nt out / hid CORRECTED  | (0.0168/0.0157) | 0.0091 / 0.0092 | 0.0098 / 0.0094 |
  | kl_ci_masked target         | 0.0033 | 0.0035 | 0.0034 |
  | L0 target out / hid         | 19.7 / 32.2 | 19.9 / 32.2 | 22.3 / 35.5 |
  | L0 nt out / hid             | 0.35 / 0.39 | 0.77 / 0.83 | 0.88 / 0.87 |
  VERDICT: the merged nontarget adversary (out + hid, c-shape, adv ramp 0->0.5/100,
  n_warmup 0) tightens the corrected nontarget worst case ~45% below the untreated
  baseline at ZERO step-time cost, with the target stream at full parity (ntmerged) —
  and the mid-training nt-L0 inflation (6->15/48 at 8k) anneals away to a negligible
  +0.4-0.5 absolute by 20k. ntmerged-bsc is the seat-worthy recipe; allmerged-bsc
  trades a hair of target PGD/L0 and +6% step for -3.3 GB/rank. Remaining decisions
  for Oli: T4/T5/T7/T12 amendment sign-offs (incl. the eval delta-pinning), whether
  0.009-vs-0.017 justifies seat promotion, and the E2b-style strong arm's 2x-faster
  fine-tune path as the patch-up recipe for existing runs.
- 2026-08-21: SUBSPACE SCATTER PORTED TO JAX RUNS (jobs 10228/10235/10236/10237).
  Scratch-side pipeline (`~/pd_scratch/dual_obj_jax/subspace_{export,build}.py` +
  `subspace_applet.sbatch`, per CONFIGS.md one-off convention): a JAX exporter
  (open_jax_run + clean_forward captures `post_attn/mlp_in/mlp_hidden/site .out`,
  output-role CI, x.V-hat inner acts) writes the torch validation pipeline's exact
  artifact formats; the torch-era `compute_subcomp_periods` + `build_subspace_scatter`
  then run UNCHANGED from the probe-linear-frame worktree venv, with only
  `load_component_uv` shimmed to an exported `uv_directions.npz`. Deliberate v1
  deviation: the alive candidate list is the mean-CI>0.1 filter alone — the
  `find_alive_subcomponents` KL-sweep gate is not reproduced (a few extra pick cards
  possible). Applets (self-contained index.html):
  - ntmerged-bsc  (p-5b7fa697): analysis/subspace_scatter/ — 85 add / 71 sub alive
  - allmerged-bsc (p-07eaa3d1): analysis/subspace_scatter/ — 93 add / 81 sub alive
  - 14-4x         (p-bdf13b20): analysis/subspace_scatter/ — 91 add / 76 sub alive
  Plane scatter (alive_plane_scatter) remains unported (needs a JAX collection pass;
  ridge_cv_probes jsons are base-model and reusable).
