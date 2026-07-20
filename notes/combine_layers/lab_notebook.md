# Lab notebook — combine_layers

## 2026-07-18

### Deferred-anneal results — best merge so far

- **Slip:** the 10k-step edit to the singles' YAMLs raced their job start (the
  svd-tpd pipeline freed GPUs ~16h before its ETA and the jobs had already read the
  configs) — they ran the full 20k/20k/24k. Net effect: step-matched to the original
  sources after all; only the merge (4831) ran at the requested 10k.
- Flat-schedule singles ≈ sources on recon (0.0056–0.0070) at higher L0
  (11.9/10.2/18.8 vs 9.6/8.0/12.6) — marginal components survive p=2, as predicted.
- Anneal merge `combine-L16-18-anneal-01`: **rounded 0.0133 / PGD 0.0217 / L0 40.3 /
  ntgt 0.0108 / ntgt L0 0.40**. Recon 2.2× singles (4-block merges: ~4×), L0 = sum of
  singles, ntgt L0 below the singles' sum. Raw combination of the flat singles still
  fails (0.2316). At step 2k (old budget) already 0.0183 < both-02's 0.0239.
- Control launched (job 4941, `combine-L16-18-endstate10k-01`): original annealed
  L16/17/18 sources, end-state impmin, 10k steps — isolates deferred-anneal from
  the 3-block/10k confounds.

### Control result (2026-07-19): deferred anneal is a wash

- Control: rounded 0.0139 / PGD 0.0290 / L0 38.5 / ntgt 0.0098 / ntgt L0 0.39 —
  within noise of the anneal merge on everything except PGD (0.0290 vs 0.0217, the
  anneal's one consistent ~25% edge; n=1, suggestive only).
- Trajectories overlap (0.0188 vs 0.0183 at 2k) → my "soft phase repairs fast"
  claim was an LR-schedule artifact (10k cosine still ~0.9× LR at step 2k; the 2k
  merges were fully decayed). Both merges converge to L0 ≈ 39 from opposite sides.
- Net: the 0.023→0.013 improvement over the 4-block merges = budget (+ maybe
  3-vs-4 blocks); *where* the anneal happens doesn't move the joint objective.
  Report section rewritten accordingly.

## 2026-07-17

### Deferred-anneal experiment launched (flat-schedule singles + annealing merge)

- Idea (user): the sources annealed coeff ×2→×1 and p 2→0.5 *during single-block
  training*, so pruning decisions were binarised in the wrong (intact-model) context.
  Retrain L16/L17/L18 with **pinned schedules** (pnorm 2 throughout, coeff pinned at
  the source's peak = 2× base: 1e-4/1e-4/6e-5), then run the *entire* anneal during
  the merge. Per the threshold account, p=2 keeps interior optima (graded CIs, no
  hard pruning), so cross-block redundancy should still be represented when the
  binarising anneal happens — in the combined context, where its marginal value is
  priced correctly.
- New configs `~/pd_scratch/combine_layers/configs/addsub-L1{6,7,8}-06-flatsched.yaml`
  (copies of the sources; only impmin schedule fields changed; **all 10k steps** —
  user shortened from the sources' 20k/24k, so absolute quality is not directly
  comparable to the originals, only the schedule contrast is). New `--impmin_anneal`
  in `combine/finetune.py`: merge impmin replays the source schedule over the merge
  steps (coeff 2×→1× of `--impmin_coeff`=3e-5, i.e. 6e-5→3e-5; p 2.0→0.5). Merge =
  "both" recipe (grouped CI fns, everything trains), batch 32, **10k steps**, label
  `combine-L16-18-anneal-01`. Note the merge is 3 blocks (no L19, per the ask).
- Scheduling: user's svd-tpd pipeline holds all 6 GPUs. Jobs 4819 (L16, after the
  3 svd-tpd jobs) and 4820/4821 (L17/L18, additionally after the 3 psep-analyze
  jobs) each take 2 GPUs — worst-case concurrent usage stays ≤ 6. Merge + eval job
  4831 (re-submitted after the 10k-step change; sbatch spools scripts, so the queued
  4822 had to be cancelled) gated afterok on all three. Expected: singles start
  ~+17h, run ~8–9.5h; merge ~6h at batch 32 (2.1 s/it per the frzalive run); results
  in ~33h.

## 2026-07-16

### freeze_alive_train_dead results

- Job 4791 completed 2000 steps (~70 min, 2.08 s/it); standalone eval (job 4796):
  rounded **0.0359**, PGD 0.0703, L0 152.6, ntgt rounded 0.0131, ntgt L0 2.50.
- Beats the over-sparse run (0.0431/0.126) on both recon metrics with zero freedom on
  alive weights → the repair is substantially routing + new glue, not weight
  adjustment. Gap to the unfrozen fresh-CI variant remains (0.0359 vs 0.0266), L0
  highest of all variants, still falling at cutoff.
- Per-block L0 45.2/26.4/62.9/18.1 (L16/17/18/19): the L16/L18 redundancy-carrier
  pattern replicates the obj-4 resurrection finding via an independent protocol.
- First launch (4790) crashed instantly: fire parses `--tags=a,b` into a tuple but
  `init_pd_run` expects a comma string — pass a single tag.
- Report section + obj2 figures updated (new subject in dot plot and trajectories).

### freeze_alive_train_dead launched

- New variant (job 4790, `combine-L16-19-freeze_alive_train_dead-01`): freeze the
  sources' reference-alive subcomponents (per-run `alive_subcomponents.tsv`:
  146/100/177/177 for L16/17/18/19), train only the dead subcomponents + ONE fresh
  global CI fn (obj-3 settings: ci_fn_lr 1.6e-4, components_lr 1e-4, batch 32).
  Hypothesis: all repair is forced into previously-dead capacity, so the validated
  single-block mechanisms provably cannot be polluted (stronger guarantee than the
  completeness protocol's per-block attributability).
- Machinery: per-subcomponent freezing is new —
  `Components.freeze_subcomponents(frozen)` in core (grad hooks zero frozen columns
  of V / rows of U; non-persistent buffer that `_apply_ci_scaled_weight_decay`
  respects), `--freeze_alive_components` in `combine/finetune.py`. Test:
  `test_frozen_subcomponents.py`.
- Note: the frozen-alive weights are frozen, but the fresh CI fn is still free to
  mask them — "frozen" constrains weights, not masks.

### Analysis: AB heatmaps, subspace scatter, alive-threshold fix

- **kl-thr convention violation caught (user)**: I ran `find_alive_subcomponents` on
  the three fine-tuned combined runs with the default `--kl-thr=0.008`, but the
  convention is *the run's own observed rounded recon* (0.008 is just the reference
  run's 0.0074). Re-cut on CPU from the npz (no GPU rerun): both-02 thr 0.0239 →
  2270 alive, freshci-01 thr 0.0266 → 2270, complete-joint-01 thr 0.0228 → 2849
  (all previously 5634/7072). Heatmaps barely changed (the CI > 0.1 collection filter
  dominated) but the alive TSVs are now the real reference lists. Convention added
  to `scripts/validation/commands.md`.
- AB heatmaps for the three fine-tuned runs (add+sub, all positions) in each run's
  `analysis/ab_heatmaps_*`; facet labels now layer-qualified (`L16 gate_proj`) —
  `plot_ab_heatmaps` fix, since combined runs have 4 layers of same-named matrices.
- Subspace-scatter applet for both-02 (`analysis/subspace_scatter/index.html`,
  L18 MLP): chain needed two combined-run fixes — `collect_hidden_activations
  --layer=18` (its single-decomposed-MLP autodetect asserts on 4 layers) and an
  L18-only alive TSV for `collect_inner_activations` (the applet infers its layer
  from the alive list). 169 (add) / 82 (sub) alive-filtered MLP components.
- Report: hyperparameter-summary section; formal account of completeness training +
  mathematical pruning-threshold criterion (prune ⇔ ΔKL < τ = λ/w; redundant pairs;
  ε < τ < F′ repair window).

## 2026-07-15

### Setup / reconnaissance

- The four test runs have near-identical configs: same C per matrix (456 MLP, 72 q/k,
  128 v/o), same `global_shared_transformer` CI fn (d_model 512, 4 blocks, 8 heads,
  MLP 2048), same losses. Only differences: importance-minimality coeff (5e-5 for the
  L16/L17 "04" runs, 3e-5 for the "05" runs), steps (24k for L17, 20k for the rest),
  L18/L19 use `binomial` sampling — as do L16/L17 (checked: all four `binomial`).
  → hyperparameter combination is easy; per the sparsity rule-of-thumb we take the
  3e-5 coeff for fine-tuning (obj 2).
- Each run's CI fn is ONE transformer over the concat of its block's 7 matrix inputs
  (`ci_fn._global_ci_fn.*` in the checkpoint). Input projector / output head are sized
  to that block only ⇒ the trained CI fns cannot be merged into one bigger global CI
  fn without retraining, but they CAN coexist as separate per-block networks. That is
  exactly what the new `grouped_global` CI mode does.
- Checkpoints embed the entire frozen 8B target (16 GB each). Assembly mmap-loads and
  slices only `_components.*` + `ci_fn._global_ci_fn.*`, never materialising the
  target copies.
- No pre-existing combine/merge machinery anywhere in the repo (checked; layerwise.py
  only splits configs, never re-assembles).

### Decisions (with Antoine)

- `--ci-thr` default 0.01, matching `rounding_threshold` of the runs' own
  `TargetReconLoss` evals, so combined numbers are directly comparable to the
  end-of-training `eval/target_recon/rounded` logs. **This deviates from the
  roadmap, which specified default 0.1** — changed by explicit decision (Antoine,
  2026-07-15) for comparability with the training logs; 0.1 remains the aliveness
  threshold for L0 (`--ci_alive_thr`).
- The per-block CI-fn multiplexer is a small core addition (`param_decomp/ci_fns.py`),
  not a lab-side hack — objective 2 needs to train through it.
- Nontarget (FineWeb, delta=1) is evaluated everywhere, including objective 1.

### Obj 1 implementation notes

- Grouped wrapper verified equal to the original single-run global CI fn when loaded
  with the same weights (`test_grouped_ci_fn.py`).
- Singles are evaluated through the identical assembly path as the combined subject —
  if a single's numbers reproduce its end-of-training logs, the assembly + eval driver
  are trustworthy end-to-end.
- Eval batches: same seed re-applied per subject; PGD eval config copied from the
  runs' own eval block (20 steps, step_size 0.1, shared_across_batch).
- Cluster: 5/6 GPUs busy with my other runs → obj-1 eval runs on 1 GPU. Job 4719.

### Obj 1 — assembly validation (job 4719)

Singles evaluated through the assembly path reproduce their end-of-training logs
(differences consistent with different eval batches/seed):

| run | rounded (assembled) | rounded (training log) | L0 (assembled) | L0 (log) |
|---|---|---|---|---|
| L16-04-init-proj | 0.00721 | 0.00682 | 9.60 | 9.56 |
| L17-04-init-proj | 0.00620 | 0.00590 | 8.00 | 7.99 |
| L18-05-coupled | 0.00552 | 0.00521 | 12.61 | 12.51 |

→ the grouped-CI assembly + standalone eval driver are trustworthy end-to-end.

### Obj 1 results (job 4720; `~/out/combine/obj1_readily_combined.json`)

**The decompositions do NOT readily combine.** Rounding threshold 0.01, means over
10 batches × 128 prompts:

| subject | rounded | PGD | ci_masked | stochastic | ntgt rounded | total L0 |
|---|---|---|---|---|---|---|
| L16 single | 0.0072 | 0.0076 | 0.0072 | 0.0058 | 0.0034 | 9.6 |
| L17 single | 0.0062 | 0.0069 | 0.0061 | 0.0051 | 0.0032 | 8.0 |
| L18 single | 0.0055 | 0.0059 | 0.0055 | 0.0045 | 0.0038 | 12.6 |
| L19 single | 0.0059 | 0.0060 | 0.0059 | 0.0045 | 0.0032 | 7.9 |
| **combined (4)** | **0.257** | **0.403** | 0.252 | 0.238 | 0.0122 | 38.1 |

- Combined target recon is **~40× the worst single** and **~10× the sum of the four
  singles' losses** — strongly superadditive, not error accumulation alone.
- Nontarget degrades much less (0.003 → 0.012): off-distribution the delta carries
  the output, so component errors matter less.
- Per-block CI/L0 in the combined model is identical to the singles (e.g. L16:
  9.6009 vs 9.6053) — expected by construction, since CIs are computed from the
  *clean* target-model activations (`cache_type="input"` pass has no masking). The
  damage is entirely in the masked forward: each block's components+masks were
  trained with all other blocks intact, and replacing all four compounds the errors
  (downstream blocks see off-manifold inputs their masks were never tuned for).
- First OOM'd (job 4719): combined model + fp32 KL over 128×64×128k vocab logits on
  the nontarget eval at batch 128 exceeds 44 GiB. Nontarget eval batch now defaults
  to 64 (`--nontarget_batch_size`), per-batch values recorded for raw-data plots.
- Prefix-scaling eval submitted (job 4721: L16+L17, L16+L17+L18, and off-chain pair
  L18+L19) to see how the error grows with the number of replaced blocks.

**Implication for obj 2/3:** fine-tuning has real work to do (not just polish); the
target recon needs to come down ~40×, back to the ~0.006 range.

Prefix scaling (job 4721): rounded recon 0.0072 (L16) → 0.0437 (+L17) → 0.2154
(+L18) → 0.2567 (+L19); off-chain L18+L19 = 0.0252. Superadditive from the first
pair on; figures in `report_figures/obj1_{recon,scaling}.png`.

### Obj 2 memory probes

- 64/rank (target 16 tok) + 64/rank nontarget (64 tok): **OOM** during the nontarget
  train pass (job 4722; fp32 log_softmax over the 128k vocab; peak 43+ GiB).
  Training the 4-block assembly is heavier than the source runs (weight deltas for
  4 blocks + 4 CI fns + PPGD state).
- 32/rank probe submitted (job 4723). If it fits: dp=4 reproduces the sources'
  global batch of 128; dp=2 gives global 64 (feasibility-grade).
- Probe runs write a throwaway run dir (`p-*` under ~/out/runs; Trainer always
  checkpoints the final step) — delete after each probe.
- 32/rank probe (4723) **fits and completes** (peak ~46.8 GB by nvidia-smi, with the
  eval at 128/rank — real dp=2 runs eval at 64/rank, so slightly lighter). Its
  step-0 eval reproduces obj-1 exactly (rounded 0.2565, PGD 0.4036) → the
  finetune init path is verified.
- sbatch gotcha: a backgrounded nvidia-smi sampler without a `trap ... EXIT` keeps
  the SLURM job alive after a crash (zombie holding the GPU until walltime). Fixed
  with a trap; also bit job 4721/4722 (cancelled by hand).

### Obj 2 fine-tune runs

- First attempt dp=2 at 32/rank (`-01`, jobs 4725/4726) **OOM'd at step 1**: DDP
  gradient buckets + NCCL buffers add ~1.5 GB/rank over the single-process probe.
  Cancelled, run dir deleted.
- Relaunched single-GPU (world=1, batch 32 global — probe-proven), one GPU per
  variant so both run concurrently:
  - `combine-L16-19-both-02` (job 4727): train components + CI fns.
  - `combine-L16-19-frozenci-02` (job 4728): CI fns frozen (lr 1e-12).
  - Common: steps 2000, components_lr 1e-4, ci_fn_lr 5e-5 (cosine → 0.1×), impmin
    3e-5 constant at p=0.5 (end-state objective), full nontarget pass (ratio 2.0).
  - Caveat for the record: global batch 32 vs the sources' 128 — gradient noise is
    4× higher; feasibility-grade, not final-quality. A dp=4 (4×32) run reproducing
    batch 128 needs 4 free GPUs + the DDP overhead issue solved (e.g. batch 24/rank).
  - Step-0 eval of the `-01` attempt matched obj-1 exactly (rounded 0.2557,
    PGD 0.4033) before OOMing — init path confirmed under the Trainer too.
- `frozenci-02` (4728) OOM'd around its step-0 eval at ~43.8 GiB: single-process
  runs at batch 32 sit within ~0.5 GiB of the ceiling and live or die by
  fragmentation luck. The remaining fat was the target **eval** batch (ref 128 in
  one process; the sources only ever ran 64/rank). Added `--eval_batch_size` to
  finetune.py; relaunched as `combine-L16-19-frozenci-03` (job 4729) with eval 64.
  `both-02` (4727) cleared its step-0 eval and keeps running — evals are the peak,
  so it should stay under.
- Memory ledger vs sources (64/rank, 1 block, ~42.6 GiB): the 4-block assembly adds
  ~4.5 GiB static (4×weight deltas 1.75 + 3 extra CI fns + optimizer states), paid
  for by halving the train batch to 32 — but the eval batch had to shrink too.
- `frozenci-03` (4729, eval 64) STILL OOM'd — in the nontarget train pass
  (`recon_loss_kl` kl_div, the 1002 MiB fp32 logits tensor: 32×64×128256×4B). At
  batch 32/32 the config is coin-flip close to the ceiling (both-02 survived the
  identical allocation pattern). Real fix: nontarget train batch 16
  (`frozenci-04`, job 4730) — ~3 GiB off the failing pass. Config asymmetry vs
  both-02 (nontarget 32): only affects nontarget gradient noise; documented.

### Obj 2 trajectories (step 0 → 500 → 1000)

| run | rounded | PGD | target L0 | ntgt rounded | ntgt L0 |
|---|---|---|---|---|---|
| both-02 | 0.257 → 0.035 → **0.028** | 0.40 → 0.21 → **0.10** | 38 → 88 → 77 | 0.011 → 0.014 → 0.026 | 0.30 → 0.63 → 0.90 |
| frozenci-04 | 0.254 → 0.068 → 0.050 | 0.42 → 0.33 → 0.20 | 38.1 (pinned) | 0.012 → 0.011 → 0.012 | 0.32 → 0.27 → 0.45 |

Reading:

- **Fine-tuning works** — both variants recover most of the combination damage
  within 1000 steps at global batch 32; neither has plateaued.
- **Training the CI fns is much stronger on target recon** (0.028 vs 0.050 rounded,
  0.10 vs 0.20 PGD) but pays in two currencies: target L0 doubles (38→77 — CI fns
  open more components to fix recon) and the *targeting erodes* (nontarget rounded
  0.011→0.026, nontarget L0 0.30→0.90: components waking up off-distribution).
- **Frozen CI fns pin the masks** (CIs are computed from unmasked activations, which
  don't change → L0 exactly constant): components alone recover ~5×, targeting
  stays clean.
- Suggests a possible middle path if needed later: train both but with lower CI-fn
  LR and/or higher nontarget impmin ratio to hold the targeting.

### Obj 2 finals (step 2000)

| run | rounded | PGD | target L0 | ntgt rounded | ntgt L0 |
|---|---|---|---|---|---|
| both-02 | **0.0236** | **0.0584** | 67.8 | 0.0122 | 0.65 |
| frozenci-04 | 0.0437 | 0.1470 | 38.1 | ~0.012 | ~0.45 |

- both-02's nontarget erosion self-corrected (0.026 at step 1000 → 0.0122 at 2000)
  and its L0 re-sparsified from the step-500 peak (88 → 68).
- frozenci-04 plateaus (last 500 steps: 0.047 → 0.044).
- Neither reaches single-block quality (~0.006): 4× (both) / 7× (frozen) above.
  Caveats: global batch 32 vs sources' 128, only 2000 steps, LR already decayed.
  Trajectories suggest longer training closes more of the gap.
- **Verdict: feasible.** Training both components and CI fns is clearly the better
  variant (2× better rounded, 2.5× better PGD) at the cost of ~1.8× the L0 —
  consistent with the "prefer excellent PGD recon over low L0" rule.
- Final apples-to-apples eval (same script/seed as obj-1): job 4731 →
  `~/out/combine/obj2_finetuned_eval.json`.

### Code review pass (2026-07-15)

Ran /code-review (medium) focused on spec fidelity. Fixed: wrong nontarget-L0 column
in the obj-2 report table (singles are 0.07–0.11, not 0.27–0.35 — raw combination
already raises off-distribution L0 3–5×); the "end-of-training coeff" claim (L16/L17
converged at 5e-5, fine-tune uses min=3e-5 — now stated as a caveat); additive
expectation 0.0114/2.2× (was 0.0111/2.3×); per-panel figure legends (PGD only in
target panel, delta_only now plotted in both); `faithfulness_warmup_steps` pinned to
0 in combined configs (would otherwise replay warmup at lr 1e-3 on the loaded
assembly for sources that use it); data/nontarget-equality asserts across sources;
target-equality assert before evaluating finetuned subjects. Deferred (reported
only): CI_L0 regex-vs-fnmatch group matching (latent, needs a core fix),
eval_kwargs dataclass refactor, `_build_ctx` reuse of core `_build_metric_context`.

### Obj 4 design (completeness training)

Roadmap hypothesis: single-block decompositions each dropped their copy of
redundant cross-layer mechanisms (impmin pressure + the other blocks' intact copies
covering during training) → the combined model is missing ALL copies. Supporting
evidence already in hand: frozen-CI fine-tuning (masks pinned → cannot resurrect
anything) plateaus at 0.044, while CI-free fine-tuning reaches 0.024 at +30 L0 —
consistent with "something must be woken up".

Design decisions for stage 2 (per-block fine-tune):
- **Both the block's components and its CI fn train** — resurrection requires masks
  to open, which pinned CI fns cannot do.
- **"Rest of network = over-sparse" is realised by hard-freezing** the other blocks
  (requires_grad False on their components + CI fns) while keeping them as
  decomposition targets: the recon losses' masked forwards already replace all
  blocks, so block k trains against the other blocks' (frozen) reconstructions.
  Core fix required: `_apply_ci_scaled_weight_decay` now skips frozen components
  (it decays weights directly at the scheduled components-LR, outside the
  optimizer, and would otherwise shrink "frozen" blocks' dormant subcomponents
  ~3%/1000 steps).
- **Init from the over-sparse checkpoint** (`--init_from=combine-L16-19-frozenci-04`):
  components = over-sparse, CI fns = the sources' originals (frozenci never moved
  them).
- requires_grad freezing is single-process only (DDP's reducer would hang on
  post-hoc frozen params) — asserted.
- Validation built into the protocol: each per-block run's step-0 eval must
  reproduce frozenci-04's final eval (≈0.0437 rounded) since nothing has trained.
- Stage 3 will frankenstein-assemble the four completed blocks (each block's
  components + CI fn taken from its own per-block run) and evaluate; joint
  short fine-tune only if the franken assembly degrades.

### Obj 3 finals (standalone eval, `obj3_eval.json`)

| model | rounded | PGD | L0 | ntgt rounded | ntgt L0 | CI-fn params |
|---|---|---|---|---|---|---|
| both-02 (4 per-block CI fns) | 0.0239 | 0.0551 | 67.6 | 0.0133 | 0.59 | ~124M |
| obj3 fresh single CI fn | 0.0266 | **0.0484** | 97.9 | 0.0132 | 2.73 | ~90M |

A single from-scratch CI fn over all 28 matrices reaches comparable recon (slightly
worse rounded, best PGD of all variants) with a lighter CI stack, but is less sparse
(L0 98 vs 68, still falling at 2000 steps: 187→128→111→97) and notably less targeted
(ntgt L0 2.7 vs 0.6). Verdict: viable; needs longer training to catch up on
sparsity/targeting. Distillation not needed.

### Obj 4 stage 2 — all four blocks (steps 0 → 1000, others frozen at over-sparse)

| block | rounded | PGD | total L0 |
|---|---|---|---|
| L16 | 0.0426 → 0.0362 | 0.150 → 0.114 | 38.1 → 50.7 (+12.6) |
| L17 | 0.0426 → 0.0451 | 0.150 → 0.126 | 38.1 → 42.1 (+4.0) |
| L18 | 0.0426 → 0.0351 | 0.150 → 0.108 | 38.1 → 52.9 (+14.8) |
| L19 | 0.0426 → 0.0487 | 0.150 → 0.136 | 38.1 → 42.7 (+4.6) |

Every step-0 eval reproduced frozenci-04's final (0.0426/0.150/38.1) — init/freeze
machinery validated four times. Clear per-block heterogeneity: **L16 and L18 are the
resurrectors** (+13–15 L0 and real recon gains); L17/L19 wake little and their
rounded recon even drifts slightly up (they mostly re-tune PGD robustness). Also
faster than joint runs (~1.5 s/it vs 2.7: frozen blocks skip gradient work).

### Obj 4 stage 3 — franken assembly does NOT compose (obj4_franken_eval.json)

franken (each block from its own per-block run): rounded **0.0605**, PGD 0.135,
L0 72.8, ntgt rounded **0.0447** (vs over-sparse 0.0431 / 0.126 / 38.1 / 0.0132).
Worse than the over-sparse baseline it grew from, and 3.4× worse nontarget recon.
Naive additive expectation from the per-block gains was ≈0.037; interaction penalty
≈ +0.023 — the obj-1 superadditivity in miniature: each block was tuned against the
over-sparse *others*, and all four changed at once.

→ Reconciliation pass launched (`complete-joint-01`, job 4747): joint fine-tune
from the franken state with frozen CI fns (masks keep the resurrected components
alive; components re-align), 1000 steps. Success criterion: beat over-sparse
(0.0431) substantially at its ~73 L0, ideally approaching both-02 (0.0239 @ 68) —
which would show the completeness protocol buys recon that frozen-mask training
alone could not.

### Obj 4 stage 3 — reconciliation succeeds (2026-07-16, complete-joint-01)

Trajectory: rounded 0.0598 → 0.0235 (step 500) → 0.0229 (step 1000); PGD 0.163 →
0.070; L0 flat at ~72–74 (CI fns frozen ⇒ masks pinned); nontarget rounded healed
0.052 → 0.0148. Standalone eval (`obj4_joint_eval.json`): rounded **0.0228**, PGD
0.0598, L0 72.8, ntgt rounded 0.0163, ntgt L0 0.454.

**Success criterion met and exceeded**: frozen-mask training that plateaued at
0.0431 from the raw assembly reaches 0.0228 from the resurrected assembly — the
difference is attributable to the ~35 L0 of per-block-resurrected components. The
protocol matches joint "both" fine-tuning on recon (0.0228 vs 0.0239) at similar
L0 with better nontarget sparsity (0.45 vs 0.59), and every mask change happened in
an isolated, per-block phase. Figures regenerated with both obj-4 subjects.

### Obj 3 launched

- `combine-L16-19-obj3-freshci-01` (job 4732): components from the sources, ONE
  fresh `global_shared_transformer` CI fn over all 28 matrices (source arch:
  d_model 512, 4 blocks ⇒ ~90M params vs 4×31M for the per-block bundle — lighter,
  as the roadmap wants). CI fn LR 1.6e-4 (the sources' from-scratch value),
  components LR 1e-4, otherwise identical to obj-2 config.
- Risk to watch: at init the random CI fn produces ~0.5 masks everywhere; early
  gradients could damage the pretrained components before the CI fn organises.

### Obj 2 preparation

- `combine/finetune.py` written: combined config = union targets + grouped CI +
  end-state impmin (constant coeff 3e-5, p=0.5 — the sources anneal coeff ×2→×1 and
  p 2→0.5 across training, so the *end-state* is base coeff at p 0.5).
- Freezing (obj 2a) is `FROZEN_LR = 1e-12` on the CI-fn optimizer: schedules require
  start_val > 0, and `requires_grad_(False)` after DDP construction would stall the
  reducer. Adam moves params by ≈ lr per step regardless of grad scale → ~1e-8 total.
- GPU situation: my other runs hold 5/6 GPUs (L18-05-hid_sched + L20-05-coupled at
  2 ea., fa-L19-nodelta at 1, with dependent jobs queued behind them). Plan: 1-GPU
  memory probe (batch 64/rank ≈ one dp=2 rank) as soon as the eval job frees its GPU;
  real fine-tunes on 2 GPUs when a pair frees up.
