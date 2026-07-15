# Lab notebook — combine_layers

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
  end-of-training `eval/target_recon/rounded` logs. (0.1 in the roadmap text is the
  *aliveness* threshold used for L0.)
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
