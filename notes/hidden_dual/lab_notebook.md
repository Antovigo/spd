# Dual hidden-acts CI — lab notebook

Newest entries at the bottom. Spec in `plan.md`. Runs under `~/out/runs/`; ad-hoc configs
and sbatch under `~/pd_scratch/hidden_dual/`.

## 2026-07-29 — implementation

Branch `feature/dual_hidden_acts` off `experiment/8B_targeted`, worktree
`~/Code/param-decomp/dual_hidden_acts`. Three separable commits so the scheme can be
replayed onto other branches:

| commit | contents |
|---|---|
| `4bbccc335` | core: `CIRole`, `ci_fn_hidden`, `site_outputs` early exit, `hidden_acts.py`, `StochasticHiddenReconSubsetLoss`, `PGDHiddenActsReconLoss`, `NamedMetricConfig`, trainer wiring |
| `568252262` | lab: `ci_role` on `CIHiddenActsReconLoss`, ab_grids dual payload + green/magenta applet |
| `4eec1168a` | docs: CLAUDE.md pointers + `plan.md` |

Commit boundaries are core / lab / docs rather than the finer split in `plan.md`: pre-commit
type-checks the whole tree, so the `MetricContext.ci_hidden` field and the test call sites
that must pass it cannot land in separate commits, and `configs.py` carries both the
`dual_hidden_ci` field and the new loss's union entry.

### Decisions taken during implementation, beyond the plan

- **Early-exit condition is cache size, not execution order.** `site_outputs` aborts when
  `len(cache) == len(mask_infos)`. Determining "the last decomposed module" would need the
  model's execution order, which config order does not give (the L18 config lists gate/up/down
  before q/k/v/o, but Llama runs attention first). Cache size needs no such knowledge.
- **`CIHiddenActsReconLoss` switched from raw per-module MSE to the same relative error** as
  the loss and the PGD probe, and now uses the truncated forward instead of a full clean +
  full masked pair. All three hidden-acts numbers are then directly comparable, which is the
  whole point of running dual against ctrl. Cost: its logged values are **not** comparable to
  those from earlier runs. Raw MSE is still available via the legacy
  `StochasticHiddenActsReconLoss`.
- **`pgd_masked_objective_update`** added to `pgd_utils.py` so the hidden PGD probe reuses the
  PGD driver instead of importing private helpers; `pgd_masked_recon_loss_update` is now a
  thin wrapper over it and `_forward_with_adv_sources` folded away.
- **fp32 before subtracting** in `site_squared_errors`. Under bf16 autocast the prediction and
  target are close and large, so a bf16 difference discards most significant bits.
- **Merge colours are subtractive on white**, not additive on black: white = neither,
  green = hidden-only (expected), magenta = output-only (the anomaly), black = both. Keeps
  "inactive = white" consistent with every other tile in the applet, which matters when
  scanning a gallery of hundreds.
- **Two latent bugs fixed in passing**, both of which the dual scheme would have triggered:
  the nontarget loss log key and the nontarget eval dedup assert were keyed by metric *class
  name*, so with one importance-minimality instance per CI net the second would have silently
  overwritten the first / falsely tripped the assert.

### Verification

- `498 passed` on the pre-existing suite (`-m "not slow"`), plus 12 new tests in
  `param_decomp/tests/test_dual_hidden_ci.py`. basedpyright: 0 errors across the tree.
- The new tests pin the parts that could silently be wrong: `site_outputs` matches the full
  forward's output cache tensor-for-tensor; the forward really does abort early (a hook on
  the target model's root never fires); the cached tensors keep their autograd graph;
  relative error is exactly 0 when components+delta reproduce `W` and exactly 1 when a site
  is fully ablated; `clean_site_outputs` reproduces what the frozen model itself computes.
- All three run configs validate through `LMExperimentConfig`, and the derived nontarget loss
  sets are as specified: both impmins at 1e-4 (2x ratio), both recon losses kept,
  `UnmaskedRecon` and PPGD dropped.

### Per-step cost of the dual scheme

Marginal over the `addsub-L18-09-one-im` recipe: two CI-net forwards (~34 M params each,
negligible against an 8 B target), one truncated masked forward per pass, and one extra CI
net of optimizer state (~0.55 GB). The truncation is what keeps the extra graph to one
block's internals instead of the whole tail of the model plus `lm_head`.

### Memory probes

Submitted `probe-L18-dual` (2 GPUs, jobid 6043) and `probe-L18to20-dual` (3 GPUs, jobid
6044): `steps: 3`, no wandb, no checkpoints, so step-0 slow eval (ABGridDataset + the two
20-step PGD probes) is included in the peak. `run_ddp_dual.sbatch` samples per-GPU memory
every 3 s and prints the peak.

L18to20 starting point: batch held at 128/96, C shrunk to 304 MLP / 48,48,88,88 attn — above
the 6L sizing (228) that fit 6 blocks on 4 GPUs at batch 48, below the L18 sizing (456).

Probe results (peak per-GPU, cards 46068 MiB) and what each one taught:

| config | GPUs | batch/nt | C | peak | verdict |
|---|---|---|---|---|---|
| L18 dual | 2 | 128/128 | 456 | 45641 | 427 MiB headroom — rejected |
| L18 dual | 2 | 128/96 | 456 | 39657 | ✓ launched |
| L18to20 dual | 3 | 126/96 | 304 | 46253 | over the 45 GB cards; only fit a 48 GB one |
| L18to20 dual | 3 | 126/96 | 228 | 45383 | 685 MiB — too tight |
| L18to20 dual | 4 | 128/96 | 304 | 42204 | ✓ launched |
| L18to20 ctrl | 4 | 128/96 | 304 | 37821 | ✓ queued |

**C is a weak memory lever**: 304 → 228 (the floor) bought under 1 GB, because the
weight-delta tensors dominate and are full-weight-shaped independent of C (~2.7 GB for 3
blocks). Per-rank batch and GPU count are the real levers. Hence 4 GPUs for the 3-block runs
rather than a smaller C — 4 GPUs at C=304 beats 3 GPUs at C=228 on every axis (2.7 GB more
headroom, a third more components, clean divisibility), and the user pre-approved 4 GPUs for
this case.

Three non-memory constraints surfaced during probing:

- **Every batch size must divide the DDP world size.** 128 is not divisible by 3, which is a
  second reason the 3-block runs went to 4 GPUs.
- **`eval.batch_size` must equal `pd.batch_size`.** `PersistentPGDReconLoss` sizes its
  persistent adversarial sources from the train batch and is auto-evaluated, so a different
  eval batch trips `source leading dim 42 must divide batch dim 21`. Both reference configs
  happen to set them equal, which is why this had never surfaced.
- **QOS caps a job at 24 h** (48 h and 72 h are rejected; `--test-only` does *not* enforce
  this, so it silently accepts 48 h). The single-block run should finish in one leg at ~16.5 h;
  the 3-block runs will likely need a resume leg, which the SIGTERM checkpoint handles.

## 2026-07-29 — code review, then launch

Review (single agent, adversarial, against `plan.md`) found **no defect that computes a wrong
number**, and verified the load-bearing claims by direct experiment on a real
`LlamaForCausalLM` rather than by reading: the early exit really does skip layers after the
last site and `lm_head`; cached tensors match a full forward bit-for-bit and keep their graph;
the PGD stash holds the final sources' values and the eval `no_grad` does not break the inner
ascent (error grows monotonically with `n_steps`); snapshot round-trips all optimizer state
for both nets.

Its one important finding was a **spec gap, not a bug**: every CI-density eval metric read
`ctx.ci` unconditionally, so `plan.md`'s own primary step-5000 check ("`n_alive` on the hidden
net") was unmeasurable — the whole dashboard would have shown output-net numbers. Fixed by
adding `ci_role` to `NAlive`, `CI_L0`, `CIMeanPerComponent`, `CIHistograms` plus
`Metric.key_prefix` so two instances of a dict-returning metric can coexist without colliding.
Also took from review: `sub_` in `site_squared_errors` (one fewer fp32 buffer live, ~0.4 GiB
of nontarget peak), an assert for the single-hook-fire invariant the early exit rests on, and
an assert that the two nets agree on module keys.

Deliberately not taken: removing `ComponentModel.__init__`'s `dual_hidden_ci=False` default
(27 test call sites, no functional gain), and the `measure_site_errors` helper collapsing the
four-line measure sequence in three consumers (worth doing if a fourth probe appears).

Launched with `run_ddp_dual.sbatch` (worktree-pointing copy of `run_ddp.sbatch`, 24 h):
jobs 6076 (L18 dual, 2 GPU), 6077 (L18to20 dual, 4 GPU), 6078 (L18to20 ctrl, 4 GPU, queued
behind the 6-GPU per-user cap — it starts automatically).

Step-0 verification on 6076 confirmed every piece live: both impmin instances under distinct
keys (identical values, as expected — `zero_init_readout` starts both nets at logit 0.5), both
recon losses on both passes, all four hidden-acts probes with per-site breakdowns, both nets'
`n_alive`, and `ab_grids/step_0.js` carrying `"ci_roles": ["output", "hidden"]`. Probe ordering
is as it should be: adversarial 1.81 > CI-masked 0.95 > stochastic 0.47.
