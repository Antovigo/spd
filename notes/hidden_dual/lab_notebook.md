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


## 2026-07-30 — hidden-acts probes moved to the fast cadence

At `slow_every: 5000` the hidden-acts probes only produced 5 points per run, which is too
thin to see *when* the two CI nets diverge or to judge early whether the hidden-acts
reconstruction actually needs adversarial optimisation. Changed for future runs:

- `PGDHiddenActsReconLoss.slow` → `False`, and its config now mixes in a new
  `EvalCadenceConfig` (`slow: bool | None`), so cadence is a per-instance config choice.
  `Metric.is_slow` resolves class default vs override and the trainer gates on that.
- `CIHiddenActsReconLoss.slow` → `False` too. It was slow when it ran two *full* forwards;
  since it moved to `site_outputs` it costs one truncated forward per eval batch, which is
  ~6x cheaper than `CEandKLLosses`, and that has never been slow.

**The three launched runs are untouched.** Jobs 6076/6077 loaded their code at process start,
so editing the worktree cannot affect them. Job 6078 (ctrl) had *not* started, so it would
have picked up the new default and ended up on a different cadence from the dual runs,
breaking the matched comparison — it was held (`scontrol hold`), its config pinned with
`slow: true` on both hidden probes, then released.

Two facts worth recording, both verified rather than assumed:

- **Eval cannot perturb the decomposition.** The eval loop runs *after* backward and *before*
  the optimizer step, with `.grad` already populated, so a leaky probe would corrupt the
  update. It doesn't: `_run_pgd_loop` uses `torch.autograd.grad` w.r.t. the sources only and
  never `.backward()`, so gradients come out bitwise identical. Pinned by
  `TestEvalDoesNotPerturbTraining`.
- **But eval does advance the global RNG.** Stochastic masks and PGD source init draw from
  the global stream, so changing eval cadence changes the draws subsequent training steps
  see. Runs with different eval cadence are therefore *not* bit-comparable — a different
  sample path from the same distribution, not a different algorithm, and not a bias.

Cost of the fast PGD probe at `n_steps: 20` is roughly 5% wall clock (210 truncated forwards
+ 200 source backwards per eval, every 500 steps). `n_steps: 5` brings that to ~1.5% and is
the recommended setting for the frequent instance, with a 20-step `slow: true` sibling for
the definitive number.


## 2026-07-30 — persistent adversary for the hidden objective too

`addsub-L18-09-dual` finished cleanly (20001 steps, 16.9 h, peak 39717 MiB), freeing 2 GPUs.
Launched `addsub-L18-09-dual-ppgd` (job 6105) — identical to it in every respect except that
the hidden-acts objective now also gets a persistent adversary, with its own sources.

Motivation: the two CI nets were not under equal masking pressure. The output net faced
stochastic masks *and* a persistent adversary sharpening across every prior step; the hidden
net faced only stochastic masks. A denser hidden net could then have been an artefact of the
weaker pressure rather than a property of hidden-activation importance. This run removes that
confound, so `dual` vs `dual-ppgd` isolates exactly it.

Loss composition (as agreed): hidden = Stochastic 1.0 + PPGD 0.5; output = Stochastic 1.0 +
Unmasked 0.5 + PPGD 0.5. Both PPGD instances share the output run's hyperparameters (Adam
lr 0.01, betas 0.5/0.99, `n_warmup_steps: 2`, `n_samples: 1`, `per_batch_per_position`). No
PPGD on the nontarget pass. Evals unchanged.

Implementation: `PersistentPGDState` no longer holds a reconstruction loss — it takes a
`PGDObjective` per call, the same seam already used for per-step PGD, so one resumable state
machine serves both objectives. Separate sources fall out of the metric-instance boundary
(lazy per-instance state + snapshot keyed by `instance_key`) rather than needing special
handling; pinned by a test asserting the tensors are distinct, diverge once stepped, and
round-trip under separate keys.

Migrating `test_spd_losses.py` to the new API needed doing twice: the first regex stripped
`reconstruction_loss=` from unrelated helper calls, so the file was reverted and the
substitution re-scoped to the state helper by name. Worth remembering that a bare
`reconstruction_loss=recon_loss_mse,` pattern appears in several unrelated call sites.

Step-0 sanity: each PPGD loss sits above its stochastic counterpart, as an adversarial mask
must — output 0.173 vs 0.107, hidden 0.514 vs 0.452. Memory 39659 MiB, statistically
identical to the plain dual run's 39657 (the extra sources are ~22 MB, the extra graphs
transient). Rate 3.19 s/step → ~17.7 h projected, inside the 24 h cap in one leg.

## Exchange-rate probe (job 6125, CPU)

Asked how to balance the impmin coefficient between a KL and an MSE objective. First design
swept a threshold `tau` on CI values and re-measured both objectives — wrong: under
`leaky_hard` the CI is ~98% saturated at 0 and ~2% at 1, with only 0.11% in (0.01, 0.5), so
the sweep would have been flat until everything died at once. Caught before it ran. Redesigned
around ablating components (ranked by mean hidden CI, which under saturation *is* firing rate,
which is what the impmin penalty is proportional to) and (component, position) entries.

Ran CPU-only: all 6 GPUs were on the two training runs, and a no-gres job still *sees* the
cards, so `CUDA_VISIBLE_DEVICES=""` is set in both the sbatch and the script, plus a
`torch.cuda.is_available()` assert. fp32 throughout — the KL differences are ~1e-3 nats and
bf16 logits would swamp them. Target built directly in fp32 (`spec.model_copy(update=...)`)
rather than loading bf16 and converting, to avoid holding both copies. First submission at
150G bounced on `QOSMaxMemoryPerUser`; 96G ran. Prompts tokenize to 5 tokens, not the
config's `max_seq_len: 16`, so forwards were much cheaper than budgeted — 256 prompts, ~35 min
for ~40 forwards.

Two interpretation errors, both caught by the data:

Extrapolated kappa from the first single 25-component step (0.183) and quoted a 3.6x
coefficient change; the cumulative slope settled at 0.365 over 500 components, giving 1.9x.
Single-segment slopes ranged 0.18-0.69 before converging.

Claimed the per-site KLs summing to more than the joint total showed destructive interference
between site residuals. It doesn't. The all-sites injection overrides every site output, so
gate/up reach the logits only through the clamped `down_proj` and q/k/v only through the
clamped `o_proj` — jointly they contribute nothing, and their effect is already folded into
the write sites' residuals. `down_proj` alone gives 97% of the joint KL. A causal bottleneck,
not interference. This also flipped the `site_patterns` advice.

Headline: kappa spans 200x by direction (0.0015 for the hidden-only surplus, 0.35 for
marginal shared components), so no unit conversion balances the two losses. What does set the
coefficient is that the *value* distribution is bimodal in the same place — bulk surplus ~200x
above its keep-threshold, marginal fringe at ~2x — making `lambda_hidden: 5e-5 -> 1e-4`
surgical. Full numbers and cross-checks in `report.md`.

## 2026-08-01 — which hidden activations? the addsub-L18-11-* site-target series

Asked which *part* of the hidden activations the second CI net should reconstruct. Four
arms, everything but the measured site set held identical to `addsub-L18-10-dual-ppgd`:

| arm | hidden objective measured at |
|---|---|
| `addsub-L18-11-baseline` | all 7 decomposed matrices (the status quo) |
| `addsub-L18-11-module-out` | `o_proj` + `down_proj` — what the modules add to the stream |
| `addsub-L18-11-resid` | the residual stream itself, post-attn and post-MLP |
| `addsub-L18-11-down-only` | `down_proj` alone |

15000 steps, gamma annealed over the last 5000 (`gamma_anneal_start_frac` 2/3). Two lanes
of 2 GPUs, chained `afterok` so at most two run at once.

### Readout sites

The stream is not any matrix's output, so measuring there needed a new concept:
`pd.hidden_readout_sites`, a `{name: module_path}` map whose module *input* is captured —
clean and masked — and joined to the decomposed sites in `ComponentModel.measurement_sites`.
`site_patterns` then selects it like anything else, so all four hidden-acts metrics gained
residual support with no per-metric code. In a Llama block the two capture points are
`layers.18.post_attention_layernorm` (post-attention stream) and `layers.19.input_layernorm`
(post-MLP stream).

A readout is measured at **every position**, unlike a decomposed site, which is restricted
to the positions its routing mask selects. That restriction is sound at a matrix output — an
unrouted position ran the frozen module and its error is identically zero — but false on the
stream: attention mixes positions, so a position routed to nothing still receives error from
the routed positions it attends to.

**All four arms declare the same two readout sites** and differ only in the *training*
losses' `site_patterns`. Every arm therefore logs the same eval panel — hidden error at all
7 matrices and at both stream points, under both CI nets — which is what makes questions
like "does training on `down_proj` alone also fix the stream?" answerable.

### One bug, found by the probe

The global CI-fn wrapper transforms *every* key of the activation cache rather than indexing
by its own layer list, so the extra readout entries crashed it with
`KeyError: 'resid_post_attn'`. `calc_causal_importances` now selects the decomposition
targets explicitly. Pinned by a test; the assumption that "extra keys are inert" was checked
against `get_all_component_acts` (which does skip unknown keys) and wrongly generalised.

### The residual objective needed recalibrating — by 2830x

Measured at step 0, the same CI mask gives hidden error **0.942** at the matrices and
**0.000333** at the stream: a 2830x ratio, because the stream's `Σ tgt²` denominator is
dominated by the frozen incoming residual. Left at `coeff: 1.0` the resid arm's objective
would sit ~2830x below the sparsity penalty it competes with, and the run would have
measured "hidden objective switched off" rather than "the stream is a worse target".

The arm's coefficients are therefore scaled by 2830 (stochastic 1.0 -> 2830, PPGD 0.5 ->
1415), equalising the two objectives at step 0. The reported quantity stays the true
relative error; only the arm's weight changes, so the variable under test is *which*
activations rather than *how strongly*. Note the calibration is probe-dependent: the
CI-masked ratio is 2830, the stochastic-masked one ~2100. The 35% gap is immaterial against
the correction itself, but the number is not a constant of nature.

### C raised 4x — measured, not guessed

The -10 baseline was at its alive-count ceiling (hidden net: `q_proj` and `k_proj` at exactly
128/128), which would have clipped the very readout the series is about. Probes at 10 steps:

| C factor | total C | peak / GPU (of 46068) | headroom |
|---|---|---|---|
| 1x | 1536 | 39002 | 7.0 GB |
| 4x | 6144 | 41546 | 4.4 GB |

4x the components costs 2.5 GB — C really is a weak memory lever here, as the -09 series
found. Stopped at 4x rather than 6x: extrapolation puts 6x near 43.2 GB, the same knife-edge
that OOM'd earlier 8B runs on this node's smaller cards, and a mid-run OOM costs 13 h.

## 2026-08-01 — fifth arm: `mlp-only`, and the wandb rename

Added `addsub-L18-11-mlp-only` (hidden objective on `*.mlp.*` — gate, up, down; no
attention) as a third concurrent lane, GPU budget raised to 6. Config diffs from
`baseline` in exactly three lines: the label and the two training losses' `site_patterns`.

It exists because of the baseline result. The hidden net's per-position surplus over the
output net is ~4.4x in attention but only ~1.6x in the MLP, and the anomaly census puts
essentially zero magenta in `k_proj`/`v_proj` — so the attention surplus is real activity
that the output objective cannot see. `baseline` vs `mlp-only` differ in exactly the four
attention-internal sites and therefore price that surplus against output quality directly.

The first two arms were launched without `--run_id`, so their run dirs and wandb runs took
auto-generated `p-*` names. Local dirs renamed to their labels; wandb display names renamed
from each run's own `config.label` (asserting rather than guessing when absent). Wandb run
*ids* are immutable, so those two keep `p-*` in their URLs. Later arms pass `--run_id`.

## 2026-08-02 — phase 2 plan, fixed before the results are in

Once the five arms land: pick the best-performing locus, return to the 20000-step formula,
and re-test it there under **increased** hidden-acts reconstruction pressure — two
different increases, 2 GPUs each, 4 total.

Writing the selection rule down now, before seeing arms 3-5, so it cannot be fitted to
whichever answer arrives. The stated criterion is *best output decomposition*, so the
primary key is output quality, and the tie-breaks are the readouts the series exists to
produce:

1. **Output quality at matched sparsity.** `kl_ci_masked` and `PGDReconLoss` read against
   `CI_L0_output`, not raw. The arms will not land at equal sparsity — the hidden loss is a
   mean over sites, so narrowing the site set multiplies the per-site gradient weight
   (7 sites -> 2 sites is 3.5x) and buys more components. Comparing raw KL across arms would
   reward whichever arm happened to end least sparse.
2. **Hidden reconstruction where it was never trained.** Each arm logs the full panel, so
   generalisation off the trained sites is directly visible. An arm that repairs sites it
   never measured is doing something structural rather than fitting its own objective.
3. **Anomaly rate** (magenta cells, cut 0.5) — cell counts only, which are cut-robust;
   component counts are an ordering, not a magnitude.
4. **Saturation** — any arm near the ceiling is disqualified from the comparison rather
   than ranked, since its counts are clipped.

Phase-2 shape: 20000 steps, `gamma_anneal_start_frac` 0.5 (annealed over the last 10000, as
in `-10-dual-ppgd`), C kept at 4x — it is strictly better at matched step count and is what
lifted the alive-count ceiling, so reverting it would reintroduce the clip. Only the
schedule returns to the 20k formula.

Pressure: the winning arm's hidden recon coefficients (stochastic 1.0 / PPGD 0.5 at
baseline weighting) scaled by two multipliers. These multiply whatever the arm's own
calibration already is — for `resid` that is on top of the 2830x, which is a unit
conversion rather than a pressure choice.

## 2026-08-02 — selection metric fixed: PGD nats per active component

The arm is chosen by **PGD reconstruction nats per active component** —
`PGDReconLoss / CI_L0_output`, lower is better. This supersedes the matched-sparsity
reading-off written above; it is the same idea reduced to one number, which removes the
judgement call about what "matched" means when no two arms land at the same sparsity.

Two definitional forks, both reported because they can in principle disagree:

- **Literal**: adversarial recon error carried per active component. An arm with more error
  *and* many more components can score well here, which is the failure mode to watch.
- **Rate-distortion**: nats of KL recovered against the fully-ablated reference
  (`kl_zero_masked` = 0.24215) per active component, higher is better. Immune to that
  failure mode.

Denominator reported both as `CI_L0_output` (mean active per position) and `n_alive_output`
(distinct components alive). The output net's count is the denominator because
`PGDReconLoss` is the output objective under an output-CI mask; the hidden net's components
are the *cost* being tested, and enter through their effect on the output net.

First two arms — all four framings agree, so the ambiguity is not load-bearing so far:

| arm | PGDRecon | CI_L0 out | alive out | nats/active | recovered/active |
|---|---|---|---|---|---|
| baseline | 0.00547 | 25.4 | 1229 | **2.151e-04** | **0.0093** |
| resid | 0.00692 | 28.2 | 1421 | 2.452e-04 | 0.0083 |

`resid` is 14% worse. If a later arm splits the two framings, that gets reported rather
than resolved silently.

### Correction: denominator is the union over both nets

The metric is **output-PGD nats per total alive component**, where "total alive" counts
components alive under *either* CI net. The two nets score the **same** shared pool of
subcomponents, so the sum of the two logged `NAlive` values double-counts every component
both nets keep — which is most of them. It has to be a union.

Neither logged count gives the union, since `NAlive` reduces to a scalar per net. The
overlap is recoverable from `ab_grids`, which stores per-component mean CI for both roles
over all C, so the union is computed there: a component counts as alive if its
per-position mean CI reaches 0.1 under either net.

That definition is stricter than the `NAlive` eval metric — mean over the prompt pool
rather than max over individual examples — so these counts run lower than the logged ones.
They are computed identically for every arm, which is what the ranking needs, and they are
the only place the per-net overlap exists at all. The `NAlive` figures stay in the report
as the headline absolute counts; the union is what the selection metric divides by.

## 2026-08-02 — metric correction: the ranking was inverted

`nats / alive` is wrong-signed. Both quantities are costs — we want few nats *and* few
components — so dividing one by the other credits an arm for keeping more components. Under
that ratio the densest arm on the Pareto frontier came out on top; the correct combination
is the **product**, `PGDReconLoss * alive-either`.

Corrected ranking:

| arm | PGD | alive either | nats x alive | Pareto |
|---|---|---|---|---|
| **mlp-only** | 0.00596 | **298** | **1.7756** | optimal |
| down-only | 0.00627 | 315 | 1.9735 | dominated by mlp-only |
| baseline | **0.00547** | 374 | 2.0456 | optimal |
| module-out | 0.00602 | 406 | 2.4429 | dominated by baseline, mlp-only |
| resid | 0.00692 | 367 | 2.5394 | dominated by down-only, mlp-only |

The frontier is exactly `{mlp-only, baseline}`. `mlp-only` trades 9% more adversarial error
for 20% fewer components and wins the product by 13%. `module-out` and `resid` are beaten on
both axes at once, which no exchange rate rescues.

This inverts the earlier conclusion. **Dropping attention from the hidden objective is the
best move**, not the worst — the attention surplus the baseline analysis identified (4.4x
the output net's per-position activity) is real but expensive, and buying it back costs a
fifth of the component budget for ~9% output fidelity.

Domination is now reported alongside the product, because the product fixes one particular
exchange rate between two costs while domination is exchange-rate-free — and a near-tie in
any scalar score should be read against the two factors separately.

Phase 2 relaunched on `mlp-only` (jobs 6641 / 6642); the earlier pair based on `baseline`
was cancelled under 2 minutes in.

## 2026-08-02 — phase 2, third run: the all-matrices locus at highest pressure

Added `addsub-L18-12-press10-allmat` — the `baseline` locus (all 7 matrix outputs) at the
10x hidden-recon coefficient. Three runs, 6 GPUs.

| run | locus | hidden coeff multiplier |
|---|---|---|
| `addsub-L18-12-press3` | `mlp-only` (`*.mlp.*`) | 3x |
| `addsub-L18-12-press10` | `mlp-only` (`*.mlp.*`) | 10x |
| `addsub-L18-12-press10-allmat` | `baseline` (all 7 matrices) | 10x |

Its config differs from the running `press10` in exactly two places, the label and
`site_patterns`, so `press10` vs `press10-allmat` is a clean single-variable pair at the
pressure where any locus effect should be largest. That matters because the phase-1 loci were
separated by only 13% on `nats x alive`, and both survivors were on the Pareto frontier —
the ordering could plausibly change once the hidden objective is weighted 10x, and this pair
detects that directly rather than by extrapolation.

It also probes the phase-1 mechanism from the other side. Per-site gradient weight is
`1/n_sites`, so the `mlp-only` arm already gives each MLP site 1/3 against `baseline`'s 1/7 —
a 2.3x concentration. A 10x global multiplier on `baseline` overshoots that concentration
while keeping attention in the objective, which separates "concentrate the pressure" from
"increase the pressure".

## 2026-08-02 — the `-11` name reused: C scaling and hidden pressure

Cancelled the site-target work (the running `-12` pressure runs) and purged the `-11`
series: 407 GB across five completed `-11` arms and two partial `-12` runs. Before deleting,
copied every run's `metrics.jsonl`, `experiment_config.yaml` and `run_metadata.json` into
`~/pd_scratch/hidden_site_targets/archive/` (8.9 MB total), so every number in
`site_targets_report.md` stays checkable. Checkpoints and `ab_grids` payloads are gone, so
the anomaly census cannot be recomputed at a different threshold — that is the one
irreversible loss. The wandb runs were left in place; say the word to delete those too.

The new `-11` is three exact replications of `addsub-L18-10-dual-ppgd` at 4x C:

| run | C | hidden recon coeffs |
|---|---|---|
| `addsub-L18-11-bigc` | 6144 | 1.0 / 0.5 (unchanged) |
| `addsub-L18-11-press2` | 6144 | 2.0 / 1.0 |
| `addsub-L18-11-press5` | 6144 | 5.0 / 2.5 |

Generated from the reference's own YAML rather than from any descendant, so exactness is by
construction rather than by inspection. Diffs confirm it: `bigc` differs from the reference
in the label and seven C values only; each pressure run differs from `bigc` in the label and
two coefficients only. Output-side hyperparameters — the three output recon losses and both
importance-minimality coefficients — are byte-identical throughout, which is what makes the
comparison a pressure comparison rather than a rebalancing.

Deliberately **no `hidden_readout_sites`** here. They would not have hurt, but with
`site_patterns: null` the hidden objective is "every measurement site", so declaring readouts
would have widened it from 7 matrices to 9 and quietly broken the replication.

Reference schedule already anneals gamma over the second half (`start_frac: 0.5`, 20000
steps), so nothing there changed.
