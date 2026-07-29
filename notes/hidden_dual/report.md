# Dual CI networks: output-importance vs hidden-activation-importance

Status as of 2026-07-29: **implemented, verified, three runs launched.** No scientific
results yet — first substantive snapshot at step 5000. Spec in `plan.md`, chronology in
`lab_notebook.md`.

## What was built

Two CI networks over one shared pool of subcomponents, differing only in the reconstruction
loss that trains them: one scores importance for the model's **final output**, the other for
the **decomposed sites' activations**. Enabled by `pd.dual_hidden_ci`; metrics select a net
with `ci_role`, defaulting to `"output"` so every pre-existing config is untouched.

The falsifiable claim the setup exists to test: **output-importance should imply
hidden-importance, but not the reverse.** A component that matters for the logits must
matter for the activations that produce them; a component that only carries interference
cancelled before the output should matter for the activations alone.

## The measurement

All three hidden-acts probes report the *same* quantity — the relative per-site error
`Σ(out − tgt)² / Σ tgt²`, averaged over sites — under three different masks, so they are
directly comparable:

| probe | mask | delta |
|---|---|---|
| `StochasticHiddenReconSubsetLoss` (training loss) | stochastic subset ablation | random |
| `CIHiddenActsReconLoss` (eval) | CI itself | off |
| `PGDHiddenActsReconLoss` (eval) | 20 steps of sign-PGD, adversarial | random |

Relative rather than raw MSE because site activation scales differ by orders of magnitude
(MLP `down_proj` vs attention `q_proj`), so raw MSE would silently weight the objective by
activation variance and would not transfer across blocks. Numerator and denominator are
accumulated and DDP-reduced separately: the reported number is a ratio of sums over the whole
eval pass, never a mean of per-batch or per-rank ratios.

Targets are the frozen model's own site outputs, recomputed as `F.linear(x_clean, W, b)` from
the clean activations each step already caches. This costs **no extra forward pass** and
measures accumulated drift from the target model rather than each site's local error given an
already-perturbed input — the latter would be blind to exactly the chained-block failure the
experiment is about.

## Evidence that it works

From `addsub-L18-09-dual` at step 0 (job 6076), all four hidden-acts probes plus both nets'
density diagnostics reporting:

```
eval/loss/CIHiddenActsRecon_hiddenCI            0.952     (CI mask, no delta)
eval/loss/StochasticHiddenReconSubsetLoss       0.466     (stochastic mask, random delta)
eval/loss/PGDHiddenActsReconLoss                1.809     (20-step adversarial)
eval/n_alive/NAlive_output/total                1768      (= total C, as expected at init)
eval/n_alive/NAlive_hidden/total                1768
train/loss/SmoothL0ImpMin_output             3537.99
train/loss/SmoothL0ImpMin_hidden             3537.99      (identical at init — see below)
train/nontarget/loss/StochasticReconSubsetLoss        0.00183
train/nontarget/loss/StochasticHiddenReconSubsetLoss  0.01336
```

Four things worth noting in that table:

- **The probe ordering is right**: adversarial (1.81) > CI-masked (0.95) > stochastic (0.47).
  The adversary must find more error than any sampled mask, and the stochastic pass is lower
  because it admits the delta component at random strength while the CI-masked probe has no
  delta at all.
- **Both impmin instances log separately and are identical at step 0.** Identical is the
  correct value, not a bug: `zero_init_readout` starts both nets' readouts at logit exactly
  0.5, so the two nets emit the same CI until gradients separate them. That they appear under
  *distinct keys* is the fix for a real defect (below).
- **Both nets' `n_alive` is visible.** Without this the hidden net's density — the primary
  step-5000 diagnostic — would have been unobservable.
- **The nontarget pass carries both recon losses.** With the delta forced on, the hidden loss
  drives components toward being inactive at each *site* on the broad distribution, a more
  local version of the pressure the output recon applies.

`ab_grids/step_0.js` carries `"ci_roles": ["output", "hidden"]`, so the two-colour heatmaps
are populated from the first snapshot.

### Two latent defects found and fixed

Both were pre-existing and both would have been triggered by this scheme, silently:

1. The **nontarget loss log key** and the **nontarget eval duplicate-detection assert** were
   keyed by metric *class name*. With one importance-minimality instance per CI net, the
   second would have overwritten the first in the logs / falsely tripped the assert.
2. Every CI-density eval metric (`NAlive`, `CI_L0`, `CIMeanPerComponent`, `CIHistograms`)
   read the output net unconditionally, so on a dual run all of them would have silently
   reported output-net numbers. They now take `ci_role` and namespace their keys via
   `Metric.key_prefix`, which engages only for explicitly-named instances so single-CI runs'
   log keys are unchanged.

A code review also confirmed by direct experiment on a real `LlamaForCausalLM` that the
early-exit forward skips the tail (layers after the last decomposed site and `lm_head` never
execute), that its cached tensors match a full forward's bit-for-bit and keep their autograd
graph, and that the PGD probe's inner ascent is not broken by the eval loop's `no_grad`
(errors grow monotonically with `n_steps`: 0 → 2.70, 5 → 4.18, 20 → 4.44).

## Cost

The early-exit forward (`ComponentModel.site_outputs`) aborts once every decomposed site is
cached, so nothing past the last site is computed *or* retained for backward. It cannot skip
the *prefix*, though: for an L18-only decomposition it still runs embeddings + layers 0–17,
so it costs ≈19/32 of a forward, not a sliver. Autograd retains nothing from those frozen
layers, so the memory claim holds even where the compute framing is optimistic.

Marginal cost of the scheme per step: two CI-net forwards (~34 M params each, negligible
against 8 B), one truncated masked forward per pass, one extra CI net of optimizer state
(~0.55 GB).

## Measured memory (peak per-GPU, cards are 46068 MiB)

| config | GPUs | batch / nontarget | C (MLP) | peak | headroom |
|---|---|---|---|---|---|
| `addsub-L18-09-dual` | 2 | 128 / 128 | 456 | 45641 | **427 MiB — rejected** |
| `addsub-L18-09-dual` | 2 | 128 / 96 | 456 | 39657 | 6.4 GB ✓ |
| `L18to20` dual | 3 | 126 / 96 | 304 | 46253 | **overflowed — only fit a 48 GB card** |
| `L18to20` dual | 3 | 126 / 96 | 228 (floor) | 45383 | 685 MiB — too tight |
| `L18to20` dual | 4 | 128 / 96 | 304 | 42204 | 3.9 GB ✓ |
| `L18to20` ctrl | 4 | 128 / 96 | 304 | 37821 | 8.2 GB ✓ |

**C is a weak memory lever here.** Dropping C from 304 to the 228 floor bought under 1 GB,
because the weight-delta tensors dominate and are full-weight-shaped regardless of C
(~2.7 GB for 3 blocks). The effective levers are per-rank batch and GPU count. That is why
the 3-block runs went to 4 GPUs rather than shrinking C: 4 GPUs at C=304 beats 3 GPUs at
C=228 on *every* axis — 2.7 GB more headroom, a third more components, and clean batch
divisibility. The user pre-approved 4 GPUs for exactly this case.

Two non-memory constraints also bit and are worth recording: every batch size must divide the
DDP world size, and **`eval.batch_size` must equal `pd.batch_size`**, because
`PersistentPGDReconLoss` sizes its persistent adversarial sources from the train batch and is
auto-evaluated. Both reference configs happen to satisfy the latter, which is why it had
never surfaced.

## Runs launched

| run | blocks | GPUs | batch / nt | C | job |
|---|---|---|---|---|---|
| `addsub-L18-09-dual` | 18 | 2 | 128 / 96 | 456, (72,72,128,128) | 6076 |
| `addsub-L18to20-01-dual` | 18,19,20 | 4 | 128 / 96 | 304, (48,48,88,88) | 6077 |
| `addsub-L18to20-01-ctrl` | 18,19,20 | 4 | 128 / 96 | 304, (48,48,88,88) | 6078 (queued) |

20 000 steps, `slow_every: 5000`, `ABGridDataset` in the slow set. The two `L18to20` runs are
identical but for the scheme — same batch, same C, same GPU count — so the comparison is
clean; the ctrl simply uses less of its allocation. The ctrl is queued behind the 6-GPU
per-user cap and starts automatically.

Wall-clock: the QOS caps a job at 24 h and the single-block reference took ~16.5 h, so
`addsub-L18-09-dual` should finish in one leg. The 3-block runs will likely need a resume
leg; the trainer checkpoints on SIGTERM, so a killed job resumes from its last step.

## What to check first, at step 5000

1. **The sanity check itself** — the two-colour (a,b) heatmaps in `ab_grids/`. Expect white
   (both) and green (hidden-only) regions; **magenta means output-important but
   hidden-unimportant, which should not happen.** Any magenta is either a real finding about
   the CI networks or a bug in one of them.
2. **`NAlive_hidden/total` vs `NAlive_output/total`.** The hidden net should be denser. If it
   saturates at total C, the shared SmoothL0 coeff (5e-5) is too low for it — a result, not a
   bug, but it changes what the run can tell us.
3. **`CIHiddenActsRecon_outputCI` vs `_hiddenCI`.** The gap is the quantitative version of the
   whole hypothesis: how much hidden-activation error does the output net's CI assignment
   leave on the table? The ctrl run's single value is the baseline.
4. **Dual vs ctrl on the 3-block runs**, on both output KL and hidden-acts error. The premise
   is that hidden-acts reconstruction helps most when chaining blocks; this pair is the test.

Note that `CIHiddenActsReconLoss` changed definition (raw per-module MSE → relative error, no
delta), so its values are **not** comparable to those logged by earlier runs such as
`addsub-L18-04-hidden`. Do not overlay those curves.
