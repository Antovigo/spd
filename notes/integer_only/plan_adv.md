# Integer-only-adv CI filter of addsub-answers-09

Code: `param_decomp/param_decomp/integer_only/adv/` (CLI `python -m param_decomp.integer_only.adv.run`).
Config: `notes/integer_only/addsub-answers-09-adv.yaml`. Pod driver: `notes/integer_only/pod/run_adv.sh`,
called by `notes/integer_only/pod/run.sh` after the global-mask stages. Companion of the
integer-only global mask (`notes/integer_only/plan.md`), whose notation, pool, check set and
scoring this pipeline reuses.

## What this produces

Run p-512d73ab ("addsub-answers-09") decomposes every linear site of Llama-3.1-8B (32 blocks,
124,928 components). A *site* is one linear module, named like `layers.26.mlp.down_proj`; a
*component* is a site name plus its index within that site's decomposition (an integer from 0
to the site's component count minus 1). The run's CI function (*CI fn*) maps the original
model's activations on a prompt to a causal importance (*CI*, a real number in [0, 1]) per
component and token position.

The integer-only global mask keeps ONE fixed component set for every prompt. This pipeline
instead keeps the components frozen and fine-tunes the run's CI fn so that, on the arithmetic
prompts, the components it turns on reproduce the original model's distribution among the
integer tokens at the last position, under the run's own adversarial reconstruction term. The
fine-tuned CI fn is the *integer-only-adv CI*. The *integer-only-adv components* are those
whose integer-only-adv CI exceeds 0.1 at some position of some pool prompt; the
*integer-only-adv model* has exactly those components on at every prompt and position, every
other component and the weight delta off. It is scored on the same check set as the global
masks, and its components' activity over the pool goes into a dataset and an (a, b)-grid
applet.

## Notation

As in `plan.md`: the pool is the 240,200 prompts `Answer: {a} plus {b} is ` /
`Answer: {a} minus {b} is ` (`a`, `b` integer operands 0..399, subtraction only `b <= a`), each
10 tokens (positions `t`, integer, 0..9; position 9 is the last); `r` (integer) is a prompt's
true answer; `P` the prompts with `r >= 0` (the whole pool); `Q` the integer-only check set
(32,768 prompts of `P`, `$RUN/integer_only/check_set.npy`); `V_int` the 1000 tokens of the
integers 0..999 in integer order; `p` a model's next-token distribution at the last position;
`q(x) = p(x) / Σ_{x' ∈ V_int} p(x')` for integers `x` in 0..999; `q_Llama` the original
model's `q`;

    KL_int = Σ_x q_Llama(x) (log q_Llama(x) − log q_m(x))   (nats)

for a model `m`. `KL_full`, `ce_int` and accuracy are `plan.md`'s.

For one site with `C` components (integer), input `x` (a `d_in`-vector per position) and target
weight `W` (`d_out` x `d_in`): `V` (`d_in` x `C`) holds the components' input vectors as
columns, `U` (`C` x `d_out`) their output vectors as rows, `m` (length `C`, entries in [0, 1]) a
mask, `d` (real in [0, 1]) the weight-delta mask. The site's masked output is

    y = ((x V) ⊙ m) U + d · x (W − V U)^T

(`⊙` the entrywise product). *Weight delta off* means `d = 0`. `ci_c` (real in [0, 1]) is
component `c`'s CI at one (prompt, position); `s_c` (real in [0, 1]) a *source* value. A mask
built from a source is `m_c = ci_c + (1 − ci_c) s_c`.

## The fine-tune

What trains: every floating parameter of the run's CI fn at step 40000 (its output head gets
the gradient). The components (`U`, `V`) are frozen. Initialization: the run's CI fn at step
40000.

Data: the pool rows of `P` outside `Q` (207,432 prompts), target data only. No non-target
batch, no hidden-activation reconstruction. Each Adam step `k` (integer, 0..2499) draws 4096
distinct rows (`ci_filter.step.sample_batch`, a pure function of the seed 0 and `k`). Training
outside `Q` keeps `Q` held out, as for the global masks.

Reconstruction: the run's own merged stochastic-subset persistent-PGD term
(`MergedStochasticSubsetPPGDReconLoss`, SPEC S34, `ci_filter/ppgd.py`), scored by `KL_int` at
the last position. Per step and prompt, with probability `α` (real, the adversarial fraction)
the prompt is adversarial: every site is routed and its masks use the persistent adversary's
sources; otherwise its sources are fresh `U[0,1]` draws and a uniform-k subset of sites is
routed (an unrouted site computes `x W^T`, the target's own output). The weight delta is ablated
exactly as in the run's target stream: at every routed site its mask is the adversary's persistent
delta source on an adversarial prompt and a fresh `U[0,1]` draw otherwise, in the warmup ascents
and the main forward alike (`MergedPPGDRecon.weight_delta: on`; Antoine, 2026-11-13). The
integer-only-adv model and every score keep the delta off. The reference `q_Llama` is the original model's clean
forward in the same jit (the run's bf16 weights), not the cached `reference_log_q.npy`.

Per step: `n_w` (integer) warmup ascents of the adversary against the all-adversarial forward
(CI detached), then the CI fn's gradient of

    L = c_r · mean KL_int + c_a · A(ci_upper; γ) + c_f · F(ci_upper; γ, a')

over the batch (microbatches summed), then one more adversary ascent from the same backward.
`A` and `F` are the run's importance-minimality activity (Geman-McClure smooth L0 with width
`γ` (real > 0), normalized at CI 1) and frequency penalty (`f log2(1 + a' f)` summed over
components, `f` (real in [0, 1]) a component's firing frequency over the microbatch's CI
entries, `a'` (integer) its reference datapoint count), on the upper-leaky CI over all 10
positions. `c_r`, `c_a`, `c_f` (reals) are the coefficients.

Ceiling ON: every CI the filter reads (training masks, imp-min, alive list, collection) is the
entrywise minimum of the trained fn's and the run's own CI (`ci_filter.step.CIConstraints`,
the run's CI fn frozen in bf16). Over the cap only a gradient that lowers the CI passes. Prune
OFF: no component is removed at the start (`CIFilterConfig.prune: false`).

## Coefficients: the run's values at its final step

Every loss coefficient and the adversary's optimizer are CONSTANT at the run's value at its
last training step. The run has `S` = 40,000 steps (integer) and evaluates each schedule at the
run fraction `k / (S − 1)` of its 0-based step `k`, so its last update (`k` = 39,999) reads every
schedule at fraction 1. `integer_only/adv/coefficients.py` (`resolve_final`) reads the run's
pinned `pd` section and evaluates each schedule there; stage `train` writes the result to
`coefficients.json` and the resolved training config to `training/filter_config.yaml`.
Resolved for p-512d73ab:

| quantity | value | run schedule |
|---|---|---|
| `c_r` | 1.5 | constant |
| `α` | 0.9 | 0 → 0.9 by fraction 0.25 |
| `n_w` | 2 | `n_warmup_steps` |
| adversary Adam `β₁`, `β₂`, `ε` | 0.5, 0.99, 1e-8 | constant |
| adversary learning rate | 0.01 | 0 → 0.01 by fraction 0.0125 |
| `c_a` | 5e-5 | 2e-4, × 0.25 from fraction 0.25 |
| `γ` | 0.01 | 1.0 until 0.75, then → 0.01 |
| `c_f` | 2.5e-5 | 1e-4, × 0.25 from fraction 0.25 |
| `a'` | 1280 | `auto` = the run's target batch 128 × 10 positions |
| CI-fn Adam `β₁`, `β₂` | 0.9, 0.999 | (no weight decay, no gradient clip) |

The run writes `a'` as `auto`: each call's own `B·T` (`B` the batch, `T` the positions), 1280
on its target stream. The filter holds 1280 fixed, so the penalty per firing rate is the run's
at any filter (micro)batch.

Not carried (with `coefficients.json`'s `dropped`): `UnmaskedReconLoss` (coefficient 0.5; every
component on with the delta off does not depend on the CI fn, so its gradient there is 0); the
hidden pass (its recon, KL rider and imp-min); the non-target pass; the CI-scaled weight decay
and the components' optimizer (the components are frozen); the CI fn's learning-rate schedule
(replaced below).

Schedule (the `IntegerOnlyAdvConfig` default; the addsub-answers-09 pod config overrides it with batch 2048 × 2000 steps): 2500 steps, batch 4096, CI learning rate 8e-5 constant (no decay), Adam `β` as the
run's, `ε` 1e-8 (the run's AdamW `ε`). Mesh: every visible device (4 H100 on the pod), each
microbatch split over them, the weights whole on each (`placement: ddp`, below). The batch is
split into microbatches of `microbatch` prompts (hardware-only; the per-row source gradients and
the CI-fn gradient are summed).

## The adversary's initial state

The run's checkpoint at step 40000 holds the target-stream adversary under
`training/adversaries/MergedStochasticSubsetPPGDReconLoss/` (the hidden pass's is a separate
`HiddenMergedStochasticPPGDRecon` entry, not read): per matrix kind, component sources
`(32, 128, 10, C)` and delta sources `(32, 128, 10)` float32 (32 blocks, `B` = 128 batch rows,
10 positions, `C` the kind's components per block), their Adam moments, and the ascent counter.
`adversary_init: run` restores all of it unchanged (`integer_only/adv/adversary.py`), so the
first filter step continues the run's ascent (Adam bias correction included).

Shapes: the default batch (4096) has 32 times the run's rows (the pod's 2048: 16 times). Batch element `j` (0-based, in
batch order) reads row `j mod 128` (`ci_filter/ppgd.py`), and each row's gradient is the sum
over its 32 elements. In the run a row is not tied to a prompt either: every step draws a fresh
batch and row `i` attacks whichever prompt sits at slot `i`. Tiling the rows (instead of 4096
fresh rows) keeps the trained state and costs 1.9 GB per device (sources and moments,
replicated); 4096 rows would be 61 GB.

Semantics that differ from the run's: (1) the run's adversary ascended the run's own objective
(full-vocabulary KL); here it ascends `KL_int` at the last position, so its component and delta
sources start from the run's attack on a different objective; (2) each row now attacks
`batch / 128` prompts per step instead of 1 (16 at the pod's batch 2048). The restored rows are the closest
available start; `adversary_init: fresh` (U[0,1] sources, zero moments) is the alternative.

## Stages

Each stage is `python -m param_decomp.integer_only.adv.run <stage> --run_dir $RUN --data_root
$DATA_ROOT --config notes/integer_only/addsub-answers-09-adv.yaml`, from the repository root.
It reads `$RUN/integer_only/` (`config.yaml`, `pool.npz`, `check_set.npy`, and for `score`
`sweep/summary.tsv`); the integer-only stages `candidates` and `problem` must be done. Output
goes to `$RUN/integer_only_adv/`. A stage writes `status/<stage>.done` (JSON facts, including
`peak_device_gib`, the allocator's peak per device) or `status/<stage>.failed` (the traceback); a
done stage is skipped. The first stage pins the config as `config.yaml`; later stages refuse
other settings (the hardware-only `microbatch`, `accumulate_on_host`, `save_every`, `log_every`,
`ci_batch` excepted).

1. `train`: resolves the coefficients, restores the run and its adversary, runs the 2500 steps
   and writes `ci_fn/`. Every `save_every` (250) steps it saves the CI fn, its Adam state, the
   adversary and the step to `training/state_<k>/` (`training/latest_state.json` names the
   newest complete one); a rerun resumes there on the same trajectory. `ci_fn/` is written as
   `ci_fn.partial/` and renamed; the state folders are deleted after the rename, and a rerun
   that finds `ci_fn/` only deletes them.
2. `alive`: the integer-only-adv CI over all 240,200 prompts (batches of `ci_batch`): writes
   `max_ci.npz` (per site a `(10, C)` array, the max over prompts) and `integer-only-adv.json`.
3. `score`: the integer-only-adv model on all of `Q`, exactly as stage 4 of `plan.md` scores
   the global masks (`integer_only.pick.score_masks`, the same factored forward, precision and
   original-model reference): size, `KL_int`, `KL_full`, `ce_int`, accuracy, for all prompts,
   additions and subtractions. Writes `final.json` and `comparison.tsv` (below).
4. `collect`: as stage 5 of `plan.md`, for every integer-only-adv component and every position
   `t` where it is ever ON: (a) its inner activation `x V_c` in the integer-only-adv model (fp32
   forward, stored float16); (b) its binarized integer-only-adv CI (> 0.1), from a second CI
   pass; the stored pairs are those ON in stage 2 or in this pass.
5. `applet`: `ab_grids.html` and `ab_grids/` from `datasets/`, the page and conventions of
   `$RUN/integer_only/ab_grids.html`; its two views are the binarized integer-only-adv CI and the
   inner activation in the integer-only-adv model divided by `||V_c||`.

## Layout of `$RUN/integer_only_adv/`

    README.md               index (rewritten after every stage)
    config.yaml             the pinned settings
    status/                 <stage>.done / <stage>.failed
    coefficients.json       the final-step coefficients, adversary and CI-optimizer settings, `dropped`
    training/               filter_config.yaml, metrics.jsonl (state_<k>/ while training)
    ci_fn/                  the fine-tuned CI fn (orbax)
    max_ci.npz              {site: (10, C) float32}
    integer-only-adv.json   the integer-only-adv components
    final.json              the integer-only-adv model on Q
    comparison.tsv          the global-mask summary rows plus `integer_only_adv`
    datasets/               README.md, index.npz, inner_pos<t>.npy, ci_on_pos<t>.npy
    ab_grids.html, ab_grids/  the applet and its data
    logs/                   the pod driver's stage logs

`integer-only-adv.json` has the schema of `$RUN/integer_only/integer-only.json`: `rule`, `source`,
`threshold` (0.1), `n_alive` (number of listed components), `n_components` (124,928),
`components` (`{site: [component indices]}` over all 224 sites). `datasets/` has the format of
`$RUN/integer_only/datasets/` (`plan.md`), its own `README.md` stating the CI it binarizes.

`comparison.tsv` is `$RUN/integer_only/sweep/summary.tsv` (the original model, the
all-candidates model and every λ mask, `chosen` marking the integer-only mask) with one more row,
`integer_only_adv`, in the same columns. All rows are on the same `Q` against the same original
model. The global-mask summary itself is left untouched, so rerunning the integer-only `pick`
stage cannot drop the row. If `summary.tsv` does not exist when `score` runs, the table holds
the original model's row (scored here) and the adv row.

## Running on the pod (4 GPUs)

`notes/integer_only/pod/run.sh` runs the integer-only stages, then calls `run_adv.sh`, which runs
`train` on all GPUs, `alive`, `score` and `collect` on GPU 0, and `applet` on the CPU, each with
its log in `$RUN/integer_only_adv/logs/<stage>.log`, with `--xla_gpu_deterministic_ops=true` and
`--xla_gpu_enable_command_buffer=` (no CUDA command buffers, as in the run's own launch
environment; without it the first L40 smoke's training step failed with "Failed to add memset
node to a CUDA graph") added to `XLA_FLAGS`. It stops at the first failure. `run.sh`'s log ends
with one `=== all stages done` line only after both pipelines finish; `run_adv.sh` prints
`=== adv stages done`.

`train` places the restored run with `placement: ddp` (the target, the components and the CI fn
whole on every device, the batch split over the devices). Under `zero1` (`restore_jax_run`'s
default, every weight sharded and gathered in each forward) a 2-L40 step took 22 s at both
microbatch 32 and 128, against 3.7 s under `ddp` at microbatch 128: the gathers dominated.
`score` and `collect` build the single-device factored model and assert one visible device (on
2 L40s that build under a 2-device mesh ran for over 20 minutes without finishing); `alive`
runs on the same GPU so its CI pass and `collect`'s are one device's programs, as integer-only
stages 1 and 5.

## Smoke (L40, 2026-11-13)

On the p-512d73ab step-32000 copy (the step-16000 copy's `training/` item is truncated: 11 GB
of the 44 GB a full checkpoint stores, its adversary unreadable), with the integer-only smoke's
pool (`a`, `b` in 0..109, 18,205 prompts, `Q` 2,048 of them) and its `summary.tsv` (made at step
16000), every stage ran: `train` (20 steps, batch 64, microbatch 32, `zero1`, 2 L40s), then
`alive`, `score`, `collect` on one L40 and `applet` on the CPU.

- The adversary restored with 96,000 ascents (32,000 steps × (2 warmups + 1)); the filter's
  first step continued from it (96,003 after step 0).
- 20 steps: `KL_int` 0.030 at step 0, L0 (components with CI > 0 per token) 1148 → 568.
- 8,153 integer-only-adv components; the integer-only-adv model on `Q`: `KL_int` 0.0267,
  `KL_full` 0.0271, `ce_int` 0.0552, accuracy 1.000 (`comparison.tsv`).
- `collect` stored 15,138 pairs; with `alive` on the same single GPU, 0 pairs were ON in only
  one of the two CI passes (57 when `alive` had run on 2 GPUs).

Training step time and peak device memory (`peak_device_gib`), `m` (integer) the prompts per
device per microbatch:

| GPUs | placement | microbatch | `m` | s/step | microbatches/step | peak GiB/device |
|---|---|---|---|---|---|---|
| 2 L40 | zero1 | 32 | 16 | 21.9 | 2 | 29.2 |
| 2 L40 | zero1 | 128 | 64 | 22.7 | 2 | 33.4 |
| 2 L40 | ddp | 64 | 32 | 2.73 | 2 | 39.3 |
| 2 L40 | ddp | 128 | 64 | 3.73 | 2 | 41.4 |
| 2 L40 | ddp | 256 | 128 | out of memory | | |
| 4 L40 | ddp | 256 | 64 | 10.3 | 2 | 40.6 |

The 4-L40 row is after the microbatch source slice was split over the batch axes; before, the
same setting ran out of memory, and the process then waited in an NCCL rendezvous instead of
exiting (no `.failed` marker; the job ran until cancelled).

Pod estimate (extrapolated, not measured on an H100): microbatch 1024 (`m` = 256, 4
microbatches per step). Memory: 40.6 + 0.0656 × (256 − 64) ≈ 53 GiB from the measured slope,
plus the held source-gradient slices (≈ 5 GiB per device at 4 microbatches) — under the ~75 GiB
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` leaves. Time: on 2 L40s a ddp microbatch took
0.87 + 0.0156 `m` seconds; at `m` = 256 that is 4.9 s, and an H100 is 3 to 4 times faster per
device, so 1.2–1.6 s per microbatch plus the inter-device traffic (3.3 s per microbatch over 4
L40s' PCIe, much less over NVLink): about 5.5–7 s per step, 4–5 h for 2500 steps. `alive`,
`score`, `collect` scale with the pool (13.2 × the smoke's): ~5–15 min each on one H100.
