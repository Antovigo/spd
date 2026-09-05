# Multi-block targeted decomposition: what had to change, and what it means for the full network

2026-08-24. Written while scaling the addsub targeted (tPD dual-objective) line from ONE
decomposed block (`addsub-L18-16-ntmerged-bsc`) to FOUR (`addsub-4L17-20-01`, then
`addsub-4L13-16-01`) on 4x L40. Companion to `notes/l40_tpd_jax_blockers.md` (driver/OOM
mechanics on this box) — that note covers why a failed multi-GPU step hangs instead of
exiting; this one covers what was actually wrong and what to upstream.

Everything below is measured on `l40-worker` (L40, sm_89, 45–48 GB/card, no peer-to-peer),
llama-3.1-8B target, seq 64 broad stream / 5-token prompt pool, dp 4.

---

## 1. The one core change: `stacked_tail` (commit `14df37a81`)

**Symptom.** Four decomposed blocks OOM'd at EVERY batch size and EVERY `fsdp` width. The
smallest shape tried (target 64 / non-target 48 — a quarter of the proven single-block
per-rank batch) failed exactly like the largest.

**What made it diagnosable.** Three facts, in order:

1. The temp arena was **batch-invariant**: 27.84 / 27.22 / 26.85 GiB across a 2x batch
   range. Memory that does not move with batch is not activations.
2. It was **fsdp-invariant** in the way that matters: rank 0's request went 27.8 GiB
   (fsdp 1) -> 20.8 GiB (fsdp 2) -> still OOM at fsdp 4. Sharding moved it, but did not
   remove it.
3. It was **block-count-invariant**: the failing allocation for ONE decomposed block at dp4
   was byte-identical (28.32 / 31.30 GiB) to FOUR blocks at fsdp 2. Memory that does not
   scale with the thing you added is not caused by the thing you added.

Together those say: the arena scales with the FROZEN SUFFIX (11–13 layers in every variant),
not with the decomposition. Confirmed by dumping buffer assignment —
`XLA_FLAGS=--xla_dump_to=...` is already wired via `experiments/lm/training.py::enable_hlo_dump`
(writes to `<run_dir>/hlo`, survives an exec-time OOM because compile completes first), and
`core/tools/memreport.py` parses it. The dump named ~6 concurrent `wrapped_slice` copies of
`bf16[11,14336,4096]` (1.2 GiB each, in both matmul layouts).

**Root cause.** `glu_transformer.py` kept ONE post-prefix stack and sliced it per masked
forward (`slice_layers`: `self.stacked[lo-start:hi-start]`). A step runs ~10 masked forwards
(recon grid x adversary ascents x streams x CI roles), and XLA cannot CSE those slices across
the `optimization_barrier`s that `sequential_passes` inserts, nor across peeled scan loops.
So each forward materialized its own multi-GB copy of the frozen blocks it was not even
decomposing.

The repo had already learned this lesson once: `stacked_prefix` is a separate field, and its
comment says *"an in-graph slice of a multi-GB stack materializes copies and breaks
command-buffer capture (measured ~8x slower)"*. The frozen TAIL never got the same treatment.

**The change.** Symmetric with the prefix, three stored stacks instead of two:

| field | blocks | mask-dependent? | why a field |
|---|---|---|---|
| `stacked_prefix` | `[0, split_layer)` | no — reused once per step (S3/S18) | pre-existing |
| `stacked` | `[split_layer, tail_layer)` — the DECOMPOSED SPAN | yes | pre-existing, now narrowed |
| `stacked_tail` | `[tail_layer, n_layer)` | YES — every forward re-runs it | NEW: so no forward slices it |

Plus two consequences that fall out of the same edit:

- the per-kind V/U, CI and mask stacks size to `span_layers` (4 layers) instead of
  `scan_layers` (15) — the old stacks zero-filled a dummy entry for every frozen layer;
- at the common all-sites-live shape the per-kind entries are passed WHOLE rather than
  re-sliced per forward (an identity slice is skipped, not merely elided).

Layout only. Both scans run the same layers in the same order, so this is not even a D4
float-reassociation change: the frozen torch-equivalence goldens and the prefix
bit-equality test pass unchanged.

**Effect (4 blocks, 28 sites, target 128 / non-target 96):**

| layout | before | after |
|---|---|---|
| `fsdp: 1` | OOM at every batch | **31.8 GB, 2.8 s/step** |
| `fsdp: 4` | 21.4 GB, 25.5 s/step | unchanged (still available, still slow here) |

20k steps: ~5.9 days -> **~19 h**. For calibration, torch's 3-block dual run is 3.83 s/step
and its 1-block dual run 3.04 s/step vs JAX's 3.35 — i.e. the fixed JAX step sits where the
torch line says it should, and the pre-fix 7x gap was entirely this bug.

**Validation run before committing:** targets 67/67 (incl. frozen torch-equivalence goldens
and `test_prefix_reuse` bit-equality), core 564/564, library 526/526, basedpyright clean;
at 4 simulated devices 630 passed. Three tests updated because they pinned the OLD two-stack
field layout (`test_activation_capture`'s compact-graph lowering oracle, `test_prefix_reuse`,
`test_qwen3_8b`'s q_norm shape assert).

---

## 2. Pre-existing failures found on the way (candidates for upstream)

Both reproduce on UNMODIFIED `feature/dual_obj_jax` HEAD — verified by running them in a
worktree at the parent commit — so they are not fallout from the change above, but they do
mean the documented validation stack is partly red:

- `param_decomp/targets/invariance_check.py` FAILS at 4 simulated devices ("trajectory
  diverged across shardings — SPMD correctness broken (SPEC D4)"). The reported deltas are
  ~1e-6 relative on grad norms, i.e. plausibly a tolerance that no longer fits the current
  loss set rather than a real SPMD break — but it is asserting, so nobody can use it as the
  gate it is meant to be.
- `param_decomp/core/tests/test_weight_init.py::test_placed_init_matches_the_eager_values`
  refuses to run under `XLA_FLAGS=--xla_force_host_platform_device_count=4` ("single-device
  by construction"), so the documented "run the suite at 4 devices too" step cannot be run
  clean as written.

---

## 3. Config-side facts (no code change, but load-bearing)

**`fsdp: 1` is right on this box and multi-block made it possible.** The L18 line used
`fsdp: 1` to dodge the per-layer weight gather (l40-worker reports CNS between every GPU
pair, so NCCL routes it over shared memory: ~9x on the step here — 2.8 s vs 25.5 s). Four
blocks could not use it until the tail split. On NVLink hardware the trade reverses and
`fsdp` is nearly free, which is why the H100 plan re-enables it.

**Measured sizing, 4 blocks / 28 sites / dp4 / fsdp1** (`train/mem/peak_gb_per_rank`,
median `train/perf/step_time_s`; cap is ~46.9 GB usable at `mem_fraction 0.97`):

| blocks | target/non-target batch | hidden points | peak | s/step |
|---|---|---|---|---|
| 17–20 | 128 / 96 | 15 | 31.8 GB | 2.80 |
| 17–20 | 192 / 144 | 15 | 32.9 GB | 3.45 |
| 17–20 | 256 / 192 | 15 | OOM | — |
| 13–16 | 128 / 96 | 15 | 31.8 GB | 3.11 |
| 13–16 | 128 / 96 | 19 | 31.9 GB | 3.11 |
| 13–16 | 192 / 144 | 15 | 33.1 GB | 3.81 |
| 13–16 | 192 / 144 | 19 | **OOM** | — |

Three things to take from that table:

1. **Decomposing LOWER costs more.** Same 4 blocks, same batch: 3.45 s (17–20) vs 3.81 s
   (13–16), because the frozen tail every masked forward must run grows 11 -> 15 blocks.
   The prefix is free (computed once per step, S3/S18); the tail is not.
2. **Hidden points are nearly free in isolation** (31.8 -> 31.9 GB, no step-time change for
   15 -> 19 points) — but they are NOT free in combination with a larger batch: 192/144 with
   19 points OOMs while either knob alone fits. Memory here is not additive across knobs;
   probe the exact production shape, never infer it.
3. **Batch is the strong knob for both axes** (+1.2 GB and +0.7 s going 128 -> 192).

**Which knob to give up.** For 13–16 the batch was kept at 192/144 and the hidden points cut
19 -> 15 (`layers.13..27`), preserving the 17–20 run's hidden-pass STRUCTURE exactly (4
in-span + 11 downstream). Rationale: batch changes optimizer dynamics the LRs were tuned
around and would confound the block-to-block comparison the run exists to make; the hidden
points change WHICH activations are matched, and dropping the four furthest-downstream ones
is the smaller perturbation. Eval-only knobs (`eval.batch_size`, `PGDReconLoss.n_batches`)
are the next thing to trade — they cost diagnostic precision, not training semantics.

**Allocator.** Probes and both runs set `runtime.launch_env.xla_python_client_allocator:
platform`. It was adopted while chasing the OOM (to rule out BFC fragmentation — it did not
help, which was itself informative) and kept because it is what the passing configuration was
validated under. Worth an A/B against default BFC now that the real cause is fixed; if BFC is
equal or better, drop it.

---

## 4. Launcher lessons (not repo code, but they cost hours)

- **A wedged multi-GPU job does not die on `scancel`.** The trainer catches SIGTERM to save,
  and a rank stuck in a clique-acquire never gets there, so the process survives its own job
  holding ~33 GB/card. It then poisons the NEXT job, which OOMs and looks like a capacity
  result. This happened twice; the second time it invalidated a whole diagnostic ladder.
  Use `timeout -k` (SIGKILL fallback) around the trainer, prefer `kill -9` over `scancel`
  for wedged jobs, and verify `nvidia-smi --query-compute-apps` is empty before submitting.
- **Keep the contention ABORT, not just the retry.** Probe scripts that waited for free cards
  but then proceeded anyway turned contention into fake OOM data. The guard must exit
  non-zero and print the offending PIDs.
- **Probes must cover the EVAL envelope.** Probes with eval disabled do not exercise the
  step-0 slow eval (20-ascent PGD + hidden-acts), which is where a long run would die at
  minute one. Smoke at the production shape with eval ON, plus save/resume.

---

## 5. What this means for the FULL-NETWORK scale-up

**The tail fix does not help the 32-block run — by construction.** With every block
decomposed, `split_layer = 0` and `tail_layer = n_layer`: no prefix, no tail, span = the whole
model. The fix matters for every SUBSET rung (4, 8, 16 blocks) and for any targeted run that
decomposes a window, which is the entire addsub line. The span-sized V/U/CI/mask stacks
likewise collapse to "all layers" and stop being a saving.

**So the full-network constraints are different ones**, and the numbers here do not
extrapolate to them:

- V/U + Adam state dominate (the existing full-32L seat is ~18.3 B V/U params at ΣC 1.245 M,
  sized for dp 64). At targeted C's (ΣC ~7.8 k for 4 blocks) that term is negligible today;
  at 32 blocks of targeted C it is ~62 k ΣC — still small. The full-32L *plain* recipe is the
  one that needs 64 GPUs, not a targeted 32-block run.
- Every masked forward runs the whole model regardless, so step time at 32 blocks is roughly
  the 4-block cost with span 32 instead of 4 live sites — i.e. the LIVE-site cost (~9x per
  site) becomes the dominant term rather than the frozen tail. That is what
  `ChunkwiseSubsetReconLoss` exists for and it is the first thing to re-tune at 8+ blocks.
- The hidden pass scales with points x batch and is already the knob that broke first at 4
  blocks. At 32 blocks with "span + downstream" points it needs an explicit policy
  (sub-sample the points, or cap them).

**The mechanism is expected to span the whole network** (Antoine, 2026-08-24), and the
4-block runs are consistent with that — per-block CI L0 was flat across 17–20 (14.7 / 17.5 /
12.1 / 14.2). So walking the window down the network block-by-block is NOT the path: it
answers a question whose answer is already assumed, at ~a day per rung. Subset runs are worth
running only as ENGINEERING rungs (do the memory/step-time properties hold as the span grows)
or to compare recipes, not to locate the mechanism.

**Recommended order:** one 8-block L40 rung as an engineering check (does the tail fix keep
holding; price `ChunkwiseSubsetReconLoss` against the all-sites plan at a span where the
live-site cost starts to dominate), then straight to the FULL network on 8x H100 SXM with
`fsdp` re-enabled — where the gather this whole note is about becomes cheap, the tail fix
becomes irrelevant by construction (§5 above), and the binding constraints are the live-site
recon cost and the hidden-point policy instead.

---

## 6. Upstream PR proposal

One PR, already isolated as `14df37a81` (4 files + a `core/CLAUDE.md` entry):

- **Title:** store the frozen tail as its own stack (`stacked_tail`)
- **Claim:** multi-block subset decompositions are memory-infeasible without it; with it,
  4 blocks fit at `fsdp: 1` and run ~9x faster than the sharded workaround.
- **Evidence to include:** the batch/fsdp/block invariance triple (§1), the buffer-assignment
  dump naming `wrapped_slice`, the before/after table, and the torch cross-check.
- **Risk:** layout-only; goldens and bit-equality tests unchanged. The three updated tests are
  layout assertions, not semantics.

Two smaller items worth separating out, if the maintainers want them:

- fix or re-tolerance `invariance_check.py` at >1 device, and make `test_weight_init` skip
  (rather than assert) under a forced multi-device flag — so the documented validation stack
  runs clean (§2);
- a launcher-side note (or a `timeout -k` example in the docs) for the wedged-job reaping
  problem in `notes/l40_tpd_jax_blockers.md` §"when these runs fail, they often don't exit",
  which is now confirmed to bite through `scancel` as well (§4).

---

## 7. Upstream state as of `facf2e7b1` (checked 2026-08-24)

Upstream PR #1000 ("Explicit placement, broader transformer targets, and evaluation
refinements") landed on 2026-08-24 — 264 files, +25k/−16k, one squashed commit. Local
`main` now carries it (merge `f455f6cb7`, clean; 53-test config/frequency/nonlinearity
subset green). `feature/dual_obj_jax` does **not** — see
`~/pd_scratch/dual_obj_jax/upstream-merge-2026-08-24/NOTES.md`.

Three things that change what §6 should say.

**7.1 The `stacked_tail` PR is still un-preempted — and now worth MORE.** Tip
`glu_transformer.py` still slices the single frozen `stacked` per masked forward
(`slice_layers(lo, hi) → jax.tree.map(a[lo:hi], self.stacked)`, line ~1714). Upstream has
neither `stacked_tail` NOR the `stacked_prefix` this branch already carried, so **both**
halves of the frozen-stack split are ours to contribute. §6's claim stands unchanged; the
PR just got a second, independent component. Upstream rewrote the file heavily around the
new anatomy abstraction, so the patch needs rebasing, not re-deriving.

**7.2 Recon chunking was deleted upstream** (SPEC amendment 2026-08-05, "Oli, dechunk").
`ChunkwiseSubsetReconLoss`, `CIMaskedReconLayerwiseLoss` and the entire live-set plan API
(`ReconPlan`, `ReconForward`, `make_plan`, `all_sites_live`, `each_site_live`,
`live_groups`, `subset_chunk_plan`) are gone. S2 is amended so masked forwards are TOTAL
over the model's sites; S10′ so every term routes over all sites, "matching the published
VPD recipe". The chunkwise **CI fn** (§4.6) survives — the glossary now reserves "chunk"
for that alone.

No production config here uses it: `addsub-L18-16-ntmerged-bsc` and both 4L configs route
all-sites (`StochasticReconSubsetLoss`, `PersistentPGDReconLoss`,
`MergedStochasticSubsetPPGDReconLoss`, `CIMaskedReconLoss`) and take the chunkwise CI fn
only. `addsub-4L17-20-01.yaml` says so in a comment: "Chunkwise recon is a roadmap
experiment, not this run." So the cost is a roadmap arm, not a run — but the
~9×-cheaper-masked-forward idea that made chunked recon attractive for the full network is
no longer something upstream will carry for us. If it is wanted at 32 blocks it is now a
local feature to re-add and defend, not a config knob.

**7.3 The mesh is now explicitly authored.** `hsdp_mesh(replicate, fsdp, tp)` /
`hsdp_abstract_mesh(replicate, fsdp, tp)` replace the `dp` + `gpus_per_node` derivation.
This SUBSUMES the branch's local `fsdp` override and `assert_inline_topology` — the
`fsdp: 1` trick that made the L40 runs ~9× faster becomes a first-class authored shape
rather than a workaround. Config impact: the `runtime.dp` / `runtime.fsdp` spelling in
every config here needs migrating when the merge lands.

Also gone upstream, for awareness: `param_decomp/autointerp/` and `param_decomp/harvest/`
were deleted wholesale (release-line trimming), along with `infra/` helpers and
`adapters/`. `targets/qwen3_8b.py` → `qwen3.py`, `llama8b.py` → `llama31.py`.

---

## 8. The merge, as landed (2026-08-24)

Branch `feature/dual_obj_jax-upstream-merge` (forked from `feature/dual_obj_jax`
at `14df37a81`, pushed to origin). Two commits:

| commit | what |
|---|---|
| `4616e8ce1` | the merge — 1280 tests pass, basedpyright and ruff clean |
| `73c79ae2a` | tests pinning the nonlinearity prior as a once-per-step weight-space term |

`feature/dual_obj_jax` itself is UNTOUCHED at `14df37a81`, and job 10409's frozen
worktree is pinned there, so the 13-16 run is unaffected either way.

### What made it a port rather than a merge

Both sides refactored the same core. Upstream split the shared `_StepAtoms` into
`ForwardSubstrate` + `ReconGrid` + free functions; this branch had grown per-pass
scoping, dual CI, delta-pinned ascents and `sequential_passes` on that same class.
`make_targeted_train_step` is therefore rewritten on upstream's primitives, keeping
all four passes. The checks that make that trustworthy are
`test_sequential_and_fused_passes_give_the_same_step` (the two paths must give the
same gradient) and the three-step tPD golden — both green.

### Three things that would have shipped broken

1. `persistent_delta_pinned_masks` merged with NO conflict but still assumed the
   pre-#1000 raw-array source layout; upstream had made `Sources` a typed
   `SiteSource(components, delta)` record.
2. The branch's zero/coupled inits called `component_stacks_from_sites`, which
   assigns one group PER SITE — silently mismatching the placement census.
3. The dual CI fn's hidden-head bias sharding referenced an undefined name; only
   reachable on a mesh with `dual: true`, so no CPU test would have caught it.

### Config migration (every config here needs all four)

```yaml
target:
  attention_implementation: xla   # NOT `auto` — see below
runtime:
  replicate: 2                    # was `dp: 2` + `gpus_per_node: 2`
  fsdp: 1
  tp: 1
  compilation_cache_dir: ~/.cache/param-decomp/xla   # now required
  compiler_options: tuned-v1                          # now required
cadence:
  checkpointing:                  # was `save_every` + `checkpoint_retention`
    kind: periodic
    save_every: 5000
    retention: {kind: keep_last, n: 2}
```

`attention_implementation: xla` is load-bearing, not cosmetic. Upstream turned this
branch's hard-coded cuDNN-off into a config knob; `auto` re-selects cuDNN, whose
graph API cannot run against this box's pre-CUDA-12.8 driver — the failure that
blocked every dp>=1 targeted run. A migrated config that leaves it at `auto` will
reproduce that.

`~/pd_scratch/dual_obj_jax/addsub-L18-16-nlpenalty.yaml` is a migrated, parsing
example of all of it.

### Still open

- **`stacked_prefix` / `stacked_tail` are NOT in the merge.** Upstream rewrote
  `glu_transformer.py` around a new anatomy abstraction at the same time, so the
  split needs re-deriving rather than re-applying. It is excluded deliberately so
  the merge commit is green; `test_prefix_reuse.py` came out with it and
  `core/model.py`'s `ResidualStart` / `SupportsPrefixResidual` are marked dormant.
  **This is what makes 4 blocks fit on 4x L40 — it must land before any multi-block
  run uses this branch.** §7.1 still stands: upstream has neither half, so both are
  ours to contribute.
- The merged branch has had no GPU smoke. Step-parity against a known run (same
  seed, same config, first ~50 steps) is the check worth doing before trusting it.

### 8.1 Measured: what losing the frozen-stack split actually costs (2026-08-25)

Three 60-step smokes on 2x L40, production shape (`allmerged-bsc` recipe, `replicate: 2`),
steady-state windows only — the first window includes compilation and is discarded.

| config | step time | peak/rank | run dir |
|---|---|---|---|
| `allmerged-bsc`, PRE-merge branch (with `stacked_prefix`/`stacked_tail`) | 3.581 s | 36.45 GB | (recorded baseline) |
| control: merged branch, **no** penalty, no prefix/tail | 7.134–7.160 s | 43.79 GB | p-eb3628e4 |
| `nlpenalty`: merged branch, **with** penalty, no prefix/tail | 7.149–7.192 s | 43.79 GB | p-db5a008c |

Two readings, and the control is what separates them:

- **The nonlinearity prior is free**: +0.02 s/step (~0.3%) and +0.00 GB against its own
  control. Expected — it is a weight-space term with no forward of its own — but now
  measured rather than assumed.
- **The frozen-stack split is worth 2x**: 3.581 → 7.14 s/step is **1.99x**, plus 7.3 GB
  (+20%) of peak. Every masked forward re-running blocks 0-17 instead of sharing one
  frozen lead per step is the whole difference; nothing else changed between the baseline
  and the control except the merge.

Consequence for the 20k run: 7.16 s x 20000 = **~39.8 h**, i.e. two legs against the 24 h
wall-clock cap, versus ~20 h with the split. The `RUN_ID=p-xxxxxxxx sbatch` resume path
handles two legs, so it is affordable — but re-deriving the split first halves it.

Peak at 43.79 GB on a 48 GB L40 is comfortable but no longer roomy; a heavier recipe on
this branch should re-measure before assuming it fits.

### 8.2 Two GPU-only bugs the CPU suite could not reach

Both found by these smokes, both fixed on the branch:

- `63db6fa0c` — the T11 CI-scaled decay factor is derived from CI (C on the ACTIVATION
  layout, `tp`) and multiplied into the component stacks (C on the PERSISTENCE layout,
  `('tp','replicate')`). A `ShardingTypeError` at step 0 under explicit axes. Needs a real
  mesh, so 1280 CPU tests and a clean type check said nothing.
- `a9111a49e` — the chunkwise CI fn chose its OWN attention backend with a literal
  `attn_implementation("auto", ...)`, ignoring `target.attention_implementation`. At the
  non-target stream's 64-token length that selects cuDNN, which this box's pre-CUDA-12.8
  driver cannot run; it surfaces as a bare `cudaErrorInvalidValue` naming
  `jit_targeted_step`, with no mention of attention. The backend is now an authored arch
  field (default `auto`, so upstream behaviour is unchanged) and **both** knobs must say
  `xla` here.

Three more schema migrations the smokes surfaced, beyond §8's list: the entry point's CLI
is positional with a required `local_device_count` (`CONFIG DATA_ROOT LOCAL_DEVICE_COUNT
--run_id`); `CIHistograms` now demands `eval.n_steps: 1` because its histograms are binned
exactly on device per batch; and `n_batches_accum` is gone from the schema.

### 8.3 The frozen-stack split, ported and measured stage by stage (2026-08-25)

Re-derived on upstream's rewritten `glu_transformer.py` in two commits, each measured by
the same 60-step production-shape smoke on 2x L40:

| stage | step time | peak/rank | commit |
|---|---|---|---|
| pre-merge baseline (`allmerged-bsc`, old branch) | 3.581 s | 36.45 GB | — |
| merged, no split | 7.14 s | 43.79 GB | (merge) |
| + three-stack split (`stacked_prefix`/`stacked_tail`) | 6.53 s | **28.23 GB** | `ff219fa9d` |
| + prefix REUSE (`ResidualStart`, once per stream) | **6.32 s** | 28.23 GB | `8784009be` |

1291 tests / basedpyright / ruff green at each stage.

**What the staging bought, and what it did not.** The split is a MEMORY fix, decisively:
peak falls 36% and lands 23% BELOW the pre-merge baseline — the per-forward `wrapped_slice`
copies of the frozen stack were the whole story there. But the two stages together recover
only 11% of the step time (7.14 → 6.32 s), and prefix reuse specifically — the part I
expected to dominate, since it removes 18 of every masked forward's 32 blocks — bought 3%.

**So the remaining 1.76x is NOT the frozen-stack layout.** 6.32 s against 3.581 s, same
GPUs, same recipe, same batch shapes, and (checked) a byte-identical XLA flag set —
`compiler_options: tuned-v1` is the same dict the branch defaulted to. Whatever it is came
in with the merge itself; the leading suspect is upstream's explicit-placement rewrite
(`AxisType.Explicit` + reshard-based collectives), which is the substance of #1000 and
changes how every collective is emitted. At `replicate: 2, fsdp: 1, tp: 1` the branch's
`fsdp: 1` trick meant essentially no parameter sharding, and the explicit path may be
emitting reshards the old implicit GSPMD path elided.

That is a profiling question, not a reading-the-diff question: `enable_hlo_dump` plus
upstream's new `core/tools/hlo_census.py` against the two branches at the same config is
the way to localize it. Not done.

Consequence for the 20k run as it stands: 6.32 s x 20000 = **35.1 h**, still two legs.
Worth knowing before spending them.

### 8.4 CORRECTION: the real baseline, and what the 2x is not (2026-08-25)

§8.1–8.3 compared against 3.581 s / 36.45 GB — a figure RECORDED in the branch's plan
notes on 2026-08-20, in a different measurement context. That was the wrong baseline to
reason from. Running the pre-merge branch (`14df37a81`) with the ORIGINAL
`addsub-L18-16-allmerged-bsc.yaml` through the SAME 60-step smoke harness, same two L40s,
same `train_log_every: 10`:

| | step time | peak/rank |
|---|---|---|
| pre-merge, like-for-like (run p-9307e684) | **3.099–3.147 s** | **21.16 GB** |
| merged + full frozen-stack port (run p-ef6dd44c) | 6.311–6.347 s | 28.23 GB |

So the regression is **2.02x step time and +33% peak memory** — and §8.3's claim that the
split put memory 23% BELOW baseline is WRONG: it was measured against the stale 36.45 GB.
The merged branch uses MORE memory than pre-merge, not less. The split still moved peak
43.79 -> 28.23 GB; it just does not get back under the real baseline.

What the 2x is NOT, each ruled out by measurement rather than argument:

- **Not the nonlinearity penalty.** Controlled on identical code: 7.16 s with, 7.14 s
  without (+0.3%), same peak.
- **Not the frozen-stack layout.** Ported in full (`ff219fa9d`, `8784009be`) and measured
  stage by stage; it is a memory fix worth 11% of step time.
- **Not XLA flags.** `compiler_options: tuned-v1` is byte-identical to the dict the
  pre-merge branch defaulted to — diffed, not assumed.
- **Not retracing.** Both runs compile `jit_targeted_step` exactly 3 times over 60 steps.
- **Not raw kernel count.** 1123 -> 1227 PTX kernels (+9%), nowhere near 2x.

Which leaves the merge's own numerics/placement. The leading suspect stays upstream's
explicit-sharding rewrite (`AxisType.Explicit` + reshard-based collectives): more time AND
more memory together reads like extra materialization or serialized collectives, not extra
arithmetic. The pre-merge dump has 309 all-reduce / 94 all-gather / 171 copy in the
optimized step.

Next diagnostic, blocked only on format: the two branches dump differently — pre-merge
writes `*_after_optimizations.txt`, the merged branch writes `.hlo.pb` plus `module.mlir`.
Convert the merged proto to text (or re-dump with the text flag) and run upstream's own
`core/tools/hlo_census.py` over both; it counts collectives inside while-loops vs entry and
across replicate groups, which is exactly the axis the explicit-placement rewrite changed.

### 8.5 LOCALIZED: it is the `zero1` preset, whose layout changed in #1000 (2026-08-25)

Collective census (`core/tools/hlo_census.py` over each run's optimized `jit_targeted_step`)
plus per-op all-reduce sizing. Same 60-step production-shape smoke throughout:

| config | step time | peak/rank | all-reduce ops | bytes all-reduced /step |
|---|---|---|---|---|
| pre-merge, `zero1` | 3.10–3.15 s | 21.16 GB | 13 | **299.7 MB** |
| merged, `zero1` | 6.32 s | 28.23 GB | 69 | **7968.9 MB** |
| **merged, `ddp`** | **2.76–2.83 s** | 26.79 GB | 12 | **356.1 MB** |

The merged branch under `zero1` all-reduces **26x more data across the replicate axis every
step** — 8.0 GB against 0.30 GB, 32 of those reductions ≥64 MB (combined buffers of
200–500 MB, op-named in `pd_value_and_grad_target/.../add_any`, i.e. gradient
accumulation). On this box that is the entire regression: its GPUs have no peer-to-peer
(the config's own comment records CNS between every pair), so NCCL routes cross-device
traffic over shared memory.

`zero1` did not mean the same thing before and after. SPEC D4, amended 2026-08-18: "the
per-group fallback is REMOVED entirely… `zero1`'s faithfulness rows are now its
intra-matrix master layout for ALL groups (the weights transition is the identity;
previously tiling groups took the stack-preferring pair)". The config asks for `zero1` on
both sides and gets two different layouts.

**`ddp` is the fix, and it is better than where we started**: 2.83 s/step is 10% FASTER
than the pre-merge baseline, at 12 all-reduces and 356 MB. Memory is 26.79 GB — +27% on
pre-merge, which is the expected price of replicating optimizer state rather than sharding
it, and comfortable on a 48 GB card.

`owner` is not an option at this shape: it shards the component stack axis over
`replicate`, and a single-block decomposition gives every semantic group stack length 1,
which cannot tile 2 — it refuses at config build (upstream's fail-closed placement claim,
working correctly).

Also measured, and NOT the answer: `replicate: 1, fsdp: 2` is 14.01 s/step / 22.85 GB —
2.2x worse than `fsdp: 1`, confirming the pre-merge lore that the parameter-sharding plane
is a bad trade on a no-P2P box, and ruling out "the explicit path is bad specifically at
fsdp=1".

Ruled out earlier and still ruled out: the nonlinearity penalty (+0.3%, controlled), the
frozen-stack layout (ported in full; a memory fix worth 11% of step time), XLA flags
(byte-identical), retracing (3 compiles both sides), kernel count (+9%).

**Worth raising upstream**: at a mesh with a degenerate `fsdp` axis, `zero1` emits per-leaf
cross-replicate reductions of the full gradient tree rather than one combined reduce. Their
own seats run `fsdp: 8`, so the shape that exposes it is one they do not exercise.

Consequence for the 20k run: 2.83 s x 20000 = **15.7 h — a SINGLE leg**, comfortably inside
the 23 h cap, against 35 h / two legs under `zero1`.

### 8.6 Launched: `addsub-L18-16-nlpenalty`, job 10631 (2026-08-25)

Run `p-f3728189`, 20000 steps, one leg. Frozen worktree `nlpenalty` @ `8784009be`.

Live confirmation at step 200:

- **2.87 s/step** (100→200 delta 4:47), ETA **15:44:24** — the 2.83 s/step smoke
  prediction holds in production. The step-100 ETA of 40:52:46 is the cumulative
  average still carrying compile + the step-0 slow eval; it converges on the second
  line, exactly as the 13-16 run's did.
- Mesh `replicate=2, fsdp=1, tp=1` with `components/optimizer_state (replicated)` —
  `ddp` in force, not `zero1`. This is the §8.5 fix doing its job.
- The prior is live AND working: `NonlinearityLocalityLoss` 1873 → 1316 over the
  first 100 steps, split `_neuron=1870` / `_attention_head=2.849` (both unit kinds
  present, matching `unit_kind_coefficients`). Coeff ramping on schedule —
  4.95e-05 / 9.95e-05 / 1.495e-04 at steps 100/200/300, i.e. linear to 1e-3 at 2000.
  `relative_threshold` still at its 4.0 knot.

Note the term's scale: at 1873 raw against a coeff reaching 1e-3, its weighted
contribution lands near 1.9 at full ramp, versus a `total` currently ~0.22. That is
the intended pressure, but it is the thing to watch in the curves — tPD has no
faithfulness role (T3), so nothing pushes back on U-concentration the way plain PD does.

**Unversioned recipe.** `~/pd_scratch/dual_obj_jax/` is not a git repo, so this config
and the 13 sbatch scripts (including the SIGTERM forwarding fix) exist in exactly one
place on scratch. Worth moving under the repo.

### 8.7 Run 10631 died at step 4000: the A/B grid got the raw CI fn (2026-08-25)

`p-f3728189` ran healthy for 3900 steps at 2.87 s/step, then died in the first
`ABGridDataset` snapshot:

    AttributeError: 'ChunkwiseTransformerCIFn' object has no attribute 'fn'

`save_every` was 5000, so the newest checkpoint was step 0 — 3h21m lost.

**Cause.** `ci_preactivations` runs the CI COMPUTE lifecycle, and post-#1000
`materialize_ci_compute_weights` reads `.fn` / `.placement` off a `PlacedCIFn` to
rebuild a chunkwise fn's compute weights against the resolved placement.
`ab_grid_operation.run` reached past the eval invocation to
`state.decomposition.ci_fn` and handed the bare fn down. `EvalInvocation` exists to
prevent exactly this — its docstring says operations consume `placed_ci_fn`, "never a
raw (fn, rules) pair" — and the grid op was the tree's only violator.

**Why three checks missed it.**
1. The grid path is branch-only, so upstream's placement threading never touched it.
2. Both seams annotated the parameter `ci_fn: Any` — the only two `ci_fn: Any` in the
   tree — so basedpyright had nothing to check. Now typed `PlacedCIFn`.
3. `test_ab_grid_dataset.py` DOES exist (378 lines) but builds a proper `PlacedCIFn`
   and passes it in, so it was green either way. The untested seam was the OPERATION's
   wiring, not the dataset code.

Fixed in `41ca10c1a`; regression test in `03600e3a4`, which poisons
`state.decomposition.ci_fn` with an object that raises on any attribute read and
requires the snapshot to come back. Verified adversarially: restoring the old argument
fails with the poison naming attribute `'fn'`.

**The scheduling trap that hid it, and that cost two smokes.** `schedule_for` gives
`ABGridDatasetConfig` its OWN schedule, `EveryAfterFirst(every, slow_every)`, firing
when `step != every and step > 0 and step % slow_every == 0` — it deliberately skips
the first eval pass, where an untrained decomposition clears any floor and the snapshot
is enormous and uninformative. At `every: 500 / slow_every: 4000` the first firing is
step 4000 exactly, which is where 10631 died — independent confirmation.

Consequence for smoke design: **a short smoke can NEVER exercise the grid** unless it
is aimed at the predicate. Setting `every: 20 / slow_every: 20 / steps: 25` made step 20
the SKIPPED pass — smoke 10649 exited rc=0 with no grid at all, a green run that proved
nothing. `every: 10 / slow_every: 20 / steps: 25` fires at step 20; smoke 10653 wrote
`step_20.js` + `index.html` + `manifest.js` with zero tracebacks. `saved_components=0`
there is expected, not a fault: readout heads init at `W=0`, so nothing reaches the 0.05
floor at step 20; the mean-CI vectors are stored for every component regardless.

**Relaunched** as job 10654 from worktree @ `03600e3a4`, with `save_every` 5000 -> 2500.
2500 still divides 20000, so every allmerged-bsc comparison point (5000/10000/15000/
20000) is still written — a strict superset — and the worst case halves.

### 8.8 Run 10654 died at step 4000 too — a SECOND sharding bug, one line later

Not a regression of 41ca10c1a: that fix worked and execution reached the next statement.
`_take_columns` gathers the saved components off the component axis, which under explicit
axes carries `tp` while the prompt axis carries `('replicate', 'fsdp')`:

    ShardingTypeError: Use `.at[...].get(out_sharding=)` ...
    Got operand=ShapedArray(float32[1000@(replicate,fsdp),1,512@tp]),
        indices=ShapedArray(int32[128,1])

A gather ALONG a sharded axis needs collectives, so JAX refuses to infer an output
sharding. Fixed in `fb07ebf3a`: keep the prompt/position layout, replicate the gathered
axis. The result goes to host immediately and is the saved columns only, so the implied
all-gather is negligible.

**Why the smoke that cleared the first fix missed this one — my error.** That smoke ran
the grid at step 20, where an untrained decomposition leaves every component below the
0.05 floor. `saved` was empty for every site, so `live` was empty and the chunk loop
`break`s BEFORE the gather. `saved_components=0` was correctly explained as expected, but
it also meant only the first half of `collect_ab_grid_snapshot` ran. I reported the path
as validated on the strength of a run that never executed the failing line.

The corrected smoke sets `mean_ci_floor: 0.0`, which saves everything and forces the
widest gather: 1952 components over 7 sites, a 104 MB `step_20.js`, rc=0, no tracebacks.
**Rule for this eval: reaching the snapshot is not covering it. Only a non-empty `saved`
exercises the gather.** (The 104 MB also shows concretely why the schedule skips the first
pass — at a zero floor the snapshot really is enormous.)

**Cost.** `save_every: 2500` limited the loss to ~1400 steps rather than 3900; the step-2500
checkpoint was there to resume from. Lowering it after the first crash paid for itself
within a day.

**Both fixes are eval-path only** — no training numerics — so the control resuming across
them (leg 1 without, leg 2 with) remains a valid comparison against allmerged.

### 8.9 State at the end of 2026-08-25

| run | id | job | status |
|---|---|---|---|
| `addsub-L18-16-nlpenalty` | `p-43e77281` | 10658 | resumed from step 2500, heading to 20000 |
| `addsub-L18-16-nlcontrol` | `p-cbb66ad1` | 10656 | resumed from step 2153, heading to 20000 |

The control's first 2000 steps against the pre-merge `allmerged-bsc` run (leg-1 log vs
job 10137), 20 matched points:

| metric | median abs rel diff | signed bias |
|---|---|---|
| `train/loss/total` | 2.28% | -1.21% |
| `hidden_ci/loss/total` | 2.44% | +0.36% |
| `MergedStochasticSubsetPPGDReconLoss` | 3.62% | -1.82% |
| `UnmaskedReconLoss` | 4.36% | -2.93% |
| `SmoothL0ImportanceMinimalityLoss` | 6.55% | -6.43% |
| `FrequencyMinimalityLoss` | 8.39% | -8.21% |

All 12 schedule keys byte-identical, which is the check that the fraction-preserving
`steps: 20000` decision was right. Exact agreement was never available: `ddp` vs `zero1`
reorders reductions and the stochastic/adversarial terms draw fresh masks. The signed bias
near zero on the aggregate losses is the real evidence the trajectories overlay; the two
importance/frequency terms carry a consistent negative bias worth a look in the curves.

Production-shape performance, median over steps 200-2000: **3.568 -> 2.912 s/step (-18.4%)**
and **36.82 -> 28.61 GB peak per GPU (-22.3%)**.

**Non-target PGD across a leg boundary: NORMAL, do not re-investigate.** The pre-merge
allmerged run's non-target PGD arms sit at ~0.22-0.27 through step 8000 and drop ~28x from
step 12000 (its leg-2 resume), while the merged control reports ~0.013-0.016 from its first
firing. Flagged 2026-08-26 as a possible control-validation failure; Antoine confirmed this
is expected behaviour. Compare the control to allmerged on the TARGET arms (which track to
within 2-11%); the non-target arms across that boundary are not a discrepancy.

### 8.10 COMPLETE: both 20k runs finished clean (2026-08-26)

| run | id | final ckpts | grids |
|---|---|---|---|
| `addsub-L18-16-nlpenalty` | `p-43e77281` | 17500, 20000 | 4000..20000, all five |
| `addsub-L18-16-nlcontrol` | `p-cbb66ad1` | 19000, 20000 | 4000..20000, all five |

Both rc=0, zero tracebacks end to end after the two ab_grid fixes.

**Control recovers allmerged at the endpoint.** PGD target arms at step 20000:
control 0.004539 / 0.003355 vs allmerged 0.004828 / 0.003413 (within 2-6%, control
slightly better). Together with the first-2000-step overlay (§8.9) the merged branch
reproduces the pre-merge trajectory at both ends, 18% faster and 22% lighter.

**The penalty's census effect is stable to the end.** gate_proj 42→33, up_proj 42→35
(the two `Neurons()`-partitioned sites, deficit ~-8 at every firing from 4000 to 20000);
unpenalized sites o_proj/down_proj drift both ways; totals 201 vs 184.

**The PGD story inverted twice and ended nuanced** (target/output arm, pen vs ctrl):
-7.8% @4k, +27.9% @8k, +65.9% @12k, +17.1% @16k, **+1.7% @20k**. The mid-run
faithfulness gap opened while `relative_threshold` sat at its 4.0 plateau and closed as
it annealed 4.0→1.0 over the second half (and the LR cosine wound down). At 20000:

| arm | control | penalty | gap |
|---|---|---|---|
| target/out | 0.004539 | 0.004617 | +1.7% |
| target/hidden | 0.003355 | 0.004333 | +29.2% |
| nontarget/out | 0.01158 | 0.01767 | +52.6% |
| nontarget/hidden | 0.01158 | 0.02031 | +75.4% |

So the end-state cost is concentrated in the HIDDEN role and the NON-TARGET stream —
the target/output arm, the one the run optimizes hardest, ends essentially free. Note
`train/loss/total` (0.09498 vs 0.04673) is NOT a like-for-like number: it includes the
weighted penalty itself (~0.027 at the end) on one side only.

**Nonlinearity term trajectory** (raw, coeff 1e-3): 92 @2600 -> 38 @8000 -> 25 @16000,
ticking back to 27 @20000 as the threshold anneal reached 1.0 and sharpened the count —
the term was still actively shaping at run end. The attention_head part (~1-1.4
throughout) is two orders below the neuron part; at these unit counts the head prior
barely binds.

Read: at coeff 1e-3 the prior delivers a persistent ~20% reduction in above-floor MLP
components at a real cost in non-target/hidden adversarial reconstruction. A coeff/
threshold-schedule sweep is the obvious next arm; snapshots for both runs are browsable
via each run dir's `ab_grids/index.html`.

### 8.11 The 4L arm: (192,144) is REFUSED on the merged branch; allocator envs are dead (2026-08-27)

First multi-block run on the merged branch. Smoke at the pre-merge production shape
(192,144) OOM'd in the FIRST train step: the executable's fused temp arena is 14.88 GiB
(empty op name = arena, and the byte count divides by no tensor shape) and the runtime's
BFC ("XLA_backend_N_bfc", a TF-lineage pool) cannot place it after the step-0 eval churns
the pool. The step-0 slow eval itself fits and passes at 28 sites, including the standing
nonlinearity eval.

**Four allocator knobs, four identical failures (jobs 10691/92/93/95):** the config's
`runtime.launch_env` platform allocator; the same exported at shell level;
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`; `TF_GPU_ALLOCATOR=cuda_malloc_async`.
jax 0.10.1's jaxlib parses XLA_PYTHON_CLIENT_* into plugin options (xla_client.py:177-208)
and the plugin binary carries the option keys, yet the failing allocator ignores them all
— and the pool bar (51% free, request refused) implies its LIMIT (~30 GB) ignores
MEM_FRACTION too. **Consequence: `launch_env.xla_python_client_allocator` is a silent
no-op on the merged branch's jax. The pre-merge 4L ladder's platform-allocator fix is
dead.** Worth raising with upstream/jax; do not trust launch_env allocator knobs until
someone finds where jax 0.10 actually routes them.

**What worked: shrink the demand.** Batch (128,96) — the probe ladder's own arm, and the
shape the original 4L header admits the LRs were tuned for — fits: smoke 10696, 60 steps,
**33.91 GB/rank**, rc=0. (It also carried TF_GPU_ALLOCATOR=cuda_malloc_async; whether that
contributed is unresolved — the production run keeps it for recipe identity.)

Production: `addsub-4L17-20-nlpenalty` (job 10701), 20k steps, penalty 1e-3, baseline
pre-merge `addsub-4L17-20-01` = p-b44674e1 (32.9 GB / 3.45 s/step at (192,144); 31.8 GB /
2.8 s/step at (128,96), fsdp:1). Merged step rate TBD from the first log windows.

### 8.12 The coefficient sweep's answer: 2e-3 doesn't break, it just pays more (2026-08-28)

`addsub-L18-16-nlpenalty2x` (p-3c8c727c) finished clean. Final table @20000:

| | control | 1x (1e-3) | 2x (2e-3) |
|---|---|---|---|
| gate_proj saved | 42 | 33 | 29 |
| up_proj saved | 42 | 35 | 31 |
| down_proj / o_proj | 80 / 31 | 76 / 35 | 78 / 35 |
| PGD target/out | 0.004539 | +1.7% | +15.8% |
| PGD target/hidden | 0.003355 | +29.2% | +38.7% |
| PGD nontgt/out | 0.01158 | +52.6% | +69.9% |
| PGD nontgt/hidden | 0.01158 | +75.4% | +92.0% |
| raw penalty term | — | 26.8 | 20.7 |

Findings:
1. **Dose-response is real but SUB-LINEAR**: the penalized-site deficit goes -16 -> -24
   components for 2x the coefficient. Stable at every firing from 4000 to 20000.
2. **Specificity holds at both doses** — unpenalized sites (down, o) show no reduction.
3. **No breakdown at 2e-3.** Trains stably to 20k; and the mid-run faithfulness spike is
   SMALLER than 1x's at the same steps (1x @12000: +66/+82% on target arms; 2x: +24/+31%).
   The spike tracks the relative_threshold anneal dynamics, not the coefficient — an
   argument for higher coeffs with gentler anneals rather than the reverse.
4. **The endpoint cost is where 2x pays**: 1x ends nearly free on target/out (+1.7%);
   2x pays +15.8%, and every other arm is ~+10-17 points worse than 1x.

Operating-point read: ~8 extra suppressed components for ~14 points of target/output
faithfulness — 1e-3 looks like the better default; 2e-3 is usable if write-locality is
the priority. The binding cost at BOTH doses is the non-target/hidden arm (tPD has no
faithfulness role to push back — the §8.6 caveat, measured).

Subspace scatter applets: building (job 10703) for control + 1x at step 20000; the 2x
can follow with `RUNS=p-3c8c727c sbatch subspace_applet_merged.sbatch`.

### 8.13 COMPLETE: addsub-4L17-20-nlpenalty, the first merged-branch multi-block run (2026-08-28)

`p-78852086` (job 10701) ran 20000 steps at 3.75 s/step on 4x L40 with zero failures.
Batch (128,96), penalty 1e-3, save_every 2500. ~20.8 h single leg.

**The census effect generalizes to 4 blocks.** Against the pre-merge baseline
p-b44674e1, aggregated over layers 17-20 (penalty/baseline):

| step | gate+up (penalized) | down | attn |
|---|---|---|---|
| 4000 | 161/163 (-2) | 155/141 | 16/15 |
| 8000 | 164/172 (-8) | 151/146 | 25/31 |
| 12000 | 171/191 (-20) | 157/160 | 49/44 |
| 20000 | 171/192 (**-21**) | 152/157 | 42/37 |

The baseline's penalized-site census GROWS 163->192 over training; the penalty run's
holds at ~171 — the prior suppresses growth of above-floor components on exactly its
sites, deficit stable from 12000. Per-layer @20000: L17 -5, L18 -11, L19 -4, L20 -1 —
concentrated where the mechanism is heaviest (L18), echoing the distributed-mechanism
weighting of the original 4L run.

**PGD @20000 is NOT attributable**: penalty reads +58-70% over baseline on all arms, but
this comparison stacks THREE differences (penalty, pre-merge vs merged code, batch
(128,96) vs (192,144) — and the eval batch differs with it, 96 vs 144, so even the
absolute probe values are not like-for-like). The L18 sweep's clean +1.7% endpoint says
most of this gap is probably NOT the penalty, but only a merged-code no-penalty 4L
control at (128,96) would settle it. Worth running if the 4L faithfulness number matters;
~21 h on 4 cards.

Artifacts: ckpts 17500/20000, ab_grids applet at runs/p-78852086/ab_grids/index.html,
wandb p-78852086.

### 8.14 Four-point sweep complete: the census effect SATURATES below 1e-3 (2026-08-29)

`addsub-L18-16-nlpenalty05x` (p-55ee815f) finished clean. Final table @20000, all four
arms (control / 0.5x / 1x / 2x):

| | control | 0.5x (5e-4) | 1x (1e-3) | 2x (2e-3) |
|---|---|---|---|---|
| gate_proj | 42 | 35 | 33 | 29 |
| up_proj | 42 | 32 | 35 | 31 |
| gate+up deficit | — | **-17** | **-16** | **-24** |
| down_proj / o_proj | 80/31 | 76/35 | 76/35 | 78/35 |
| PGD target/out | base | +5.7% | +1.7% | +15.8% |
| PGD target/hidden | base | +26.2% | +29.2% | +38.7% |
| PGD nontgt/out | base | +67.4% | +52.6% | +69.9% |
| PGD nontgt/hidden | base | +62.0% | +75.4% | +92.0% |

**Headline: half the dose produces the FULL 1x census effect.** 0.5x's deficit (-17) is
indistinguishable from 1x's (-16); only 2x (-24) pushes past. The dose-response is not
sub-linear — it is SATURATED between 5e-4 and 1e-3, with a second regime above.

PGD: 0.5x and 1x are within each other's firing-to-firing noise on every arm (target/out
+5.7 vs +1.7; the hidden/nontgt arms interleave). 2x is worse across the board.

Reading: at this recipe the prior's census effect turns on well below 5e-4 and plateaus;
1e-3 buys nothing over 5e-4. The knee is UNLOCATED — it is at or below 5e-4. A 0.25x
(2.5e-4) arm would bracket it. Both sub-1e-3 arms pay the same background faithfulness
cost (mostly the non-target/hidden arm), suggesting that cost is tied to HAVING the
prior, not its strength — consistent with the 2x mid-run finding (§8.12) that the
transient tracks anneal dynamics, not coefficient.

### 8.15 4L control complete: the attribution splits cleanly (2026-08-29)

`addsub-4L17-20-nlcontrol` (p-b7fd77f3, job 10710) ran 20k clean — and en route gave the
ab_grid fixes their first full-scale no-penalty exercise (all four PGD arms, five grid
firings, 28 sites, zero tracebacks).

**True 4L census effect = -35, not the confounded -21.** Matched-pair (only the penalty
differs): gate+up 206/171 at 20000, deficit -26 @4k -> -35 @12k, flat after. The
pre-merge-baseline comparison (§8.13) understated it by 40% and mistimed its onset —
the "slowly emerging deficit" was the baseline confound, not the physics.

**PGD @20000 decomposes** (penalty-vs-baseline was +58-70% on everything; now split):

| arm | penalty vs ctrl (= penalty effect) | ctrl vs pre-merge base (= code+batch) |
|---|---|---|
| target/out | +20.9% | +30.9% |
| target/hidden | +34.7% | +23.6% |
| nontgt/out | +63.8% | +3.5% |
| nontgt/hidden | +65.9% | +0.1% |

On the TARGET arms, roughly half the raw gap was code+batch (plausibly the batch: (128,96)
sees 33% fewer tokens than (192,144) at equal steps). On the NON-TARGET arms the penalty
owns essentially all of it. So at 4 blocks the prior's faithfulness cost is REAL and
larger than at L18 (target/out +21% vs +1.7%) — the census effect scales up
(-35 for 28 sites vs -16 for 7) but so does the price.

Fine-tune pair (nlft-keep / nlft-drop, §8.13 questions) starts on the freed cards.

### 8.16 Plots: nlpenalty dose-response (2026-08-29)

All figures in `plots/nlpenalty/`. Apples-to-apples throughout: the L18 family is
control/0.5x/1x/2x (p-cbb66ad1 / p-55ee815f / p-43e77281 / p-3c8c727c), all merged code,
same seed/recipe/batch, values at step 20000. The 4L family is the matched merged pair
(p-b7fd77f3 / p-78852086); the confounded pre-merge baseline appears in NO figure.
L18 and 4L are never mixed in one axes.

| family | coeff (×1e-3) | neuron units/comp | L0 total | alive total | KL rounded (tgt) | KL rounded (nt) | PGD tgt/out | PGD nt/hidden |
|---|---|---|---|---|---|---|---|---|
| L18 | 0 | 2179.0 | 22.0 | 201 | 0.003225 | 0.1182 | 0.004539 | 0.01158 |
| L18 | 0.25 | 44.0 | 23.1 | 189 | 0.00351 | 0.1182 | 0.00459 | 0.01549 |
| L18 | 0.5 | 29.4 | 24.0 | 183 | 0.003594 | 0.1181 | 0.004797 | 0.01876 |
| L18 | 1 | 20.1 | 24.1 | 184 | 0.00381 | 0.1181 | 0.004617 | 0.02031 |
| L18 | 2 | 17.6 | 25.0 | 178 | 0.003942 | 0.1182 | 0.005255 | 0.02223 |
| 4L | 0 | 2206.0 | 61.0 | 405 | 0.008983 | 0.506 | 0.01909 | 0.04059 |
| 4L | 1 | 40.2 | 64.9 | 365 | 0.009799 | 0.506 | 0.02308 | 0.06735 |

*(Sweep completed 2026-08-31 with 0.125x and 0.25x; all figures carry six points. The
"0.25x has the lowest non-target cost" claim made here on 2026-08-30 was read off a
SINGLE endpoint firing and does not survive multi-firing means — see §8.20.)*

**0) Does the penalty make components interface with fewer nonlinearities? Emphatically.**
Soft neuron count per component: **2179 -> 29 at the SMALLEST dose (74x)**, then 20 (1x)
and 18 (2x) — the census saturation of §8.14 was the tail of an effect that is nearly
complete at 5e-4. Attention heads: ~14 -> ~1-2 per component. Same shape at 4L (2206 -> 40).

![L18 neurons/component](plots/nlpenalty/l18_nonlinearity_neuron.png)
![L18 heads/component](plots/nlpenalty/l18_nonlinearity_attn.png)
![4L neurons/component](plots/nlpenalty/4l_nonlinearity_neuron.png)
![4L heads/component](plots/nlpenalty/4l_nonlinearity_attn.png)

**1) L0 — per stream, and the streams disagree.** Target stream: total L0 22 -> 24-25
(~10%, confined to gate_proj 4.9 -> 7.7). NON-target stream: gate/up L0 rises 4-5x with
the coefficient (gate 0.17 -> 0.82, up 0.16 -> 0.53 at 2x) while every unpenalized
matrix is flat — off the target distribution, the penalized components fire far more
promiscuously. This was invisible in the combined view.

![L18 L0 target](plots/nlpenalty/l18_l0_target.png)
![L18 L0 non-target](plots/nlpenalty/l18_l0_nontarget.png)
![4L L0 target](plots/nlpenalty/4l_l0_target.png)
![4L L0 non-target](plots/nlpenalty/4l_l0_nontarget.png)

**2) Alive components: the familiar census dip, ~9-11%**, on the penalized MLP sites
(grid census, stream-independent — the arithmetic-prompt grid).

![L18 alive](plots/nlpenalty/l18_alive.png)
![4L alive](plots/nlpenalty/4l_alive.png)

**3) Rounded-mask reconstruction: mild on target, ZERO on non-target.** Target-stream
rounded KL rises 11%/18%/22% at 0.5x/1x/2x; the non-target rounded KL is flat to four
significant figures in both families.

![L18 rounded KL target](plots/nlpenalty/l18_rounded_kl_target.png)
![L18 rounded KL non-target](plots/nlpenalty/l18_rounded_kl_nontarget.png)
![4L rounded KL target](plots/nlpenalty/4l_rounded_kl_target.png)
![4L rounded KL non-target](plots/nlpenalty/4l_rounded_kl_nontarget.png)

**4) PGD — per stream: target nearly flat until 2x; non-target carries the cost**
(+60-90% across both roles).

![L18 PGD target](plots/nlpenalty/l18_pgd_target.png)
![L18 PGD non-target](plots/nlpenalty/l18_pgd_nontarget.png)
![4L PGD target](plots/nlpenalty/4l_pgd_target.png)
![4L PGD non-target](plots/nlpenalty/4l_pgd_nontarget.png)

**Answer to the key question:** on the TARGET distribution the penalty is nearly free —
L0 ~flat, rounded KL +11-22%, PGD flat until 2x. Its entire cost lives OFF-distribution:
non-target gate/up L0 up 4-5x, non-target PGD +60-90% — the penalized components fire
more promiscuously and reconstruct worse on the broad stream. What it buys is a
~75-100x reduction in nonlinearities per component; the dose-response says 5e-4 already
delivers ~95% of that at the lowest cost. (0.25x arm running to bracket the knee.)

### 8.17 nlft-keep: more training does NOT reduce the PGD recon (2026-08-30)

`addsub-4L17-20-nlft-keep` (p-25223a32): 10k further steps from p-78852086@20000, penalty
kept, schedules held at parent end-state, LRs at the cosine floor. Batch (96,72), not the
parent's (128,96) — the S33 executable's memory plan holds the param cast and train arena
simultaneously (probes: restore itself is clean, 17.8 GiB) and (128,96) does not fit
this node's 45.4 GB-free cards; mild batch confound on absolute values.

| PGD arm | parent @20k | FT @4000 | FT @8000 (last slow eval) |
|---|---|---|---|
| target/out | 0.0231 | 0.0479 | 0.0312 |
| target/hidden | 0.0234 | 0.0551 | 0.0278 |
| nontgt/out | 0.0656 | 0.175 | 0.102 |
| nontgt/hidden | 0.0674 | 0.238 | 0.0995 |

The FT never beat the parent on any arm. S33's FRESH adversaries caused a 2-3.5x
robustness collapse by step 4000 (the restored state itself evaluated at parity at step
0), recovering toward — not past — parent by 8000. Census: flat throughout (374/375 vs
parent 365). Read: the parent's elevated PGD is the objective's equilibrium, not
undertraining; and fine-tune adversary resets cost ~8k steps of transient vulnerability.

### 8.18 nlft-drop: locality is MAINTAINED, not locked in (2026-08-30)

`addsub-4L17-20-nlft-drop` (p-7aac5314): the keep twin minus the penalty, same restore,
same (96,72). Ran clean, ckpts 7500/10000.

| | parent @20k | keep-FT @8k | drop-FT @4k | drop-FT @8k | control @20k |
|---|---|---|---|---|---|
| soft neurons/comp | 40 | 32 | 1237 | 1270 | 2206 |
| census total | 365 | 375 | 373 | 376 | 405 |
| PGD target/out | 0.0231 | 0.0312 | — | **0.0215** | 0.0191 |
| PGD nontgt/hidden | 0.0674 | 0.0995 | — | 0.0999 | 0.0406 |

Three findings:
1. **The penalty's defining property does not survive its removal.** Neurons/component
   relaxes 40 -> ~1250 within 4000 steps of lifting the pressure — then PLATEAUS at ~57%
   of the never-penalized level (1270 @8k, vs control 2206). Partial hysteresis, mostly
   relaxation: the reconstruction objective actively prefers delocalized writes.
2. **The census is blind to this.** Alive counts stay parent-shaped (373/376) while the
   structure they were proxying dissolves — consistent with §8.16's finding that the
   census understates the locality effect by orders of magnitude.
3. **Dropping the penalty buys back the target-stream adversarial cost**: drop@8k
   target/out 0.0215 beats the parent (0.0231) and keep@8k (0.0312). The non-target arm
   stays at the adversary-reset transient level both twins share.

Consequence: the prior is a MAINTENANCE term — it belongs in the objective for the whole
run (cheap at 2.5e-4, per §8.16), not as a phase that can be annealed away or applied
post-hoc. And any analysis consuming a penalized decomposition should use the checkpoint
trained WITH the penalty, not a continuation.

### 8.19 Fine-tunes: CI-masked and rounded-mask reconstruction (2026-08-30)

Fast-tier CE/KL evals over both fine-tunes, anchors = parent p-78852086 @20k and control
p-b7fd77f3 @20k. Rounding costs nothing anywhere (kl_rounded ≈ kl_ci to <2% throughout),
so the pairs move together.

**Target stream** (kl_ci_masked / kl_rounded_masked):

| step | keep | drop | anchors |
|---|---|---|---|
| 2000 | 0.0130 / 0.0127 | 0.0132 / 0.0129 | parent 0.00986 / 0.00980 |
| 6000 | 0.0120 / 0.0118 | 0.0113 / 0.0111 | control 0.00924 / 0.00898 |
| 10000 | 0.0103 / 0.0102 | **0.00968 / 0.00954** | |

1. **Both twins START ~30% worse than the parent ended** — the S33 optimizer reset costs
   masked recon too, not just adversarial robustness (fresh Adam at the floor LR takes
   the whole 10k to re-approach). Shared by both twins, so it is the reset, not the
   penalty.
2. **drop recovers faster and ends BETTER than the parent** (0.0095 vs 0.0098), touching
   the control's level — lifting the penalty frees capacity for masked recon, the
   on-distribution counterpart of §8.18's PGD buy-back.
3. **keep converges slower** (0.0102 @10k, ~4% above parent): the maintained penalty
   gradient competes with the recon gradient at the floor LR.

**Non-target stream:** keep and drop are IDENTICAL to 3 decimals at every step
(0.449-0.560, oscillating around 0.506), and parent = control = 0.506. Two conclusions:
the penalty has ZERO effect on non-target masked/rounded recon (confirming §8.16 at the
fine-tune scale), and the oscillation is eval-draw noise common to both twins, not model
movement. The broad-stream masked recon simply does not see any of this.

### 8.20 Sweep complete (six points): the benefit scales with dose, the cost does NOT (2026-08-31)

`addsub-L18-16-nlpenalty0125x` (p-240775bb) finished clean, closing the L18 dose-response
at 0 / 0.125x / 0.25x / 0.5x / 1x / 2x — all merged code, one seed, same recipe, step
20000.

**Locality (the benefit) is strongly dose-dependent**, neurons/component:
2179 -> 63 -> 44 -> 29 -> 20 -> 18. Even an EIGHTH of the default gives 34x. Each
doubling above 1.25e-4 buys roughly another 1.3-1.4x — diminishing but never flat.

**Census (alive components) needs more dose than locality does.** gate+up deficit vs
control: -6 (0.125x), -17 (0.25x), -17 (0.5x), -16 (1x), -24 (2x). At 0.125x the census
is within noise of control (195 vs 201) while locality is already 34x better — the two
effects are DISSOCIATED, which the fine-tunes showed from the other direction (§8.18:
census intact while locality relaxes).

**A methodological correction.** The non-target PGD arm swings ~2x between firings within
a single run — every arm spikes at step 16000 (control 0.0171, 0.25x 0.0240, 1x 0.0291,
2x 0.0329) and most fall back by 20000. Ranking doses on one endpoint firing is not
sound; the 0.125x endpoint (0.0313) is the highest of any arm purely because it did not
fall back. Means over the last four firings (steps 8000-20000):

| arm | PGD tgt/out | PGD nt/out | PGD nt/hidden |
|---|---|---|---|
| control | 0.00518 | 0.01305 | 0.01316 |
| 0.125x | 0.00599 | 0.02259 | 0.02171 |
| 0.25x | 0.00613 | 0.01846 | 0.01770 |
| 0.5x | 0.00633 | 0.02011 | 0.01924 |
| 1x | 0.00673 | 0.02107 | 0.02073 |
| 2x | 0.00637 | 0.01987 | 0.02254 |

**The faithfulness cost is essentially DOSE-INDEPENDENT.** Every penalized arm sits
~1.4-1.7x control on the non-target arms and +15-30% on target/out, with no monotone
ordering and a between-dose spread (0.0177-0.0226 on nt/hidden) comparable to the
within-arm noise. Turning the prior ON costs what it costs; turning it UP mostly does not
cost more. This corroborates §8.12's finding from the 2x mid-run transient (the excursion
tracked the threshold anneal, not the coefficient).

**Operating-point read, revised.** Since the cost is a step function of "prior on?" and
the benefit keeps improving with dose, there is no cost argument for staying low — the
argument for a small coefficient is only that returns diminish. 2.5e-4 to 1e-3 is a
reasonable band; 1.25e-4 gives up census effect entirely for no faithfulness saving and
is NOT recommended. A seed replicate is the missing evidence: every statement here rests
on one seed per dose, and the noise floor measured above is uncomfortably close to the
between-dose differences on the cost metrics.

### 8.21 Penalized decompositions are markedly more attack-init sensitive (2026-09-01)

Prompted by a suspicion that the PGD endpoints were noisy. Design: each finished L18
checkpoint re-evaluated with the PRODUCTION probe (`make_fresh_pgd_step`, unmodified),
16 fresh-PGD inits, on ONE batch held fixed across every init AND every checkpoint
(pass 40, the production endpoint pass). Model fixed, batch fixed — the whole spread is
attack-init sensitivity. Harness: `pgd_replicates.py` / `pgd_replicates.sbatch`.

**First, the noise decomposition** (control, 4-batch production metric): init alone gives
cv 1.9-2.3% on the non-target arms, against cv 20.8-24.7% when the batch draw is also
allowed to vary. So the eval noise that made single-firing dose rankings unsound (§8.20)
is BATCH noise, not init noise, and the fix for cross-run comparisons is a shared batch,
not more inits.

**Then the finding.** cv over 16 inits, single fixed batch (95% chi-square CI, n=16):

| coeff (×1e-3) | target/out | target/hidden | nontgt/out | nontgt/hidden |
|---|---|---|---|---|
| 0 | 4.5% | 5.6% | 3.2% | 3.4% |
| 0.125 | 15.2% | 15.5% | 7.7% | 14.7% |
| 0.25 | 16.4% | 18.8% | 12.5% | 11.3% |
| 0.5 | 17.4% | 17.2% | 21.5% | 14.7% |
| 1 | 14.5% | 15.0% | 14.3% | 20.2% |
| 2 | 19.7% | 20.0% | 38.9% | 33.5% |

Full table with CIs: `plots/nlpenalty/init_sensitivity_table.md`.

1. **Penalized vs unpenalized is a step change, not a gradient.** The control's cv
   (3.2-5.6%) sits below every penalized arm's 95% CI lower bound on all four arms. Even
   the smallest dose triples-to-quadruples it. This is a property of the MODEL, not the
   measurement: identical batch, identical probe.
2. **Among penalized arms, the trend is dose-dependent only OFF-distribution.**
   Spearman cv-vs-coefficient over the five penalized doses: non-target/output rho=+0.90
   (p=0.037), non-target/hidden rho=+0.90 (p=0.037); target arms rho=+0.40/+0.30
   (p=0.51/0.62, i.e. flat within noise at 15-20%).
3. **Interpretation.** The prior appears to roughen the adversarial landscape: with
   component writes concentrated on few nonlinearities, where the attack starts matters
   much more, and off-distribution it matters progressively more with dose. Practical
   consequence: for a penalized decomposition a single-init PGD number is a sample from a
   wide distribution — at 2e-3 the non-target arm ranges 0.0197-0.0594 across inits, a
   3x spread. Report a multi-init mean, or the max if the worst case is what matters.

![L18 init sensitivity, target](plots/nlpenalty/l18_init_sensitivity_target.png)
![L18 init sensitivity, non-target](plots/nlpenalty/l18_init_sensitivity_nontarget.png)

Caveat: one batch, one seed per dose. The control-vs-penalized step is far larger than
the CIs and is safe; the within-penalized trend rests on 5 points and one Spearman test
per arm, so treat p=0.037 as suggestive rather than established.

### 8.22 PGD loss vs number of adversarial steps: the 20-step metric understates the cost (2026-09-02)

Prompted by the possibility that a run looking worse at 20 ascent steps converges better
at 100. It does not — the opposite. Design: production probe rebuilt at `n_steps` =
5/10/20/40/80, SAME init and SAME 4 batches for every step count and every checkpoint
(sign-PGD from a fixed init is deterministic, so the k-points trace one ascent).
Harness `pgd_trajectory.py`; values in `plots/nlpenalty/adv_steps_table.md`.

![L18 adversarial steps, target](plots/nlpenalty/l18_adv_steps_target.png)
![L18 adversarial steps, non-target](plots/nlpenalty/l18_adv_steps_nontarget.png)

**The control saturates; penalized runs do not.** Non-target/hidden, control:
0.0114 (k=20) -> 0.0120 (k=80), +5%. Penalized at 1e-3: 0.0214 -> 0.0806, **+277%**.
Ratio to control grows from 1.9x at k=20 to 6.7x at k=80. Every penalized dose keeps
climbing where the control is flat by k=40.

| coeff (×1e-3) | nontgt/hidden k=20 | k=80 | ratio to control @80 |
|---|---|---|---|
| 0 | 0.0114 | 0.0120 | 1.0x |
| 0.125 | 0.0233 | 0.0422 | 3.5x |
| 0.25 | 0.0187 | 0.0303 | 2.5x |
| 0.5 | 0.0252 | 0.0455 | 3.8x |
| 1 | 0.0214 | 0.0806 | 6.7x |
| 2 | 0.0355 | 0.0752 | 6.3x |

**Consequences.**
1. **The standard 20-step probe UNDERSTATES the penalty's cost**, by ~3.5x at 1e-3 on
   the non-target/hidden arm. Every "+50-90%" figure in 8.16/8.20 is a lower bound on
   what a stronger attacker finds.
2. **No crossover.** No dose is worse at 20 and better at 80; the ordering is stable and
   the gaps widen. The 20-step ranking is directionally right, just compressed.
3. **The vulnerability needs budget to find.** At k=5 penalized and control are within
   3-15% on the non-target arm (except 2e-3); the break is between k=10 and k=20. A
   weak attacker sees a decomposition that looks nearly as robust as the control.
4. **The target stream behaves differently**: penalized curves saturate like the control
   (1e-3: +25% at k=80, converged by k=40), and dose ordering there is non-monotone
   (0.5x reaches +49%, above 1e-3's +25%). The divergence is an OFF-distribution
   phenomenon, consistent with 8.16 and 8.21.

Caveat: one init, one batch set per point (per the requested design), so a single curve
carries the 8.21 init spread (cv 15-39% on penalized non-target arms). The control-vs-
penalized separation at k>=40 is far larger than that; between-dose ordering at high k
is not.

### 8.23 Why penalized runs diverge under a stronger adversary: norm inflation, not deadness or cancellation (2026-09-02)

Antoine's hypothesis: dead components keep non-zero weights when the penalty is on, so
the adversary can switch them on and is effectively stronger. Two weight-space tests over
all six L18 checkpoints (`dead_component_norms.py`, `component_geometry.py`; both are
checkpoint arithmetic, the first plus one forward for CI).

Per component the switchable scale is `||V_c|| * ||U_c||` — the rank-1 update the mask can
turn on. DEAD = max CI over batch x positions < 0.1, i.e. never fires (a MEAN-CI threshold
is useless here: CI is sparse, L0 ~22 of thousands, so it marks ~99% dead).

| coeff (×1e-3) | dead norm | alive norm | total | dead share | MLP coherence | PGD nt/hidden @k=80 |
|---|---|---|---|---|---|---|
| 0 | 372 | 1014 | 1386 | 26.8% | 0.051 | 0.0120 |
| 0.125 | 937 | 1817 | 2754 | 34.0% | 0.046 | 0.0422 |
| 0.25 | 1042 | 1973 | 3015 | 34.6% | 0.048 | 0.0303 |
| 0.5 | 1104 | 2159 | 3263 | 33.8% | 0.050 | 0.0455 |
| 1 | 1196 | 2340 | 3536 | 33.8% | 0.052 | 0.0806 |
| 2 | 1214 | 2519 | 3733 | 32.5% | 0.056 | 0.0752 |

1. **Dead components DO carry much more weight — 372 -> 1214, a 3.3x increase.** The
   hypothesis is directly supported in absolute terms, and it correlates with the
   divergence (Pearson r=+0.83, p=0.042 vs PGD at k=80).
2. **But it is not SPECIFIC to dead components.** The dead SHARE is flat at 32-35% for
   every penalized dose (up from the control's 27%, then no trend), and alive norm grows
   in lockstep (1014 -> 2519, 2.5x) with an equal-or-better correlation to the divergence
   (r=+0.89, p=0.019). What the penalty does is inflate the WHOLE switchable norm ~2.7x;
   dead components inherit their usual ~1/3 of it.
3. **Cancellation fragility is ruled out.** Coherence
   `||sum_c V_c U_c^T|| / sum_c ||V_c U_c^T||` is flat at 0.046-0.056 across all six runs
   with no trend — penalized decompositions are no more cancellation-dependent than the
   control.

**Mechanism this supports.** The penalty is SCALE-INVARIANT by construction (S36 divides
by `||U_c||^2`), so it exerts no shrinking pressure, while forcing each component's write
onto few units makes reconstructing a dense `W` need more total norm. The adversary's
budget is proportional to that norm, so a bigger-normed decomposition is a strictly
stronger attack surface — and finding the useful subset takes ascent steps, which is
exactly the k>=20 divergence of 8.22.

**Testable follow-up if this matters:** pair the prior with an explicit norm penalty (or
raise `ci_scaled_weight_decay`) so locality is bought without inflating `||V||*||U||`. If
the PGD divergence tracks total norm rather than locality per se, that should remove most
of the adversarial cost while keeping the 8.16 locality gain.

Caveat: n=6 checkpoints, one seed each; the correlations above have 4 dof and cannot
separate dead/alive/total norm (they are collinear, all rho=+0.89).

### 8.24 The L18-17 series: the divergence is the PENALTY, not the hidden pass (2026-09-02)

Replication of 8.22 on the ntmerged-based L18-17 series, plus the output-only run.
Output-role arms throughout (`ci.dual: false` runs have no hidden readout), same fixed
init and batches as 8.22, `pgd_trajectory.py`. Values:
`plots/nlpenalty/l1817_adv_steps_table.md`.

![L18-17 adversarial steps, target](plots/nlpenalty/l1817_adv_steps_target.png)
![L18-17 adversarial steps, non-target](plots/nlpenalty/l1817_adv_steps_nontarget.png)

Non-target/output, ratio to the L18-16 control at each k:

| run | hidden role | penalty | k=20 | k=80 | x ctrl @80 |
|---|---|---|---|---|---|
| L18-16 control | dual | — | 0.0113 | 0.0119 | 1.00x |
| L18-16 0.5x (allmerged) | dual | 5e-4 | 0.0212 | 0.0416 | 3.51x |
| L18-17 nl5e4 | dual | 5e-4 | 0.0133 | **0.0732** | **6.16x** |
| L18-17 output-only | single | — | 0.0100 | 0.0113 | 0.95x |
| L18-17 output-only + 5e-4 | single | 5e-4 | 0.0151 | 0.0483 | 4.07x |

1. **The 8.22 divergence replicates across recipes.** ntmerged + 5e-4 shows the same
   flat-then-explode shape and ends at 6.2x control, above the allmerged 0.5x arm's
   3.5x. Not an artifact of the merged stochastic+PPGD recipe.
2. **The hidden pass is NOT the cause.** Output-only + 5e-4 still diverges (4.1x,
   0.0151 -> 0.0483 from k=20 to k=80) with no hidden CI role at all. The penalty alone
   suffices — consistent with 8.23's norm-inflation mechanism, which is weight-space and
   role-independent.
3. **Output-only WITHOUT the penalty is the most adversarially stable decomposition
   measured**: 0.95x control on the non-target arm, saturated by k=40 — flatter than the
   control itself. Dropping the hidden role costs target-stream reconstruction (worst at
   low budget, 0.0061 at k=5 vs control 0.0034) but that curve then converges (+68% at
   k=80) instead of running away.
4. **Target stream: penalty and role both cost, and they add.** At k=80: control 0.00454,
   output-only 0.00761 (+68%), nl5e4 0.00798 (+76%), output-only + 5e-4 0.01039
   (**+129%**, the worst of the five).

**Reading.** The off-distribution runaway tracks the penalty in every configuration
tested (5 runs, 2 recipes, 2 role settings) and never appears without it. Output-only
trades a uniformly higher but BOUNDED target-stream error for the elimination of
non-target fragility — the opposite trade from the penalty, which buys locality at the
price of an unbounded-looking off-distribution tail.

Caveats: one init and one batch set per curve (8.21 init cv 15-39% on penalized
non-target arms); recompilation alone moves a repeated measurement ~4.5% (the L18-16
control's k=20 read 0.00431 here vs 0.00422 in 8.22). The 6.16x vs 4.07x vs 3.51x
ordering among penalized runs is within that combined uncertainty; the
penalized-vs-unpenalized separation is not. No merged-code ntmerged run WITHOUT the
penalty exists, so L18-17 nl5e4's baseline is borrowed from the L18-16 line.

### 8.25 Dead vs alive breakdown: the divergence is an INTERACTION (2026-09-03)

Restricting the fresh-PGD adversary to one component group and re-measuring the
loss-vs-adversarial-steps curve. Masks compose as `mask = ci + (1-ci)*source`, so a
component frozen at source 0 keeps its natural CI and is effectively unattacked; DEAD =
max CI over the fixed batches < 0.1. Same init/batches as 8.22/8.24
(`pgd_group_ablation.py`); the `all` setting reproduces the production probe as a check.
Non-target/output arm. Values: `plots/nlpenalty/groupabl_table.md`.

![control](plots/nlpenalty/l18_groupabl_cbb66ad1.png)
![nl5e4](plots/nlpenalty/l18_groupabl_118386d3.png)
![output-only + 5e-4](plots/nlpenalty/l18_groupabl_204fa1bc.png)

| run | alive/total | setting | k=20 | k=80 | k20->k80 |
|---|---|---|---|---|---|
| control | 832/1952 | all | 0.0118 | 0.0122 | 1.04x |
| | | alive | 0.0076 | 0.0080 | 1.05x |
| | | dead | 0.0062 | 0.0063 | 1.01x |
| nl5e4 (dual) | 670/1952 | all | 0.0132 | 0.0285 | **2.15x** |
| | | alive | 0.0087 | 0.0093 | 1.07x |
| | | dead | 0.0066 | 0.0067 | 1.01x |
| output-only + 5e-4 | 1104/1952 | all | 0.0146 | 0.0619 | **4.24x** |
| | | alive | 0.0068 | 0.0071 | 1.04x |
| | | dead | 0.0076 | 0.0139 | 1.83x |

1. **Neither group alone reproduces the divergence.** In both penalized runs the
   alive-only attack saturates by k=20 (1.04-1.07x from k=20 to k=80) — flat, exactly
   like the control. Dead-only is flat too for nl5e4 (1.01x); it grows somewhat for
   output-only + 5e-4 (1.83x) but still reaches only 0.0139 against 0.0619 unrestricted.
2. **The effect is super-additive, and the excess tracks the divergence.** At k=80,
   alive+dead summed vs the joint attack: control 0.0143 vs 0.0122 (**0.86x** — SUB-additive,
   the two groups partly substitute), nl5e4 0.0160 vs 0.0285 (**1.78x**), output-only +
   5e-4 0.0210 vs 0.0619 (**2.95x**). The penalty turns a sub-additive attack surface
   into a strongly super-additive one, and the super-additivity ranks the runs the same
   way the divergence does.
3. **Reading.** The adversary must switch dead components ON *while* perturbing the live
   circuit; neither move alone is damaging. That explains why the divergence needs many
   ascent steps (8.22) — a coordinated subset is a harder search — and it refines 8.23:
   the extra dead-component norm is a necessary ingredient, not a sufficient one.

**Caveat, and a correction to 8.24's error budget.** The `all` curves here reproduce
8.24 at k=20 (within 0.6-3%) but NOT at k=80: nl5e4 reads 0.0285 here vs 0.0732 there
(2.6x), output-only + 5e-4 0.0619 vs 0.0483 (1.28x). Same model, batches and init seed —
only the compilation differs. So the penalized k>=40 measurements are far less stable
than 8.24's stated ~4.5%: at high budget the attack appears to be multi-modal, and a
single curve can land in a much worse or much better basin. The QUALITATIVE claims
(penalized diverges, control saturates, restricted attacks saturate) hold in every
measurement; the specific high-k ratios (6.16x etc.) should be treated as one draw, not
a point estimate. Quantifying this properly needs the 8.21 multi-init treatment applied
at k=80.

### 8.26 CORRECTION: "non-target stream" in 8.16-8.23 means the HIDDEN CI head (2026-09-03)

Antoine flagged that the arm labels were ambiguous. To be exact about what those numbers
are, and what was re-measured:

`make_fresh_pgd_step(..., role=...)` picks **which CI readout head builds the mask floor**
(`mask = ci_lower(role) + (1 - ci_lower(role)) * source`). It does NOT change what is
reconstructed: `hidden_acts_reconstruction` is unset in every sweep config, so the loss is
always end-to-end OUTPUT KL. So `nontarget/hidden` = *output reconstruction, non-target
stream, masks gated by the hidden head* — never hidden-activation reconstruction.

Still, gating by a differently-trained head is a different adversarial problem, and 8.22's
figures mixed heads across doses (control and 0.5x on the output head, the rest on the
hidden head). Those figures are WITHDRAWN. Re-harvested with `pgd_curve.py`, output head
only, both streams, and the conclusion is unchanged and now clean:

| penalty (×10⁻³) | PGD general text, k=20 | k=80 | ×control @80 |
|---|---|---|---|
| 0 | 0.0111 | 0.0119 | 1.00x |
| 0.125 | 0.0182 | 0.0452 | 3.79x |
| 0.25 | 0.0188 | 0.0309 | 2.60x |
| 0.5 | 0.0192 | 0.0540 | 4.53x |
| 1 | 0.0197 | 0.0722 | 6.06x |
| 2 | 0.0279 | 0.0784 | 6.58x |

Baseline saturates (+7% from k=20 to k=80), every penalised dose keeps climbing, and the
ratio is now monotone in dose — cleaner than the hidden-gated version.

Two infrastructure findings from the re-harvest:
1. `open_jax_run` builds its OWN mesh — `hsdp_mesh(1, n_devices, 1)` with `zero1` —
   regardless of how the run trained. On this box that layout needs ~26.5 GiB plus a
   ~21.9 GiB transient and OOMs on 2 cards, while the run's own `ddp` / fsdp-1 layout ran
   the same probe in its slow eval at 26.8 GiB and is ~4x faster per step. Forcing the
   training layout is what made 2-GPU (and hence parallel) harvesting possible. Consumer
   memory does NOT follow from training memory — this bit three times.
2. Recording the loss inside the ascent (one 80-step run yielding every k) is illegal
   under SPMD: `io_callback` cannot carry a replicated sharding. Hence the per-k sweep,
   which costs sum(k) rather than max(k).

The shareable write-up built on this data is `notes/nonlinearity_penalty.md`.

### 8.27 Per-stream figures, output-only replication, second adversary start (2026-09-03)

Three questions closed on the output-head harvest from 8.26. Jobs 11029-11032 (2x L40 each,
~58 min per 3-run job, all COMPLETED, zero error lines). Seed 1234 output lives in
`pgd_curve/`, seed 5678 in `pgd_curve_s5678/`; `pgd_curve.sbatch` now takes `INIT_SEED` and
`OUTDIR` env vars so the two never collide.

**(a) The share figures were mixing streams silently.** Audit of the 8-figure set committed
in `39c02a572`: `01` (nonlinearity soft-unit count) is weight-space and belongs to neither
stream; `02` (L0) and `05` (rounded recon) read `eval/l0/...` and `eval/ce_kl/...`, i.e.
TARGET only, with no label saying so; `03` (alive components) is an AB-grid census over the
task prompt grid. Only the four PGD figures were per-stream. Rebuilt as 12 figures with one
per stream wherever the quantity has a stream, and an on-plot note where it doesn't.

Two findings that were invisible in the target-only set:

- `eval/nontarget_data/ce_kl/kl_rounded_masked` is **0.1182 at every dose**, flat to four
  digits, against 0.0032 -> 0.0039 on the target stream. Off-distribution, ordinary
  reconstruction does not notice the penalty at all.
- `eval/nontarget_data/l0/0.0_*` total goes **0.90 -> 1.94** across the sweep (gate alone
  0.175 -> 0.82, 4.7x) while target L0 is flat at 22 -> 25. The penalty leaves components
  materially more willing to fire on text they do not explain. This is the first mechanism
  candidate for the 8.22/8.24 off-distribution runaway that is measurable from the training
  logs alone, and it is dose-monotone, which the PGD numbers are not.

**(b) The runaway is not an artefact of the hidden-acts objective.** Harvested the L18-17
pair (`p-f9417595` output-only control, `p-204fa1bc` output-only + 5e-4) plus `p-118386d3`
(dual + 5e-4, same lineage). At 80 steps on general text the penalty costs 4.98x / 3.06x
(output-only) against 4.53x / 3.19x (dual, `p-cbb66ad1` -> `p-55ee815f`) at the two starts.
Same size. Separately: output-only decompositions are less adversarially robust on the task
stream to begin with (0.0070 vs 0.0043 at k=20) — the hidden-acts term buys on-task
robustness and does nothing for the off-distribution cost.

**(c) Second adversary start replicates the shape, not the ranking.** Ratio to control at
k=80, general text, seed 1234 / seed 5678: 0.125x -> 3.79/3.85, 0.25x -> 2.60/3.62,
0.5x -> 4.53/3.19, 1x -> 6.06/6.15, 2x -> 6.58/5.47. Flat baseline and a 3-6x penalised
envelope rising with dose in both; individual doses move up to 40%. `p-118386d3` (the dual
5e-4 repeat from a different training seed/lineage) gives 3.76x / 1.99x against
`p-55ee815f`'s 4.53x / 3.19x — **training seed moves the magnitude about as much as attacker
start does**. Consequence for anything downstream: quote this cost as a range, never as a
point estimate; 8.24's single-init ratios should be read the same way.

Share doc `notes/nonlinearity_penalty.md` rewritten around this: 7 sections, 12 figures,
every adversarial figure a two-panel start-point-1 / start-point-2 comparison, and the
caveat list now states the seed and lineage limits explicitly.

### 8.28 Config audit of every within-figure comparison (2026-09-04)

Antoine asked whether the runs sharing a plot differ *only* by the nonlinearity penalty. Ran
a full launch-config diff (list-of-dicts re-keyed by `type` first — a naive index-based diff
is useless here, because inserting `NonlinearityLocalityLoss` into `pd.loss_metrics` shifts
every later index and manufactures ~35 phantom diffs).

**Sweep, figures 01-10 (p-cbb66ad1 / 240775bb / 54078d02 / 55ee815f / 43e77281 / 3c8c727c):
CLEAN.** 20 differing keys, all accounted for: 18 are the `NonlinearityLocalityLoss` block
(absent in the control; byte-identical across the five penalised runs apart from
`coeff.max_val`, which IS the swept variable), plus `run_name`, plus
`cadence.checkpointing.save_every` (1000 on the control, 2500 on the rest). save_every is
checkpoint cadence only and 20000 is a multiple of both, so it touches nothing. seed=0
everywhere; all nine runs harvested at step 20000, n_batches 4, pass_index 40.

**Output-only pair (p-f9417595 / p-204fa1bc): CLEAN.** 19 diffs = the penalty block +
run_name.

**Dual-vs-output-only WITHIN the -17 lineage (p-118386d3 / p-204fa1bc): CLEAN.** 75 diffs,
every one of them `decomposition.ci.dual` or machinery that only exists when it is true
(`pd.hidden.*`, `nontarget.hidden.*`, the two hidden eval metrics). This is a valid
isolation of the hidden-acts objective.

**BUT -16 vs -17 is NOT a seed replicate, and the doc said it was.** The two lineages train
DIFFERENT reconstruction losses:
  -16: `MergedStochasticSubsetPPGDReconLoss` (output 1.5, hidden 3.0) + Unmasked 0.5
  -17: `PersistentPGDReconLoss` (0.5 / 1.0) + `StochasticReconSubsetLoss` (1.0 / 2.0) + Unmasked 0.5
Seeds are identical (0) across all nine. So 8.27's claim that "training seed moves the
magnitude about as much as attacker start does" was **wrong on attribution** — the training
RECIPE does. Corrected in the share doc.

Two consequences, both fixed:

1. Old figures 11/12 plotted -16 and -17 curves on shared axes, inviting a level comparison
   that the recipe difference does not license. Replaced with **self-normalised cost curves**
   — each pair divided by its own penalty-off control, so the recipe cancels. The two pairs
   then land on top of each other (80 steps, general text: 4.53x/3.19x dual vs 4.98x/3.06x
   output-only), which is a strictly stronger statement of the section's claim than the raw
   curves were. Figures 11 and 12 now share a y-range so flat-vs-runaway reads at a glance.
2. The old claim "output-only decompositions are less adversarially robust to begin with
   (0.0070 vs 0.0043)" compared p-f9417595 (-17) against p-cbb66ad1 (-16) — confounded by
   recipe. Restated using the clean within-lineage pair p-118386d3 vs p-204fa1bc, both 5e-4,
   both -17: dropping the hidden-acts objective costs **1.49x / 1.76x** at k=20 on the task
   stream (0.0065 -> 0.0098, 0.0052 -> 0.0092). The claim survives with a valid comparison.

Lesson for the next write-up: run the type-keyed config diff BEFORE asserting "everything
else is identical". The sweep earned that sentence; the cross-lineage runs never did.

### 8.29 Per-block replication on the 4L17-20 pair (2026-09-05)

Independent replication of 8.22/8.27's adversarial-step result, on a decomposition that
shares neither the block nor the loss recipe with the L18-16 sweep.

**Pair:** `addsub-4L17-20-nlcontrol` (p-b7fd77f3) vs `addsub-4L17-20-nlpenalty`
(p-78852086), 1e-3, blocks 17-20, 28 sites, `PersistentPGD` + `StochasticReconSubset`
recipe. Type-keyed config diff: 19 keys, 18 of them the `NonlinearityLocalityLoss` block,
plus `run_name`. Clean.

**Block-restricted probe** (`pgd_curve_block.py`, `--blocks`). While block N is attacked the
other sites are pinned every ascent step to component source 1 AND delta source 1, so
mask == 1 and the site sums back to the ORIGINAL matrix. The first draft pinned them to
source 0 (mask == ci_lower), which ablates every component with CI < 1 across three quarters
of the network -- Antoine caught it. Delta semantics are unchanged from production: pinned
1.0 throughout the non-target arm, adversarially optimised on the target arm.

**Result -- penalty cost (on/off), general text:**

    block   k=5    k=10   k=20   k=40   k=80
    17      1.33   1.58   1.97   2.25   2.47
    18      1.07   1.32   1.76   2.15   2.52
    19      1.15   1.31   2.08   2.72   2.83
    20      1.12   1.35   1.82   2.40   2.95
    all 4   1.07   1.20   1.60   2.25  12.31

Task distribution: flat, 1.00-1.16 at every k for every block. So the off-distribution
runaway reproduces at FOUR separate depths, with a control that saturates in each
(e.g. block 18 off: 0.0055 -> 0.0091 over k=5..80, on: 0.0059 -> 0.0228).

**Superadditivity.** All four blocks attacked jointly reach 12.31x, >4x the best single
block. The exposure the penalty leaves is distributed, not localised -- consistent with
[[multi-block-targeted-runs]]'s finding that the addsub mechanism spans the network.

**Free reproducibility check.** Block 20 was harvested twice by two sessions with identical
settings (seed 1234, nb 4, pass 40). Target arm agrees to <1.3%; NON-TARGET arm differs by
up to **8.9%**. Same seed, same batches, same code -- so ~9% is this probe's own noise floor
on the non-target arm, presumably bf16 reduction order. Ratios of 2.5-3x clear it; do not
report non-target differences below ~10% as real.

**Two launcher traps burned time here.** (1) `sbatch --export` uses commas as its OWN
separator, so `BLOCKS="17,18,19,20"` silently arrives as `BLOCKS=17` -- three jobs looked
like deliberate single-block runs when they were truncated. `pgd_curve.sbatch` now takes a
colon-separated list and translates it. (2) Each harvest process writes only its own
`results` dict, so two block-sets sharing an output filename silently erase each other;
every block set now gets its own OUTDIR and the figure script globs `pgd_block*` and merges.

Concurrency note: two other sessions were submitting into the same scratch tree during this
work. Check `squeue` and the JSON `blocks`/`init_seed` metadata before assuming a file is
yours.

### 8.30 Second adversary start for the joint 4-block attack (2026-09-05)

8.29's superadditivity number (12.31x) rested on one adversary start. Job 11079 repeats the
all-blocks-at-once harvest on the same pair at `--init-seed 5678`. Raw 4-batch means:

```
                     k5      k10     k20     k40     k80
off  start1       0.0230  0.0357  0.0423  0.0447  0.0453
off  start2       0.0218  0.0338  0.0414  0.0447  0.0456
on   start1       0.0247  0.0429  0.0676  0.1004  0.5584
on   start2       0.0255  0.0450  0.0706  0.2508  1.2306
```

Cost ratios, general text: start1 1.07 / 1.20 / 1.60 / 2.25 / **12.31**; start2
1.17 / 1.33 / 1.71 / 5.61 / **26.99**. The direction replicates and is far above any single
block (2.47-3.02), but the magnitude moves by 2.2x between starts, and the divergence sets
in one budget earlier at start 2 (k40 already 5.61 vs 2.25). The CONTROL is stable to 0.7%
at k80 across the same two starts (0.0453 vs 0.0456), so the instability is a property of
the penalised decomposition, not of the probe.

Task stream is weaker and start-dependent: start1 1.17/1.16/1.14/1.68/2.13 vs start2
1.14/1.03/0.98/1.03/1.34. Both exceed the flat per-block curves (1.00-1.16) at k80, but a
20-step probe sees nothing at either start, and start 2 sees essentially nothing until k80.
Do NOT quote an on-distribution coordination cost from one start.

Figure `plots/blocks_4l/03_per_block_cost.png` now carries both joint curves; per-block
curves remain single-start (they are stable enough that a second start was not the
bottleneck). Share doc section 8 reworded from "12.3x" to "12.3x and 27.0x", with the
instability stated.

**Process note.** This work collided with a parallel session on the same task: both wrote
`~/pd_scratch/dual_obj_jax/block_figs.py` and both harvested block 20 (jobs 11072/11073 vs
11077/11078). The duplicate block-20 harvest turned out to be useful — it is the only
independent repeat of the probe, and 8.29 uses it as the noise floor — but the scratch-file
overwrite was silent and cost a figure set. Scratch analysis scripts are not a safe shared
namespace between sessions.
