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

All figures in `notes/plots/nlpenalty/`. Apples-to-apples throughout: the L18 family is
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

![L18 neurons/component](notes/plots/nlpenalty/l18_nonlinearity_neuron.png)
![L18 heads/component](notes/plots/nlpenalty/l18_nonlinearity_attn.png)
![4L neurons/component](notes/plots/nlpenalty/4l_nonlinearity_neuron.png)
![4L heads/component](notes/plots/nlpenalty/4l_nonlinearity_attn.png)

**1) L0 — per stream, and the streams disagree.** Target stream: total L0 22 -> 24-25
(~10%, confined to gate_proj 4.9 -> 7.7). NON-target stream: gate/up L0 rises 4-5x with
the coefficient (gate 0.17 -> 0.82, up 0.16 -> 0.53 at 2x) while every unpenalized
matrix is flat — off the target distribution, the penalized components fire far more
promiscuously. This was invisible in the combined view.

![L18 L0 target](notes/plots/nlpenalty/l18_l0_target.png)
![L18 L0 non-target](notes/plots/nlpenalty/l18_l0_nontarget.png)
![4L L0 target](notes/plots/nlpenalty/4l_l0_target.png)
![4L L0 non-target](notes/plots/nlpenalty/4l_l0_nontarget.png)

**2) Alive components: the familiar census dip, ~9-11%**, on the penalized MLP sites
(grid census, stream-independent — the arithmetic-prompt grid).

![L18 alive](notes/plots/nlpenalty/l18_alive.png)
![4L alive](notes/plots/nlpenalty/4l_alive.png)

**3) Rounded-mask reconstruction: mild on target, ZERO on non-target.** Target-stream
rounded KL rises 11%/18%/22% at 0.5x/1x/2x; the non-target rounded KL is flat to four
significant figures in both families.

![L18 rounded KL target](notes/plots/nlpenalty/l18_rounded_kl_target.png)
![L18 rounded KL non-target](notes/plots/nlpenalty/l18_rounded_kl_nontarget.png)
![4L rounded KL target](notes/plots/nlpenalty/4l_rounded_kl_target.png)
![4L rounded KL non-target](notes/plots/nlpenalty/4l_rounded_kl_nontarget.png)

**4) PGD — per stream: target nearly flat until 2x; non-target carries the cost**
(+60-90% across both roles).

![L18 PGD target](notes/plots/nlpenalty/l18_pgd_target.png)
![L18 PGD non-target](notes/plots/nlpenalty/l18_pgd_nontarget.png)
![4L PGD target](notes/plots/nlpenalty/4l_pgd_target.png)
![4L PGD non-target](notes/plots/nlpenalty/4l_pgd_nontarget.png)

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
