# Making JAX tPD fast on 2× L40

The tPD JAX run on `l40-worker` was slow enough that a 20 000-step run meant multiple requeue
segments against the 24-hour QOS cap. It now runs at **~1.15 s/step**. This records why,
because neither cause is visible on the hardware the code was designed for.

How slow it actually was is less certain than it looked: `l40_tpd_jax_blockers.md` records
~10 s/step, but the A/B below measures ~40 s/step for that layout even with prefix reuse
helping, which the old figure cannot be reconciled with (see "Which fix did what"). Treat the
~10 as unverified.

An earlier attempt at the same problem, in July, had already taken a comparable run from 250
to 3.0 s/step, and that work was never committed — it lived as uncommitted modifications in
the `8B_targeted_jax` worktree, on a branch whose tip predated it by two weeks. It is now
preserved at `ac7d1ce0c` on `experiment/8B_targeted_jax`; the findings below were
reconstructed from it before anything else happened.

The July recipe had five parts. Three already held on `experiment/tpd_jax_tests`, either
because they were ported properly or because the tree evolved into them. Two had not
transferred, and they were the two that produced almost all of the speedup.

## What already transferred

The **latency-hiding scheduler** is on: `RuntimeConfig.compiler_options` ships
`xla_gpu_enable_latency_hiding_scheduler: True` as a tuned default, so no config has to ask.

**CUDA graphs are on**, though by accident rather than intent. The same default map carries
`xla_gpu_enable_command_buffer: ""`, documented in the schema as a correctness guard that
disables graph capture. It does not: the empty string does not survive serialisation into
native compiler options, XLA falls back to its own default, and graphs get captured anyway.
July found that leaving graphs on was worth a large part of the 55 → 5–10 s/step step, so the
inert flag is currently doing the right thing for the wrong reason. It should be asserted or
deleted rather than left as a default that lies.

The **poisoned autotune cache** is handled. `~/out/xla_compilation_cache.poisoned-2026-07-13`
is quarantined and the live cache is clean, and no committed config sets
`gpu_autotune_level: "0"`. One loose end: the YAML example in
`param_decomp/experiments/CLAUDE.md` still shows `xla_flags: { gpu_enable_command_buffer: "",
gpu_autotune_level: "0" }`. Given that a single autotune-0 compile poisons the shared
per-fusion cache for every later full-autotune compile — silently, with flag changes becoming
no-ops — that example is a live footgun and should go.

## What did not transfer

### The frozen target is FSDP-sharded across two cards that cannot talk to each other

This is the same root cause July found behind 250 s/step, and it is back in full.

`build_target` places the loaded 8B through `place_target(model, mesh)`, and
`GLUDecomposedModel.shardings` shards the ~14 GiB stacked layer bulk on the `fsdp` mesh axis,
gathering one layer at a time inside the `lax.scan`. With `dp: 2, gpus_per_node: 2, tp: 1`,
`_hsdp_shape(2, 1, 2)` yields `(replicate 1, fsdp 2, tp 1)` — every block of the frozen target
is split across the two L40s and re-gathered on every pass through the scan, in every forward
of every step.

The docstring justifies this with "gathered per layer inside the scan, on NVLink". There is no
NVLink on this host. `nvidia-smi topo -p2p r` reports CNS between every GPU pair — no
peer-to-peer at all — and `libibverbs` fails to load, so NCCL routes every one of those
gathers over shared memory. The assumption is written into the code as a statement of fact and
is false here. It is true on the H100 nodes the layout was designed for, which is why it went
unnoticed.

July's fix was a `PD_SUBNODE_REPLICATE=1` env gate inside `hsdp_mesh` forcing `fsdp = 1` for
sub-node worlds, measured at 250 → 55 s/step. On this tree an env gate is the wrong shape —
the library reads no ambient environment, and `runtime.sharding` is already an authored
placement policy — but note that `sharding: zero1` governs the *trainable* state only; the
frozen target sits outside the placement table entirely, which is what the
`NOT AUDITED (legacy mesh-vocabulary .shardings): ci_fn, frozen target, …` banner in every log
is telling you. So this wants either a mesh-shape knob or a frozen-target row in the table.

The cost is memory. Today the target occupies ~9.1 GiB per card (7.0 GiB of sharded blocks
plus 2.1 GiB of replicated embed and head); full replication takes that to ~16.1 GiB, so
+7.0 GiB against roughly 10.5 GiB of headroom at the current peak of 30.9 of 41.4 GiB. It
fits, but not comfortably — July ran `xla_python_client_mem_fraction: 0.97` for exactly this
reason, against the 0.92 default here.

### Prefix reuse does not exist at all

For an L18-only decomposition of a 32-block model, blocks 0–17 are frozen, mask-independent,
and identical across every forward in the step. Nothing can flow back into them: all seven
decomposed sites live at block 18. Yet every forward in the step — the clean forward, the tap
read, each recon-grid forward, each PPGD ascent — currently embeds tokens and runs all 32
blocks, and `remat_recon_forwards: true` then replays those 18 wasted blocks in the backward
as well.

July restored the torch-era residual-start as a pure performance optimisation: a `ResidualStart`
wrapper carrying the stop-gradient residual entering block `split_layer`, computed once per
stream per step and accepted by every forward in place of token inputs. The arithmetic is
`F × 32` blocks becoming `18 + F × 14`, which at six to ten forwards per step is a 1.9–2.0×
reduction — matching the measured 5.9 → 3.0 s/step exactly.

Two implementation details from that work are worth more than the diff itself, because both
were found the expensive way:

- **The prefix and suffix stacks must be separate model fields** (`stacked_prefix` and
  `stacked`, split at build time), never an in-graph `stacked[split:]` slice. The slice
  materialises copies of a 16 GiB stack and breaks command-buffer capture; it regressed the
  step from 3.0 to 25 s.
- **Never pass a bound method as a `lax.scan` body.** `lax.scan` hashes its body function, and
  a bound method's hash reaches `self` — the traced, unhashable module — so it raises
  `TypeError`. Use a local closure.

Porting is a reimplementation rather than a patch application. `targets/glu_transformer.py` is
the direct descendant of the old `targets/llama8b.py` and keeps the same skeleton — `stacked`,
`_stack_per_kind_vu`, `_attach_per_kind_masks`, `_reconstruct_compute_weights` — but it has
since grown the capture-key grammar (`_scan_capture_layout`), the per-kind CI envelope stack
(`_stack_ci_per_kind`), and the unified `clean_forward` / `masked_forward` pair that replaced
the old `clean_output` / `read_activations` / `masked_output` grid in the 2026-07-30 amendment.
The July diff is still the best map of *which* layer-indexed lookups need a `- split_layer`
offset: the `collect` dictionary, the capture `sink`, `weight_deltas`, and the tap loop in
`read_activations`.

There is also a SPEC question to settle first. S3 and S18 were amended on 2026-06-24, with
Oli's approval, specifically to *remove* residual-start in favour of a whole-model token-input
engine. The reasoning lives in `REMOVE_RESIDUAL_START_DESIGN.md`, which SPEC still cites but
which was deleted from the tree in `edb299119` — read it there. July's re-add is marked
"pending Oli" and is a narrower claim than the thing that was removed — an optional target
capability gated on `split_layer > 0`, semantically transparent, rather than a return to a
separately harvested prefix. That distinction is what makes it arguable, and it should be
argued before the code lands.

## Doing better than either branch

The two fixes interact in a way neither branch exploited. Once the prefix is reused, its
weights are needed for exactly one forward per stream per step — so keeping the prefix FSDP-ed
costs one gather per step instead of one per forward, while the suffix, which every forward
touches, is what actually wants replicating.

That hybrid layout — prefix ÷fsdp, suffix replicated — puts roughly 12.1 GiB per card
(2.1 replicated embed and head, 3.9 of sharded prefix, 6.1 of replicated suffix) against
today's 9.1 and July's 16.1. It buys the full speedup for +3.0 GiB rather than +7.0, which
matters directly: the non-target batch is currently cut to 24 against the torch reference's 96,
and that headroom is the only way back toward it.

## What was built

Both fixes are on `perf/jax-tpd-2xl40-speed`, and the shapes differ from July's in two ways
that matter.

The frozen-target fix is **`runtime.fsdp`**, a mesh-shape knob, not July's
`PD_SUBNODE_REPLICATE` environment gate — the library reads no ambient environment, and the
mesh is where this decision actually lives. It defaults to `None` (the `gpus_per_node`-derived
shape), so no NVLink run changes; `fsdp: 1` degenerates the parameter-sharding axis, which
replicates every `P(..., "fsdp", ...)` spec and moves the world onto `replicate`, leaving
batch sharding intact.

**Prefix reuse** is the July design — `split_layer`, `stacked_prefix`, `ResidualStart`, hoisted
once per stream in `prep_stream` — reimplemented against the current `glu_transformer`, whose
capture-key grammar and unified forwards did not exist in `llama8b`. Both of July's hard-won
details carried over: separate stack FIELDS rather than an in-graph slice, and a local closure
rather than a bound method as the `lax.scan` body.

The port turned up a latent bug worth noting on its own: `component_activation_forward` indexed
the per-kind V stack by GLOBAL layer, which is wrong for any decomposition not starting at
block 0 — including the L18 seat.

## Measured, 2026-08-13

Both fixes landed on `perf/jax-tpd-2xl40-speed` (`189c403c0`) and ran as
`addsub-L18-jax-speed-01`, 5000 steps at the seat's own batch sizes so the comparison is
clean — only the two perf knobs differ from the config that measured ~10 s/step.

| step | elapsed | s/step | peak GB/rank |
|---|---|---|---|
| 100 | 2:54 | 1.736 (includes warmup) | 32.93 |
| 200 | 4:49 | 1.139 | 32.93 |
| 300 | 6:45 | 1.159 | 32.93 |
| 400 | 8:41 | 1.159 | 32.93 |

**~1.15 s/step steady state against ~10 s/step: about 8.7×.** That is well past the ~2×
the block arithmetic predicted for prefix reuse and past July's best of 3.0 s/step, which
says the per-layer gather over shared memory was costing more than the July numbers alone
implied. A 20 000-step run drops from about 56 hours — three requeue segments against the
24-hour QOS cap — to roughly six, inside a single allocation.

Memory landed better than feared. Replicating the frozen target adds ~7 GiB/card, and the
estimate was 37.9 of a 41.4 GiB pool; the run sits at **32.9 GB/rank** against ~43.6 GiB at
`mem_fraction: 0.97`. Prefix reuse shrinks the temp arena at the same time it saves compute
— every grid forward is 14 blocks instead of 32 — and the two effects nearly cancel the
replication cost. About 10 GiB is spare.

So the note above is wrong where it says the non-target batch cut to 24 is forced. It was
forced by an arena sized for 32-block forwards. Raising it back toward the torch reference's
96 is now the obvious next move, and `remat_recon_forwards: false` is the second — that flag
trades against an activation peak far smaller than when it was set.

## Which fix did what

Prefix reuse is unconditional in the code, so an A/B on `runtime.fsdp` alone isolates the
mesh knob. `addsub-L18-jax-ab-fsdp2` is the identical code and config with `fsdp` reverted to
the derived shape, logging every step:

| variant | frozen target | s/step | GB/rank |
|---|---|---|---|
| `fsdp` derived (=2) | sharded, gathered per layer | **40.3** (median, n=12) | 24.5 |
| `fsdp: 1` | replicated | **1.15** | 32.9 |

**The mesh knob alone is worth ~35× here, for +8.4 GB/rank.** The sharded figure is the
median of steps 3–14 (mean 39.5, range 35.2–43.5); steps 1–2 read 95 and 99 s, which is the
PPGD warmup (`n_warmup_steps: 2`), not settling.

One honest caveat: the probe shared `l40-worker` with the 5000-step run, and this host routes
all GPU↔GPU traffic over shared memory, so the sharded figure carries some cross-job
contention and should be read as an upper bound on that layout's cost.

**A discrepancy worth resolving before anyone trusts the old number.** `l40_tpd_jax_blockers.md`
records ~10 s/step for the sharded layout at this config. That cannot be reconciled with this
measurement: prefix reuse strictly *removes* work, so the pre-change branch — sharded AND
without prefix reuse — must be no faster than ~38 s/step, not 10. The ~38 figure sits much
closer to July's 250 s/step for the same layout (250 / 38 ≈ 6.5, of which prefix reuse
explains ~2× and config differences the rest). The peaks disagree too: that note reports 30.9
GiB for the sharded path where this probe measures 24.5. So the honest claim for the end-to-end
speedup is **at least the ~8.7× implied by the recorded baseline, and the A/B suggests
substantially more**; the recorded baseline itself should be re-measured rather than quoted.

## What this leaves open

`ascend_replicate` was never measured: under `fsdp: 1` the compute weights are already
replicated, so it is a no-op there and setting both is redundant. It stays the right first
lever on any run that keeps a real `fsdp` axis.

The SPEC amendment is **pending Oli**. S3/S18 were amended on 2026-06-24 specifically to
remove residual-start, and prefix reuse contradicts the letter of S18 even though it does
not reinstate what was removed — nothing is harvested, stored, or crosses a step boundary,
and the shared value is bit-identical to recomputing it. That conversation should happen
before this merges to main.

One capability note, since it is easy to misread the code: splitting the stack costs
nothing. From token inputs the prefix still runs through the capture machinery, so points
below `split_layer` resolve exactly as before (pinned against an unsplit model with
identical weights). Only a `ResidualStart` cannot answer them, because it has already
consumed the prefix — and no training capture point lives below the first decomposed block,
since masking cannot reach one.
