# Making JAX tPD fast on 2× L40: what the earlier attempt found, and what still hasn't transferred

The tPD JAX run on `l40-worker` currently sits at ~10 s/step, which puts a 20 000-step run at
roughly 56 hours. An earlier attempt at the same problem, in July, took a comparable run from
250 s/step to 3.0 s/step on the same two cards. That work was never committed: it lives as
uncommitted modifications in the `8B_targeted_jax` worktree (eleven modified files plus an
untracked `param_decomp/tests/test_prefix_reuse.py`), on a branch whose tip predates it by two
weeks. It is not on any branch and not pushed. **Preserve it before anything else** — a
`git stash` or a WIP commit in that worktree costs nothing and the findings below are
reconstructed from it.

The July recipe had five parts. Three of them already hold on `experiment/tpd_jax_tests`,
either because they were ported properly or because the tree evolved into them. Two did not
transfer, and they are the two that produced almost all of the speedup.

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

## Suggested order

Start with **`ascend_replicate: true`**, which is already implemented
(`runtime.ascend_replicate`, `train.py::replicate_for_ascend`), defaults to false, and is not
set in `llama8b_l18_addsub_targeted_2xl40.yaml`. It gathers the ÷fsdp compute weights once
before the adversary ascents so the `n_warmup` ascend forwards skip their per-layer gathers,
collapsing `n_warmup × n_layer × (fwd + bwd)` collectives into one. It is the same family of
win as the frozen-target fix, applied to V/U instead, the numerics are bit-identical, and it is
a one-line config edit. Free information about how much the collectives are actually costing.

Then **un-shard the frozen target** for sub-node worlds. Biggest single win on the July
measurement, moderate implementation, moderate memory risk.

Then **prefix reuse**, which is worth about another 2× but needs the SPEC conversation, a real
port into `glu_transformer.py`, and a parity test in the shape of July's `test_prefix_reuse.py`
(it bit-matched step-100 losses against the slow baseline, which is the bar).

## Caveat

None of this is measured on the current tree. The ~10 s/step figure is from
`l40_tpd_jax_blockers.md`; the 250 / 55 / 5.9 / 3.0 figures are July's, on a differently-sized
config (non-target batch 64, `max_seq_len` 6, `remat_recon_forwards` off). The ranking comes
from reading the code and the memory record, not from benchmarking, and the first action above
is chosen partly because it is the cheapest way to start replacing it with data.
