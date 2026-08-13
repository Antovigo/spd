# Making JAX tPD fast on 2× L40

The targeted (tPD) JAX trainer was painfully slow on `l40-worker` — slow enough that a
20 000-step run meant babysitting several requeue segments against the 24-hour QOS cap. Two
changes took it to **1.15 s/step**, and a 20 000-step run now fits in about six and a half
hours, inside a single allocation.

Both problems come from the same blind spot: the trainer's layout was designed for an NVLink
node, and neither cause is visible there. An earlier attempt at this in July had found both
and measured the same kind of win, but that work was never committed. It is now preserved at
`ac7d1ce0c` on `experiment/8B_targeted_jax`, and the fixes below were rebuilt from it against
the current tree.

## Problem 1: the frozen model was split across two cards that cannot talk to each other

`build_target` runs the loaded 8B through `place_target`, and `GLUDecomposedModel.shardings`
splits the ~14 GiB of stacked layer weights across the `fsdp` mesh axis. Every block of the
frozen model then has to be re-gathered from the other card on every pass through the scan —
in every forward, of every step.

That is a fine trade on a machine where the cards share NVLink. This one is not that machine.
`nvidia-smi topo -p2p r` reports CNS between every GPU pair — no peer-to-peer at all — and
`libibverbs` fails to load, so NCCL routes every one of those gathers over shared memory. The
docstring for that sharding actually asserted the gather happens "on NVLink", which is simply
false here, and the cost is enormous: **~40 s/step of the step time was those gathers.**

## Problem 2: every forward re-ran the frozen first half of the model

The addsub config decomposes layer 18 of a 32-block model. Blocks 0–17 are frozen, no gradient
can reach them, and no mask touches them — so they compute exactly the same thing in every
forward of a step. But each forward embedded the tokens and ran all 32 blocks anyway: the
clean forward, the tap read, every recon-grid draw, every PPGD ascent. With
`remat_recon_forwards: true`, the backward then replayed those 18 wasted blocks as well.

## The fixes

**`runtime.fsdp`** is a new field controlling how wide the parameter-sharding plane is. It
defaults to `None`, meaning "derive it from `gpus_per_node`" — exactly today's behaviour, so
nothing on an NVLink node changes. Setting `fsdp: 1` collapses that axis, which makes every
`P(..., "fsdp", ...)` spec replicate rather than shard, and moves the devices onto the
`replicate` axis instead. Batch sharding is unaffected, so the run is still data-parallel
across both cards.

Putting this in the mesh rather than in the target is what keeps it simple: no target code
knows about it, and the frozen model, the CI function and the V/U compute weights all stop
gathering at once. (July used an environment variable for the same effect; a config field is
the right shape here, since the library deliberately reads no ambient environment.)

**Prefix reuse** teaches the target that it has a frozen lead worth computing once. A model
whose decomposed sites all sit past block *k* reports `split_layer = k`, keeps blocks `[0, k)`
in a separate `stacked_prefix` field, and accepts a `ResidualStart` — the activation entering
block *k* — in place of token inputs. The engine computes that once per stream per step in
`prep_stream` and substitutes it for the batch, so every downstream forward resumes from it.
At the L18 config that removes 18 of every forward's 32 blocks.

This is numerically identical, not merely close: the output and every captured activation are
bit-equal to the token path, which `targets/tests/test_prefix_reuse.py` pins directly. It also
costs no capability — given token inputs the prefix still runs through the normal capture
machinery, so activations below `split_layer` resolve exactly as before.

Two implementation details are worth remembering, because both were found the expensive way:

- The prefix and suffix must be **separate model fields**, never an in-graph `stacked[split:]`
  slice. Slicing a multi-GB stack per forward materialises copies and breaks command-buffer
  capture — July measured that regression at roughly 8×.
- **Never pass a bound method as a `lax.scan` body.** `lax.scan` hashes its body function, and
  a bound method's hash reaches `self` — the traced, unhashable module. Use a local closure.

## What it bought

Measured on the addsub L18 tPD config at its usual batch sizes (128 target / 24 non-target,
`dp: 2`), so the comparison is clean:

| frozen model | s/step | GB/rank |
|---|---|---|
| sharded across both cards (`fsdp` derived) | **39.9** | 24.5 |
| replicated (`fsdp: 1`) | **1.15** | 32.9 |

Prefix reuse is unconditional in the code, so reverting only `runtime.fsdp` isolates the mesh
knob: **it is worth ~35× on its own**, for +8.4 GB per card. Both figures are medians over
every post-warmup step of their runs — 39.9 from a completed 20-step probe (mean 39.7, range
35.2–44.1), and 1.15 from the 5000-step run (mean 1.151, range 1.13–1.17). The first two steps
of any run read much higher because `n_warmup_steps: 2` gives the PPGD adversary extra work;
that is not the model settling.

Prefix reuse is what separates the 39.9 above from July's 250 s/step for the same sharded
layout. It is the smaller of the two wins here, but it is the one that also shrinks the
executable's temp arena, and that turned out to matter for memory.

**Memory came out better than the arithmetic predicted.** Replicating the frozen model costs
about 7 GiB per card, which should have pushed the peak to roughly 38 of 41.4 GiB. Instead the
run sits at **32.9 GB/rank against ~43.6 GiB** at `mem_fraction: 0.97`, because prefix reuse
shrinks the arena at the same time — every grid forward is now 14 blocks instead of 32. About
10 GiB is spare.

That matters more than it sounds. The non-target batch was cut from the torch reference's 96
down to 24 to fit, and the notes blamed the config. It was really an arena sized for 32-block
forwards. Raising that batch back up is now the obvious next move, and turning
`remat_recon_forwards` off is the second — it trades against an activation peak far smaller
than when it was set.

The run itself is healthy end to end: eval, the slow-eval figure tier, and the step-2500
checkpoint (587 MB, both items on disk) all work unchanged, and the losses track the
configured schedules — including the expected bump when the PPGD adversary ramps in between
steps 1000 and 1500.

## A bug found on the way

`component_activation_forward` indexed the per-kind V stack by *global* layer number. That is
wrong for any decomposition that does not start at block 0 — including the L18 config it ships
with. Fixed as part of this work, with a test covering the case where a live chunk starts above
`split_layer`, which is what the multi-layer chunkwise configs hit on every draw.

## What is still open

The SPEC amendment is **pending Oli**. S3 and S18 were amended on 2026-06-24 specifically to
*remove* residual-start, and prefix reuse contradicts the letter of S18. It does not reinstate
what was removed — nothing is harvested, nothing is stored, nothing crosses a step boundary,
and the shared value is bit-identical to recomputing it — but that is a conversation to have
before this merges, which is why the code sits on a branch.

`ascend_replicate` was never measured. Under `fsdp: 1` the compute weights are already
replicated, so it is a no-op there and setting both is redundant. It remains the right first
lever on any run that keeps a real `fsdp` axis.

One number in `l40_tpd_jax_blockers.md` does not survive this work: it records ~10 s/step for
the sharded layout at this config. That cannot be right, because prefix reuse only removes work
and the sharded path still measures ~40 s/step *with* prefix reuse helping. The peak memory in
that note disagrees too (30.9 GiB where this probe measures 24.5). Treat the ~10 as unverified.
One caveat in the other direction: the 39.9 s/step probe shared the node with the 5000-step
run, and this host routes GPU traffic over shared memory, so it carries some cross-job
contention and should be read as an upper bound.
