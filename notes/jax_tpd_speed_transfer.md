# Speeding up targeted PD on 2× L40

A 20 000-step tPD run on `l40-worker` took upwards of 60 hours. With the changes below it
takes **6.4 hours** — 1.147 s/step, measured over a 5000-step run.

Both changes address the same thing: the trainer's default layout assumes the GPUs share a
fast fabric. On this host they do not. `nvidia-smi topo -p2p r` reports **CNS between every
GPU pair** — no peer-to-peer at all — and `libibverbs` fails to load, so NCCL routes all
GPU↔GPU traffic over shared memory. Anything the layout gathers between cards is therefore
far more expensive here than the code assumes.

## Change 1 — stop sharding the frozen model across the two cards

This is the big one: **~35× on its own.**

`GLUDecomposedModel.shardings` splits the ~14 GiB of stacked layer weights across the `fsdp`
mesh axis, so every block of the frozen 8B is re-gathered from the other card on every pass
through the scan — in every forward, of every step. Over shared memory, those gathers are
essentially the entire step time.

The fix is a new `runtime.fsdp` field that sets how wide the parameter-sharding plane is.
`fsdp: 1` collapses that axis, so every `P(..., "fsdp", ...)` spec replicates instead of
sharding, and the devices land on the `replicate` axis instead. Batch sharding is unaffected —
the run is still data-parallel across both cards. It costs the full frozen model resident per
card (~16 GiB instead of ~9).

The field defaults to `None` ("derive it from `gpus_per_node`"), which is exactly the existing
behaviour, so nothing on an NVLink node changes.

Putting the switch in the mesh rather than in the target is what keeps it small: no target
code knows about it, and the frozen model, the CI function and the V/U compute weights all
stop gathering at once. It also subsumes `ascend_replicate`, whose whole job is hoisting the
same gather off the adversary ascents — don't set both.

## Change 2 — reuse the frozen prefix instead of recomputing it every forward

The addsub config decomposes layer 18 of a 32-block model. Blocks 0–17 are frozen, no gradient
reaches them, and no mask touches them, so they compute the same thing in every forward of a
step. Each forward nonetheless embedded the tokens and ran all 32 blocks — the clean forward,
the tap read, every recon-grid draw, every PPGD ascent — and with `remat_recon_forwards: true`
the backward replayed those 18 wasted blocks too.

A model whose decomposed sites all sit past block *k* now reports `split_layer = k`, holds
blocks `[0, k)` in a separate `stacked_prefix` field, and accepts a `ResidualStart` (the
activation entering block *k*) in place of token inputs. The engine computes it once per stream
per step in `prep_stream` and substitutes it for the batch, so every downstream forward resumes
from there.

This is numerically identical, not approximately: output and every captured activation are
bit-equal to the token path, pinned by `targets/tests/test_prefix_reuse.py`. It also costs no
capability — given token inputs the prefix still runs through the normal capture machinery, so
activations below `split_layer` resolve exactly as before.

Two implementation details matter a great deal:

- The prefix and suffix must be **separate model fields**, never an in-graph `stacked[split:]`
  slice. Slicing a multi-GB stack per forward materialises copies and breaks command-buffer
  capture, which costs roughly 8×.
- **Never pass a bound method as a `lax.scan` body.** `lax.scan` hashes its body function and a
  bound method's hash reaches `self` — the traced, unhashable module. Use a local closure.

## The config change

Starting from the shipped `llama8b_l18_addsub_targeted_2xl40.yaml`, only the `runtime:` block
changes:

```yaml
runtime:
  dp: 2
  gpus_per_node: 2
  tp: 1
  fsdp: 1                                  # <-- change 1
  sharding: zero1
  remat_recon_forwards: true
  remat_ci_fn: true
  launch_env:
    xla_python_client_mem_fraction: 0.97   # <-- headroom for the replicated model
```

Change 2 needs no config: it activates automatically for any target whose decomposed sites all
sit above block 0.

## The code change

`runtime.fsdp` threads from `RuntimeConfig` (`experiments/lm/runtime.py`) through
`config.py` / `training.py` / `training_targeted.py` into `_hsdp_shape`, `hsdp_mesh` and
`hsdp_abstract_mesh` (`core/sharding.py`).

Prefix reuse adds `ResidualStart` and the `SupportsPrefixResidual` protocol to `core/model.py`,
the once-per-stream hoist to `core/train.py::prep_stream`, and `split_layer` / `stacked_prefix`
/ `_start` plus the layer-index offsets to `targets/glu_transformer.py`.

It also fixes a real bug: `component_activation_forward` indexed the per-kind V stack by
*global* layer number, which is wrong for any decomposition not starting at block 0 — including
this L18 config.

## Measurements

Same code, same config, batch 128 target / 24 non-target, `dp: 2`. The only difference is
`runtime.fsdp`:

| frozen model | s/step | 20k steps | GB/rank |
|---|---|---|---|
| sharded across both cards (`fsdp` unset) | 39.9 | ~220 h | 24.5 |
| replicated (`fsdp: 1`) | **1.147** | **6.4 h** | 32.9 |

Medians over every post-warmup step: 39.9 from a 20-step probe (mean 39.7, range 35.2–44.1);
1.147 from the 5000-step run (mean 1.149, range 1.13–1.17, n=46). The first two steps of any
run read much higher because `n_warmup_steps: 2` gives the PPGD adversary extra work — that is
not the model settling. The 39.9 probe shared the node with the other run, and this host routes
GPU traffic over shared memory, so read it as an upper bound.

Memory came out better than expected. Replicating the frozen model costs ~7 GiB/card, which
should have pushed the peak to ~38 of 41.4 GiB. It sits at **32.9 GB/rank against ~43.6 GiB**,
because change 2 shrinks the executable's temp arena at the same time — every grid forward is
14 blocks instead of 32. About 10 GiB is spare.

That spare capacity is worth spending. The non-target batch is currently 24, cut down from the
torch reference's 96 to fit an arena sized for 32-block forwards. Raising it back up is the
obvious next move, and turning `remat_recon_forwards` off is the second — it now trades against
a much smaller activation peak.

The full training loop is unaffected: eval, the slow-eval figure tier, and checkpointing
(587 MB, both items) all work unchanged, and losses track the configured schedules.

## One thing to settle before merging

SPEC S3 and S18 were amended on 2026-06-24 to *remove* residual-start, and change 2
contradicts the letter of S18. It does not reinstate what was removed — nothing is harvested,
nothing is stored, nothing crosses a step boundary, and the shared value is bit-identical to
recomputing it — but the amendment is marked **pending Oli** and should be agreed before this
lands on main.
