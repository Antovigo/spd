# Fixes and changes worth sending upstream

Written 2026-08-25, after merging upstream PR #1000 (`facf2e7b1`, "Explicit placement,
broader transformer targets, and evaluation refinements") into the dual-objective branch
and getting the result to actually run on GPU.

**Who this is for.** Someone deciding what to contribute back, or someone hitting one of
these problems and wanting to know whether it is already understood. Each item says what
breaks, *who it breaks for*, and whether the fix is needed by everyone or only by us.

**The setup everything below was measured on**, since several items only appear under
particular conditions:

| | |
|---|---|
| Model | Llama-3.1-8B, bfloat16 weights |
| What is decomposed | one transformer block (layer 18), 7 sites: the three MLP projections and the four attention projections |
| Sequence length | 64 tokens |
| Hardware | 2 x NVIDIA L40 (48 GB each), single machine |
| Notable hardware limit | no direct GPU-to-GPU links (every pair reports "CNS"), so cross-GPU traffic goes through shared host memory and is slow |
| Notable software limit | the NVIDIA driver predates CUDA 12.8, so cuDNN's newer attention path cannot run at all |
| Run type | targeted parameter decomposition, the branch's four-pass dual-objective variant |

A map of everything, then the details:

| # | Change | Upstream affected? | Only matters if... |
|---|---|---|---|
| 1 | CI-scaled weight decay crashes at step 0 | **Yes, upstream code** | you run on more than one device *and* enable that option |
| 2 | CI network's attention backend cannot be chosen | **Yes, upstream code** | your machine cannot run cuDNN attention |
| 3 | Nonlinearity penalty refused in targeted runs | **Yes, upstream rule** | you want that penalty in a targeted run |
| 4 | `zero1` sends 26x more data per step | **Yes, upstream behaviour** | your mesh has an FSDP axis of 1 |
| 5 | Frozen layers re-sliced and re-run every pass | **Yes, upstream lacks the fix** | your model is much larger than the part you decompose |
| 6 | A/B-grid eval: two sharding/placement bugs | No, our file only | (kept for the general lesson, which does apply upstream) |
| 7 | Delta-pinned masks silently broken by the merge | No, our file only | (kept for the general lesson) |

---

## 1. CI-scaled weight decay crashes at step 0 on any real multi-device mesh

**What happens.** The run dies immediately with a `ShardingTypeError` inside
`ci_scaled_weight_decay.apply`, before a single training step completes.

**Why.** The decay factor is computed from CI values, whose component axis is laid out
the way *activations* are laid out. It is then multiplied into the stored component
weights, whose component axis is laid out the way *stored parameters* are laid out. Under
#1000's explicit-axis mode those two layouts are not interchangeable, so the multiply is
rejected. The two stored tensors (`V` and `U`) need not agree with each other either.

**Who this affects.** Anyone on upstream `main` who enables CI-scaled weight decay and
runs on more than one device. Not specific to our branch, our model, or our hardware.
Upstream's own targeted configuration does not set this option, which is very likely why
it has not been hit yet.

**Why no test catches it.** It cannot happen on a single device, because then there are no
layouts to disagree. Our full 1280-test suite runs on CPU and passes. It appeared on the
very first GPU run of the merged branch.

**The fix** (`63db6fa0c`, about 19 lines in `core/train.py`): before multiplying, re-label
the factor to match each tensor's own layout, read off that tensor individually. The
numbers produced are unchanged; this is purely about labelling. It is skipped when no mesh
is active, so single-device behaviour is untouched.

---

## 2. The CI network always picks its own attention backend

**What happens.** On our machine the run dies with:

    error before calling cuModuleGetFunction (1): cudaErrorInvalidValue

naming `jit_targeted_step`. Nothing in that message mentions attention or cuDNN.

**Why.** #1000 added a setting for which attention implementation the *target model* uses.
The chunkwise CI network, however, chooses its own with a hard-coded `"auto"`
(`core/ci_fn.py`, around line 403 upstream). At a 64-token sequence `"auto"` selects
cuDNN. So a run that explicitly asked for the XLA implementation still got cuDNN inside
the CI network, and on a machine that cannot run cuDNN the whole run fails.

**Who this affects.** Anyone whose machine cannot use cuDNN attention, typically because
of an older driver, as here. On a fully up-to-date machine this is invisible. Not specific
to our branch.

**Cost of not fixing it.** This is a hard failure, not a slowdown, and the error message
points at the wrong place, so it is expensive to diagnose. It blocked every multi-device
targeted run until found.

**The fix** (`a9111a49e`): make the backend a normal setting on the CI network's attention,
defaulting to `"auto"` so existing behaviour is unchanged. We deliberately kept it separate
from the target model's setting: the CI network is a different, much narrower attention,
and someone could reasonably want cuDNN for one and not the other.

---

## 3. The nonlinearity penalty cannot be used in a targeted run

**What happens.** A targeted config listing `NonlinearityLocalityLoss` is rejected when the
config is parsed, before anything runs.

**Why.** Two places forbid it: `TargetedLossMetricConfig` does not list the penalty as an
allowed member, and `build_targeted_objective` asserts it is absent, citing SPEC S36/T3.

**Who this affects.** Anyone wanting that penalty in a targeted run. The penalty itself
exists upstream and works; only the targeted combination is closed off.

**Is it a bug?** Genuinely unclear, and worth asking rather than assuming. The targeted
loss list is an intentional allow-list, and the neighbouring restriction (no faithfulness
role in a targeted run) is a real design decision. But the penalty is a weight-space term
that does not involve CI or a forward pass at all, so the reasoning behind the faithfulness
restriction does not obviously carry over. Our reading is that it is an omission, since the
penalty is newer than the targeted loss list.

**What we did locally.** Added it as an allowed member and dropped the assertion. The
penalty is scored once per step against the component weights, outside the pass loop, and
added once regardless of how many passes the run has. Two tests pin this: one measures its
contribution in a 2-pass and a 4-pass run and checks they agree and that it scales linearly
with its coefficient; the other checks that a hidden pass cannot produce it.

**One caveat for interpreting results.** A targeted run has no faithfulness role, so nothing
pushes back on components concentrating their writes the way it would in plain
decomposition. Curves should be read with that in mind.

---

## 4. `zero1` sends 26x more data per step when the FSDP axis is 1

The largest practical finding here, and a *performance* problem rather than a crash.

**What happens.** The same configuration, on the same hardware, was roughly twice as slow
after the merge as before it. We measured this by counting the cross-device communication
in each run's compiled step:

| Configuration | Time per step | Peak memory per GPU | Number of all-reduces | Data reduced per step |
|---|---|---|---|---|
| before the merge, `zero1` | 3.10-3.15 s | 21.16 GB | 13 | 300 MB |
| after the merge, `zero1` | 6.32 s | 28.23 GB | 69 | **7 969 MB** |
| after the merge, `ddp` | **2.76-2.83 s** | 26.79 GB | 12 | 356 MB |
| after the merge, FSDP axis 2 | 14.01 s | 22.85 GB | not measured | not measured |

Those four rows were measured on a reduced short run at the same model shape, so their
memory column is lower across the board than a full production run's and should be read
only *against each other*. For production-shape memory numbers see item 5.

**Why.** #1000 changed what `zero1` means (SPEC D4 amendment, 2026-08-18): the per-group
fallback layout was removed, and `zero1`'s layout now applies to all groups. At our shape
(one decomposed block, replicate 2, FSDP 1) the new layout ends up reducing the full
gradient tree across the replicate axis every step, in 32 separate pieces of 64 MB or more.
On a machine with no direct GPU-to-GPU links, that traffic is the entire slowdown.

**Who this affects.** Anyone running with an FSDP axis of 1 and more than one replica.
Upstream's own setups appear to run FSDP 8, where the layout behaves as intended, which is
presumably why this has not surfaced there.

**The fix on our side is a one-line config change:** use `ddp` instead of `zero1`. That is
*faster than where we started*, 2.83 s against 3.10 s before the merge, at the cost of 27%
more memory. That extra memory is simply the price of replicating optimizer state rather
than sharding it, and is comfortable on a 48 GB card.

**Checked and ruled out** before finding this, all by measurement rather than reasoning: the
nonlinearity penalty itself (+0.3%), the frozen-stack changes, the compiler flags
(byte-identical), recompilation (3 compilations on both sides), and kernel count (+9%,
nowhere near 2x). We also tried `owner` partitioning, which refuses at this shape: it splits
the component-stack axis across replicas, and a single-block decomposition gives each group
a stack of length 1, which cannot be split in two. That refusal is correct behaviour, not a
bug.

**Suggestion for upstream.** Either make `zero1` fall back to a sensible layout when the
FSDP axis is 1, or refuse the combination outright with a clear message. Silently becoming
twice as slow is the worst of the three options. The method that found it is worth reusing:
dump the optimized compiled step and count the collective operations and their sizes. Note
that the merged branch writes that dump in binary form by default, so the text form has to
be requested explicitly.

---

## 5. Frozen layers are re-sliced and re-run on every masked forward

**What happens.** No failure; this is purely speed and memory.

**Why.** When you decompose one block of a 32-block model, the other 31 blocks are frozen
and identical across every masked forward in a step. Upstream's transformer target keeps
all layers in one stack and slices out the range it needs on each forward, which copies a
large amount of data repeatedly, and re-runs the leading frozen blocks once per pass even
though every pass computes exactly the same thing.

**Who this affects.** Anyone whose target model is much larger than the portion being
decomposed, which is the normal case for single-block studies on an 8B model. It matters
more the more masked forwards a step performs, and our four-pass targeted runs perform
many.

**What we did** (`ff219fa9d`, then `8784009be`): split the model into three parts, the
frozen blocks before the decomposed range, the range itself, and the frozen blocks after
it, so no slicing copy is needed; then run the leading frozen part **once per stream** and
share its output across passes.

**Honest accounting of the benefit.** Neither of these is the reason the merged branch was
slow; item 4 was, and we initially and wrongly blamed these. The prefix sharing on its own
measured about 3% of step time at this one-block configuration.

The clearest evidence arrived later, from a control run: the same recipe, same seed, same
schedules, run on the merged branch (with `ddp` and both changes above) against the
original pre-merge production run, compared at matched steps.

| | pre-merge production run | merged branch + `ddp` + these changes |
|---|---|---|
| time per step | 3.54-3.58 s | **2.91-2.94 s** (18% faster) |
| peak memory per GPU | 36.65-36.82 GB | **28.61 GB** (22% less) |

Both numbers are steady from step 200 to step 1000, and the loss curves track each other
throughout (see below), so this is a like-for-like production comparison rather than a
micro-benchmark. The two effects are combined here and we have not separated them: the
speed is mostly item 4's `ddp` switch, the memory mostly this item's frozen-stack split.

They are worth contributing regardless of the split of credit: they are what makes
multi-block work practical, and the benefit grows with the number of passes.

**Status upstream:** absent. Upstream's transformer target still slices per forward.
Contributing this means rebasing onto #1000's new model-anatomy abstraction, which is what
`ff219fa9d` did on our side.

---

## 6 and 7. Two problems in our own files, kept for the lesson rather than to contribute

**The A/B-grid evaluation passed the wrong object** (`41ca10c1a`). #1000 requires that
anything reading CI values receives the CI network *paired with its resolved placement*.
The eval machinery hands operations exactly that pair, and its own documentation says
operations should use it and never the raw network. Our grid evaluation reached past it and
used the raw network, so it failed with `'ChunkwiseTransformerCIFn' object has no attribute
'fn'`.

Three things let this reach a 16-hour run and kill it after 3 900 steps:

- The file is ours, so upstream's placement work never touched it.
- Both function signatures typed the parameter as `Any`, so the type checker had nothing to
  verify. These were the *only* two such annotations in the entire codebase.
- The grid's schedule deliberately skips the first evaluation pass, so at our cadence its
  first run was step 4 000. Short test runs cannot reach it. Our first attempt at a short
  test run finished successfully **without running the grid at all**, a green result proving
  nothing, because we had accidentally aimed it at the skipped pass.

**Then it failed a second time, in the same function, one statement later.** With the first
fix in place the run reached step 4 000 again and died on the *next* line: the gather that
pulls out the saved components runs along an axis that is split across GPUs, and under
#1000's explicit-sharding mode a gather along a split axis has no single correct answer for
how the result should be laid out, so JAX refuses rather than guess. Fixed by declaring the
answer (`fb07ebf3a`).

The short test run that cleared the first fix could not see the second one, and the reason
is worth stating plainly: it ran the grid at step 20, when the decomposition is untrained
and no component passes the cut-off, so the list of saved components was empty and the loop
exited *before* the gather. Zero saved components exercises exactly half of that function. A
test run that means to cover it has to force the list non-empty by setting the cut-off to
zero. Reaching a piece of code is not the same as covering it.

**The genuinely useful outcome, and the part that transfers.** Both failures were sharding
problems, and neither was visible to the twelve existing tests for that file, because those
all run on a single device where the operations in question are unproblematic. But the
repository already has everything needed to catch them without a GPU: a `multidevice` test
marker and a `make test-multidevice` target that runs the suite on eight simulated CPU
devices. It simply was not pointed at this code.

We added one test there (`4cbd48a67`). It builds the same layout production uses and
performs the same gather, and with the fix reverted it fails on CPU with the production
error, down to the shape structure: `float32[16@(replicate,fsdp),2,8@tp]` against
production's `float32[1000@(replicate,fsdp),1,512@tp]`. It runs in about two seconds.

A bug that had cost two multi-hour GPU runs to discover became a two-second CPU test. If
there is one thing worth taking from this whole exercise, it is that **explicit sharding
made a new class of bug possible that single-device tests cannot see, and the cheap defence
already exists and is under-used.** Anywhere #1000 introduced explicit layouts is worth a
multidevice test, upstream code included.

Two smaller lessons from the same episode: an `Any` annotation at a boundary whose entire
purpose is *which object you pass* will eventually cost someone a long run; and a short test
run is only meaningful once you have checked it actually reaches — and fully exercises — the
code you are testing.

**Delta-pinned mask construction was silently broken by the merge.** The file merged with no
conflict at all, but the surrounding data structure had changed from raw arrays to a typed
one, so the merged result was wrong while looking perfectly fine. Worth remembering that a
conflict-free merge is not evidence of a correct merge.

---

## Suggested order if contributing

1. **Item 1** (weight-decay crash): small, self-contained, clearly a bug, affects upstream
   users directly.
2. **Item 2** (attention backend): small, and the default preserves existing behaviour.
3. **Item 4** (`zero1` at FSDP 1): report the measurements first and let upstream choose
   between a fallback and a refusal; we have no strong view on which.
4. **Item 3** (nonlinearity penalty): ask before patching, it may be deliberate.
5. **Item 5** (frozen-stack split): the largest piece of work and the one needing the most
   discussion, since it touches the model-anatomy abstraction #1000 just introduced.

---

Fuller technical detail, including the measurement runs behind item 4 and the failure
timeline behind item 6, is in `report.md`, sections 7 and 8.
