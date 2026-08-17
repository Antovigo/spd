# Running JAX tPD on a pre-12.8 driver

Two bugs stopped the targeted root from executing a single step on `l40-worker`, and a third
constraint decides how large the run can be. This note covers what they are, how they work,
and which of them you'd hit on other hardware.

Our setup: L40 cards (sm_89, 45–48 GB depending on the card), driver **535.247.01 / CUDA
12.2**, and a venv on the `cuda` extra — which means a CUDA **12.8** userspace with cuDNN 9.8
and jax 0.10.1. One VM node, eight GPUs. The gap between the 12.2 driver and the 12.8
userspace is what causes the first bug, and it is worth keeping in mind throughout.

Fixes live in `a7b2636ce` (code and dependency) and `381ea3c71` (run-config sizing).

One practical warning before the rest: when these runs fail, they often don't exit. Multi-GPU
steps are synchronised — each GPU's worker blocks inside a communication step until all the
others reach it — so when one worker dies, typically on an out-of-memory, the survivors wait
for a partner that will never arrive, with no timeout. The job then looks alive while doing
nothing: no output, no CPU, and still holding every byte of its GPU memory. Worse, `scancel`
does not reliably reap it; you have to find the PID and `srun … kill -9` it by hand, and until
you do that memory is unavailable to everyone else on the node.

XLA will not rescue you here. It notices within ten seconds and logs `... Acquire clique ...
and may be stuck`, but the matching terminate timeouts
(`--xla_gpu_executable_terminate_timeout`,
`--xla_gpu_first_collective_call_terminate_timeout_seconds`) cover collective *execution*,
not the clique-acquire rendezvous, and the rendezvous path has only `warn_stuck` variants.
Setting them is accepted and changes nothing — measured: a deliberately over-budget dp2 run
still hung for the full 1200 s.

So the launcher has to do it. Wrap the trainer in `timeout --signal=KILL <secs>` to bound the
waste — the timeout fires from inside the job's own cgroup, where signalling the process is
permitted — and, better, have a watchdog kill the moment `ran out of memory trying to
allocate` appears in the log. That line is written at the instant of failure, ten seconds
before the stuck warning, so you reap in seconds with the cause already recorded, instead of
inferring a stall from log silence much later.

## The error that hides its own cause

Almost everything below first shows up as this, naming a different kernel each time:

```
INTERNAL: [0] There was an error before calling cuModuleGetFunction (1):
cudaErrorInvalidValue : invalid argument [executable_name='jit_targeted_step']
```

It is tempting to read that as "this kernel failed to load". It isn't. XLA's phrasing is
literal — the message comes from a `cudaPeekAtLastError` guard, so CUDA was *already* in an
error state and this is simply the first place XLA looks. In practice that's hundreds of
kernel loads after whatever actually went wrong. The kernel it names is a bystander, and
debugging it leads nowhere.

Three things follow. The traceback tells you nothing about where the fault is.
`CUDA_LAUNCH_BLOCKING=1` won't move it, because it synchronises kernel *launches* and the
failing call is an ordinary synchronous API call. And shrinking the batch or `C` to simplify
the reproducer won't make it go away — these faults are structural — though it does buy you a
hundred-second repro instead of a fifteen-minute one.

To find the real origin, run under the sanitizer that matches your **driver**, not your venv:

```bash
/usr/local/cuda-12.2/bin/compute-sanitizer --tool memcheck \
  --report-api-errors all --target-processes all \
  python -m param_decomp.experiments.lm.run_targeted --config ... --data_root ...
```

`--report-api-errors all` prints every CUDA API call that returns an error, with a host
backtrace. It named the cause below in about ninety seconds. Reach for it early.

## cuDNN attention doesn't work on this driver

`attn_implementation` in `vendored_jax/llama.py` asked for
`jax.nn.dot_product_attention(..., implementation="cudnn")` whenever the sequence length was
a multiple of 64, on GPU, in fp16 or bf16. cuDNN 9's graph API can't run against driver 535:

```
xla::gpu::CuDnnThunk::Initialize
  -> CudnnSupport::DeserializeGraph
    -> cudnn_frontend::graph::Graph::deserialize -> warmup -> execute
      -> run_auxiliary_kernels
        -> cudaMemcpyAsync  =>  cudaErrorInvalidValue
```

Nobody checks that return value, so CUDA stays poisoned and the error resurfaces later as the
message above.

This looked like a targeted-only bug for a while, but it isn't. The tPD target stream uses
prompts five tokens long (SPEC T8), which never selects cuDNN; the non-target stream runs at
seq 64, which always does. Any run at a multiple of 64 is exposed, plain or targeted — and
any host whose driver predates its CUDA userspace will hit it.

The fix makes `attn_implementation` always return the XLA composite. Its arguments are still
there, so restoring the capability check on a newer driver is a one-line edit, plus one in
`param_decomp/tests/test_attn_implementation.py` where the contract is pinned.

That does mean cuDNN flash attention is now off everywhere, including on machines where it
works fine. The better long-term shape is a `runtime:` field —
`attention_implementation: auto | xla`, defaulting to `auto` — following the `remat_ci_fn`
pattern, so the choice is visible in the pinned `launch_config.yaml` and healthy hardware
keeps the fast path.

One dead end worth flagging: XLA's own cuDNN flags
(`--xla_gpu_cudnn_gemm_fusion_level=0`, `--xla_gpu_enable_cudnn_fmha=false`) do nothing here.
They control XLA's pattern-matching passes, and this is an explicit JAX-level custom call
they never see.

## The pinned NCCL is too old for this jaxlib

With attention fixed, every multi-GPU run died on:

```
NCCL operation ncclAlltoAll(...) failed: unhandled system error
```

`pyproject.toml`'s `cuda` extra pinned `nvidia-nccl-cu12==2.25.1` to keep the NVIDIA
components on a single CUDA 12.8.1 release train. But XLA in jax 0.10.1 emits `ncclAlltoAll`
when it lowers a reshard, and that entry point doesn't exist before NCCL 2.28:

```console
$ nm -D .../nvidia/nccl/lib/libnccl.so.2 | grep -iE 'ncclAlltoAll|ncclAllGather'
ncclAllGather        # 2.25.1: AllGather is there, AlltoAll isn't
```

Nothing about the transport is involved, which is worth knowing because the error message
invites you to go looking there. `NCCL_SHM_DISABLE`, `NCCL_P2P_DISABLE`,
`NCCL_CUMEM_ENABLE=0`, and switching `sharding` between `zero1` and `ddp` all change nothing.
The operation simply isn't implemented, and "unhandled system error" is how NCCL says so.

The fix bumps the pin to `nvidia-nccl-cu12>=2.28` (resolving to 2.31.2), deliberately off the
12.8.1 train. This is a packaging bug rather than anything to do with our hardware — anyone
running `dp > 1` on this extra hits it.

## What fits, and why the batch size barely matters

Once it runs, memory decides the scale. The binding constraint is the executable's temp arena
— XLA's single buffer for all of an executable's intermediates — sitting on top of the
resident frozen target, which is already FSDP-sharded to roughly 9 GiB per card (see the
correction below). The surprise is how little the arena responds to batch size:

| non-target batch | temp arena | fits the 41.4 GiB pool? |
|---|---|---|
| 48 | 23.05 GiB | no |
| 32 | 22.27 GiB | no |
| 24 | 17.93 GiB | yes — peak 30.9 GiB, ~10 s/step |

Halving *both* streams (target 64, non-target 16) moved the arena from 17.93 to 17.83 GiB.
Both remat flags were already on. So if you're over budget, shrinking the batch is a weak
lever and you should expect to give up a lot of it for very little.

Watch `xla_python_client_mem_fraction`, which defaults to 0.92. Dropping it to 0.75 caps BFC
at roughly 34 of 45 GB and turns a config that fits into one that hangs.

Two consequences worth weighing. The torch reference ran non-target 96 on this same hardware,
so 24 is a fourfold cut to the broad stream — a real fidelity loss, not a free win. And at
~10 s/step a 20 000-step run is about 56 hours against a 24-hour QOS cap, so plan on roughly
three requeue segments.

**Correction (2026-08-13).** An earlier revision of this section claimed the frozen target was
replicated at 16 GiB per card and proposed sharding it as the untouched lever. That is
backwards on both counts. `build_target` already runs the target through
`place_target(model, mesh)`, and `GLUDecomposedModel.shardings` FSDPs the ~14 GiB layer bulk
on the `fsdp` axis; at `dp: 2, gpus_per_node: 2, tp: 1` the mesh is `(replicate 1, fsdp 2,
tp 1)`, so the target is *already* split across the two cards at roughly 9 GiB each (7 GiB of
sharded blocks plus the 2.1 GiB replicated embed and head). The memory arithmetic corroborates
it: peak 30.9 GiB minus a 17.93 GiB arena leaves ~13 GiB of resident state, which cannot
contain a 16 GiB target. Sharding is not available to be pulled, and pulling it further is the
wrong direction anyway — see `jax_tpd_speed_transfer.md`, where that same sharding turns out to
be the dominant cost on a host with no P2P.

## A multi-GPU OOM hangs instead of raising

This one is general JAX/XLA behaviour and worth internalising. When one rank runs out of
memory mid-allocation, it dies quietly while the other blocks in
`Acquire clique: devices=2:[0,1]`, and the job sits there until the wall clock kills it. So a
"clique deadlock" is far more often an out-of-memory condition than a rendezvous fault.

On any `dp > 1` hang, grep the log for `ran out of memory trying to allocate` before you go
near NCCL.

## XLA flags don't behave the way the schema suggests

Three gotchas, all general.

`xla_gpu_enable_command_buffer: ''` in `compiler_options` silently does nothing. The schema
documents it as a correctness guard that disables CUDA-graph capture, but with it set,
`TF_CPP_VMODULE=cuda_executor=5` still logs `Create CUDA command buffer (CUDA graph)`. The
empty string doesn't survive serialisation into native compiler options. Only
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` actually disables them — verified by the count
dropping to zero. Any run that assumed graphs were off has been running with them on.

`compiler_options` and `XLA_FLAGS` accept different sets of flags. cuDNN and other *debug*
options are rejected outright by `compiler_options` (`No such compile option:
'xla_gpu_enable_cudnn_fmha'`). Both paths reject unknown names loudly; the empty-string case
above is the one silent failure.

Finally, `compiler_options` are part of the compile-cache key and `XLA_FLAGS` are not. An A/B
driven by `XLA_FLAGS` will happily load a cached executable compiled without them and tell
you nothing. Clear `<data_root>/xla_compilation_cache/jit_targeted_step-*` first.

## About this node specifically

`nvidia-smi topo -p2p r` reports `CNS` between every GPU pair, and `libibverbs` fails to load
— no peer-to-peer, no InfiniBand — so NCCL routes GPU-to-GPU traffic over shared memory. On a
tiny config that costs a lot: dp2 measured 51 s/step against dp1's 1.6 s. At realistic batch
sizes the penalty is much smaller (~10 s/step), but the shape of the tradeoff stands. We run
two cards because one can't hold the working set, not because two are faster.

`/dev/shm` is 229 GB, so shared-memory size is never the problem here.

## Still open: the CI grid doesn't fit in-loop

> Measured against `ArithmeticCIGrid`, which `ABGridDataset` has since replaced. The new
> metric runs ONE frozen `component_activation_forward` per chunk where the old tier ran a
> clean forward plus a masked one, and never gathers the full `(n_prompts, C)` grids — so
> the numbers below are an upper bound on what it asks for. Whether that clears the
> contiguous-region ceiling on this card is unmeasured.

`ArithmeticCIGrid` can't run alongside training. It wants a single allocation of about
19.4–19.8 GiB, and the largest BFC will serve on this card is 17.98 GiB.

```
Pool limit   40.78 GiB      In use at failure   9.01 GiB
Peak in use  27.00 GiB      Largest ever served 17.98 GiB
```

Note that only 9 GiB was in use when a 19.77 GiB request failed against a 40.78 GiB pool. This
is a contiguous-region ceiling rather than a capacity problem: the training step's arena
carves up the pool first, and what's left can't host one region that large.

What's frustrating is that the request doesn't move:

| what was tried | request |
|---|---|
| 100×100 grid, one forward | 19.77 GiB |
| `chunk_prompts: 1000` plus `probe_metrics: null` | 19.77 GiB |
| `chunk_prompts: 100` | 19.77 GiB |
| `xla_python_client_allocator: platform` | 17.98 GiB |
| non-target 24 → 16 | 19.38 GiB |

`chunk_prompts` and the nullable `probe_metrics` are committed in `1499ab721` and do bound the
forward, but neither is what sizes this allocation. Whatever does, it isn't the prompt axis,
and it hasn't been identified — so there's no telling yet whether other setups hit this.

The obvious untried route is to compute the grid offline against a saved checkpoint, where it
has the card to itself and no training arena to compete with. Everything else in
`eval.metrics` runs fine in-loop.
