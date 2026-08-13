# JAX tPD on a pre-12.8 driver: bugs, mechanisms, and what generalizes

Reference environment: NVIDIA L40 (sm_89, 45 GB), driver **535.247.01 / CUDA 12.2**, venv on
the `cuda` extra (CUDA **12.8** userspace, cuDNN 9.8, jax 0.10.1), single VM node, 8 GPUs.

Fixes in `a7b2636ce` (code + dependency), `381ea3c71` (run-config sizing).

---

## 1. cuDNN attention fails against a pre-12.8 driver

**Generalizes** to any host whose driver predates the venv's CUDA userspace. Not L40- or
tPD-specific; tPD merely selects the failing path more often.

`vendored_jax/llama.py::attn_implementation` requested
`jax.nn.dot_product_attention(..., implementation="cudnn")` whenever `seq_len % 64 == 0` on
GPU in fp16/bf16. cuDNN 9's graph API cannot run against driver 535:

```
xla::gpu::CuDnnThunk::Initialize
  -> CudnnSupport::DeserializeGraph
    -> cudnn_frontend::graph::Graph::deserialize -> warmup -> execute
      -> run_auxiliary_kernels
        -> cudaMemcpyAsync  =>  cudaErrorInvalidValue
```

The return is unchecked, so CUDA stays sticky and the error surfaces later (§5).

**Why targeted runs hit it and plain runs may not:** the tPD target stream's prompt length
is 5 (SPEC T8) and never selects cuDNN; the non-target stream at seq 64 always does. Any run
whose sequence length is a multiple of 64 is exposed.

**Fix:** `attn_implementation` returns the XLA composite unconditionally. Args are retained
so restoring the capability check is one line here and one in
`param_decomp/tests/test_attn_implementation.py`.

**Open:** this disables cuDNN flash attention globally, including where it works. The
upstream-worthy form is a `runtime:` field (`attention_implementation: auto | xla`,
defaulting to `auto`), matching the `remat_ci_fn` pattern.

**Does not help:** XLA's cuDNN flags (`--xla_gpu_cudnn_gemm_fusion_level=0`,
`--xla_gpu_enable_cudnn_fmha=false`). Those govern XLA's pattern-matching passes; this is an
explicit JAX-level custom call and is invisible to them.

---

## 2. The pinned NCCL predates `ncclAlltoAll`

**Generalizes** — a repo packaging bug, independent of host. Affects every `dp > 1` run.

```
NCCL operation ncclAlltoAll(...) failed: unhandled system error
```

`pyproject.toml`'s `cuda` extra pinned `nvidia-nccl-cu12==2.25.1` to keep NVIDIA components
on one CUDA 12.8.1 train. XLA in jax 0.10.1 emits `ncclAlltoAll` when lowering a reshard,
and that entry point first exists in NCCL 2.28:

```console
$ nm -D .../nvidia/nccl/lib/libnccl.so.2 | grep -iE 'ncclAlltoAll|ncclAllGather'
ncclAllGather        # 2.25.1: AllGather present, AlltoAll absent
```

**Not a transport problem.** `NCCL_SHM_DISABLE`, `NCCL_P2P_DISABLE`, `NCCL_CUMEM_ENABLE=0`
and `zero1` vs `ddp` change nothing — the operation is not implemented, and "unhandled
system error" is how NCCL reports that.

**Fix:** `nvidia-nccl-cu12>=2.28` (resolves 2.31.2), deliberately off the 12.8.1 train.

---

## 3. Memory: the temp arena is batch-independent

**Mechanism generalizes; the numbers are 45 GB-L40-specific.**

The binding constraint is the executable's **temp arena** (XLA's single buffer for an
executable's intermediates), stacked on the **16 GiB replicated frozen target**. The arena
barely responds to batch, so batch is a weak lever:

| non-target batch | temp arena | fits 41.4 GiB pool? |
|---|---|---|
| 48 | 23.05 GiB | no |
| 32 | 22.27 GiB | no |
| 24 | **17.93 GiB** | **yes** — peak 30.9 GiB, ~10 s/step |

Halving target *and* non-target (64/16) moved the arena 17.93 → 17.83 GiB. Both remat flags
were already on.

`xla_python_client_mem_fraction` defaults to **0.92**; lowering it to 0.75 caps BFC at ~34 GB
of 45 and turns a fitting config into a hang.

**Cost of the 24 setting:** the torch reference ran non-target 96 on the same hardware, so
this is a 4× reduction in the broad stream. The untouched lever is the frozen target —
replicated at 16 GiB/card and outside the placement policy (`NOT AUDITED (legacy
mesh-vocabulary .shardings): ci_fn, frozen target, …`). Sharding it would buy back the most.

At ~10 s/step, 20000 steps ≈ 56 h against a 24 h QOS cap: about three requeue segments.

---

## 4. A `dp > 1` OOM hangs instead of raising

**Generalizes** — JAX/XLA multi-device behaviour.

Rank 0 dies mid-allocation while rank 1 blocks in `Acquire clique: devices=2:[0,1]`; the job
sits until the wall clock. A "clique deadlock" is therefore far more often an OOM than a
rendezvous fault. **On any dp>1 hang, grep for `ran out of memory trying to allocate` before
suspecting NCCL.**

---

## 5. Diagnosing sticky CUDA errors

**Generalizes.**

```
INTERNAL: [0] There was an error before calling cuModuleGetFunction (1):
cudaErrorInvalidValue : invalid argument [executable_name='jit_targeted_step']
```

This is **not** a module-load failure. XLA's wording is literal: a `cudaPeekAtLastError`
guard. CUDA was already in an error state and this is the first place XLA checks, typically
hundreds of kernel loads after the offending call. The named kernel is a bystander.

- The traceback location says nothing about the fault location.
- `CUDA_LAUNCH_BLOCKING=1` does not help — it synchronises *launches*; the failing call is
  an ordinary synchronous API call.
- The fault is structural, so shrinking batch/C reproduces it identically — useful only as a
  fast reproducer.

Name the true origin with the sanitizer matching the **driver** (not the venv):

```bash
/usr/local/cuda-12.2/bin/compute-sanitizer --tool memcheck \
  --report-api-errors all --target-processes all \
  python -m param_decomp.experiments.lm.run_targeted --config ... --data_root ...
```

`--report-api-errors all` reports every CUDA API call returning an error, with a host
backtrace.

---

## 6. XLA flag plumbing

**Generalizes.**

- **`xla_gpu_enable_command_buffer: ''` in `compiler_options` silently does nothing.** The
  schema documents it as a correctness guard disabling CUDA-graph capture; with it set,
  `TF_CPP_VMODULE=cuda_executor=5` still logs `Create CUDA command buffer (CUDA graph)`. The
  empty string does not survive serialisation into native compiler options. Only
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` disables them (verified: count → 0).
- **`compiler_options` and `XLA_FLAGS` accept different flag sets.** cuDNN and other *debug*
  options are rejected by `compiler_options` (`No such compile option:
  'xla_gpu_enable_cudnn_fmha'`). Both reject unknown names loudly; only the empty-string case
  fails silently.
- **`compiler_options` are in the compile-cache key; `XLA_FLAGS` are not.** An `XLA_FLAGS`
  A/B will reuse a cached executable compiled without them. Clear
  `<data_root>/xla_compilation_cache/jit_targeted_step-*` first.

---

## 7. Node interconnect

**Specific to `l40-worker`.**

`nvidia-smi topo -p2p r` reports `CNS` between every GPU pair and `libibverbs` fails to load:
no P2P, no InfiniBand, so NCCL routes GPU↔GPU `via SHM/direct`. At tiny configs dp2 measured
51 s/step against dp1's 1.6 s; at real batch ~10 s/step. **dp2 here is for capacity, not
speed** — two cards because one cannot hold the working set. `/dev/shm` is 229 GB, never the
constraint.

---

## 8. Open: `ArithmeticCIGrid` does not fit in-loop

**Unresolved. Generalization unknown** — the sizing driver has not been identified.

The pass wants a single **~19.4–19.8 GiB** allocation; BFC's largest servable here is
**17.98 GiB**.

```
Pool limit   40.78 GiB      In use at failure   9.01 GiB
Peak in use  27.00 GiB      Largest ever served 17.98 GiB
```

With only 9 GiB in use against a 40.78 GiB pool, this is a **contiguous-region** ceiling, not
capacity: the training step's arena carves the pool first and what remains cannot host one
region that large.

The request is invariant to every lever tried:

| lever | request |
|---|---|
| 100×100 grid, one forward | 19.77 GiB |
| `chunk_prompts: 1000` + `probe_metrics: null` | 19.77 GiB |
| `chunk_prompts: 100` | 19.77 GiB |
| `xla_python_client_allocator: platform` | 17.98 GiB |
| non-target 24 → 16 | 19.38 GiB |

`chunk_prompts` and nullable `probe_metrics` are committed (`1499ab721`) and do bound the
forward, but neither is what sizes this allocation — **it is not the prompt axis**.

Untried alternative: run the grid **offline against a saved checkpoint**, where it owns the
card and competes with no training arena. Everything else in `eval.metrics` runs fine
in-loop.
