# Getting the JAX targeted (tPD) root to run on `l40-worker`

What had to be fixed before `python -m param_decomp.experiments.lm.run_targeted` would
execute a single step on our 2×L40 box, and the sizing it settles at. Written up because
several of these are traps that will cost the next person the same day they cost this one.

Environment: NVIDIA L40 (sm_89, 45 GB), driver **535.247.01 / CUDA 12.2**, venv on the
`cuda` extra (CUDA **12.8** userspace, cuDNN 9.8, jax 0.10.1). Single node, 8 GPUs, VM.

Commits: `a7b2636ce` (the two code/dependency fixes), `381ea3c71` (run-config sizing).

---

## 0. The error everything looked like

Every failure below the first fix presented as exactly this, always naming a different
innocent kernel:

```
INTERNAL: [0] There was an error before calling cuModuleGetFunction (1):
cudaErrorInvalidValue : invalid argument [executable_name='jit_targeted_step']
```

**This is not a module-load failure.** XLA's wording is literal — it is a
`cudaPeekAtLastError` guard. CUDA was *already* in an error state and this is merely the
first place XLA bothers to check, which is typically hundreds of kernel loads after the
call that actually failed. The kernel it names is a bystander; do not debug it.

Consequences worth internalising:

- The traceback's location carries no information about the fault's location.
- `CUDA_LAUNCH_BLOCKING=1` does **not** help — it synchronises kernel launches, and the
  failing call was an ordinary synchronous API call, not a launch.
- Shrinking batch/C to "simplify the repro" proves nothing: the fault is structural, so a
  C=8 run fails identically. That *is* useful — it gives a 100-second reproducer.

**What actually found it**, in ninety seconds, after eleven hypotheses had been tested and
killed one job at a time:

```bash
/usr/local/cuda-12.2/bin/compute-sanitizer --tool memcheck \
  --report-api-errors all --target-processes all \
  python -m param_decomp.experiments.lm.run_targeted --config ... --data_root ...
```

`--report-api-errors all` reports every CUDA API call that returns an error, with a host
backtrace, so the origin is named rather than inferred. Use the sanitizer matching the
**driver** (12.2 here), not the venv. Reach for this early on any sticky-CUDA-error hunt.

---

## 1. cuDNN attention is unusable on a pre-12.8 driver

**Symptom.** The error in §0, on every targeted run, at any scale, at dp1 and dp2.

**Cause.** `vendored_jax/llama.py::attn_implementation` asked for
`jax.nn.dot_product_attention(..., implementation="cudnn")` whenever
`seq_len % 64 == 0`. cuDNN 9's graph API cannot run against driver 535:

```
xla::gpu::CuDnnThunk::Initialize
  -> CudnnSupport::DeserializeGraph
    -> cudnn_frontend::graph::Graph::deserialize -> warmup -> execute
      -> run_auxiliary_kernels
        -> cudaMemcpyAsync  =>  cudaErrorInvalidValue
```

Nothing checks that return, so CUDA goes sticky and resurfaces later as §0.

**Why it looked targeted-only.** The tPD *target* stream's prompt length is 5 (SPEC T8),
which never selected cuDNN. The *non-target* stream runs at seq **64**, which always did.
A plain full-data run at a non-multiple-of-64 sequence length sails past this.

**Fix.** `attn_implementation` now always returns the XLA composite. Arguments are kept so
restoring the capability check on a newer driver is a one-line edit (and one in
`param_decomp/tests/test_attn_implementation.py`, which pins the contract).

**Open design question for the team.** This disables cuDNN flash attention *globally*,
including on machines where it works. The upstream-worthy version is a `runtime:` config
field (`attention_implementation: auto | xla`, defaulting to `auto`), matching the
`remat_ci_fn` pattern — explicit in the pinned `launch_config.yaml`, and no penalty for
healthy hardware. Not built here because it threads through shared interfaces and deserves
a deliberate decision.

**Note.** XLA's own cuDNN flags (`--xla_gpu_cudnn_gemm_fusion_level=0`,
`--xla_gpu_enable_cudnn_fmha=false`, …) do **not** help. They govern XLA's pattern-matching
passes; this is an explicit JAX-level custom call and is invisible to them.

---

## 2. The pinned NCCL predates a collective this jaxlib emits

**Symptom.** With §1 fixed, every `dp > 1` run died with:

```
NCCL operation ncclAlltoAll(...) failed: unhandled system error
```

**Cause.** `pyproject.toml`'s `cuda` extra pinned `nvidia-nccl-cu12==2.25.1` to keep the
NVIDIA components on one coherent CUDA 12.8.1 release train. But XLA in jax 0.10.1 emits
`ncclAlltoAll` when lowering a reshard, and that entry point does not exist before NCCL
2.28:

```console
$ nm -D .../nvidia/nccl/lib/libnccl.so.2 | grep -iE 'ncclAlltoAll|ncclAllGather'
ncclAllGather        # 2.25.1: AllGather present, AlltoAll absent
```

**This is not a transport problem.** `NCCL_SHM_DISABLE`, `NCCL_P2P_DISABLE`,
`NCCL_CUMEM_ENABLE=0`, and switching `sharding` between `zero1` and `ddp` all changed
nothing, because there is no transport involved — the operation simply is not implemented.
"Unhandled system error" is NCCL's unhelpful way of saying so.

**Fix.** `nvidia-nccl-cu12>=2.28` (resolves to 2.31.2), deliberately off the 12.8.1 train,
with the reason recorded inline so nobody re-pins it back.

---

## 3. Sizing: what actually fits, and why batch is a weak lever

**Symptom.** With §1 and §2 fixed, the full-scale dp2 run still failed — see §4 for how
that failure presents.

**Cause.** The binding constraint is the executable's **temp arena** (XLA's single buffer
for an executable's intermediates), stacked on top of the **16 GiB replicated frozen
target**. The arena is largely *batch-independent*, so cutting batch buys much less than
expected:

| non-target batch | temp arena | fits 41.4 GiB pool? |
|---|---|---|
| 48 | 23.05 GiB | no |
| 32 | 22.27 GiB | no |
| 24 | **17.93 GiB** | **yes** — peak 30.9 GiB, ~11 s/step |

Halving target *and* non-target together (64/16) moved the arena from 17.93 to 17.83 GiB —
i.e. essentially not at all. Both remat flags were already on.

A second, self-inflicted factor: a scratch config had
`xla_python_client_mem_fraction: 0.75`, capping BFC at ~34 GB of 45 and stranding 11 GB.
**The schema default is 0.92 — do not lower it.** At 0.75 the non-target-24 config misses
by a hair (16 + 17.93 = 33.9 GiB against a 34 GB cap); at 0.92 it fits comfortably.

**Fix.** `llama8b_l18_addsub_targeted_2xl40.yaml` now runs non-target 24 (and the eval
geometry that tracks it), with the measurements recorded in the config comment.

**Two things this costs us, for the team to weigh.**

1. The torch reference ran non-target 96 on the same hardware. 24 is a 4× reduction in the
   broad stream, which is a genuine fidelity gap, not a free win. The untouched lever is
   the frozen target: it is **replicated** at 16 GiB/card and sits outside the placement
   policy (every log says `NOT AUDITED (legacy mesh-vocabulary .shardings): ci_fn, frozen
   target, …`). Sharding it would buy back the most.
2. 20000 steps × ~11 s ≈ **61 hours** against a 24 h QOS cap — about three requeue
   segments. SIGTERM → save → requeue → resume is supported, but plan for it.

---

## 4. Traps that cost real time

**A `dp > 1` OOM hangs; it does not raise.** Rank 0 dies mid-allocation while rank 1 blocks
in `Acquire clique: devices=2:[0,1]`, and the job sits there until the wall clock. Every
"clique deadlock" seen while debugging this turned out to be an OOM in disguise — hours
went into NCCL rendezvous theories that were never the problem. **On any dp>1 hang, grep
the log for `ran out of memory trying to allocate` before suspecting NCCL.**

**`xla_gpu_enable_command_buffer: ''` in `compiler_options` silently does nothing.** The
schema documents it as "a correctness guard" disabling CUDA-graph capture. With it set,
`TF_CPP_VMODULE=cuda_executor=5` still logged 38 × `Create CUDA command buffer (CUDA
graph)`. The empty string does not survive serialisation into native compiler options.
Only `XLA_FLAGS=--xla_gpu_enable_command_buffer=` actually disables them (verified: count
drops to 0). Every run assuming graphs were off has had them on. Worth fixing or asserting
in the schema.

**`compiler_options` and `XLA_FLAGS` accept different flag sets.** cuDNN and several other
knobs are XLA *debug* options and are rejected by `compiler_options`
(`No such compile option: 'xla_gpu_enable_cudnn_fmha'`). Both reject unknown names loudly;
only the empty-string case above fails silently.

**`compiler_options` are in the compile-cache key; `XLA_FLAGS` are not.** A test driven by
`XLA_FLAGS` will happily reuse a cached executable compiled without them and tell you
nothing. Clear `<data_root>/xla_compilation_cache/jit_targeted_step-*` when A/B-ing via
env.

**This node has no P2P and no InfiniBand.** `nvidia-smi topo -p2p r` reports `CNS` between
every GPU pair, and `libibverbs` fails to load; NCCL routes GPU↔GPU `via SHM/direct`. At
the tiny-config extreme dp2 measured 51 s/step against dp1's 1.6 s. At real batch it is far
better (~11 s/step), but **dp2 here is for capacity, not speed** — we need two cards
because one cannot hold the working set, not because two are faster. `/dev/shm` is 229 GB,
so SHM size is never the issue.

---

## 5. Also on this branch (not fixes)

Ported from the torch reference so the run mirrors `addsub-L18-09-one-im`: `coupled` /
`zero_u` component init, `normalize_at_one` on smooth-L0 importance minimality, the
target-pool eval pass (fresh-PGD + scalars), two-stream CI-mean plots, weight-magnitude
plots, and the 20 000-prompt addsub pool. See `1131f1bdc`, `0c0fb2c52`, `97e89a57b`.

One straggler fixed here: three of those eval configs (`TargetPoolScalarsConfig`,
`TwoStreamCIMeanPerComponentConfig`, `WeightMagnitudeConfig`) were never registered in
`param_decomp/tests/test_eval_tier.py`'s tier sets. The default testmon inner loop did not
cover that test, so it stayed green locally — worth running `make test-all` before
declaring a new metric done.
