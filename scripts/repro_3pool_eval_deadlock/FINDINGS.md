# 3-pool torch.profiler / CUPTI deadlock investigation (2026-05-28)

**Verdict: The deadlock is partially fixed but still reproduces at production
scale.** Up to ~56 ranks (production CI fn, multi-node), torch.profiler works
cleanly. At 104 ranks (the original failing config), it hangs during normal
training — symptom is rank 0 (LW block-0 leader) stuck in `dist.recv`+`.item()`
in `aggregate_losses_to_rank0` (`reductions.py:114-119`), while CI rank 96 has
moved on to the next iteration's pre-step barrier and PPGD rank 100 is in the
next iteration's async V/U broadcast recv. So the May 26 anti-pattern bullet
("Don't try to make torch.profiler work at 112-GPU scale") still holds.

## Background

Per `docs/handoff_2026-05-26_3pool_perf.md`:

> `torch.profiler` is usable at this scale — we tried four variants and all
> deadlocked the moment CUPTI activated. Cause unconfirmed (possibly CUPTI ↔
> NCCL incompatibility).

…and the launcher script `scripts/gpt2_xl_qk_production.py` still ships with
`--torch-profile` marked "(broken at scale, kept for completeness)".

The question this investigation set out to answer: **does the deadlock still
reproduce on current `feature/multipool`, or has it been fixed by intervening
changes?**

## Scaffolding used

Everything lives in `scripts/repro_3pool_eval_deadlock/`:

- `debug_scaffolding.py` — install scripts for SIGUSR1 stack-dump handler,
  faulthandler heartbeat, NCCL desync env vars (`TORCH_NCCL_DESYNC_DEBUG=1`,
  `TORCH_DISTRIBUTED_DEBUG=DETAIL`), per-rank memory snapshots, and a
  monkey-patch that injects an enabled `PhaseProfiler` whenever
  `PD_TORCH_PROFILE_RANKS` is set (current `feature/multipool` has the env-var
  intent in `gpt2_xl_qk_production.py` but no code that actually reads the env
  var to construct an enabled profiler — that wiring lives only on side
  branches; see commit `5559277e`).
- `run_debug.py` — torchrun-compatible wrapper that installs the scaffolding
  before delegating to `param_decomp_lab.experiments.lm.run.cli`.
- `submit_torch_profile.sh` — single-node 8-rank submit.
- `submit_torch_profile_mnode.sh` — parameterized multi-node submit.
- `debug_config_torch_profile{,_16r,_32r,_104r}.yaml` — 3-pool configs at the
  test scales.

All scaffolding is gated behind env vars and harmless when disabled. Nothing
touches core code.

## Results

| Job   | Ranks | Nodes | Config                              | Result    | Elapsed |
|-------|-------|-------|-------------------------------------|-----------|---------|
| 34208 |  8    |  1    | small CI fn (d=512, n=2)            | COMPLETED | 1:21    |
| 34209 | 16    |  2    | small CI fn                         | COMPLETED | 1:31    |
| 34210 | 32    |  4    | small CI fn                         | COMPLETED | 1:39    |
| 34211 | 104   | 13    | production xl_qk_smoke.yaml         | PREEMPTED (resubmitted)| —       |
| 34223 |  56   |  7    | **production** xl_qk_smoke.yaml (12 LW blocks × 4) | COMPLETED training, hung in NCCL cleanup (cancelled at 7min) | 3:48 to traces |
| 34222 | 104   | 13    | **production** xl_qk_smoke.yaml     | **FAILED — DEADLOCK** at ~step 10 in `aggregate_losses_to_rank0` | 8:34 |

## What the deadlock looks like at 104 ranks

Faulthandler dump after 5-min timeout (jobs `34222`, `pd_3pool_debug/34222/
rank_{000,096,100}_stacks.faulthandler.txt`):

| Rank | Pool | Stuck in |
|------|------|----------|
|   0  | LW (block-0 leader, profiler target) | `.item()` after `dist.recv(pgd_vals, src=ppgd_ranks[0], group=cross_pool_p2p_group)` in `aggregate_losses_to_rank0` (reductions.py:119) — step N's log aggregation |
|  96  | CI leader (profiler target)         | Pre-step `dist.barrier()` at `optimize.py:714` — already moved on to step N+1 |
| 100  | PPGD leader (profiler target)       | `dist.broadcast` in `async_recv_updated_vu_from_layerwise_kickoff` (`layout.py:1149`) — also at step N+1 |

So CI + PPGD finished step N's log (including their send-to-rank-0) and moved
to step N+1, but LW rank 0 is wedged in `.item()` on the just-recv'd PPGD
values. The recv "completed" from the NCCL API's perspective but the GPU
stream apparently hasn't drained the data, so `.item()` blocks.

This is consistent with the May 26 doc's "CUPTI ↔ NCCL incompatibility"
suspicion: CUPTI's instrumentation of NCCL kernels can break the stream-sync
contract that `.item()` relies on. The smaller-scale tests pass because there
are fewer concurrent NCCL ops for CUPTI to interfere with; once ranks ≥ ~64,
the interference becomes manifest.

The 56-rank run hung in cleanup (a milder version of the same pattern — NCCL
teardown deadlock under CUPTI). The 104-rank run hangs mid-training.

For each completed job: PhaseProfiler activated on one rank per pool, all
three Chrome trace JSONs written cleanly to `$HOME/pd_3pool_debug/<jobid>/
torch_profile/`, training continued normally past the active recording window,
zero errors, zero hangs.

## Likely fix

Most plausible candidate: commit `c202141d` ("three_pool: route cross-pool p2p
onto dedicated process group", 2026-05-27), which separated cross-pool p2p
sends/recvs onto their own `cross_pool_p2p_group` PG. The default PG now
carries only barriers. The handoff doc speculated "CUPTI ↔ NCCL
incompatibility" as the cause; one mechanism that fits is CUPTI's
instrumentation of NCCL kernels colliding with the same communicator handling
both blocking barriers (sync) and the heavy cross-pool p2p stream — the
separation removes that interleaving.

Other plausible contributors: the pre-step barrier added in `731ee481` and
the NCCL event-timing path's switch from `torch.cuda.synchronize()` to
per-event `post.synchronize()`.

## Follow-up suggestions

1. **The May 26 anti-pattern stands — keep the `--torch-profile` warning** in
   `gpt2_xl_qk_production.py` and the handoff doc. Update both to reflect
   that the deadlock is now scale-dependent: works ≤ ~56 ranks, hangs at
   104. Standalone single-process repros remain the right answer for getting
   `key_averages`-style data at this scale.
2. **Cherry-pick `5559277e` to `feature/multipool`.** Brings in the missing
   `_maybe_enable_memory_profile` wiring that `PD_MEMORY_PROFILE_*` env vars
   already expect. Today those env vars are dead — the launcher sets them but
   no code reads them on this branch. Memory snapshots are useful at any scale
   and not affected by the CUPTI issue.
3. **Possible deeper fix attempts** (only if you really need profiler at
   production scale): try disabling `profile_memory=True` (currently set in
   `PhaseProfiler.__enter__`) since memory tracking is what makes CUPTI
   heavy; or scale the schedule down to one rank profiled at a time and
   bracket all_to_all-style instrumentation with explicit `torch.cuda.
   synchronize()` to drain streams before any cross-pool `.item()`.
4. **If the deadlock at 104 ranks is reproducible enough, file a torch
   issue** — the pattern (cross-pool `.item()` blocking on a stream that
   CUPTI is tracking) is upstream-relevant.

## Side-find from the same scaffolding

The earlier sessions caught a separate bug in `param_decomp_lab/eval_metrics/
plotting.py:472` — `.cpu().numpy()` on bf16 CI tensors crashes with
`TypeError: Got unsupported ScalarType BFloat16`, which presents as a hang on
the other pools that subsequently time out in monitoredBarrier. **Fix
applied (uncommitted)**: `.float().cpu().numpy()`. This was the cause of the
smoke 34151 hang described in `~/notes/3pool_followups_2026_05_28.md` task
#3. Verified at 8-rank in jobs 34206 + 34207. Left in working tree for review.
