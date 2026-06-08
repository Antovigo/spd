GOAL: produce a clean, controlled {jax,torch}×{1-pool,2-pool} GPT-2-XL throughput matrix — all FOUR
cells run FRESH at the SAME GPU count + SAME global batch, recorded as tok/s/GPU + total tok/s — and
write it to lore `2026-06-08--4way-gpt2xl-clean-ab`. You are running interactively on the Andromeda
cluster; KEEP WORKING TOWARD THIS GOAL ON YOUR OWN until it's done or it's morning.

CONTEXT (read first): `~/pd-nano-jax/jax_spike/HANDOFF.md`,
`~/pd-nano-jax/jax_spike/remote/CONTINUE_INTERACTIVE.md`, and lore
`2026-06-08--4way-results-jax-torch-1pool-2pool`. The GPT-2-XL bench code already exists + is committed
(`stage12_gpt2xl_1pool.py`, `stage13_gpt2xl_2pool.py`, matched torch 1/2-pool configs) — your job is to
RUN it cleanly and collect four numbers, not rebuild it.

PERSISTENCE — this is the important part. Do NOT stop or hand back when blocked:
- Before each run, check capacity (`sinfo`, `squeue --me`). If there aren't enough free H200s for the
  cell, SLEEP and retry: `sleep 1200` (20 min), then re-check — loop this. Cluster is contended; be
  patient, not give-up-y. (Use ScheduleWakeup / a sleep loop — whatever lets you resume yourself.)
- When you submit a job, POLL it to completion inline (`while squeue -j "$JID" -h | grep -q .; do
  sleep 30; done`) then read the log. Don't end your turn "waiting to be re-invoked."
- If a cell OOMs, shrink C / batch / #dec-layers for ALL cells equally and rerun (keep it controlled).
- Commit progress to `feature/nano-pd-jax` after every cell so nothing is lost.
- Run jax cells via the proven `jax_spike/remote/gpu.sh` (NOT a hand-rolled launcher — that broke
  before with coordinator "Connection refused"). Torch via `remote/torch_gpu.sh` / `pd-lm(-2pool)`.

GUARDRAILS:
- Target single-node 8 GPU; NEVER exceed 16 GPU. Leave `oli-dev` and `opus-dispatcher` alone.
- Throughput runs only: ~30–100 steps past compile, read tok/s, then `scancel`. Never leave a big job
  idle or queued indefinitely (if queued >~45 min with no start, scancel and try a smaller cell/size).
- `jax.block_until_ready` warmup before timing; TF32 matched both sides.

STOP CONDITIONS (then write a final lore note and summarize):
- DONE: all four cells measured at equal GPU + equal batch, committed, and written to
  `2026-06-08--4way-gpt2xl-clean-ab` (linked to the round-1 results doc). OR
- TIME: it's ~morning — about 8–9 hours after you start (note your start time; stop by then even if
  incomplete). At the time-stop, write what's done + what remains to lore so it's a clean handoff.

Order of work: jax 1-pool → jax 2-pool → torch 1-pool → torch 2-pool. Begin now: read the context,
check capacity, and start cell 1 (sleeping/looping if the GPUs aren't free yet).
