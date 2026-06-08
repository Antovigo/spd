Continue the "jaxer" project. You're running INTERACTIVELY on the Andromeda cluster (the human is
watching and can steer) — so normal long-running work is fine: poll jobs, schedule wakeups, iterate.

START: read `~/pd-nano-jax/jax_spike/HANDOFF.md` (self-contained), then lore
`2026-06-08--4way-results-jax-torch-1pool-2pool`.

MISSION: a CLEAN, controlled **{jax,torch} × {1-pool,2-pool}** throughput matrix — all FOUR cells
FRESH at the SAME GPU count + SAME global batch → true apples-to-apples tok/s/GPU + total tok/s.
Round 1 finished the 4-way but had to anchor torch-2pool on a lore number (cluster was full); round 2
wrote the GPT-2-XL benches but its runs failed and it gave up early. So most code already exists —
your job is to RUN it cleanly and collect the four numbers.

CONFIG (controlled, sized to fit):
- **GPT-2-XL** (gpt2-xl: 48L, d1600, 25 heads, ffn6400) — parity-verified both sides
  (`jax_spike/vendored_jax/gpt2.py` ↔ torch vendored gpt2), small enough that 1-pool fits replicated.
- Decompose a few MLP layers (`mlp.c_fc` + `mlp.c_proj`), bf16, TF32 matched, same losses
  (faith/imp/stoch-layerwise/persistent-PGD) + same CI fn across all cells. Size C / batch / #layers /
  seq so ALL FOUR fit at the chosen GPU count, and use IDENTICAL values for every cell.
- **Single node, 8 GPU** is the clean target (no inter-node confound; verify capacity with `sinfo`;
  the cluster is busy — never exceed 16 GPU; leave oli-dev / opus-dispatcher alone). Throughput runs
  only: ~30–100 steps past compile, read tok/s, then scancel.

EXISTING CODE (committed on `feature/nano-pd-jax`): `jax_spike/stage12_gpt2xl_1pool.py`,
`jax_spike/stage13_gpt2xl_2pool.py`, and matched torch GPT-2-XL 1-pool/2-pool configs.

DO (commit after each step):
1. jax 1-pool — run stage12 at 8 GPU **via `jax_spike/remote/gpu.sh`** (`NODES=1 GPN=8 bash
   jax_spike/remote/gpu.sh "python stage12_gpt2xl_1pool.py <args>"`). DO NOT roll your own jax
   launcher — round 2 did and got coordinator "Connection refused" errors; gpu.sh uses the proven
   `distributed_util.init_distributed()` recipe (validated to 16 GPU). If stage12/13 don't call
   `distributed_util.init_distributed()`, fix them so gpu.sh works.
2. jax 2-pool — run stage13 at 8 GPU.
3. torch 1-pool + torch 2-pool GPT-2-XL configs at 8 GPU (`jax_spike/remote/torch_gpu.sh`, or
   `PYTHONPATH=. .venv/bin/python -m param_decomp_lab.experiments.lm.run / .two_pool_run`).
4. If any cell OOMs, shrink C/batch for ALL cells equally and rerun.
5. Write lore `2026-06-08--4way-gpt2xl-clean-ab`: the equal-GPU/equal-batch matrix (4 numbers) + exact
   config (model, #dec layers, C, seq, global batch) + caveats; link to the round-1 results doc.

GOTCHAS (full list in HANDOFF.md): `jax.block_until_ready` warmup before timing (else compile time
leaks in); TF32 matched both sides; git already has `merge.conflictstyle diff3`; lab env is
`~/pd-nano-jax/.venv` (torch, `PYTHONPATH=. .venv/bin/python`), jax env is
`jax_spike/.venv-cuda` (used by gpu.sh). Work in ~/pd-nano-jax, commit to feature/nano-pd-jax.
