You are a Claude launched on the Andromeda cluster (SLURM) to AUTONOMOUSLY continue the "jaxer"
project with --dangerously-skip-permissions. Act decisively; don't wait for confirmation.

ROUND 2. First read `~/pd-nano-jax/jax_spike/HANDOFF.md`, then lore
`2026-06-08--4way-results-jax-torch-1pool-2pool`. Round 1 finished the 4-way but could NOT do a clean
same-hardware A/B (cluster full; real Llama-8B OOMs 1-pool; torch-2pool was anchored on a lore number;
cells ran at mixed scales). Fix exactly that with a SMALLER workload.

MISSION: a CLEAN, controlled **{jax,torch}×{1-pool,2-pool}** matrix where ALL FOUR cells run FRESH at
the **SAME total GPU count and SAME global batch** → true apples-to-apples tok/s/GPU + total tok/s.

HARD CONSTRAINTS (cluster is BUSY — verify with `sinfo` yourself):
- As of launch only ~1 fully-idle node (8 H200) + partially-free "mix" nodes. **Target a clean
  SINGLE-NODE 8-GPU A/B** (cleanest: no inter-node confound, definitely schedulable). Go to 16 GPU
  ONLY if SLURM actually packs it from free GPUs quickly; otherwise stay at 8. **Never exceed 16 GPU.**
  Leave the user's jobs (oli-dev, opus-dispatcher) alone. Throughput runs only (~30–100 steps past
  compile, read tok/s, scancel); never leave jobs idle/queued for long.
- **Target = GPT-2-XL** (gpt2-xl: 48L, d1600, 25 heads, ffn6400) — already parity-verified BOTH sides
  (`jax_spike/vendored_jax/gpt2.py` ↔ torch vendored gpt2), and small enough that 1-pool fits
  replicated on 8 GPU → enables the clean A/B (sidesteps the Llama-8B jax-1pool global-batch OOM).
  Decompose a few MLP layers (e.g. 3 layers' `mlp.c_fc` + `mlp.down_proj`); pick C, seq (~1024), and a
  global batch sized so ALL FOUR cells fit at 8 GPU. Keep GPU count + global batch IDENTICAL across
  cells — that equality is the entire point. (Fallback: small-batch Llama-8B-L18 + reduced C. Document.)
- Match across cells: bf16 weights, TF32 matmul, same losses (faith/imp/stoch-layerwise/persistent-
  PGD), same CI fn.

DO:
1. Adapt jax cells to GPT-2-XL: fork/generalize `stage10_real_pd_bench.py` (1-pool) +
   `stage11_real_pd_2pool.py` (2-pool) onto `vendored_jax/gpt2.py`.
2. Write matched torch GPT-2-XL configs: 1-pool (`pd-lm`, kind:hf transformers.GPT2LMHeadModel) +
   2-pool (`pd-lm-2pool`). 8-GPU topologies.
3. Run all four fresh at 8 GPU + identical global batch. Record tok/s/GPU + total tok/s + step ms +
   a few steps of loss-down (sanity, not just timing). If a cell OOMs at 8 GPU, shrink C/batch for ALL
   cells equally and rerun — keep it controlled.

RULES: work in ~/pd-nano-jax (branch feature/nano-pd-jax). torch: `PYTHONPATH=. .venv/bin/python`.
jax: `jax_spike/.venv-cuda` via `remote/gpu.sh`. Apply ALL round-1 fixes (block_until_ready warmup,
TF32 matching, `git config --global merge.conflictstyle diff3`, `uv sync --all-packages`, frozen
weights as jit ARG not constant). Commit progress frequently. When done, write lore
`2026-06-08--4way-gpt2xl-clean-ab` with the clean equal-GPU/equal-batch matrix + caveats, linked to
the round-1 results doc. Be honest about any cell that needed shrinking.

Begin: read HANDOFF.md + the lore doc, `sinfo` to see real capacity, pick a fitting GPT-2-XL config,
build + run all four at 8 GPU.
