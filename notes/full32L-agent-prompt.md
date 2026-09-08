# Task: full-network (32-block) targeted addsub decomposition of Llama-3.1-8B on 8x H100

## Goal

Scale the single-block targeted dual-objective decomposition `addsub-L18-23-neuronaligned`
(tPD, SPEC §11 / S36 / T12) to ALL 32 blocks of Llama-3.1-8B (224 sites), on one Runpod pod
with 8x H100 SXM 80 GB (NVLink). Same recipe, same kind of result: a sparse neuron-aligned
decomposition whose AB grids show clean arithmetic structure.

Deliverables, all committed to the repo (outside `param_decomp/`, e.g. `runpod/full32L/`):

1. `full32L.yaml` — the run config.
2. A probe ladder: generated trial configs + one driver script + a smoke aimed at the AB grid.
3. A launcher for the real run (fixed run id, SIGTERM-save trap, hang watchdog, detached).
4. `full32L-pod-guide.md` — step-by-step pod setup, launch, retrieval, resume, teardown.

Work in `~/Code/param-decomp/dual_obj_jax` (branch `feature/dual_obj_jax`, tip 24c36ae25).
You may edit library code under `param_decomp/` only after confirming with me; configs,
scripts and docs need no confirmation. Read `CLAUDE.md`, `docs/handbook.md`,
`docs/skill.md`, `CONFIGS.md`, `param_decomp/core/SPEC.md`, `PERF_NOTES.md` first.

## Reference material

- `~/pd_scratch/dual_obj_jax/addsub-L18-23-neuronaligned.yaml` + `.sbatch` — the 1-block
  recipe and the source of every scale-independent value. Its header explains each choice.
- `~/pd_scratch/dual_obj_jax/addsub-4L18-21-neuronaligned.yaml` + `.sbatch` — the 4-block
  scale-up (run dir `~/out/runs/by-name/addsub-4L18-21-neuronaligned/`, 4.6 s/step on
  4x L40 at batch 128/96, 30 GB/rank). Shows what was and was not adjusted at 1->4.
- `~/pd_scratch/dual_obj_jax/addsub-4L-scaleup-roadmap.md` and `notes/report.md` §1-§5,
  §8.5, §8.7-8.8, §8.11 — probe-ladder method, mesh measurements, the two AB-grid crashes,
  the silent dp>1 OOM hang. Many workarounds there are L40-specific (no GPU peer-to-peer,
  pre-CUDA-12.8 driver, `attention_implementation: xla`, `fsdp: 1`, allocator env vars):
  for each, decide whether it still applies on H100 SXM and say why.
- `param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b64_dp64.yaml` — the
  full-data 32-block seat, H100-validated. Useful for `remat_*` flags and the chunkwise CI
  fn schema, NOT for its mesh (it shards because its V/U are 18B params; ours are ~1B).
- `param_decomp/core/checkpoint.py`, `param_decomp/core/run.py` (~L670-700) — resume:
  same run id restores optimizers + PPGD sources + step; the pinned launch config
  byte-compares, so nothing may change between legs.
- `param_decomp/experiments/lm/ab_grid_dataset.py` — grids land in
  `<run_dir>/ab_grids/step_<n>.js` + `manifest.js` + `index.html`.
- `~/pd_scratch/tpd_jax_tests/prestage_fineweb64.sbatch`,
  `~/pd_scratch/dual_obj_jax/harvest-neuron-ranks.sbatch` — how the datasets and the
  neuron-ranks artifact were made.

## Decisions (fixed)

**Copied verbatim from the 1-block recipe:** per-site C (q 72, k 72, v 128, o 256,
gate 456, up 456, down 512 in every block), `initialization: neuron_aligned_targeted` with
`neuron_ranks: addsub1-100_llama31-8b_unit-energy`, every loss type and coefficient
magnitude (imp-min, frequency, stochastic/unmasked/PPGD recon, merged nontarget terms,
`adv_fraction` 0.5, source shapes), optimizer betas and grad-clip, both LRs
(3.2e-4 / 1.6e-4 — no LR tuning), `zero_init_readout`, `dual: true`,
`sequential_passes`, the eval metric set, the prompts file, the datasets, wandb project.
Rationale for the coefficients: imp-min and frequency are sums over components, so a fixed
coefficient keeps per-component pressure constant.

**Changed on purpose:** `ci_scaled_weight_decay: null` — the term is REMOVED (reference
0.3; `null` is the schema's off switch, a positive float is required otherwise). Record it
in the header as a deliberate recipe change, not a scale adjustment.

**Sites:** `layers: {kind: all}`.

**Hidden pass:** kept on both streams. Points = every block's MLP output,
`layers.0.mlp.down_proj.out` … `layers.31.mlp.down_proj.out`. The hidden loss is a mean
over points, so coefficients stay as they are; note this in the header. If memory forces
a cut, the nontarget hidden pass is the first thing to drop — tell me before doing so.

**Batch:** target 256, nontarget 128. `eval.batch_size` 128.

**Steps and schedules:** `pd.steps: 40000`. Gamma held at 1.0 until step 30000, then
linear to 0.01 at 40000. Every other schedule keeps the reference's ABSOLUTE step counts:
imp-min and frequency coefficients 4x final at step 0, linear to 1x at step 10000, then
flat (target, hidden, nontarget, nontarget-hidden); PPGD adversary LR ramp 0 -> 0.01 over
the first 500 steps; merged nontarget `adv_fraction` ramp 0 -> 0.5 over the first 100
steps. Components/CI-fn LR: cosine to 0.1x over the full 40k, as the reference. Fast eval
every 500; slow eval (PGD probe, AB grids) every 4000 with `slow_on_first_step: true`.
Checkpoint every 4000, `keep_last 2`. The run must resume on its own id after any
interruption; the guide gives the exact command.

**CI function:** `chunkwise_transformer`, `input_tap: all_block_taps`. Design it yourself:
one block per chunk is the validated shape at 1 and 4 blocks and scales linearly; wider
chunks couple sites across blocks and pay for an input projection over 8x ~26.6k tap dims
(the 14336-wide MLP hidden tap dominates). Budget by parameter count and FLOPs, compare to
the reference's per-component ratio, and pick the cheaper design that keeps capacity per
component at least at the reference's. State the numbers in the header.

**Mesh:** A/B in the ladder (below). Candidates: (a) `replicate 8, fsdp 1, tp 1, ddp` —
the layout measured fastest on the 1-block line; V/U + Adam (~12 GB) + frozen model
(16 GB) fit replicated on 80 GB; (b) `replicate 1, fsdp 8, tp 1, zero1` — the HSDP seat's
layout; NVLink makes the gathers cheap. Do not use `zero1` at `fsdp 1` (26x all-reduce
regression, report §8.5). `attention_implementation: auto` on H100 unless you find a
reason not to.

## Probe ladder (all on the pod, unattended, one summary file)

Total pod time target: under one hour including compiles. Each trial 30 steps with the
step-0 slow eval ON (that is where a long run dies first), under `timeout -k`; a timeout
or hang counts as "does not fit". Read `train/mem/peak_gb_per_rank` and the steady-state
`train/perf/step_time_s` from `metrics.jsonl`. Trials, in order:

1. Mesh (a) at batch 256/128.
2. Mesh (b) at batch 256/128.
3. Only if neither fits: 128/96 on the better mesh.
4. AB-grid smoke at the chosen shape: `eval.every 10, slow_every 20, steps 25,
   mean_ci_floor 0.0`, and shrink the grid's `a_range`/`b_range` to `[1, 10]` so the
   payload is ~100 prompts. The floor MUST be 0.0: the failing gather only runs when
   some components are saved (report §8.8), and at step 20 nothing clears a positive
   floor. Pass = `step_20.js` written, `saved_components/total` > 0, no traceback.

The driver runs trials sequentially, survives shell disconnect, skips trials whose output
exists, keeps the XLA compilation cache on the volume, and writes one markdown table
(trial, fits?, peak GB, s/step, projected hours for 40k steps) I can paste back to you.

## Pod guide (exact commands, in this order)

1. **Pod:** 8x H100 SXM, image/template and minimum CUDA driver for the jax build in
   `uv.lock`, persistent NETWORK VOLUME (budget it: HF cache with Llama weights ~15 GB,
   venv + XLA cache, 2 datasets x 1.7 GB, checkpoints — estimate one 32-block checkpoint
   from the 4-block one's 2.5 GB, times `keep_last` plus trial outputs, plus margin).
   Everything that must survive a pod restart lives on the volume.
2. **Code:** `git clone https://github.com/Antovigo/spd.git -b feature/dual_obj_jax`, pinned
   commit stated; install with uv per `README.md`.
3. **Data:** upload `~/out/datasets/fineweb_llama_tok_64{,_eval}` (1.7 GB each) and
   `~/out/neuron_ranks/addsub1-100_llama31-8b_unit-energy` (6 MB) from this cluster to the
   volume with rsync/scp (placeholders for host/port/key). Regeneration via
   `prestage_tokenized` is the fallback only; give the command. Llama weights download
   from HF on the pod. The loader cycles the dataset in epochs; at batch 128 over 40k
   steps that is ~0.5 epoch of the 11M-row shard, so no extra data is needed.
4. **Secrets:** every token needed (HF with Llama-3.1 access, wandb), where I find each,
   how to set them on the pod without committing them. wandb online is fine.
5. **Layout and env:** the `DATA_ROOT` tree (`runs/`, `datasets/`, `neuron_ranks/`,
   `wandb/`, XLA cache), `PARAM_DECOMP_OUT_DIR`, `WANDB_DIR`, `HF_HOME`, `TMPDIR` (on the
   volume), `XLA_PYTHON_CLIENT_MEM_FRACTION`; entry point
   `python -m param_decomp.experiments.lm.run_targeted <config> <data_root> 8 --run_id <id>`.
6. **Detached running:** I start the ladder or the run, close the shell, shut my laptop,
   and it keeps going. tmux or nohup recipe, reattach, tail logs, plus the log-age
   watchdog and SIGTERM-forwarding trap ported from the reference sbatch scripts (no
   SLURM on the pod).
7. **Retrieval while running:** rsync/scp commands to pull `<run_dir>/ab_grids/`
   (open `index.html` locally), `metrics.jsonl`, `launch_config.yaml`, the ladder summary,
   and optionally a checkpoint (state its size).
8. **Stop / resume / teardown:** clean stop (SIGTERM so it saves), resume on the same run
   id after a pod restart, stop the pod without losing the volume, what to delete at the end.

## Output

- `full32L.yaml` with a header in the reference style: every field that differs from
  `addsub-L18-23-neuronaligned.yaml`, what changed and why; every scale-dependent knob
  deliberately kept, said so.
- Trial configs generated from `full32L.yaml` by a script (not hand-copied), the driver,
  the smoke config, the launcher, the guide.
- Final message: open decisions for me, the CI-fn sizing numbers, and your step-time and
  wall-clock estimate for 40k steps with the assumptions behind it.
