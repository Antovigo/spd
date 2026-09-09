# addsub-all-layers-4xh100-01 on a Runpod 4x H100 SXM pod — step by step

The same full-network (32-block, 224-site) targeted dual-objective decomposition of
Llama-3.1-8B as `addsub-all-layers-guide.md`, on **four** H100 SXM 80 GB instead of eight,
at batch **128 target / 128 non-target**. Config: `addsub-all-layers-4xh100-sota.yaml`.
Run id `p-b4132c01`, run name `addsub-all-layers-4xh100-01`.

**This guide is the delta.** Steps 1, 3, 4, 7 and 8 of `addsub-all-layers-guide.md` (pod
prep, data and weights, secrets, retrieval, teardown) apply unchanged except where noted
here, and that guide's "If the image is missing tools" section covers pods without `uv`,
`apt-get`, `git` or `rsync`. Everything below is what differs.

Code: `https://github.com/Antovigo/spd.git`, branch `feature/dual_obj_jax`. Same caution as
the 8-GPU guide: the run's `launch_config.yaml` is byte-compared on every resume but the
code is not, so avoid pushing to the branch while the run is in flight.

---

## Why this variant exists, and what it costs

Runpod prices GPUs linearly, so four H100s cost $13.96/h against $27.92/h for eight, at the
same price per unit of compute. Four is also easier to get. The catch is that halving the
devices doubles the per-rank batch, and **this codebase has no gradient accumulation**:
`pd.batch_size` is GLOBAL and shards over the data mesh, so the global batch is the only
lever. Carrying the 8-GPU config's 256/128 onto four cards would put 64 target prompts and
32 broad rows on each rank, which the memory estimate does not support.

So the target batch halves to 128. That is not purely a concession: every run on this line,
the one-block seat and both four-block scale-ups included, trained at `pd.batch_size: 128`.
The 8-GPU file's 256 is the scale-up brief's untested bet that a bigger batch helps at 32
blocks. This variant declines that bet rather than testing it.

**Read the two runs as different experiments.** At equal steps they see different amounts of
target data, so do not compare a curve from this run against one from the 8-GPU config
step-for-step.

The non-target batch stays at 128 (32 rows per rank), per Antoine 2026-09-09. That is where
the memory risk sits, because the broad stream carries seq 64 against the pool's 5 tokens
and therefore dominates the activation term. If it does not fit, the ladder escalates to
128/96 on its own; nothing is manual.

---

## 1. Pod

As step 1 of the 8-GPU guide, with two changes.

**GPU:** 4x H100 SXM 80 GB, one node, NVLink. Driver r570 or newer, for the same reason
(`attention_implementation: auto` uses the cuDNN graph API). Verify before installing:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # 4 rows, driver >= 570
```

**Network volume:** the budget is unchanged at roughly 185 GB used, so **250 GB**. The run's
checkpoints still dominate at about 20 GB each with `keep_last 2`, and the ladder's two
smokes hold about 80 GB more until you delete them. Checkpoints are topology-free, so their
size does not change with the device count.

## 2. Code and environment

Identical to the 8-GPU guide, steps 2 and 5. `env.sh` is shared; nothing in it is
profile-specific.

```bash
cd /workspace
git clone -b feature/dual_obj_jax https://github.com/Antovigo/spd.git spd
cd spd && git log -1 --oneline
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
source notes/dual_objective/addsub-all-layers/env.sh
uv python install 3.12
uv sync --frozen --no-dev --extra cuda
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # 0.10.1, 4x CudaDevice
```

Note the entry point takes the device count as a positional argument, so it is `4` here, not
`8`. The scripts derive it from the config's own mesh and refuse a mismatch, so you never
type it:

```bash
python -m param_decomp.experiments.lm.run_targeted <config> $DATA_ROOT 4 --run_id p-b4132c01
```

Sanity check before spending GPU time, which parses and builds every config for this profile
against the pod's data root:

```bash
cd $REPO/notes/dual_objective/addsub-all-layers
$VENV_PY make_trials.py --profile 4xh100 --check --data-root $DATA_ROOT
```

## 3. Data, weights, secrets

Unchanged. Follow steps 3 and 4 of the 8-GPU guide. The datasets, the neuron-ranks artifact
and the Llama weights do not depend on the device count, and if you already staged them on
this volume for the 8-GPU route they are reused as they are.

---

## 4. Probe ladder

Everything is driven by `--profile 4xh100`, which selects the 4-GPU config, its own trials
directory, its own ladder directory and its own run ids. The two profiles can share a
`$DATA_ROOT` without colliding.

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd $REPO/notes/dual_objective/addsub-all-layers
./run_ladder.sh --profile 4xh100 --detach
tail -f $DATA_ROOT/ladder-4xh100/driver.log
```

Each trial is 30 steps with the step-0 slow eval ON, under `timeout -k`, so a timeout or a
hang counts as "does not fit". The order:

1. `mesh-f4-b128x128` — `replicate 1 / fsdp 4 / zero1`, the natural HSDP layout
2. `mesh-r2f2-b128x128` — `replicate 2 / fsdp 2 / zero1`, half the frozen-target gathers,
   bought with more static memory
3. **only if neither fits**, the same two meshes again at 128/96, automatically
4. `smoke-abgrid-<winner>` — aimed at the AB grid's own schedule: `eval.every 10`,
   `slow_every 20`, `mean_ci_floor 0.0`, grid `[1,10]^2`, checkpointing ON. Pass =
   `ab_grids/step_20.js` written, `saved_components/total` > 0, no traceback
5. `smoke-resume-<winner>` — `save_every 20`; the driver SIGTERMs the trainer after the
   step-20 log, expects a SIGTERM save, relaunches on the same run id, expects
   `resumed from checkpoint step N` and a clean finish at step 30

The static-memory arithmetic the ladder is testing, per rank at `mem_fraction 0.95`:

| mesh | trainable (÷4) | frozen target | grad accumulator | static | left for activations |
|---|---|---|---|---|---|
| `f4` = 1x4 zero1 | 5.6 GB | 4.0 GB (÷4) | 1.9 GB | 11.4 GB | ~64 GB |
| `r2f2` = 2x2 zero1 | 5.6 GB | 8.0 GB (÷2) | 1.9 GB | 15.4 GB | ~60 GB |
| `ddp` = 4x1 (extra) | 22.3 GB | 16.0 GB | 7.4 GB | 45.7 GB | ~30 GB |

Under `zero1` the optimizer state shards over the full 4-wide data mesh in every row, so the
meshes differ in how the frozen target is sharded and how often it is gathered, not in
optimizer footprint. `ddp` is generated but never in the default order; run it with
`./run_ladder.sh --profile 4xh100 --only mesh-ddp-b128x128` if you want the number. Never
`zero1` at `fsdp: 1`: the config builder accepts it, so that guard is lore rather than code,
and on a degenerate sharding axis it measured 26 times the all-reduce volume.

Extrapolating the four-block L40 run's roughly 10.5 GB of activations at 28 live sites and
32/24 per rank to 224 sites gives 30 to 60 GB here, depending on how much of that term is
site-proportional. It should fit, but not by much, which is exactly why step 3 exists.

Output is one table at `$DATA_ROOT/ladder-4xh100/summary.md`. Paste it back. To redo a
trial, delete its `.rc` file AND its run dir: run ids are fixed per trial, derived from the
trial's name, and listed in `trials-4xh100/manifest.tsv`.

## 5. The real run

Two edits before launching, and they must happen now, because the pinned
`launch_config.yaml` is byte-compared on every resume:

1. Set `runtime.replicate` and `runtime.fsdp` in `addsub-all-layers-4xh100-sota.yaml` to the
   winning mesh. Authored as `1 / 4`; `r2f2` is `2 / 2`.
2. If the ladder escalated, set `nontarget.batch_size` to 96 and `eval.batch_size` to 96 to
   match the shape that actually fit.

Then re-check and free the volume:

```bash
$VENV_PY make_trials.py --profile 4xh100 --check --data-root $DATA_ROOT
awk -F'\t' '$4 ~ /smoke/ {print $2}' trials-4xh100/manifest.tsv | while read -r id; do rm -rf "$DATA_ROOT/runs/$id"; done
./launch_run.sh --profile 4xh100
tail -f $DATA_ROOT/logs/addsub-all-layers-4xh100-01.latest.log
```

A healthy launch prints `targeted run addsub-all-layers-4xh100-01 | 4 GPU / 1 proc | target
B=128 nontarget B=128 seq=64 sites=224 steps=40000`, then goes quiet for the compile, then
the step-0 slow eval, `checkpoint saved @ step 0`, and a `[step N]` line every 100 steps. AB
grids appear at step 4000. The run detaches with `setsid nohup`, so close the shell freely.

**Expected wall clock.** Per-rank work here is the 8-GPU plan's target geometry with twice
its broad rows, and the broad stream dominates the step, so expect roughly 6 to 10 s/step
against the 8-GPU estimate of 4 to 6. That is 3 to 5 days for 40k steps including the
measured 6% eval and checkpoint overhead, so **$950 to $1550** at $13.96/h. The eight-GPU
route was 2 to 3 days at $1340 to $2010. You are trading calendar time for money, and taking
on more resume legs.

## 6. Stop, resume, teardown

As step 8 of the 8-GPU guide, with `--profile 4xh100` on both scripts:

```bash
./stop_run.sh --profile 4xh100      # SIGTERM -> checkpoint save -> exit, then waits
./launch_run.sh --profile 4xh100    # resumes on p-b4132c01 from the newest checkpoint
```

A resume needs the config byte-identical to the run dir's pinned copy and a pod with **4**
GPUs, since the mesh is in the config. Everything else, including retrieval of `ab_grids/`,
`metrics.jsonl` and the checkpoint, is identical to the 8-GPU guide's steps 7 and 8; only
the run id changes, to `p-b4132c01`.
