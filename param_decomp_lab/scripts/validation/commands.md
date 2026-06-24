# Validation — sample commands

Self-contained example invocations. Run the setup block once per shell, then paste any
command below. Each command uses `uv run`, which resolves the project venv automatically —
no separate `source .venv/bin/activate` needed.

## Setup

```bash
# The targeted 8B addition run.
MODEL_PATH=~/out/runs/llama8b-add-02/model_20000.pth
RUN_DIR=$(dirname "$MODEL_PATH")
JSON="$RUN_DIR/alive_components_per_position.json"
```

## sample_target_data

For every position of each sampled target sequence, compare the original model's vs. the
circuit's top-5 next-token predictions (circuit = only subcomponents with CI > `--ci-thr`
active, inactive subcomponents and delta off). Long format: one row per
`(sequence, position, model)`.

A forward pass of the 8B target needs a GPU, so submit it to SLURM with `--slurm` (the
login node has none). Output lands in the run folder as `sample_target_data.tsv`.

```bash
# Submit to SLURM (single GPU). Tail the log path it prints to watch progress.
uv run python -m param_decomp_lab.scripts.validation.sample_target_data "$MODEL_PATH" --slurm

# More examples, stricter CI threshold, top-10 tokens:
uv run python -m param_decomp_lab.scripts.validation.sample_target_data "$MODEL_PATH" \
    --n-examples=100 --ci-thr=0.2 --top-n=10 --slurm

# Override SLURM knobs (partition / time / memory):
uv run python -m param_decomp_lab.scripts.validation.sample_target_data "$MODEL_PATH" \
    --slurm --partition=h100 --slurm-time=0:30:00 --slurm-mem=64G
```

To run inline instead (e.g. inside an `srun --pty` shell that already holds a GPU), drop
`--slurm`:

```bash
srun --gpus=1 --time=1:00:00 --pty bash   # then, in the shell it opens:
uv run python -m param_decomp_lab.scripts.validation.sample_target_data "$MODEL_PATH"
```

## find_alive_components

Run every target prompt and record which subcomponents are ever active (CI > `--ci-thr`).
Writes `alive_components.tsv` (one row per alive subcomponent) and
`alive_components_per_prompt.json` (active components per prompt) to the run folder.

```bash
# Submit to SLURM (single GPU).
uv run python -m param_decomp_lab.scripts.validation.find_alive_components "$MODEL_PATH" --slurm

# Lower threshold to catch weakly-firing components:
uv run python -m param_decomp_lab.scripts.validation.find_alive_components "$MODEL_PATH" \
    --ci-thr=0.01 --slurm
```

## ablate_component_groups

Causal test of what each pos-`=` component *family* does. For each named group (defined
in `_GROUPS` in the script — units-digit lattice, sum-bands, operand-magnitude, …) force
exactly that group off in the circuit mask and re-read the single-token answer at `=`,
comparing the predicted integer to `X+Y`. Writes `ablate_component_groups.tsv` (one row
per prompt × condition) to the run folder.

```bash
# Submit to SLURM (single GPU). 1024 random prompts × all groups, ~1 min after load.
uv run python -m param_decomp_lab.scripts.validation.ablate_component_groups "$MODEL_PATH" \
    --slurm --slurm-time=0:20:00
```

## plot_ci_heatmaps / plot_ab_heatmaps

CPU-only — read the per-position JSON from `find_alive_components` and render heatmaps. No
GPU/SLURM needed; run directly on the login node.

```bash
JSON="$RUN_DIR/alive_components_per_position.json"   # RUN_DIR=$(dirname "$MODEL_PATH")

# Prompt × subcomponent heatmaps, faceted by matrix, one PNG per position.
uv run python -m param_decomp_lab.scripts.validation.plot_ci_heatmaps "$JSON"
uv run python -m param_decomp_lab.scripts.validation.plot_ci_heatmaps "$JSON" --grep="2+" --n-prompts=100

# a×b CI grids: rows = matrices, cols = per-position active subcomponents, one PNG per position.
# --op picks the operator (writes to figures/ab_heatmaps_{add,sub}/); --grep filters prompts.
uv run python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps "$JSON" --op=+ --ci-thr=0.5
uv run python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps "$JSON" --op=- --ci-thr=0.5
```

## build_addition_explorer

Interactive, GPU-free HTML explorer for `a+b=` runs: detects each component's periodic
*base* (mod 2/5/10/...) via an η² residue-variance fingerprint, and reads the gate/up/down
neuron-space overlap from the checkpoint U/V (mmap, CPU-only — no forward pass). Writes a
self-contained `index.html` + `data.js` (open from `file://`, no server/CDN/GPU) into
`figures/addition_explorer/`. Reads the `find_alive_components` JSON next to the checkpoint.

```bash
uv run python -m param_decomp_lab.scripts.validation.build_addition_explorer "$MODEL_PATH"
uv run python -m param_decomp_lab.scripts.validation.build_addition_explorer "$MODEL_PATH" --no-weights
```

## collect_ablation_kl

Per-component ablation effect on every `a+b=` prompt — a cleaner metric than CI. Ablates
each alive subcomponent (removes `U_c V_c^T`) and records, at the `=` position: KL of the
next-token distribution vs the un-ablated model, the ablated argmax token + prob, the
normalized inner activation `(x·V_c)/||V_c||`, and CI. Reference = all components + delta on
(= exact reconstruction of the target). 8B forward → SLURM.

```bash
uv run python -m param_decomp_lab.scripts.validation.collect_ablation_kl "$MODEL_PATH" --slurm
# smoke test first (subset, short):
uv run python -m param_decomp_lab.scripts.validation.collect_ablation_kl "$MODEL_PATH" \
    --max-prompts=256 --max-components=12 --output-dir="$RUN_DIR/ablation_kl_smoke" --slurm --slurm-time=0:20:00
```

## build_arith_ablation_explorer

GPU-free HTML explorer over `collect_ablation_kl`'s `data.npz` (Objective 2). Detects each
component's period by autocorrelation of the ablation-KL marginal (spiky, non-sinusoidal),
and packs the (a,b) grids of all five switchable color metrics (CI, ablation KL, inner
activation, original token, ablated token). Writes `index.html` + `data.js` to
`figures/arith_ablation_explorer/`.

```bash
uv run python -m param_decomp_lab.scripts.validation.build_arith_ablation_explorer "$RUN_DIR"
```

## headless_check

Smoke-test an HTML applet (e.g. the addition explorer) in headless Chromium — no display,
no GPU. Fails on any JS console error / uncaught exception, clicks through selectors, and
screenshots each step. Runs in its own toolchain venv (NOT `uv run`), built once by
`headless_setup.sh` (Playwright + Chromium + the system libs bare login nodes lack, all
without root).

```bash
bash param_decomp_lab/scripts/validation/headless_setup.sh   # once; idempotent

PY=~/.cache/pd-headless/venv/bin/python
APP=~/out/runs/llama8b-add-refine-treat-01/figures/addition_explorer/index.html
$PY param_decomp_lab/scripts/validation/headless_check.py "$APP" \
    --clicks='[data-view=bases];;[data-view=interplay];;[data-view=inspector];;[data-view=gallery]' \
    --probes="document.querySelectorAll('#gallery .card').length"
```

## screen_components_on_data

Screen the broad **nontarget** distribution (fineweb) to find natural-text contexts where
the L18 components fire. Records per-component firing frequency (`screen_components_on_data.tsv`)
and top-`--top-k` max-activating contexts (`screen_components_on_data.jsonl`). Checkpoints
both every 50 batches, so a wall-clock kill still leaves usable partial output.

8B forward over many tokens is the cost — keep `--n-batches` modest (≈300 × bs 128 ≈
2.5M tokens ≈ 20 min on one L40). Submit to SLURM:

```bash
uv run python -m param_decomp_lab.scripts.validation.screen_components_on_data "$MODEL_PATH" \
    --n-batches=300 --slurm --slurm-time=0:28:00
```

## Arithmetic analysis pipeline (`roadmap_addition_analysis`)

One operation at a time (`add` / `sub` / `mult`) over the `1..100 × 1..100` grid. Reference
run `addmult-L18-03`. Set `OP=add` and run top to bottom; swap `OP` for `sub` / `mult`.

```bash
MODEL_PATH=~/out/runs/addmult-L18-03/model_28000.pth
RUN_DIR=$(dirname "$MODEL_PATH")
OP=add
V=param_decomp_lab.scripts.validation
```

### 0. Ever-alive set on the original data (existing script, GPU — run once)

`find_alive_components` finds subcomponents ever causally important on the run's **original**
distribution; it is op-agnostic and writes the **unsuffixed** `alive_components.tsv` +
`alive_components_per_position.json`. Run it once with defaults (no `--prompts` / `--output`);
the arithmetic scripts read these and apply the per-op + last-position + mean-CI filtering
themselves.

```bash
uv run python -m $V.find_alive_components "$MODEL_PATH" --slurm --slurm-time=0:30:00
```

### 1-2. Hidden + inner activations (GPU)

```bash
uv run python -m $V.collect_hidden_activations "$MODEL_PATH" --op=$OP --slurm --slurm-time=0:30:00
uv run python -m $V.collect_inner_activations  "$MODEL_PATH" --op=$OP --slurm --slurm-time=0:30:00
# collect_inner_activations writes inner_activations_<op>.tsv + alive_filtered_<op>.tsv
# (lower --mean-ci-thr to widen the alive set; default 0.1 is fairly strict).
```

### 3-5. Periods, cosine heatmaps, explorer (CPU — run on the login node)

```bash
uv run python -m $V.compute_subcomp_periods "$RUN_DIR/inner_activations_$OP.tsv"
uv run python -m $V.plot_subcomp_cosine "$MODEL_PATH" --op=$OP
# inner-activation (a,b) heatmaps — same layout as plot_ab_heatmaps, written next to it:
uv run python -m $V.plot_ab_inner_heatmaps "$RUN_DIR/inner_activations_$OP.tsv"
uv run python -m $V.build_neuron_connection_explorer "$MODEL_PATH" --op=$OP
# lower --conn-floor (default 0.1) to let the UI threshold reach weaker connections,
# at the cost of a larger data.js. The applet's "hover shows" toggle switches the
# subcomponent heatmap between causal importance and (signed) inner activation.
```

### 6. Real dimensionality of the representation (CPU)

Projects the stored `mlp_input` / `mlp_output` onto the alive subcomponents' read / write
subspaces and reports the real dimensionality (geometric rank, PCA participation ratio,
TwoNN intrinsic dim) + the variance-captured completeness check. Writes
`dimensionality_<op>.{npz,json}` and a self-contained Plotly applet.

```bash
uv run python -m $V.reduce_dimensionality "$MODEL_PATH" --op=$OP
# open figures/dimensionality_<op>/index.html in a real browser (3D is WebGL).
```

### 7. Independent subspaces (ISA, CPU)

Runs ICA on the reduced `z` and groups components into independent subspaces by energy
correlation (so circular features stay one block), checking near-orthogonality via principal
angles. Writes `independent_subspaces_<op>.json` and a self-contained Plotly applet.

```bash
uv run python -m $V.find_independent_subspaces "$MODEL_PATH" --op=$OP
# tune --var-keep (PCs before ICA) and --group-distance (subspace cut) if grouping looks off;
# it warns if FastICA didn't converge.
```

Smoke-test the explorer applet in headless Chromium (see `headless_check` below):

```bash
PY=~/.cache/pd-headless/venv/bin/python
$PY param_decomp_lab/scripts/validation/headless_check.py \
    "$RUN_DIR/figures/neuron_explorer_$OP/index.html" --wait-ms=2000 --timeout-ms=30000 \
    --probes="document.querySelectorAll('.node').length;;document.getElementById('hint').textContent"
```
