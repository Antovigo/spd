# Validation — sample commands

Self-contained example invocations. Run the setup block once per shell, then paste any
command below. Each command uses `uv run`, which resolves the project venv automatically —
no separate `source .venv/bin/activate` needed.

## Setup

```bash
# The targeted 8B addition run.
MODEL_PATH=~/out/runs/llama8b-add-02/model_20000.pth
RUN_DIR=$(dirname "$MODEL_PATH")
# Analysis artifacts live under <run>/analysis/: figures + applets directly in it, shared
# datasets in analysis/datasets/. (figures/ is reserved for training-loop figures.)
DATASETS="$RUN_DIR/analysis/datasets"
JSON="$DATASETS/alive_subcomponents_per_position.json"
```

## sample_target_data

For every position of each sampled target sequence, compare the original model's vs. the
circuit's top-5 next-token predictions (circuit = only subcomponents with CI > `--ci-thr`
active, inactive subcomponents and delta off). Long format: one row per
`(sequence, position, model)`.

A forward pass of the 8B target needs a GPU, so submit it to SLURM with `--slurm` (the
login node has none). Output lands in `analysis/datasets/` as `sample_target_data.tsv`.

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

## find_alive_subcomponents

The reference alive set + CI-ranked sufficiency curve. Ranks all subcomponents by mean
lower-leaky CI at the last (`=`) position, sweeps top-k subsets (rest hard zero at `=`,
delta fully on) and measures last-position KL / argmax agreement vs the raw target model.
The alive subset is the top-k for the smallest swept k with mean KL <= `--kl-thr`
(default 0.008). Every downstream script reads this output.

Outputs (in the run's `analysis/` layout):
- `datasets/alive_subcomponents.tsv` — the reference alive list
- `datasets/alive_subcomponents_curve.tsv` + `datasets/alive_subcomponents_kl.npz` —
  the sweep, for re-thresholding without a GPU
- `datasets/alive_subcomponents_per_position.json` — per-(prompt, position) CI of the
  alive subcomponents above `--ci-thr` (the input to the CI/AB heatmap plots and applets)
- `alive_subcomponents/recon_vs_k.png` — the KL-vs-sparsity curve with the alive cut marked

```bash
# Submit to SLURM (single GPU).
uv run python -m param_decomp_lab.scripts.validation.find_alive_subcomponents "$MODEL_PATH" --slurm

# Looser KL threshold, dense k grid around the knee:
uv run python -m param_decomp_lab.scripts.validation.find_alive_subcomponents "$MODEL_PATH" \
    --kl-thr=0.02 --ks=0,8,16,32,64,128 --slurm
```

## ablate_component_groups

Causal test of what each pos-`=` component *family* does. For each named group (defined
in `_GROUPS` in the script — units-digit lattice, sum-bands, operand-magnitude, …) force
exactly that group off in the circuit mask and re-read the single-token answer at `=`,
comparing the predicted integer to `X+Y`. Writes `ablate_component_groups.tsv` (one row
per prompt × condition) to `analysis/datasets/`.

```bash
# Submit to SLURM (single GPU). 1024 random prompts × all groups, ~1 min after load.
uv run python -m param_decomp_lab.scripts.validation.ablate_component_groups "$MODEL_PATH" \
    --slurm --slurm-time=0:20:00
```

## plot_ci_heatmaps / plot_ab_heatmaps

CPU-only — read the per-position JSON from `find_alive_subcomponents` and render heatmaps.
No GPU needed, but submit as a (CPU) SLURM job anyway to keep load off the login node.

```bash
JSON="$DATASETS/alive_subcomponents_per_position.json"   # DATASETS="$RUN_DIR/analysis/datasets"

# Prompt × subcomponent heatmaps, faceted by matrix, one PNG per position.
uv run python -m param_decomp_lab.scripts.validation.plot_ci_heatmaps "$JSON"
uv run python -m param_decomp_lab.scripts.validation.plot_ci_heatmaps "$JSON" --grep="2+" --n-prompts=100

# a×b CI grids: rows = matrices, cols = per-position active subcomponents, one PNG per position.
# --op picks the operator (writes to analysis/ab_heatmaps_{add,sub}/); --grep filters prompts.
uv run python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps "$JSON" --op=+ --ci-thr=0.5
uv run python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps "$JSON" --op=- --ci-thr=0.5
```

## build_addition_explorer

Interactive, GPU-free HTML explorer for `a+b=` runs: detects each component's periodic
*base* (mod 2/5/10/...) via an η² residue-variance fingerprint, and reads the gate/up/down
neuron-space overlap from the checkpoint U/V (mmap, CPU-only — no forward pass). Writes a
self-contained `index.html` + `data.js` (open from `file://`, no server/CDN/GPU) into
`analysis/addition_explorer/`. Reads the `find_alive_subcomponents` JSON from `analysis/datasets/`.

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
    --max-prompts=256 --max-components=12 --output-dir="$DATASETS/ablation_kl_smoke" --slurm --slurm-time=0:20:00
```

## build_arith_ablation_explorer

GPU-free HTML explorer over `collect_ablation_kl`'s `data.npz` (Objective 2). Detects each
component's period by autocorrelation of the ablation-KL marginal (spiky, non-sinusoidal),
and packs the (a,b) grids of all five switchable color metrics (CI, ablation KL, inner
activation, original token, ablated token). Writes `index.html` + `data.js` to
`analysis/arith_ablation_explorer/`.

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
APP=~/out/runs/llama8b-add-refine-treat-01/analysis/addition_explorer/index.html
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

### 0. Reference alive set on the original data (GPU — run once)

`find_alive_subcomponents` (see its section above) produces the reference alive list on
the run's **original** distribution; it is op-agnostic and writes the **unsuffixed**
`alive_subcomponents.tsv` + `alive_subcomponents_per_position.json`. Run it once with
defaults (no `--prompts` / `--output`); the arithmetic scripts read these and apply the
per-op + last-position + mean-CI filtering themselves.

```bash
uv run python -m $V.find_alive_subcomponents "$MODEL_PATH" --slurm
```

### 1-2. Hidden + inner activations (GPU)

```bash
uv run python -m $V.collect_hidden_activations "$MODEL_PATH" --op=$OP --slurm --slurm-time=0:30:00
uv run python -m $V.collect_inner_activations  "$MODEL_PATH" --op=$OP --slurm --slurm-time=0:30:00
# collect_inner_activations writes inner_activations_<op>.tsv + alive_filtered_<op>.tsv to analysis/datasets/
# (lower --mean-ci-thr to widen the alive set; default 0.1 is fairly strict). For mult, the
# log-periodic components fire strongly but only on a few % of prompts, so use a low threshold:
#   ... --op=mult --mean-ci-thr=0.02
```

### 3-5. Periods, cosine heatmaps, explorer (CPU — run on the login node)

```bash
uv run python -m $V.compute_subcomp_periods "$DATASETS/inner_activations_$OP.tsv"
uv run python -m $V.plot_subcomp_cosine "$MODEL_PATH" --op=$OP
# inner-activation (a,b) heatmaps — same layout as plot_ab_heatmaps, into analysis/ab_heatmaps_<op>/:
uv run python -m $V.plot_ab_inner_heatmaps "$DATASETS/inner_activations_$OP.tsv"
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
# open analysis/dimensionality_<op>/index.html in a real browser (3D is WebGL).
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

### 8. Pick-3-subcomponent subspace scatter (CPU)

A Plotly applet spanning every available task: pick up to 3 subcomponents from a thumbnail
grid (organised into task → period sections); a points selector chooses which single task's
activations are scattered onto those 3 directions, coloured by result (a+b / a×b …) / a / b.
A **side** selector (input → pre-nonlinearity → post-nonlinearity → output) chooses which MLP
activation each direction projects (and which matrices the picks come from: up/gate for
input & pre-nonlinearity, down for post-nonlinearity & output).
Auto-detects the tasks with a `hidden_activations_<op>.npz` + `alive_filtered_<op>.tsv`.

```bash
uv run python -m $V.build_subspace_scatter "$MODEL_PATH"            # all detected tasks
uv run python -m $V.build_subspace_scatter "$MODEL_PATH" --ops=add,mult
# open analysis/subspace_scatter/index.html in a real browser (3D is WebGL).
```

### 10. Neuron investigator (CPU)

A self-contained HTML applet to investigate which neurons take part in the task. Left half:
a neuron × subcomponent heatmap of the interaction score (the std over the target grid of what
each subcomponent writes to / reads from each neuron; subcomponents sorted by period → matrix →
period-confidence; neurons by total interaction score per frequency, paged; write blue / read
red, with an input/output/all + threshold neuron filter). Clicking a cell selects a (neuron,
subcomponent) pair;
the right half shows that subcomponent's inner-activation `(a,b)` heatmap and the neuron's up /
gate / output heatmaps — plus, for write (gate/up) subcomponents, its contribution to the
neuron's gate/up and the counterfactual without it — with an **operation toggle** to re-render them on a different task's
activations (add / sub / mult — any op with a saved npz), a **plot-size** control (px per operand
value; plots pack as many per row as fit), and a **drag-resizable divider** between the panels.

```bash
uv run python -m $V.build_neuron_investigator "$MODEL_PATH" --op=$OP
# raise --top-neurons (default 512) for deeper paging at the cost of data.js size.
# open analysis/neuron_investigator_<op>/index.html in a real browser.
```

Smoke-test the explorer / investigator applets in headless Chromium (see `headless_check`):

```bash
PY=~/.cache/pd-headless/venv/bin/python
$PY param_decomp_lab/scripts/validation/headless_check.py \
    "$RUN_DIR/analysis/neuron_explorer_$OP/index.html" --wait-ms=2000 --timeout-ms=30000 \
    --probes="document.querySelectorAll('.node').length;;document.getElementById('hint').textContent"
$PY param_decomp_lab/scripts/validation/headless_check.py \
    "$RUN_DIR/analysis/neuron_investigator_$OP/index.html" --clicks='#matrix' \
    --probes="document.querySelectorAll('#grids canvas').length"
```

### 11. Per-prompt accuracy, original vs ablated (GPU)

Probability the model puts on the correct answer (and on each wrong answer in a `±--range`
window) for every `a<op>b=` prompt, on the all-on reconstruction and with subcomponents ablated.
Writes `<run_dir>/model_accuracy/accuracy[_<ablation>].json` (filename suffixed with the ablated
subcomponents) plus a marimo notebook to plot them. GPU — submit with `--slurm`.

```bash
CKPT=~/out/runs/llama8b-add-02/model_20000.pth
# original model (no ablation)
uv run python -m $V.measure_model_accuracy "$CKPT" --slurm
# gate-163 + down-240 from L18's MLP ablated (range defaults to ±5)
uv run python -m $V.measure_model_accuracy "$CKPT" --ablate=gate_proj:163,down_proj:240 --slurm
# plot: pip/uv install marimo, then
marimo edit ~/out/runs/llama8b-add-02/model_accuracy/model_accuracy_notebook.py
```

### 13. Fourier (circular) features around L18's MLP (CPU)

Recovers the plane + center of the circular feature the model uses — the operands `a`, `b` in the
post-RMSNorm MLP input and the task result in the MLP output (Feucht et al. 2026). CPU-only: it
consumes the `hidden_activations_<op>.npz` grids from step 1, so collect those first (one per
task). Fit separately per task; writes `~/out/runs/fourier_features/coordinates_<op>.json`
(`features[side][variable][period]` → `{period, r2, offset, cos, sin}`, each vector `d_model`-long).

For **add/sub** the fit is in linear space at `T ∈ {2,5,10,20,50,100}`. For **mult** it fits in
**log space** (`θ = 2π·log(v)/log(r)`) — multiplication is periodic in `log v`. Which ratios? Find
them first with `find_log_periods` (13a), then let `find_fourier_features` read the canonical
clustered ratios straight from the sibling `subcomp_periods_mult.tsv`.

```bash
for OP in add sub mult; do
  uv run python -m $V.find_fourier_features \
    ~/out/runs/llama8b-add-02/analysis/datasets/hidden_activations_$OP.npz
done
# override the space/periods if needed (mult defaults to log + the TSV ratios):
uv run python -m $V.find_fourier_features "$NPZ_MULT" --space=log --periods=1.26,2.0
```

### 13a. Find the log-space periods for multiplication (CPU)

Multiplication has no integer period; the operand is encoded as a **circle whose phase advances
with `log v`**. This finds those circles without any frequency grid: per value, average out the
nuisance operand, detrend (DC + linear-in-`log v`), SVD to get the dominant 2D plane, and read the
log-period off how fast the phase winds per unit `log v` (`P = 2π/median ω`, ratio `e^P`).
Diagnostics (`sv_ratio≈1`, `radius_cv≈0`, `omega_cv≈0`) flag a genuine circle. Writes a figure +
JSON to `~/out/runs/fourier_features/log_periods_mult.{png,json}`. The clean result is the second
operand `b` at ratio ≈×1.26 (matching the `subcomp_periods_mult.tsv` clusters).

```bash
uv run python -m $V.find_log_periods \
  ~/out/runs/addmult-L18-03/analysis/datasets/hidden_activations_mult.npz --v-min=10
```

### 14. Fourier-feature scatter applet — subcomponents vs neurons (CPU)

Interactive applet (canvas, no CDN/GPU): scatters one task's activations projected onto another
task's Fourier plane, one plot per period, side by side. Dropdowns for **activation task**, **basis
task**, **operand** (`a` / `b` / result), **colour by** (`a`/`b`/result) with a **mod** + **offset**
form (colours by `(value−offset) mod m` on a cyclic wheel, like the subspace-scatter applet), and
**overlay** (subcomponent unit directions, or individual gate/up read-row / down write-column
neuron directions) with a **threshold** on in-plane norm. A **CI (selected)** colour option (when
`inner_activations_<op>.tsv` has a `ci` column — rerun `collect_inner_activations`) paints points by
the selected subcomponent's causal importance per prompt. Arrows start at the activation-space zero
and a marker shows the Fourier circle centre, so an off-zero centre is visible. Scroll to zoom,
drag to pan; hover an arrow for its label + ‖proj‖; click a subcomponent arrowhead to see its
inner-activation `(a, b)` heatmaps (one per task). Reads the bases from `find_fourier_features`'
`~/out/runs/fourier_features/coordinates_<op>.json` (override with `--coordinates-dir`), plus the
run's `hidden_activations_<op>.npz` / `alive_filtered_<op>.tsv` / `subcomp_periods_<op>.tsv` and the
checkpoint U/V + target MLP weights. Writes `<run_dir>/analysis/fourier_scatter/{index.html,data.js}`.

```bash
CKPT=~/out/runs/addmult-L18-03/model_28000.pth
uv run python -m $V.build_fourier_scatter "$CKPT"        # add,mult auto-detected (npz + basis present)
# smoke-test in headless Chromium:
PY=~/.cache/pd-headless/venv/bin/python
$PY $V_DIR/headless_check.py ~/out/runs/addmult-L18-03/analysis/fourier_scatter/index.html
```

### 14b. Result feature construction — before/after MLP18 with in-browser ablation (CPU)

Does MLP 18 build the `a+b` circular features or add to pre-existing structure — and which
neurons / subcomponents build them? One plot per canonical period on the residual-stream
Fourier probes (`runs/fourier_probes/probes_{pre,post}.json`), five linked rows: pre-MLP
residual on the pre-fit probes, pre-MLP residual on the post-fit probes, post-MLP residual on
the post probes, the same with **one** ablated **neuron** (measured max KL > 0.01, from the
census) or **subcomponent** (the alive set, restricted to those whose last-token CI reaches
`--last-ci-thr` = 0.01 on ≥1 prompt; a period dropdown filters the list) — always exact on
the full grid — and the **alive-components-only MLP** (binary mask, delta off). Hovering a
point marks the same prompt in every plot; the ablated item draws red read-direction arrows
on row 1 (pre frame) and write-direction arrows on row 2 (post frame); colour modes include
the ablation displacement and the signed alignment with the item's read/write directions.
Search items by id (`12023` / `g124`).

```bash
CKPT=~/out/runs/addsub-L18-04-8x-beta0.75-LR/model_24000.pth
uv run python -m $V.build_result_feature_construction "$CKPT"
# smoke-test:
PY=~/.cache/pd-headless/venv/bin/python
$PY param_decomp_lab/scripts/validation/headless_check.py \
    "$(dirname "$CKPT")/analysis/result_feature_construction/index.html" \
    --probes="document.querySelectorAll('#grid canvas').length;;document.querySelectorAll('#items .item').length"
```

### 15. Polytope explorer (CPU)

A self-contained HTML applet mapping the operand grid into SwiGLU gate-sign polytopes. The
`(a, b)` map is coloured by which combination of **alive gates** (gate preactivation > 0 —
alive = takes both signs on the op's grid, top `--top-gates` kept by output relevance) — or,
in a second mode, of **causally-important subcomponents** (CI > thr) — is active on each
prompt; one colour = one combination = one region where the MLP is roughly linear. A top-k
control + per-thumbnail checkboxes choose the combination bits; rare combinations pool into
grey; hovering a legend row highlights its region; hovering / clicking a map pixel shows
which gate / subcomponent `(a, b)` thumbnails are active there. Operation selector across
every op with saved `hidden_activations_<op>.npz` + `alive_filtered_<op>.tsv` +
`inner_activations_<op>.tsv` (the last must carry the `ci` column).

```bash
uv run python -m $V.build_polytope_explorer "$MODEL_PATH"                # all detected ops
uv run python -m $V.build_polytope_explorer "$MODEL_PATH" --ops=add,mult --top-gates=64
# open analysis/polytope_explorer/index.html; smoke-test headless:
PY=~/.cache/pd-headless/venv/bin/python
$PY param_decomp_lab/scripts/validation/headless_check.py \
    "$RUN_DIR/analysis/polytope_explorer/index.html" \
    --probes="document.querySelectorAll('#legend .legrow').length"
```

### 16. Neuron census — L18 neurons over the 0..200 add/sub grids (GPU + CPU)

Decomposition-free: probes the frozen base model's L18 MLP neurons. Outputs land in
`~/out/runs/neurons/` (the shared census dir), not a run's `analysis/`.

```bash
N=param_decomp_lab.scripts.validation.neurons
CKPT=~/out/runs/addsub-L18-04-4x/model_24000.pth

# activations + mlp_input + answer baseline, both ops (~30 min, 1 GPU)
uv run python -m $N.collect_neuron_activations "$CKPT" --slurm

# dense ablation-KL screen: all 14336 neurons x 41x41 subgrid, one op per job (~2 h each)
uv run python -m $N.collect_neuron_ablation_kl "$CKPT" --op=add --stride=5 --slurm --slurm-time=6:00:00
uv run python -m $N.collect_neuron_ablation_kl "$CKPT" --op=sub --stride=5 --slurm --slurm-time=6:00:00

# full 201x201 grid for the screened candidates, sharded (after candidates.tsv exists)
uv run python -m $N.collect_neuron_ablation_kl "$CKPT" --op=add --stride=1 \
    --neurons-tsv=~/out/runs/neurons/candidates.tsv --shard-index=0 --shard-count=2 --slurm
```

```bash
# after the screens land: candidates, periodicity, subspace, applet (all CPU)
uv run python -m $N.select_candidate_neurons --kl-thr=0.01
uv run python -m $N.compute_neuron_periodicity ~/out/runs/neurons/activations_add.npz
uv run python -m $N.compute_neuron_periodicity ~/out/runs/neurons/activations_sub.npz
uv run python -m $N.compute_neuron_subspace "$CKPT" --candidates-tsv=~/out/runs/neurons/candidates.tsv
uv run python -m $N.build_neuron_census "$CKPT"
PY=~/.cache/pd-headless/venv/bin/python
$PY param_decomp_lab/scripts/validation/headless_check.py ~/out/runs/neurons/applet/index.html

# subcomponent phase (reference run addsub-L18-04-2x-beta0.75-LR, addition only)
SCKPT=~/out/runs/addsub-L18-04-2x-beta0.75-LR/model_20000.pth
uv run python -m $N.collect_subcomp_ablation_kl "$SCKPT" --op=add --stride=5 --slurm   # screen, all C
uv run python -m $N.collect_subcomp_ablation_kl "$SCKPT" --op=add --stride=1 --slurm \
    --components-tsv=<causal-components.tsv>                                            # full grid
uv run python -m $N.compute_subcomp_neuron_links "$SCKPT"                               # CPU links + R²
```

```bash
# subcomponent applet (after subcomp screen/full + links exist)
uv run python -m $N.build_subcomp_census "$SCKPT"
$PY param_decomp_lab/scripts/validation/headless_check.py \
    ~/out/runs/addsub-L18-04-2x-beta0.75-LR/analysis/subcomp_census/index.html
```

### 17. Minimal sufficient subset — CI-ranked sufficiency curve (GPU)

This is `find_alive_subcomponents` — see its section at the top of this file.

### 18. Alive neurons — greedy removal on the census grids (GPU)

Neuron analog of §17 without CI: orders L18 neurons by mean |post-SwiGLU activation| at
`=` (ascending) and greedily zeroes them (adaptive batch + bisection) while the mean KL
vs the base model on a scoring subset of the 0..200 add/sub grids stays <= `--kl-thr`.
Kept neurons are the alive set; decomposition-independent, so outputs go to `runs/neurons/`.

```bash
uv run python -m $N.find_alive_neurons "$CKPT" --slurm
uv run python -m $N.find_alive_neurons "$CKPT" --kl-thr=0.02 --score-prompts=8000 --slurm
# outputs: runs/neurons/alive_neurons.tsv, alive_neurons_curve.tsv, alive_neurons.npz,
#          alive_neurons_curve.png
```

### 19. Subspace-projection KL — alive subspaces vs alive circuit (GPU)

Projects the original model's L18 MLP input (onto the alive gate/up `V` span) or output
(onto the alive down `U` span) at the `=` position, weights unchanged, and compares the
last-position KL vs the target against running only the alive subcomponents (dead + delta
off at `=`). If the decomposition found the causally relevant subspace, the projections
should hurt no more than the circuit. Reads `alive_subcomponents.tsv`; one heatmap PNG per op.

```bash
uv run python -m $V.collect_projection_kl "$MODEL_PATH" --slurm
# add-only, custom alive list:
uv run python -m $V.collect_projection_kl "$MODEL_PATH" --ops=add --alive-tsv="$DATASETS/alive_subcomponents.tsv" --slurm
# outputs: datasets/projection_kl/{data_add.npz,data_sub.npz,summary.tsv,meta.json},
#          analysis/projection_kl/kl_heatmaps_{add,sub}.png

# per-prompt sets instead of the static alive list: subcomponents with lower-leaky CI > 0.01
# at the `=` position of each prompt define that prompt's subspaces and circuit (`ci_only`)
uv run python -m $V.collect_projection_kl "$MODEL_PATH" --ci-thr=0.01 --slurm
# outputs: datasets/projection_kl_ci0.01/..., analysis/projection_kl_ci0.01/kl_heatmaps_{add,sub}.png
```

