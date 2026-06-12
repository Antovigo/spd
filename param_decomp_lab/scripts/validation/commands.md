# Validation — sample commands

Self-contained example invocations. Run the setup block once per shell, then paste any
command below. Each command uses `uv run`, which resolves the project venv automatically —
no separate `source .venv/bin/activate` needed.

## Setup

```bash
# The targeted 8B addition run.
MODEL_PATH=~/out/runs/llama8b-add-02/model_20000.pth
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

# a×b CI grids for `a+b=` prompts: rows = matrices, cols = all alive subcomponents, one PNG per position.
uv run python -m param_decomp_lab.scripts.validation.plot_ab_heatmaps "$JSON"
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
