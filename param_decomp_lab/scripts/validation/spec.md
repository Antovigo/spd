# Validation script specs

One entry per script. Each describes the arguments, behaviour, and output schema. See
`CLAUDE.md` in this folder for the shared CLI design.

**Output layout.** Analysis artifacts live under the run's `analysis/` folder, never in
`figures/` (which is reserved for the figures the training loop emits). Shared **datasets**
(alive lists, activation grids/TSVs, periods, dimensionality / ISA summaries, ablation
bundles) go in `<run_dir>/analysis/datasets/`; **figures and applets** go directly in
`<run_dir>/analysis/<name>/`. `--output*` flags still override per file/dir. Helpers in
`common.py`: `analysis_dir(run_dir)`, `analysis_datasets_dir(run_dir)`, and
`run_dir_of_dataset(path)` (recovers the run dir from a dataset under `analysis/datasets/`,
for scripts whose first positional arg is a dataset rather than a checkpoint).

**sample_target_data.py**

args:
- the path to a decomposed model (a checkpoint `model_<step>.pth`, or a run dir / W&B path)
- `--n-examples`: number of target sequences to sample (default 50)
- `--ci-thr`: lower-leaky causal-importance threshold above which a subcomponent counts as
  active in the circuit (default 0.1)
- `--top-n`: how many top next-token predictions to report per model (default 5)
- `--batch-size`: forward-pass chunk size over the sampled sequences (default 8) — bounds
  GPU memory for large vocab / long sequences; does not affect which examples are sampled
- `--seed`: seed for the random sample of target sequences (default 0)
- `--output`: overrides the output TSV path
- `--slurm`: submit this invocation as a single-GPU SLURM job instead of running locally
  (the login node has no GPU). `--partition`, `--gpus` (default 1), `--slurm-time`
  (default `1:00:00`), and `--slurm-mem` tune the job. All other args are forwarded
  verbatim, with paths expanded to absolute. See `CLAUDE.md` → "GPU scripts run via SLURM".

Draws a random sample of `--n-examples` sequences from the run's **target** distribution
(`cfg.data`) and, for each one, compares the original target model against the *circuit*.

The circuit (the "ablated" model): for every decomposed matrix, the component mask is
**binary** — exactly 1 on every subcomponent whose lower-leaky CI exceeds `--ci-thr` at
that position (active), 0 on every subcomponent at or below it (inactive) — and the
**delta component is disabled** (no `weight_deltas_and_masks` is passed, so the residual
weight-delta is set to zero). Inactive subcomponents and the delta are therefore both off;
only the active subcomponents reconstruct the matrix. The mask is binary, not the CI
magnitude — active subcomponents are set to 1, not to their CI value.

For **every real position** of each sampled sequence the script reports both models' top-n
next-token predictions. For prompt-based target data the sequences are right-padded, so the
real positions are `0 .. last_non_pad` (pad id taken from the tokenizer's `pad_token_id`,
falling back to `eos_token_id`); for packed dataset-based target data there is no padding
and every position is real.

Implementation:
- A single batch of size `--n-examples` is drawn from the target loader (`split="train"`)
  to form the sample; the prompt-pool loader caps the sample at the number of prompts and
  the script warns when fewer than `--n-examples` are returned. The sample is then
  processed in chunks of `--batch-size`.
- Per chunk, one `cache_type="input"` forward pass yields both the original logits
  (`output`) and the pre-weight-act cache feeding `calc_causal_importances`
  (`sampling="continuous"`, i.e. deterministic — no noise injected). The circuit mask is
  built from `ci.lower_leaky` and a second masked forward pass yields the circuit logits.
- Both forward passes run under `torch.no_grad()` and `bf16_autocast` (gated by
  `cfg.runtime.autocast_bf16`). Logits are cast to float before softmax, and the top-n is
  taken at every position.
- LM tasks only (next-token predictions and a tokenizer are required).

Output TSV (default `sample_target_data.tsv` in the run's `analysis/datasets/`), **long
format**: one row per `(sequence, position, model)`, so each sequence yields
`2 * n_positions` rows. Columns:
- `example` — index of the sequence within the sample
- `pos` — the position whose next-token prediction this row reports
- `token` — the input token at `pos`, decoded and `repr`'d (whitespace/specials visible)
- `model` — `original` or `ablated`
- `rank_1 .. rank_{n}` — that model's top-n next tokens at `pos`, one per column in
  descending-probability order, each formatted `'<tok>' (<prob>)` (token `repr`;
  probability to 3 dp)

Rows are emitted in `(example, pos, model)` order, so the `original` and `ablated` rows for
the same position are adjacent for easy comparison.

**find_alive_components.py**

args:
- the path to a decomposed model (a checkpoint `model_<step>.pth`, or a run dir / W&B path)
- `--ci-thr`: lower-leaky CI threshold above which a subcomponent counts as active (default
  0.1 — matches the circuit threshold in `sample_target_data.py`, so the alive set is
  exactly the components the circuit ever uses)
- `--batch-size`: forward-pass chunk size over the prompt pool (default 8)
- `--output`: overrides the TSV path
- `--output-json`: overrides the JSON path
- `--slurm` (+ `--partition` / `--gpus` / `--slurm-time` / `--slurm-mem`): submit as a
  single-GPU SLURM job (see `CLAUDE.md` → "GPU scripts run via SLURM")

Runs **every** prompt in the run's target distribution (read in file order from
`cfg.data.prompts_file`) through the decomposed model and records which subcomponents reach
lower-leaky CI > `--ci-thr` on at least one (prompt, position). Requires prompts-based
target data (asserts `cfg.data.prompts_file` is set). Only real (non-pad) positions are
counted — prompts are right-padded, so positions where the token is the pad id (falling
back to `eos_token_id`) are masked out of every statistic.

Implementation:
- The prompt pool is tokenised once (via `load_prompts_dataset`) in file order, so prompt
  index = file line, and processed in `--batch-size` chunks. Per chunk, one
  `cache_type="input"` forward pass feeds `calc_causal_importances`
  (`sampling="continuous"`, deterministic) for the CI and `get_all_component_acts` for the
  per-component inner activations `V^T x`. Runs under `torch.no_grad()` + `bf16_autocast`.
- Per-component running stats (over valid positions across all prompts): `count_active`,
  `max_ci`, and `activation_sum` (sum of `V^T x` over active positions); `count_total` is
  the total valid-position count.

Output 1 — TSV (default `alive_components.tsv` in `analysis/datasets/`), one row per alive
subcomponent (`count_active > 0`), sorted by `(layer, matrix, component)`. Same schema as
the SPD `find_alive_components.py`:
- `layer` — block number from the module path (e.g. 18)
- `matrix` — the rest of the module path (e.g. `mlp.gate_proj`)
- `component` — the subcomponent index
- `fraction_active` — `count_active / count_total` (fraction of seen valid positions where
  it was active)
- `max_ci` — max observed lower-leaky CI
- `mean_activation` — mean `V^T x` over the positions where it was active

Output 2 — JSON (default `alive_components_per_position.json` in `analysis/datasets/`), the active
components per (prompt, position), organised **prompt > position > matrix > list**:
```json
{
  "<prompt text>": {
    "<position>": {
      "<full module path>": [{"component": <idx>, "ci": <CI at this position>}, ...],
      ...
    },
    ...
  },
  ...
}
```
Keys are the prompt strings (asserted unique); position keys are stringified ints over the
prompt's real positions. For each (prompt, position), only modules with ≥1 active component
appear; the component list is sorted by descending CI (each component's `ci` is its
lower-leaky CI at that position, rounded to 3 dp). Written compactly (no indent). Consumed
by `plot_ci_heatmaps.py` and `plot_ab_heatmaps.py`.

**ablate_component_groups.py**

args:
- the path to a decomposed model (a checkpoint `model_<step>.pth`, or a run dir / W&B path)
- `--n-examples`: number of target prompts to randomly subsample (default 1024)
- `--ci-thr`: lower-leaky CI threshold defining the circuit (default 0.1)
- `--batch-size`: forward-pass chunk size (default 128)
- `--seed`: seed for the random prompt subsample (default 0)
- `--output`: overrides the output TSV path
- `--slurm` (+ `--partition` / `--gpus` / `--slurm-time` / `--slurm-mem`): submit as a
  single-GPU SLURM job (see `CLAUDE.md` → "GPU scripts run via SLURM")

Causal probe of what each *family* of `=`-position components contributes to the predicted
sum. The baseline is the circuit (every subcomponent with CI > `--ci-thr` on at the
position, delta off). For each named group in the module-level `_GROUPS` dict (a
`{short_matrix: [component, ...]}` mapping, e.g. the units-digit lattice or the sum-band
down components), the script clones the circuit mask, forces that group's components to 0
at every position, and re-runs the forward pass. Because every `X+Y` answer is a single
token, the argmax at the `=` position is the full predicted integer, so per-digit effects
are read off directly. LM prompts-based target data only (asserts `cfg.data.prompts_file`).

`_GROUPS` is edited in-source to match the families found by upstream grid analysis — it is
not a CLI argument; this script is the place to encode and re-run a specific ablation
hypothesis.

Output TSV (default `ablate_component_groups.tsv` in `analysis/datasets/`), one row per
`(prompt, condition)` where condition ∈ {`baseline`, *group names*}:
- `example` — index within the subsample
- `x`, `y` — the operands (parsed from the `X+Y=` prompt)
- `correct` — `x + y`
- `condition` — `baseline` or the ablated group name
- `pred` — the decoded argmax token at `=`, repr'd
- `pred_int` — the prediction parsed to int, or empty if non-numeric
- `correct_flag` — 1 iff `pred_int == correct`

**plot_ci_heatmaps.py**

args:
- the per-position JSON from `find_alive_components.py`
- `--n-prompts`: cap on prompts shown, in file order (default 50)
- `--grep`: keep only prompts containing this substring (default none → all)
- `--output-dir`: overrides the figure folder (default `<run_dir>/analysis/ci_heatmaps/`)

CPU-only (no model loaded). For each token position, draws one heatmap with prompts on the
y-axis (tiny text) and alive subcomponents on the x-axis, faceted by matrix, coloured by
lower-leaky CI (`RdPu`, 0–1, shared colorbar). The x-axis alive set is the union of
components active across the *selected* prompts (so it adapts to `--grep`); a cell is the
component's CI at that position for that prompt, 0 where inactive. Writes one PNG per
position (`position_<pos>.png`).

**plot_ab_heatmaps.py**

args:
- the per-position JSON from `find_alive_components.py` (prompts must be `a+b=`)
- `--output-dir`: overrides the figure folder (default `<run_dir>/analysis/ab_heatmaps_<op>/`)

CPU-only (no model loaded). A variant view for `a+b=` arithmetic prompts. For each token
position, writes one figure whose subplots form a grid: matrices down the rows, every alive
subcomponent across the columns (row widths differ; unused cells are blank). Each subplot is
an `a`-by-`b` heatmap (x = a, y = b, both 1..N parsed from the prompts) coloured by that
subcomponent's lower-leaky CI on `a+b=` at this position (`RdPu`, 0–1, single shared
colorbar). All alive subcomponents are shown; subplot titles are the component index, row
labels the matrix, and the figure carries `a` / `b` operand axis titles. Writes one PNG per
position (`position_<pos>.png`).

**screen_components_on_data.py**

args:
- the path to a decomposed model (a checkpoint `model_<step>.pth`, or a run dir / W&B path)
- `--n-batches`: number of broad-data batches to stream (default 600; the dominant cost —
  total tokens = `n_batches × batch_size × max_seq_len`)
- `--batch-size`: forward-pass / stream batch size (default 128)
- `--ci-thr`: lower-leaky CI threshold above which a component counts as active (default 0.1)
- `--top-k`: max-activating contexts kept per component (default 30)
- `--context-window`: tokens of left context stored per firing (default 24)
- `--seed`: stream shuffle seed (default 0)
- `--alive-tsv`: alive-on-addition list used to flag components (default `alive_components.tsv`
  in `analysis/datasets/`)
- `--output-tsv` / `--output`: override the TSV / JSONL paths
- `--slurm` (+ `--partition` / `--gpus` / `--slurm-time` / `--slurm-mem`): submit as a
  single-GPU SLURM job (see `CLAUDE.md` → "GPU scripts run via SLURM")

Streams the run's broad **nontarget** distribution (`cfg.nontarget.data`; asserts it exists)
through the decomposed model, computes lower-leaky CI for every decomposed component, and
finds where each fires (CI > `--ci-thr`) on general text — screening which components are
addition-specific vs. generic numeric, and surfacing the non-`a+b=` situations that drive
them. Uses a GPU-side running top-k per component (keyed by the global flat position id) and
keeps every batch's input_ids on CPU to reconstruct context windows for the survivors at the
end. Checkpoints both outputs every 50 batches (the write logic is a `flush()` helper, safe
to call mid-stream), so a wall-clock kill leaves usable partial output. LM only.

Output 1 — TSV (default `screen_components_on_data.tsv`), one row per component that fires
at least once, sorted by descending `frac_active`:
- `matrix`, `component` — the decomposed matrix (e.g. `mlp.gate_proj`) and component index
- `alive_on_addition` — 1 iff the component is in `--alive-tsv`
- `count_active` — positions where CI > `--ci-thr` over the whole screen
- `frac_active` — `count_active / positions_seen`
- `max_ci` — max observed lower-leaky CI

Output 2 — JSONL (default `screen_components_on_data.jsonl`), one object per component:
`{matrix, component, alive_on_addition, contexts: [{ci, pos, token, left_context}, ...]}`,
contexts sorted by descending CI (the firing token repr'd, plus decoded left context).

---

## Arithmetic analysis (`roadmap_addition_analysis`)

A pipeline that probes the L18 MLP decomposition one **operation** at a time over its
`1..100 × 1..100` prompt grid (`add` `+`, `sub` `-`, `mult` `×`). Every output file is
suffixed with the operation (`_add` / `_sub` / `_mult`) and the three scripts that read the
grid all act at the **last token** (the `=` answer position).

**"Alive" here means two things at once:** ever causally important on the run's **original**
data — flagged by `find_alive_components` run once with defaults, writing the *unsuffixed*
`alive_components.tsv` / `alive_components_per_position.json` (op-agnostic) — **and** mean
lower-leaky CI over *this op's* grid above `--mean-ci-thr` (default 0.1). The downstream
scripts read `find_alive_components`'s output and do the per-op / last-position / mean-CI
filtering themselves; `collect_inner_activations` materialises the intersection once into
`alive_filtered_<op>.tsv`, which the period / cosine / explorer scripts consume. Only the
three L18 MLP matrices are considered (decomposed attention is dropped). Shared helpers in
`common.py`: `op_symbol` / `op_prompts_file` / `parse_operands` / `MLP_MATRICES`,
`read_alive_components` / `read_subcomp_periods`, `square_grid_size` (asserts full grid
coverage), `load_component_uv` (mmap U/V).

Pipeline (run `find_alive_components` with defaults once first):
`collect_hidden_activations` + `collect_inner_activations` → `compute_subcomp_periods` →
`plot_subcomp_cosine` / `build_neuron_connection_explorer`.

**collect_hidden_activations.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--layer`: decomposed-MLP block (default: inferred, the single MLP layer in the run)
- `--batch-size`: forward chunk (default 256)
- `--output`, plus `--slurm` (+ knobs)

Forward-hooks the L18 MLP at five points and stores each one's **last-token** activation over
the grid. A plain forward (no masks) runs the bare target model, so the captures are the true
module in/outputs. GPU. Output (default `hidden_activations_<op>.npz`): five `[N, N, dim]`
float16 grids indexed `[a-1, b-1]` — `resid_pre_mlp`, `mlp_input` (post-RMSNorm),
`gate_preact`, `up_preact` (both pre-SwiGLU), `mlp_output` (post-down-proj) — plus `a`, `b`,
`op`, `layer`. Stored as npz (not JSON): the full hidden states are ~410M floats. The
post-SwiGLU neuron activation `silu(gate_preact)*up_preact` is derivable, not stored.

**collect_inner_activations.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--mean-ci-thr`: mean-CI cutoff for the alive filter (default 0.1)
- `--alive-tsv`: `find_alive_components` output (default `alive_components.tsv` in `analysis/datasets/`)
- `--batch-size` (default 256), `--output`, `--output-alive`, plus `--slurm` (+ knobs)

For every existing-alive MLP subcomponent and every prompt, computes the normalized inner
activation `(x · V_c) / ||V_c||` at the last token (`x` = the cached module input, already
post-RMSNorm for gate/up and post-SwiGLU for down, so nothing is reapplied). Also computes
each component's mean last-token CI over the grid and keeps those above `--mean-ci-thr`. GPU.
Outputs:
- `inner_activations_<op>.tsv` — one row per (kept component, prompt): `a, operation, b,
  matrix, subcomponent, inner_act, ci` (`operation` is the infix symbol; `matrix` the bare proj;
  `ci` = that component's last-token lower-leaky causal importance on the prompt).
- `alive_filtered_<op>.tsv` — the surviving alive set: `layer, matrix, component, mean_ci`.

**compute_subcomp_periods.py**

args:
- the `inner_activations_<op>.tsv` from the previous script (positional)
- `--log-bar`: held-out R² a log fit must clear to count (default 0.45)
- `--output`

CPU. Rebuilds each subcomponent's `[N, N]` inner-activation grid and measures periodicity of
its `f(a)` (mean over b) and `f(b)` (mean over a) marginals **two ways**:

- **additive** (add/sub): integer period of the marginal — **autocorrelation** (best lag in
  `1..N//2`; score = unit-`r(0)` autocorrelation) and **FFT** (peak nonzero frequency →
  `round(N/k)`; score = fraction of DC-removed power), with a representative `period` /
  `period_axis` (stronger FFT peak).
- **logarithmic** (mult): the marginal repeats each time the operand grows by a fixed
  multiplicative ratio `r`. A sinusoid is fit in `log(operand)` over `operand > threshold`
  (the periodicity is usually only resolved above some value); the period is chosen by the
  most **cross-validated** evidence (fit on half the points, scored on the held-out half — so a
  few high-value points lining up by chance don't pass), reported at the lowest threshold whose
  held-out fit clears `--log-bar`. Detected ratios are **clustered** (in log-ratio space) so
  they snap to a handful of canonical periods.

`period_type` ∈ {additive, log, none} is decided by comparing the additive and log
cross-validated R² (log wins a near-tie, since the linear/log sinusoids are near-degenerate for
long periods and log is the meaningful reading on mult). Reads `alive_filtered_<op>.tsv` for
`layer`/`matrix`. Output (`subcomp_periods_<op>.tsv`): the additive columns above, the log
columns (`log_{a,b}_ratio/thr/cvr2`, representative `log_period` ratio / `log_axis` /
`log_threshold`), the `additive_cvr2` / `log_cvr2` used for the decision, and `period_type`.

**plot_subcomp_cosine.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--output-dir`

CPU (mmap U/V, no forward). Cosine-similarity heatmaps between alive subcomponents' V and U
vectors, sorted by representative period with a thick separator between period groups
(`RdBu_r`, symmetric ±1; positive=red). The SwiGLU boundary splits the vectors into
incompatible dimensions, so two figures:
- `cosine_gate_up_<op>.png` — gate + up together (they share both spaces): V (residual,
  d_model) and U (neuron, d_int) side by side.
- `cosine_down_<op>.png` — down: V (neuron) and U (residual) side by side.

Outputs to `<run_dir>/analysis/subcomp_cosine/`.

**plot_ab_inner_heatmaps.py**

args:
- the `inner_activations_<op>.tsv` from `collect_inner_activations` (positional)
- `--output-dir`

CPU. The inner-activation twin of `plot_ab_heatmaps`: identical figure layout (matrices down
the rows, subcomponents across the columns, one `a×b` tile each — same tile size, gaps, fonts,
margins, colorbar, via the shared `plot_ab_heatmaps._plot_position`) but coloured by the
subcomponent's normalized inner activation `(x·V_c)/||V_c||` at the last token instead of CI.
Inner activations are signed, so a diverging `RdBu_r` on a symmetric shared scale
(`±max|inner|`, positive=red) replaces CI's 0..1 `RdPu`. One figure (inner activations are
last-token only), written next to the CI heatmaps as
`<run_dir>/analysis/ab_heatmaps_<op>/inner_activations.png`.

**build_neuron_connection_explorer.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--conn-floor`: min |connection strength| stored per (subcomponent, neuron) (default 0.1)
- `--top-neurons`: cap on neurons stored per subcomponent (default 60)
- `--output-dir`

CPU. Emits a self-contained HTML applet (`index.html` + `data.js`, `file://`-openable, no
server/CDN/GPU) into `<run_dir>/analysis/neuron_explorer_<op>/`. Connection strength uses the
V-unit normalization (V→V/||V||, U→U·||V||): gate/up (pre-SwiGLU) write strength to neuron
`j` is `U[c,j]·||V_c||`; down (post-SwiGLU) read strength is `V[j,c]/||V_c||`. The user picks
`(a, b)` and a connection threshold; the page shows active gate/up subcomponents (left, up on
top, period-sorted), the neurons they connect above threshold (center, sorted by strongest
gate/up driver then strength), and active down subcomponents (right), with lines coloured by
connection sign (red +, blue −). A "hover shows" toggle switches the subcomponent heatmap
between causal importance (0..1, red ramp) and signed normalized inner activation (red +,
blue −, per-component scaled); per-prompt activity (which subcomponents/neurons appear) stays
CI-based regardless. Hovering a neuron shows its up / gate / `silu(gate)·up` output.

Reads `alive_filtered_<op>.tsv`, `subcomp_periods_<op>.tsv`, the `find_alive_components`
per-position JSON (`alive_components_per_position.json` — unsuffixed/op-agnostic, filtered to
this op's symbol with an assert that ≥1 prompt matched, for CI patterns + activity),
`inner_activations_<op>.tsv` (inner-activation heatmaps), and `hidden_activations_<op>.npz`
(neuron up/gate grids, fp16 base64). Limitation: the UI threshold cannot surface neurons whose
connection is below `--conn-floor` (they aren't stored) — lower `--conn-floor` to widen the
universe, at the cost of `data.js` size. Smoke-test with `headless_check.py`.

**reduce_dimensionality.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--rank-eps`: relative cutoff on G eigenvalues for the orthonormal basis (default 1e-6 —
  drops only numerically-zero directions, so the geometric rank is essentially full)
- `--output-dir`

CPU (mmap U/V + the stored activations, no forward). Measures the real dimensionality of the
last-token MLP representation on two sides: **input** = the post-RMSNorm MLP input
(`mlp_input` from `collect_hidden_activations`) projected onto the span of the alive up/gate
unit V directions; **output** = the MLP output (`mlp_output`) projected onto the span of the
alive down unit U directions. For each side it builds an orthonormal basis `Q = Dᵀ E Λ^(-1/2)`
of the direction span (`G = D Dᵀ = E Λ Eᵀ`) and reduces the real activation to `z = Qᵀ x` — a
plain projection, so several subcomponents reading/writing the same plane collapse exactly.

Reports per side: geometric rank (non-negligible G eigenvalues), linear effective dimension
(participation ratio of the cov(z) / PCA spectrum), intrinsic dimension (TwoNN via
`scikit-dimension`, on z), and the completeness fraction `‖z‖²/‖x‖²` (centered) — the share of
activation variance living in the subcomponent subspace. Reads `alive_filtered_<op>.tsv` and
`hidden_activations_<op>.npz` (from `analysis/datasets/`). Outputs:
- `analysis/datasets/dimensionality_<op>.npz` — `z` (raw orthonormal) and `pca` (PCA-rotated)
  per side, plus the `(a, b)` per row. Consumed by the ISA script (Objective 7).
- `analysis/datasets/dimensionality_<op>.json` — the scalar summary (rank / PR / TwoNN / var
  fraction per side).
- `analysis/dimensionality_<op>/index.html` — a single self-contained **Plotly** applet
  (plotly.js inlined, `file://`-openable): a scree plot with an eigenvalue-threshold dialog,
  rotatable 3D scatter (floor-shadow projection) over both the raw z-axes and the PCA-ordered
  axes in groups of three, a colour selector (a / b / a+b), for both sides, with the TwoNN /
  rank / PR / variance table at the bottom. The 3D uses WebGL, so headless screenshots show
  only the colorbar — open it in a real browser to see the points.

**find_independent_subspaces.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--var-keep`: PCA cumulative-variance kept before ICA (default 0.9 — drops near-noise
  directions that hurt ICA convergence)
- `--group-distance`: average-linkage cut on `1 − |energy-corr|` for grouping components into
  subspaces (default 0.75, i.e. group when energy correlation > 0.25)
- `--seed`, `--max-iter` (FastICA; default 20000), `--output-dir`

CPU. Independent Subspace Analysis of the reduced representation `z` from
`reduce_dimensionality` (reads `dimensionality_<op>.npz`). Per side: reduce z to the PCs
capturing `--var-keep` of the variance, run scikit-learn FastICA, then group the independent
components into subspaces by the **magnitude (energy) correlation** `|corr(|s_i|, |s_j|)|` —
so a circular feature (components linearly uncorrelated but jointly dependent) is recovered as
one subspace. Blocks are checked for near-orthogonality via the principal angles between their
z-space directions (logs a warning if FastICA didn't converge). Discovery uses no `(a, b)`
labels; they are only used afterward to colour the projections. Outputs:
- `analysis/datasets/independent_subspaces_<op>.json` — per side, the component→subspace grouping.
- `analysis/independent_subspaces_<op>/index.html` — a self-contained Plotly applet: a 3D
  scatter of the selected subspace's components (colour a / b / a+b), the energy-correlation
  heatmap (components ordered by subspace), and the min-principal-angle heatmap between
  subspaces. As with Objective 6, the 3D is WebGL (real browser to see the points).

**build_subspace_scatter.py**

args:
- the path to a decomposed model
- `--ops`: comma-separated tasks (default: auto-detect every op with a
  `hidden_activations_<op>.npz` + `alive_filtered_<op>.tsv` in `analysis/datasets/`)
- `--output-dir`

CPU. A self-contained Plotly applet for exploring the activation geometry in a user-picked
3-subcomponent subspace, **spanning every available task** (add / sub / mult). The right panel
is a thumbnail grid (4 per row) of the alive subcomponents, organised into high-level **task**
sections, then by period group — `period N` (additive), `×r` (log multiplicative ratio), or
`no period` — from `subcomp_periods_<op>.tsv` (via `read_subcomp_period_groups`; absent file →
all "no period"). Each thumbnail is that subcomponent's inner-activation `(a, b)` pattern on its
task, signed `RdBu_r`. The user clicks up to 3 directions (from any task); a **points** selector
chooses which single task's last-token activations are scattered onto those 3 unit directions.
A **side** selector (MLP order) sets which activation each direction projects and which matrices
the picks come from: **input** = `mlp_input · V̂` (up/gate); **pre-nonlinearity** = each up/gate
subcomponent's own preactivation `up_preact`/`gate_preact` onto its `Û` (what the matrix writes,
pre-SwiGLU); **post-nonlinearity** = the post-SwiGLU neuron output `silu(gate)·up` onto the down
`V̂` (what down reads); **output** = `mlp_output · Û` (down). Each direction's
sign (an arbitrary gauge) is flipped so the median projection (over all tasks' points) is
positive, so its arrow points toward the data. The directions are kept at their **true mutual
angles**: each point is embedded via the Cholesky factor of the picked sub-Gram (`P = L⁻¹s`)
and the three directions are drawn as red oblique arrows (≈ ¾ of the data range, since the V/U
norm is also an arbitrary gauge), with `aspectmode:"data"`. A colour selector — `result` (the
task's operation, `a+b` / `a−b` / `a×b`), `a`, or `b`, with an optional modulo and a phase
offset — recolours the points. The modulo options track the **points task's** subcomponent
frequencies: integer residues (`(x − offset) mod m`) for an additive task, or the detected log
ratios for a log task (mult), which colour by the **multiplicative phase**
`frac((log x − offset)/log r)` on a cyclic scale. The current camera is re-applied on every
redraw so changing colour/mod/picks/points doesn't reset the view. A dark-grey floor shadow aids
reading. Reads each task's `alive_filtered_<op>.tsv`, optional `subcomp_periods_<op>.tsv`, and
`hidden_activations_<op>.npz`; directions come from the checkpoint. Output:
`analysis/subspace_scatter/index.html`. The 3D is WebGL (real browser to see the points); a
screen-fixed (non-orbiting) shadow isn't possible in a single Plotly 3D scene, so the shadow is
the standard floor projection.

**build_neuron_investigator.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--top-neurons`: number of neurons kept (by total interaction score; default 512)
- `--output-dir`

CPU. Emits a self-contained HTML applet (`index.html` + `data.js`, `file://`-openable, no
server/CDN/GPU) into `<run_dir>/analysis/neuron_investigator_<op>/`. The **interaction score**
between a subcomponent and a neuron is the std, over the target grid, of what the subcomponent
writes to / reads from that neuron (always ≥ 0): gate/up (pre-SwiGLU, *write*)
`std(inner_act_c)·||V_c||·|U[c,j]|` (std of the contribution `(x·V_c)·U[c,j]`); down
(post-SwiGLU, *read*) `std_grid(silu(gate_j)·up_j)·|V[j,c]|/||V_c||` (the neuron's post-SwiGLU
activation std × unit read weight) — two slightly different metrics for input vs output, sharing
a scale. The left half is a neuron × subcomponent heatmap. Subcomponents (columns) are ordered
by **period group, then matrix** (gate > up > down), then the confidence the period is correct
(the chosen fit's CV R²). Period groups are additive (`p10`) < multiplicative (`×1.27`, for
mult's log-periodic components) < none (`—`), via `read_subcomp_period_groups` — with period
band labels above the names, thick delimiters between groups and thin between matrices. Neurons
(rows) are ordered **by total interaction score per frequency** (grouped by the period group
they couple to most strongly, then by that coupling), paged
50 at a time (adjustable). A **neuron filter** (a `input`/`output`/`all` dropdown + a typed
threshold) hides neurons whose total interaction score over the chosen subcomponent scope is
below the threshold. Write scores render blue, read red (the down
columns' sign is flipped and an RdBu scale applied), on a shared `|score|` scale.
Clicking a cell selects that (neuron, subcomponent) pair (black border); the right half then
lays the subcomponent's inner-activation `(a, b)` heatmap and the neuron's up / gate /
post-SwiGLU output (`silu(gate)·up`) `(a, b)` heatmaps, each signed `RdBu_r` on a per-heatmap
scale with its own colour bar. For a **write (gate/up) subcomponent** it adds two more grids: the
subcomponent's contribution to the neuron's gate/up preactivation (`inner_act_c·||V_c||·U[c,j]`)
and the counterfactual preactivation with it removed (final − contribution) — the neuron's gate/up
without that subcomponent. (Omitted for read (down) subcomponents.) Hovering any heatmap pixel
shows a tooltip with the operands `a`, `b` and that cell's value. A **plot-size** control sets the heatmaps' pixels-per-operand-value
(each `(a, b)` cell is that many px wide); the heatmaps then pack as many per row as the right
panel's current width allows. The **divider** between the panels is drag-resizable (set the left
panel's width; the right takes the rest and its heatmaps reflow). An **operation toggle** in the right panel re-renders
all four grids on a *different* task's activations — defaulting to the build op, switchable to
any operation with a saved `hidden_activations_<o>.npz` (hidden when only one is available) — so
the same neuron / subcomponent can be compared across add / sub / mult.

Only the top `--top-neurons` neurons are kept — their per-op up/gate grids (for the right panel)
are the payload's bulk, so the cap bounds `data.js` size (~28 MB at 512 for one op, scaling with
the number of available ops). Reads `alive_filtered_<op>.tsv` (mean CI),
`subcomp_periods_<op>.tsv`, the build op's `inner_activations_<op>.tsv` (for the score), and one
`hidden_activations_<o>.npz` per available op (per-op neuron up/gate + subcomponent inner grids,
fp16 base64); U/V from the checkpoint (mmap). No forward pass. Smoke-test with `headless_check.py`.

**measure_model_accuracy.py**

args:
- the path to a decomposed model
- `--ablate`: comma-separated `matrix:component` to ablate (e.g. `gate_proj:163,down_proj:240`;
  bare `gate:163` also accepted). The matrices are matched by suffix against the decomposed
  module paths and must each resolve to exactly one module. Omit for the un-ablated model.
- `--range`: half-width `n` of the answer window (default 5)
- `--batch-size`: forward-pass chunk size (default 512)
- `--output-dir`: overrides the output directory
- `--slurm` (+ `--partition` / `--gpus` / `--slurm-time` / `--slurm-mem`): submit as a single-GPU
  SLURM job (see `CLAUDE.md` → "GPU scripts run via SLURM")

GPU. Runs every `a<op>b=` prompt of the run's target distribution through the **all-on
reconstruction** (every component on + delta on, which reproduces the target model) and, per
prompt, records the probability on the correct answer token and on every wrong answer in a `±n`
window (offset `k` → the first token of `str(result + k)`). With `--ablate`, the listed
subcomponents are masked off (`U_c V_c^T` removed, delta + all others stay on), so original vs
ablated differ only by the ablation. A first-batch reconstruction check (asserts un-ablated only)
guards the mask/delta wiring. The op (add/sub/mult) and the result are auto-detected from the
prompt format; the answer is read at the last (`=`) position (the prompt pool shares one length).
For the `1..100` grid every windowed result is a 1–3 digit number — a single Llama-3 token — so
the first-token probability is the number's probability.

Output (note: **`<run_dir>/model_accuracy/`**, not `analysis/`, per the objective's request):
`accuracy[_<ablation>].json` — the filename is suffixed with the ablated subcomponents
(`accuracy_gate163_down240.json`; empty un-ablated). JSON: run/op/range/ablation metadata +
`accuracy` + `mean_p_correct`, then per prompt `a`, `b`, `result`, the correct token + its
probability, the argmax token + whether it is correct, and `offset_probs` mapping each offset in
`-n..+n` to its token's probability (offset 0 = correct). A sibling `model_accuracy_notebook.py`
(marimo) is copied in to plot the results: per-offset mean ±1 std curves (4 operand-parity classes
× original/ablated) and `(a, b)` P(correct) heatmaps. Open with `marimo edit model_accuracy_notebook.py`.

**find_fourier_features.py**

args:
- the path to a `hidden_activations_<op>.npz` grid (from `collect_hidden_activations`; the op is
  read from the file)
- `--periods`: periods/ratios to fit (default: linear `2,5,10,20,50,100`, or — in log space — the
  clustered ratios read from the sibling `subcomp_periods_<op>.tsv`)
- `--space`: `linear` or `log` (default: `log` for mult, `linear` otherwise)
- `--output`: overrides the output path

CPU (no forward pass — it reuses the saved activation grids). Replicates Feucht et al. (2026)'s
probing for the circular ("Fourier") features around L18's MLP. For each period and each probed
variable it fits the generative model `x̄(v) ≈ offset + cos(θ)·cos_vec + sin(θ)·sin_vec` by least
squares on the **mean activation per distinct probed value** (equal weight per value, which
isolates the probed variable from the nuisance operand). The angle is `θ = 2πv/T` in **linear**
space (`period` = integer `T`, add/sub) or `θ = 2π·log(v)/log(r)` in **log** space (`period` =
multiplicative ratio `r`, mult — one turn per `×r`; multiplication is periodic in `log v`). In log
space the default ratios come from the run's `subcomp_periods_mult.tsv` clusters (the frequencies
the period analysis already found; see `find_log_periods` for how those are located from scratch).
`offset` is the circle's center; `(cos_vec, sin_vec)` span its plane; `r2` is the fraction of the
conditional mean's variance that period explains (each explains only a fraction — the mean is a sum
over several). Two sides, matching the paper: **input** = the post-RMSNorm MLP input (`mlp_input`
grid), probed for each operand `a` and `b`; **output** = the MLP's residual write (`mlp_output`
grid), probed for the task result (`a+b` / `a-b` / `a×b`). Fit separately per task.

Output (fixed dir per the objective: **`<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/`**, not the
source run's `analysis/`): `coordinates_<op>.json` — op/symbol/layer/`space`/source/grid metadata +
`features[side][variable][period]` → `{period, r2, offset, cos, sin}`, each vector `d_model`-long
(`period` is the ratio `r` in log space). For addition/subtraction the input operands and output
sums show clear circular structure (higher-period `r2`); for multiplication only the second operand
`b` fits cleanly (log ratio ≈×1.27), while `a` and the product do not.

**find_log_periods.py**

args:
- the path to a `hidden_activations_<op>.npz` grid (from `collect_hidden_activations`)
- `--v-min`: smallest operand/result value used (default 10) — below it `log v` is sampled too
  coarsely and the phase step aliases past the Nyquist limit `π`
- `--n-planes`: how many SVD pairs (candidate planes) to report per variable (default 3)
- `--output`: overrides the output path (a `.png`; the `.json` sits beside it)

CPU. Finds the **log-space periods** of the multiplication circular features **without scanning any
frequency grid**. Multiplication is periodic in `log v`: the operand is a circle whose phase
advances with `log v`. Per probed variable (input `a`, `b`; output result): average out the
nuisance operand → `x̄(v)`; remove DC + the linear-in-`log v` trend (the magnitude direction); SVD
over `v` (a circular feature is a near-degenerate singular-value **pair** whose scores are a
`cos`/`sin` of the same phase); for each consecutive pair `(2k, 2k+1)` project onto the plane and
take the **signed angle increment** between consecutive values ÷ `Δ log v` → angular velocity `ω`.
The log-period is `P = 2π/|median ω|`, ratio `r = e^P`. Diagnostics: `sv_ratio` (≈1 for a
degenerate pair), `radius_cv` (0 = perfect circle), `omega_cv` (0 = phase exactly linear in
`log v`), `var_share`. Only `v ≥ --v-min` is used (Nyquist).

Output (same fixed dir): `log_periods_<op>.png` — per-variable figure (top-plane trajectory
coloured by `log v`, plus unwrapped phase vs `log v` with the fitted slope) — and `.json` with each
variable's planes and diagnostics. Empirically only the second operand `b` is a clean log-circle
(ratio ≈×1.26, `omega_cv≈0.17`), reproducing the `subcomp_periods_mult.tsv` dominant cluster; `a`
and the product are not clean single-period log-circles.

**build_fourier_scatter.py**

args:
- the path to a decomposed model (checkpoint)
- `--coordinates-dir`: where the `coordinates_<op>.json` bases live (default
  `<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/`)
- `--ops`: comma-separated tasks to include (default: auto — those with both a
  `hidden_activations_<op>.npz` and a `coordinates_<op>.json`)
- `--arrow-floor`: minimum in-plane norm for an arrow to be shipped (default 0.1); the applet's
  threshold form filters further, and inner grids are only shipped for subcomponents clearing it
- `--output-dir`: overrides the output dir

CPU (no forward pass). A self-contained canvas applet (vanilla JS, no CDN) for comparing
subcomponents / neurons against the circular features. For a chosen **basis task** and **operand**
(first operand `a`, second operand `b`, or the output result), each canonical period's plane is the
orthonormalised `(cos_vec, sin_vec)` of that Fourier feature; one plot per period, side by side. It
scatters the chosen **activation task**'s activations projected onto that plane (input operands ←
`mlp_input`, result ← `mlp_output`) — so e.g. subtraction activations can be viewed on addition's
basis. The plane is the orthonormalised `(cos_vec, sin_vec)`; when the sin axis is degenerate
(e.g. period 2, where `sin(2πv/2)=0` for every integer `v` so the circle would collapse to the
`e1` line), the second axis falls back to the direction of most activation variance orthogonal to
`e1` — an arbitrary but informative viewing axis, as in Feucht et al. Everything is in **raw
projection coords** (`x·e1, x·e2`): points, the subcomponent arrows (which start at the
**activation-space zero** `(0,0)`), and a crosshair+ring marker at the Fourier circle centre (the
projected `offset`) share one origin, so an off-zero centre is visible. Points colour by `a` / `b`
/ result, either raw or by `(value − offset) mod m` / the multiplicative log phase via a `mod` +
`offset` form (options from the task's `subcomp_periods`, like the subspace-scatter applet); a
further **CI (selected)** colour option (per task: shown when some `inner_activations_<op>.tsv`
carries a `ci` column, greyed for tasks lacking it or for a subcomponent absent from a task's TSV)
paints each point by the currently-selected subcomponent's causal importance on that prompt. All
colouring uses a **viridis 0→1** map with a single shared legend. Colour/mod/offset changes and
selecting a new subcomponent recolour in place (zoom preserved). Scroll zooms, drag pans. The **unit** subcomponent directions (gate/up `V` for
input operands, down `U` for the result) are drawn as arrows scaled to the point cloud; only those
whose in-plane norm ≥ the typed threshold show; hovering shows the label + ‖proj‖; clicking a
subcomponent arrowhead opens its inner-activation `(a, b)` heatmaps (one per task) at the bottom.
An **overlay** toggle swaps subcomponents for **individual neurons'** directions — gate/up read
rows or the down write columns of the frozen target weight (a neuron-matrix dropdown) — so
directions captured by neurons but not subcomponents (or vice versa) are visible. Reads the bases
(`coordinates_<op>.json`; asserts each was fit at the checkpoint's layer), the run's
`hidden_activations_<op>.npz` / `alive_filtered_<op>.tsv` / `subcomp_periods_<op>.tsv`, and the
checkpoint U/V + target MLP weights (mmap). No forward pass.

Output: `<run_dir>/analysis/fourier_scatter/{index.html,data.js}` — `data.js` holds, per
(basis task, operand, period): the projected point clouds for every activation task, the projected
circle centre, the floor-passing subcomponent and neuron arrow coords + in-plane norms, plus the
kept subcomponents' inner-activation and (when available) CI grids per task (fp16 base64).
Smoke-test with `headless_check.py`.
