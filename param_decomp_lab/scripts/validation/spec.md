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

**find_alive_subcomponents.py**

args:
- the path to a decomposed model (a checkpoint `model_<step>.pth`, or a run dir / W&B path)
- `--kl-thr`: mean last-position KL below which a top-k subset counts as sufficient; the
  alive set is the smallest swept k under it. Default `rounded`: anchor to the run's own
  rounded circuit — per-(prompt, position) masks with CI > `--rounding-thr`, delta off —
  evaluated in-sweep at the last position on the same prompts (commensurable with the
  sweep, unlike the whole-sequence `eval/target_recon/rounded` training metric). Pass a
  float for an explicit absolute cut.
- `--rounding-thr`: CI rounding threshold defining the anchor circuit (default 0.01,
  matching `TargetReconLoss`)
- `--ci-thr`: lower-leaky CI threshold above which a subcomponent is recorded in the
  per-position JSON (default 0.1 — matches the circuit threshold in
  `sample_target_data.py`)
- `--batch-size`: forward-pass chunk size over the prompt pool (default 256)
- `--n-points`: size of the default log-spaced k grid (default 40)
- `--ks`: explicit comma-separated k grid overriding `--n-points` (pass a dense grid
  around the knee to tighten the alive cut)
- `--prompts`: override the LM `prompts_file`
- `--output` / `--output-curve` / `--output-npz` / `--output-fig` / `--output-json`:
  override the per-file paths
- `--slurm` (+ `--partition` / `--gpus` / `--slurm-time` / `--slurm-mem`): submit as a
  single-GPU SLURM job (see `CLAUDE.md` → "GPU scripts run via SLURM")

Produces the run's **reference alive list**. Ranks every subcomponent by its
max-over-positions mean lower-leaky CI (per position, mean over the target prompts — read
in file order from `cfg.data.prompts_file`; requires prompts-based target data — then max
over positions, so a subcomponent that only fires early ranks by its early-position
strength), then sweeps top-k prefixes of that ranking: for each k the top-k subcomponents
are enabled and all others zeroed **at every position**, with the weight-delta pinned
**off** everywhere (`TargetReconLoss` target-data semantics: the components must do the
work), and the masked model's last-position output is compared to the raw target
model's (KL + argmax agreement) — the KL is read at `=` because that is where the answer
is read, but masking acts everywhere, so a component matters iff masking it anywhere
moves the `=` output. The alive subcomponents are the top-k for the smallest swept k
whose mean KL is ≤ `--kl-thr`. Every downstream script consumes this output.

Implementation:
- Phase 1 (ranking + JSON): per `--batch-size` chunk, one `cache_type="input"` forward
  feeds `calc_causal_importances` (`sampling="continuous"`, deterministic); accumulates
  per-(position, subcomponent) mean CI and the sparse per-(prompt, position) record of
  subcomponents with CI > `--ci-thr`. Runs under `torch.no_grad()` + `bf16_autocast`.
- Phase 2 (sweep): outer loop over chunks (one target-reference forward each), inner loop
  over ks ascending, growing the enabled set incrementally. Asserts the all-on (no-delta)
  masked model reproduces the raw target (mask-wiring check).

Output 1 — TSV (default `alive_subcomponents.tsv` in `analysis/datasets/`), the alive
subset (top-k_alive rows of the ranking), one row per subcomponent:
- `layer` — block number from the module path (e.g. 18)
- `matrix` — the rest of the module path (e.g. `mlp.gate_proj`)
- `component` — the subcomponent index
- `rank` — position in the max-over-positions mean-CI ranking (0 = highest)
- `mean_ci` / `mean_ci_last` / `max_mean_ci` — mean lower-leaky CI over all positions /
  at the last position / at the subcomponent's strongest position (the rank key)

Output 2 — TSV (default `alive_subcomponents_curve.tsv`), one row per swept k:
`k, max_mean_ci_at_k, mean_kl, q5_kl, q95_kl, max_kl, argmax_agree`.

Output 3 — npz (default `alive_subcomponents_kl.npz`): per-(k, prompt) KL + argmax
agreement, the rounded-circuit per-prompt KL (`rounded_kl`), and the full CI ranking,
for per-prompt analysis / re-thresholding without a GPU.

Output 4 — JSON (default `alive_subcomponents_per_position.json` in `analysis/datasets/`),
the **alive** subcomponents active per (prompt, position), organised
**prompt > position > matrix > list**:
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
Keys are the prompt strings (asserted unique); position keys are stringified ints. For each
(prompt, position), only modules with ≥1 active alive subcomponent appear; the component
list is sorted by descending CI (each component's `ci` is its lower-leaky CI at that
position, rounded to 3 dp). Written compactly (no indent). Consumed by
`plot_ci_heatmaps.py`, `plot_ab_heatmaps.py`, `build_addition_explorer.py`, and
`build_neuron_connection_explorer.py`.

Output 5 — figure (default `analysis/alive_subcomponents/recon_vs_k.png`): the
recon-vs-k curve (mean + q5–q95 ribbon + max KL, argmax agreement on a twin axis) with
the alive cut marked.

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
- the per-position JSON from `find_alive_subcomponents.py`
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
- the per-position JSON from `find_alive_subcomponents.py` (prompts must be `a+b=`)
- `--output-dir`: overrides the figure folder (default `<run_dir>/analysis/ab_heatmaps_<op>/`)

CPU-only (no model loaded). A variant view for `a+b=` arithmetic prompts. For each token
position, writes one figure whose subplots form a grid: matrices down the rows, every alive
subcomponent across the columns (row widths differ; unused cells are blank). Each subplot is
an `a`-by-`b` heatmap (x = a, y = b, both 1..N parsed from the prompts) coloured by that
subcomponent's lower-leaky CI on `a+b=` at this position (`RdPu`, 0–1, single shared
colorbar). All alive subcomponents are shown; subplot titles are the component index, row
labels the matrix, and the figure carries `a` / `b` operand axis titles. Writes one PNG per
position (`position_<pos>.png`).

**score_period_separation.py**

args:
- the per-position JSON from `find_alive_subcomponents.py` (prompts must be `a<op>b=`)
- `--min-mass`: mean-CI cutoff below which a (position, subcomponent) grid is skipped (default 0.01)
- `--output` / `--output-summary`: override the two TSV paths

CPU-only (no model loaded). Quantifies how cleanly each subcomponent's CI pattern isolates a
single operand period. Per (op, position, subcomponent) the `[b, a]` CI grid is DC-removed and
2D-FFT'd; power is grouped into conjugate frequency orbits labelled `a` / `b` / `a+b` / `a-b` /
`mixed2d` with an integer period. Reports `purity` (top orbit's power share), `band_purity`
(top orbit + its harmonics — the headline cleanliness number, `> 0.5` = clean),
`n_orbits_50/90` (orbits to reach that power share), and the top-3 orbits. Always-on grids
(std < 0.05) are labelled `flat` and excluded from aggregates. Subtraction's triangular
prompt coverage falls back to 1D marginal FFTs (diagonal structure invisible there — compare
runs on the `+` rows). Writes `period_separation.tsv` (per-subcomponent rows) and
`period_separation_summary.tsv` (per op × position × matrix: `n_clean`, `n_flat`,
median / mass-weighted `band_purity`, `mean_n_orbits_50`, per-period counts). A
`..._step<k>` JSON yields `..._step<k>.tsv` outputs.

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
- `--alive-tsv`: alive-on-addition list used to flag components (default `alive_subcomponents.tsv`
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

**"Alive" here means two things at once:** in the run's **reference alive list** — from
`find_alive_subcomponents` run once with defaults, writing the *unsuffixed*
`alive_subcomponents.tsv` / `alive_subcomponents_per_position.json` (op-agnostic) — **and** mean
lower-leaky CI over *this op's* grid above `--mean-ci-thr` (default 0.1). The downstream
scripts read `find_alive_subcomponents`'s output and do the per-op / last-position / mean-CI
filtering themselves; `collect_inner_activations` materialises the intersection once into
`alive_filtered_<op>.tsv`, which the period / cosine / explorer scripts consume. Only the
three L18 MLP matrices are considered (decomposed attention is dropped). Shared helpers in
`common.py`: `op_symbol` / `op_prompts_file` / `parse_operands` / `MLP_MATRICES`,
`read_alive_components` / `read_subcomp_periods`, `square_grid_size` (asserts full grid
coverage), `load_component_uv` (mmap U/V).

Pipeline (run `find_alive_subcomponents` with defaults once first):
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
- `--alive-tsv`: `find_alive_subcomponents` output (default `alive_subcomponents.tsv` in `analysis/datasets/`)
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
server/CDN/GPU) into `<run_dir>/analysis/neuron_explorer_<op>/`. Each subcomponent's connection
vector is **unit-normalised** (`V[:,c]/||V_c||` for down reads, `U[c,:]/||U_c||` for gate/up
writes), so an edge value is that neuron's share of the subcomponent's connection energy
(`∑_j w²=1`) and one threshold / colour ramp means the same on the read and write sides. The user
picks `(a, b)` and a connection threshold; the page shows active gate/up subcomponents (left, up on
top, period-sorted), the neurons they connect above threshold (center, sorted by strongest
gate/up driver then strength), and active down subcomponents (right), with lines coloured by
connection sign (red +, blue −). A "hover shows" toggle switches the subcomponent heatmap
between causal importance (0..1, red ramp) and signed normalized inner activation (red +,
blue −, per-component scaled); per-prompt activity (which subcomponents/neurons appear) stays
CI-based regardless. Hovering a neuron shows its up / gate / `silu(gate)·up` output.

Reads `alive_filtered_<op>.tsv`, `subcomp_periods_<op>.tsv`, the `find_alive_subcomponents`
per-position JSON (`alive_subcomponents_per_position.json` — unsuffixed/op-agnostic, filtered to
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
between a subcomponent and a neuron is the **fraction of the target's variance** over the grid that
their rank-1 term explains (always ≥ 0), so write and read sides are on one comparable scale:
gate/up (pre-SwiGLU, *write*) `std((x·V_c)·U[c,j]) / std(neuron j's own gate/up preactivation)`;
down (post-SwiGLU, *read*) `std(silu(gate_j)·up_j · V[j,c]) / std(the down read total)` =
`std(silu(gate_j)·up_j)·|V[j,c]|/||V_c|| / std(inner_act_c)`. The left half is a neuron × subcomponent heatmap. Subcomponents (columns) are ordered
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
probing for the circular ("Fourier") features around L18's MLP **exactly** (their Eq. 9, bias
included). For each period `T` and probed variable `v` (operand `a`, `b`, or the result) it fits two
linear probes over the **individual prompts** — `cos(θ) ≈ w_cos·x + b_cos`, `sin(θ) ≈ w_sin·x +
b_sin` — where `θ = 2πv/T` in **linear** space (`period` = integer `T`, add/sub) or `θ =
2π·log(v)/log(r)` in **log** space (`period` = multiplicative ratio `r`, mult — one turn per `×r`).
In log space the default ratios come from the run's `subcomp_periods_mult.tsv` clusters. Solved by
normal equations (float64; a 4098×4098 solve, orders faster than lstsq's full SVD) on a fixed 80/20
train split; `r2` is the **held-out** fraction of variance explained, averaged over the cos and sin
probes (period 2 has `sin(2πv/2)=0`, so only cos counts). Probes are fit at two **sites**: **mlp** —
`a`,`b` at the post-RMSNorm `mlp_input`, the result at `mlp_output` (where the SPD components read /
write); **resid** — `a`,`b` at `resid_pre_mlp` (the residual stream, Feucht's site), the result at
`resid_pre_mlp + mlp_output` (residual after the MLP write). Fit separately per task.

Output (fixed dir per the objective: **`<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/`**, not the
source run's `analysis/`): `coordinates_<op>.json` — op/symbol/layer/`space`/`sites`/source/grid
metadata + `features[site][operand][period]` → `{period, r2, w_cos, b_cos, w_sin, b_sin}`, the `w_*`
vectors `d_model`-long (`period` is the ratio `r` in log space). Empirically the operands fit
cleanly at every period (held-out `r2 ≈ 0.98–0.99`); the result is cleanest at long periods and
weakest around period 20 (add) / mid periods (sub, mult); the two sites give near-identical `r2`
(they differ only by RMSNorm, which the linear probe absorbs).

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
- `--feucht-probes-dir`: dir of Feucht et al.'s *downloaded* probes (default
  `<coordinates-dir>/feucht_addition_resid`, `probe_period{T}_{cos,sin}.pt`); when present adds a
  `feucht` site (add/result only). Absent → the site is silently omitted
- `--output-dir`: overrides the output dir

CPU (no forward pass). A self-contained canvas applet (vanilla JS, no CDN) for comparing
subcomponents / neurons against the circular features. For a chosen **basis task**, **site**, and
**operand** (first operand `a`, second operand `b`, or the output result), each period's probe gives
one plot, side by side. Points are the chosen **activation task**'s activations mapped to the
probe's **predicted `(cos, sin)`** (`w_cos·x + b_cos`, `w_sin·x + b_sin`) — a clean feature traces
the unit circle, exactly Feucht's plot — so e.g. subtraction activations can be viewed on addition's
probe. The **site** dropdown picks which probe: `mlp` (a/b ← `mlp_input`, result ← `mlp_output`; the
spaces the SPD components read/write), `resid` (our probe on the residual stream), or — when Feucht
et al.'s downloaded probes are staged — `feucht`, projecting onto their *shipped* addition-output
probe (variable a+b at the resid layer-output; add/result only, other basis/operand locked) to
eyeball our fit against theirs. When the sin axis is degenerate (period 2, `sin(2πv/2)=0` so `w_sin≈0`) the plot falls back
to an orthonormal frame — `e1` along `w_cos`, `e2` the top activation-variance direction ⊥ `e1`
(rescaled to the cos-axis spread so the residue split stays visible), as in Feucht et al. A
crosshair+ring marks the projected circle centre. Points colour by `a`, `b`,
`a+b`, `a-b`, `a×b` (computed from the operand grids, independent of the active task), each either
raw or by `(value − offset) mod m` via a `mod` + `offset` form (options from the task's
`subcomp_periods`, like the subspace-scatter applet); a **model accuracy** option (1 if the base
model's argmax next token is the correct answer, else 0) and a **P(correct)** option (its
probability mass on the correct token) — both shown when the shared `arithmetic_map/results.tsv`
covers a task, for the active task's operation; and a **CI (selected)** option (per task: shown when
some `inner_activations_<op>.tsv` carries a `ci` column, greyed for tasks lacking it or for a
subcomponent absent from a task's TSV) paints each point by the currently-selected subcomponent's
causal importance on that prompt. All
colouring uses a **viridis 0→1** map with a single shared legend (for a `mod m` residue the scale
runs 0…`m−1`, the reachable max). Colour/mod/offset changes and selecting a new subcomponent
recolour in place (zoom preserved). Scroll zooms, drag pans. Hovering a point shows its selected
colour value and prompt (`<value> (a<sym>b=)`). At the **mlp** site only (the components' V/U
directions live in MLP space), the **unit** subcomponent directions (gate/up `V` for input operands,
down `U` for the result) are drawn as bright-red arrows in the same predicted-`(cos,sin)` frame
(same colour for subcomponents and the neuron overlay), scaled to the point cloud; only those whose
in-plane norm ≥ the typed threshold show; hovering shows the label + ‖proj‖. At the **resid** site
the arrows / overlay / threshold controls are disabled (cross-space projection isn't meaningful) —
just the Feucht circle. Clicking a subcomponent **or** neuron arrowhead selects it and draws a bar
chart of its **angle to every MLP-site Fourier plane** (arccos of the in-plane norm; 0° = the
direction lies in the plane), grouped by basis task · operand with one bar per period — using the
full un-floored projection so orthogonal planes appear too; a clicked subcomponent additionally
opens its inner-activation `(a, b)` heatmaps (one per task) at the bottom. An **overlay** toggle
swaps subcomponents for **individual neurons'** directions — gate/up read rows or the down write
columns of the frozen target weight (a neuron-matrix dropdown). Reads the probes
(`coordinates_<op>.json`; asserts each was fit at the checkpoint's layer and carries `sites`), the
run's `hidden_activations_<op>.npz` (incl. `resid_pre_mlp`) / `alive_filtered_<op>.tsv` /
`subcomp_periods_<op>.tsv`, and the checkpoint U/V + target MLP weights (mmap). No forward pass.

Output: `<run_dir>/analysis/fourier_scatter/{index.html,data.js}` — `data.js` holds, per
(site, basis task, operand, period): the predicted-`(cos,sin)` point clouds for every activation
task and the projected circle centre (both sites); plus, per (basis task, operand, period): the
floor-passing subcomponent and neuron arrow coords + in-plane norms (mlp site), and the kept
subcomponents' inner-activation and (when available) CI grids per task (fp16 base64). Smoke-test
with `headless_check.py`.

**build_result_feature_construction.py**

args:
- the path to a decomposed model (checkpoint)
- `--probes-dir`: dir holding the `probes_post.json` + `probes_pre.json` files from the
  `probes/` pipeline (default the shared `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`; each
  file's site and layer are asserted)
- `--census-dir`: the neuron census dir (default `runs/neurons/`) — supplies the measured
  ablation KL (`candidates.tsv`, overridden by `ablation_full_add.npz` where present)
- `--kl-thr`: measured max-KL floor for a neuron to be selectable (default 0.01)
- `--last-ci-thr`: last-token CI an alive subcomponent must reach on ≥1 prompt to be included
  (default 0.01) — the alive filter may admit components acting only at operand positions,
  which are invisible/ablatable-to-no-effect at the `=` token this applet reads; the per-prompt
  `ci` column of `inner_activations_add.tsv` supplies the max
- `--periods`: comma list of probe periods shown, one plot per period (default `2,5,10,20,50,100`)
- `--output-dir`

CPU (no forward pass). Asks whether MLP 18 **builds** the result (`a+b`) circular features or
**adds to** structure already in the residual stream — and which neurons / subcomponents build
them. A self-contained canvas applet (`index.html` + `data.js`): one plot per period, in the
probes' predicted-`(cos, sin)` frame (red ring = unit circle; column captions carry both
sites' r²). **Five rows** share one zoomable view per column: (1) pre-MLP residual on the
**pre**-fit probes — the pre-existing structure in its own best frame; (2) pre-MLP residual on
the **post**-fit probes — how much of it already lies in the final representation's frame;
(3) post-MLP residual on the post probes; (4) row 3 with **one ablated neuron or
subcomponent**; (5) the **alive-components-only MLP** — the residual after an MLP whose
decomposed matrices are reconstructed from just the applet's alive subcomponents (binary
mask, delta off; computed at build time) — how much of the representation the kept circuit
rebuilds. **Hovering a point marks the same prompt in every plot** (overlay canvases, no
repaint), so a prompt's trajectory across the rows reads directly. Colour by `a` / `b`
/ `a+b` with mod + offset, or by the **ablation displacement** — `Δ (plane)` (the 2D
original→ablated move in each column's own probe plane) or `Δ (total)` (the norm over all
planes) — prompt-keyed so every row colours, scaled to the 99th percentile, grey where
undefined (no ablation), value in the tooltip; or by the **alignment** of the activations with
the selected item's unit direction (signed dot product, diverging blue–grey–red on a symmetric
99th-pct scale): `align · read dir` = `x·V̂` (gate/up; down uses `h·V̂`) or the neuron's
normalized gate preactivation, `align · read dir (up row)` = the neuron's normalized up
preactivation (neuron-only), `align · write dir` = the gate/up preactivation vector onto `Û`,
or the post residual onto a down `Û` / the neuron's unit down column. The ablated item also
draws **red direction arrows**: a gate/up subcomponent's unit read direction (`V̂`; a neuron
its gate + up rows) on row 1's pre frame, a down subcomponent's unit write direction
(`Û`; a neuron its down column) on row 2's post frame — mapped through the same linear map
as the points (`v̂·W` per plane, so lengths compare across periods; display-scaled per row to
the unit ring, raw ‖proj‖ on arrowhead hover). RMSNorm between the pre residual and the
gate/up input is absorbed by the probes, so raw directions are used (as in
`build_direction_scatter`). An **ablate**
dropdown picks neurons or subcomponents; a **period dropdown** (subcomponents
only) filters the list to one period group (`period N` / `×r` / `no period`, labels from
`subcomp_periods_add.tsv`; searching a filtered-out id resets it);
the side panel lists them with checkboxes, `(a, b)` thumbnails (post-SwiGLU activation /
inner activation), KL / mean-CI / period info, and a **search-by-id** box (`12023` for neurons,
`g124` for subcomponents). **One item is ablated at a time** (picking a new one replaces the
old), always exactly on the full grid: a neuron subtracts `act_j · (w · W_down[:, j])`
(additive over neurons), a down subcomponent subtracts `(h · V_c) · (w · U_c)`, and gate/up
subcomponents ship their exact full-grid deltas (SwiGLU re-evaluated at build time with the
rank-1 term removed).

Ablation math lives entirely in the post frame (the only frame with an ablated row); the pre
probes contribute one projected cloud + the caption r². Reads the two probes JSONs, the run's
`hidden_activations_add.npz` / `alive_filtered_add.tsv` / `subcomp_periods_add.tsv` /
`inner_activations_add.tsv`, the
census `candidates.tsv` (+ `ablation_full_add.npz`), and the checkpoint U/V + frozen MLP
weights (mmap). Addition only (the probes' result variable is `a+b`).
Output: `<run_dir>/analysis/result_feature_construction/{index.html,data.js}`.
Smoke-test with `headless_check.py`.

**build_polytope_explorer.py**

args:
- the path to a decomposed model
- `--ops`: comma list of operations to include (default: every op with
  `hidden_activations_<op>.npz` + `alive_filtered_<op>.tsv` + `inner_activations_<op>.tsv`
  in `analysis/datasets/`)
- `--top-gates`: alive gates stored per op, by output relevance (default 64)
- `--output-dir`

CPU. Emits a self-contained HTML applet (`index.html` + `data.js`, `file://`-openable, no
server/CDN/GPU) into `<run_dir>/analysis/polytope_explorer/`. A SwiGLU MLP is piecewise
(approximately) linear: wherever every gate preactivation keeps its sign, the MLP applies one
roughly fixed linear map (ignoring the negative silu bump near zero). The applet colours the
op's `(a, b)` operand grid by **which combination of alive gates is positive** — one colour =
one combination = one polytope — answering "which prompts activate the same combination of
gates?"; a second mode colours by the combination of **causally-important subcomponents**
(CI > a typed threshold) instead.

**Alive gates** are the neurons of the decomposed layer's MLP whose gate preactivation takes
both signs across the op's grid (a never-flipping gate contributes no polytope boundary
there). Most of `d_int` flips somewhere, so only the `--top-gates` most output-relevant are
stored, ranked by `std over the grid of silu(gate_j)·up_j` · `||down column j||` (the size of
the neuron's contribution to the MLP output). In the applet a **top-k** control (default 8;
on the reference add run the top-20 combinations then cover ~70% of the grid) plus per-gate
checkboxes choose which stored gates form the combination; CI mode has its own top-k
(default the 8 highest-mean-CI subcomponents) and per-subcomponent checkboxes — including
every subcomponent shatters the map into thousands of singleton combinations.
Combinations are ranked by pixel count; the most frequent
`max colours` get distinct colours (golden-angle HSL), the rest pool into grey. The legend
lists each coloured combination (count + which bits are on); hovering a row highlights its
region on the map. Hovering a map pixel shows a tooltip (`a`, `b`, result, combination rank)
and drives the right panel; clicking pins the pixel (black square marker).

The right panel shows one `(a, b)` heatmap thumbnail per stored gate (its preactivation grid;
title shows output relevance + fraction-positive) and per filtered-alive subcomponent (CI or
inner-activation grid, per a dropdown). At the current pixel, active thumbnails (gate > 0 /
CI > thr) get a red border and inactive ones dim; a dot marks the pixel on every thumbnail
and a value readout shows each item's value there. **Operation selector** across every
included op; map and thumbnail pixels-per-value controls; drag-resizable divider.

Reads, per op: `hidden_activations_<op>.npz` (gate/up grids), `alive_filtered_<op>.tsv`, and
`inner_activations_<op>.tsv` (must carry the `ci` column — rerun `collect_inner_activations`
if not); plus the checkpoint's target down-projection weight (mmap). No forward pass.
`data.js` holds fp16-base64 gate / CI / inner `(a, b)` grids (~14 MB for two ops at the
defaults). Smoke-test with `headless_check.py`.

**collect_projection_kl.py**

args:
- the path to a decomposed model
- `--ops`: comma list of operations to run (default `add,sub`)
- `--batch-size`: forward-pass chunk size over the prompt grid (default 512)
- `--ci-thr`: switch to **per-prompt** sets — subcomponents with lower-leaky CI above
  this at the last position of each prompt (default off = static alive mode)
- `--alive-tsv`: override the alive-components TSV (default the run's
  `analysis/datasets/alive_subcomponents.tsv`; static mode only, incompatible with `--ci-thr`)
- `--output-dir` / `--output-fig-dir`: override the dataset dir / figure dir
- `--slurm` + knobs: submit as a single-GPU job (see `CLAUDE.md` → "GPU scripts run via
  SLURM"); the login node has no GPU.

Tests whether the decomposition's subspaces are the *causally relevant* subspaces of the
decomposed layer's MLP, not just a sufficient circuit: if they are, projecting the
**original model's** activations onto them (weights unchanged) should preserve the output
about as well as running the circuit itself. Per prompt of each op's `1..100 × 1..100`
grid it measures `KL(P_target || P_variant)` of the last-position next-token distribution
for three variants, each intervening **only at the last (`=`) position** (where the set
is defined) and leaving decomposed attention and earlier positions untouched:

- `projected_inputs` — a forward pre-hook on the MLP module projects the post-RMSNorm MLP
  input onto the span of the set's gate_proj + up_proj `V` columns; the unchanged MLP
  weights consume the projected input.
- `projected_outputs` — a forward hook projects the MLP output (the residual-stream
  write) onto the span of the set's down_proj `U` rows before it is added to the residual.
- `alive_only` / `ci_only` — the MLP weight is replaced by the set's subcomponent sum:
  component masks 1 on the set / 0 off it and the weight-delta **off** at the last
  position; all components + delta on at earlier positions (exact reconstruction there).

The subcomponent set is either **static** (default): the alive set from
`alive_subcomponents.tsv`, one subspace shared by every prompt, circuit variant
`alive_only`; or **per prompt** (`--ci-thr=X`): the subcomponents whose lower-leaky CI
(continuous sampling) at the last position exceeds `X` on that prompt — the reference
forward then also feeds the CI fn (`cache_type="input"`), each prompt gets its own
SVD bases and circuit mask, the circuit variant is `ci_only`, and outputs land in
`projection_kl_ci<X>/` instead of `projection_kl/`. An empty per-prompt set projects to
the zero vector.

Bases are orthonormalised by SVD in float32 (singular values > `1e-5 · s_max` kept); the
projections run in float32 and cast back to the model dtype. The reference is the raw
target model, so the circuit variant carries the decomposition's reconstruction-error
floor while the projections don't. A first-batch wiring check asserts all-on + delta
reproduces the target (mean KL < 0.5). One reference + three variant forwards per chunk.

Outputs (defaults in the run's `analysis/` layout; `<name>` = `projection_kl` or
`projection_kl_ci<X>`):
- `datasets/<name>/data_<op>.npz` — `kl` [3, N, N] (variant order in `variants`),
  `token` [3, N, N] (per-variant argmax token id), `orig_token` / `orig_prob` [N, N];
  with `--ci-thr` also `n_ci_in` / `n_ci_out` [N, N] (per-prompt set sizes: gate+up V
  vectors / down U vectors).
- `datasets/<name>/summary.tsv` — per (op, variant): mean/median/q95/max KL + argmax
  agreement.
- `datasets/<name>/meta.json` — set definition (alive counts + subspace ranks, or
  `ci_thr`), KL direction, token-id → string decode map.
- `<name>/kl_heatmaps_<op>.png` — the three KL(a, b) heatmaps, shared log colour scale,
  per-panel mean KL in the title.

---

## Neuron census (`neurons/`)

A decomposition-free pipeline probing the frozen base model's L18 MLP **neurons** over the
0..200 operand grids (`a<op>b=` for add / sub — 201×201 prompts per op, every prompt exactly
5 Llama-3 tokens: `<BOS> a op b =`). Everything lands in the shared
`<PARAM_DECOMP_OUT_DIR>/runs/neurons/` census dir (like `fourier_probes/`), not a run's
`analysis/` — the model_path argument only locates the frozen target model. Shared helpers in
`neurons/common.py`: `VALUES` (0..200), `PERIODS` (2, 5, 10, 20, 25, 33, 50, 100), `OFFSETS`
(±1, ±2, ±5, ±10, ±20, ±25, ±50, ±100), prompt/answer-token grid builders, `token_value_map`.

**neurons/collect_neuron_activations.py**

args:
- the path to a decomposed model (only to locate the frozen base model)
- `--ops`: comma list of ops to run (default `add,sub`)
- `--layer` (default 18), `--batch-size` (default 256), `--out-dir` (default the census dir)
- `--slurm` + the usual SLURM knobs

Per op, one clean forward over the grid capturing at the `=` position: `gate_preact` /
`up_preact` `[201, 201, 14336]` fp16 (post-SwiGLU `silu(gate)·up` is derivable), `mlp_input`
`[201, 201, 4096]` fp16 (the post-RMSNorm MLP input — what neuron gate/up rows and gate/up
subcomponent V vectors read, so subcomponent inner activations are CPU-derivable on the same
grid), and the model-answer baseline: `orig_token` / `orig_prob` (argmax next token),
`correct_token` / `correct_prob` / `correct_logprob` (the true answer's **first** token — sums
are single tokens, negative differences start with `-`), `is_correct`.

Outputs: `activations_<op>.npz`, `baseline_<op>.npz`.

**neurons/collect_neuron_ablation_kl.py**

args:
- the path to a decomposed model (only to locate the frozen base model)
- `--op` (default `add`), `--stride` (default 5): prompts are `a, b in VALUES[::stride]` —
  stride 5 → the 41×41 dense screen, stride 1 → the full 201×201 grid
- `--neurons-tsv`: restrict to a candidate set (any TSV with a `neuron` column); default all
  14336 neurons
- `--shard-index` / `--shard-count`: contiguous split of the neuron set across SLURM jobs
- `--layer` (default 18), `--batch-size` (prompts per clean forward, default 64), `--chunk`
  (neurons per patched batch, default 256), `--output`, `--slurm` + knobs

Zeroes one neuron's post-SwiGLU activation **at the `=` position only** (equivalently removes
`act_j · W_down[:, j]` from the MLP output there) and measures the next-token effect vs the
clean model. Positions before `=` keep their clean K/V, so the ablated forward re-runs only
layers `layer+1..31` + final norm + lm_head on the single patched token against the clean KV
cache, batched over `(prompt × neuron-chunk)` rows — this is what makes all-14336 × grid
tractable (~1–2 h/op for the screen on one L40). The tail is hand-rolled (RMSNorm → QKV →
RoPE at position 4 → SDPA over 4 cached keys + own → MLP; GQA via `repeat_interleave`);
validated to ~1e-7 max |Δlogit| against a real hooked ablation on a toy Llama. A **null
patch** (delta = 0) per prompt batch measures the bf16 noise floor; its KL grid ships as
`null_kl` and its max is asserted `< 0.02`. Caveat by construction: a neuron acting purely at
operand positions is invisible (its effect is frozen into the clean K/V).

Per (neuron, prompt): `kl` (KL(P_clean ‖ P_ablated), full vocab), `abl_token` / `abl_prob`
(argmax under ablation — decode with `token_value_map` for error-mode analysis: 44 → 43 vs
54), `answer_flip`, `delta_correct_logprob`, and (stride 1 only) `offset_logprob` — ablated
logprob of the first token of `str(answer + δ)` for δ in `OFFSETS`, with the clean
counterpart per prompt in `clean_offset_logprob` (offsets crossing zero degenerate to the
bare `-` token; mask by comparing token ids).

Output: `ablation_screen_<op>.npz` (stride > 1) or `ablation_full_<op>[_shard<i>of<k>].npz`:
`kl` / `abl_prob` / `delta_correct_logprob` fp16 + `abl_token` int32 + `answer_flip` bool
`[n_neurons, n_a, n_b]`, `offset_logprob` fp16 `[n_neurons, n_a, n_b, n_offsets]` (stride 1),
per-prompt `null_kl` / `orig_token` / `clean_offset_logprob`, and `neuron_ids`, `a`, `b`,
`offsets`, `layer`, `op`, `stride`.

**neurons/compute_neuron_periodicity.py**

args: the `activations_<op>.npz`; `--ablation-npz` (full-grid ablation npz — adds KL-grid
scores for its candidate neurons); `--output`.

Translation-invariance score per neuron × channel (gate / up / combined) × lag, where the lag
set (`common.translation_lags()`) is pure-a `(p, 0)`, pure-b `(0, p)` and all mixed `(p, ±q)`
for p, q in `PERIODS` — Pearson correlation between the **planar-detrended** grid and its
shifted self over the overlap. No sinusoid assumption; mixed lags catch checkerboards and
diagonals (`a+b mod p` is invariant along `(k, -k)`, `a-b mod p` along `(k, k)`). A true
period-p pattern also scores ~1 at multiples of p: consumers read profiles, not argmaxes.
Detrending matters: a shifted linear trend self-correlates perfectly, so without it every
magnitude-trending neuron scores ~1 everywhere. CPU-only, minutes for all 14336 × 3 × 2 ops.

Output: `periodicity_<op>.npz` — `score [14336, 3, n_lags]`, `lags`, plus `kl_score` /
`kl_neuron_ids` with `--ablation-npz`.

**neurons/select_candidate_neurons.py**

args: `--census-dir`, `--kl-thr` (default 0.01), `--floor-margin` (default 3 — `kl_thr` must
exceed `floor_margin ×` the measured null-patch noise floor, asserted), `--bound-top`
(default 256), `--output`.

Union of two nets, both feeding the stride-1 full-grid ablation (the unaliased ground
truth): **screen** — max screen KL over any op exceeds `--kl-thr` (argmax flips alone do NOT
qualify: ~10k neurons flip near-tied argmaxes at negligible KL); **bound** — the stride-5
screen only samples `a, b ≡ 0 (mod 5)`, phase-aliasing exactly the periodic neurons this
census is after, so per op the top `--bound-top` neurons by the full-grid perturbation bound
`max |silu(gate)·up| · ‖down col‖` (from `activations_<op>.npz` + `subspace.npz` norms) join
regardless of screen KL. Output `candidates.tsv`: `neuron`, `source` (screen/bound/both),
per-op `max_kl / mean_kl / n_flip / min_dlp / bound`, sorted by overall max KL. This TSV is
the `--neurons-tsv` input of the full-grid ablation run and the candidate list every
downstream neuron script consumes.

**neurons/compute_neuron_subspace.py**

args: model path (only to mmap the frozen L18 MLP weights); `--candidates-tsv` (adds a PCA
over the candidate set's read/write vectors); `--probes-dir` (default the shared
`runs/fourier_probes/`); `--layer`; `--output`.

Projects each neuron's read rows (gate/up) onto the Fourier probe planes fitted at the MLP
input (`norm` site) and its write column (down) onto the planes at the MLP output
(`mlp_out`), per variable (`a`, `b`, `a+b`) and period in `PERIODS`. Ships the in-plane
fraction of the **unit** direction plus each plane's held-out r² (to discount junk planes).
Output `subspace.npz`: `read_frac [14336, 2, 3, 8]`, `write_frac [14336, 3, 8]`,
`read_r2` / `write_r2`, `norms`, and `pca_sv_{gate,up,down}` + `candidate_ids` when given.

**neurons/collect_subcomp_ablation_kl.py**

The subcomponent analogue of `collect_neuron_ablation_kl`, for a run's L18 MLP matrices only
(gate/up/down — attention k/v ablation would invalidate the frozen prefix KV cache, asserted
away). Removes one component's rank-1 `U_c V_c^T` from the frozen weight at the `=` position
(down: direct write patch; gate/up: recompute the SwiGLU with the component's contribution
subtracted from the preactivation — algebraically exact, checked to 6e-14 in f64) and runs the
same patched tail. Defaults to **all C components per matrix** (the point is not trusting the
learned CI); `--components-tsv` (matrix + component columns) restricts. Same stride semantics
and output arrays as the neuron script, with `matrix` / `component` keys; writes
`subcomp_ablation_<screen|full>_<op>.npz` in the run's `analysis/datasets/`.

**neurons/compute_subcomp_neuron_links.py**

args: model path (checkpoint mmap); `--acts-npz` (census `activations_add.npz`),
`--candidates-tsv` (census neuron candidates), `--subcomp-screen-npz` (the run's
`subcomp_ablation_screen_add.npz`), `--subcomp-kl-thr` (measured-KL causality threshold for
components), `--layer`, `--output`.

CPU one-pass producer for the subcomponent story: per matrix the subcomponent
inner-activation grids on the 0..200 addition grid (`x·V_c`, x = MLP input for gate/up /
post-SwiGLU acts for down) + their translation-invariance periodicity + measured-causal
flags; the coupling weights to the candidate neurons (`U[c, j]` for gate/up writes,
`V[j, c]` for down reads) with per-component inner-act stds (`std·|U|` = functional
interaction strength); and the **explanation R²** per candidate neuron × channel — variance
of the neuron's gate/up preactivation grid explained by the sum of all components vs only
the measured-causal ones. Low causal-R² on a high-KL neuron = a causally-important neuron
the decomposition's causal components do not explain. Output
`subcomp_neuron_links_add.npz` in the run's `analysis/datasets/`.

**neurons/build_neuron_census.py**

args: model path (tokenizer only — decodes ablated answer tokens to numbers);
`--census-dir`, `--top-k` (candidates that ship full grids, default 200), `--output-dir`.

The census applet (`<census_dir>/applet/{index.html,data.js}`, vanilla JS/canvas, `file://`).
Left: summary (candidate count, null floors, model accuracy), sortable/filterable candidate
table (min-KL + per-lag periodicity filters), KL-vs-periodicity scatter, baseline accuracy
maps. Right, per selected neuron: KL / Δcorrect-logprob / (ablated-answer − truth) heatmaps
(the last with residue-class error-mode histograms: condition on `a ≡ r (mod m)` /
`b ≡ r (mod m)`), gate/up/combined activation grids, the full per-channel lag-score profile
(click a lag to recompute an **in-browser local windowed periodicity map**), the
answer-offset Δlogprob profile (all prompts and flip-only), and the probe-plane in-plane
fraction table. Grids ship uint8-quantized base64; falls back to 41×41 screen grids when the
full-grid ablation npz is absent. Smoke-test with `headless_check.py`.

**neurons/build_subcomp_census.py**

args: the decomposition run's model path; `--census-dir` (the shared neuron census),
`--output-dir`.

The subcomponent applet (`<run>/analysis/subcomp_census/{index.html,data.js}`), three tabs:
**components** — sortable table (measured ablation KL stats, causal flag, inner-act std, top
periodicity lags) with a detail panel (KL / Δcorrect-logprob / error-mode / inner-activation
grids, offset profile, and the top coupled candidate neurons with their own period chips and
combined-act thumbnails); **connection matrix** — candidate neurons (rows, grouped by
dominant combined-act period; ◆ marks multi-period neurons, i.e. two incommensurate pure
periods above 0.5) × components (cols, grouped by matrix then dominant inner-act period),
cell = signed coupling `U[c, j]` (optionally × inner-act std) or down-read `V[j, c]`,
period-matched blocks being the thing to look for; **explanation** — per candidate neuron,
neuron ablation max KL vs the R² of its gate/up preact explained by measured-causal
components (r2_all as small dots for reference); clicking reconstructs actual / explained /
residual grids in-browser from the shipped inner grids × couplings. Prefers
`subcomp_ablation_full_add.npz` over the screen when present.
