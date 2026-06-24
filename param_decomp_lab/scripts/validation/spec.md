# Validation script specs

One entry per script. Each describes the arguments, behaviour, and output schema. See
`CLAUDE.md` in this folder for the shared CLI design.

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

Output TSV (default `sample_target_data.tsv` in the decomposed model's folder), **long
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

Output 1 — TSV (default `alive_components.tsv` in the run folder), one row per alive
subcomponent (`count_active > 0`), sorted by `(layer, matrix, component)`. Same schema as
the SPD `find_alive_components.py`:
- `layer` — block number from the module path (e.g. 18)
- `matrix` — the rest of the module path (e.g. `mlp.gate_proj`)
- `component` — the subcomponent index
- `fraction_active` — `count_active / count_total` (fraction of seen valid positions where
  it was active)
- `max_ci` — max observed lower-leaky CI
- `mean_activation` — mean `V^T x` over the positions where it was active

Output 2 — JSON (default `alive_components_per_position.json` in the run folder), the active
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

Output TSV (default `ablate_component_groups.tsv` in the run folder), one row per
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
- `--output-dir`: overrides the figure folder (default `<json_dir>/ci_heatmaps/`)

CPU-only (no model loaded). For each token position, draws one heatmap with prompts on the
y-axis (tiny text) and alive subcomponents on the x-axis, faceted by matrix, coloured by
lower-leaky CI (`RdPu`, 0–1, shared colorbar). The x-axis alive set is the union of
components active across the *selected* prompts (so it adapts to `--grep`); a cell is the
component's CI at that position for that prompt, 0 where inactive. Writes one PNG per
position (`position_<pos>.png`).

**plot_ab_heatmaps.py**

args:
- the per-position JSON from `find_alive_components.py` (prompts must be `a+b=`)
- `--output-dir`: overrides the figure folder (default `<run_dir>/figures/ab_heatmaps/`)

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
  in the run folder)
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
- `--alive-tsv`: `find_alive_components` output (default `alive_components.tsv` in the run folder)
- `--batch-size` (default 256), `--output`, `--output-alive`, plus `--slurm` (+ knobs)

For every existing-alive MLP subcomponent and every prompt, computes the normalized inner
activation `(x · V_c) / ||V_c||` at the last token (`x` = the cached module input, already
post-RMSNorm for gate/up and post-SwiGLU for down, so nothing is reapplied). Also computes
each component's mean last-token CI over the grid and keeps those above `--mean-ci-thr`. GPU.
Outputs:
- `inner_activations_<op>.tsv` — one row per (kept component, prompt): `a, operation, b,
  matrix, subcomponent, inner_act` (`operation` is the infix symbol; `matrix` the bare proj).
- `alive_filtered_<op>.tsv` — the surviving alive set: `layer, matrix, component, mean_ci`.

**compute_subcomp_periods.py**

args:
- the `inner_activations_<op>.tsv` from the previous script (positional)
- `--output`

CPU. Rebuilds each subcomponent's `[N, N]` inner-activation grid and measures the periodicity
of its `f(a)` (mean over b) and `f(b)` (mean over a) marginals two ways: **autocorrelation**
(best lag in `1..N//2`; score = unit-`r(0)` autocorrelation there) and **FFT** (peak nonzero
frequency → `period = round(N/k)`; score = that frequency's fraction of DC-removed power).
Reads `alive_filtered_<op>.tsv` (same dir) for the `layer`/full-`matrix` columns. Output
(default `subcomp_periods_<op>.tsv`): `layer, matrix, component`, the four
`{autocorr,fft}_{a,b}_{period,score}` columns, and a representative `period` / `period_axis`
(the FFT axis with the stronger peak — used for downstream sorting). Note: autocorrelation
tends to return lag 1 for smooth/monotone marginals, so the representative period uses FFT.

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

Outputs to `<run_dir>/figures/subcomp_cosine/`.

**build_neuron_connection_explorer.py**

args:
- the path to a decomposed model
- `--op`: `add` (default) / `sub` / `mult`
- `--conn-floor`: min |connection strength| stored per (subcomponent, neuron) (default 0.1)
- `--top-neurons`: cap on neurons stored per subcomponent (default 60)
- `--output-dir`

CPU. Emits a self-contained HTML applet (`index.html` + `data.js`, `file://`-openable, no
server/CDN/GPU) into `<run_dir>/figures/neuron_explorer_<op>/`. Connection strength uses the
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
