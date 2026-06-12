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
