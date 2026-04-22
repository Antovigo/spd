**find_alive_components.py**
args: 
- the path to a decomposed model
--ci-thr: the minimal causal importance for a component to be considered active (default 0.01)
--n-batches: the number of batches of data to run
--nontarget: use nontarget data (nontarget_task_config). If a nontarget dataset is not specified in the config, raise an error
--output: overrides the path to write the data to

This script runs a decomposed model on <n-batches> of data (target or nontarget), and stores a list of the components that are active at least once over all the seen data. If the dataset is a series of prompts (as often in targeted decomposition's target data), it only runs one batch that contains each prompt once. Otherwise, the batch size is the same as the one defined in the decomposition's config file.

The output is a TSV file where each row is an alive component. The columns are:
- layer (the number of the block, where a block is an entire MLP or attention block)
- matrix (e.g. "attn.q_proj" or "mlp.c_fc")
- component (the component's index)
- fraction_active: fraction of seen inputs x positions on which the component is active
- max_ci: the maximum observed causal importance
- mean_activation: the mean observed inner activation of that component (V^Tx) on inputs where it is active
  
Unless --output is specified, the TSV file is saved to the decomposed model's folder. The filename is "alive_components.tsv" for target data, and "alive_components_nontarget.tsv" for nontarget data.

**nontarget_activity.py**
args:
- the path to a decomposed model
- the path to a list of alive components (TSV from `find_alive_components.py`)
--harvest-subrun-id: optional subrun id (e.g. "h-20260211_120000") inside `SPD_OUT_DIR/harvest/<run_id>/`. If omitted, uses the most recent subrun.
--output-summary: overrides the path to write the summary TSV
--output-sequences: overrides the path to write the activation sequences

This script does NOT run the model. It consumes the pre-computed statistics stored by the harvest pipeline (`spd/harvest/`), which is assumed to have already been run on the nontarget / general-distribution data for this decomposition. The harvest output lives at `SPD_OUT_DIR/harvest/<run_id>/<subrun_id>/harvest.db` (and associated `.pt` files) and is opened via `HarvestRepo`.

For each alive component from the input TSV, the script looks up its entry in the harvest data using the component key `f"{module_name}:{component_index}"` (where `module_name` is the full decomposed module path, e.g. `"h.0.attn.q_proj"`). If a component has no harvest entry (e.g. it was not decomposed in that harvest run), it is skipped with a warning.

Summary output (TSV, default filename `nontarget_activity.tsv` in the decomposed model's folder, columns):
- layer (the block number)
- matrix (e.g. "attn.q_proj" or "mlp.c_fc")
- component (the component's index)
- firing_density: fraction of harvest tokens on which the component fired (from `ComponentSummary.firing_density`)
- n_firings: absolute count of positions where the component fired (from `CorrelationStorage.count_i`)
- n_tokens_seen: total number of tokens seen by harvest (from `CorrelationStorage.count_total`, identical across rows)
- mean_ci: mean `causal_importance` activation conditional on firing (from `ComponentSummary.mean_activations["causal_importance"]` when present)
- mean_component_activation: mean `component_activation` (V^Tx) conditional on firing (from `ComponentSummary.mean_activations["component_activation"]` when present; otherwise blank)

Sequences output (JSONL, default filename `nontarget_activity_sequences.jsonl` in the decomposed model's folder). One line per (alive component × activation example) with fields:
- layer
- matrix
- component
- example_idx: index into the harvested `activation_examples` list for that component
- token_ids: list[int] — the input token window around the firing (unpadded)
- firings: list[bool] — whether the component fired at each position in the window
- activations: dict[str, list[float]] — per-position activation values, keyed by activation type (e.g. `"causal_importance"`, `"component_activation"`)
- text: the decoded token window as a single string (requires `config.tokenizer_name`)

JSONL (rather than TSV) is used for the sequences file because each row naturally contains variable-length lists and a small dict; embedding those in TSV cells would be awkward to parse.

Only LM tasks are supported (the harvest data is keyed by token sequences).


**effect_of_ablation.py**
args:
- the path to a decomposed model
- the path to a list of components, from the "find_alive_components" script
--n-batches: the number of batches of data to run
--nontarget: use nontarget data (nontarget_task_config). If a nontarget dataset is not specified in the config, raise an error
--output: overrides the path to write the data to

The list of components only needs to have at least the layer, matrix and component columns. The other columns are not mandatory.

For each alive components, we want to measure how the model behaves when it is ablated. So we make a version of the model where that one component is ablated, and all the other components (including, crucially, the delta component) are fully enabled.

For each input sequence and each position, we collect the KL-divergence compared to the original model.
  
Again, if the dataset is a series of prompts (as often in targeted decomposition's target data), it only runs one batch that contains each prompt once. Otherwise, the batch size is the same as the one defined in the decomposition's config file.

Implementation:
- Loop order is batches outer, components inner. For each batch we do a single baseline forward pass of the underlying target model (via the ComponentModel with no `mask_infos`) and reuse its logits across every ablation on that batch. Then we iterate over the rows of the components list and for each one perform a forward pass with a mask that ablates only that component.
- "Ablated version of the model": `component_mask` is all-ones for every decomposed module, except the single targeted `(module, component_index)` entry, which is set to 0. The delta component is fully enabled — `weight_deltas_and_masks` uses the model's weight deltas with a delta mask of 1.0 broadcast over the batch/position dims. This guarantees that if no component were ablated, components + delta would reconstruct the original target model exactly; so any output difference is attributable to the ablated component alone. The baseline we compare against is the original target model (equivalent to calling the ComponentModel with no `mask_infos`), not the unablated SPD reconstruction.
- KL divergence is computed per (batch_example, position) as `KL(softmax(orig_logits) || softmax(ablated_logits))`. `orig_pred` is `argmax(orig_logits)`, `orig_prob` is `softmax(orig_logits)[orig_pred]`. `token` is the input token id at that position.
- Only LM tasks are supported (tokens/positions/KL are LM-specific).

Output layout (identical schemas in target and nontarget mode; only the filenames differ). The per-(component, prompt, pos) KL and the per-(prompt, pos) token + orig-prediction data are written to two separate files because the orig-prediction columns are identical across every component for a given `(prompt, pos)` and would otherwise be duplicated N_components times.

- KL file (one row per (component, prompt, pos)), default filename `effect_of_ablation.tsv` (or `effect_of_ablation_nontarget.tsv` with `--nontarget`). Columns:
  - layer (the number of the block, where a block is an entire MLP or attention block)
  - matrix (e.g. "attn.q_proj" or "mlp.c_fc")
  - component (the component's index)
  - prompt (the prompt's index)
  - pos (the token position)
  - kl (the KL-divergence compared to the original model at that position)
- Orig-predictions file (one row per (prompt, pos)), default filename `orig_predictions.tsv` (or `orig_predictions_nontarget.tsv` with `--nontarget`). Columns:
  - prompt, pos
  - token (input token id)
  - token_str (decoded input token)
  - orig_pred (id of the most likely token under the original model)
  - orig_pred_str (decoded orig_pred)
  - orig_prob (probability of orig_pred)

The two default paths are overridable with `--output-kl=PATH` and `--output-orig=PATH`.

`ablated_pred` / `ablated_prob` are not written — downstream ranking (`find_swap_candidates`) only uses the original model's top-1 prediction (to assert correctness at the task position) and the raw KL; the ablated model's argmax carries no extra signal beyond the KL distribution itself.

**Summary-only mode (`--summary-only`)**

When the full per-(component, prompt, pos) KL file would be too large to keep around (e.g. ablating hundreds of alive components on a large nontarget dataset), pass `--summary-only`. The script runs the exact same ablation loop but does not write `effect_of_ablation*.tsv` or `orig_predictions*.tsv`; instead it maintains one streaming aggregate per component:
- a t-digest (from `pytdigest`) for the quantile,
- a running sum and count for the mean,
- a running max.

At the end it writes a single summary TSV with columns `layer, matrix, component, n_positions, mean_kl, kl_q{pct}, max_kl` — the same schema `summarize_nontarget.py` produces — defaulting to `effect_of_ablation_summary.tsv` (or `effect_of_ablation_nontarget_summary.tsv` with `--nontarget`) and overridable via `--output-summary`. The quantile percent is controlled by `--quantile` (default 0.99).

This mode intentionally **skips** the task-A/B prompt-exclusion logic that `summarize_nontarget.py` applies; it's meant for workflows (e.g. domain-decomposition runs like CSS) where there is no single task prompt to exclude. For the targeted-decomposition flow that needs exclusion + monitoring alerts, keep the two-step `effect_of_ablation` → `summarize_nontarget` pipeline.

Summary-only mode supports checkpoint/resume via `--checkpoint-every=N` (default `0`, disabled). Every N processed batches the streaming state (t-digest centroids + running totals + components list + `batches_done`) is pickled atomically to `<summary>.ckpt` next to the summary TSV. On startup, if that file exists, its state is loaded and the data loader is fast-forwarded past the batches it already covers — so the same command can be rerun after an interruption. The checkpoint file is deleted after the final summary TSV is written. Resuming assumes a deterministic loader (same `--split`/`--batch-size`/etc.); the checkpoint also asserts the components list matches, so swapping the shortlist between runs is rejected. `--checkpoint-every` is only valid with `--summary-only`.

**summarize_nontarget.py**
args:
- the path to a decomposed model
- the path to the nontarget KL file (`effect_of_ablation_nontarget.tsv`)
- the path to the nontarget orig-predictions file (`orig_predictions_nontarget.tsv`)
--task-a / --task-b: JSON dicts `{"prompt": ..., "target": ...}` (used for prompt exclusion and for the monitoring alerts)
--quantile: quantile of per-position KL used as the side-effect score, in `(0, 1)` (default 0.99)
--prompts: optional override for the prompts file
--output: overrides the output path (default `nontarget_summary.tsv` in the decomposed model's folder)

Produces a compact per-component summary that `find_swap_candidates` consumes directly instead of re-scanning the raw nontarget files. Decoupling the two steps means that tweaking `--top-k` in `find_swap_candidates`, or trying different task prompts that share the same nontarget data, doesn't require another full pass over the (large) nontarget KL file.

Prompt exclusion. Any nontarget prompt whose token sequence contains either task A's or task B's tokenised prompt as a contiguous sub-sequence is dropped entirely (all its positions, for all components). Such sequences reproduce the target context and legitimately produce large ablation-induced KL that would otherwise bias the summary.

Procedure (two streaming passes):
1. Read the nontarget orig-predictions TSV to build each prompt's token sequence (for the exclusion test) and collect the monitoring alerts below.
2. Read the nontarget KL TSV and append each row's `kl` to its component's list iff the prompt isn't excluded.
3. For each component, compute `n_positions` (count of kept KLs), `mean_kl`, `quantile_kl` (at `--quantile`), and `max_kl` with `numpy`.

Output TSV columns: `layer, matrix, component, n_positions, mean_kl, quantile_kl, max_kl`.

Nontarget monitoring (printed to stdout, not written to the TSV):
- The set of excluded prompt indices.
- One line per `(prompt, pos)` where the original model predicts one of the target tokens (deduplicated by position since `orig_pred` / `orig_prob` are identical across components). Format: `[nontarget-hit] task=A prompt=123 pos=45 ' foo bar baz qux quux' -> orig: ' np' (0.870)`. Lines on prompts that were dropped from the summary are suffixed with ` [in-excluded-prompt]`.


**find_swap_candidates.py**
args:
- the path to a decomposed model
- the path to the target KL file (`effect_of_ablation.tsv`, per-(component, prompt, pos))
- the path to the target orig-predictions file (`orig_predictions.tsv`, per-(prompt, pos))
- the path to the nontarget summary TSV (produced by `summarize_nontarget.py`)
--task-a: JSON dict `{"prompt": <prompt text>, "target": <next-token text>}` for task A
--task-b: JSON dict `{"prompt": <prompt text>, "target": <next-token text>}` for task B
--high-kl: lower cutoff for the "on-task" KL (ablating the component on its own task must swing the distribution by at least this much; default 0.5)
--low-kl: upper cutoff for both the "off-task" KL (other task) and the nontarget KL quantile (default 0.1)
--prompts: optional override for the prompts file
--output: overrides the path to write the data to

The script finds `(component_A, component_B)` pairs in the same decomposed matrix that are good candidates for swapping their `U` (output) vectors. A candidate pair `(a_comp, b_comp)` in a given `(layer, matrix)` satisfies all six cutoffs simultaneously:
- `a_comp`'s KL at task A's target position > `high_kl` (it matters for task A),
- `a_comp`'s KL at task B's target position < `low_kl` (it does NOT matter for task B),
- symmetrically for `b_comp` on task B / task A,
- `a_comp`'s and `b_comp`'s nontarget KL quantile (from the summary file) both < `low_kl`.

There is no top-k filter: every pair passing the cutoffs is written. Pairs are sorted by `margin = min(margin_i)` across the six cutoff margins, each computed in log space as `log(kl / cutoff)` for "on-task" conditions and `log(cutoff / kl)` for "off-task"/nontarget conditions (positive values = passes, larger values = further from the cutoff in log units), so the top rows of the TSV are the pairs that robustly meet every criterion. Using log-ratios means a component that is 10× below the `low_kl` cutoff contributes the same margin as one that is 10× above `high_kl`, regardless of the absolute cutoff scales.

Task resolution is the same as in `summarize_nontarget` — the prompt must appear exactly once in the prompts file and the target must be a single token under the tokenizer.

Importance signal. Read from the target KL file at the task position of each task. The script asserts `orig_pred == target_token_id` at each task position via the target orig-predictions file before going further (otherwise the original model doesn't even solve the task).

Side-effect signal. Read directly from the `kl_q<pct>` column of the nontarget summary TSV. The quantile percent `pct` is recovered from the column name and is propagated into the output column names so the TSV is self-describing.

Output TSV (one row per candidate pair that satisfies the cutoffs), columns:
- rank (1-indexed, by descending margin)
- layer, matrix
- a_comp, b_comp (component indices)
- a_on_kl, a_off_kl (A's KL on its intended task vs the other task)
- b_on_kl, b_off_kl
- a_nontarg_kl_q{pct}, b_nontarg_kl_q{pct}
- margin (minimum cutoff margin across the six conditions)

Unless `--output` is specified, the TSV file is saved to the decomposed model's folder as `swap_candidates.tsv`.

**swap_test.py**
args:
- the path to a decomposed model
- the path to an `alive_components.tsv` (produced by `find_alive_components` on the target data) — used to read the mean inner activation of each component on its original inputs
- one or more swap specs, each a positional string of the form `<layer>:<matrix>:<a_comp>/<b_comp>` (e.g. `9:attn.v_proj:52/35`). Both components of a swap must live in the same matrix (otherwise the U vectors have incompatible shapes). Every swap is applied simultaneously in a single forward pass, so the output reflects their combined effect.
--target-a / --target-b: the expected next-token strings for tasks A and B, used only for per-position target-token probability tracking (each must tokenize to exactly one token under the run's tokenizer). Shared across all swaps.
--nontarget: if set, evaluate on the nontarget dataset; otherwise (default) evaluate on the LM target prompts
--n-batches: number of nontarget batches to run (default 1; ignored in target mode, which always uses one batch containing every prompt)
--output: override the default output path

By default the script runs on the LM target prompts (one batch containing every prompt); with `--nontarget`, it runs on `--n-batches` batches of the nontarget dataset instead. Each invocation writes a single TSV. To compare target vs nontarget behaviour, run the script twice.

The script builds a modified version of the decomposed model where, for every supplied swap `(A, B)` in its matrix, the output (U) vectors of components A and B are swapped. Because the two components usually fire with different inner-activation magnitudes (V^T x), a raw U swap would rescale the output by the wrong factor — so we normalise each pair by the ratio of the mean inner activations from `alive_components.tsv`:

```
a_mean = alive_components[A].mean_activation
b_mean = alive_components[B].mean_activation

U[A] := (b_mean / a_mean) * U[B]_original
U[B] := (a_mean / b_mean) * U[A]_original
```

so that when A fires on its original inputs (activation ≈ `a_mean`), the output magnitude it produces now matches what B would produce on B's inputs (`b_mean * ||U[B]_original||`), and vice versa. Both `a_mean` and `b_mean` are asserted to be non-zero.

Multiple swaps are independent rows in different U matrices — or disjoint row pairs in the same matrix. The script reads the original U rows for every swap **before** writing any of them, so overlapping specs (two swaps that both touch the same `(module, component)`) are rejected rather than silently chained through half-modified state.

Model semantics. The swapped model is run through the same `ComponentModel` forward path that `effect_of_ablation` uses: all-ones component masks for every decomposed module, and `weight_deltas_and_masks` with delta-mask 1 everywhere using the **pre-swap** weight deltas (`target_weight - V @ U_original`). This guarantees that outside the swapped rows the model is byte-identical to the original target model, while the swapped components' contributions are reconfigured. The comparison baseline ("orig") is the original target model itself, invoked via `spd_model(batch)` with no `mask_infos`. The swap is performed in place inside a context manager that clones the affected U rows on entry and restores them on exit, so the model state is clean after the script finishes.

Datasets. In target mode, data is the LM `prompts_file` (one batch containing every prompt, same convention as `effect_of_ablation`). In nontarget mode, data is the `nontarget_task_config` dataset, iterated for `--n-batches` batches using the decomposition's `batch_size`. For each batch, `orig_logits` is computed before the swap context and `swapped_logits` inside it, so the final model state is unchanged.

Output. A single TSV, default `swap_test_<slug>.tsv` (target) or `swap_test_<slug>_nontarget.tsv` (nontarget) in the decomposed model's folder. `<slug>` encodes all swaps, joining each pair's `<module>_c<a>_c<b>` with `__` (use `--output` when this gets unwieldy). One row per `(prompt, pos)` with columns:

- prompt, pos, token, token_str (input token id and decoded string)
- orig_pred, orig_pred_str, orig_prob
- swapped_pred, swapped_pred_str, swapped_prob
- p_target_a_orig, p_target_a_swapped (probability of the task-A target token under each model)
- p_target_b_orig, p_target_b_swapped
- kl (`KL(softmax(orig_logits) || softmax(swapped_logits))`)

The nontarget-mode TSV additionally carries a `batch_idx` column so positions from different batches can be distinguished.

Only LM tasks with a `prompts_file` are supported (the script needs prompt-based target data and LM token predictions).


**compare_matched_components.py**
args:
- the path to the first decomposed model (run A)
- the path to the second decomposed model (run B)
--alive-components-a: optional override for run A's alive-components TSV (default `<run_dir_a>/alive_components.tsv`)
--alive-components-b: optional override for run B's alive-components TSV (default `<run_dir_b>/alive_components.tsv`)
--label-a / --label-b: human-readable labels used in the figure (defaults to each run's checkpoint directory name if omitted)
--output-tsv: overrides the TSV output path
--output-fig: overrides the figure output path

Compares two SPD decompositions (typically the same target model decomposed with different seeds or hyperparameters) by 1-to-1 matching their alive components per `(layer, matrix)`. "Alive" is taken directly from the TSVs written by `find_alive_components.py` — run that script on each model first. The matching maximises the absolute cosine similarity between flattened rank-one component weights (`V[:, i] @ U[i, :]`) via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). When the two runs have different numbers of alive components in a matrix, the smaller set is zero-padded so every component gets an entry (padded rows show zero cosine similarity).

Procedure:
1. Load both ComponentModels and their alive-components TSVs. The two models must share the same set of decomposed module paths (asserted), since we assume both decompose the same target architecture.
2. For each `(layer, matrix)` that is present in both alive-components TSVs, run Hungarian matching on the rank-one weight cosine-similarity matrix. Alongside the matched cosine similarity, record the per-pair V cosine similarity, U cosine similarity, and `||U|| * ||V||` norm for each side.
3. Sort pairs within each matrix by descending `|weight_cos_sim|`.

Output TSV (default `matched_components.tsv` in run A's directory). One row per matched pair, columns:
- layer, matrix
- pair_idx (0-indexed within the matrix, sorted by descending |weight cos sim|)
- weight_cos_sim (signed cosine similarity of the matched rank-one weights)
- v_cos_sim, u_cos_sim (signed cosine similarity of the underlying V and U vectors; 0 if either side is a zero vector)
- norm_a, norm_b (`||U|| * ||V||` for each side)

Output figure (default `matched_components.png` in run A's directory). A grid with rows = matrix types, columns = layer indices. Each cell has four subpanels: weight cosine similarity bars, an `||U||·||V||` scatter (run A on y, run B on x, shared axis scale), V cosine similarity bars, U cosine similarity bars. All similarity values are shown as absolute values in the figure (sign is kept in the TSV).

Only `LinearComponents` modules are matched.


**compare_components.py**
args:
- the path to the first decomposed model (run A)
- the path to the second decomposed model (run B)
--alive-components-a: optional override for run A's alive-components TSV (default `<run_dir_a>/alive_components.tsv`)
--alive-components-b: optional override for run B's alive-components TSV (default `<run_dir_b>/alive_components.tsv`)
--random-b: if set, replace run B's component weights with Kaiming-normal samples and draw `len(alive_b[(layer, matrix)])` random indices from run B's total `C` per matrix as the candidate pool (repeated `--n-random-samples` times). This is a control showing what max-cosine looks like when no training has happened, with the pool size matched per matrix to avoid inflation by extra components.
--n-random-samples: number of random draws averaged over (default 10; only used with `--random-b`)
--random-seed: base seed for the random-init and pool sampling (default 0; only used with `--random-b`)
--label-a / --label-b: human-readable labels used in the figure (defaults to each run's checkpoint directory name; in `--random-b` mode label B defaults to `<name>-random`)
--output-tsv: overrides the TSV output path
--output-fig: overrides the figure output path

Max-cosine variant of `compare_matched_components`: for each alive component in A, picks the component in B (from B's alive set, or a random subset in `--random-b` mode) with the highest *absolute* cosine similarity on the flattened rank-one weight (`V[:, i] @ U[i, :]`). The matched pair's V and U cosine similarities, plus `||U|| * ||V||` norms, are recorded for the chosen B component. Because the matching is not 1-to-1, several A components may map to the same B component.

Procedure:
1. Load both ComponentModels and their alive-components TSVs. The two models must share the same set of decomposed module paths.
2. For each shared `(layer, matrix)`, compute the pairwise cos-sim matrix between A's and B's flattened rank-one component weights; pick the argmax column for each row.
3. In `--random-b` mode, repeat step 2 `--n-random-samples` times, each time re-initializing model B's V/U with a fresh seed and sampling `len(alive_b[key])` indices from `range(C)` of the random model. The TSV is one row per `(a_component, draw)`; the figure shows the mean of absolute cos similarities across draws.

Output TSV (default `compare_components.tsv`, or `compare_components_random.tsv` with `--random-b`, in run A's directory). Columns:
- layer, matrix
- draw (0 in the non-random mode; 0..N-1 in random mode)
- a_component (alive component index in model A)
- b_component (matched component index in model B — meaningful per-draw in random mode)
- weight_cos_sim (signed cosine similarity of the matched flat weights)
- v_cos_sim, u_cos_sim (signed cosine similarity of the underlying V and U vectors of the matched pair; 0 if either side is a zero vector)
- norm_a, norm_b (`||U|| * ||V||` for each side)

Output figure (same naming as the TSV, with `.png` suffix). Grid with rows = matrix types, columns = layer indices. Each cell has four subpanels: mean |weight cos sim| bars per A component, `||U||·||V||` scatter (A on y, B on x), mean |V cos sim|, mean |U cos sim|. A components are sorted per matrix by descending mean |weight cos sim|. In the non-random mode the mean is over a single draw (so it's just the absolute value); in random mode it is over all draws.

Only `LinearComponents` modules are matched.


**compare_to_larger.py**
args:
- the path to the targeted decomposition (few alive components)
- the path to a larger decomposition of the same target model
--alive-components-targeted: optional override for the targeted alive-components TSV (default `<run_dir_targeted>/alive_components.tsv`)
--n-random-samples: number of random-init draws for the baseline (default 10)
--random-seed: base seed for the random baseline (default 0)
--chunk-size: number of larger-model components processed at once when computing cos sim (default 64). The larger model's components are iterated in chunks so that `(chunk_size, d_in*d_out)` fits in GPU memory; each chunk updates a running max.
--output-dir: overrides the output directory (default: targeted run's folder)

Sanity-checks whether a narrow targeted decomposition has rediscovered substructures already present in a larger "general" decomposition of the same target model. The intended use: the targeted run was trained against a specific task (prompts-based target data) and has only a handful of alive components; the larger run was trained on general data and has many more components (alive and inactive).

For each alive component in the **targeted** model, for each shared `(layer, matrix)`, searches across **all** components of the larger model (not restricted to its alive set) and picks the one with the highest absolute cosine similarity on the flattened rank-one weight (`V[:, i] @ U[i, :]`). Reuses the matching / TSV-writing helpers from `compare_components.py`, so the per-row schema is identical.

Random baseline: the targeted model's `V` and `U` are re-initialized (Kaiming-normal) and the match is repeated `--n-random-samples` times with different seeds, searching against the same larger-model pool. This tells us how high the max |cos sim| gets when a random rank-one slice is matched against a large component pool — the real targeted components should score noticeably higher if they encode task-specific structure that also lives in the larger decomposition.

Procedure:
1. Load both ComponentModels and the targeted model's alive-components TSV. The two models must share the same set of decomposed module paths.
2. For each `(layer, matrix)` present in the alive-components TSV, match every alive targeted component against every component of the larger model (indices `0 .. C_larger - 1`).
3. Re-init the targeted model's V/U `--n-random-samples` times; rerun step 2 each time. All draws are concatenated into the random-baseline TSV.

Outputs: two TSVs, each suffixed with the larger run's folder name (no PNGs are written):
- `compare_to_larger_<folder-larger>.tsv` — real match (one row per alive component in the targeted model, `draw` always 0).
- `compare_to_larger_random_<folder-larger>.tsv` — random baseline (one row per `(alive component, draw)`).

Both files share the same schema as `compare_components.tsv`: `layer, matrix, draw, a_component, b_component, weight_cos_sim, v_cos_sim, u_cos_sim, norm_a, norm_b`. `a_component` is the targeted-model component, `b_component` the index in the larger model.

Only `LinearComponents` modules are matched.

**completeness.py**
args:
- the path to a decomposed model
- the path to a list of alive components (TSV from `find_alive_components.py`)
--n-batches: number of batches of data to run (default 1; ignored when the target task is a prompts file, which always runs one batch with every prompt)
--nontarget: use nontarget data (`nontarget_task_config`). Errors if no nontarget config is set.
--prompts: optional override for the LM `prompts_file`
--split: optional override for the LM `eval_data_split` (dataset-based tasks only)
--batch-size: optional override for `config.batch_size` (dataset-based tasks only)
--ci-threshold: drop rows where the component's **max** lower-leaky CI on that sequence is ≤ this value (default 0.1). Also used for the per-batch skip: components whose max CI across the entire batch is below threshold are skipped entirely for that batch.
--output: overrides the output TSV path

Checks whether the alive components from the decomposition capture all of the relevant mechanisms — i.e. for each alive component X, whether X's function is also performed in parallel by either the delta component or some of the inactive (non-alive) components.

For each alive component X, two ablated models are run and each is compared to the original target model:
- **case a ("circuit")**: every decomposed module's component mask is 1 on its alive indices and 0 on all non-alive indices, delta is OFF, and X is additionally set to 0. If the alive set is mechanistically complete but X's role is redundant *within* the alive set, KL stays small; if X is uniquely needed (and neither delta nor any inactive component helps, because they are all off), KL is large.
- **case b ("all-on")**: every component is on (mask=1) and delta is ON (mask=1), except X is set to 0. If KL stays small here but is large in case a, something outside the alive set (delta or an inactive component) is doing X's job in parallel — which is the signal we're looking for.

For each sequence (prompt) we record the **mean KL across positions** for both cases, plus the **max CI of X across positions in that sequence**. Per-token KL is not written. A row is written only when `max_ci > ci_threshold` — testing on sequences where X never fires is pure noise.

**Per-batch skip:** before running ablations for X, we check X's max CI over the entire batch (all prompts × positions). If it's below threshold, X is skipped for this batch entirely — no ablation forward passes — which can cut the forward-pass count drastically when the alive set has many components but most are silent on any given chunk of data.

A shared **baseline for case a** is computed once per batch: the circuit setup with *no* component ablated. This lets downstream code subtract the irreducible imperfection of the circuit approximation from `mean_kl_circuit` to isolate X's marginal contribution. The baseline is identical across components at a given prompt, so it's duplicated across the rows for that prompt rather than stored separately — since each prompt only yields rows for the components that actually fire on it (filtered by `max_ci > ci_threshold`), the duplication is bounded. No baseline is needed for case b: running the all-on configuration without ablation yields KL=0 by construction (components + delta exactly reconstruct the target).

Implementation:
- Loop order is batches outer, components inner. For each batch we do one `spd_model(batch, cache_type="input")` forward pass — its `output` is the original target model's logits (since `mask_infos=None`) and its `cache` feeds `calc_causal_importances` to get X's CI. We then run the circuit-no-ablation baseline once. For each alive X we first check whether `ci_X.max() > ci_threshold` over the whole batch; if not, skip. Otherwise we do two further forward passes with `mask_infos`: one for case a, one for case b, with X temporarily zeroed in each mask and restored afterwards.
- KL divergence is computed per (batch_example, position) as `KL(softmax(orig_logits) || softmax(ablated_logits))` (same as `effect_of_ablation.py`), then averaged over positions to get one `mean_kl` per sequence.
- `max_ci` is `ci_X.amax(dim=-1)` (max over positions) for each prompt in the batch.
- Component masks and `mask_infos` are built lazily from the first batch's shape and reused across all batches (the loader guarantees a fixed shape).
- Only LM tasks are supported (tokens/positions are LM-specific).

Output TSV (default filename `completeness.tsv`, or `completeness_nontarget.tsv` with `--nontarget`, in the decomposed model's folder). One row per (alive component × prompt) where `max_ci > ci_threshold`, columns:
- layer (the block number)
- matrix (e.g. "attn.q_proj" or "mlp.c_fc")
- component (the component's index)
- prompt (the prompt's index)
- mean_kl_circuit (mean over positions of KL under case a — only alive components, no delta, X off)
- mean_kl_circuit_baseline (mean over positions of the case-a KL with no component ablated; same value across all rows for a given prompt; subtract from `mean_kl_circuit` to isolate X's marginal effect)
- mean_kl_all (mean over positions of KL under case b — every component + delta, X off)
- max_ci (max over positions of the component's lower-leaky causal importance on that sequence)
