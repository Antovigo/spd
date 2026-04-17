Implement the following scripts in spd/scripts/validation. When dealing with paths, all scripts should support "~" for home.

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
--top-k: number of candidate pairs to keep (default 20)
--prompts: optional override for the prompts file
--output: overrides the path to write the data to

The script ranks (component_A, component_B) pairs in the same decomposed matrix that are good candidates for swapping their `U` (output) vectors. A good candidate pair has: (a) both components are important for their own task (ablating them pushes the model away from the correct next token), (b) both are in the same `(layer, matrix)` (so the U vectors have the same dimensions), (c) small side effects on nontarget data.

Task resolution. For each task `{"prompt": P, "target": T}` the script:
- Loads the prompts file referenced by `config.task_config.prompts_file` and finds the row whose text matches `P` exactly, giving `prompt_idx`.
- Tokenises `P` with `config.tokenizer_name` (no padding) and uses `last_pos = len(tokens) - 1` — the position whose logits predict the next token.
- Tokenises `T` and asserts it is a single token id (`target_token_id`).
Only LM tasks with a `prompts_file` are supported.

Importance score (per task, per component). The original model's probability at the task's target position is read once from the target orig-predictions file (and asserted to have `orig_pred == target_token_id`; otherwise the task isn't even solved by the unablated model). From the target KL file, rows with `prompt == prompt_idx` and `pos == last_pos` give the component-wise importance: the `kl` value on that row is how much ablating the component swings the next-token distribution. Larger KL means more important.

Side-effect score (per component). Read directly from the `quantile_kl` column of the nontarget summary TSV produced by `summarize_nontarget.py`. See that script for the exclusion rule and the definition of the quantile.

Pair ranking. For every `(layer, matrix)` that contains both a task-A candidate and a task-B candidate, enumerate all pairs `(a_comp, b_comp)` with `a_comp != b_comp`. The combined score for a pair is `min(a_targ_kl, b_targ_kl) / (1e-6 + mean(a_nontarg_kl, b_nontarg_kl))`, which rewards pairs that are jointly important for their own tasks while keeping the tail of their nontarget disruption low. Pairs are sorted by score and the top `--top-k` are written.

The quantile percent is recovered from the summary TSV's column name (`kl_q<pct>`) and is propagated into the output column names so the TSV is self-describing.

Output TSV (one row per candidate pair), columns:
- rank (1-indexed)
- layer
- matrix
- a_comp, b_comp (component indices)
- a_targ_kl, b_targ_kl (KL at each task's target row after ablating the component)
- a_nontarg_kl_q{pct}, b_nontarg_kl_q{pct} (nontarget KL quantile from the summary, `{pct}` is e.g. 99 for quantile 0.99)
- score (combined score)

Unless `--output` is specified, the TSV file is saved to the decomposed model's folder as `swap_candidates.tsv`.

**swap_test.py**
args:
- the path to a decomposed model
- the path to an `alive_components.tsv` (produced by `find_alive_components` on the target data) — used to read the mean inner activation of each component on its original inputs
--layer / --matrix: the decomposed matrix whose components will be swapped (both components must live in the same matrix, otherwise the U vectors have incompatible shapes)
--a-component / --b-component: the two component indices to swap
--target-a / --target-b: the expected next-token strings for tasks A and B, used only for per-position target-token probability tracking (each must tokenize to exactly one token under the run's tokenizer)
--n-nontarget-batches: number of nontarget batches to run (default 1)
--output-target / --output-nontarget: override the default output paths

The script builds a modified version of the decomposed model where the output (U) vectors of components A and B are swapped in the selected matrix. Because the two components usually fire with different inner-activation magnitudes (V^T x), a raw U swap would rescale the output by the wrong factor — so we normalise by the ratio of the mean inner activations from `alive_components.tsv`:

```
a_mean = alive_components[A].mean_activation
b_mean = alive_components[B].mean_activation

U[A] := (b_mean / a_mean) * U[B]_original
U[B] := (a_mean / b_mean) * U[A]_original
```

so that when A fires on its original inputs (activation ≈ `a_mean`), the output magnitude it produces now matches what B would produce on B's inputs (`b_mean * ||U[B]_original||`), and vice versa. Both `a_mean` and `b_mean` are asserted to be non-zero.

Model semantics. The swapped model is run through the same `ComponentModel` forward path that `effect_of_ablation` uses: all-ones component masks for every decomposed module, and `weight_deltas_and_masks` with delta-mask 1 everywhere using the **pre-swap** weight deltas (`target_weight - V @ U_original`). This guarantees that outside components A and B the swapped model is byte-identical to the original target model, while A and B's contributions are reconfigured by the swap. The comparison baseline ("orig") is the original target model itself, invoked via `spd_model(batch)` with no `mask_infos`. The swap is performed in place inside a context manager that clones the two affected U rows on entry and restores them on exit, so the model state is clean after the script finishes.

Datasets. Target data is the LM `prompts_file` (one batch containing every prompt, same convention as `effect_of_ablation`). Nontarget data is the `nontarget_task_config` dataset, iterated for `--n-nontarget-batches` batches using the decomposition's `batch_size`. Both passes run under the same swap configuration; for each batch, `orig_logits` is computed before the swap context and `swapped_logits` inside it, so the final model state is unchanged.

Output. Two TSVs, default `swap_test_target.tsv` and `swap_test_nontarget.tsv` in the decomposed model's folder. One row per `(prompt, pos)` with columns:

- prompt, pos, token
- orig_pred, orig_prob
- swapped_pred, swapped_prob
- p_target_a_orig, p_target_a_swapped (probability of the task-A target token under each model)
- p_target_b_orig, p_target_b_swapped
- kl (`KL(softmax(orig_logits) || softmax(swapped_logits))`)

Nontarget rows additionally carry a `batch_idx` column so positions from different batches can be distinguished.

Only LM tasks with a `prompts_file` are supported (the script needs prompt-based target data and LM token predictions).
