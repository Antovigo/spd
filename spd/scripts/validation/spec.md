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
- KL divergence is computed per (batch_example, position) as `KL(softmax(orig_logits) || softmax(ablated_logits))`. `orig_pred` is `argmax(orig_logits)`, `orig_prob` is `softmax(orig_logits)[orig_pred]`, and similarly for `ablated_pred` / `ablated_prob`. `token` is the input token id at that position.
- Only LM tasks are supported (tokens/positions/KL are LM-specific).
  
The data is saved as a tab-separated file, with columns:
- layer (the number of the block, where a block is an entire MLP or attention block)
- matrix (e.g. "attn.q_proj" or "mlp.c_fc")
- component (the component's index)
- prompt (the prompt's index)
- pos (the token position)
- token (the input token at that position)
- kl (the KL-divergence compared to the original model at that position)
- orig_pred (the most likely token according to the original model)
- orig_prob (the associated probability)
- ablated_pred (the most likely token according to the ablated model)
- ablated_prob (the associated probability)
  
Unless --output is specified, the TSV file is saved to the decomposed model's folder. The filename is "effect_of_ablation.tsv" for target data, and "effect_of_ablation_nontarget.tsv" for nontarget data.
