# Integer-only global mask of addsub-answers-09

Code: `param_decomp/param_decomp/integer_only/` (CLI `python -m param_decomp.integer_only.run`).
Config: `notes/integer_only/addsub-answers-09.yaml`. Pod driver: `notes/integer_only/pod/run.sh`.

## What this produces

Run p-512d73ab ("addsub-answers-09") is a targeted parameter decomposition of Llama-3.1-8B
(32 blocks, every linear site of every block). A *site* is one linear module, named like
`layers.26.mlp.down_proj`; a *component* is one rank-one piece of a site's decomposition,
identified by the site name plus its index within that site (an integer from 0 to the site's
component count minus 1). The decomposition has 124,928 components.

The pipeline finds one fixed set of components, the *integer-only* set, such that the model
with exactly those components on, at every prompt and position, and every other component and
the weight delta off, keeps the original model's next-token distribution among the integer
tokens at the last position of the arithmetic prompts. It then records what those components
do over the whole prompt pool and shows it in an (a, b)-grid applet.

## The prompts

The pool is `param_decomp/param_decomp/experiments/lm/prompts/answer_words_addition_subtraction_0-399.txt`,
the run's own target stream. Each line is `Answer: {a} plus {b} is ` or `Answer: {a} minus {b} is `
(trailing space kept), where `a` and `b` are integer operands from 0 to 399. Measured on the file
and the Llama-3.1 tokenizer:

- 240,200 lines (not 320,000): first the 160,000 additions, all `(a, b)` with `a, b` in 0..399,
  row-major (`a` outer, `b` inner); then the 80,200 subtractions, only `b <= a`, row-major
  (`a` outer from 0 to 399, `b` inner from 0 to `a`). Every true answer `r` (an integer, `a + b` or
  `a - b`) is therefore in 0..798.
- Every line tokenizes, with the BOS token, to the same 10 tokens. Positions `t` (integer, 0..9):
  0 `<|begin_of_text|>`, 1 `Answer`, 2 `:`, 3 a space, 4 `a`, 5 ` plus` or ` minus`, 6 a space,
  7 `b`, 8 ` is`, 9 a space. Every integer from 0 to 999 is one token.
- The answer that follows the prompt is the bare integer token: for every line, the text of the
  line followed by the digits of `r` tokenizes to the line's 10 tokens plus the single token of
  `r` (checked for all 240,200 lines in stage 1). This is the run's own tokenization of the
  target stream (`experiments/lm/targeted_data.py`: `encode(line, add_special_tokens=True)`).

The pipeline reads the prompts file through `IntegerOnlyConfig.prompts` and `templates`; each line
must match exactly one template, and the position labels are derived from the tokens (a position
whose token depends on `a` is labelled `a`, and so on; constant positions show their token).

## Notation

- `P`: the prompts whose true answer `r >= 0`. Here `P` is the whole pool (240,200 prompts).
- `V_int`: the 1000 tokens of the integers 0..999, in integer order.
- `p`: a model's next-token distribution at the last position (position 9) of a prompt, over
  the full vocabulary. `q(x) = p(x) / Σ_{x' ∈ V_int} p(x')` is `p` renormalized over `V_int`,
  indexed by the integer `x` (0..999).
- `q_Llama`: `q` of the ORIGINAL Llama-3.1-8B (dense weights), not of any decomposed model.
- `KL_int(m) = Σ_x q_Llama(x) (log q_Llama(x) − log q_m(x))` (nats) for a mask `m` on one
  prompt; `KL_full` is the same over the full vocabulary with `p` in place of `q`.
- `ce_int = −log q_m(r)`; accuracy = the fraction of prompts with `argmax_x q_m(x) = r`.
- CI: a component's causal importance at a (prompt, position), here always the run's own CI
  function's OUTPUT head (the output-reconstruction head of the dual-head CI fn), lower-leaky
  squash, evaluated on the ORIGINAL model's activations, with no ceiling and no pruning. A
  component is ON at `(prompt, t)` when its CI there exceeds 0.1. The -08 alive lists used the
  same head and squash (`filtered/alive-last-position.json`: "filter output lower-leaky CI on the
  original model > threshold", at threshold 0.01 and with a CI filter's fine-tuned CI fn).
- `Q`: the check set, 32,768 prompts of `P` drawn without replacement with a fixed seed.
- `c` indexes candidate components (canonical order: sites sorted by name, then component index);
  `ℓ_c` (real) is candidate `c`'s learned logit; `σ(z) = 1 / (1 + e^{−z})` is the logistic
  function; `T > 0` (real) the surrogate temperature; `λ >= 0` (real, nats per kept component)
  the sparsity weight; `m_c` in {0, 1} candidate `c`'s mask entry.

## Model under a global mask

Every site computes `y = ((x V) ⊙ m) U`: `x` is the site input (a `d_in`-vector per position),
`V` (`d_in` x `C`) the components' input vectors as columns, `U` (`C` x `d_out`) their output
vectors as rows, `m` (length `C`) the mask, `⊙` the entrywise product and `C` the site's number of
stored components. `x V_c` (real) is component `c`'s *inner activation*. The mask is the same for
every prompt and position; the weight delta (the target weight minus the components' sum) is
off. Embedding, norms, attention and unembedding are the original model's. Only candidates are
stored; every other component is permanently 0. Residual stream, norms, attention and softmax
run in fp32; the site and unembedding matmuls in bf16 with fp32 accumulation while learning
and scoring (as -08), in fp32 for the collection. Stage 2 checks this forward against the
decomposition's own masked forward and the dense target against the run's clean forward
(`status/problem.done`, `checks`).

## Method

Learner (the -08 `kl_sampled` path, `learn.py`): each candidate `c` has a logit `ℓ_c`, starting
at `ln 99 = 4.59512` (`σ(ℓ_c / T) = 0.99`). Each Adam step `k` (0-based, 0..2999) draws one mask
for the whole batch, `m_c ~ Bernoulli(σ(ℓ_c / T))`, and takes the loss

    L = mean over the batch of KL_int(m) + λ Σ_c σ(ℓ_c / T)

on a batch of 512 prompts of `P \ Q`, with the straight-through gradient
`∂L/∂ℓ_c = (∂ mean KL_int / ∂m_c + λ) · σ(ℓ_c / T) (1 − σ(ℓ_c / T)) / T` (the mask derivative
taken at the drawn mask). Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8), learning rate
`0.01 · (1 + cos(π k / 3000)) / 2`. The batch rows and the mask draw are pure functions of the
seed and `k`, so a resumed run continues the same trajectory. The result is the hard mask
`m_c = 1[ℓ_c > 0]`. Differences from -08: the reference of `KL_int` is the original Llama
(-08 trained against its alive-only model `S₀`), candidates come from the run's own CI at 0.1
(-08: a CI filter's alive set at 0.01), `P` includes `r = 0` (-08: `r >= 1`), and `Q` keeps its
random draw order, so the 4,096-prompt probe (the first 4,096 of `Q`) is a uniform random
subset (-08 sorted `Q`, so its probe was the 4,096 lowest pool rows).

Settings (all from p-ba8b0c08 `analysis/pruning/step_40000/addsub-08-prune/learn/kl_sampled/lam_5e-05/learn_config.json`):
3000 steps, batch 512, learning rate 0.01, cosine schedule, initial logit 4.59512, `T = 1`,
probe every 100 steps on 4,096 prompts of `Q`, bf16 matmuls, seed 0.

λ sweep: `λ` in {5e-5, 1e-4, 2e-4, 5e-4}, one mask each, one process (one GPU) per `λ`.

Rule: the chosen `λ` is the HIGHEST `λ` whose mask has mean `KL_int` over all of `Q` strictly
below 0.16479748860001564. That bar is the -08 integer-only mask's mean `KL_int` to Llama on
-08's check set: p-ba8b0c08
`analysis/pruning/step_40000/addsub-08-prune/learn/kl_table.json`, entry
`kl_sampled/lam_5e-05`, field `kl_int_llama` (that mask has 6,519 components). If no `λ`
qualifies, stage 4 fails and writes no `integer-only.json`.

## Stages

Each stage is `python -m param_decomp.integer_only.run <stage> --run_dir $RUN --data_root $DATA_ROOT
--config notes/integer_only/addsub-answers-09.yaml`, run from the repository root. `$RUN` is the
run directory; everything is written under `$RUN/integer_only/`. A stage writes
`status/<stage>.done` (JSON facts) when it finishes and `status/<stage>.failed` (the traceback) when
it fails; a finished stage is skipped when run again. The first stage pins the config as
`config.yaml`; later stages refuse other settings (batch sizes excepted).

1. `candidates`: the original model and the run's CI fn over all 240,200 prompts (batches of 256).
   Writes `pool.npz`, `max_ci.npz` (per site a `(10, C)` array, `C` the site's number of components: the max CI over all prompts at each
   position) and `candidates.json` (the components with max CI > 0.1 at some position).
2. `problem`: `P`, `Q` (`check_set.npy`), the original model's `log q` for every prompt
   (`reference_log_q.npy`, 240,200 x 1000 float32, 0.96 GB; the learners read it instead of running
   the dense model), and the factored-model checks.
3. `learn --lam λ`: one mask (`sweep/lam_<λ>/`). Resumes from `state.npz` (saved every 100 steps).
4. `pick`: scores every mask, the original model and the all-candidates model on all of `Q`
   (size, `KL_int`, `KL_full`, `ce`, `ce_int`, the integer mass `p_int`, the entropy of `q`,
   accuracy; all prompts, additions, subtractions), writes `sweep/lam_<λ>/final.json` and
   `sweep/summary.tsv`, applies the rule and writes `integer-only.json`.
5. `collect`: over all 240,200 prompts, for every integer-only component and every position `t`
   where it is ever ON: its inner activation in the integer-only model (fp32 forward, stored
   float16) and its ON bit. The ON bits come from a second pass of the CI fn (same batches as
   stage 1); the stored pairs are those ON on some prompt in stage 1 (`max_ci.npz`) or in this
   pass. Writes `datasets/` (format below); `status/collect.done` counts the pairs ON in only
   one of the two passes and lists any integer-only component with no stored pair.
6. `applet`: `ab_grids.html` and `ab_grids/`, from `datasets/` (no forward pass).

## Layout of `$RUN/integer_only/`

`N` = 240,200 is the number of pool rows (row `i` = line `i` of the prompts file, counting from
0); `C` is a site's number of components (stored or not); `T` = 10 positions.

    README.md               index (rewritten after every stage)
    config.yaml             the pinned settings
    status/                 <stage>.done / <stage>.failed (learn: learn_lam_<λ>)
    pool.npz                tokens (N, 10), op, a, b, r (N,), position_labels, ops
    max_ci.npz              {site: (10, C) float32}
    candidates.json         candidates, schema of integer-only.json
    check_set.npy           Q as pool rows, in draw order
    reference_log_q.npy     (N, 1000) float32 original log q
    sweep/lam_<λ>/          learn_config.json, log.jsonl, state.npz, final.json
    sweep/summary.tsv       one row per mask, plus `target` and `candidates`
    integer-only.json       the chosen mask
    datasets/               README.md, index.npz, inner_pos<t>.npy, ci_on_pos<t>.npy
    ab_grids.html, ab_grids/  the applet and its data
    logs/                   the pod driver's stage logs


`integer-only.json` and `candidates.json` use the schema of p-ba8b0c08 `filtered/integer-only.json`:
`rule` (text), `source` (the folder the list came from), `threshold` (the CI cut, 0.1),
`n_alive` (number of listed components), `n_components` (124,928), `components`
(`{site: [component indices]}` over all 224 sites, a list possibly empty).

`datasets/`: `index.npz` holds the pool columns and the column table `col_position`,
`col_site`, `col_component`, `col_v_norm` (`||V_c||`), one row per stored (position, component)
pair, sorted by position, then site, then component index. With `K_t` the number of stored pairs
at position `t`, `inner_pos<t>.npy` is `(N, K_t)` float16 and `ci_on_pos<t>.npy` is
`(ceil(N/8), K_t)` uint8 (`ceil(N/8)` = `N/8` rounded up), bit-packed along prompts (numpy
`packbits`), columns in the table's order. Pairs that are never ON
are not stored. -08 stored the same two quantities as `datasets/causal_importances/masks.parquet`
(one boolean list per (position, site, component) row) and `inner_pos<t>.npy` + `index.npz`
`pos<t>_cols`.

`ab_grids.html`: the page of `ci_filter/grids_app.html` with `ab_grids/` as its base URL. One slice
per operation and position `t`; each slice shows every integer-only component ever ON at `t`
(sorted by its ON fraction over that operation's prompts at `t`; the slider's lowest setting
shows every component ON at least once there) as a grid over `(a, b)`, `a` the row, `b` the column.
The view toggles between the binarized original CI (CI > 0.1) and the inner activation in the
integer-only model divided by `||V_c||` (as `alive.html`). Cells with no prompt (`b > a` for
subtraction) are grey. Grids wider than 100 cells ship as 4 x 4-cell-mean thumbnails, the full
400 x 400 grid loading on click.

## Reproducibility of the CI

The run's CI fn computes in bf16. On an L40, two processes evaluating it on the same prompts
with XLA's default GPU kernels gave different max CI values (largest difference 0.068;
457 of the ~30,000 (position, component) pairs ON in either process were ON in only one, on
the 18,205-prompt smoke pool). With `XLA_FLAGS=--xla_gpu_deterministic_ops=true` two
processes agreed exactly (0 pairs, largest difference 0), and the CI pass took 54 s instead of
61 s. The pod driver sets that flag for every stage; it made a learner step 0.90 s instead of
0.75 s on the L40. Stage 1 (a max reduction) and stage 5 (a column gather) are different
compiled programs; with the flag, they agreed on every pair in the smoke, and stage 5 stores the
union of both passes' pairs should they differ.

## Running on the pod (4 GPUs)

`notes/integer_only/pod/run.sh` runs stages 1 to 6 in order, stage 3 as four processes at once
(`λ` number `i` on GPU `i`), every other GPU stage on GPU 0, and stops at the first failed stage.
Stages already done are skipped, so rerunning it resumes.
