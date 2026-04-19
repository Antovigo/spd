# Multi-language ablation eval

`multilang_ablation.py` measures how ablating a fixed set of components affects
the model's next-token distribution across several programming languages (and
optionally plain English). Intended use: validate that a hypothesised
language-specific component set (e.g. CSS components) only perturbs its target
language.

## Inputs

1. **Decomposed model path** — same formats as the other validation scripts
   (WandB path or local checkpoint).
2. **Components** — second positional arg, either a file path or an inline
   comma-separated spec. Each spec is `<layer>:<matrix>:<component>`.

   File form (blank lines and `#` comments ignored):
   ```
   # CSS components to ablate
   3:attn.v_proj:17
   5:mlp.down_proj:42
   7:mlp.down_proj:8
   ```

   Inline forms:
   ```
   multilang_ablation <model> 3:attn.v_proj:17
   multilang_ablation <model> "3:attn.v_proj:17,5:mlp.down_proj:42"
   ```

   A file path is detected by `Path.is_file()`; otherwise the argument is
   parsed as a comma-separated spec list.

## What it does

1. Builds a ComponentModel with:
   - every listed component's mask set to 0
   - every other component's mask set to 1
   - the delta component fully enabled (mask = 1)
   So the ablated model equals the original target model *minus* the listed
   components' contributions, and nothing else.

   With `--invert`, the mask semantics flip: listed components stay on (mask =
   1), every other component is off (mask = 0), and the delta is disabled.
   This isolates a model that can only run the listed components (useful for
   e.g. probing whether a hypothesised subnetwork alone still produces the
   target-task predictions).

2. For each language, streams files from a HuggingFace dataset, tokenises
   them, and packs the token stream into batches of shape
   `(batch_size, seq_len)`.

3. For each batch, runs both the original and ablated models, and records two
   TSVs: per-position KL and predictions, and per-sequence mean KL + mean
   next-token cross-entropy of both models (see Output below).

## Languages and data sources

| Language      | Dataset                              | Notes                                                |
|---------------|--------------------------------------|------------------------------------------------------|
| `css`         | `Antovigo/pile-css-chunks`           | Pre-tokenised CSS training data of the decomposition |
| `css_bigcode` | `bigcode/the-stack-smol` `data/css`  | Independent CSS sample; `/* ... */` comments stripped|
| `html`        | `bigcode/the-stack-smol` `data/html` | `<style>...</style>` bodies stripped                 |
| `javascript`  | `bigcode/the-stack-smol` `data/javascript` |                                                |
| `python`      | `bigcode/the-stack-smol` `data/python` |                                                    |
| `c`           | `bigcode/the-stack-smol` `data/c`    |                                                      |
| `rust`        | `bigcode/the-stack-smol` `data/rust` |                                                      |
| `english`     | `wikitext` `wikitext-103-raw-v1`     | Plain prose baseline                                 |

Default language set is all of the above. Override with `--languages` (comma
separated, e.g. `--languages=css,html,python`).

HTML is cleaned with `re.sub(r"<style\b[^>]*>.*?</style>", "", text,
flags=DOTALL|IGNORECASE)` so the `html` sample measures HTML without embedded
CSS. Any other embedded-language cases (JS template literals, inline `style=`
attributes, etc.) are not handled.

## Sequence format

`css` reads the already-tokenised `pile-css-chunks` directly — each 512-token
chunk is flattened and re-packed into the eval's `(batch_size, seq_len)`
shape.

All other languages tokenise text on-the-fly in the same way
`extract_css_from_pile.py` built `pile-css-chunks`:

- `tokenizer(..., add_special_tokens=False)` — no BOS/EOS added anywhere.
  (The gpt-neox-20b tokenizer used by the CSS models also has
  `add_bos_token=add_eos_token=False`, so `tokenizer.encode(text)` behaves
  the same way.)
- Files are concatenated with **no separator** between them, then sliced into
  fixed-size `(batch_size, seq_len)` chunks.
- Any partial final chunk is dropped.

Known mismatches against training:

- **Non-CSS languages (HTML, JS, Python, C, Rust, English)** were not in the
  CSS target distribution at all — they correspond to the model's nontarget
  training data (`pile-uncopyrighted-tok-shuffled`), which may have its own
  inter-document separator conventions. The no-separator choice matches the
  CSS pipeline, which is the more important match for interpreting
  CSS-ablation KL. Any format bias for other languages applies uniformly, so
  cross-language KL comparisons remain meaningful.

## Output

Two TSVs:

**Per-position**: `<run_dir>/multilang_ablation.tsv` (override with
`--output`), one row per (language, prompt, position):

| Column             | Description                                           |
|--------------------|-------------------------------------------------------|
| `lang`             | Language label                                        |
| `prompt`           | Sequence index within the language (resets per lang)  |
| `pos`              | Position within the sequence                          |
| `token_str`        | Input token at this position, decoded                 |
| `orig_pred_str`    | Argmax prediction from the original model             |
| `orig_prob`        | Probability of `orig_pred_str` under the original     |
| `ablated_pred_str` | Argmax prediction from the ablated model              |
| `ablated_prob`     | Probability of `ablated_pred_str` under the ablated   |
| `kl`               | `KL(orig || ablated)` at this position                |

`(lang, prompt)` together identify one sequence; group by that pair to pick
out all tokens from a single contiguous input.

**Per-sequence**: `<run_dir>/per_sequence_loss.tsv` (override with
`--output-sequences`), one row per (language, prompt) — useful for matching
the ablated model's CE against the target-model training-loss curve to
estimate how much training compute the ablation has "undone":

| Column        | Description                                               |
|---------------|-----------------------------------------------------------|
| `lang`        | Language label                                            |
| `prompt`      | Sequence index within the language                        |
| `n_positions` | Number of next-token positions averaged over (seq_len - 1)|
| `mean_kl`     | Mean `KL(orig \|\| ablated)` across positions             |
| `orig_ce`     | Mean next-token CE loss of the original model             |
| `ablated_ce`  | Mean next-token CE loss of the ablated model              |

Both CE values and `mean_kl` are averaged over positions 0..seq_len-2 (next-
token positions; the final position has no label).

All `*_str` columns are TSV-escaped (backslash-escape of tab/newline/CR/`\`)
to keep the file splittable by tab everywhere.

## Parameters

- `--tokens-per-lang=N` (default 10000) — minimum tokens emitted per language.
  Actual count is rounded up to a whole number of batches, so the true total
  is `ceil(N / (batch_size * seq_len)) * batch_size * seq_len`.
- `--languages=a,b,c` — override the default language set.
- `--batch-size=N` — defaults to `config.batch_size`.
- `--seq-len=N` — defaults to the task config's `max_seq_len`.
- `--output=PATH` — override per-position TSV path.
- `--output-sequences=PATH` — override per-sequence TSV path.
- `--invert` — flip mask semantics: keep only the listed components on and
  disable every other component and the delta.

## Usage

```bash
uv run python -m spd.scripts.validation.multilang_ablation \
    "$MODEL_PATH" components_to_ablate.txt
```

With overrides:

```bash
uv run python -m spd.scripts.validation.multilang_ablation \
    "$MODEL_PATH" components_to_ablate.txt \
    --tokens-per-lang=50000 \
    --languages=css,html,python \
    --batch-size=16 --seq-len=1024
```

## Expected result

If the listed components are language-specific (e.g. CSS-specific):
- `css` KL distribution has a heavy right tail and `ablated_ce` shifts
  sharply above `orig_ce` (crossing an earlier point on the training curve).
- Other languages sit near zero on KL, with `ablated_ce ≈ orig_ce`.
- `html` sits near zero too, since `<style>` bodies were stripped.
