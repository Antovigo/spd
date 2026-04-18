# Multi-language evaluation for CSS ablation

## Goal

After ablating CSS-specific components from a decomposed LM, we want to measure
the distributional effect on the model's output *per programming language*. The
hypothesis is that ablation should shift output distributions sharply on CSS
while leaving other languages ~unchanged.

Concretely, we want to produce per-position KL(orig || ablated) values, each
tagged with a language label, so the downstream analysis can plot KL
distributions per language.

We don't need a lot of data per language — a few hundred positions per
language is already enough to see distribution shape. Prioritise breadth of
languages and cleanliness of labels over size.

## Dataset choice

Primary candidate: **`bigcode/the-stack-smol`** on HuggingFace.

- Ships a deduplicated, permissively-licensed subset of The Stack
- Organised by language (`data/<lang>/` subdirectories): `css`, `html`,
  `javascript`, `python`, `c`, `rust`, `yaml`, `json`, ...
- Loaded per-language via `load_dataset("bigcode/the-stack-smol",
  data_dir="data/<lang>")` — each row is one file, with fields including
  `content` (string) and `lang` (string).
- "Smol" means ~10k files per language, small enough to subsample cheaply.

Alternatives considered:

- `codeparrot/github-code` — bigger, same idea, but overkill here.
- `code_search_net` — only 6 languages, no CSS.
- `bigcode/the-stack-v2` — gated and much larger; unnecessary.

Languages to include (initial pick): `css`, `html`, `javascript`, `python`,
`c`, `rust`, `json`, `yaml`, plain English text (e.g. a wikitext slice) as a
non-code baseline. Skip `markdown` — it commonly embeds many other languages
in fenced code blocks and would pollute the per-language KL distribution.

## Labelling strategy

One label per file (the dataset's `lang` field). No per-token labels.

### Cleaning HTML

To keep the `html` sample from being contaminated by embedded CSS, strip
`<style>...</style>` bodies from each HTML file *before* tokenising:

- Regex: `<style\b[^>]*>.*?</style>` (case-insensitive, dotall).
- Remove matches (including the tags themselves), keep the rest.

No splitting, no per-token labels — the file is tokenised as a single string
after cleanup and every token gets the label `html`.

Not handled (out of scope for v1):

- JS template literals containing CSS — no standard delimiter.
- HTML inline `style="..."` attributes — short, noisy.
- CSS pulled in via `<link rel="stylesheet">` — already covered by `lang=css`.

### Label taxonomy

Flat strings matching the dataset's `lang` field, plus `text` for the
non-code baseline:

```
css, html, javascript, python, c, rust, json, yaml, text
```

## Output format

A single JSONL file, one line per tokenised file:

```json
{
  "lang": "html",
  "file_id": "the-stack-smol/html/abc123",
  "token_ids": [1, 523, 77, ...]
}
```

Storing whole files (rather than pre-chunking to `max_seq_len`) keeps the
downstream window logic in one place — the ablation eval script already
handles chunking via the existing data loader.

## Loader sketch

New file: `spd/scripts/validation/multilang_dataset.py`.

```python
def load_multilang_eval(
    languages: list[str],
    files_per_lang: int,
    tokenizer_name: str,
    output_path: Path,
    seed: int = 0,
) -> None:
    """Sample `files_per_lang` files per language from the-stack-smol,
    strip <style> blocks from HTML, tokenise, and write JSONL."""
```

Key steps:

1. `load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}",
   split="train", streaming=True)` per language, take the first
   `files_per_lang` rows after shuffling with `seed`.
2. For `html`, apply the `<style>` stripper; for everything else, pass through.
3. Tokenise and write JSONL.

Kept as a one-shot script (not streamed through SPD's data loader) because
the dataset is small and the labelling is not worth generalising into the
training-data pipeline.

## Downstream use

A follow-up eval script (e.g. `css_ablation_eval.py`) would:

1. Load the JSONL.
2. For each file, run the original model and the ablated model (the latter
   with the chosen components masked to 0, delta=1 as in
   `effect_of_ablation.py`).
3. Compute per-position `KL(orig || ablated)`.
4. Group by file-level `lang`, emit a TSV of (lang, kl) rows.
5. Plot KL distributions per language.

Expected result if the ablation is clean: `css` has a heavy right tail;
other languages sit near zero.

## Open questions

- Is `the-stack-smol` CSS sample representative? It's GitHub-sourced and
  probably biased toward web frameworks. Acceptable for a first pass.
- Tokeniser mismatch: we need to use the decomposition's tokeniser, not a
  generic one. The loader takes `tokenizer_name` as an arg and asserts it
  matches `config.tokenizer_name`.
- Licence: the-stack-smol is permissively-licensed; fine for research use.
