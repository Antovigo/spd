# Multi-language evaluation for CSS ablation

## Goal

After ablating CSS-specific components from a decomposed LM, we want to measure
the distributional effect on the model's output *per programming language*. The
hypothesis is that ablation should shift output distributions sharply on CSS
(and on CSS embedded in other languages, e.g. HTML `<style>` blocks) while
leaving other languages ~unchanged.

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
  `javascript`, `python`, `c`, `rust`, `markdown`, `yaml`, `json`, ...
- Loaded per-language via `load_dataset("bigcode/the-stack-smol",
  data_dir="data/<lang>")` — each row is one file, with fields including
  `content` (string) and `lang` (string).
- "Smol" means ~10k files per language, small enough to subsample cheaply.

Alternatives considered:

- `codeparrot/github-code` — bigger, same idea, but overkill here.
- `code_search_net` — only 6 languages, no CSS.
- `bigcode/the-stack-v2` — gated and much larger; unnecessary.

Languages to include (initial pick): `css`, `html`, `javascript`, `python`,
`c`, `rust`, `markdown`, `json`, `yaml`, plain English text (e.g. a wikitext
slice) as a non-code baseline.

## Labelling strategy

Every token gets a label string. File-level labels come from the dataset's
`lang` field. Embedded-CSS labels are recovered by regex-splitting `content`
before tokenising.

### File-level labels

For a file tagged `lang=X`, every token gets label `X`. Simplest case.

### Embedded CSS in HTML

No dataset labels this — all HTML files are tagged `html` regardless of
`<style>` content. Workaround:

1. For each HTML file, find all `<style ...>...</style>` blocks (case-insensitive,
   non-greedy, multiline). Regex: `<style\b[^>]*>(.*?)</style>`.
2. Split `content` into an ordered list of `(text_chunk, label)` pairs, where
   label is `"css"` inside `<style>` bodies and `"html"` outside.
3. Tokenise each chunk independently, concatenate the token id lists, and
   build a parallel list of per-token labels the same length as the ids.

This keeps byte-level label accuracy: a token that straddles the `<style>`
boundary is extremely rare in practice since tags end with `>` which tends to
terminate BPE merges, but if it happens we assign the label of the first byte
of the token and accept the small inaccuracy.

Same trick extends to:

- JS template literals containing CSS (e.g. `css\`\`\`` tagged templates) —
  harder because there's no standard delimiter; skip for now.
- Markdown code fences — `\`\`\`<lang>\n...\n\`\`\`` is well-defined; could
  produce labels like `markdown/python`, `markdown/css` etc. Useful for
  another pass, not v1.
- HTML inline `style="..."` attributes — short, noisy; skip.

### Suggested label taxonomy

Flat strings, no hierarchy:

```
css, html, html/css, javascript, python, c, rust, markdown, json, yaml, text
```

`html/css` is the "CSS embedded in HTML" label. Plain `html` is HTML outside
any `<style>` block. We expect ablation to hit `css` and `html/css`, spare
the rest.

## Output format

A single JSONL file, one line per tokenised file, suitable for later KL
analysis:

```json
{
  "source_lang": "html",
  "file_id": "the-stack-smol/html/abc123",
  "token_ids": [1, 523, 77, ...],
  "token_labels": ["html", "html", "html/css", "html/css", ...]
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
    label tokens (including CSS-in-HTML), and write JSONL."""
```

Key steps:

1. `load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}",
   split="train", streaming=True)` per language, take the first
   `files_per_lang` rows after shuffling with `seed`.
2. For `html`, run the `<style>` splitter; for everything else, pass through.
3. Tokenise chunks, build token_ids + token_labels lists.
4. Write JSONL to `output_path`.

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
4. Group by `token_labels[pos]`, emit a TSV of (label, kl) rows.
5. Plot KL distributions per label.

Expected result if the ablation is clean: `css` and `html/css` distributions
have a heavy right tail; other labels sit near zero.

## Open questions

- Is `the-stack-smol` CSS sample representative? It's GitHub-sourced and
  probably biased toward web frameworks. Acceptable for a first pass.
- Tokeniser mismatch: we need to use the decomposition's tokeniser, not a
  generic one. The loader takes `tokenizer_name` as an arg and asserts it
  matches `config.tokenizer_name`.
- Licence: the-stack-smol is permissively-licensed; fine for research use.
- `<style>` extraction misses CSS in `<link rel="stylesheet">` external
  files (those live in separate CSS files anyway, covered by `lang=css`).
