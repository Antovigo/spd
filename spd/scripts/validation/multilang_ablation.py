"""Ablate a fixed set of components and measure per-language per-position KL divergence.

Reads a plain-text file of components (one `<layer>:<matrix>:<component>` per line), ablates
all of them simultaneously (their masks set to 0, every other component and the delta kept
fully enabled), and runs sequences from several programming languages through both the original
and the ablated model. For every (language, position) it writes one TSV row with the input
token, both models' top predictions and probabilities, and the per-position KL divergence.

With `--invert`, the listed components are the only ones kept on: every other component
*and* the delta are disabled. This isolates a model that can only run the listed components
(e.g. only the alive components from a CSS decomposition).

Code samples come from `bigcode/the-stack-smol` (one subset per language). English comes from
`wikitext-103-raw-v1`. HTML files have `<style>...</style>` blocks stripped so the 'html' sample
measures HTML without embedded CSS; CSS files have `/* ... */` comments stripped to match how
the CSS training data (`pile-css-chunks`) was built.

Usage:
    python -m spd.scripts.validation.multilang_ablation <model_path> <components_txt> \\
        [--tokens-per-lang=10000] [--languages=css,html,javascript,python,c,rust,english] \\
        [--batch-size=N] [--seq-len=N] [--output=PATH] [--invert]
"""

import csv
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import LMTaskConfig
from spd.log import logger
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import (
    build_module_lookup,
    escape_tsv_value,
    load_spd_run,
)
from spd.spd_types import ModelPath

_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

_PREPROCESSORS: dict[str, Callable[[str], str]] = {
    "html": lambda s: _STYLE_RE.sub("", s),
    "css": lambda s: _CSS_COMMENT_RE.sub("", s),
}

_DEFAULT_LANGUAGES: tuple[str, ...] = (
    "css",
    "html",
    "javascript",
    "python",
    "c",
    "rust",
    "english",
)

FIELDS = [
    "lang",
    "token_str",
    "orig_pred_str",
    "orig_prob",
    "ablated_pred_str",
    "ablated_prob",
    "kl",
]


def _parse_component_line(raw: str) -> tuple[int, str, int]:
    """Parse a '<layer>:<matrix>:<component>' line."""
    parts = raw.split(":")
    assert len(parts) == 3, f"Component line {raw!r} must be '<layer>:<matrix>:<component>'"
    return int(parts[0]), parts[1], int(parts[2])


def _load_component_list(
    path: Path, module_lookup: dict[tuple[int, str], str]
) -> list[tuple[str, int]]:
    """Return [(module_name, component_idx), ...] from a plain-text file; skip blank/# lines."""
    out: list[tuple[str, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        layer, matrix, component = _parse_component_line(line)
        key = (layer, matrix)
        assert key in module_lookup, (
            f"No decomposed module matches layer={layer}, matrix={matrix}. "
            f"Available: {sorted(module_lookup.keys())}"
        )
        out.append((module_lookup[key], component))
    assert out, f"No components found in {path}"
    return out


def _open_lang_dataset(lang: str) -> tuple[Iterator[dict[str, Any]], str, Callable[[str], str]]:
    """Return (row iterator, content field name, preprocess fn) for a language."""
    if lang == "english":
        ds: Any = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        return iter(ds), "text", _PREPROCESSORS.get(lang, lambda s: s)
    ds = load_dataset(
        "bigcode/the-stack-smol", data_dir=f"data/{lang}", split="train", streaming=True
    )
    return iter(ds), "content", _PREPROCESSORS.get(lang, lambda s: s)


def _iter_tokens(lang: str, tokenizer: PreTrainedTokenizer) -> Iterator[int]:
    """Stream token IDs from a language's dataset with no inter-file separator.

    Matches how `extract_css_from_pile.py` builds the CSS training chunks (raw
    `tokenizer.encode(text)` per doc, concatenated, then sliced). The gpt-neox-20b tokenizer
    used by the CSS models has `add_bos_token=add_eos_token=False`, so no special tokens are
    added anywhere in training — and we replicate that here.
    """
    row_iter, content_field, preprocess = _open_lang_dataset(lang)
    for row in row_iter:
        text = preprocess(row[content_field])
        if not text:
            continue
        encoded: Any = tokenizer(text, add_special_tokens=False)  # pyright: ignore[reportCallIssue]
        ids: list[int] = encoded["input_ids"]
        if ids:
            yield from ids


def _pack_batches(
    token_iter: Iterator[int],
    batch_size: int,
    seq_len: int,
    n_batches: int,
) -> Iterator[Tensor]:
    """Pack a token stream into up to `n_batches` tensors of shape (batch_size, seq_len)."""
    tokens_per_batch = batch_size * seq_len
    buffer: list[int] = []
    emitted = 0
    for tid in token_iter:
        buffer.append(tid)
        if len(buffer) >= tokens_per_batch:
            flat = buffer[:tokens_per_batch]
            buffer = buffer[tokens_per_batch:]
            yield torch.tensor(flat, dtype=torch.long).reshape(batch_size, seq_len)
            emitted += 1
            if emitted >= n_batches:
                return


def _make_decoder(tokenizer: PreTrainedTokenizer) -> Callable[[int], str]:
    """Memoised, TSV-safe single-token decoder."""
    cache: dict[int, str] = {}

    def decode(tid: int) -> str:
        s = cache.get(tid)
        if s is None:
            s = escape_tsv_value(tokenizer.decode([tid]))  # pyright: ignore[reportAttributeAccessIssue]
            cache[tid] = s
        return s

    return decode


def multilang_ablation(
    model_path: ModelPath,
    components_path: str,
    tokens_per_lang: int = 10000,
    languages: str | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    output: str | None = None,
    invert: bool = False,
) -> Path:
    """Ablate the listed components and record per-position KL across programming languages.

    If `invert` is True, keep only the listed components on and disable everything else
    (other components and the delta).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    decode = _make_decoder(tokenizer)

    assert isinstance(config.task_config, LMTaskConfig), "multilang_ablation only supports LM tasks"
    resolved_batch_size = batch_size if batch_size is not None else config.batch_size
    resolved_seq_len = seq_len if seq_len is not None else config.task_config.max_seq_len

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    comps = _load_component_list(Path(components_path).expanduser(), module_lookup)

    batch_shape = (resolved_batch_size, resolved_seq_len)
    fill_value = 0.0 if invert else 1.0
    set_value = 1.0 if invert else 0.0
    component_masks = {
        name: torch.full((*batch_shape, C), fill_value, device=device)
        for name, C in spd_model.module_to_c.items()
    }
    for module_name, component in comps:
        component_masks[module_name][..., component] = set_value
    if invert:
        deltas_and_masks = None
    else:
        weight_deltas = spd_model.calc_weight_deltas()
        delta_mask = torch.ones(batch_shape, device=device)
        deltas_and_masks = {name: (weight_deltas[name], delta_mask) for name in weight_deltas}
    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=deltas_and_masks)

    langs = tuple(s.strip() for s in languages.split(",")) if languages else _DEFAULT_LANGUAGES

    tokens_per_batch = resolved_batch_size * resolved_seq_len
    n_batches = max(1, -(-tokens_per_lang // tokens_per_batch))

    output_path = Path(output).expanduser() if output else run_dir / "multilang_ablation.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()

        for lang in langs:
            token_iter = _iter_tokens(lang, tokenizer)
            batch_iter = _pack_batches(token_iter, resolved_batch_size, resolved_seq_len, n_batches)

            rows_written = 0
            for batch in tqdm(batch_iter, desc=lang, total=n_batches):
                batch = batch.to(device)
                orig_logits = spd_model(batch)
                ablated_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(orig_logits, Tensor) and isinstance(ablated_logits, Tensor)

                orig_log_probs = F.log_softmax(orig_logits, dim=-1)
                orig_probs = orig_log_probs.exp()
                orig_prob_max, orig_pred = orig_probs.max(dim=-1)

                abl_log_probs = F.log_softmax(ablated_logits, dim=-1)
                abl_prob_max, abl_pred = abl_log_probs.exp().max(dim=-1)

                kl = (orig_probs * (orig_log_probs - abl_log_probs)).sum(dim=-1)

                batch_cpu = batch.cpu().tolist()
                orig_pred_cpu = orig_pred.cpu().tolist()
                orig_prob_cpu = orig_prob_max.cpu().tolist()
                abl_pred_cpu = abl_pred.cpu().tolist()
                abl_prob_cpu = abl_prob_max.cpu().tolist()
                kl_cpu = kl.cpu().tolist()

                bsz, slen = batch.shape
                for b in range(bsz):
                    for t in range(slen):
                        writer.writerow(
                            {
                                "lang": lang,
                                "token_str": decode(batch_cpu[b][t]),
                                "orig_pred_str": decode(orig_pred_cpu[b][t]),
                                "orig_prob": orig_prob_cpu[b][t],
                                "ablated_pred_str": decode(abl_pred_cpu[b][t]),
                                "ablated_prob": abl_prob_cpu[b][t],
                                "kl": kl_cpu[b][t],
                            }
                        )
                rows_written += bsz * slen

            if rows_written == 0:
                logger.warning(f"{lang}: dataset produced no batches (empty source?)")
            else:
                logger.info(f"{lang}: wrote {rows_written} rows")

    logger.info(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    fire.Fire(multilang_ablation)
