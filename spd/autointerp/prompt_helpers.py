"""Shared prompt-building helpers for autointerp and graph interpretation.

Pure functions for formatting component data into LLM prompt sections.
"""

import re

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import ExampleRenderingConfig
from spd.autointerp.schemas import DECOMPOSITION_DESCRIPTIONS, DecompositionMethod
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md

DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": (
        "SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. "
        "Simple vocabulary, common narrative elements."
    ),
    "danbraunai/pile-uncopyrighted-tok-shuffled": (
        "The Pile (uncopyrighted subset): diverse text from books, "
        "academic papers, code, web pages, and other sources."
    ),
    "danbraunai/pile-uncopyrighted-tok": (
        "The Pile (uncopyrighted subset): diverse text from books, "
        "academic papers, code, web pages, and other sources."
    ),
}

WEIGHT_NAMES: dict[str, str] = {
    "attn.q": "attention query projection",
    "attn.k": "attention key projection",
    "attn.v": "attention value projection",
    "attn.o": "attention output projection",
    "mlp.up": "MLP up-projection",
    "mlp.down": "MLP down-projection",
    "glu.up": "GLU up-projection",
    "glu.down": "GLU down-projection",
    "glu.gate": "GLU gate projection",
}

_ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def ordinal(n: int) -> str:
    if 1 <= n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    return f"{n}th"


def human_layer_desc(canonical: str, n_blocks: int) -> str:
    """'0.mlp.up' -> 'MLP up-projection in the 1st of 4 blocks'"""
    m = re.match(r"(\d+)\.(.*)", canonical)
    if not m:
        return canonical
    layer_idx = int(m.group(1))
    weight_key = m.group(2)
    weight_name = WEIGHT_NAMES.get(weight_key, weight_key)
    return f"{weight_name} in the {ordinal(layer_idx + 1)} of {n_blocks} blocks"


def layer_position_note(canonical: str, n_blocks: int) -> str:
    m = re.match(r"(\d+)\.", canonical)
    if not m:
        return ""
    layer_idx = int(m.group(1))
    if layer_idx == n_blocks - 1:
        return "This is in the final block, so its output directly influences token predictions."
    remaining = n_blocks - 1 - layer_idx
    return (
        f"This is {remaining} block{'s' if remaining > 1 else ''} from the output, "
        f"so its effect on token predictions is indirect — filtered through later layers."
    )


def density_note(firing_density: float) -> str:
    if firing_density > 0.15:
        return (
            "This is a high-density component (fires frequently). "
            "High-density components often act as broad biases rather than selective features."
        )
    if firing_density < 0.005:
        return "This is a very sparse component, likely highly specific."
    return ""


def build_data_presentation(
    seq_len: int,
    context_tokens_per_side: int,
    decomposition_method: DecompositionMethod,
) -> Md:
    window_size = 2 * context_tokens_per_side + 1
    md = Md()

    md.h(3, "Decomposition method")
    md.p(DECOMPOSITION_DESCRIPTIONS[decomposition_method])

    md.h(3, "Data")
    md.p(
        f"The model processes sequences of {seq_len} tokens. "
        f"Each activation example below shows a {window_size}-token window centered on the "
        f"firing token, with up to {context_tokens_per_side} tokens of context on each side. "
        f"Windows are truncated at sequence boundaries. "
        f"Examples are sampled uniformly at random from all firings across the dataset."
    )

    md.h(3, "Metric definitions")
    md.p("The token statistics below use these metrics:")
    md.bullets(
        [
            "**Precision**: P(component fires | token). Of all occurrences of token X in "
            "the dataset, what fraction had this component firing?",
            "**PMI** (pointwise mutual information, in nats): How much more likely is "
            "co-occurrence than chance? 0 = no association, 1 ≈ 3x, 2 ≈ 7x, 3 ≈ 20x.",
        ]
    )
    md.p(
        "**Input** metrics concern the token at the position where the component fires. "
        "**Output** metrics concern what the model predicts (at its final logits) at "
        "those positions — not the component's direct output."
    )
    return md


def build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if output_pmi:
        md.labeled_list(
            "**Output PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in output_pmi[:10]],
        )
    if output_stats.top_precision:
        md.labeled_list(
            "**Output precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in output_stats.top_precision[:10]],
        )
    return md


def build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if input_pmi:
        md.labeled_list(
            "**Input PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in input_pmi[:6]],
        )
    if input_stats.top_precision:
        md.labeled_list(
            "**Input precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in input_stats.top_precision[:8]],
        )
    return md


def describe_example_rendering(rendering: ExampleRenderingConfig) -> str:
    if rendering.format == "legacy_delimited":
        return (
            "Examples use grouped `<<delimiters>>` around contiguous active tokens. "
            "Tokens inside the delimiters are positions where the component is active."
        )

    delim = "`<<<token>>>`" if rendering.highlight_delimiter == "angle" else "`[[[token]]]`"
    if rendering.annotation_style == "activation":
        delim = (
            "`<<<token (ci:X, act:Y)>>>`"
            if rendering.highlight_delimiter == "angle"
            else "`[[[token]]] (ci:X, act:Y)`"
            if rendering.format == "single_line"
            else "`[[[token (ci:X, act:Y)]]]`"
        )

    if rendering.format == "xml":
        raw_desc = (
            "The `<raw>` block preserves literal whitespace and control characters."
            if not rendering.xml_sanitize_raw
            else "The `<raw>` block sanitizes control characters for readability, e.g. newline as `↵`."
        )
        highlighted_desc = (
            "The `<highlighted>` block also preserves literal token text."
            if not rendering.xml_sanitize_highlighted
            else "The `<highlighted>` block sanitizes control characters for readability while preserving token boundaries."
        )
        return (
            "Each example is an XML-style block with `<raw>` and `<highlighted>` sections. "
            f"`<highlighted>` repeats the same window with firing tokens wrapped as {delim}. "
            f"{raw_desc} {highlighted_desc}"
        )

    if rendering.annotation_style == "activation":
        return (
            "Each example is one annotated line. Firing tokens are wrapped as "
            f"{delim}. `ci` is causal importance and `act` is the component activation at that position. "
            "Control characters are rendered visibly, e.g. newline as `↵`."
        )

    return (
        "Each example is one line with firing tokens wrapped as "
        f"{delim}. Control characters are rendered visibly, e.g. newline as `↵`."
    )


def _build_legacy_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
    shift_firings: bool,
) -> Md:
    items: list[str] = []
    for ex in component.activation_examples[:max_examples]:
        if not any(ex.firings):
            continue
        spans = app_tok.get_spans(ex.token_ids)
        firings = [False] + ex.firings[:-1] if shift_firings else ex.firings
        tokens = list(zip(spans, firings, strict=True))
        items.append(delimit_tokens(tokens))
    md = Md()
    if items:
        md.numbered(items)
    return md


def build_fires_on_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> Md:
    return _build_legacy_examples(component, app_tok, max_examples, shift_firings=False)


def build_says_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> Md:
    return _build_legacy_examples(component, app_tok, max_examples, shift_firings=True)


def _fmt_ann(activations: dict[str, float]) -> str:
    """Format activation annotations for a single firing token.

    For SPD: (ci:0.82, act:-0.05)
    For other methods: just the activation value, e.g. (act:3.21)
    """
    parts: list[str] = []
    if "causal_importance" in activations:
        parts.append(f"ci:{activations['causal_importance']:.2g}")
    if "component_activation" in activations:
        parts.append(f"act:{activations['component_activation']:.2g}")
    if "activation" in activations:
        parts.append(f"act:{activations['activation']:.2g}")
    return f"({', '.join(parts)})"


def _delimited_token(
    display_span: str,
    token_id: int,
    ann: str | None,
    app_tok: AppTokenizer,
    delimiter_style: str,
    annotation_inside: bool,
    sanitize_fallback: bool,
) -> str:
    stripped = display_span.lstrip()
    whitespace = display_span[: len(display_span) - len(stripped)]
    if stripped:
        token_text = stripped
    else:
        token_text = (
            app_tok.get_tok_display(token_id) if sanitize_fallback else app_tok.decode([token_id])
        )

    open_delim, close_delim = ("[[[", "]]]") if delimiter_style == "brackets" else ("<<<", ">>>")
    if not ann:
        return f"{whitespace}{open_delim}{token_text}{close_delim}"
    if annotation_inside:
        inner = f"{token_text} {ann}".rstrip()
        return f"{whitespace}{open_delim}{inner}{close_delim}"
    return f"{whitespace}{open_delim}{token_text}{close_delim} {ann}".rstrip()


def _delimit_annotated(
    spans: list[str],
    token_ids: list[int],
    firings: list[bool],
    per_token_activations: list[dict[str, float]],
    app_tok: AppTokenizer,
    delimiter_style: str,
    annotation_style: str,
    annotation_inside: bool,
    sanitize_fallback: bool,
) -> str:
    """Join token strings, wrapping active tokens with configured delimiters."""
    parts: list[str] = []
    for token_id, span, active, acts in zip(
        token_ids, spans, firings, per_token_activations, strict=True
    ):
        if active:
            ann = _fmt_ann(acts) if annotation_style == "activation" else None
            parts.append(
                _delimited_token(
                    display_span=span,
                    token_id=token_id,
                    ann=ann,
                    app_tok=app_tok,
                    delimiter_style=delimiter_style,
                    annotation_inside=annotation_inside,
                    sanitize_fallback=sanitize_fallback,
                )
            )
        else:
            parts.append(span)
    return "".join(parts)


def _cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def _build_xml_example(raw_text: str, highlighted_text: str) -> str:
    return (
        "<example>\n"
        f"<raw><![CDATA[{_cdata(raw_text)}]]></raw>\n"
        f"<highlighted><![CDATA[{_cdata(highlighted_text)}]]></highlighted>\n"
        "</example>"
    )


def build_annotated_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
    rendering: ExampleRenderingConfig,
    shift_firings: bool = False,
) -> Md:
    """Build activation examples in the configured presentation format."""
    if rendering.format == "legacy_delimited":
        return _build_legacy_examples(component, app_tok, max_examples, shift_firings)

    assert rendering.highlight_delimiter != "legacy", (
        "legacy delimiter only supported with legacy_delimited format"
    )
    items: list[str] = []
    for ex in component.activation_examples[:max_examples]:
        if not any(ex.firings):
            continue
        firings = [False] + ex.firings[:-1] if shift_firings else ex.firings
        act_keys = list(ex.activations.keys())
        per_token_acts = [
            {k: ex.activations[k][i] for k in act_keys} for i in range(len(ex.token_ids))
        ]
        if rendering.format == "xml":
            raw_spans = (
                app_tok.get_spans(ex.token_ids)
                if rendering.xml_sanitize_raw
                else app_tok.get_raw_spans(ex.token_ids)
            )
            highlighted_spans = (
                app_tok.get_spans(ex.token_ids)
                if rendering.xml_sanitize_highlighted
                else app_tok.get_raw_spans(ex.token_ids)
            )
            items.append(
                _build_xml_example(
                    raw_text="".join(raw_spans),
                    highlighted_text=_delimit_annotated(
                        spans=highlighted_spans,
                        token_ids=ex.token_ids,
                        firings=firings,
                        per_token_activations=per_token_acts,
                        app_tok=app_tok,
                        delimiter_style=rendering.highlight_delimiter,
                        annotation_style=rendering.annotation_style,
                        annotation_inside=True,
                        sanitize_fallback=rendering.xml_sanitize_highlighted,
                    ),
                )
            )
        else:
            items.append(
                _delimit_annotated(
                    spans=app_tok.get_spans(ex.token_ids),
                    token_ids=ex.token_ids,
                    firings=firings,
                    per_token_activations=per_token_acts,
                    app_tok=app_tok,
                    delimiter_style=rendering.highlight_delimiter,
                    annotation_style=rendering.annotation_style,
                    annotation_inside=False,
                    sanitize_fallback=True,
                )
            )
    md = Md()
    if items:
        md.numbered(items)
    return md
