"""Probe what the interpreter model finds confusing about our prompt.

Generates real baked prompts for a sample of components, appends a
"what's confusing?" meta-question, sends to Gemini, and collects feedback.

Usage:
    source .venv/bin/activate
    python -m spd.autointerp.scripts.probe_prompt_confusion \
        --decomposition_id s-55ea3f9b \
        --harvest_subrun_id h-20260227_010249 \
        --n_components 10
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import ExampleRenderingConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_annotated_examples,
    build_separated_examples,
    describe_example_rendering,
    human_layer_desc,
)
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift, get_output_token_stats
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.log import logger
from spd.utils.markdown import Md

COMPONENT_SUBSET_PATH = Path(
    "/mnt/polished-lake/home/oli/spd/component_subsets/"
    "jose_coherent_200_seed0_common_ci0_0_ci0_1_ci0_5.txt"
)

CONFUSION_SUFFIX = """

---

**Meta-task (ignore the labeling task above):** Instead of labeling this component, please do two things:

**Part 1: Attempt the interpretation.** Give your best label and reasoning as if you were doing the task for real.

**Part 2: Reflect on the prompt.** Answer these questions about the prompt you just read:
1. What parts of the prompt were confusing or unclear?
2. Were there any terms or concepts that weren't well-explained?
3. Did any of the instructions seem contradictory?
4. What assumptions did you find yourself making that the prompt didn't explicitly address?
5. Was there anything about the activation examples that was hard to interpret or potentially misleading?
6. Did you understand the relationship between CI and inner activation?
7. Did you understand the sign convention? Were you tempted to interpret positive/negative activations as excitation/suppression?
8. Was the context about sequence boundaries and special tokens clear?

Be specific and honest. We're iterating on this prompt.
"""

CONFUSION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "reasoning": {"type": "string"},
        "confusing_or_unclear": {"type": "string"},
        "unexplained_terms": {"type": "string"},
        "contradictions": {"type": "string"},
        "implicit_assumptions": {"type": "string"},
        "example_issues": {"type": "string"},
        "ci_vs_inner_activation_understanding": {"type": "string"},
        "sign_convention_understanding": {"type": "string"},
        "sequence_boundary_understanding": {"type": "string"},
        "other_feedback": {"type": "string"},
    },
    "required": [
        "label",
        "reasoning",
        "confusing_or_unclear",
        "unexplained_terms",
        "contradictions",
        "implicit_assumptions",
        "example_issues",
        "ci_vs_inner_activation_understanding",
        "sign_convention_understanding",
        "sequence_boundary_understanding",
    ],
}


ExampleLayout = Literal["interleaved", "separated"]


def _build_draft_prompt(
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift,
    context_tokens_per_side: int,
    layout: ExampleLayout = "interleaved",
) -> str:
    """Build a prompt matching the draft v1 structure."""
    rendering = ExampleRenderingConfig(
        format="xml",
        highlight_delimiter="brackets",
        annotation_style="activation",
        xml_sanitize_raw=False,
        xml_sanitize_highlighted=False,
    )

    if layout == "interleaved":
        examples_interleaved = build_annotated_examples(
            component,
            app_tok,
            max_examples=30,
            rendering=rendering,
        )
        examples_raw = None
        examples_annotated = None
    else:
        examples_interleaved = None
        examples_raw, examples_annotated = build_separated_examples(
            component,
            app_tok,
            max_examples=30,
            rendering=rendering,
        )

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)

    dataset_desc = DATASET_DESCRIPTIONS.get(
        model_metadata.dataset_name, model_metadata.dataset_name
    )

    # Build output PMI section
    output_pmi_lines: list[str] = []
    if output_token_stats.top_pmi:
        for tok, pmi in output_token_stats.top_pmi[:10]:
            output_pmi_lines.append(f"- `{repr(tok)}`: {pmi:.2f}")

    # Determine special tokens
    eos_id = app_tok.hf_tokenizer.eos_token_id
    assert isinstance(eos_id, int)
    eos_display = app_tok.get_tok_display(eos_id)

    md = Md()

    # --- Context: SPD method ---
    md.h(3, "Context")
    md.p(
        "Below you will be presented with data about a component of a neural network as isolated by "
        'a brand new Mechanistic Interpretability technique "Stochastic Parameter Decomposition". '
        "You will be tasked with describing the component in terms of its activation patterns on "
        "various text examples from a pretraining dataset, and other supporting evidence."
    )
    md.p(
        "In Stochastic Parameter Decomposition, each weight matrix of a network is decomposed "
        'into C rank-1 parts, called "subcomponents", where C is usually greater than the rank '
        "of the weight matrix. These are parameterised as U \u2022 V, where V is the "
        '"read direction" (dimension `d_in` \u2014 what input patterns the component responds to) '
        'and U is the "write direction" (dimension `d_out` \u2014 what the component contributes '
        "to the output). They multiply to a rank-1 matrix of the shape of the original matrix, "
        "and can be thought to represent a one-dimensional slice of the computation the weight "
        "matrix does."
    )
    md.p("These subcomponents are learned in an unsupervised manner under 3 main losses:")
    md.bullets(
        [
            "Faithfulness: the C rank-1 subcomponent matrices must sum to the original weight "
            "matrix - this should be a direct factorisation of the original weight matrix.",
            "Minimality / Simplicity: For a given datapoint, as few subcomponents as possible "
            "should be necessary for the network. Alternatively - as many subcomponents as "
            "possible should be ablatable",
            "Reconstruction: The network should nonetheless reproduce the behaviour of the "
            "target network.",
        ]
    )
    md.p(
        'In order to facilitate this, we train a small auxiliary "Causal Importance Network" '
        "which produces a mask of Causal Importance values in (0, 1) for each datapoint/token. "
        "The minimality/simplicity loss above incentivizes this mask to be sparse."
    )
    md.p("At each token position, each component has 2 values:")
    md.numbered(
        [
            "**Causal Importance (CI):** The primary measure of component activity. CI is how "
            "causally important this component is for the model's output at this token \u2014 "
            'i.e. how much would ablating it change the output. CI is the basis for "firing": '
            "a component fires when its CI exceeds a threshold. **When interpreting a component, "
            "CI is the signal you should weight most heavily.**",
            "**Inner Activation (act):** The dot product of the input with the component's read "
            "direction (`x @ V`), scaled by the write direction norm. This measures how much the "
            "input aligns with what the component reads, regardless of whether that alignment "
            "matters for the output.",
        ]
    )
    md.p(
        "These two values are correlated but meaningfully different. A large inner activation "
        "with low CI means the input happens to align with the component's read direction, but "
        "the component's contribution isn't needed at this token. Treat low-CI positions as "
        "weak or incidental evidence."
    )
    md.p(
        'A component is said to "fire" when its causal importance exceeds a threshold. '
        "The data below uses a CI threshold of 0.0, so even very weakly important positions "
        "are included as firings."
    )
    md.p(
        "**Sign convention:** Negating both u\u1d62 and v\u1d62 produces the same rank-1 "
        'matrix, so the absolute sign of inner activations is arbitrary \u2014 "positive" does '
        'not mean excitation and "negative" does not mean suppression. What *is* meaningful is '
        "**relative sign within a component**: positive and negative activations produce opposite "
        "contributions to the output, so if you see a component's examples split into a cluster "
        "with positive act values and another with negative act values, those two clusters likely "
        "serve distinct roles (e.g. one token class activates positively, another negatively). "
        'Focus on whether the examples split by sign, not on what the sign itself "means".'
    )

    # --- Context: this specific component ---
    md.h(3, "Context")
    md.p(
        f"The component you will be labeling today comes from a decomposition of a "
        f"{model_metadata.n_blocks}-block transformer trained on {dataset_desc}. "
        f"Specifically, it is part of the {layer_desc}. It has a firing rate of "
        f"{component.firing_density * 100:.2f}% (fires {rate_str}). "
        f"The target model has ~42M parameters \u2014 keep its expected capability in mind, "
        f"it is not a smart model."
    )

    # --- Evidence ---
    md.h(2, "Evidence:")

    # Output token statistics
    md.h(3, "Output token statistics")
    md.p(
        "At each position where the component fires, we look at the model's next-token "
        "prediction distribution. The following tokens have the highest PMI (pointwise mutual "
        "information) between the component firing and the model assigning high probability to "
        "that token as its next-token prediction. A positive PMI value means this token is "
        "predicted more often than its base rate when the component fires. The value is in nats: "
        "0 = no association, 1 \u2248 3\u00d7 base rate, 2 \u2248 7\u00d7, 3 \u2248 20\u00d7."
    )
    if output_pmi_lines:
        md.p("**Top output tokens by PMI:**")
        md.p("\n".join(output_pmi_lines))

    # Activating examples
    md.h(3, "Activating examples")
    md.p(
        "The following **activating examples** are sampled uniformly at random from all positions "
        "in the dataset where the component fires (CI above threshold). For each sampled "
        f"activation location, we extract both a leading and trailing window of tokens centered "
        f"on the firing position, with up to {context_tokens_per_side} tokens of context on each "
        f"side. Windows are truncated at sequence boundaries \u2014 so a firing at the beginning "
        "of a training sequence will have little or no left context. This truncation is itself "
        "evidence (e.g. a component that consistently fires near the start of sequences). "
        "We include annotations for **all** firing positions in the window - not just the firing "
        "which was sampled to produce the window, however we don't include inner activations for "
        "all tokens - this would be too noisy - all tokens have at least epsilon inner activation "
        "on almost all components."
    )
    md.p(
        f"The training data consists of variable-length documents concatenated with "
        f"`{eos_display}` separator tokens between them, then sliced into fixed "
        f"{model_metadata.seq_len}-token sequences. This means `{eos_display}` tokens can appear "
        "anywhere within a sequence (not just at the start), and a single sequence may contain "
        f"parts of multiple documents. If you see `{eos_display}` in examples, it is a literal "
        "token the model processed, not a formatting artifact."
    )
    if layout == "interleaved":
        md.p("Each example is shown as an XML block with two views:")
        md.bullets(
            [
                "`<raw>`: the literal token text of the window",
                "`<annotated>`: the same window with firing tokens wrapped as "
                "`[[[token (ci:X, act:Y)]]]`",
            ]
        )
    else:
        md.p(
            "Examples are shown in two sections: first all raw text windows, then all "
            "annotated windows (with the same numbering). The annotated version wraps "
            "firing tokens as `[[[token (ci:X, act:Y)]]]`."
        )
    md.p("**Annotation legend:**")
    md.bullets(
        [
            "**ci** (causal importance): 0\u20131. How essential this component is at this "
            "position. This is the primary signal \u2014 high CI means the component genuinely "
            "matters here. Low CI (e.g. <0.05) means the component is barely involved; treat "
            "those positions as background context, not as evidence of what the component does.",
            "**act** (inner activation): alignment of the input with the component's read "
            "direction. See the sign convention note above \u2014 relative sign differences "
            "within a component are meaningful. These values are normalised so that typical "
            "magnitudes fall roughly in (-1, 1).",
        ]
    )
    if layout == "interleaved":
        assert examples_interleaved is not None
        md.p(describe_example_rendering(rendering))
        md.extend(examples_interleaved)
    else:
        assert examples_raw is not None and examples_annotated is not None
        md.h(4, "Raw text")
        md.extend(examples_raw)
        md.h(4, "Annotated")
        md.extend(examples_annotated)

    # Task
    md.h(2, "Task")
    md.p(
        "Based on all the above context and evidence, please give a label of 8 words or fewer "
        "for this component. The label should read like a short description of the job this "
        "component does in the network. Please also provide a short summary of your reasoning. "
        "Use all the evidence: activation examples, token statistics, and activation values. "
        "Be epistemically honest \u2014 express uncertainty when the evidence is weak, ambiguous, "
        "or mixed. Lowercase only."
    )

    return md.build()


def main(
    decomposition_id: str = "s-55ea3f9b",
    harvest_subrun_id: str = "h-20260227_010249",
    n_components: int = 10,
    seed: int = 42,
    out_dir: str = "scratch/prompt_probe_results",
    layout: ExampleLayout = "interleaved",
) -> None:
    load_dotenv()

    adapter = adapter_from_id(decomposition_id)
    harvest = HarvestRepo(decomposition_id, subrun_id=harvest_subrun_id, readonly=True)
    token_stats = harvest.get_token_stats()
    assert token_stats is not None

    harvest_config = harvest.get_config()
    raw_ctx = harvest_config["activation_context_tokens_per_side"]
    assert isinstance(raw_ctx, int)
    context_tokens_per_side = raw_ctx

    app_tok = AppTokenizer.from_pretrained(adapter.tokenizer_name)

    # Load component subset
    component_keys = COMPONENT_SUBSET_PATH.read_text().strip().splitlines()
    rng = random.Random(seed)
    sample = rng.sample(component_keys, min(n_components, len(component_keys)))

    logger.info(f"Probing {len(sample)} components: {sample}")

    from spd.autointerp.providers import GoogleAILLMConfig, create_provider

    provider = create_provider(GoogleAILLMConfig())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    async def _run() -> None:
        for key in sample:
            logger.info(f"Processing {key}")
            component = harvest.get_component(key)
            assert component is not None, f"Component {key} not found"

            output_stats = get_output_token_stats(
                token_stats, key, app_tok, top_k=20, pmi_min_count=2.0
            )
            assert output_stats is not None

            prompt = _build_draft_prompt(
                component=component,
                model_metadata=adapter.model_metadata,
                app_tok=app_tok,
                output_token_stats=output_stats,
                context_tokens_per_side=context_tokens_per_side,
                layout=layout,
            )

            # Save the baked prompt
            prompt_file = out_path / f"{key.replace(':', '_')}_prompt.md"
            prompt_file.write_text(prompt)

            # Send with confusion suffix
            confusion_prompt = prompt + CONFUSION_SUFFIX
            try:
                response = await provider.chat(
                    prompt=confusion_prompt,
                    max_tokens=8000,
                    response_schema=CONFUSION_SCHEMA,
                    timeout_ms=120_000,
                )
                feedback = json.loads(response.content)
            except Exception as e:
                logger.error(f"  Skipping {key}: {e}")
                continue

            result_file = out_path / f"{key.replace(':', '_')}_feedback.md"
            md_lines = [f"# Feedback for {key}\n"]
            for field, answer in feedback.items():
                md_lines.append(f"## {field.replace('_', ' ').title()}\n")
                md_lines.append(f"{answer}\n")
            md_lines.append(
                f"---\nInput tokens: {response.input_tokens}, "
                f"Output tokens: {response.output_tokens}\n"
            )
            result_file.write_text("\n".join(md_lines))

            json_file = out_path / f"{key.replace(':', '_')}_feedback.json"
            json_file.write_text(
                json.dumps(
                    {
                        "component_key": key,
                        "feedback": feedback,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    },
                    indent=2,
                )
            )

            logger.info(f"  -> {result_file}")

    asyncio.run(_run())

    # Generate HTML debug page
    html_path = _generate_debug_page(out_path)
    logger.info(f"Done. Results in {out_path}")
    logger.info(f"Debug page: {html_path}")


def _generate_debug_page(results_dir: Path) -> Path:
    """Generate an HTML page showing all probed components with their prompts and feedback."""
    import html as html_mod

    from spd.settings import SPD_OUT_DIR

    json_files = sorted(results_dir.glob("*_feedback.json"))
    _ = {f.stem.replace("_feedback", "_prompt"): f for f in json_files}

    cards: list[str] = []
    for jf in json_files:
        data = json.loads(jf.read_text())
        key = data["component_key"]
        fb = data["feedback"]

        prompt_file = results_dir / f"{key.replace(':', '_')}_prompt.md"
        prompt_text = prompt_file.read_text() if prompt_file.exists() else "(prompt not found)"

        label = fb.get("label", "—")
        reasoning = fb.get("reasoning", "—")

        feedback_fields = [
            ("Confusing / Unclear", fb.get("confusing_or_unclear", "")),
            ("Unexplained Terms", fb.get("unexplained_terms", "")),
            ("Contradictions", fb.get("contradictions", "")),
            ("Implicit Assumptions", fb.get("implicit_assumptions", "")),
            ("Example Issues", fb.get("example_issues", "")),
            ("CI vs Inner Activation", fb.get("ci_vs_inner_activation_understanding", "")),
            ("Sign Convention", fb.get("sign_convention_understanding", "")),
            ("Sequence Boundaries", fb.get("sequence_boundary_understanding", "")),
            ("Other", fb.get("other_feedback", "")),
        ]

        feedback_html = ""
        for name, value in feedback_fields:
            if value:
                feedback_html += (
                    f'<div class="fb-field">'
                    f"<strong>{html_mod.escape(name)}:</strong> "
                    f"{html_mod.escape(value)}</div>\n"
                )

        cards.append(f"""
        <details class="component-card">
          <summary>
            <span class="key">{html_mod.escape(key)}</span>
            <span class="label">{html_mod.escape(label)}</span>
            <span class="tokens">({data.get("input_tokens", "?")} in / {data.get("output_tokens", "?")} out)</span>
          </summary>
          <div class="card-body">
            <div class="section">
              <h3>Interpretation</h3>
              <div class="label-box"><strong>Label:</strong> {html_mod.escape(label)}</div>
              <div class="reasoning">{html_mod.escape(reasoning)}</div>
            </div>
            <div class="section">
              <h3>Prompt Feedback</h3>
              {feedback_html}
            </div>
            <details class="prompt-details">
              <summary>Full prompt ({len(prompt_text)} chars)</summary>
              <pre class="prompt">{html_mod.escape(prompt_text)}</pre>
            </details>
          </div>
        </details>
        """)

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Autointerp Prompt Probe — Debug</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em;
         background: #faf9f6; color: #2a2520; }}
  h1 {{ color: #5a3e28; border-bottom: 2px solid #d4c5b0; padding-bottom: 0.5em; }}
  .component-card {{ margin: 1em 0; border: 1px solid #d4c5b0; border-radius: 8px;
                     background: #fff; }}
  .component-card > summary {{ padding: 0.8em 1em; cursor: pointer; display: flex;
                               gap: 1em; align-items: center; font-size: 0.95em; }}
  .component-card[open] > summary {{ border-bottom: 1px solid #e8e0d4; }}
  .key {{ font-family: monospace; font-weight: bold; color: #8e3500; }}
  .label {{ color: #2a6e3f; font-style: italic; flex: 1; }}
  .tokens {{ color: #888; font-size: 0.85em; }}
  .card-body {{ padding: 1em; }}
  .section {{ margin-bottom: 1.5em; }}
  .section h3 {{ color: #5a3e28; margin: 0 0 0.5em; font-size: 1em; }}
  .label-box {{ background: #f0ebe3; padding: 0.5em 0.8em; border-radius: 4px; margin-bottom: 0.5em; }}
  .reasoning {{ color: #555; line-height: 1.5; }}
  .fb-field {{ margin: 0.4em 0; padding: 0.4em 0.6em; background: #fdf8f0; border-left: 3px solid #d4a55a;
               line-height: 1.4; }}
  .prompt-details {{ margin-top: 1em; }}
  .prompt-details > summary {{ cursor: pointer; color: #888; font-size: 0.85em; }}
  .prompt {{ background: #f5f0e8; padding: 1em; border-radius: 4px; font-size: 0.8em;
             max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
  <h1>Autointerp Prompt Probe — Debug</h1>
  <p>{len(json_files)} components probed. Click each to expand.</p>
  {"".join(cards)}
</body>
</html>"""

    out_dir = SPD_OUT_DIR / "www" / "autointerp" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "prompt_probe.html"
    html_path.write_text(html_doc)
    return html_path


if __name__ == "__main__":
    import fire

    fire.Fire(main)
