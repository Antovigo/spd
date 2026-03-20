"""Generate a static HTML gallery of full baked prompt strategies on toy data."""

import html
from pathlib import Path

import yaml

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import (
    CompactSkepticalConfig,
    DualViewConfig,
    RichExamplesConfig,
)
from spd.autointerp.schemas import ModelMetadata
from spd.autointerp.scripts.prepare_jose_threshold_sweep import (
    SWEEP_ROOT,
    build_sweep_variants,
)
from spd.autointerp.strategies.dispatch import format_prompt
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI
from spd.settings import SPD_OUT_DIR

OUT_PATH = SPD_OUT_DIR / "www" / "autointerp" / "prompt_strategy_gallery.html"
TOKENIZER_NAME = "openai-community/gpt2"


def _toy_component() -> ComponentData:
    return ComponentData(
        component_key="toy.component:0",
        layer="h.1.mlp.up_proj",
        component_idx=0,
        mean_activations={"causal_importance": 0.21, "component_activation": 0.33},
        firing_density=0.013,
        activation_examples=[
            ActivationExample(
                token_ids=[64, 65, 66],
                firings=[False, True, False],
                activations={
                    "causal_importance": [0.02, 0.92, 0.01],
                    "component_activation": [0.05, 0.45, 0.03],
                },
            )
        ],
        input_token_pmi=ComponentTokenPMI(top=[(65, 2.1)], bottom=[(66, -0.8)]),
        output_token_pmi=ComponentTokenPMI(top=[(64, 1.7)], bottom=[(66, -0.5)]),
    )


TOY_MODEL_METADATA = ModelMetadata(
    n_blocks=4,
    model_class="toy.model.TinyTransformer",
    dataset_name="danbraunai/pile-uncopyrighted-tok-shuffled",
    layer_descriptions={"h.1.mlp.up_proj": "1.mlp.up"},
    seq_len=8,
    decomposition_method="spd",
)

TOY_INPUT_STATS = TokenPRLift(
    top_recall=[],
    top_precision=[("b", 0.43)],
    top_lift=[],
    top_pmi=[("b", 2.1)],
    bottom_pmi=None,
)

TOY_OUTPUT_STATS = TokenPRLift(
    top_recall=[],
    top_precision=[("x", 0.21)],
    top_lift=[],
    top_pmi=[("x", 1.7)],
    bottom_pmi=None,
)


def _bake_prompt(
    strategy: CompactSkepticalConfig | DualViewConfig | RichExamplesConfig,
    app_tok: AppTokenizer,
) -> str:
    component = _toy_component()
    if isinstance(strategy, RichExamplesConfig):
        return format_prompt(
            strategy=strategy,
            component=component,
            model_metadata=TOY_MODEL_METADATA,
            app_tok=app_tok,
            input_token_stats=None,
            output_token_stats=TOY_OUTPUT_STATS,
            context_tokens_per_side=1,
        )
    return format_prompt(
        strategy=strategy,
        component=component,
        model_metadata=TOY_MODEL_METADATA,
        app_tok=app_tok,
        input_token_stats=TOY_INPUT_STATS,
        output_token_stats=TOY_OUTPUT_STATS,
        context_tokens_per_side=1,
    )


def _build_card(
    slug: str,
    description: str,
    strategy: CompactSkepticalConfig | DualViewConfig | RichExamplesConfig,
    prompt: str,
) -> str:
    strategy_yaml = yaml.safe_dump(
        strategy.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=False,
    ).strip()
    return f"""
    <article class="card">
      <div class="card-head">
        <p class="eyebrow">Strategy</p>
        <h2>{html.escape(slug)}</h2>
        <p class="desc">{html.escape(description)}</p>
      </div>
      <pre class="prompt">{html.escape(prompt)}</pre>
      <details>
        <summary>Config</summary>
        <pre class="config">{html.escape(strategy_yaml)}</pre>
      </details>
    </article>
    """


def main(out_path: str | None = None) -> None:
    app_tok = AppTokenizer.from_pretrained(TOKENIZER_NAME)
    cards = []
    for variant in build_sweep_variants():
        prompt = _bake_prompt(variant.strategy, app_tok)
        cards.append(_build_card(variant.slug, variant.description, variant.strategy, prompt))

    target = Path(out_path) if out_path is not None else OUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Autointerp Prompt Strategy Gallery</title>
  <style>
    :root {{
      --bg: #efe7d8;
      --panel: #fffaf1;
      --ink: #1d1812;
      --muted: #6a5b4b;
      --line: #d9cbb4;
      --accent: #8f3600;
      --shadow: 0 18px 40px rgba(29, 24, 18, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(143, 54, 0, 0.10), transparent 24%),
        linear-gradient(180deg, #f8f1e6 0%, var(--bg) 100%);
    }}
    main {{ padding: 28px 0 40px; }}
    .hero {{
      width: min(1680px, calc(100vw - 32px));
      margin: 0 auto 16px;
      padding: 24px 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,248,238,0.96));
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 90ch;
    }}
    .rail-wrap {{
      overflow-x: auto;
      padding: 6px 16px 16px;
    }}
    .rail {{
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(680px, 760px);
      gap: 16px;
      align-items: start;
      width: max-content;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 18px;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      color: var(--accent);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .desc {{
      margin: 0 0 12px;
      color: var(--muted);
      min-height: 2.4em;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.77rem;
      line-height: 1.42;
    }}
    .prompt {{
      max-height: 78vh;
      overflow: auto;
      padding: 16px;
      background: #f7ecdb;
      border: 1px solid #e2d2b8;
    }}
    details {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .config {{
      margin-top: 10px;
      padding: 12px;
      background: #f7ecdb;
      border: 1px solid #e2d2b8;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Autointerp Prompt Strategy Gallery</h1>
      <p>
        Full baked prompts for the nine sweep variants, rendered on minimal toy data.
        The prompt text is real strategy output; only the baked evidence is compressed.
        Sweep root: <code>{html.escape(SWEEP_ROOT.as_posix())}</code>
      </p>
    </section>
    <div class="rail-wrap">
      <section class="rail">
        {"".join(cards)}
      </section>
    </div>
  </main>
</body>
</html>
"""
    target.write_text(html_doc)
    print(target)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
