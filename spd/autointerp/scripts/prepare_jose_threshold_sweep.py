"""Prepare a config-driven Jose autointerp sweep across harvest thresholds.

This writes:
- one stable subset file shared across harvest thresholds
- one AutointerpSlurmConfig YAML per strategy/rendering variant x threshold
- a JSON + Markdown manifest
- a shell script with submit commands

Usage:
    python -m spd.autointerp.scripts.prepare_jose_threshold_sweep
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from spd.autointerp.config import (
    AutointerpConfig,
    AutointerpEvalConfig,
    AutointerpSlurmConfig,
    CompactSkepticalConfig,
    DetectionEvalConfig,
    DualViewConfig,
    ExampleRenderingConfig,
    FuzzingEvalConfig,
    RichExamplesConfig,
)
from spd.autointerp.providers import GoogleAILLMConfig
from spd.autointerp.subsets import save_component_keys_file
from spd.harvest.repo import HarvestRepo

DECOMPOSITION_ID = "s-55ea3f9b"
THRESHOLD_SUBRUNS = {
    "ci0_0": "h-20260227_010249",
    "ci0_1": "h-20260319_121635",
    "ci0_5": "h-20260318_223737",
}

SWEEP_ROOT = Path("scratch/autointerp_sweeps/jose_threshold_matrix_v2_200")
CONFIG_DIR = SWEEP_ROOT / "configs"
MANIFEST_JSON = SWEEP_ROOT / "manifest.json"
MANIFEST_MD = SWEEP_ROOT / "README.md"
SUBMIT_SCRIPT = SWEEP_ROOT / "submit_all.sh"

SUBSET_SIZE = 200
SUBSET_SEED = 0
MIN_FIRING_RATE_PERCENT = 0.005
MAX_FIRING_RATE_PERCENT = 5.0
COMMON_SUBSET_PATH = Path(
    f"component_subsets/jose_coherent_{SUBSET_SIZE}_seed{SUBSET_SEED}_common_ci0_0_ci0_1_ci0_5.txt"
)

INTERPRET_LLM = GoogleAILLMConfig(model="gemini-3-flash-preview", thinking_level="minimal")
EVAL_LLM = GoogleAILLMConfig(model="gemini-3-flash-preview", thinking_level="minimal")

INTERPRET_RPM = 4000
INTERPRET_CONCURRENT = 384
EVAL_RPM = 1800
EVAL_CONCURRENT = 128
INTERPRET_TIME = "06:00:00"
EVAL_TIME = "06:00:00"


@dataclass(frozen=True)
class SweepVariant:
    slug: str
    description: str
    strategy: CompactSkepticalConfig | DualViewConfig | RichExamplesConfig


def _sample_common_subset_file() -> tuple[Path, dict[str, int]]:
    min_density = MIN_FIRING_RATE_PERCENT / 100.0
    max_density = MAX_FIRING_RATE_PERCENT / 100.0
    eligible_by_threshold: dict[str, set[str]] = {}
    for threshold_slug, harvest_subrun_id in THRESHOLD_SUBRUNS.items():
        repo = HarvestRepo(DECOMPOSITION_ID, harvest_subrun_id, readonly=True)
        summary = repo.get_summary()
        eligible_by_threshold[threshold_slug] = {
            key
            for key, comp in summary.items()
            if min_density <= comp.firing_density <= max_density
        }

    common = sorted(set.intersection(*eligible_by_threshold.values()))
    assert len(common) >= SUBSET_SIZE, (
        f"Only found {len(common)} common eligible components across thresholds; need {SUBSET_SIZE}"
    )

    rng = random.Random(SUBSET_SEED)
    sampled = sorted(rng.sample(common, SUBSET_SIZE))
    save_component_keys_file(COMMON_SUBSET_PATH, sampled)

    eligible_counts = {slug: len(keys) for slug, keys in eligible_by_threshold.items()}
    eligible_counts["common"] = len(common)
    return COMMON_SUBSET_PATH, eligible_counts


def build_sweep_variants() -> list[SweepVariant]:
    xml_angle = ExampleRenderingConfig(format="xml", highlight_delimiter="angle")
    xml_brackets = ExampleRenderingConfig(format="xml", highlight_delimiter="brackets")

    return [
        SweepVariant(
            slug="rich-singleline-brackets",
            description="rich_examples with one-line activation annotations",
            strategy=RichExamplesConfig(
                max_examples=30,
                include_dataset_description=True,
                label_max_words=8,
                output_pmi_min_count=2.0,
                example_rendering=ExampleRenderingConfig(
                    format="single_line",
                    highlight_delimiter="brackets",
                    annotation_style="activation",
                ),
            ),
        ),
        SweepVariant(
            slug="rich-xml-angle",
            description="rich_examples with raw + highlighted XML examples using angle delimiters",
            strategy=RichExamplesConfig(
                max_examples=30,
                include_dataset_description=True,
                label_max_words=8,
                output_pmi_min_count=2.0,
                example_rendering=ExampleRenderingConfig(
                    format="xml",
                    highlight_delimiter="angle",
                    annotation_style="activation",
                ),
            ),
        ),
        SweepVariant(
            slug="rich-xml-brackets",
            description="rich_examples with raw + highlighted XML examples using brackets",
            strategy=RichExamplesConfig(
                max_examples=30,
                include_dataset_description=True,
                label_max_words=8,
                output_pmi_min_count=2.0,
                example_rendering=ExampleRenderingConfig(
                    format="xml",
                    highlight_delimiter="brackets",
                    annotation_style="activation",
                ),
            ),
        ),
        SweepVariant(
            slug="compact-legacy",
            description="compact skeptical baseline with legacy delimited examples",
            strategy=CompactSkepticalConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=ExampleRenderingConfig(),
            ),
        ),
        SweepVariant(
            slug="compact-xml-angle",
            description="compact skeptical with XML examples and angle delimiters",
            strategy=CompactSkepticalConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=xml_angle,
            ),
        ),
        SweepVariant(
            slug="compact-xml-brackets",
            description="compact skeptical with XML examples and bracket delimiters",
            strategy=CompactSkepticalConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=xml_brackets,
            ),
        ),
        SweepVariant(
            slug="dual-legacy",
            description="dual_view baseline with legacy delimited examples",
            strategy=DualViewConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=ExampleRenderingConfig(),
            ),
        ),
        SweepVariant(
            slug="dual-xml-angle",
            description="dual_view with XML examples and angle delimiters",
            strategy=DualViewConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=xml_angle,
            ),
        ),
        SweepVariant(
            slug="dual-xml-brackets",
            description="dual_view with XML examples and bracket delimiters",
            strategy=DualViewConfig(
                max_examples=30,
                include_pmi=True,
                include_dataset_description=True,
                label_max_words=8,
                example_rendering=xml_brackets,
            ),
        ),
    ]


def _build_slurm_config(
    strategy: CompactSkepticalConfig | DualViewConfig | RichExamplesConfig,
    subset_path: Path,
) -> AutointerpSlurmConfig:
    return AutointerpSlurmConfig(
        config=AutointerpConfig(
            llm=INTERPRET_LLM,
            component_keys_path=subset_path.as_posix(),
            max_requests_per_minute=INTERPRET_RPM,
            max_concurrent=INTERPRET_CONCURRENT,
            template_strategy=strategy,
        ),
        time=INTERPRET_TIME,
        evals=AutointerpEvalConfig(
            llm=EVAL_LLM,
            component_keys_path=subset_path.as_posix(),
            seed=SUBSET_SEED,
            max_requests_per_minute=EVAL_RPM,
            max_concurrent=EVAL_CONCURRENT,
            detection_config=DetectionEvalConfig(
                n_activating=5,
                n_non_activating=5,
                n_trials=5,
            ),
            fuzzing_config=FuzzingEvalConfig(
                n_correct=8,
                n_incorrect=8,
                n_trials=5,
            ),
        ),
        evals_time=EVAL_TIME,
    )


def _manifest_row(
    threshold_slug: str,
    harvest_subrun_id: str,
    subset_path: Path,
    variant: SweepVariant,
    config_path: Path,
) -> dict[str, object]:
    return {
        "threshold_slug": threshold_slug,
        "harvest_subrun_id": harvest_subrun_id,
        "subset_path": subset_path.as_posix(),
        "variant_slug": variant.slug,
        "description": variant.description,
        "config_path": config_path.as_posix(),
        "strategy": variant.strategy.model_dump(mode="json"),
    }


def _write_manifest(rows: list[dict[str, object]], eligible_counts: dict[str, int]) -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Jose Threshold Matrix v2 (200)",
        "",
        f"- decomposition: `{DECOMPOSITION_ID}`",
        f"- subset size: `{SUBSET_SIZE}`",
        f"- firing-density band: `{MIN_FIRING_RATE_PERCENT}%` to `{MAX_FIRING_RATE_PERCENT}%`",
        f"- subset seed: `{SUBSET_SEED}`",
        f"- shared subset file: `{COMMON_SUBSET_PATH.as_posix()}`",
        f"- common eligible components across thresholds: `{eligible_counts['common']}`",
        f"- eligible by threshold: `ci0_0={eligible_counts['ci0_0']}`, `ci0_1={eligible_counts['ci0_1']}`, `ci0_5={eligible_counts['ci0_5']}`",
        f"- interpret model: `{INTERPRET_LLM.model}`",
        f"- eval model: `{EVAL_LLM.model}`",
        f"- interpret limits: `{INTERPRET_RPM}` req/min, `{INTERPRET_CONCURRENT}` concurrent",
        f"- eval limits: `{EVAL_RPM}` req/min, `{EVAL_CONCURRENT}` concurrent",
        "",
        "| Threshold | Harvest subrun | Variant | Config | Subset |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['threshold_slug']}` | "
            f"`{row['harvest_subrun_id']}` | "
            f"`{row['variant_slug']}` | "
            f"`{row['config_path']}` | "
            f"`{row['subset_path']}` |"
        )
    MANIFEST_MD.write_text("\n".join(lines) + "\n")


def _write_submit_script(rows: list[dict[str, object]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "cd /mnt/polished-lake/home/oli/spd",
        "",
    ]
    for row in rows:
        lines.append(
            "uv run spd-autointerp "
            f"{DECOMPOSITION_ID} "
            f"--config {row['config_path']} "
            f"--harvest_subrun_id {row['harvest_subrun_id']}"
        )
    SUBMIT_SCRIPT.write_text("\n".join(lines) + "\n")
    SUBMIT_SCRIPT.chmod(0o755)


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    variants = build_sweep_variants()
    rows: list[dict[str, object]] = []
    subset_path, eligible_counts = _sample_common_subset_file()

    for threshold_slug, harvest_subrun_id in THRESHOLD_SUBRUNS.items():
        for variant in variants:
            config = _build_slurm_config(variant.strategy, subset_path)
            config_path = CONFIG_DIR / f"{threshold_slug}__{variant.slug}.yaml"
            config.to_file(config_path)
            rows.append(
                _manifest_row(
                    threshold_slug=threshold_slug,
                    harvest_subrun_id=harvest_subrun_id,
                    subset_path=subset_path,
                    variant=variant,
                    config_path=config_path,
                )
            )

    _write_manifest(rows, eligible_counts)
    _write_submit_script(rows)

    print(f"prepared_configs={len(rows)}")
    print(f"manifest_json={MANIFEST_JSON}")
    print(f"manifest_md={MANIFEST_MD}")
    print(f"submit_script={SUBMIT_SCRIPT}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
