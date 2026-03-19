"""Compare intruder detection scores across harvest subruns.

Guarantees the same set of component keys is scored in all subruns, making it
a fair comparison of activation pattern coherence.

Usage:
    python -m spd.harvest.scripts.run_intruder_comparison \
        s-55ea3f9b \
        --subruns h-20260227_010249,h-20260319_121635,h-20260318_223737 \
        --limit 200
"""

import asyncio
import random
import tempfile
from functools import reduce
from pathlib import Path

from dotenv import load_dotenv

from spd.adapters import adapter_from_id
from spd.autointerp.providers import OpenRouterLLMConfig, create_provider
from spd.harvest.config import IntruderEvalConfig
from spd.harvest.db import HarvestDB
from spd.harvest.intruder import run_intruder_scoring
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.log import logger


def main(
    decomposition_id: str,
    subruns: str,
    limit: int = 200,
    n_trials: int = 10,
    seed: int = 42,
) -> None:
    load_dotenv()

    subrun_ids = [s.strip() for s in subruns.split(",")]
    assert len(subrun_ids) >= 2, "Need at least 2 subruns to compare"

    eval_config = IntruderEvalConfig(
        llm=OpenRouterLLMConfig(model="deepseek/deepseek-v3.2", reasoning_effort="none"),
        n_trials=n_trials,
        max_concurrent=200,
        max_requests_per_minute=5000,
    )
    min_examples = eval_config.n_real + 1

    repos = {sid: HarvestRepo(decomposition_id, subrun_id=sid, readonly=True) for sid in subrun_ids}

    logger.info("Getting eligible component keys...")
    eligible_sets = {}
    for sid, repo in repos.items():
        keys = set(repo.get_eligible_component_keys(min_examples))
        eligible_sets[sid] = keys
        logger.info(f"  {sid}: {len(keys)} eligible")

    shared = sorted(reduce(lambda a, b: a & b, eligible_sets.values()))
    logger.info(f"Shared eligible: {len(shared)}")
    assert len(shared) >= limit, f"Only {len(shared)} shared eligible components, need {limit}"

    rng = random.Random(seed)
    target_keys = rng.sample(shared, limit)
    logger.info(f"Selected {limit} component keys for comparison")

    components_by_subrun: dict[str, list[ComponentData]] = {}
    for sid, repo in repos.items():
        logger.info(f"Loading {limit} components from {sid}...")
        components_by_subrun[sid] = list(repo.get_components_bulk(target_keys).values())
        assert len(components_by_subrun[sid]) == limit

    tokenizer_name = adapter_from_id(decomposition_id).tokenizer_name
    provider = create_provider(eval_config.llm)

    async def run_all() -> None:
        scores_by_subrun: dict[str, dict[str, float]] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for sid in subrun_ids:
                scratch_db = HarvestDB(Path(tmpdir) / f"scratch_{sid}.db")
                logger.info(f"Scoring {sid}...")
                results = await run_intruder_scoring(
                    components=components_by_subrun[sid],
                    provider=provider,
                    tokenizer_name=tokenizer_name,
                    score_db=scratch_db,
                    eval_config=eval_config,
                    limit=None,
                    cost_limit_usd=eval_config.cost_limit_usd,
                )
                scores_by_subrun[sid] = {r.component_key: r.score for r in results}

        scored_keys = sorted(
            reduce(lambda a, b: a & b, (set(s) for s in scores_by_subrun.values()))
        )

        print("\n" + "=" * 60)
        print("INTRUDER DETECTION COMPARISON")
        print("=" * 60)
        print(f"  Components scored: {len(scored_keys)}")
        print(f"  Trials per component: {n_trials}")
        print()

        for sid in subrun_ids:
            scores = scores_by_subrun[sid]
            mean = sum(scores[k] for k in scored_keys) / len(scored_keys)
            print(f"  {sid}  mean={mean:.3f}")

        if len(subrun_ids) == 2:
            a, b = subrun_ids
            sa, sb = scores_by_subrun[a], scores_by_subrun[b]
            n_a = sum(1 for k in scored_keys if sa[k] > sb[k])
            n_b = sum(1 for k in scored_keys if sb[k] > sa[k])
            n_t = sum(1 for k in scored_keys if sa[k] == sb[k])
            print(f"\n  {a} better: {n_a}")
            print(f"  {b} better: {n_b}")
            print(f"  Tied: {n_t}")

        print("=" * 60)

    asyncio.run(run_all())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
