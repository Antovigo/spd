"""Sample a stable subset of component keys from a harvest subrun.

Usage:
    python -m spd.autointerp.scripts.sample_component_keys \
        --decomposition_id s-55ea3f9b \
        --harvest_subrun_id h-20260318_223737 \
        --output_path jose_coherent_100_component_keys.txt \
        --limit 100 \
        --min_firing_rate_percent 0.005 \
        --max_firing_rate_percent 5.0 \
        --seed 0
"""

import random
from pathlib import Path

from spd.autointerp.subsets import save_component_keys_file
from spd.harvest.repo import HarvestRepo


def main(
    decomposition_id: str,
    harvest_subrun_id: str,
    output_path: str,
    limit: int = 100,
    min_firing_rate_percent: float = 0.005,
    max_firing_rate_percent: float = 5.0,
    seed: int = 0,
) -> None:
    repo = HarvestRepo(decomposition_id, harvest_subrun_id, readonly=True)
    summary = repo.get_summary()

    min_density = min_firing_rate_percent / 100.0
    max_density = max_firing_rate_percent / 100.0

    eligible = sorted(
        [
            key
            for key, comp in summary.items()
            if min_density <= comp.firing_density <= max_density
        ]
    )

    assert len(eligible) >= limit, (
        f"Only found {len(eligible)} eligible components in "
        f"[{min_firing_rate_percent}%, {max_firing_rate_percent}%], need {limit}"
    )

    rng = random.Random(seed)
    sampled = rng.sample(eligible, limit)
    sampled.sort()

    out_path = Path(output_path)
    save_component_keys_file(out_path, sampled)

    print(f"eligible_count={len(eligible)}")
    print(f"sampled_count={len(sampled)}")
    print(f"output_path={out_path}")
    print("first_10_keys=")
    for key in sampled[:10]:
        print(key)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
