"""Write one pod launch config for the CI filter from the committed template seats.

    $VENV_PY make_config.py <stage> <out.yaml> [--init-id cf-xxxxxxxx]

stage: `smoke` (objective 1 on a 1..30 pool, 30 steps, evals every 10 steps, then the grid),
`obj1` (last-position KL), `obj2` (last-position integer KL, init from --init-id).
Hardware knobs come from the environment: MICRO (prompts per microbatch; unset = the whole
1024-prompt batch in one forward), EVAL_BATCH (default 1000), GRID_CHUNK (default 1000).
Constraint knobs (env), on every stage: `PRUNE_DEAD=1` sets `prune_dead: true` (components dead
on the pool at the start are removed), `CI_CEILING=1` sets `ci_ceiling: true` (CI can only
decrease from the starting point's). The run is referenced by id, so it resolves to `$DATA_ROOT/runs/p-ba5a0c05`."""

import argparse
import os
from pathlib import Path

from param_decomp.ci_filter.config import CIFilterConfig

TEMPLATES = Path(__file__).resolve().parents[3] / "param_decomp" / "ci_filter" / "configs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["smoke", "obj1", "obj2"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--init-id", default=None)
    args = ap.parse_args()

    template = "last_position_integer_kl.yaml" if args.stage == "obj2" else "last_position_kl.yaml"
    raw = CIFilterConfig.from_file(TEMPLATES / template).model_dump(mode="json")
    micro = os.environ.get("MICRO")
    raw["microbatch_size"] = int(micro) if micro else None
    raw["eval_batch_size"] = int(os.environ.get("EVAL_BATCH", "1000"))
    raw["grid"]["chunk_prompts"] = int(os.environ.get("GRID_CHUNK", "1000"))
    raw["ci_ceiling"] = os.environ.get("CI_CEILING", "0") == "1"
    raw["prune_dead"] = os.environ.get("PRUNE_DEAD", "0") == "1"
    match args.stage:
        case "smoke":
            raw |= {"steps": 30, "eval_every": 10, "wandb": None}
            raw["pool"] |= {"a_range": [1, 30], "b_range": [1, 30]}
        case "obj1":
            pass
        case "obj2":
            assert args.init_id, "obj2 starts from objective 1's CI fn: pass --init-id"
            raw["init"] = {"kind": "ci_filter", "id": args.init_id}
    config = CIFilterConfig.model_validate(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config.to_file(args.out)
    print(
        f"{args.stage}: {args.out} (microbatch {config.microbatch_size}, "
        f"eval {config.eval_batch_size}, prune_dead {config.prune_dead}, "
        f"ci_ceiling {config.ci_ceiling})"
    )


if __name__ == "__main__":
    main()
