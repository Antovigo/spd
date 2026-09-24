"""Write one pod launch config for the CI filter from the committed template seats.

    $VENV_PY make_config.py <stage> <out.yaml> [--init-id cf-xxxxxxxx]

stage: `smoke` (objective 1 on a 1..30 pool, 30 steps, evals every 10 steps, then the grid),
`obj1` (last-position KL), `obj2` (last-position integer KL, init from --init-id), `obj3`
(last-position cross-entropy of the TRUE answer, init from --init-id).
Hardware knobs come from the environment: MICRO (prompts per microbatch; unset = the whole
1024-prompt batch in one forward), EVAL_BATCH (default 1000), GRID_CHUNK (default 1000).
Every filter runs under the ceiling and prune constraints (see `CIFilterConfig`). The run is
referenced by id, so it resolves to `$DATA_ROOT/runs/<that id>`. TASK=addsub (default) or
TASK=mult selects the seat, i.e. which run and which prompt pool; RUN_ID and STEP override the
seat's run and checkpoint."""

import argparse
import os
from pathlib import Path

from param_decomp.ci_filter.config import CIFilterConfig

TEMPLATES = Path(__file__).resolve().parents[3] / "param_decomp" / "ci_filter" / "configs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["smoke", "obj1", "obj2", "obj3"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--init-id", default=None)
    args = ap.parse_args()

    # TASK picks WHICH decomposition is filtered: the seats differ only in `run.id` and
    # `pool.operations`, so they are separate template files rather than a flag here.
    task = os.environ.get("TASK", "addsub")
    assert task in ("addsub", "mult"), f"TASK must be addsub or mult, got {task!r}"
    template = {
        "obj2": "last_position_integer_kl.yaml",
        "obj3": "last_position_answer_ce.yaml",
    }.get(args.stage, "last_position_kl.yaml")
    if task == "mult":
        assert args.stage in ("smoke", "obj1"), (
            f"no mult seat for {args.stage!r}: only objective 1 (last-position KL) is authored "
            "for multiplication. The answer-supervised seats need their own template with "
            "`include_minus: false` — a product is never negative."
        )
        template = "last_position_kl_mult.yaml"
    raw = CIFilterConfig.from_file(TEMPLATES / template).model_dump(mode="json")
    micro = os.environ.get("MICRO")
    raw["microbatch_size"] = int(micro) if micro else None
    raw["eval_batch_size"] = int(os.environ.get("EVAL_BATCH", "1000"))
    raw["grid"]["chunk_prompts"] = int(os.environ.get("GRID_CHUNK", "1000"))
    match args.stage:
        case "smoke":
            raw |= {"steps": 30, "eval_every": 10, "wandb": None}
            raw["pool"] |= {"a_range": [1, 30], "b_range": [1, 30]}
        case "obj1":
            pass
        case "obj2" | "obj3":
            assert args.init_id, f"{args.stage} starts from another filter's CI fn: pass --init-id"
            raw["init"] = {"kind": "ci_filter", "id": args.init_id}
        case _:
            raise AssertionError(f"unknown stage {args.stage!r}")  # argparse choices guard this
    # RUN_ID / STEP come from run_pipeline.sh, which exports whatever it preflighted — so the
    # generated config names the SAME checkpoint the caller checked exists. The template's own
    # `run.id` / `step` are the standalone default for a hand-run make_config.py.
    if run_id := os.environ.get("RUN_ID"):
        raw["run"] = {"kind": "id", "id": run_id}
    if step := os.environ.get("STEP"):
        raw["step"] = int(step)
    config = CIFilterConfig.model_validate(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config.to_file(args.out)
    print(
        f"{args.stage}: {args.out} (microbatch {config.microbatch_size}, "
        f"eval {config.eval_batch_size})"
    )


if __name__ == "__main__":
    main()
