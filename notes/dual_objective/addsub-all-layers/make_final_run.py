#!/usr/bin/env python
"""Build the SOTA config for the 32-block addsub task from the balanced recipe.

    python make_final_run.py --lr-mult 0.5 --c-mult 2 --rho <measured> [-o <out.yaml>]

Default output is `../addsub-all-layers-4xh100-sota.yaml` — this task's sota config, the one
`PROFILES['4xh100']` names, so `launch_run.sh` / `make_trials.py` pick it up with no further
edit. It supersedes the pre-balance sota of the same name (git history holds that).

Edits the source TEXTUALLY, never through a yaml round-trip: `yaml.safe_dump` drops every
comment, and in these configs the comments ARE the rationale — each one records why a value
is what it is. Each edit is anchored on a unique line and the result is re-parsed and
compared field-by-field against the intent before it is written.

What it folds in, all measured on the 4x H100 seat:

* `--lr-mult`  BOTH learning rates (the recipe couples them 2:1). The 2026-09-10 sweep put
  the optimum at 0.5x: score 2.953, against 4.864 at 0.125x and 43.535 at 2x (mean of the
  last 5 `train/loss/total` at step 2000, seed 0 throughout, bracket closed on both sides).
  CAVEAT recorded in the header: that sweep ran at C1x under `batch` normalization with the
  DILUTED hidden coefficients, so 0.5x is the best available estimate for this objective,
  not a measured optimum under it.
* `--c-mult`   every entry of `sites.cs`. C is tiled across layers by schema — the chunkwise
  CI fn scans over chunks and asserts they are homogeneous in the per-slot C tuple — so C
  scales for all 32 blocks or none. Measured at C2x: 41.6 GB/rank of 80, 2.49 s/step.
* `--rho`      DIVIDES the three hidden recon coefficients. `per_position` restores the L18
  per-point WEIGHT (x32/14) but replaces the per-point QUANTITY with a different functional,
  whose value is larger wherever high-energy positions reconstruct well and low-energy ones
  badly. Dividing by the measured per_position/batch ratio keeps the L18 recon/sparsity
  exchange rate while still moving pressure off the BOS position. Measure it with
  `arms-4xh100/normcal.yaml` under `PD_LOG_ALT_HIDDEN_NORM=1`. `--rho 1` opts out.
"""

import argparse
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "addsub-all-layers-4xh100-bal-perpos.yaml"
DEFAULT_OUT = HERE.parent / "addsub-all-layers-4xh100-sota.yaml"
RUN_NAME = "addsub-all-layers-4xh100-sota-01"
CS_KEYS = ("q", "k", "v", "o", "gate", "up", "down")


def _yaml_float(x: float) -> str:
    """A float literal YAML 1.1 parses as a FLOAT. `8e-05` is a STRING to yaml (1.1 needs a
    dot and a signed exponent), which would ride into the config as a str and fail far from
    here; `%g` also truncates 4.5714286 to 4.57143. Keep 10 significant digits and force a
    dotted mantissa on anything exponential."""
    out = f"{x:.10g}"
    if "e" in out:
        mantissa, _, exponent = out.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        sign = "-" if exponent.startswith("-") else "+"
        out = f"{mantissa}e{sign}{exponent.lstrip('+-').zfill(2)}"
    return out


def _sub_once(text: str, pattern: str, repl: str, what: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    assert n == 1, f"{what}: expected exactly 1 match for {pattern!r}, got {n}"
    return out


def build(text: str, lr_mult: float, c_mult: int, rho: float) -> str:
    src = yaml.safe_load(text)

    for opt in ("components_optimizer", "ci_fn_optimizer"):
        old = src["pd"][opt]["lr_schedule"]["max_val"]
        new = float(f"{old * lr_mult:.10g}")
        # Anchor on the literal token, not the parsed value: the file writes `3.2e-04`
        # while yaml parses it to 0.00032, so a value-based pattern never matches.
        text = _sub_once(
            text,
            rf"^(  {opt}:\n    lr_schedule:\n      max_val: )\S+$",
            rf"\g<1>{_yaml_float(new)}",
            f"{opt} lr",
        )

    for key in CS_KEYS:
        old = src["decomposition"]["sites"]["cs"][key]
        text = _sub_once(text, rf"^(      {key}: ){old}$", rf"\g<1>{old * c_mult}", f"cs.{key}")

    seen = 0
    for block, path in (("pd", ("pd", "hidden")), ("nontarget", ("nontarget", "hidden"))):
        node = src
        for k in path:
            node = node[k]
        for term in node["recon"]:
            old = term["coeff"]
            new = float(f"{old / rho:.10g}")
            text = _sub_once(
                text,
                rf"^(        coeff: ){re.escape(str(old))}$",
                rf"\g<1>{_yaml_float(new)}",
                f"{block} coeff",
            )
            seen += 1
    assert seen == 3, f"expected 3 hidden recon coefficients, edited {seen}"

    text = _sub_once(text, r"^run_name: .*$", f"run_name: {RUN_NAME}", "run_name")
    return text


def verify(text: str, src: dict, lr_mult: float, c_mult: int, rho: float) -> dict:
    """Re-parse the edited text and check every field against the intent. The edits are
    textual, so this is what stands between a bad regex and a 30 h run at the wrong value."""
    got = yaml.safe_load(text)
    for opt in ("components_optimizer", "ci_fn_optimizer"):
        want = float(f"{src['pd'][opt]['lr_schedule']['max_val'] * lr_mult:.10g}")
        assert got["pd"][opt]["lr_schedule"]["max_val"] == want, (opt, got, want)
    for key in CS_KEYS:
        want = src["decomposition"]["sites"]["cs"][key] * c_mult
        assert got["decomposition"]["sites"]["cs"][key] == want, (key, want)
    for path in (("pd", "hidden"), ("nontarget", "hidden")):
        a, b = src, got
        for k in path:
            a, b = a[k], b[k]
        for old_term, new_term in zip(a["recon"], b["recon"], strict=True):
            want = float(f"{old_term['coeff'] / rho:.10g}")
            assert new_term["coeff"] == want, (path, new_term, want)
    assert got["run_name"] == RUN_NAME

    # Nothing else may move: every other leaf must be identical to the source.
    def leaves(node, prefix=()):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from leaves(v, (*prefix, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from leaves(v, (*prefix, i))
        else:
            yield prefix, node

    touched = {("run_name",)}
    diffs = {k: (a, b) for (k, a), (_, b) in zip(leaves(src), leaves(got), strict=True) if a != b}
    unexpected = {
        k: v
        for k, v in diffs.items()
        if k not in touched and "lr_schedule" not in k and "cs" not in k and k[-1] != "coeff"
    }
    assert not unexpected, f"edits leaked into unrelated fields: {unexpected}"
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-mult", type=float, required=True)
    ap.add_argument("--c-mult", type=int, default=1)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    text = SOURCE.read_text()
    src = yaml.safe_load(text)
    edited = build(text, a.lr_mult, a.c_mult, a.rho)
    got = verify(edited, src, a.lr_mult, a.c_mult, a.rho)

    banner = (
        f"# SOTA for the 32-block addsub task. GENERATED by make_final_run.py from\n"
        f"# {SOURCE.name} (the balanced hidden recipe) with every measured result folded in:\n"
        f"#   LR x{a.lr_mult:g} — the 2026-09-10 sweep optimum (2.953 at 0.5x, vs 4.864 at 0.125x\n"
        f"#     and 43.535 at 2x; mean of the last 5 train/loss/total at step 2000, seed 0).\n"
        f"#     That sweep ran at C1x under `batch` normalization with the DILUTED hidden\n"
        f"#     coefficients, so this is the best estimate for this objective, not a measured\n"
        f"#     optimum under it.\n"
        f"#   C x{a.c_mult:g} — measured 41.6 GB/rank of 80 and 2.49 s/step at C2x.\n"
        f"#   hidden recon coefficients / rho={a.rho:g} — the measured per_position/batch scale\n"
        f"#     ratio, so the balanced recipe keeps the L18 recon/sparsity exchange rate rather\n"
        f"#     than an unmeasured multiple of it.\n"
        f"# Comments below are the source's, preserved: the edits are textual, never a yaml\n"
        f"# round-trip. Regenerate rather than hand-editing.\n"
    )
    Path(a.out).write_text(banner + edited)
    cs = got["decomposition"]["sites"]["cs"]
    lrs = tuple(
        got["pd"][o]["lr_schedule"]["max_val"] for o in ("components_optimizer", "ci_fn_optimizer")
    )
    hid = [(t.get("name", t["type"]), t["coeff"]) for t in got["pd"]["hidden"]["recon"]]
    nth = [(t.get("name", t["type"]), t["coeff"]) for t in got["nontarget"]["hidden"]["recon"]]
    print(
        f"wrote {a.out}\n  run_name {got['run_name']}\n  LR {lrs}\n  cs {cs}\n  hidden {hid}\n  nt-hidden {nth}"
    )


if __name__ == "__main__":
    main()
