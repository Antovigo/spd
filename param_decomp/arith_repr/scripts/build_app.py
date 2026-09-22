"""Assemble the per-read-point analyses into the applet's data file.

    python -m param_decomp.arith_repr.scripts.build_app --analysis <dir> --out <app dir>

Writes `<app dir>/data.js` (`window.ARITH_REPR = {...}`) and copies `app.html` beside it as
`index.html`; open over `file://`."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

APP = Path(__file__).parent.parent / "app.html"


def rounded(x: Any, nd: int = 3) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    return x


LABELS = {
    "add": {"res": "a+b", "cross": "a-b"},
    "sub": {"res": "a-b", "cross": "a+b"},
    "both": {"res": "a+b|a-b", "cross": "a-b|a+b"},
}
"""Display names of the result quantities per operation set (`res` = a op b, `cross` = the
other operation's result, a negative control); on the pooled set the name spells out what the
quantity is on addition | subtraction prompts. The analysis keeps the generic names."""


def relabel(x: Any, table: dict[str, str]) -> Any:
    """Rename `res`/`cross` in every hypothesis name (`res:10` -> `a+b:10`) and quantity."""
    if isinstance(x, str):
        head, sep, tail = x.partition(":")
        return table.get(head, head) + sep + tail
    if isinstance(x, list):
        return [relabel(v, table) for v in x]
    if isinstance(x, dict):
        return {relabel(k, table): relabel(v, table) for k, v in x.items()}
    return x


def compact(record: dict[str, Any]) -> dict[str, Any]:
    """Drop the bulky per-direction arrays the applet does not draw; round the rest."""
    out: dict[str, Any] = {
        "key": record["key"],
        "k": record["k"],
        "n_reads": record["n_reads"],
        "positions": {},
    }
    for p, pos in record["positions"].items():
        ops: dict[str, Any] = {}
        for op_name, op in pos["ops"].items():
            fits = [
                {
                    "name": f["name"],
                    "quantity": f["quantity"],
                    "kind": f["kind"],
                    "period": f["period"],
                    "dim_hypothesis": f["dim_hypothesis"],
                    "marginal": f["marginal"],
                    "unique": f["unique"],
                    "generalising": f["generalising"],
                    "dim_S": f["dim_S"],
                    "linear_r2": f["linear_r2"],
                    "sv": [round(x, 4) for x in f["singular_values"][:8]],
                    "r2": [round(x, 3) for x in f["heldout_r2"][:8]],
                }
                for f in op["fits"]
            ]
            table = LABELS[op_name]
            ops[op_name] = {
                "joint_energy": op["joint_energy"],
                "explained_by_kept": op["explained_by_kept"],
                "null_threshold": op["null_threshold"],
                "y_spectrum": [round(x, 5) for x in op["y_spectrum"][:20]],
                "fits": [
                    {
                        **f,
                        "name": relabel(f["name"], table),
                        "quantity": relabel(f["quantity"], table),
                    }
                    for f in fits
                ],
                "separability": relabel(rounded(op["separability"]), table),
                "clusters": relabel(rounded(op["clusters"]), table),
            }
        out["positions"][p] = {
            "outside_basis_energy": pos["outside_basis_energy"],
            "ops": ops,
            # Compared across operations: `res` is a+b on add vs a-b on sub.
            "add_vs_sub_cosines": relabel(
                rounded(pos.get("add_vs_sub_cosines", {})),
                {"res": "result(a+b|a-b)", "cross": "cross(a-b|a+b)"},
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in sorted(args.analysis.glob("*.json")):
        records[path.stem] = compact(json.loads(path.read_text()))
    layers = sorted({int(k.split(".")[1]) for k in records})
    keys = [
        f"{s}.{layer}"
        for layer in layers
        for s in ("attn_in", "mlp_in")
        if f"{s}.{layer}" in records
    ]
    payload = {"keys": keys, "records": records}
    (args.out / "data.js").write_text("window.ARITH_REPR = " + json.dumps(payload) + ";")
    shutil.copy(APP, args.out / "index.html")
    print(f"{len(keys)} read points -> {args.out}")


if __name__ == "__main__":
    main()
