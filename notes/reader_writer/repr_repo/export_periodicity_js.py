"""Write the periodicity index (periodicity.json, seed_periodicity.py) as `periodicity.js` for the AB-grids applet.

The applet (param_decomp/ci_filter/grids_app.html, branch experiment/comp_pruning) loads it next to its slice files
and offers a "period" filter. Layout:

    window.PERIODICITY = {
      taus:      [2, 4, 5, 10, 20, 25, 50, 100]   the exact periods tau (dividing 100) the index tests
      blocks:    {"<p>": [block, ...]}            blocks whose gate / up readers are indexed at position p
      variables: {"<p>": [v, ...]}                variables tested at position p ("a", "b", "a+b")
      positions: {"<p>": {"<site>": {"<c>": {"<v>": [tau, ...]}}}}
      constancy: {"<p>": {"<site>": {"<c>": "constant" | "op_constant"}}}   labelled components only
      constancy_positions: ["1", "2", "3", "4"]  positions where every kept component was tested for constancy
      const_threshold: 0.95
    }

p is the applet's position index (1 = the token a, 3 = the token b, 4 = "=", addition prompts only), site a module
name such as "layers.0.mlp.gate_proj", c the component index, v a variable, and [tau, ...] the periods of that
component in v at that position (empty: tested, no period found). Components absent from `positions` were not
indexed. Sites: MLP gate / up / down and attention q / k / v / o. constancy: "constant" = one constant explains
>= const_threshold of the inner activation's uncentred energy over all prompts (both ops); "op_constant" = one constant
per op does (and one constant alone does not).

    python export_periodicity_js.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo import ROOT  # noqa: E402

OUT = Path(
    "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/addsub-08-last-pos-40000/ab_grids/periodicity.js"
)


def main() -> None:
    doc = json.loads((ROOT / "periodicity.json").read_text())
    positions: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {}
    constancy: dict[str, dict[str, dict[str, str]]] = {}
    for rd in doc["components"].values():
        site, c = rd["site"], str(rd["component"])
        for t, by_var in rd["positions"].items():
            positions.setdefault(t, {}).setdefault(site, {})[c] = {
                v: entry["periods"] for v, entry in by_var.items()
            }
        for t, cs in rd["constancy"].items():
            if cs["label"] is not None:
                constancy.setdefault(t, {}).setdefault(site, {})[c] = cs["label"]
    js = {
        "taus": [int(tau) for tau in doc["index"]["1"]["a"]],
        "variables": doc["variables"],
        "blocks": doc["blocks"],
        "source": str(ROOT / "periodicity.json"),
        "positions": positions,
        "constancy": constancy,
        "constancy_positions": sorted(
            {t for rd in doc["components"].values() for t in rd["constancy"]}
        ),
        "const_threshold": doc["const_threshold"],
    }
    OUT.write_text("window.PERIODICITY = " + json.dumps(js, separators=(",", ":")) + ";\n")
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
