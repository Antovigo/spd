"""Quantity lists of the agentic run in the format qdirs.py and qapplet.py read (not an agent tool).

Writes qagent/qloop_t<t>_agentic_app.json per position: per site the final list, each quantity named by
its formula (the agent's own name in "label"); "drop_one_loss" = the held-out R^2 of the reads lost
when the quantity is removed (inner folds, training rows, qa.Scorer), "chance" = 0, "stability" = 1.
Candidates come from qagent/agentic_cands.py.

    python applet_prep.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import qa
from qdata import DATA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qagent"))
from agentic_cands import RUN1  # noqa: E402


def main() -> None:
    for t in (1, 2, 3, 4):
        sites = []
        f = np.load(DATA / f"site_data_t{t}.npz")
        labels = sorted({k.split("/")[0] for k in f.files if "/" in k},
                        key=lambda x: (int(x[1:].split(".")[0]), x.endswith("mlp")))  # fmt: skip
        for lab in labels:
            spec = RUN1 / f"t{t}" / lab / "final.py"
            if not spec.exists():
                sites.append({"site": lab, "constant": True, "accepted": []})
                continue
            Q, seen = [], set()
            for nm, src in qa.load_spec(spec):
                if src not in seen:
                    Q.append((nm, src))
                    seen.add(src)
            sc = qa.Scorer(qa.Site(t, lab), Q)
            allq = list(range(len(Q)))
            r2 = sc.heldout_r2(allq)
            acc = [{"name": src, "label": nm, "drop_one_loss": round(r2 - sc.heldout_r2([j for j in allq if j != i]), 6),
                    "chance": 0.0, "stability": 1.0} for i, (nm, src) in enumerate(Q)]  # fmt: skip
            sites.append({"site": lab, "accepted": acc, "heldout_r2": round(r2, 4)})
            print(t, lab, round(r2, 3), flush=True)
        (DATA / f"qloop_t{t}_agentic_app.json").write_text(
            json.dumps(
                {
                    "t": t,
                    "nested": True,
                    "source": str(RUN1),
                    "extra": "agentic_cands.cands",
                    "sites": sites,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
