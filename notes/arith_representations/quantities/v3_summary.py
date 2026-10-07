"""Print one line per site from v3_token_a.json."""

import json

for s in json.load(open("v3_token_a.json")):  # noqa: SIM115
    names = ", ".join(a["name"].split(" (")[0] for a in s["accepted"])
    print(
        f"{s['site']:<9} n={s['n']:3d} dims={s['dims']:2d} R2 {s['r2_in']:.3f} heldout {s['heldout_r2']:.3f} "
        f"imp {s['important']['share']:.3f} imp_cv {s['important_cv']['share']:.3f}  {names}"
    )
