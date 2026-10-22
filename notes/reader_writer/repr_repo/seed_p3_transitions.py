"""Add the transitions L<b>.p3.b_mod_10, b = 0..11 (position 3, the token b) to the representation repository.

From track_mod10_p3/{analysis,context}.json (track_p3.py, p3_page.py). Definitions as in seed_blocks_mod10.py, with
r = b mod 10 and residue classes pooled over 20 contexts (operation, a); in addition, per transition, the attention
weight of position 3 on positions 0..3, the context offsets of the b circle in D, and the per-context separability.
The description is generated from the numbers. Nothing is centred.

    python seed_p3_transitions.py
"""

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN  # noqa: E402
from repo import ROOT, Repo  # noqa: E402

TRACK = RUN / "analysis/reader_writer/track_mod10_p3"


def main() -> None:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copytree(ROOT, ROOT.parent / f"representation_repository.bak-{stamp}")
    repo = Repo()
    an = json.loads((TRACK / "analysis.json").read_text())
    ctx = json.loads((TRACK / "context.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()  # fmt: skip
    today = datetime.date.today().isoformat()
    prov = {"script": "notes/reader_writer/repr_repo/seed_p3_transitions.py", "commit": commit, "date": today,
            "data": str(TRACK), "notebook": "Mod 10 representations at b (position 3)"}  # fmt: skip
    defs = json.loads((ROOT / "transitions.json").read_text())["transitions"][0]["definitions"] | {
        "prompts": "8,000: op in (+, -), a in (7, 48, 89, 130, 171, 212, 253, 294, 335, 376), b = 1..400; a context "
        "is one (op, a) pair; residue classes (b mod 10) pool 800 prompts over the 20 contexts",
        "op_offset": "norm of the difference between the mean context offsets of + and - in D coordinates; a context "
        "offset is the mean over residues of the context's residue means minus the average over contexts",
        "offset_share": "share of the across-context variation of the context residue means reproduced by one offset per context",
        "per_context_ovr": "per context, number of residues linearly separable (affine) from the other nine in D",
    }  # fmt: skip
    rep_of = {b: [r for r in repo.representations if r.startswith(f"L{b}.p3.")] for b in range(13)}
    for b in range(12):
        blk = an["blocks"][str(b)]
        att, mlp = blk["attn"], blk["mlp"]
        after, c_after = an["boundaries"][f"pre_{b + 1}"], ctx[f"pre_{b + 1}"]
        g = mlp["out_gain"]
        w = att["weight_by_position"]
        desc = (f"Attention puts {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} / {w[3]:.2f} of its weight on <BOS> / a / op / b and has "
                f"period-10 gain <= {max(att['out_gain']):.2f}. The MLP's period-10 gain is {min(g):.2f}-{max(g):.2f} "
                f"(secant r2 {mlp['mechanism']['r2_secant_sum']:.2f}, additivity {mlp['mechanism']['r2_sum_single']:.2f}). "
                f"Leaving the block: closest-pair separation {after['clouds']['sep_min']:.2f}, accuracy "
                f"{after['clouds']['acc_loo']:.3f}, {after['sep_D_affine_ovr']} of 10 residues separable with contexts pooled "
                f"({min(c_after['per_context_ovr'])} in the worst single context), offset between the + and - circles "
                f"{c_after['op_offset']:.2f} against a circle radius of {c_after['radius']:.2f}.")  # fmt: skip
        tr = {
            "id": f"L{b}.p3.b_mod_10", "block": b, "position": 3,
            "input_reps": rep_of[b], "output_reps": rep_of[b + 1] if b < 11 else [],
            "attention": {k: att[k] for k in ("out_gain", "out_share_from_D", "r2_lin", "delta_over_input_D",
                                              "r2_value_path", "weight_by_position")},
            "mlp": {k: mlp[k] for k in ("out_gain", "out_share_from_D", "r2_lin", "delta_over_input_D", "beta_re")}
            | {"secant": mlp["mechanism"]["r2_secant_sum"], "additivity": mlp["mechanism"]["r2_sum_single"],
               "top_readers": mlp["mechanism"]["top90"], "mechanism_prompts": mlp["mechanism"]["prompts"],
               "plane_cos_across": [float(np.mean(c)) for c in after["cos_prev"]]},
            "residues": {"before": an["boundaries"][f"mid_{b}"]["clouds"], "after": after["clouds"],
                         "sep_D_affine_ovr_after": after["sep_D_affine_ovr"], "context_after": c_after,
                         "a_mod10_amp_after": after["a_amp"], "a_mod10_in_D_share_after": after["a_in_D_share"]},
            "description": desc, "definitions": defs, "provenance": prov,
        }  # fmt: skip
        repo.transitions[tr["id"]] = tr
        print(tr["id"], desc[:160])
    repo.save()


if __name__ == "__main__":
    main()
