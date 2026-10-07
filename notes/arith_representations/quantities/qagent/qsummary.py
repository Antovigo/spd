"""Summary of one position's loop and patching results (AGENT.md step 8), printed as markdown.

* how often each quantity is in a site's final list, split into attention and MLP inputs;
* medians of dims, held-out R^2 (each scheme run) and important residual share;
* patching (if patch_t<t>_<model>[_tag].json exist): sites whose mean patch costs KL > 0.01, the
  median fraction of that KL the reconstruction removes (in-sample and held-out), the worst sites,
  and the all-sites patches.

    python qsummary.py <t> [--tag name]
"""

import argparse
import json
from collections import Counter

import numpy as np
from qdata import DATA

MATTERS = 0.01  # a site matters at t if patching its mean costs more than this KL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    run = json.loads((DATA / f"qloop_t{args.t}{tag}.json").read_text())
    sites = [s for s in run["sites"] if not s.get("constant")]
    const = [s["site"] for s in run["sites"] if s.get("constant")]
    print(f"## t = {args.t} ({run['position']}), D = {run['D']}, primary folds: {run['scheme']}")
    if const:
        print(f"constant over the domain (not fit): {', '.join(const)}")
    kinds = {"attn": [s for s in sites if s["site"].endswith("attn")],
             "mlp": [s for s in sites if s["site"].endswith("mlp")]}  # fmt: skip
    cnt = {k: Counter(a["name"] for s in v for a in s["accepted"]) for k, v in kinds.items()}
    names = sorted(
        set(cnt["attn"]) | set(cnt["mlp"]), key=lambda n: -(cnt["attn"][n] + cnt["mlp"][n])
    )
    print(
        f"\n| quantity | attention inputs ({len(kinds['attn'])}) | MLP inputs ({len(kinds['mlp'])}) |"
    )
    print("|---|---|---|")
    for n in names:
        print(f"| {n} | {cnt['attn'][n]} | {cnt['mlp'][n]} |")
    keys = [
        ("dims", "dims of the accepted list"),
        ("heldout_r2", f"held-out R^2 ({run['scheme']})"),
    ]
    keys += [
        (k, f"held-out R^2 ({k.removeprefix('heldout_r2_')})")
        for k in sites[0]
        if k.startswith("heldout_r2_")
    ]
    keys += [("important_share_in", "important residual share, in-sample"),
             ("important_share_heldout", "important residual share, held-out")]  # fmt: skip
    print("\n| median | attention inputs | MLP inputs |\n|---|---|---|")
    for k, lab in keys:
        print(f"| {lab} | {np.median([s[k] for s in kinds['attn']]):.3g} | {np.median([s[k] for s in kinds['mlp']]):.3g} |")  # fmt: skip
    for model in ("alive", "original"):
        p = DATA / f"patch_t{args.t}_{model}{tag}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        rows = [(k, v["mean"]["kl_mean"], v["recon"]["kl_mean"], v["heldout"]["kl_mean"], v.get("r2_stream"))
                for k, v in r["sites"].items()]  # fmt: skip
        imp = [x for x in rows if x[1] > MATTERS]
        print(f"\n**Patching, {model} model** ({r['n_prompts']} prompts; unpatched accuracy {r['baseline_acc']:.3f}): "
              f"{len(imp)} of {len(rows)} sites cost KL > {MATTERS} when mean-patched", end="")  # fmt: skip
        if imp:
            fr = np.median([1 - x[2] / x[1] for x in imp])
            fh = np.median([1 - x[3] / x[1] for x in imp])
            print(
                f"; median fraction of that KL removed: {fr:.3f} (in-sample), {fh:.3f} (held-out)"
            )
            print(
                "\n| site | mean patch KL | reconstruction KL | held-out KL | stream R^2 |\n|---|---|---|---|---|"
            )
            for k, mk, rk, hk, r2 in sorted(imp, key=lambda x: -x[2] / x[1])[:6]:
                print(
                    f"| {k} | {mk:.4f} | {rk:.4f} | {hk:.4f} | {'' if r2 is None else f'{r2:.3f}'} |"
                )
        else:
            print()
        if r.get("dead_readers_off"):
            m = r["masked_only"]
            print(f"\n{r['dead_readers_off']} readers dead at t switched off at t in every run; that alone: "
                  f"KL {m['kl_mean']:.4f} (median {m['kl_median']:.4f}), accuracy {m['acc']:.4f}")  # fmt: skip
        print(
            "\n| all sites together | KL mean | KL median | KL vs dead-off model | accuracy |\n|---|---|---|---|---|"
        )
        for v in ("recon", "heldout", "mean"):
            a = r["all_sites_" + v]
            print(f"| {v} | {a['kl_mean']:.4f} | {a['kl_median']:.4f} | {a.get('kl_vs_masked_mean', float('nan')):.4f} | {a['acc']:.4f} |")  # fmt: skip


if __name__ == "__main__":
    main()
