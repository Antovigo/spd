"""The op switches of layers 16-31 and the neurons they gate (alive-only model, `=`).

    python -m param_decomp.arith_repr.vectors.flip_switches   # CPU -> stdout + OUT/flip/opflow/switches.json
    (needs OUT/flip/opflow/h_eq.npz from `flip_opflow capture`)

Switch: a gate or up component of layers 16-31 whose op term carries >= 0.5 of its inner activation's
variance over the 20000 prompts (`flip_readers.switch_cols`). Its effect on neuron n's pre-activation is
U_c[n] * h_c; its op effect is U_c[n] * (mean h_c on add - mean h_c on sub). Per layer, the switches
with the largest |op effect| summed over neurons; per switch, the 5 neurons with the largest |op
effect|. Per neuron: mean act per op; the act's ANOVA shares (`flip_opflow.anova`) and the share of its
op-odd part D explained by the class means of a + b and of a - b; and its direct effect on the answer:
the logits of w_n (its down write, scaled by the final norm gain) on the number tokens 0..200 and the
"-" token, summarised by the correlation of the number-token logits with [v >= 100], with v, and the
minus logit relative to the number-token mean, in units of the number-token logits' standard
deviation."""

import json
from typing import Any

import numpy as np

from param_decomp.arith_repr.isa.components_model import ComponentsModel
from param_decomp.arith_repr.vectors.flip_opflow import OUT, anova, grid
from param_decomp.arith_repr.vectors.flip_readers import switch_cols


def main() -> None:
    cm = ComponentsModel()
    g = grid(cm)
    H = dict(np.load(OUT / "h_eq.npz"))
    sw = switch_cols(cm, H)
    tok = np.r_[np.asarray(cm.num_ids), int(cm.minus)]
    Eg = (
        np.asarray(cm.unembed[tok], np.float64) * np.asarray(cm.final, np.float64)[None]
    )  # (202, d)
    v = np.arange(201)
    res: dict[str, Any] = {}
    for li in range(16, 32):
        Vd = np.asarray(cm.sites[li]["down"].V, np.float64)
        Ud = np.asarray(cm.sites[li]["down"].U, np.float64)
        pre = {
            k: H[f"{li}.{k}"].astype(np.float64) @ np.asarray(cm.sites[li][k].U, np.float64)
            for k in ("gate", "up")
        }
        act = pre["gate"] / (1 + np.exp(-pre["gate"])) * pre["up"]
        rows = []
        for k in ("gate", "up"):
            h = H[f"{li}.{k}"].astype(np.float64)
            U = np.asarray(cm.sites[li][k].U, np.float64)
            names = cm.sites[li][k].names
            for j in np.flatnonzero(sw[(li, k)]):
                dh = h[cm.op == 0, j].mean() - h[cm.op == 1, j].mean()
                eff = U[j] * dh
                rows.append((float(np.abs(eff).sum()), k, j, names[j], dh, eff))
        rows.sort(key=lambda r: -r[0])
        out = []
        print(f"\n== L{li}: {len(rows)} switches", flush=True)
        for tot, k, j, nm, _dh, eff in rows[:4]:
            h = H[f"{li}.{k}"][:, j].astype(np.float64)
            top = np.argsort(-np.abs(eff))[:5]
            ra = anova(act[:, top], g)
            neurons = []
            for i, n in enumerate(top):
                w = Vd[n] @ Ud
                lg = Eg @ w
                ln = lg[:201]
                sd = ln.std()
                rec = {"neuron": int(n), "op effect": float(eff[n]),
                       "act add/sub": [float(act[cm.op == o, n].mean()) for o in (0, 1)],
                       "shares": {q: float(ra[q][i] / max(ra["total"][i], 1e-30)) for q in ("op", "a", "b", "op*a", "op*b", "a*b", "op*a*b")},
                       "D|a+b": float(ra["D|a+b"][i]), "D|a-b": float(ra["D|a-b"][i]),
                       "corr >=100": float(np.corrcoef(ln, v >= 100)[0, 1]), "corr v": float(np.corrcoef(ln, v)[0, 1]),
                       "minus": float((lg[201] - ln.mean()) / sd),
                       "up tokens": [int(t) for t in np.argsort(-ln)[:3]], "down tokens": [int(t) for t in np.argsort(ln)[:3]]}  # fmt: skip
                neurons.append(rec)
            out.append(
                {
                    "name": nm,
                    "mean h add/sub": [float(h[cm.op == o].mean()) for o in (0, 1)],
                    "neurons": neurons,
                }
            )
            print(
                f"  {nm} h {out[-1]['mean h add/sub'][0]:+.1f}/{out[-1]['mean h add/sub'][1]:+.1f} (total |op effect| {tot:.1f})",
                flush=True,
            )
            for rec in neurons:
                s = rec["shares"]
                print(f"    n{rec['neuron']:5d} op eff {rec['op effect']:+.2f} act {rec['act add/sub'][0]:+.2f}/{rec['act add/sub'][1]:+.2f}"
                      f" | op {s['op']:.2f} a {s['a']:.2f} b {s['b']:.2f} a*b {s['a*b']:.2f} op*a*b {s['op*a*b']:.2f}"
                      f" | D a+b {rec['D|a+b']:.2f} a-b {rec['D|a-b']:.2f} | logits: corr[v>=100] {rec['corr >=100']:+.2f}"
                      f" corr v {rec['corr v']:+.2f} minus {rec['minus']:+.1f}sd up {rec['up tokens']} down {rec['down tokens']}", flush=True)  # fmt: skip
        res[str(li)] = out
    (OUT / "switches.json").write_text(json.dumps(res, indent=1))
    print("saved", OUT / "switches.json", flush=True)


if __name__ == "__main__":
    main()
