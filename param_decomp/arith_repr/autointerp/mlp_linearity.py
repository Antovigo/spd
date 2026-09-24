"""Is an MLP a linear map on the operand codes? Exact split of each MLP's write into a linear part,
the silu nonlinearity (gates switching) and the gate x up product, on the model's own neurons.

    python -m param_decomp.arith_repr.autointerp.mlp_linearity --run <run_dir> --resid <dir> <layer>

No decomposition anywhere: the frozen model's MLP at `layer`, fed the recorded post-norm residual
at `a`, `b` and `=`. Population of a position: every prompt of the pool (both ops; at `a` the
100 distinct prefixes). Per neuron n, over that population, g = gate pre-activation, u = up,
s = silu(g), h = s u; bars are population means, d = deviation from the mean, and
s ~ alpha + beta g is the least-squares line of silu over the neuron's own range of g, with
residual r (zero mean, uncorrelated with g). Then exactly

    h - h_bar =  s_bar du                 `up`     linear: up read through a frozen gate
               + u_bar beta dg            `gate`   linear: the gate's own read, silu as a line
               + u_bar r                  `silu`   silu's bend: a gate switching on / off
               + ds du - mean             `prod`   gate x up: a product of two reads

`up` + `gate` is one affine map of the input, J = W_d [diag(s_bar) W_u + diag(u_bar beta) W_g]
(the MLP if it were linear). A neuron's gate never "flips" on this population iff r = 0.

For each variable (classes v), the MLP's write to the variable's arrangement (class means of
W_d (h - h_bar), centred, per op) is the sum of the four terms' class means. Stored per
(position, op, variable) in `mlp_lin/L<layer>.parquet`: energy of the write (`E`, weighted
between-class variance) and `E_in` of the stream it reads (raw post_attn), each term's share
(projection on the write, shares sum to 1), `r2_lin` = 1 - |write - up - gate|^2 / |write|^2
(how much of the write one linear map gives), `r2_noflip` = 1 - |silu|^2 / |write|^2, and how
concentrated the silu term is: `n90_silu` neurons carry 90 % of its projection, with the share
of those that actually cross zero (`cross`: 2-98 % of the population has g > 0). Variable
`prompt` = every prompt its own class (the per-prompt write). `mlp_lin/L<layer>.npz` keeps
the population statistics (for the linearised forward, `mlp_patch.py`) and the per-neuron
projections of each term for the main variables."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.autointerp.mechanisms import AI
from param_decomp.arith_repr.resid import load_raw, load_read_input

POSITIONS = {1: "a", 3: "b", 4: "="}
VARIABLES = {
    1: ("a%100", "a%10", "a//10", "a%2"),
    3: ("b%100", "b%10", "b//10", "b%2", "prompt"),
    4: ("res%100", "res%10", "res//10", "prompt"),
}
TERMS = ("up", "gate", "silu", "prod")
NEURON_VARS = ("a%10", "a%100", "b%10", "b%100", "res%10", "res%100")


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def value(labels: np.ndarray, var: str) -> np.ndarray:
    """Class of every prompt under `var` (labels: (n, 3) = op, a, b)."""
    op, a, b = labels.T
    res = np.where(op == 0, a + b, a - b)
    src = {"a": a, "b": b, "res": res}[var.split("%")[0].split("//")[0]]
    if "//" in var:
        return src // int(var.split("//")[1])
    return src % int(var.split("%")[1])


def class_means(x: np.ndarray, cls: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(classes, d) means of x per class, centred with the class weights; and those weights."""
    if cls is None:  # every prompt its own class
        out = x.astype(np.float64)
        wts = np.full(len(x), 1 / len(x))
        return out - wts @ out, wts
    keys, inv = np.unique(cls, return_inverse=True)
    onehot = np.zeros((keys.size, len(x)), np.float32)
    onehot[inv, np.arange(len(x))] = 1
    n = onehot.sum(1).astype(np.float64)
    out = (onehot @ x).astype(np.float64) / n[:, None]
    wts = n / n.sum()
    return out - wts @ out, wts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("layer", type=int)
    args = parser.parse_args()
    layer = args.layer
    out_dir = args.run / AI / "mech" / "mlp_lin"
    out_dir.mkdir(parents=True, exist_ok=True)

    w = Weights()
    Wg = w.get(f"model.layers.{layer}.mlp.gate_proj.weight")  # (14336, 4096)
    Wu = w.get(f"model.layers.{layer}.mlp.up_proj.weight")
    Wd = w.get(f"model.layers.{layer}.mlp.down_proj.weight")  # (4096, 14336)
    WdT = Wd.T.astype(np.float64)
    meta = json.loads((args.resid / "pool.json").read_text())
    labels_all = np.array([(0 if o == "add" else 1, a, b) for o, a, b in meta["labels"]])

    rows: list[dict[str, object]] = []
    stats: dict[str, np.ndarray] = {}
    for p, pname in POSITIONS.items():
        x_all = load_read_input(args.resid, f"mlp_in.{layer}", p)
        raw_all = load_raw(args.resid, f"post_attn.{layer}", p)
        if p == 1:  # the residual at `a` depends on a only: one row per value
            sel = np.flatnonzero((labels_all[:, 0] == 0) & (labels_all[:, 2] == 1))
            assert np.array_equal(x_all[sel[5]], x_all[sel[5] + 37]), "not causal at `a`?"
        else:
            sel = np.arange(len(labels_all))
        x, raw, labels = x_all[sel], raw_all[sel].astype(np.float64), labels_all[sel]
        del x_all, raw_all
        g = x @ Wg.T
        u = x @ Wu.T
        s = silu(g)
        g_bar, u_bar, s_bar = g.mean(0), u.mean(0), s.mean(0)
        dg, du, ds = g - g_bar, u - u_bar, s - s_bar
        beta = (dg * ds).mean(0) / np.maximum((dg * dg).mean(0), 1e-12)
        r = ds - beta * dg
        prod = ds * du
        h_bar = s_bar * u_bar + prod.mean(0)
        alpha = s_bar - beta * g_bar
        # `silu as a line, product kept`: (alpha + beta g) u + c reproduces h_bar on average
        c_sl = h_bar - ((alpha + beta * g) * u).mean(0)
        frac_on = (g > 0).mean(0)
        cross = (frac_on > 0.02) & (frac_on < 0.98)
        for k, v in dict(
            g_bar=g_bar,
            u_bar=u_bar,
            s_bar=s_bar,
            beta=beta,
            alpha=alpha,
            h_bar=h_bar,
            c_sl=c_sl,
            frac_on=frac_on,
            x_bar=x.mean(0),
        ).items():
            stats[f"{pname}.{k}"] = v.astype(np.float32)
        hid = {"up": s_bar * du, "gate": u_bar * beta * dg, "silu": u_bar * r, "prod": prod}
        # sanity: the four terms rebuild the model's write, and it matches the recorded one
        write = (s * u) @ Wd.T
        rec = load_raw(args.resid, f"resid.{layer + 1}", p)[sel] - raw
        rel = float(np.linalg.norm(write - rec) / np.linalg.norm(rec))
        n_on_both = int(cross.sum())
        ops = (None,) if p == 1 else (0, 1)
        for o in ops:
            m = np.ones(len(labels), bool) if o is None else labels[:, 0] == o
            for var in VARIABLES[p]:
                cls = None if var == "prompt" else value(labels[m], var)
                C_t = {t: class_means(hid[t][m], cls)[0] @ WdT for t in TERMS}
                C = C_t["up"] + C_t["gate"] + C_t["silu"] + C_t["prod"]
                A_in, wts = class_means(raw[m], cls)
                E = float(wts @ (C * C).sum(1))
                E_in = float(wts @ (A_in * A_in).sum(1))
                shares = {t: float(wts @ (C_t[t] * C).sum(1)) / E for t in TERMS}
                resid_lin = C - C_t["up"] - C_t["gate"]
                row: dict[str, object] = dict(
                    layer=layer,
                    pos=pname,
                    op={None: "both", 0: "add", 1: "sub"}[o],
                    var=var,
                    E=E,
                    E_in=E_in,
                    r2_lin=1 - float(wts @ (resid_lin * resid_lin).sum(1)) / E,
                    r2_noflip=1 - float(wts @ (C_t["silu"] ** 2).sum(1)) / E,
                    r2_noprod=1 - float(wts @ (C_t["prod"] ** 2).sum(1)) / E,
                    n_cross=n_on_both,
                    rel_err_write=rel,
                    **{f"share_{t}": shares[t] for t in TERMS},
                )
                if var != "prompt":
                    # per-neuron projections of each term on the write (neurons x terms)
                    proj_dir = C @ WdT.T  # (classes, 14336): write read back per neuron
                    per_n = np.stack(
                        [wts @ (class_means(hid[t][m], cls)[0] * proj_dir) / E for t in TERMS], 1
                    )
                    sil = per_n[:, 2]
                    order = np.argsort(-np.abs(sil))
                    cum = np.cumsum(np.abs(sil[order]))
                    n90 = int(np.searchsorted(cum, 0.9 * cum[-1]) + 1) if cum[-1] > 0 else 0
                    row.update(
                        n90_silu=n90, cross_top=float(cross[order[:n90]].mean()) if n90 else np.nan
                    )
                    if var in NEURON_VARS:
                        stats[f"{pname}.{row['op']}.{var}.neuron_share"] = per_n.astype(np.float32)
                rows.append(row)
                print(
                    f"L{layer} {pname} {row['op']} {var}: E/E_in {E / E_in:.3f} "
                    f"r2_lin {row['r2_lin']:.3f} noflip {row['r2_noflip']:.3f} "
                    f"shares "
                    + " ".join(f"{t}:{shares[t]:.2f}" for t in TERMS)
                    + f" (write rel err {rel:.3f})",
                    flush=True,
                )
        del g, u, s, dg, du, ds, r, prod, hid, write
    pd.DataFrame(rows).to_parquet(out_dir / f"L{layer}.parquet")
    np.savez(out_dir / f"L{layer}.npz", **cast(dict[str, Any], stats))


if __name__ == "__main__":
    main()
