"""Does an MLP down component create its result code, or pass on one it was given? (from `mlp_units`)

    python -m param_decomp.arith_repr.vectors.mlp_analysis   # -> OUT/mlp_units.parquet

For each alive down component d and position p (4 = `=`; also 1, 3 for layers 0-5), per op, with
q_res = sum on add and diff on sub:

* `in_<line>`: share of the line in the variance of its neurons' LINEAR reads g and u (weights
  V_d[n]^2 var) — what the gate/up components hand it;
* `out_<line>`: share of the line in the variance of its inner `sum_n V_d[n] act_n`;
* on the result line, act's coefficient A(k) splits into `pass` = s_0 u_res + u_0 kappa g_res (the
  result code already in g / u, carried through; kappa = cov(s, g) / var(g)) and `new` = A - pass;
  `created` = |new|^2 / |A|^2 summed over k;
* `cross_fit` = Re<new, X> / |new| |X| with X(k) = s_a u_b + s_b u_a + u_0 (s_res - kappa g_res) on
  add (u_b -> conj u_b, s_b -> conj s_b on sub): 1 when the new result code is exactly what the
  product of an a-code and a b-code (in silu(g) * u, or inside silu(g)) predicts, line terms only;
* `k_top`, `peak_out`: the result-line harmonic with most power and the value it peaks at;
  `peak_a`, `peak_b`: peak values of the a- and b-line parts of g + u at that k (weighted), so that
  a product predicts peak_out = peak_a + peak_b (add) / peak_a - peak_b (sub) mod 100 / k."""

import numpy as np
import pandas as pd

from param_decomp.arith_repr.vectors.common import LINES, OUT, comp_table, peak_value
from param_decomp.arith_repr.vectors.load import W50


def analyse_layer(layer: int, comps: dict[str, np.ndarray]) -> list[dict[str, float]]:
    z = dict(np.load(OUT / f"mlp/L{layer}.npz"))
    ds, Vd = z["down_cols"], z["Vd"]  # Vd (n_neur, n_down)
    rows = []
    positions = sorted({int(k.split("_p")[1].split("_")[0]) for k in z if k.startswith("act_p")})
    for p in positions:
        for o in range(2):
            S, Uu, G, A = (z[f"{n}_p{p}_o{o}"] for n in ("s", "u", "g", "act"))  # (n, 4, 50)
            s0, u0 = z[f"s_mean_p{p}_o{o}"], z[f"u_mean_p{p}_o{o}"]
            gvar = z[f"g_var_p{p}_o{o}"]
            svar = z[f"s_var_p{p}_o{o}"]
            uvar = z[f"u_var_p{p}_o{o}"]
            # kappa = cov(s, g) / var(g) estimated from the line spectra (ratio of powers, signed by overlap)
            kap = (S * np.conj(G)).sum((1, 2)).real / np.maximum(
                (np.abs(G) ** 2).sum((1, 2)), 1e-12
            )
            res = 2 if o == 0 else 3
            if o == 0:
                cross = S[:, 0] * Uu[:, 1] + S[:, 1] * Uu[:, 0]
            else:
                cross = S[:, 0] * np.conj(Uu[:, 1]) + np.conj(S[:, 1]) * Uu[:, 0]
            cross = cross + u0[:, None] * (S[:, res] - kap[:, None] * G[:, res])
            P = s0[:, None] * Uu[:, res] + u0[:, None] * kap[:, None] * G[:, res]
            for j, d in enumerate(ds):
                v = Vd[:, j]
                if np.abs(v).max() == 0:
                    continue
                Ad = (v[:, None, None] * A).sum(0)  # (4, 50)
                avar = (np.abs(Ad) ** 2 * W50).sum()
                if avar < 1e-10:
                    continue
                out = (np.abs(Ad) ** 2 * W50).sum(-1) / avar
                w_in = v**2
                gin = (w_in[:, None] * (np.abs(G) ** 2 * W50).sum(-1)).sum(0)
                uin = (w_in[:, None] * (np.abs(Uu) ** 2 * W50).sum(-1)).sum(0)
                gin_t = (w_in * gvar).sum()
                uin_t = (w_in * uvar).sum()
                a_res = Ad[res]
                new = a_res - (v[:, None] * P).sum(0)
                x = (v[:, None] * cross).sum(0)
                created = (np.abs(new) ** 2).sum() / max((np.abs(a_res) ** 2).sum(), 1e-12)
                fit = (new * np.conj(x)).sum().real / max(
                    np.linalg.norm(new) * np.linalg.norm(x), 1e-12
                )
                k0 = int(np.argmax(np.abs(a_res)))
                k = k0 + 1
                # a / b parts of the linear inputs at k, V-weighted (both g and u)
                ca = (
                    v
                    * (
                        G[:, 0, k0] / np.sqrt(np.maximum(gvar, 1e-12))
                        + Uu[:, 0, k0] / np.sqrt(np.maximum(uvar, 1e-12))
                    )
                ).sum()
                cb = (
                    v
                    * (
                        G[:, 1, k0] / np.sqrt(np.maximum(gvar, 1e-12))
                        + Uu[:, 1, k0] / np.sqrt(np.maximum(uvar, 1e-12))
                    )
                ).sum()
                row: dict[str, float] = {
                    "col": int(d), "layer": layer, "cidx": int(comps["cidx"][d]), "pos": p, "op": o,
                    "var_act": float(avar), "created": float(created), "cross_fit": float(fit),
                    "k_top": k, "peak_out": float(peak_value(a_res[k0], k)),
                    "peak_a": float(peak_value(ca, k)), "peak_b": float(peak_value(cb, k)),
                    "res_share_k": float(np.abs(a_res[k0]) ** 2 * W50[k0] / avar),
                    "s_var": float((w_in * svar).sum()),
                }  # fmt: skip
                for li, line in enumerate(LINES):
                    row[f"out_{line}"] = float(out[li])
                    row[f"in_g_{line}"] = float(gin[li] / max(gin_t, 1e-12))
                    row[f"in_u_{line}"] = float(uin[li] / max(uin_t, 1e-12))
                rows.append(row)
    return rows


def main() -> None:
    comps = comp_table()
    rows = []
    for layer in range(32):
        if (OUT / f"mlp/L{layer}.npz").exists():
            rows += analyse_layer(layer, comps)
            print(layer, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "mlp_units.parquet")
    print(df.shape)


if __name__ == "__main__":
    main()
