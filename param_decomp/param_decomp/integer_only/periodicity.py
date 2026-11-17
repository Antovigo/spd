"""Periodicity and constancy labels of the collected components' inner activations, for the
`(a, b)`-grid applet's period filter. The tests are those of addsub-08's
`notes/reader_writer/repr_repo/seed_periodicity.py` (branch experiment/arith_representations),
generalized to every position, both operations and unequal class sizes.

Components: every stored (position, component) pair of a collected dataset (`collect.py`): the
components of the list that are ON (original CI > the ON threshold) at that position on at least
one prompt. A component is a site name (e.g. `layers.26.mlp.down_proj`) plus its index within
the site's decomposition; its id here is `L<block>.<kind>.<index>`, kind in gate, up, down, q, k,
v, o.

Notation:
  t          token position, an integer index into the pool's position labels
  i          a prompt (a row of the pool)
  op         the operation, add or sub; every periodicity test runs on one operation's prompts
  a, b       the integer operands of prompt i; r its true answer (a + b or a - b); all >= 0
  f_c(i)     component c's inner activation x V_c / ||V_c|| at position t on prompt i, in the
             dataset's model (x the site's input, V_c the component's input vector)
  v          the variable tested: a, b or r (`VARIABLES`); v_i its value on prompt i
  rho        a class, rho = v mod 100, an integer in 0..99; n_rho its number of prompts
  m_c(rho)   the mean of f_c over the class's prompts
  M_c(k)     sum_rho m_c(rho) exp(-2 pi j k rho / 100), k in 0..99 (j the imaginary unit)
  tau(k)     100 / gcd(k, 100): the exact period of frequency k (tau(0) = 1, the constant)
  P_tau m_c  the part of m_c made of the frequencies k with tau(k) = tau, tau a divisor of 100 in
             {1, 2, 4, 5, 10, 20, 25, 50, 100} (`TAUS`); the parts are orthogonal and sum to m_c.
             A part of exact period tau holds nothing a divisor of tau already explains: a function
             of v mod 2 has all its non-constant energy in P_2 and none in P_10
  E          sum_i f_c(i)^2 over the operation's prompts, uncentred (the network has no biases)
  share_tau  sum_i (P_tau m_c)(v_i mod 100)^2 / E: the energy, over the prompts, of the part of f_c
             that is a function of v mod 100 with exact period tau. With equal class sizes (every
             variable of the full addition grid) the shares of one variable sum to
             1 - share_rest; with unequal ones (sub, where b <= a) they need not
  share_rest 1 - sum_rho n_rho m_c(rho)^2 / E: the energy no function of v mod 100 explains
             (dependence on another variable, on v // 100, ...)
  null_tau   99th percentile of share_tau over NPERM relabellings (seed 0) v -> pi(v), pi a random
             permutation of v's distinct values: what the same values get with no arrangement in v
  align_tau  the same with pi rotating each hundred 100 q .. 100 q + 99 (q >= 1) of v's values by an
             independent random offset in 0..99, the first hundred fixed: keeps the shape within each
             hundred and breaks only their alignment. Used for tau in `ALIGN_TAUS` = {20, 25, 50, 100},
             where a smooth non-periodic f_c beats the permutation null; for tau <= 10 the chance that
             every offset is a multiple of tau (1/8 for tau = 2 and three hundreds) is too large for a
             1% test

exact_periods = [tau > 1 with share_tau >= THRESHOLD, share_tau > null_tau and, for tau in
ALIGN_TAUS, share_tau > align_tau]. periods = the members of exact_periods that divide no other
member: the fewest periods whose periodic functions add up to every present part. The indicator of
v mod 10 = 3 has parts of exact period 2, 5 and 10 and is labelled [10]; [v mod 2 = 0] + [v mod 5 =
3] has parts 2 and 5 only and is labelled [2, 5]. minimal_period = lcm(periods).

Tested: at every position whose activation can depend on the operands (a's token and after), per
operation, in a; from b's token on also in b and r. A component periodic in a and in b carries a
label in each.

Constancy, at every stored position, on all prompts of both operations:
  share_const   N mean(f_c)^2 / E (N all prompts, E over all of them): the energy one constant explains
  share_op      sum over ops of N_op mean_op(f_c)^2 / E: the energy one constant per operation explains
  label         "constant" if share_const >= CONST_THRESHOLD, else "op_constant" if share_op >=
                CONST_THRESHOLD, else null. Before the operator's token share_op = share_const.

Outputs: `<datasets>/periodicity/periodicity.json` (+ README.md) and the applet's
`<applet data>/periodicity.js` (`grids_app.html`'s period menu), written by `main`."""

import argparse
import datetime
import json
import subprocess
import time
from math import gcd, lcm
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from param_decomp.ci_filter.pool import SYMBOLS
from param_decomp.core.log import logger
from param_decomp.integer_only.applet import applet_page_text
from param_decomp.integer_only.layout import INNER_DIR, Layout

TAUS = (1, 2, 4, 5, 10, 20, 25, 50, 100)
ALIGN_TAUS = (20, 25, 50, 100)
TAU_OF_K = np.asarray([100 // gcd(k, 100) for k in range(100)])
THRESHOLD = 0.05
"""Minimum share_tau of a present exact period (addsub-08's response threshold)."""
CONST_THRESHOLD = 0.95
NPERM = 1000
PERM_BATCH = 8
VARIABLES = ("a", "b", "r")
KINDS = {"mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "down",
         "self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v",
         "self_attn.o_proj": "o"}  # fmt: skip
PERIODICITY_DIR = "periodicity"


def projectors() -> np.ndarray:
    """`(len(TAUS) * 100, 100)` real: stacked `P_tau`, the orthogonal projectors of a function of
    `rho` onto its exact-period-`tau` part."""
    k = np.arange(100)
    W = np.exp(-2j * np.pi * np.outer(k, k) / 100)
    out = [np.real(W.conj().T @ np.diag(tau == TAU_OF_K) @ W) / 100 for tau in TAUS]
    return np.concatenate(out)


PROJ = projectors()


def class_shares(Sv: np.ndarray, nv: np.ndarray, rho: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Per-value sums `Sv (B, V, C)` and counts `nv (V,)`, class `rho (B, V)` of each value under
    `B` labellings -> `(B, len(TAUS) + 1, C)`: share_tau per `TAUS`, then share_rest."""
    B, V, C = Sv.shape
    rows = (np.arange(B)[:, None] * 100 + rho).ravel()
    onehot = sp.csr_matrix((np.ones(B * V), (rows, np.tile(np.arange(V), B))), (B * 100, B * V))
    sums = (onehot @ Sv.reshape(B * V, C)).reshape(B, 100, C)
    n = (onehot @ np.tile(nv, B)).reshape(B, 100)
    m = sums / np.maximum(n, 1)[..., None]
    parts = (PROJ @ m).reshape(B, len(TAUS), 100, C)
    shares = (parts**2 * n[:, None, :, None]).sum(2) / E
    rest = 1 - (m**2 * n[..., None]).sum(1) / E
    return np.concatenate([shares, rest[:, None]], 1)


def relabellings(values: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """One null relabelling of the distinct values `values` (ascending, >= 0) -> new values."""
    if kind == "perm":
        return values[rng.permutation(values.size)]
    hundred = values // 100
    offset = np.concatenate([[0], rng.integers(0, 100, hundred.max())])[hundred]
    return 100 * hundred + (values - 100 * hundred + offset) % 100


def variable_shares(F: np.ndarray, v: np.ndarray, E: np.ndarray) -> dict[str, np.ndarray]:
    """`F (n, C)` on one operation's prompts, `v (n,)` the variable -> {"share": (len(TAUS) + 1, C),
    "null": (len(TAUS), C), "align": (len(TAUS), C)}."""
    values, idx = np.unique(v, return_inverse=True)
    onehot = sp.csr_matrix((np.ones(v.size), (idx, np.arange(v.size))), (values.size, v.size))
    Sv = np.asarray(onehot @ F)
    nv = np.bincount(idx, minlength=values.size).astype(np.float64)
    out = {"share": class_shares(Sv[None], nv, (values % 100)[None], E)[0]}
    rng = np.random.default_rng(0)
    for kind, key in (("perm", "null"), ("align", "align")):
        acc = []
        for lo in range(0, NPERM, PERM_BATCH):
            B = min(PERM_BATCH, NPERM - lo)
            rho = np.stack([relabellings(values, kind, rng) % 100 for _ in range(B)])
            acc.append(class_shares(np.broadcast_to(Sv, (B, *Sv.shape)), nv, rho, E)[:, :-1])
        out[key] = np.quantile(np.concatenate(acc), 0.99, axis=0)
    return out


def labels(S: dict[str, np.ndarray], j: int) -> tuple[list[int], list[int]]:
    share, null, align = S["share"][:, j], S["null"][:, j], S["align"][:, j]
    exact = [tau for s, tau in enumerate(TAUS)
             if tau > 1 and share[s] >= THRESHOLD and share[s] > null[s]
             and (tau not in ALIGN_TAUS or share[s] > align[s])]  # fmt: skip
    periods = [tau for tau in exact if not any(u != tau and u % tau == 0 for u in exact)]
    return exact, periods


def r5(x: float) -> float:
    return float(f"{x:.5g}")


def entry(S: dict[str, np.ndarray], j: int, rms: float) -> dict[str, object]:
    exact, periods = labels(S, j)
    return {
        "periods": periods,
        "exact_periods": exact,
        "minimal_period": lcm(*periods) if periods else None,
        "share": {str(tau): r5(S["share"][s, j]) for s, tau in enumerate(TAUS)},
        "share_rest": r5(S["share"][-1, j]),
        "null_p99": {str(tau): r5(S["null"][s, j]) for s, tau in enumerate(TAUS) if tau > 1},
        "align_null_p99": {
            str(tau): r5(S["align"][s, j]) for s, tau in enumerate(TAUS) if tau in ALIGN_TAUS
        },
        "rms": r5(rms),
    }


def component_id(site: str, c: int) -> str:
    block, rest = site.removeprefix("layers.").split(".", 1)
    return f"L{block}.{KINDS[rest]}.{c}"


def tested_variables(labels_: list[str]) -> dict[int, tuple[str, ...]]:
    """Position -> the variables tested there: from a's token on a, from b's token on a, b, r."""
    t_a, t_b = labels_.index("a"), labels_.index("b")
    return {t: ("a",) if t < t_b else VARIABLES for t in range(t_a, len(labels_))}


def periodicity(datasets: Path) -> dict[str, Any]:
    index = dict(np.load(datasets / "index.npz"))
    pos_labels = [str(x) for x in index["position_labels"]]
    ops = [str(x) for x in index["ops"]]
    op_of = index["op"]
    var_values = {"a": index["a"], "b": index["b"], "r": index["r"]}
    assert all((x >= 0).all() for x in var_values.values())
    tested = tested_variables(pos_labels)
    comps: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for t in range(len(pos_labels)):
        cols = np.flatnonzero(index["col_position"] == t)
        if cols.size == 0:
            continue
        F_all = np.load(datasets / INNER_DIR / f"inner_pos{t}.npy").astype(np.float64)
        F_all /= np.maximum(index["col_v_norm"][cols], 1e-12)[None, :]
        rows = []
        for col in cols:
            site, c = str(index["col_site"][col]), int(index["col_component"][col])
            rows.append(comps.setdefault(component_id(site, c), {
                "site": site, "block": int(site.split(".")[1]),
                "kind": KINDS[site.split(".", 2)[2]], "component": c,
                "positions": {}, "constancy": {}}))  # fmt: skip
        E = np.maximum((F_all**2).sum(0), 1e-30)
        mean = F_all.mean(0)
        op_means = np.stack([F_all[op_of == k].mean(0) for k in range(len(ops))])
        op_n = np.bincount(op_of, minlength=len(ops))
        share_const = len(op_of) * mean**2 / E
        share_op = (op_n[:, None] * op_means**2).sum(0) / E
        for j, e in enumerate(rows):
            label = ("constant" if share_const[j] >= CONST_THRESHOLD
                     else "op_constant" if share_op[j] >= CONST_THRESHOLD else None)  # fmt: skip
            e["constancy"][str(t)] = {
                "share_const": r5(share_const[j]),
                "share_op": r5(share_op[j]),
                **{f"mean_{op}": r5(op_means[k, j]) for k, op in enumerate(ops)},
                "rms": r5(np.sqrt(E[j] / len(op_of))),
                "label": label,
            }
        for k, op in enumerate(ops):
            if t not in tested:
                break
            sel = np.flatnonzero(op_of == k)
            F = F_all[sel]
            E_op = np.maximum((F**2).sum(0), 1e-30)
            rms = np.sqrt(E_op / sel.size)
            for v in tested[t]:
                S = variable_shares(F, var_values[v][sel], E_op)
                for j, e in enumerate(rows):
                    e["positions"].setdefault(str(t), {}).setdefault(op, {})[v] = entry(
                        S, j, rms[j]
                    )
            logger.info(
                f"periodicity: t={t} {op}, {cols.size} components ({time.time() - t0:.0f}s)"
            )
        del F_all
    kinds = list(KINDS.values())

    def order(i: str) -> tuple[int, int, int]:
        return comps[i]["block"], kinds.index(comps[i]["kind"]), comps[i]["component"]

    index_out = {
        op: {str(t): {v: {str(tau): sorted((i for i, e in comps.items()
                                            if tau in e["positions"].get(str(t), {}).get(op, {}).get(v, {}).get("periods", [])),
                                           key=order) for tau in TAUS if tau > 1}
                      for v in vs} for t, vs in tested.items()}
        for op in ops
    }  # fmt: skip
    cindex = {
        str(t): {lab: sorted((i for i, e in comps.items()
                              if e["constancy"].get(str(t), {}).get("label") == lab), key=order)
                 for lab in ("constant", "op_constant")}
        for t in range(len(pos_labels))
    }  # fmt: skip
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()  # fmt: skip
    return {
        "definitions": __doc__,
        "threshold": THRESHOLD,
        "const_threshold": CONST_THRESHOLD,
        "nperm": NPERM,
        "taus": [tau for tau in TAUS if tau > 1],
        "align_taus": list(ALIGN_TAUS),
        "ops": ops,
        "position_labels": pos_labels,
        "variables": {str(t): list(vs) for t, vs in tested.items()},
        "index": index_out,
        "constancy_index": cindex,
        "components": comps,
        "provenance": {
            "script": "param_decomp/param_decomp/integer_only/periodicity.py",
            "commit": commit,
            "date": datetime.date.today().isoformat(),
            "source": str(datasets),
        },
    }


def applet_index(doc: dict[str, Any]) -> dict[str, Any]:
    """`window.PERIODICITY` for `grids_app.html`: keys `"<op symbol>:<t>"` for the periods (one
    slice each), `"<t>"` for constancy (both operations)."""
    sym = {op: SYMBOLS[op] for op in doc["ops"]}
    positions: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = {}
    constancy: dict[str, dict[str, dict[str, str]]] = {}
    blocks: dict[str, list[int]] = {}
    for e in doc["components"].values():
        site, c = e["site"], str(e["component"])
        for t, by_op in e["positions"].items():
            for op, by_var in by_op.items():
                key = f"{sym[op]}:{t}"
                positions.setdefault(key, {}).setdefault(site, {})[c] = {
                    v: x["periods"] for v, x in by_var.items()
                }
                blocks.setdefault(key, []).append(e["block"])
        for t, cs in e["constancy"].items():
            if cs["label"] is not None:
                constancy.setdefault(t, {}).setdefault(site, {})[c] = cs["label"]
    variables = {f"{sym[op]}:{t}": vs for t, vs in doc["variables"].items() for op in sym}
    return {
        "taus": doc["taus"],
        "variables": variables,
        "blocks": {k: [min(b), max(b)] for k, b in blocks.items()},
        "positions": positions,
        "constancy": constancy,
        "constancy_positions": sorted(
            {t for e in doc["components"].values() for t in e["constancy"]}, key=int
        ),
        "const_threshold": doc["const_threshold"],
    }


README = """\
# Periodicity

`periodicity.json`: how periodic and how constant the inner activations of the dataset's stored
(position, component) pairs are, in the model of `../{inner_dir}/`. `p` the loaded JSON;
components are keyed `L<block>.<kind>.<component>`, `component` the original decomposition's
index within the site (`p["components"][id]["site"]` names the site):

- `p["components"][id]["positions"][t][op][v]["periods"]`: the periods of that component's inner
  activation at position `t` (integer, a pool position index) on operation `op` (`add`, `sub`) in
  variable `v` (`a`, `b`, or `r` the true answer); `share` holds the fraction of activation
  energy at each exact period, `null_p99` / `align_null_p99` the thresholds it had to beat.
- `p["components"][id]["constancy"][t]["label"]`: `constant`, `op_constant` or null.
- `p["index"][op][t][v][tau]`: every component with period `tau` in `v` at `t` on `op`.
- `p["constancy_index"][t][label]`: every component with that constancy label at `t`.

Positions: {positions}. Tested: {tested}. Periods tested: {taus}; a period is kept when its share
is >= {threshold}, beats the 99th percentile of {nperm} random relabellings of the variable's
values and, for 20, 25, 50, 100, of {nperm} rotations of each hundred; only the periods that
divide no other kept one are listed. Constancy (all prompts, both operations) at every position:
`constant` when one value explains >= {const} of the uncentred energy, `op_constant` when one
value per operation does. `p["definitions"]` holds the exact definitions; `p["provenance"]` the
script and commit. The applet's copy is `../../ab_grids/periodicity.js` (period menu).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Periodicity labels of a collected dataset.")
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    layout = Layout.of_run(args.run_dir)
    doc = periodicity(layout.datasets)
    out = layout.datasets / PERIODICITY_DIR
    out.mkdir(exist_ok=True)
    (out / "periodicity.json").write_text(json.dumps(doc, separators=(",", ":")))
    labels_ = doc["position_labels"]
    (out / "README.md").write_text(README.format(
        inner_dir=INNER_DIR,
        positions=", ".join(f"{t} = `{x}`" for t, x in enumerate(labels_)),
        tested="; ".join(f"t = {t}: {', '.join(vs)}" for t, vs in doc["variables"].items()),
        taus=", ".join(map(str, doc["taus"])), threshold=THRESHOLD, nperm=NPERM,
        const=CONST_THRESHOLD))  # fmt: skip
    js = layout.applet_data / "periodicity.js"
    js.write_text(
        "window.PERIODICITY = " + json.dumps(applet_index(doc), separators=(",", ":")) + ";\n"
    )
    layout.applet_page.write_text(applet_page_text(layout.applet_data.name))
    for op, by_t in doc["index"].items():
        for t, by_v in by_t.items():
            print(
                op, t, {v: {tau: len(x) for tau, x in ids.items() if x} for v, ids in by_v.items()}
            )
    for t, labs in doc["constancy_index"].items():
        print("constancy", t, {lab: len(x) for lab, x in labs.items()})


if __name__ == "__main__":
    main()
