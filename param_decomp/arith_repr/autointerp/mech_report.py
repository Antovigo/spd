"""Generate report_mechanisms.md (+ mech_appendix/) from the mechanism clustering products.

    python -m param_decomp.arith_repr.autointerp.mech_report --run <run_dir> --notes <notes dir>

Narrative in `mech_report_template.md` with `{{fragment}}` placeholders; tables from
`mech/{mechanisms,codes,members,code_pairs,arrangements}.parquet`."""

import argparse
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.mech_figures import with_classes
from param_decomp.arith_repr.autointerp.mechanisms import AI
from param_decomp.arith_repr.autointerp.report_md import details, fill, md_table

POS = ["<BOS>", "a", "op", "b", "="]
OPS = ("add", "sub")
FIGS = (
    "mech_code_power",
    "mech_geometry",
    "mech_snapshots",
    "mech_drift",
    "mech_map",
    "mech_checks",
    "mech_mlp_linear",
    "mech_mlp_patch",
)
POINTS = [0, 2, 8, 24, 30, 32, 34, 36, 38, 40, 50, 64]


def point_label(t: int) -> str:
    if t == 0:
        return "embed"
    layer, s = divmod(t - 1, 2)
    return f"L{layer} {'attn' if s == 0 else 'MLP'}"


def f2(x: float | None) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


class R:
    def __init__(self, run: Path) -> None:
        mech = run / AI / "mech"
        self.m = pd.read_parquet(mech / "mechanisms.parquet")
        self.c = pd.read_parquet(mech / "codes.parquet")
        self.mem = pd.read_parquet(mech / "members.parquet")
        self.pairs = pd.read_parquet(mech / "code_pairs.parquet")
        self.arr = pd.read_parquet(mech / "arrangements.parquet")
        self.m["twin"], self.m["twin_j"] = self.twins()

    def twins(self) -> tuple[list[str], list[float]]:
        """The mechanism of the other op with the largest member overlap (same position, variable)."""
        cols = self.mem.groupby("mech")["col"].apply(set)
        twin, tj = [], []
        for _, r in self.m.iterrows():
            other = self.m[(self.m.p == r.p) & (self.m.o != r.o) & (self.m["var"] == r["var"])]
            best, bj = "", 0.0
            for mid in other.mech:
                a, b = cols[r.mech], cols[mid]
                j = len(a & b) / len(a | b)
                if j > bj:
                    best, bj = mid, j
            twin.append(best)
            tj.append(bj)
        return twin, tj

    def kind(self, r: pd.Series) -> str:
        parts = []
        codes = self.c[self.c.mech == r.mech]
        if (codes.n_members > 1).any():
            til = codes[(codes.n_members > 2) & (codes.overlap_p <= 0.05)]
            parts.append("tiling" if len(til) else "block code")
        if r.n_codes > 1:
            parts.append(f"{r.n_codes} codes of the same shape")
        if "src_pos" in r and not np.isnan(r.src_pos) and r.src_cka >= 0.5:
            parts.append(f"copy from `{POS[int(r.src_pos)]}`")
        return ", ".join(parts) or "single component"


def mech_summary(rep: R, r: pd.Series) -> str:
    twin = f"; on {OPS[1 - r.o]}: {rep_twin(r)}"
    return (
        f"<b>{r.mech}</b> `{r['var']}` @ `{POS[r.p]}` ({OPS[r.o]}) — {rep.kind(r)}; "
        f"{r.n_members} comps, {r.layers}; tells apart {r.groups}/{r.n_classes} classes "
        f"(best member {r.groups_member}){twin}"
    )


def rep_twin(r: pd.Series) -> str:
    return f"{r.twin} (member overlap {r.twin_j:.2f})" if r.twin_j > 0 else "no counterpart"


def check_table(r: pd.Series) -> str:
    rows = [
        ["classes in some member's support", f"{r.coverage:.2f}"],
        [
            "classes told apart: joint / best code / best member",
            f"{r.groups} / "
            f"{'' if np.isnan(cast(float, r.get('groups_part', np.nan))) else int(r.groups_part)} / "
            f"{r.groups_member} (of {r.n_classes})",
        ],
        ["members whose removal merges classes", f2(r.get("loo_needed", np.nan))],
        [
            "support overlap (1 = tiling) / random sets / p",
            f"{f2(r.get('overlap', np.nan))} / {f2(r.get('overlap_null', np.nan))} / "
            f"{f2(r.get('overlap_p', np.nan))}",
        ],
        ["mean CKA between its codes", f2(r.get("part_cka", np.nan))],
        ["purity of the joint write (per prompt)", f2(r.purity)],
        [
            "decoding acc. joint / best code / best member (chance)",
            f"{f2(r.acc)} / {f2(r.get('acc_part', np.nan))} / {f2(r.acc_member)} ({r.chance:.2f})",
        ],
        ["kappa (1 orthogonal, >1 constructive)", f2(r.kappa)],
        ["consumers / read jointly", f"{r.consumers} / {r.joint_consumers}"],
    ]
    return md_table(["check", "value"], rows)


def transform_table(r: pd.Series) -> str:
    rows = [
        ["input point", point_label(int(r.point_in))],
        ["CKA(arrangement before, joint write)", f2(r.cka_in)],
        ["cos with the arrangement before (>0 amplify, <0 erase)", f2(r.cos_in)],
        ["share of the write inside the old arrangement's span", f2(r.in_span)],
        ["write energy / code energy before", f2(r.write_vs_code)],
    ]
    if isinstance(r.get("out_shape"), str):
        rows += [
            ["arrangement before: shape (spectrum k:share)", f"{r.in_shape} ({r.in_spectrum})"],
            ["joint write: shape (spectrum k:share)", f"{r.out_shape} ({r.out_spectrum})"],
            ["frequencies new in the write", r.new_freqs or "none"],
            [
                "write: shift-symmetric part / dimension (PR)",
                f"{f2(r.out_circulant)} / {f2(r.out_pr)}",
            ],
        ]
    if not np.isnan(cast(float, r.get("src_pos", np.nan))):
        rows.append(["best source position (CKA)", f"`{POS[int(r.src_pos)]}` ({r.src_cka:.2f})"])
    return md_table(["arrangement", "value"], rows)


def member_table(rep: R, mid: str) -> str:
    out = []
    for cid, g in cast(pd.DataFrame, rep.mem[rep.mem.mech == mid]).groupby("code", sort=False):
        c = rep.c[rep.c.code == cid].iloc[0]
        head = (
            f"code {cid} ({c.layers}): {c.n_members} comps, tells apart {c.groups}/{c.n_classes}, "
            f"coverage {c.coverage:.2f}, overlap {f2(c.get('overlap', np.nan))}"
            f" (random {f2(c.get('overlap_null', np.nan))}, p {f2(c.get('overlap_p', np.nan))})"
        )
        rows = [
            [x.name, x.block, x.support, f"{x.r2:.2f}", f"{x.on_rate:.3f}"]
            for x in cast(Iterable[Any], g.sort_values(["layer", "support"]).itertuples())
        ]
        out.append(
            f"**{head}**\n\n"
            + md_table(
                [
                    "component",
                    "block",
                    "support (classes it moves; sign of its write)",
                    "R^2",
                    "on",
                ],
                rows,
            )
        )
    return "\n".join(out)


def mech_block(rep: R, r: pd.Series, open_: bool = False) -> str:
    body = (
        "Checks\n\n"
        + check_table(r)
        + "\nHow it transforms the arrangement\n\n"
        + transform_table(r)
        + "\n"
        + details("codes and components", member_table(rep, r.mech))
    )
    return details(mech_summary(rep, r), body, open_)


def position_catalogue(rep: R, p: int, o: int) -> str:
    mm = cast(pd.DataFrame, rep.m[(rep.m.p == p) & (rep.m.o == o)])
    n_mem = cast(pd.Series, mm.groupby("var")["n_members"].sum())
    order = n_mem.sort_values(ascending=False).index
    blocks = []
    for var in order:
        mv = cast(pd.DataFrame, mm[mm["var"] == var]).sort_values(["layer_min", "n_members"])
        inner = "\n".join(mech_block(rep, r) for _, r in mv.iterrows())
        summary = f"`{var}`: {len(mv)} mechanisms, {int(mv.n_members.sum())} components"
        blocks.append(details(summary, inner))
    return "\n".join(blocks)


def frag_orthogonality(run: Path) -> str:
    """Pairwise cos of the contributions m_c (x) U_c of writers with the same variable, vs the
    same with U_c' replaced by a random writer's direction."""
    from param_decomp.arith_repr.autointerp.mechanisms import coarsest, profile_corr

    mech = run / AI / "mech"
    writers = np.load(run / AI / "wiring.npz")["writers"]
    wrow = {int(c): j for j, c in enumerate(writers)}
    G = np.load(mech / "u_gram.npy")
    dg = np.sqrt(np.diag(G))
    rng = np.random.default_rng(0)
    rows = []
    for p in (1, 2, 3, 4):
        for o in (0, 1):
            pr = dict(np.load(mech / f"prof_{p}_{o}.npz"))
            names = [str(n) for n in pr["var_names"]]
            by: dict[str, list[int]] = {}
            for j, c in enumerate(pr["cols"]):
                if int(c) in wrow:
                    v = coarsest(pr, names, "r2_w", j)
                    if v is not None:
                        by.setdefault(v, []).append(j)
            real, null, same_u, rand_u = [], [], [], []
            for v, js in by.items():
                if len(js) < 2:
                    continue
                wts = pr[f"n.{v}"] / pr[f"n.{v}"].sum()
                C = profile_corr(pr[f"w.{v}"][js], wts)
                wr = np.array([wrow[int(pr["cols"][j])] for j in js])
                cu = np.abs(G[np.ix_(wr, wr)] / np.outer(dg[wr], dg[wr]))
                rw = rng.choice(writers.size, len(js))
                cr = np.abs(G[np.ix_(wr, rw)] / np.outer(dg[wr], dg[rw]))
                iu = np.triu_indices(len(js), 1)
                real += list((C * cu)[iu])
                null += list((C * cr)[iu])
                same = C[iu] >= 0.9
                same_u += list(cu[iu][same])
                rand_u += list(cr[iu][same])
            real_a, null_a = np.array(real), np.array(null)
            su, ru = np.array(same_u), np.array(rand_u)
            rows.append(
                [
                    f"`{POS[p]}`",
                    OPS[o],
                    real_a.size,
                    f"{np.quantile(real_a, 0.5):.3f}",
                    f"{np.quantile(real_a, 0.99):.3f}",
                    f"{np.quantile(null_a, 0.99):.3f}",
                    f"{(real_a > 0.3).mean():.4f}",
                    su.size,
                    f"{np.median(su):.3f} / {np.quantile(su, 0.99):.3f}" if su.size else "",
                    f"{np.median(ru):.3f} / {np.quantile(ru, 0.99):.3f}" if ru.size else "",
                ]
            )
    return md_table(
        [
            "position",
            "op",
            "writer pairs, same variable",
            "median abs cos",
            "q99",
            "q99 with random U",
            "share > 0.3",
            "pairs with the same pattern (abs corr >= 0.9)",
            "abs cos(U) median / q99",
            "random U: median / q99",
        ],
        rows,
    )


def frag_counts(rep: R) -> str:
    rows = []
    for p in (1, 2, 3, 4):
        for o in (0, 1):
            mm = rep.m[(rep.m.p == p) & (rep.m.o == o)]
            cc = rep.c[(rep.c.p == p) & (rep.c.o == o)]
            til = cc[(cc.n_members > 2) & (cc.overlap_p <= 0.05)]
            rows.append(
                [
                    f"`{POS[p]}`",
                    OPS[o],
                    int(mm.n_members.sum()),
                    len(cc),
                    int((cc.n_members > 1).sum()),
                    len(til),
                    len(mm),
                    int((mm.n_codes > 1).sum()),
                    f"{(mm.twin_j >= 0.5).mean():.2f}",
                ]
            )
    return md_table(
        [
            "position",
            "op",
            "writers with a variable",
            "codes",
            "multi-comp codes",
            "tilings (p <= 0.05)",
            "mechanisms",
            "with > 1 code",
            "share with a twin on the other op (overlap >= 0.5)",
        ],
        rows,
    )


def frag_pairs(rep: R) -> str:
    pairs = with_classes(rep.pairs, rep.m)
    rows = []
    for lo, hi, lab in (
        (2, 3, "2-3 classes"),
        (4, 11, "4-11 classes"),
        (12, 1000, ">= 12 classes"),
    ):
        s = cast(pd.DataFrame, pairs[pairs["nc"].between(lo, hi)])
        for name, col in (("code pairs", "cka"), ("one code's classes permuted", "cka_null")):
            x = cast(pd.Series, s[col])
            rows.append(
                [
                    lab,
                    name,
                    len(x),
                    f"{x.quantile(0.5):.3f}",
                    f"{x.quantile(0.9):.3f}",
                    f"{x.quantile(0.99):.3f}",
                    f"{(x >= 0.7).mean():.4f}",
                ]
            )
    return md_table(["variable", "", "pairs", "median CKA", "q90", "q99", "share >= 0.7"], rows)


def frag_checks(rep: R) -> str:
    mm = rep.m[rep.m.n_members > 1]
    cc = rep.c[rep.c.n_members > 2]
    rows = [
        ["mechanisms with > 1 component", len(mm)],
        [
            "joint write tells apart more classes than its best member",
            f"{(mm.groups > mm.groups_member).mean():.2f}",
        ],
        [
            "joint decoding accuracy > best member + 0.05",
            f"{(mm.acc > mm.acc_member + 0.05).mean():.2f}",
        ],
        ["purity of the joint write >= 0.8", f"{(mm.purity >= 0.8).mean():.2f}"],
        [
            "kappa in [0.8, 1.25] (writes near-orthogonal)",
            f"{mm.kappa.between(0.8, 1.25).mean():.2f}",
        ],
        ["kappa > 1.25 (constructive)", f"{(mm.kappa > 1.25).mean():.2f}"],
        ["kappa < 0.8 (cancelling)", f"{(mm.kappa < 0.8).mean():.2f}"],
        ["read jointly by >= 1 consumer", f"{(mm.joint_consumers >= 1).mean():.2f}"],
        ["codes with >= 3 comps", len(cc)],
        ["... tiling better than random sets (p <= 0.05)", f"{(cc.overlap_p <= 0.05).mean():.2f}"],
        ["... every member needed (loo_needed = 1)", f"{(cc.loo_needed >= 0.999).mean():.2f}"],
    ]
    return md_table(["check", "value"], rows)


ARR_SPECS = [
    (1, "a", 10),
    (3, "b", 10),
    (4, "a", 10),
    (4, "b", 10),
    (4, "res", 10),
    (4, "res_int", 10),
    (4, "res", 100),
    (4, "res_int", 100),
]


def frag_arrangements(rep: R) -> str:
    out = []
    for p, q, tau in ARR_SPECS:
        for o in (0, 1):
            s = rep.arr[
                (rep.arr.p == p) & (rep.arr.o == o) & (rep.arr.q == q) & (rep.arr.tau == tau)
            ]
            s = s.set_index("point")
            rows = [
                [
                    point_label(t),
                    f"{s.loc[t, 'share']:.3f}",
                    s.loc[t, "shape"],
                    s.loc[t, "spectrum"],
                    f2(s.loc[t, "circulant"]),
                    f2(s.loc[t, "pr"]),
                    f2(s.loc[t, "ordered"]),
                    f2(s.loc[t, "cka_prev"]),
                    f2(s.loc[t, "in_span_prev"]),
                ]
                for t in POINTS
                if t in s.index
            ]
            steps = s.dropna(subset=["in_span_prev"])
            att = steps[steps.index % 2 == 1]["in_span_prev"].mean()
            mlp = steps[(steps.index % 2 == 0) & (steps.index > 0)]["in_span_prev"].mean()
            out.append(
                details(
                    f"`{q} mod {tau}` at `{POS[p]}` ({OPS[o]}) — mean share kept in place per step: "
                    f"attention {att:.2f}, MLP {mlp:.2f}",
                    md_table(
                        [
                            "point",
                            "code share",
                            "shape",
                            "spectrum (k:share)",
                            "shift-sym.",
                            "PR",
                            "ordered",
                            "CKA prev",
                            "in prev. span",
                        ],
                        rows,
                    ),
                )
            )
    return "\n".join(out)


def pick(rep: R, **kw: object) -> pd.DataFrame:
    mm = rep.m
    for k, v in kw.items():
        mm = cast(pd.DataFrame, mm[mm[k] == v])
    return mm


def frag_highlights(rep: R) -> dict[str, str]:
    f: dict[str, str] = {}
    a10 = pick(rep, p=1, o=0, var="a%10").sort_values("layer_min")
    f["hl_a10"] = "\n".join(mech_block(rep, r) for _, r in a10.iterrows())
    b10 = pick(rep, p=3, o=0, var="b%10").sort_values("layer_min")
    f["hl_b10"] = "\n".join(mech_block(rep, r) for _, r in b10.iterrows())
    eq = cast(pd.DataFrame, rep.m[(rep.m.p == 4) & (rep.m.o == 0)])
    copies = cast(pd.DataFrame, eq[(eq.src_cka >= 0.5) & (eq.n_members >= 2)])
    copies = copies.sort_values("layer_min")
    f["hl_copies"] = "\n".join(mech_block(rep, r) for _, r in copies.iterrows())
    pairs = cast(
        pd.DataFrame,
        eq[eq["var"].isin(["units(a,b)", "tens(a,b)", "carry", "cmp(a,b)", "res>=100"])],
    )
    pairs = cast(pd.DataFrame, pairs[pairs.layer_max <= 20]).sort_values("layer_min")
    f["hl_pairs"] = "\n".join(mech_block(rep, r) for _, r in pairs.iterrows())
    res = cast(pd.DataFrame, eq[eq["var"].str.startswith("res")]).sort_values(["var", "layer_min"])
    rows = [
        [
            r.mech,
            f"`{r['var']}`",
            r.layers,
            r.n_members,
            rep.kind(r),
            f"{r.groups}/{r.n_classes}",
            f2(r.acc),
            f2(r.cka_in),
            f2(r.in_span),
            r.get("out_shape") or "",
            r.consumers,
            rep_twin(r),
        ]
        for _, r in res.iterrows()
    ]
    f["hl_result"] = md_table(
        [
            "mech",
            "variable",
            "layers",
            "comps",
            "kind",
            "classes apart",
            "acc",
            "CKA in",
            "in span",
            "shape",
            "consumers",
            "sub twin",
        ],
        rows,
    )
    return f


def frag_mlp_linear(run: Path) -> dict[str, str]:
    """Per-layer split of each MLP's write (mlp_linearity), switching-gate structure, and the
    linearised forwards (mlp_patch)."""
    mech = run / AI / "mech"
    d = pd.read_parquet(mech / "mlp_lin.parquet")
    fl = pd.read_parquet(mech / "mlp_flips.parquet")
    pt = pd.read_parquet(mech / "mlp_patch.parquet")
    f = {}
    for key, (p, o, v) in {
        "mlp_lin_a": ("a", "both", "a%10"),
        "mlp_lin_a100": ("a", "both", "a%100"),
        "mlp_lin_b": ("b", "add", "b%10"),
        "mlp_lin_bp": ("b", "add", "prompt"),
        "mlp_lin_res": ("=", "add", "res%10"),
    }.items():
        x = cast(pd.DataFrame, d[(d.pos == p) & (d.op == o) & (d["var"] == v)])
        x = x.sort_values("layer")
        fx = fl[(fl.pos == p) & (fl.op == o)].set_index("layer") if p in ("a", "b") else None
        rows = []
        for _, r in x.iterrows():
            row = [
                f"L{r.layer}",
                f2(r.E / r.E_in),
                f2(r.r2_lin),
                f2(r.r2_noflip),
                f2(r.share_up),
                f2(r.share_gate),
                f2(r.share_silu),
                f2(r.share_prod),
                "" if v == "prompt" else int(r.n90_silu),
            ]
            if fx is not None:
                q = p + "%10"
                row += [int(fx.loc[r.layer, "n_cross"]), f2(fx.loc[r.layer, f"eta_w_{q}"])]
            rows.append(row)
        head = [
            "layer",
            "write / stream",
            "R² one linear map",
            "R² no switching",
            "up",
            "gate read",
            "silu bend",
            "gate × up",
            "neurons for 90 % of the bend",
        ]
        if fx is not None:
            head += ["gates crossing 0", "switch between mod-10 classes (weighted)"]
        f[key] = md_table(head, rows)
    rows = []
    x = cast(pd.DataFrame, pt[pt.op == "both"])
    cleans = []
    for n in sorted(x.n.unique(), reverse=True):
        c = pt[(pt.kind == "clean") & (pt.run == x[(x.kind == "clean") & (x.n == n)].run.iloc[0])]
        acc = dict(zip(c.op, c.acc, strict=True))
        cleans.append(
            f"{n} prompts: {acc['both']:.3f} (add {acc['add']:.3f} / sub {acc['sub']:.3f})"
        )
    for _, r in cast(pd.DataFrame, x[x.kind != "clean"]).iterrows():
        add = pt[(pt.run == r.run) & (pt.variant == r.variant) & (pt.op == "add")].iloc[0]
        sub = pt[(pt.run == r.run) & (pt.variant == r.variant) & (pt.op == "sub")].iloc[0]
        rows.append(
            [
                {"lin": "one linear map", "sl": "no switching", "mean": "write removed"}[r.kind],
                f"`{r.positions}`",
                "all" if r.layers == "all" else f"L{r.layers}",
                f"{r.kl:.3f}",
                f"{r.agree:.3f}",
                f"{r.acc:.3f}",
                f"{add.acc:.3f} / {sub.acc:.3f}",
                r.n,
            ]
        )
    f["mlp_patch_table"] = (
        "Clean accuracy (float32 forward), per sample: "
        + "; ".join(cleans)
        + ".\n\n"
        + md_table(
            [
                "replacement",
                "positions",
                "MLPs",
                "KL (nats)",
                "same top token",
                "accuracy",
                "accuracy add / sub",
                "prompts",
            ],
            rows,
        )
    )
    return f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    rep = R(args.run)
    cc = rep.c[rep.c.n_members > 2]
    frags = (
        {
            "counts": frag_counts(rep),
            "pairs": frag_pairs(rep),
            "checks": frag_checks(rep),
            "arrangements": frag_arrangements(rep),
            "orthogonality": frag_orthogonality(args.run),
            "n_writers": str(int(rep.m.n_members.sum())),
            "n_codes": str(len(rep.c)),
            "n_mechs": str(len(rep.m)),
            "tiling_share": f"{(cc.overlap_p <= 0.05).mean():.0%} ({int((cc.overlap_p <= 0.05).sum())} of {len(cc)})",
        }
        | frag_highlights(rep)
        | frag_mlp_linear(args.run)
    )
    app = args.notes / "mech_appendix"
    app.mkdir(exist_ok=True)
    for p in (1, 2, 3, 4):
        links = []
        for o in (0, 1):
            name = f"M{p}_{POS[p].replace('=', 'equals')}_{OPS[o]}.md"
            (app / name).write_text(
                "[← back to the report](../report_mechanisms.md)\n\n"
                f"# Every mechanism at `{POS[p]}`, {OPS[o]}\n\n" + position_catalogue(rep, p, o)
            )
            links.append(f"[{OPS[o]}](mech_appendix/{name})")
        frags[f"app_{p}"] = " · ".join(links)
    figdir = args.notes / "figures_auto_interp"
    for fname in FIGS:
        shutil.copy(args.run / AI / "figs" / f"{fname}.png", figdir / f"{fname}.png")
    template = (Path(__file__).parent / "mech_report_template.md").read_text()
    (args.notes / "report_mechanisms.md").write_text(fill(template, frags))
    print("wrote", args.notes / "report_mechanisms.md")


if __name__ == "__main__":
    main()
