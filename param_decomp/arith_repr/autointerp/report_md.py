"""Build report_auto_interp.md and its appendices from the analysis products.

    python -m param_decomp.arith_repr.autointerp.report_md --run <run_dir> --notes <notes dir>

The narrative lives in `report_template.md` (next to this file); every `{{name}}` in it is
replaced by the fragment `name` generated here (component lists, evidence tables). Appendices
(complete catalogues) are written to `<notes>/auto_interp_appendix/`."""

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd

FILTER = "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
POS = ["&lt;BOS&gt;", "a", "op", "b", "="]
OPS = ("add", "sub")
QS = ("a", "b", "res")
PERIODS = (100, 50, 25, 20, 10, 5, 4, 2)
K_OF = {t: [k for k in range(1, 51) if 100 // np.gcd(k, 100) == t] for t in PERIODS}
SHARE_MIN = 0.02
POWER_MIN = 1e-3


class OnsetRow(NamedTuple):
    """One row of `onsets_all.parquet` (see `onset_tables.py`), typed for attribute access."""

    col: int
    pos: int
    op: str
    on_rate: float
    label: str
    r2: float
    desc: str


def details(summary: str, body: str, open_: bool = False) -> str:
    return f"<details{' open' if open_ else ''}><summary>{summary}</summary>\n\n{body.strip()}\n\n</details>\n"


def md_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out) + "\n"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%" if abs(x) >= 0.005 else "0"


class Data:
    def __init__(self, run: Path) -> None:
        ai = run / "analysis/arith_repr/autointerp"
        self.ai = ai
        self.df = pd.read_parquet(ai / "catalogue.parquet")
        ons = pd.read_parquet(ai / "onsets_all.parquet")
        self.ons = {
            (int(c), int(p), str(o)): OnsetRow(
                int(c), int(p), str(o), float(rate), str(lab), float(r2), str(desc)
            )
            for c, p, o, rate, lab, r2, desc in zip(
                ons["col"],
                ons["pos"],
                ons["op"],
                ons["on_rate"],
                ons["label"],
                ons["r2"],
                ons["desc"],
                strict=True,
            )
        }
        self.ons_df = ons
        cc = np.load(ai / "comp_codes.npz")
        self.share = cc["share"].astype(np.float32)
        self.power = cc["power"]
        self.ca = np.load(ai / "code_attrib.npz")
        self.ca_cases = [tuple(json.loads(c)) for c in self.ca["cases"]]
        wz = np.load(ai / "wiring.npz")
        self.wz = wz
        self.head = wz["head"]
        self.dla = wz["dla"]
        self.tuning = np.load(ai / "tuning.npz")
        self.profiles = np.load(ai / "profiles.npz")
        self.att = np.load(ai / "attn_patterns.npy", mmap_mode="r")
        self.opp = np.load(ai / "op_phase.npz")
        ix = np.load(run / FILTER / "index.npz")
        self.a, self.b, self.op = ix["a"], ix["b"], ix["op"]
        self.kind = self.df.kind.values
        self.layer = self.df.layer.values
        self.cidx = self.df.cidx.values
        is_o = self.kind == "o"
        self.point = np.where(is_o, 2 * self.layer + 1, 2 * self.layer + 2)
        self._attn_mean: dict[tuple[int, int, int], np.ndarray] = {}

    # ---------------------------------------------------------------- naming
    def hd(self, c: int) -> int:
        k = self.kind[c]
        return int(self.head[c, :8].argmax() if k in ("k", "v") else self.head[c].argmax())

    def name(self, c: int) -> str:
        k = self.kind[c]
        s = f"L{self.layer[c]} {k} c{self.cidx[c]}"
        if k in ("o", "q"):
            s += f" (H{self.hd(c)})"
        elif k in ("k", "v"):
            s += f" (kv{self.hd(c)})"
        return s

    def on(self, c: int, pos: int, op: str) -> float:
        return float(self.df[f"on_{op}_p{pos}"].values[c])

    def onset(self, c: int, pos: int, op: str) -> str:
        if pos == 0:
            return f"at &lt;BOS&gt; (constant), on {pct(self.on(c, 0, op))}"
        r = self.ons.get((c, pos, op))
        if r is None:
            return f"off (on {pct(self.on(c, pos, op))})"
        if r.label in ("always", "unexplained"):
            return r.label + (f" (best R2 {r.r2:.2f})" if r.label == "unexplained" else "")
        return (
            f"**{r.label}** (R2 {r.r2:.2f}): {r.desc}"
            if r.desc
            else f"**{r.label}** (R2 {r.r2:.2f})"
        )

    def label(self, c: int, pos: int, op: str) -> str:
        r = self.ons.get((c, pos, op))
        return "off" if r is None else str(r.label)

    def label_main(self, c: int, pos: int) -> str:
        """The addition label, or `sub only: <sub label>` for a component off on addition."""
        la, ls = self.label(c, pos, "add"), self.label(c, pos, "sub")
        if la == "off":
            return "rarely on (< 0.5 % of prompts)" if ls == "off" else f"sub only: {ls}"
        return la

    # ---------------------------------------------------------------- codes
    def code(self, c: int, pos: int, op: str, q: str) -> dict[int, float]:
        """Power-weighted share of each period's code supplied by writer c, at the point after it."""
        o, qi = OPS.index(op), QS.index(q)
        sh = self.share[c, pos, o, qi]
        pw = self.power[self.point[c], pos, o, qi]
        out = {}
        for t, ks in K_OF.items():
            p = pw[ks]
            if p.sum() < POWER_MIN:
                continue
            out[t] = float((sh[ks] * p).sum() / p.sum())
        return out

    def writes(self, c: int, pos: int, op: str, qs: tuple[str, ...] = QS) -> str:
        if self.kind[c] not in ("o", "down"):
            return "(reads)"
        parts = []
        for q in qs:
            if pos < 3 and q != "a":
                continue
            d = {t: v for t, v in self.code(c, pos, op, q).items() if abs(v) >= SHARE_MIN}
            if d:
                parts.append(f"{q}: " + ", ".join(f"mod{t} {v:+.0%}" for t, v in d.items()))
        return "; ".join(parts) if parts else "-"

    def ca_case(self, case: tuple[Any, ...]) -> np.ndarray:
        return self.ca["share"][self.ca_cases.index(case)]

    # ---------------------------------------------------------------- attention
    def attn(self, li: int, h: int, q: int) -> np.ndarray:
        key = (li, h, q)
        if key not in self._attn_mean:
            A = np.asarray(self.att[li, :, h, q, : q + 1], np.float32)
            self._attn_mean[key] = np.stack([A[self.op == o].mean(0) for o in (0, 1)])
        return self._attn_mean[key]

    # ---------------------------------------------------------------- output
    def dla_str(self, c: int, n: int = 5) -> str:
        d = self.dla[c, :201]
        d = d - d.mean()
        top = np.argsort(-d)[:n]
        return ", ".join(f"{t}" for t in sorted(top.tolist()))

    def dla_corr(self, c: int, op: str) -> float:
        prof = self.profiles["w_res"][4, OPS.index(op), c]
        if op == "add":
            p, t = prof, self.dla[c, 2:201]
        else:
            p, t = prof[99:], self.dla[c, 0:100]
        if p.std() == 0 or t.std() == 0:
            return float("nan")
        return float(np.corrcoef(p, t)[0, 1])

    # ---------------------------------------------------------------- tables
    def comp_rows(
        self, cols: Iterable[int], pos: int, ops: tuple[str, ...] = OPS, writes: bool = True,
        extra: Callable[[int], list[Any]] | None = None,
    ) -> list[list[Any]]:  # fmt: skip
        rows = []
        for c in cols:
            c = int(c)
            row: list[Any] = [self.name(c), " / ".join(pct(self.on(c, pos, op)) for op in ops)]
            descs = [self.onset(c, pos, op) for op in ops]
            if len(ops) == 2 and descs[0] == descs[1]:
                descs[1] = "same"
            row += descs
            if writes:
                ws = [self.writes(c, pos, op) for op in ops]
                if len(ops) == 2 and ws[0] == ws[1] and ws[0] in ("(reads)", "-"):
                    ws[1] = "same"
                row += ws
            if extra is not None:
                row += extra(c)
            rows.append(row)
        return rows

    def comp_table(
        self, cols: Iterable[int], pos: int, ops: tuple[str, ...] = OPS, writes: bool = True,
        extra_headers: list[str] | None = None, extra: Callable[[int], list[Any]] | None = None,
    ) -> str:  # fmt: skip
        headers = ["component", "on (" + "/".join(ops) + ")"] + [f"on-set ({op})" for op in ops]
        if writes:
            headers += [f"writes ({op})" for op in ops]
        headers += extra_headers or []
        return md_table(headers, self.comp_rows(cols, pos, ops, writes, extra))


LEGEND = """**How to read the component tables.** `component` = layer, site kind, component index
(o/q components: the head holding most of their weight; k/v: the kv head). `on` = fraction of
the operation's 10,000 prompts on which the component's CI exceeds 0.01 at that position.
`on-set` = the coarsest function of (a, b) visible at the position that explains the on/off
pattern (adjusted R² within 90 % of the best function in a fixed list; `always` = on for > 95 %
of prompts, `unexplained` = no listed function reaches R² 0.5), then the on classes spelled out
completely (`res mod 100 in {1..9, 75}` = on when the result mod 100 is 1-9 or 75; `(tens) res in
{20..29}`; for digit-pair labels, the cells of the units- or tens-digit plane that are on, rows
with the same cells merged). `[coarser: …]` = a coarser period that already explains ≥ 80 % of
the on-rate profile along that quantity. `res` = a+b on addition, a−b on subtraction.
`writes` (o/down only) = the share of each Fourier code of a, b or res that the component's
write supplies, measured at the first read point after it (its contribution's projection on
that code over the code's squared norm, power-weighted over the harmonics of each period; codes
carrying < 0.1 % of the read input's variance and shares below 2 % are omitted; negative = the
component opposes the code). `(reads)` = a q/k/v/gate/up component (it writes into a head or
the MLP hidden layer, not the residual)."""


def activity(d: Data, c: int, pos: int) -> float:
    df = d.df
    return float(
        np.mean([df[f"on_{op}_p{pos}"].values[c] * df[f"wstd_{op}_p{pos}"].values[c] for op in OPS])
    )


def top_by_activity(d: Data, cols: Iterable[int], pos: int, n: int) -> list[int]:
    cols = [int(c) for c in cols]
    return sorted(
        sorted(cols, key=lambda c: -activity(d, c, pos))[:n],
        key=lambda c: (d.layer[c], d.kind[c], d.cidx[c]),
    )


def link(fname: str, what: str) -> str:
    return f"Complete list: [{what}](auto_interp_appendix/{fname})."


def period_share_table(d: Data, case: tuple[Any, ...]) -> tuple[str, np.ndarray]:
    """Per-period share of a code_attrib case, by writer group (layer x attn/mlp) + embedding."""
    i = d.ca_cases.index(case)
    sh, zp = d.ca["share"][i], d.ca["zpow"][i]
    W = d.ca["writers"]
    grp = [f"L{d.layer[c]} {'attn' if d.kind[c] == 'o' else 'MLP'}" for c in W]
    out_rows = []
    for t, ks in K_OF.items():
        w = zp[ks] / zp[ks].sum()
        s = sh[:-1][:, ks] @ w
        by = cast(pd.Series, pd.Series(s).groupby(grp).sum())
        by["embedding"] = float(sh[-1][ks] @ w)
        top = cast(pd.Series, by[by.abs() >= SHARE_MIN]).sort_values(ascending=False)
        out_rows.append(
            [f"mod {t}", f"{by.sum():.2f}", ", ".join(f"{k} {v:+.2f}" for k, v in top.items())]
        )
    return md_table(["period", "total explained", "groups supplying ≥ 2 %"], out_rows), sh


def short_onset(d: Data, c: int, pos: int, op: str) -> str:
    r = d.ons.get((c, pos, op))
    if r is None:
        return "off"
    if not r.desc:
        return str(r.label)
    return str(r.desc).split(" [coarser")[0]


def compact_writers(d: Data, case: tuple[Any, ...], pos: int, op: str, n: int = 12) -> str:
    """One bullet per period: the writers with |share| >= 2 %, largest first (at most n)."""
    i = d.ca_cases.index(case)
    sh, zp = d.ca["share"][i], d.ca["zpow"][i]
    W = d.ca["writers"]
    lines = []
    for t, ks in K_OF.items():
        s = sh[:-1][:, ks] @ (zp[ks] / zp[ks].sum())
        sel = np.flatnonzero(np.abs(s) >= SHARE_MIN)
        sel = sel[np.argsort(-s[sel])]
        if sel.size == 0:
            continue
        items = [
            f"{d.name(int(W[j]))} {s[j]:+.0%} ({short_onset(d, int(W[j]), pos, op)})"
            for j in sel[:n]
        ]
        more = f"; … {sel.size - n} more" if sel.size > n else ""
        lines.append(f"- **mod {t}** ({sel.size} writers): " + "; ".join(items) + more)
    return "\n".join(lines)


def writer_lists(d: Data, case: tuple[Any, ...], pos: int, op: str) -> str:
    i = d.ca_cases.index(case)
    sh, zp = d.ca["share"][i], d.ca["zpow"][i]
    W = d.ca["writers"]
    body = []
    for t, ks in K_OF.items():
        s = sh[:-1][:, ks] @ (zp[ks] / zp[ks].sum())
        sel = np.flatnonzero(np.abs(s) >= SHARE_MIN)
        sel = sel[np.argsort(-s[sel])]
        if sel.size == 0:
            continue
        rows = [
            [
                d.name(int(W[i])),
                f"{s[i]:+.3f}",
                pct(d.on(int(W[i]), pos, op)),
                d.onset(int(W[i]), pos, op),
            ]
            for i in sel
        ]
        body.append(
            details(
                f"period {t}: {sel.size} writers with |share| ≥ 2 %",
                md_table(["component", "share", "on", "on-set"], rows),
            )
        )
    return "\n".join(body)


# =============================================================================== fragments


def frag_pos1(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    df = d.df
    cols = df.index[df.main_pos == 1].values

    def cls(c: int) -> str:
        op = "add" if d.on(int(c), 1, "add") >= d.on(int(c), 1, "sub") else "sub"
        r = d.ons.get((int(c), 1, op))
        if r is None:
            return "rarely on (< 0.5 %)"
        lab = r.label
        if lab == "a":
            body = r.desc.split("{")[1].split("}")[0] if "{" in r.desc else ""
            if "," not in body and ".." not in body:
                return "single value"
            if "," not in body:
                return "window (one run)"
            return "scattered values"
        if lab == "a//10":
            return "tens-digit set"
        if lab.startswith("a%"):
            return f"residue mod {lab[2:]}"
        return lab

    t = pd.DataFrame({"col": cols, "layer": df.layer.values[cols], "cls": [cls(c) for c in cols]})
    t["band"] = pd.cut(
        t.layer,
        [-1, 1, 4, 9, 14, 19, 24, 31],
        labels=["L0-1", "L2-4", "L5-9", "L10-14", "L15-19", "L20-24", "L25-31"],
    )
    ct = pd.crosstab(t.cls, t.band)
    ct["total"] = ct.sum(1)
    ct = ct.sort_values("total", ascending=False)
    f["pos1_classes"] = md_table(
        ["on-set class (operation where it is on more)"] + list(ct.columns),
        [[i] + list(r) for i, r in ct.iterrows()],
    )
    # causality: on-set depends on later tokens?
    dep = []
    for c in cols:
        ra, rs = d.on(c, 1, "add"), d.on(c, 1, "sub")
        la, ls = d.ons.get((int(c), 1, "add")), d.ons.get((int(c), 1, "sub"))
        if abs(ra - rs) > 0.02 or (la is not None and ls is not None and la.desc != ls.desc):
            dep.append(int(c))
    f["pos1_noncausal_n"] = str(len(dep))
    kinds = pd.Series([str(d.kind[c]) for c in dep]).value_counts()
    f["pos1_noncausal_kinds"] = ", ".join(f"{k} {v}" for k, v in kinds.items())
    dep_sorted = sorted(dep, key=lambda c: -abs(d.on(c, 1, "add") - d.on(c, 1, "sub")))
    f["pos1_noncausal"] = details(
        f"The 25 (of {len(dep)}) `a`-position components whose on-rate differs most between + and − prompts",
        d.comp_table(dep_sorted[:25], 1, writes=False)
        + "\n"
        + link("A1_a_token.md#on-set-depends-on-the-operator", "appendix A1"),
    )
    f["_A1_noncausal"] = "## On-set depends on the operator\n\n" + details(
        f"all {len(dep)}", d.comp_table(dep_sorted, 1, writes=False)
    )
    by_cls = []
    for name, g in t.groupby("cls"):
        g = g.sort_values("layer")
        by_cls.append(details(f"{name}: {len(g)} components", d.comp_table(g.col.values, 1)))
    f["pos1_by_class_note"] = (
        "The complete per-class lists are in [appendix A1](auto_interp_appendix/A1_a_token.md)."
    )
    f["_A1_by_class"] = "## By on-set class\n\n" + "\n".join(by_cls)
    return f


def frag_code_builders(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    for key, pos, q, cases in (
        ("a_builders", 1, "a", [c for c in d.ca_cases if c[1] == 1]),
        ("b_builders", 3, "b", [c for c in d.ca_cases if c[1] == 3]),
    ):
        body, full = [], []
        for case in cases:
            tab, _ = period_share_table(d, case)
            title = f"`{case[0]}` at `{POS[pos]}` (addition prompts): who writes the {q} code"
            body.append(details(title, tab + "\n" + compact_writers(d, case, pos, "add")))
            full.append(details(title, tab + "\n" + writer_lists(d, case, pos, "add")))
        f[key] = (
            "\n".join(body)
            + "\n"
            + link(
                "S1_code_builders.md",
                f"every component supplying ≥ 2 % of a {q} code, per read point and period",
            )
        )
        f["_S1_" + key] = f"## The {q} code at the `{POS[pos]}` token\n\n" + "\n".join(full)
    return f


def frag_shared_operands(d: Data) -> dict[str, str]:
    """Components on at both operand tokens with the same value set (a at `a`, b at `b`)."""
    rows = []
    for c in d.df.index:
        c = int(c)
        r1, r3 = d.ons.get((c, 1, "add")), d.ons.get((c, 3, "add"))
        if r1 is None or r3 is None or not r1.desc or not r3.desc:
            continue
        s1 = str(r1.desc).split(" [coarser")[0].replace("a", "x")
        s3 = str(r3.desc).split(" [coarser")[0].replace("b", "x")
        if s1 == s3 and r1.label.replace("a", "x") == r3.label.replace("b", "x"):
            rows.append(c)
    rows.sort(key=lambda c: (d.layer[c], d.kind[c], d.cidx[c]))
    tab = md_table(
        [
            "component",
            "on at a / at b (add)",
            "on-set at `a`",
            "on-set at `b`",
            "writes at `a`",
            "writes at `b`",
        ],
        [
            [
                d.name(c),
                f"{pct(d.on(c, 1, 'add'))} / {pct(d.on(c, 3, 'add'))}",
                short_onset(d, c, 1, "add"),
                short_onset(d, c, 3, "add"),
                d.writes(c, 1, "add"),
                d.writes(c, 3, "add"),
            ]
            for c in rows
        ],
    )
    return {
        "shared_operands_n": str(len(rows)),
        "shared_operands": details(
            f"All {len(rows)} components with the same value set at both operand tokens", tab
        ),
    }


def frag_pos3(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    df = d.df
    cols = df.index[df.main_pos == 3].values
    fam: dict[str, list[int]] = {}
    for c in cols:
        la, ls = d.label(int(c), 3, "add"), d.label(int(c), 3, "sub")
        ra, rs = d.on(int(c), 3, "add"), d.on(int(c), 3, "sub")
        if abs(ra - rs) > 0.8:
            key = "op flag (on for one operation only)"
        elif "cmp" in (la + ls):
            key = "comparator cmp(a,b) (a<b / a=b / a>b)"
        elif la.startswith("a") or ls.startswith("a"):
            key = "function of a only (a copied into the b token)"
        elif la.startswith("b") or ls.startswith("b"):
            key = "function of b only"
        elif "(a,b)" in la + ls:
            key = "digit-pair conjunction (units or tens of a and b)"
        elif la.startswith("res") or ls.startswith("res"):
            key = "function of the result"
        else:
            key = f"{la} / {ls}"
        fam.setdefault(key, []).append(int(c))
    order = sorted(fam, key=lambda k: -len(fam[k]))
    f["pos3_counts"] = md_table(
        ["family (main position b)", "components"], [[k, len(fam[k])] for k in order]
    )
    f["_A3_families"] = "\n".join(
        details(f"{k}: {len(fam[k])}", d.comp_table(sorted(fam[k], key=lambda c: d.layer[c]), 3))
        for k in order
    )
    for key, fname in (
        ("comparator cmp(a,b) (a<b / a=b / a>b)", "pos3_cmp"),
        ("function of a only (a copied into the b token)", "pos3_copies"),
        ("op flag (on for one operation only)", "pos3_opflags"),
    ):
        cs = sorted(fam.get(key, []), key=lambda c: d.layer[c])
        f[fname] = details(f"{key}: all {len(cs)} components", d.comp_table(cs, 3))
    return f


def head_block(d: Data, li: int, h: int, open_: bool = False) -> str:
    df = d.df
    q_pos = [p for p in (1, 2, 3, 4)]
    pat_rows = []
    for q in q_pos:
        A = d.attn(li, h, q)
        pat_rows.append(
            [f"{POS[q]}"]
            + [f"{A[0, k]:.2f} / {A[1, k]:.2f}" for k in range(q + 1)]
            + [""] * (4 - q)
        )
    pat = md_table(["query \\ key (add / sub)"] + POS, pat_rows)
    parts = [pat]
    for kind in ("o", "v", "q", "k"):
        if kind in ("o", "q"):
            cs = df.index[(df.kind == kind) & (df.layer == li) & (d.head.argmax(1) == h)].values
        else:
            cs = df.index[
                (df.kind == kind) & (df.layer == li) & (d.head[:, :8].argmax(1) == h // 4)
            ].values
        if cs.size == 0:
            continue
        rows_by_pos: dict[int, list[int]] = {}
        for c in cs:
            rows_by_pos.setdefault(int(df.main_pos.values[c]), []).append(int(c))
        for p, cc in sorted(rows_by_pos.items()):
            title = f"{len(cc)} {kind} components, main position `{POS[p]}`" + (
                " (kv head shared by 4 query heads)" if kind in ("k", "v") else ""
            )
            parts.append(details(title, d.comp_table(cc, p, writes=(kind == "o")), open_=False))
    n_o = int(((df.kind == "o") & (df.layer == li) & (d.head.argmax(1) == h)).sum())
    return details(f"L{li}H{h} ({n_o} alive o components)", "\n".join(parts), open_)


KEY_HEADS = [
    (15, 13),
    (16, 21),
    (0, 23),
    (1, 6),
    (2, 2),
    (18, 30),
    (13, 7),
    (20, 2),
    (16, 22),
    (1, 24),
    (5, 22),
    (17, 9),
    (24, 10),
]


def frag_heads(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    for li, h in KEY_HEADS:
        f[f"head_{li}_{h}"] = head_block(d, li, h)
    # op-dependent attention at '=' and at b
    rows = []
    df = d.df
    heads = sorted({(int(d.layer[c]), d.hd(c)) for c in df.index[df.kind == "o"]})
    for li, h in heads:
        for q in (3, 4):
            A = d.attn(li, h, q)
            diff = A[1] - A[0]
            if np.abs(diff).max() >= 0.15:
                rows.append(
                    [f"L{li}H{h}", POS[q]]
                    + [f"{A[0, k]:.2f} → {A[1, k]:.2f}" for k in range(q + 1)]
                    + [""] * (4 - q)
                )
    f["op_attention"] = md_table(["head", "query", *[f"key {p} (add → sub)" for p in POS]], rows)
    f["_B_heads"] = "\n".join(head_block(d, li, h) for li, h in heads)
    return f


def eq_group_table(d: Data, cols: Iterable[int]) -> str:
    return d.comp_table(sorted(cols, key=lambda c: (d.layer[c], d.kind[c], d.cidx[c])), 4)


def frag_equals(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    df = d.df
    eq = df.index[df.main_pos == 4].values
    lab_add = {int(c): d.label_main(int(c), 4) for c in eq}
    # additive regime L10-16
    add_cols = [
        c for c in eq if d.layer[c] <= 16 and lab_add[c].split("%")[0].split("//")[0] in ("a", "b")
    ]
    f["eq_operand_comps"] = details(
        f"All {len(add_cols)} `=` components of L0-L16 whose on-set is a function of a alone or b alone (addition label)",
        eq_group_table(d, add_cols),
    )
    # first combination L16-18 MLP
    comb = [c for c in eq if 16 <= d.layer[c] <= 18 and d.kind[c] in ("gate", "up", "down")]
    by_lab: dict[str, list[int]] = {}
    for c in comb:
        by_lab.setdefault(lab_add[c], []).append(c)
    f["eq_l16_18_counts"] = md_table(
        [
            "on-set label (addition; `sub only:` = off on addition, label on subtraction)",
            "L16-18 MLP components at =",
        ],
        [[k, len(v)] for k, v in sorted(by_lab.items(), key=lambda kv: -len(kv[1]))],
    )
    labs_sorted = sorted(by_lab.items(), key=lambda kv: -len(kv[1]))
    f["eq_l16_18"] = (
        "\n".join(
            details(
                f"{k}: the {min(8, len(v))} most active of {len(v)}",
                eq_group_table(d, top_by_activity(d, v, 4, 8)),
            )
            for k, v in labs_sorted
        )
        + "\n"
        + link("S3_result.md#l16-l18-mlp-components-at-", "every L16-L18 MLP component at `=`")
    )
    f["_S3_first"] = "## L16-L18 MLP components at `=`\n\n" + "\n".join(
        details(f"{k}: {len(v)} components", eq_group_table(d, v)) for k, v in labs_sorted
    )
    # first result code writers
    parts = []
    for op, q in (("add", "sum"), ("sub", "diff")):
        case = ("mlp_in.19", 4, OPS.index(op), q)
        tab, _ = period_share_table(d, case)
        parts.append(
            details(
                f"`mlp_in.19`, {op}: who writes the {'a+b' if op == 'add' else 'a−b'} code",
                tab + "\n" + writer_lists(d, case, 4, op),
            )
        )
    f["first_result_writers"] = "\n".join(parts)
    # rewrites
    parts, full = [], []
    for rp in ("mlp_in.20", "mlp_in.21", "mlp_in.23", "mlp_in.26", "mlp_in.29", "mlp_in.31"):
        for op, q in (("add", "sum"), ("sub", "diff")):
            case = (rp, 4, OPS.index(op), q)
            tab, _ = period_share_table(d, case)
            title = f"`{rp}`, {op}: who writes the {'a+b' if op == 'add' else 'a−b'} code"
            parts.append(details(title, tab))
            full.append(details(title, tab + "\n" + writer_lists(d, case, 4, op)))
    f["rewrite_writers"] = (
        "\n".join(parts)
        + "\n"
        + link(
            "S3_result.md#result-code-writers-l19-l31",
            "every component supplying ≥ 2 % of a result code, per read point and period",
        )
    )
    f["_S3_rewrite"] = "## Result-code writers, L19-L31\n\n" + "\n".join(full)
    # tens / carry
    tens = [
        c
        for c in eq
        if 15 <= d.layer[c] <= 24 and (lab_add[c] in ("tens(a,b)", "res//10", "carry", "res>=100"))
    ]
    f["tens_comps"] = (
        details(
            f"The 20 most active of the {len(tens)} `=` components of L15-L24 labelled tens(a,b), res//10, carry or res>=100 on addition",
            eq_group_table(d, top_by_activity(d, tens, 4, 20)),
        )
        + "\n"
        + link("S3_result.md#tens-and-carry", "all of them")
    )
    f["_S3_tens"] = "## Tens and carry\n\n" + details(f"all {len(tens)}", eq_group_table(d, tens))
    # appendix C by layer
    for name, lo, hi in (
        ("C1_equals_L0-14", 0, 14),
        ("C2_equals_L15-19", 15, 19),
        ("C3_equals_L20-24", 20, 24),
        ("C4_equals_L25-27", 25, 27),
        ("C5_equals_L28-31", 28, 31),
    ):
        blocks = []
        for li in range(lo, hi + 1):
            cs = [c for c in eq if d.layer[c] == li]
            kinds: dict[str, list[int]] = {}
            for c in cs:
                kinds.setdefault(str(d.kind[c]), []).append(c)
            inner = "\n".join(
                details(f"{k}: {len(v)}", eq_group_table(d, v)) for k, v in sorted(kinds.items())
            )
            blocks.append(
                details(f"layer {li}: {len(cs)} components with main position `=`", inner)
            )
        f[f"_{name}"] = "\n".join(blocks)
    return f


PARITY = [(17, "down", 67), (17, "down", 22), (18, "down", 4)]


def frag_parity(d: Data) -> dict[str, str]:
    df = d.df
    col = {
        k: int(cast(int, df.index[(df.layer == li) & (df.kind == kd) & (df.cidx == ci)][0]))
        for k, (li, kd, ci) in zip(("c67", "c22", "c4"), PARITY, strict=True)
    }
    f = {"parity_comps": d.comp_table(list(col.values()), 4)}
    # feeders of L18 down c4 through the MLP neuron basis
    rows = []
    for kname in ("gate", "up"):
        M = d.wz[f"mlp_18_{kname}"]
        gids = d.wz[f"mlp_18_{kname}_ids"]
        dids = list(d.wz["mlp_18_down_ids"])
        j = dids.index(col["c4"])
        order = np.argsort(-np.abs(M[:, j]))[:8]
        for i in order:
            c = int(gids[i])
            rows.append(
                [d.name(c), f"{M[i, j]:+.2f}", pct(d.on(c, 4, "add")), d.onset(c, 4, "add")]
            )
    f["parity_feeders"] = md_table(
        ["gate/up component of L18", "cos(U, V of down c4)", "on (add)", "on-set (add)"], rows
    )
    return f


def frag_output(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    df = d.df
    late = df.index[(df.main_pos == 4) & df.kind.isin(["down", "o"]) & (df.layer >= 20)].values
    fam: dict[str, list[int]] = {}
    for c in late:
        fam.setdefault(d.label_main(int(c), 4), []).append(int(c))

    def extra(c: int) -> list[Any]:
        return [d.dla_str(c), f"{d.dla_corr(c, 'add'):+.2f} / {d.dla_corr(c, 'sub'):+.2f}"]

    order = sorted(fam, key=lambda k: -len(fam[k]))
    f["late_sub_only_n"] = str(sum(len(v) for k, v in fam.items() if k.startswith("sub only")))
    f["late_rare_n"] = str(sum(len(v) for k, v in fam.items() if k.startswith("rarely")))
    f["late_n"] = str(len(late))
    f["output_counts"] = md_table(
        [
            "on-set label (addition; `sub only:` = off on addition, label on subtraction)",
            "L20-31 `=` writers",
        ],
        [[k, len(fam[k])] for k in order],
    )
    hdr = [
        "top-5 tokens of its direct logit effect",
        "corr(tuning over res, logit of res) add / sub",
    ]
    f["output_families"] = (
        "\n".join(
            details(
                f"{k}: the {min(10, len(fam[k]))} most active of {len(fam[k])} writers",
                d.comp_table(top_by_activity(d, fam[k], 4, 10), 4, extra_headers=hdr, extra=extra),
            )
            for k in order
        )
        + "\n"
        + link("S4_output.md", "every L20-L31 `=` writer by family")
    )
    f["_S4_output"] = "\n".join(
        details(
            f"{k}: {len(fam[k])} writers",
            d.comp_table(
                sorted(fam[k], key=lambda c: (d.layer[c], d.cidx[c])),
                4,
                extra_headers=hdr,
                extra=extra,
            ),
        )
        for k in order
    )
    # hundreds: effect on res+-100 per writer (delta profiles)
    z = np.load(d.ai / "delta_profiles.npz")
    W, deltas, P = z["writers"], z["deltas"], z["add"]
    i0, ip, im = [int(np.flatnonzero(deltas == x)[0]) for x in (0, 100, -100)]
    score = P[:, i0] - 0.5 * (np.nan_to_num(P[:, ip]) + np.nan_to_num(P[:, im]))
    sel = np.argsort(-score)[:40]
    rows = [
        [
            d.name(int(W[i])),
            f"{P[i, i0]:+.3f}",
            f"{P[i, im]:+.3f}",
            f"{P[i, ip]:+.3f}",
            d.label(int(W[i]), 4, "add"),
            d.onset(int(W[i]), 4, "add"),
        ]
        for i in sel
    ]
    f["hundreds"] = md_table(["writer", "Δ=0", "Δ=−100", "Δ=+100", "label", "on-set (add)"], rows)
    return f


def frag_op(d: Data) -> dict[str, str]:
    f: dict[str, str] = {}
    df = d.df
    flags = [int(c) for c in df.index if abs(d.on(int(c), 4, "sub") - d.on(int(c), 4, "add")) > 0.8]
    flags.sort(key=lambda c: (d.layer[c], d.kind[c]))
    wz = d.wz
    writers, readers, vw, rms = wz["writers"], list(wz["readers"]), wz["resid_vw"], wz["rms"]
    diff = df.wmean_sub_p4.values[writers] - df.wmean_add_p4.values[writers]

    def upstream(c: int) -> list[Any]:
        if c not in readers:
            return ["-"]
        j = readers.index(c)
        rp = 2 * d.layer[c] + (1 if d.kind[c] in ("gate", "up") else 0)
        contrib = vw[:, j] / rms[rp, 4] * diff
        top = np.argsort(-np.abs(contrib))[:3]
        return [
            "; ".join(
                f"{d.name(int(writers[k]))} {contrib[k]:+.1f}"
                for k in top
                if abs(contrib[k]) > 0.05
            )
            or "-"
        ]

    rows = []
    for c in flags:
        rows.append(
            [
                d.name(c),
                "sub" if d.on(c, 4, "sub") > d.on(c, 4, "add") else "add",
                f"{pct(d.on(c, 4, 'add'))} / {pct(d.on(c, 4, 'sub'))}",
                *upstream(c),
            ]
        )
    f["opflags_n"] = str(len(flags))
    per_layer = pd.Series([int(d.layer[c]) for c in flags]).value_counts().sort_index()
    f["opflags_layers"] = ", ".join(f"L{k}: {v}" for k, v in per_layer.items())
    # largest attention-group share of each period of any result code (all result cases)
    W = d.ca["writers"]
    is_o = d.kind[W] == "o"
    best: dict[int, tuple[float, str]] = {t: (0.0, "") for t in K_OF}
    for i, case in enumerate(d.ca_cases):
        if case[3] not in ("sum", "diff"):
            continue
        sh, zp = d.ca["share"][i], d.ca["zpow"][i]
        for t, ks in K_OF.items():
            s_ = sh[:-1][:, ks] @ (zp[ks] / zp[ks].sum())
            for li in np.unique(d.layer[W][is_o]):
                v = float(s_[is_o & (d.layer[W] == li)].sum())
                if abs(v) > abs(best[t][0]):
                    best[t] = (v, f"L{li} attention at `{case[0]}` ({OPS[case[2]]})")
    f["attn_result_max"] = md_table(
        ["period of the result code", "largest share from one layer's attention", "where"],
        [[f"mod {t}", f"{v:+.2f}", w] for t, (v, w) in best.items()],
    )
    f["opflags"] = md_table(
        [
            "component",
            "flag for",
            "on add / sub",
            "largest upstream contributions to its op difference (virtual weight × mean write difference)",
        ],
        rows,
    )
    # mirror evidence
    S, R = d.opp["same"], d.opp["reflect"]
    rows = []
    for li in range(13, 23):
        for s_i, s in enumerate(("attn_in", "mlp_in")):
            rows.append(
                [f"{s}.{li}"]
                + [
                    f"{S[li, s_i, 1, 1, k]:.2f} / {R[li, s_i, 1, 1, k]:.2f}"
                    for k in (1, 2, 4, 5, 10, 20)
                ]
                + [f"{S[li, s_i, 1, 0, 10]:.2f} / {R[li, s_i, 1, 0, 10]:.2f}"]
            )
    f["mirror_table"] = md_table(
        [
            "read point (at =)",
            *[f"b k={k} (period {100 // np.gcd(k, 100)})" for k in (1, 2, 4, 5, 10, 20)],
            "a k=10 (control)",
        ],
        rows,
    )
    # L15 MLP at '='
    l15 = df.index[
        (df.layer == 15)
        & df.kind.isin(["gate", "up", "down"])
        & ((df.on_add_p4 > 0.005) | (df.on_sub_p4 > 0.005))
    ].values
    l15 = sorted(l15, key=lambda c: -(d.on(int(c), 4, "sub") - d.on(int(c), 4, "add")))
    f["l15_mlp"] = d.comp_table(l15, 4)
    # same components compute a-b: on-rate over res mod 100, add vs sub
    eq = df.index[(df.main_pos == 4) & (df.layer >= 19)].values
    on_res = d.profiles["on_res"][4]  # (2 ops, A, 199): add res 2..200, sub res -99..99
    res_vals = (np.arange(2, 201), np.arange(-99, 100))
    corr: dict[int, float] = {}
    for c in eq:
        c = int(c)
        prof = []
        for o in (0, 1):
            r = res_vals[o] % 100
            prof.append(np.array([on_res[o, c, r == k].mean() for k in range(100)]))
        if prof[0].std() > 0 and prof[1].std() > 0:
            corr[c] = float(np.corrcoef(prof[0], prof[1])[0, 1])
    shared = sorted(
        [c for c, v in corr.items() if v > 0.7], key=lambda c: (d.layer[c], d.kind[c], d.cidx[c])
    )
    f["shared_res_n"] = f"{len(shared)} of {len(corr)}"
    f["shared_res_median"] = f"{np.median(list(corr.values())):.2f}"
    f["shared_res_total"] = str(len(eq))

    def corr_col(c: int) -> list[Any]:
        return [f"{corr[c]:+.2f}"]

    hdr = ["corr of on-rate over res mod 100, add vs sub"]
    top = sorted(top_by_activity(d, shared, 4, 25), key=lambda c: -corr[c])
    f["shared_res"] = (
        details(
            f"The 25 most active of the {len(shared)} components whose on-rate profiles over res mod 100 correlate > 0.7",
            d.comp_table(top, 4, extra_headers=hdr, extra=corr_col),
        )
        + "\n"
        + link("S5_operator.md", f"all {len(corr)}, with their correlation")
    )
    everything = sorted(corr, key=lambda c: -corr[c])
    f["_S5_shared"] = (
        "## L19-L31 `=` components: does the same component respond to the same residues of res on both operations?\n\n"
        + details(
            f"all {len(corr)}, by decreasing correlation",
            d.comp_table(everything, 4, extra_headers=hdr, extra=corr_col),
        )
    )
    f["opflags_b"] = ""
    return f


def fill(template: str, frags: dict[str, str]) -> str:
    out = template
    for k, v in frags.items():
        out = out.replace("{{" + k + "}}", v)
    missing = [line for line in out.splitlines() if "{{" in line]
    assert not missing, missing[:5]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    d = Data(args.run)
    frags: dict[str, str] = {"legend": LEGEND}
    for fn in (
        frag_pos1,
        frag_code_builders,
        frag_shared_operands,
        frag_pos3,
        frag_heads,
        frag_equals,
        frag_parity,
        frag_output,
        frag_op,
    ):
        frags |= fn(d)
        print("done", fn.__name__, flush=True)
    app = args.notes / "auto_interp_appendix"
    app.mkdir(exist_ok=True)
    head = "[← back to the report](../report_auto_interp.md)\n\n" + LEGEND + "\n\n"
    appendices = {
        "A1_a_token.md": (
            "# Appendix A1 — every component whose main position is `a`",
            frags["_A1_by_class"] + "\n" + frags["_A1_noncausal"],
        ),
        "S1_code_builders.md": (
            "# §1 — every writer of the operand codes",
            frags["_S1_a_builders"] + "\n" + frags["_S1_b_builders"],
        ),
        "S3_result.md": (
            "# §3 — complete lists for the result computation",
            frags["_S3_first"] + "\n" + frags["_S3_rewrite"] + "\n" + frags["_S3_tens"],
        ),
        "S4_output.md": (
            "# §4 — every L20-L31 writer at `=`, by on-set family",
            frags["_S4_output"],
        ),
        "S5_operator.md": ("# §5 — complete lists for the operator", frags["_S5_shared"]),
        "A3_b_token.md": (
            "# Appendix A3 — every component whose main position is `b`, by family",
            frags["_A3_families"],
        ),
        "B_heads.md": (
            "# Appendix B — every attention head with an alive o component",
            frags["_B_heads"],
        ),
        **{
            f"{k[1:]}.md": (
                f"# Appendix {k[1:3]} — components whose main position is `=`, {k.split('_')[-1]}",
                v,
            )
            for k, v in frags.items()
            if k.startswith("_C") and "_" in k[1:]
        },
    }
    for fname, (title, body) in appendices.items():
        (app / fname).write_text(f"{title}\n\n{head}{body}\n")
        print(fname, len(body) // 1000, "kB", flush=True)
    template = (Path(__file__).parent / "report_template.md").read_text()
    (args.notes / "report_auto_interp.md").write_text(fill(template, frags))
    print("report written")


if __name__ == "__main__":
    main()
