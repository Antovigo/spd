"""Tools for the quantity-finding agent (INSTRUCTIONS.md). Training rows only: the test rows of each
position (TEST_SHARE of the units below) are never loaded by these tools.

    python qa.py sites <t>                             the read sites of position t, in stream order
    python qa.py show <t> <site> <outdir>              look at the readers' inner activations
    python qa.py fit <t> <site> <spec.py> <outdir>     fit the spec's quantities, report bits, residual figures

Units held out together (train/test split and inner folds): t = 1, 2: values of a; t = 3, 4: (a, b)
pairs, both ops together.

Description length of a site's fit (bits), two-part code:
  data:     the residual of the stream in the readers' span (Z, training rows), coded per principal
            coordinate (PCA of the centred Z, components holding 99.99% of its variance, k_eff of
            them) as Gaussian: sum_j N/2 log2(2 pi e RSS_j / N);
  params:   1/2 log2(N) per fitted coefficient: dims(q) x k_eff per quantity;
  formula:  the quantity's formula as an expression tree: log2(V) bits per node (V = the size of the
            vocabulary below), plus each integer n coded with Elias-delta (|n| + 1) and a sign bit,
            each decimal number m x 10^e with Elias-delta for m and e and two sign bits.
A quantity is worth keeping when removing it raises the total (drop-one delta > 0).
"""

import ast
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qagent"))
from qdata import DATA, load  # noqa: E402

TEST_SHARE = 0.2
SEED = 12345
INNER_FOLDS = 5
RIDGE = 1e-3
IMPORTANT = 0.01


# ---------------------------------------------------------------- split
def unit_key(t: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.copy() if t <= 2 else (a - 1) * 100 + (b - 1)


def train_rows(t: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    key = unit_key(t, a, b)
    units = np.random.default_rng(SEED).permutation(np.unique(key))
    test = units[: int(round(TEST_SHARE * len(units)))]
    return np.flatnonzero(~np.isin(key, test))


# ---------------------------------------------------------------- feature builders
def scalar(x):  # noqa: ANN001, ANN201
    return np.asarray(x, float)[:, None]


def circle(x, P):  # noqa: ANN001, ANN201
    th = 2 * np.pi * np.asarray(x, float) / P
    return np.stack([np.cos(th), np.sin(th)], 1)


def classes(x):  # noqa: ANN001, ANN201
    x = np.asarray(x)
    return np.stack([(x == v).astype(float) for v in np.unique(x)], 1)


def onehot(x, values):  # noqa: ANN001, ANN201
    x = np.asarray(x)
    return np.stack([(x == v).astype(float) for v in values], 1)


def bumps(x, lo, hi, step, width):  # noqa: ANN001, ANN201
    cs = np.arange(lo, hi + 1e-9, step)
    x = np.asarray(x, float)
    return np.stack([np.exp(-((x - c) ** 2) / (2 * width**2)) for c in cs], 1)


def spline(x, lo, hi, n):  # noqa: ANN001, ANN201
    inner = np.linspace(lo, hi, n - 2)
    knots = np.r_[[lo] * 3, inner, [hi] * 3]
    return BSpline.design_matrix(np.clip(np.asarray(x, float), lo, hi), knots, 3).toarray()


def gate(cond, F):  # noqa: ANN001, ANN201
    return np.asarray(F, float) * np.asarray(cond, float).reshape(-1, 1)


BUILDERS = {"scalar": scalar, "circle": circle, "classes": classes, "onehot": onehot, "bumps": bumps,
            "spline": spline, "gate": gate}  # fmt: skip
FUNCS = {
    "log": np.log,
    "abs": np.abs,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "sqrt": np.sqrt,
}
VARIABLES = ("a", "b", "op")  # op: 0 = +, 1 = -
OPERATORS = (
    "+",
    "-",
    "*",
    "//",
    "%",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "&",
    "|",
    "~",
    "neg",
    "number",
    "list",
)
VOCAB = len(VARIABLES) + len(BUILDERS) + len(FUNCS) + len(OPERATORS)


def elias_delta(n: int) -> int:
    N = int(math.floor(math.log2(n)))
    return N + 2 * int(math.floor(math.log2(N + 1))) + 1


def number_bits(v: float) -> float:
    if float(v).is_integer():
        return elias_delta(abs(int(v)) + 1) + 1
    m, e = float(v), 0
    while not float(m).is_integer():
        m, e = m * 10, e + 1
    return elias_delta(abs(int(m)) + 1) + elias_delta(e + 1) + 2


def formula_bits(src: str) -> float:
    """Bits of a formula: log2(VOCAB) per node of its expression tree plus the numbers' codes."""
    tree = ast.parse(src, mode="eval").body
    node_bits = math.log2(VOCAB)
    allowed = set(VARIABLES) | set(BUILDERS) | set(FUNCS)

    def walk(n: ast.AST) -> float:
        if isinstance(n, ast.Constant):
            assert isinstance(n.value, int | float) and not isinstance(n.value, bool), (
                f"constant {n.value!r}"
            )
            return node_bits + number_bits(n.value)
        if isinstance(n, ast.Name):
            assert n.id in allowed, f"unknown name {n.id}"
            return node_bits
        if isinstance(n, ast.BinOp):
            return node_bits + walk(n.left) + walk(n.right)
        if isinstance(n, ast.UnaryOp):
            return node_bits + walk(n.operand)
        if isinstance(n, ast.Compare):
            return len(n.ops) * node_bits + walk(n.left) + sum(walk(c) for c in n.comparators)
        if isinstance(n, ast.Call):
            assert isinstance(n.func, ast.Name) and n.func.id in allowed, (
                "calls must be to builders or functions"
            )
            assert not n.keywords, "positional arguments only"
            return node_bits + sum(walk(x) for x in n.args)
        if isinstance(n, ast.List | ast.Tuple):
            return node_bits + sum(walk(x) for x in n.elts)
        raise AssertionError(f"not allowed in a formula: {ast.dump(n)[:60]}")

    return walk(tree)


def evaluate(src: str, op: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    formula_bits(src)  # validates
    env = {
        "a": a.astype(np.int64),
        "b": b.astype(np.int64),
        "op": op.astype(np.int64),
        **BUILDERS,
        **FUNCS,
    }
    F = np.asarray(
        eval(compile(ast.parse(src, mode="eval"), "<formula>", "eval"), {"__builtins__": {}}, env),
        float,
    )  # noqa: S307
    if F.ndim == 1:
        F = F[:, None]
    assert F.shape[0] == len(a) and np.isfinite(F).all(), (
        "a formula must give finite features, one row per prompt"
    )
    return F


# ---------------------------------------------------------------- site data (training rows)
class Site:
    def __init__(self, t: int, label: str) -> None:
        pos = load(t, [label])
        assert label in pos.sites, f"no site {label} at t = {t}; sites: see `show`"
        tr = train_rows(t, pos.a, pos.b)
        s = pos.sites[label]
        self.t, self.label = t, label
        self.op, self.a, self.b = pos.op[tr], pos.a[tr], pos.b[tr]
        self.Z, self.W, self.CI, self.rho, self.cols = s.Z[tr], s.W, s.CI[tr], s.rho[tr], s.cols
        self.N = len(tr)
        Zc = self.Z - self.Z.mean(0)
        _, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
        keep = int(np.searchsorted(np.cumsum(sv**2) / max((sv**2).sum(), 1e-300), 0.9999) + 1)
        self.B = Vt[:keep].T  # k x k_eff principal coordinates of the stream in the span
        self.k_eff = keep
        key = unit_key(t, self.a, self.b)
        units = np.random.default_rng(SEED + 1).permutation(np.unique(key))
        self.folds = [
            np.flatnonzero(np.isin(key, units[f::INNER_FOLDS])) for f in range(INNER_FOLDS)
        ]

    @property
    def Y(self) -> np.ndarray:
        return self.Z @ self.W

    @property
    def act(self) -> np.ndarray:
        return self.Y / self.rho[:, None]


def ridge(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    PP = P.T @ P
    lam = max(RIDGE * np.trace(PP) / max(PP.shape[0], 1), 1e-12)
    return np.linalg.solve(PP + lam * np.eye(PP.shape[0]), P.T @ T)


def design(
    s: Site, quantities: list[tuple[str, str]]
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    blocks, dims = [], []
    for _, src in quantities:
        F = evaluate(src, s.op, s.a, s.b)
        F = F - F.mean(0)
        r = np.linalg.matrix_rank(F) if F.size else 0
        blocks.append(F)
        dims.append(int(r))
    P = np.hstack(blocks) if blocks else np.zeros((s.N, 0))
    return P, blocks, dims


def data_bits(E: np.ndarray, N: int) -> float:
    rss = (E**2).sum(0)
    return float((N / 2 * np.log2(2 * np.pi * np.e * np.maximum(rss, 1e-300) / N)).sum())


def score(s: Site, quantities: list[tuple[str, str]]) -> dict:
    P, blocks, dims = design(s, quantities)
    X = (s.Z - s.Z.mean(0)) @ s.B  # principal coordinates
    E = X - (P @ ridge(P, X) if P.shape[1] else 0)
    half = 0.5 * math.log2(s.N)
    fbits = [formula_bits(src) for _, src in quantities]
    total = data_bits(E, s.N) + sum(half * d * s.k_eff for d in dims) + sum(fbits)
    return {"data": data_bits(E, s.N), "params": sum(half * d * s.k_eff for d in dims), "formula": sum(fbits),
            "total": total, "dims": dims, "formula_bits": fbits, "E": E, "P": P}  # fmt: skip


def heldout_r2(s: Site, P: np.ndarray) -> float:
    Y = s.Y
    err = tot = 0.0
    for te in s.folds:
        tr = np.setdiff1d(np.arange(s.N), te)
        mY = Y[tr].mean(0)
        tot += float(((Y[te] - mY) ** 2).sum())
        if P.shape[1] == 0:
            err += float(((Y[te] - mY) ** 2).sum())
            continue
        mP = P[tr].mean(0)
        G = ridge(P[tr] - mP, Y[tr] - mY)
        err += float(((Y[te] - mY - (P[te] - mP) @ G) ** 2).sum())
    return 1 - err / tot


# ---------------------------------------------------------------- figures
def panels(
    s: Site,
    cols: list[np.ndarray],
    titles: list[str],
    path: Path,
    ci_cols: list[np.ndarray] | None = None,
) -> None:
    """One row per entry of `cols` (a value per training row): t <= 2 curves over a (one per op);
    t >= 3 (a, b) grids per op (test pairs grey). Optional CI panels."""
    n = len(cols)
    if s.t <= 2:
        fig, axes = plt.subplots(n, 1, figsize=(7, 2.0 * n), squeeze=False)
        for i, v in enumerate(cols):
            ax = axes[i, 0]
            for op in np.unique(s.op):
                m = s.op == op
                o = np.argsort(s.a[m])
                ax.plot(s.a[m][o], v[m][o], ".-", ms=3, lw=0.7, label=f"op {'+-'[op]}")
            if ci_cols is not None:
                ax2 = ax.twinx()
                ax2.bar(s.a, ci_cols[i], color="0.5", alpha=0.25, width=1)
                ax2.set_ylim(0, 1.05)
            ax.set_title(titles[i], fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=6)
        axes[-1, 0].set_xlabel("a")
    else:
        ncol = 4 if ci_cols is not None else 2
        fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 3.0 * n), squeeze=False)
        for i, v in enumerate(cols):
            for op in (0, 1):
                g = np.full((100, 100), np.nan)
                m = s.op == op
                g[s.a[m] - 1, s.b[m] - 1] = v[m]
                ax = axes[i, op]
                cm = plt.get_cmap("viridis").copy()
                cm.set_bad("0.85")
                im = ax.imshow(g, cmap=cm, interpolation="nearest")
                ax.set_title(f"{titles[i]} | a {'+-'[op]} b", fontsize=7)
                ax.set_xlabel("b", fontsize=6)
                ax.set_ylabel("a", fontsize=6)
                ax.tick_params(labelsize=5)
                fig.colorbar(im, ax=ax, fraction=0.04)
                if ci_cols is not None:
                    gc = np.full((100, 100), np.nan)
                    gc[s.a[m] - 1, s.b[m] - 1] = ci_cols[i][m]
                    ax = axes[i, 2 + op]
                    ax.imshow(gc, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(f"CI | a {'+-'[op]} b", fontsize=7)
                    ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def name(s: Site, r: int) -> str:
    return f"reader c{int(s.cols[r])}"


# ---------------------------------------------------------------- commands
def cmd_show(t: int, label: str, out: Path, top: int = 8) -> None:
    s = Site(t, label)
    A = s.act
    w = (s.CI * (A - A.mean(0)) ** 2).sum(0)
    order = np.argsort(-w)[:top]
    print(f"site {label} at t = {t}: {s.W.shape[1]} readers, reader span k = {s.W.shape[0]} (k_eff = {s.k_eff} principal "
          f"coordinates), {s.N} training rows")  # fmt: skip
    print(
        "readers ranked by CI-weighted variance of their inner activation (share of the site total; rows where CI > 0.01):"
    )
    for r in order:
        print(
            f"  {name(s, r)}: {w[r] / w.sum():.3f}; important on {(s.CI[:, r] > IMPORTANT).sum()} of {s.N} rows"
        )
    X = (s.Z - s.Z.mean(0)) @ s.B
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    print(
        f"stream principal components (share of variance): {np.round(sv[:8] ** 2 / (sv**2).sum(), 3).tolist()}"
    )
    out.mkdir(parents=True, exist_ok=True)
    panels(
        s,
        [A[:, r] for r in order],
        [f"{name(s, r)} inner activation" for r in order],
        out / "show_readers.png",
        [s.CI[:, r] for r in order],
    )
    panels(
        s,
        [U[:, j] for j in range(4)],
        [
            f"stream principal component {j + 1} ({sv[j] ** 2 / (sv**2).sum():.3f})"
            for j in range(4)
        ],
        out / "show_components.png",
    )
    print(f"figures: {out / 'show_readers.png'}, {out / 'show_components.png'}")


def load_spec(path: Path) -> list[tuple[str, str]]:
    env: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), {"__builtins__": {}}, env)  # noqa: S102
    q = env["QUANTITIES"]
    assert all(isinstance(x, tuple) and len(x) == 2 for x in q), (
        "QUANTITIES = [(name, formula), ...]"
    )
    return q


def cmd_fit(t: int, label: str, spec: Path, out: Path, top: int = 6) -> None:
    s = Site(t, label)
    Q = load_spec(spec)
    full = score(s, Q)
    base = score(s, [])
    rows = []
    for i, (nm, src) in enumerate(Q):
        without = score(s, Q[:i] + Q[i + 1 :])
        rows.append({"name": nm, "formula": src, "dims": full["dims"][i], "formula_bits": round(full["formula_bits"][i], 1),
                     "param_bits": round(0.5 * math.log2(s.N) * full["dims"][i] * s.k_eff, 1),
                     "drop_one_delta_bits": round(without["total"] - full["total"], 1)})  # fmt: skip
    Y = s.Y
    Yc = Y - Y.mean(0)
    Ehat = full["E"] @ s.B.T @ s.W  # residual of the reads (raw)
    res_act = Ehat / s.rho[:, None]
    imp = s.CI > IMPORTANT
    rep = {
        "site": label, "t": t, "N_train": s.N, "k": int(s.W.shape[0]), "k_eff": s.k_eff, "n_readers": int(s.W.shape[1]),
        "bits": {k: round(full[k], 1) for k in ("data", "params", "formula", "total")}, "bits_no_quantities": round(base["total"], 1),
        "r2_reads_in_sample": round(1 - float((Ehat**2).sum() / (Yc**2).sum()), 4),
        "r2_reads_heldout_inner": round(heldout_r2(s, full["P"]), 4),
        "important_residual_share": round(float(((Ehat**2) * imp).sum() / max(((Yc**2) * imp).sum(), 1e-300)), 4),
        "quantities": rows,
    }  # fmt: skip
    w = (s.CI * res_act**2).sum(0)
    order = np.argsort(-w)[:top]
    rep["worst_readers"] = [
        {"reader": name(s, r), "share_of_ci_weighted_residual": round(float(w[r] / w.sum()), 3)}
        for r in order
    ]
    var = {
        "op": s.op,
        "a": s.a,
        "b": s.b,
        "a+b": s.a + s.b,
        "a-b": s.a - s.b,
        "a%10": s.a % 10,
        "b%10": s.b % 10,
        "a//10": s.a // 10,
        "b//10": s.b // 10,
    }
    if t <= 2:
        var = {k: v for k, v in var.items() if "b" not in k and (t == 2 or k != "op")}
    Ec = Ehat - Ehat.mean(0)
    tot = (Ec**2).sum()
    left = []
    for k, x in var.items():
        vals, inv = np.unique(x, return_inverse=True)
        means = np.zeros((len(vals), Ec.shape[1]))
        np.add.at(means, inv, Ec)
        means /= np.bincount(inv)[:, None]
        sh = float((means[inv] ** 2).sum() / max(tot, 1e-300))
        left.append((round(sh - (len(vals) - 1) / (s.N - 1), 4), k))
    rep["residual_explained_by_variable_values"] = sorted(left, reverse=True)
    out.mkdir(parents=True, exist_ok=True)
    U, sv, _ = np.linalg.svd(Ec, full_matrices=False)
    panels(
        s,
        [Y[:, r] / s.rho for r in order] + [],
        [f"{name(s, r)} inner activation" for r in order],
        out / f"{spec.stem}_reads.png",
        [s.CI[:, r] for r in order],
    )
    panels(
        s,
        [res_act[:, r] for r in order],
        [f"{name(s, r)} residual" for r in order],
        out / f"{spec.stem}_residual.png",
    )
    panels(
        s,
        [U[:, j] for j in range(3)],
        [
            f"residual singular function {j + 1} ({sv[j] ** 2 / max((sv**2).sum(), 1e-300):.3f})"
            for j in range(3)
        ],
        out / f"{spec.stem}_residual_svd.png",
    )
    rep["figures"] = [
        str(out / f"{spec.stem}_{x}.png") for x in ("reads", "residual", "residual_svd")
    ]
    (out / f"{spec.stem}_fit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_sites(t: int) -> None:
    f = np.load(DATA / f"site_data_t{t}.npz")
    labels = sorted(
        {k.split("/")[0] for k in f.files if "/" in k},
        key=lambda x: (int(x[1:].split(".")[0]), x.endswith("mlp")),
    )
    for lab in labels:
        Z = f[lab + "/Z"]
        const = bool(np.allclose(Z, Z[:1], atol=1e-6 * (np.abs(Z).max() + 1e-30)))
        print(
            lab,
            f"{f[lab + '/W'].shape[1]} readers",
            "(constant over the domain: nothing to fit)" if const else "",
        )


if __name__ == "__main__":
    if sys.argv[1] == "sites":
        cmd_sites(int(sys.argv[2]))
        raise SystemExit
    cmd, t, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if cmd == "show":
        cmd_show(t, label, Path(sys.argv[4]))
    elif cmd == "fit":
        cmd_fit(t, label, Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        raise SystemExit(__doc__)
