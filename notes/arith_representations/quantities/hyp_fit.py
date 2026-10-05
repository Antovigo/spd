"""Hypothesis-level fit of a site: readers = sparse combinations of hypothesised quantities.

A hypothesis set H is a list of quantities q, each a feature matrix Phi_q (100, d_q) over a =
1..100 (its encoding: an indicator, a circle (cos, sin), a smooth ramp, ...) and a cost in bits.
Each reader x (100,) chooses the subset S of quantities (|S| <= K) minimising its description
length

    DL(S) = (N/2) log2(RSS_S / N) + sum_{q in S} (bits_q + d_q * (1/2) log2 N),

(RSS_S: residual sum of squares of the least-squares fit on [1, Phi_q for q in S]; N = 100);
a quantity already used by another reader costs REUSE bits less. The search is greedy (see
`fit_reader`); menu quantities (indicators, intervals, single tokens) let each reader pick its
own items.

Diagnostics on the residuals (what the next hypothesis should explain):
* shared structure: top singular values of the residual matrix vs a null that permutes each
  reader's residual over a independently;
* per reader: the strongest periodic component (|DFT| over a, period 100 / k), the best single
  step (a >= c), the best smooth trend (log a, a), each as the fraction of residual variance it
  would explain, against the same statistic on permuted residuals (95% quantile);
* outlier tokens: values of a whose residual exceeds 3 robust SDs in several readers.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

N = 100
A = np.arange(1, 101)
W_BITS = 0.5 * np.log2(N)
REUSE = 4.0


@dataclass
class Quantity:
    """A hypothesised quantity. Fixed (menu=False): a reader reads all columns of Phi together
    (e.g. a circle's cos and sin). Menu (menu=True): a reader picks individual columns (e.g. the
    indicator of one units digit, one interval of a), each costing log2(#columns) bits."""

    name: str
    Phi: np.ndarray  # (100, d)
    bits: float = 8.0
    menu: bool = False
    labels: tuple[str, ...] = ()


def _rss(F: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    c, *_ = np.linalg.lstsq(F, x, rcond=None)
    f = F @ c
    return float(((x - f) ** 2).sum()), f, c


def _dl(rss: float, bits: float) -> float:
    return 0.5 * N * np.log2(max(rss, 1e-9) / N) + bits


@dataclass
class ReaderFit:
    quantities: list[str]  # quantities used
    items: list[str]  # menu items used ("quantity: label")
    fitted: np.ndarray
    r2: float
    unique: dict[str, float]  # variance fraction lost when dropping each quantity


def _label(q: Quantity, j: int) -> str:
    return q.labels[j] if q.labels else str(j)


def fit_reader(x: np.ndarray, H: list[Quantity], used: Counter, max_moves: int = 16) -> ReaderFit:
    """Greedy MDL: each move adds a fixed quantity, or one column of a menu quantity; a quantity's
    own cost is paid once per reader (REUSE bits less if another reader already uses it)."""
    cols: list[np.ndarray] = [np.ones(N)]
    owner: list[tuple[int, int]] = [(-1, -1)]  # (quantity, column or -1 for the whole quantity)
    in_use: set[int] = set()
    bits = 0.0
    cur = _dl(_rss(np.stack(cols, 1), x)[0], bits)
    for _ in range(max_moves):
        B = np.stack(cols, 1)
        Q, _ = np.linalg.qr(B)
        res = x - Q @ (Q.T @ x)
        best = None
        for qi, q in enumerate(H):
            qbits = 0.0 if qi in in_use else q.bits - (REUSE if used[q.name] else 0.0)
            if not q.menu:
                if qi in in_use:
                    continue
                F = np.hstack([B, q.Phi])
                d = _dl(_rss(F, x)[0], bits + qbits + q.Phi.shape[1] * W_BITS)
                cand = [(d, qi, -1, qbits + q.Phi.shape[1] * W_BITS)]
            else:
                taken = {c for (o, c) in owner if o == qi}
                P = q.Phi - Q @ (Q.T @ q.Phi)
                nrm = np.linalg.norm(P, axis=0)
                sc = np.where(nrm > 1e-8, np.abs(res @ P) / np.where(nrm > 1e-8, nrm, 1), -1)
                for c in taken:
                    sc[c] = -1
                cand = []
                for c in np.argsort(-sc)[:5]:
                    if sc[c] <= 0:
                        continue
                    mb = qbits + np.log2(q.Phi.shape[1]) + W_BITS
                    d = _dl(_rss(np.hstack([B, q.Phi[:, c : c + 1]]), x)[0], bits + mb)
                    cand.append((d, qi, int(c), mb))
            for t in cand:
                if best is None or t[0] < best[0]:
                    best = t
        if best is None or best[0] >= cur:
            break
        cur, qi, c, mb = best
        bits += mb
        in_use.add(qi)
        if c < 0:
            cols.extend(H[qi].Phi.T)
            owner.extend([(qi, -1)] * H[qi].Phi.shape[1])
        else:
            cols.append(H[qi].Phi[:, c])
            owner.append((qi, c))
    B = np.stack(cols, 1)
    rss, fit, _ = _rss(B, x)
    tot = float(((x - x.mean()) ** 2).sum()) or 1e-12
    unique = {}
    for qi in sorted(in_use):
        keep = [k for k, (o, _) in enumerate(owner) if o != qi]
        unique[H[qi].name] = round((_rss(B[:, keep], x)[0] - rss) / tot, 4)
    items = [f"{H[o].name}: {_label(H[o], c)}" for (o, c) in owner if o >= 0 and c >= 0]
    return ReaderFit([H[qi].name for qi in sorted(in_use)], items, fit, 1 - rss / tot, unique)


def fit_site(X: np.ndarray, H: list[Quantity], passes: int = 2) -> list[ReaderFit]:
    used: Counter = Counter()
    fits: list[ReaderFit] = []
    for _ in range(passes):
        fits = [
            fit_reader(X[:, j], H) if False else fit_reader(X[:, j], H, used)
            for j in range(X.shape[1])
        ]
        used = Counter(q for f in fits for q in f.quantities)
    return fits


def _pattern_stats(r: np.ndarray) -> dict[str, tuple[float, str]]:
    """Fraction of residual variance captured by the best periodic / step / smooth pattern."""
    r = r - r.mean()
    tot = (r**2).sum() or 1e-12
    F = np.fft.rfft(r)
    p = np.abs(F[1:]) ** 2 * 2 / N
    k = int(np.argmax(p)) + 1
    per = (min(float(p[k - 1] / tot), 1.0), f"k={k} (period {100 / k:g})")
    cs = np.cumsum(r[::-1])[::-1]  # sum of r over a >= c, for c = 1..100
    n_up = np.arange(N, 0, -1)
    gain = cs[1:] ** 2 * N / (n_up[1:] * (N - n_up[1:]))
    c = int(np.argmax(gain)) + 2
    step = (float(gain[c - 2] / tot), f"a >= {c}")
    sm = np.stack([np.log(A), A / 100.0], 1)
    sm = sm - sm.mean(0)
    rs, _, _ = _rss(sm, r)
    smooth = (float(1 - rs / tot), "log a, a")
    return {"periodic": per, "step": step, "smooth": smooth}


def diagnostics(R: np.ndarray, names: list[str], n_null: int = 200) -> dict:
    rng = np.random.default_rng(0)
    Rc = R - R.mean(0)
    sv = np.linalg.svd(Rc, compute_uv=False)
    null_sv = [np.linalg.svd(np.stack([rng.permutation(Rc[:, j]) for j in range(R.shape[1])], 1),
                             compute_uv=False)[0] for _ in range(50)]  # fmt: skip
    per_reader = {}
    for j, nm in enumerate(names):
        st = _pattern_stats(R[:, j])
        null = {k: [] for k in st}
        for _ in range(n_null):
            s0 = _pattern_stats(rng.permutation(R[:, j]))
            for kk in st:
                null[kk].append(s0[kk][0])
        flags = {
            kk: (round(v[0], 3), v[1]) for kk, v in st.items() if v[0] > np.quantile(null[kk], 0.95)
        }
        if flags:
            per_reader[nm] = flags
    mad = np.median(np.abs(Rc - np.median(Rc, 0)), 0) * 1.4826 + 1e-12
    z = np.abs(Rc) / mad
    out_tok = Counter(int(a) for a in np.argwhere(z > 3)[:, 0] + 1)
    return {"top_sv": sv[:4].round(3).tolist(), "null_top_sv_95": round(float(np.quantile(null_sv, 0.95)), 3),
            "patterned_readers": per_reader,
            "outlier_tokens": [(t, c) for t, c in out_tok.most_common(12) if c >= 2]}  # fmt: skip
