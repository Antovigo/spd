"""Periodicity index of the readers: periodicity.json.

Part 1, the operand positions: readers of blocks 0..16 at t = 1 (in a) and t = 3 (in b), on all prompts.
Part 2, the "=" position: readers of blocks 0..31 at t = 4, each tested for periodicity in a, in b and in a + b
(not exclusive), on the addition prompts only. Part 2 is defined after part 1.

Part 1.

Notation:
  b          block, 0..16
  t          token position, 1 (the token a) or 3 (the token b); the prompt is `<BOS> a op b =`
  x          the operand at position t (a for t = 1, b for t = 3), integer 1..400
  c          a reader: a kept gate / up component of block b (readers.json)
  f_c(x)     its inner activation t_c / ||V_c|| (V_c its read vector), averaged over every stored prompt whose
             operand at position t equals x: 800 prompts per x (400 values of the other operand, 2 ops).
             At t = 1 every prompt with the same a has the same activation, so the average is exact.
  m_c(r)     mean of f_c(x) over the 4 values x in 1..400 with x mod 100 = r, r in 0..99
  M_c(k)     sum_r m_c(r) exp(-2 pi i k r / 100), k in 0..99 (i the imaginary unit)
  tau(k)     100 / gcd(k, 100): the exact period of frequency k (tau(0) = 1, the constant)
  P_tau f_c  the part of f_c made of the frequencies k with tau(k) = tau. tau runs over the divisors of 100,
             {1, 2, 4, 5, 10, 20, 25, 50, 100}; the parts are orthogonal and sum to m_c(x mod 100). A part of
             exact period tau holds nothing that a divisor of tau already explains: a function of x mod 2 has
             all its (non-constant) energy in P_2 and none in P_10.
  rest       f_c(x) - m_c(x mod 100): what no period dividing 100 explains (e.g. a dependence on x // 100)
  E          sum_x f_c(x)^2 over x = 1..400, uncentred (the network has no biases; zero is activation zero)
  share_tau  sum_x (P_tau f_c)(x)^2 / E = 4 sum_{k: tau(k) = tau} |M_c(k)|^2 / (100 E)
  share_rest sum_x rest(x)^2 / E;  share_1 + sum over tau > 1 of share_tau + share_rest = 1
  null_tau   99th percentile of share_tau over NPERM random permutations of f_c over x (seed 0): the share a
             function with the same values but no arrangement in x gets
  align_tau  99th percentile of share_tau over NPERM draws (seed 0) in which the hundreds x = 101..200, 201..300,
             301..400 are each rotated by an independent random offset in 0..99 (x -> 100 q + (x - 100 q - 1 + s_q)
             mod 100 + 1, hundred q = 1, 2, 3; the first hundred fixed): keeps the shape within each hundred and
             breaks only their alignment. Used for tau in {20, 25, 50, 100}, the low frequencies (k < 10 among
             them) where a smooth, non-periodic f_c beats the permutation null; for tau <= 10 the chance that all
             three offsets are multiples of tau (1/8 for tau = 2) is too large for a 1% test
  smooth_r2  uncentred share of E explained by a degree-3 polynomial in x (a slow trend in x puts energy in
             P_100, P_50, ... as a sawtooth; a large smooth_r2 next to a tau = 50 or 100 label flags that)

exact_periods = [tau > 1 with share_tau >= THRESH (= meta.response_threshold), share_tau > null_tau, and, for
tau in {20, 25, 50, 100}, share_tau > align_tau]: the exact-period parts that are present.
Label: periods = the members of exact_periods that divide no other member: the fewest periods whose periodic
functions add up to every present part (a function of period tau holds parts of exact period every divisor of tau).
The indicator of x mod 10 = 3 has parts of exact period 2, 5 and 10 (energy 1 : 4 : 4) and is labelled [10];
[x mod 2 = 0] + [x mod 5 = 3] has parts 2 and 5 only and is labelled [2, 5]. minimal_period = lcm(periods).

Part 2 (t = 4, the token "="; the 160,000 prompts a + b with a, b in 1..400).
  v          the variable tested: a, b or s = a + b (s in 2..800)
  f_c(a, b)  the reader's inner activation t_c / ||V_c|| on prompt a + b (no averaging: at "=" it depends on both)
  m_c(r)     mean of f_c over the prompts with v mod 100 = r, r in 0..99: exactly 1,600 prompts per class for each
             v (a and b each run over four whole periods of 100, so every value of s mod 100 is equally common)
  M_c(k), tau(k), exact-period parts P_tau as in part 1, with the sums over the 160,000 prompts:
  E          sum over prompts of f_c^2 (uncentred)
  share_tau  1600 sum_{k: tau(k) = tau} |M_c(k)|^2 / (100 E): the energy of the part of f_c that is a function of
             v mod 100 with exact period tau, as a share of all of f_c's energy (so dependence on anything else,
             e.g. on a when v = s, lowers every share of v)
  share_rest 1 - sum over tau of share_tau: the energy that no function of v mod 100 explains
  Parts in different variables are orthogonal (a non-constant function of a mod 100 averages to zero over every
  class of s mod 100, and the reverse), so a reader periodic only in a gets no share in s.
  null_tau   99th percentile of the same statistic (class means over the relabelled classes, then the same DFT
             energy) over NPERM relabellings v -> pi(v) (seed 0): pi a random permutation of the values of v
  align_tau  the same with pi rotating each hundred of v (100 q + 1 .. 100 q + 100, q >= 1) by an independent random
             offset in 0..99 and keeping the first hundred fixed; used for tau in {20, 25, 50, 100} as in part 1.
             Relabelled classes no longer hold equal counts, so null statistics use the DFT of the class means as
             the observed statistic does (for which it is exact)
  exact_periods, periods, minimal_period   as in part 1, per variable

Layout: readers[id].positions[t][v] = entry, t in {"1", "3", "4"}, v in {"a", "b", "a+b"} (t = 1: v = a only;
t = 3: v = b only); index[t][v][tau] = reader ids with tau in periods.

    python seed_periodicity.py
"""

import datetime
import subprocess
import sys
from math import gcd, lcm
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo import ROOT, Repo, _write_json  # noqa: E402

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
SRC = RUN / "analysis/reader_writer/variance/alive_only"
N = 400
BLOCKS = range(17)
TS = (1, 3)
EQ_BLOCKS = range(32)
EQ_VARS = ("a", "b", "a+b")
TAUS = (1, 2, 4, 5, 10, 20, 25, 50, 100)
K = np.arange(100)
TAU_OF_K = np.asarray([100 // gcd(int(k), 100) for k in K])
NPERM = 1000
ALIGN_TAUS = (20, 25, 50, 100)
X = np.arange(1, N + 1)
POLY = np.stack([(X / N) ** d for d in range(4)], 1)


def operand_means(T: np.ndarray, t: int) -> np.ndarray:
    """(400, K) mean of T (320000, K) over prompts whose operand at position t is x = 1..400.

    Row order of inner.npy: op * N * N + (a - 1) * N + (b - 1)."""
    R = T.reshape(2, N, N, -1)
    return R.mean(axis=(0, 2)) if t == 1 else R.mean(axis=(0, 1))


def shares(F: np.ndarray) -> np.ndarray:
    """F (..., 400, C) -> (..., len(TAUS) + 1, C): share_tau for TAUS, then share_rest."""
    E = (F**2).sum(-2)
    m = F.reshape(*F.shape[:-2], 4, 100, F.shape[-1]).mean(-3)
    P = np.abs(np.fft.fft(m, axis=-2)) ** 2 * 4 / 100
    out = [P[..., tau == TAU_OF_K, :].sum(-2) / E for tau in TAUS]
    rest = F - np.tile(m, (*([1] * (F.ndim - 2)), 4, 1))
    out.append((rest**2).sum(-2) / E)
    return np.stack(out, -2)


def null(F: np.ndarray, kind: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if kind == "perm":
        perms = np.stack([rng.permutation(N) for _ in range(NPERM)])
    else:
        shifts = np.concatenate([np.zeros((NPERM, 1), int), rng.integers(0, 100, (NPERM, 3))], 1)
        j = np.arange(100)
        perms = np.stack(
            [np.concatenate([100 * q + (j + s[q]) % 100 for q in range(4)]) for s in shifts]
        )
    acc = []
    for i in range(0, NPERM, 50):
        acc.append(shares(F[perms[i : i + 50]]))
    return np.quantile(np.concatenate(acc), 0.99, axis=0)


def label(S: np.ndarray, Q: np.ndarray, QA: np.ndarray, j: int, thresh: float) -> tuple[list, list]:
    exact = [tau for i, tau in enumerate(TAUS)
             if tau > 1 and S[i, j] >= thresh and S[i, j] > Q[i, j]
             and (tau not in ALIGN_TAUS or S[i, j] > QA[i, j])]  # fmt: skip
    periods = [tau for tau in exact if not any(u != tau and u % tau == 0 for u in exact)]
    return exact, periods


AB = np.arange(N * N)
A_OF, B_OF = AB // N + 1, AB % N + 1
VALUES = {"a": A_OF, "b": B_OF, "a+b": A_OF + B_OF}


def class_stat(Sv: np.ndarray, nv: np.ndarray, lab: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Per-value sums Sv (V, C) and counts nv (V,), class r = lab mod 100 -> (len(TAUS), C) DFT energy shares."""
    onehot = np.zeros((100, len(lab)))
    onehot[lab % 100, np.arange(len(lab))] = 1
    C = onehot @ Sv
    n = np.bincount(lab % 100, weights=nv, minlength=100)
    m = C / n[:, None]
    P = np.abs(np.fft.fft(m, axis=0)) ** 2 * (N * N / 100) / 100
    return np.stack([P[tau == TAU_OF_K].sum(0) / E for tau in TAUS])


def eq_shares(F: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """F (160000, C) at "=" on the addition prompts -> {v: (S, Q, QA)}; S has share_rest as its last row."""
    E = (F**2).sum(0)
    rng = np.random.default_rng(0)
    out = {}
    for v, vals in VALUES.items():
        vs = np.unique(vals)
        idx = np.searchsorted(vs, vals)
        Sv = np.zeros((len(vs), F.shape[1]))
        np.add.at(Sv, idx, F)
        nv = np.bincount(idx, minlength=len(vs)).astype(float)
        S = class_stat(Sv, nv, vs, E)
        S = np.concatenate([S, 1 - S.sum(0, keepdims=True)])
        hund = (vs - 1) // 100
        nulls = []
        for kind in ("perm", "align"):
            acc = []
            for _ in range(NPERM):
                if kind == "perm":
                    lab = vs[rng.permutation(len(vs))]
                else:
                    off = np.concatenate([[0], rng.integers(0, 100, hund.max())])[hund]
                    lab = 100 * hund + (vs - 1 - 100 * hund + off) % 100 + 1
                acc.append(class_stat(Sv, nv, lab, E))
            nulls.append(np.quantile(np.stack(acc), 0.99, axis=0))
        out[v] = (S, nulls[0], nulls[1])
    return out


def entry_of(S, Q, QA, j, thresh) -> dict:  # noqa: ANN001
    exact, periods = label(S, Q, QA, j, thresh)
    return {
        "periods": periods,
        "exact_periods": exact,
        "minimal_period": lcm(*periods) if periods else None,
        "share": {str(tau): float(S[i, j]) for i, tau in enumerate(TAUS)},
        "share_rest": float(S[-1, j]),
        "null_p99": {str(tau): float(Q[i, j]) for i, tau in enumerate(TAUS) if tau > 1},
        "align_null_p99": {
            str(tau): float(QA[i, j]) for i, tau in enumerate(TAUS) if tau in ALIGN_TAUS
        },
    }


def main() -> None:
    repo = Repo()
    thresh = repo.meta["response_threshold"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()  # fmt: skip
    today = datetime.date.today().isoformat()
    readers: dict[str, dict] = {}
    for b in EQ_BLOCKS:
        rds = [rd for rd in repo.readers.values() if rd["block"] == b]
        z = np.load(SRC / f"L{b}" / "comps.npz")
        col = {
            (str(k), int(i)): j for j, (k, i) in enumerate(zip(z["kind"], z["index"], strict=True))
        }
        cols = np.asarray([col[(rd["kind"], rd["component"])] for rd in rds])
        vnorm = np.asarray([np.linalg.norm(z[f"V_{rd['kind']}"][:, list(z["index"][z["kind"] == rd["kind"]])
                            .index(rd["component"])]) for rd in rds])  # fmt: skip
        inner = np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")
        for t in TS if b in BLOCKS else ():
            T = np.asarray(inner[:, t - 1, :], np.float64)[:, cols] / vnorm
            F = operand_means(T, t)
            S = shares(F)
            Q = null(F, "perm")
            QA = null(F, "align")
            coef, *_ = np.linalg.lstsq(POLY, F, rcond=None)
            smooth = 1 - ((F - POLY @ coef) ** 2).sum(0) / (F**2).sum(0)
            for j, rd in enumerate(rds):
                entry = readers.setdefault(rd["id"], {"block": b, "kind": rd["kind"],
                                                      "component": rd["component"], "positions": {}})  # fmt: skip
                entry["positions"][str(t)] = {"ab"[t // 2]: {
                    **entry_of(S, Q, QA, j, thresh),
                    "smooth_r2": float(smooth[j]),
                    "rms": float(np.sqrt((F[:, j] ** 2).mean())),
                }}  # fmt: skip
        F = np.asarray(inner[: N * N, 3, :], np.float64)[:, cols] / vnorm
        for v, (S, Q, QA) in eq_shares(F).items():
            for j, rd in enumerate(rds):
                entry = readers.setdefault(rd["id"], {"block": b, "kind": rd["kind"],
                                                      "component": rd["component"], "positions": {}})  # fmt: skip
                entry["positions"].setdefault("4", {})[v] = {
                    **entry_of(S, Q, QA, j, thresh),
                    "rms": float(np.sqrt((F[:, j] ** 2).mean())),
                }
        print(f"L{b}: {len(rds)} readers", flush=True)

    variables = {"1": ("a",), "3": ("b",), "4": EQ_VARS}
    index = {
        t: {v: {str(tau): sorted((rid for rid, e in readers.items()
                                  if t in e["positions"] and tau in e["positions"][t][v]["periods"]),
                                 key=lambda s: (readers[s]["block"], readers[s]["kind"], readers[s]["component"]))
                for tau in TAUS if tau > 1} for v in vs}
        for t, vs in variables.items()
    }  # fmt: skip
    doc = {
        "definitions": __doc__,
        "threshold": thresh,
        "nperm": NPERM,
        "align_taus": list(ALIGN_TAUS),
        "blocks": {"1": list(BLOCKS), "3": list(BLOCKS), "4": list(EQ_BLOCKS)},
        "positions": {"1": "a", "3": "b", "4": "= (addition prompts only)"},
        "variables": {t: list(vs) for t, vs in variables.items()},
        "index": index,
        "readers": readers,
        "provenance": {
            "script": "notes/reader_writer/repr_repo/seed_periodicity.py",
            "commit": commit,
            "date": today,
            "source": str(SRC),
        },  # fmt: skip
    }
    _write_json(ROOT / "periodicity.json", doc)
    for t, vs in index.items():
        for v, ids in vs.items():
            print(f"t={t} {v}:", {tau: len(x) for tau, x in ids.items()})


if __name__ == "__main__":
    main()
