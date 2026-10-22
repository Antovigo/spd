"""Periodicity index of the readers of blocks 0..16 at the operand positions: periodicity.json.

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

Label: periods = [tau > 1 with share_tau >= THRESH (= meta.response_threshold), share_tau > null_tau, and, for
tau in {20, 25, 50, 100}, share_tau > align_tau].
A reader is listed under every such tau (a set of additive parts, not one period): the indicator of x mod 10 = 3
has parts of exact period 2, 5 and 10 and is listed under all three; minimal_period is their lcm.

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


def main() -> None:
    repo = Repo()
    thresh = repo.meta["response_threshold"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()  # fmt: skip
    today = datetime.date.today().isoformat()
    readers: dict[str, dict] = {}
    for b in BLOCKS:
        rds = [rd for rd in repo.readers.values() if rd["block"] == b]
        z = np.load(SRC / f"L{b}" / "comps.npz")
        col = {
            (str(k), int(i)): j for j, (k, i) in enumerate(zip(z["kind"], z["index"], strict=True))
        }
        cols = np.asarray([col[(rd["kind"], rd["component"])] for rd in rds])
        vnorm = np.asarray([np.linalg.norm(z[f"V_{rd['kind']}"][:, list(z["index"][z["kind"] == rd["kind"]])
                            .index(rd["component"])]) for rd in rds])  # fmt: skip
        inner = np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")
        for t in TS:
            T = np.asarray(inner[:, t - 1, :], np.float64)[:, cols] / vnorm
            F = operand_means(T, t)
            S = shares(F)
            Q = null(F, "perm")
            QA = null(F, "align")
            coef, *_ = np.linalg.lstsq(POLY, F, rcond=None)
            smooth = 1 - ((F - POLY @ coef) ** 2).sum(0) / (F**2).sum(0)
            for j, rd in enumerate(rds):
                periods = [tau for i, tau in enumerate(TAUS)
                           if tau > 1 and S[i, j] >= thresh and S[i, j] > Q[i, j]
                           and (tau not in ALIGN_TAUS or S[i, j] > QA[i, j])]  # fmt: skip
                entry = readers.setdefault(rd["id"], {"block": b, "kind": rd["kind"],
                                                      "component": rd["component"], "positions": {}})  # fmt: skip
                entry["positions"][str(t)] = {
                    "periods": periods,
                    "minimal_period": lcm(*periods) if periods else None,
                    "share": {str(tau): float(S[i, j]) for i, tau in enumerate(TAUS)},
                    "share_rest": float(S[-1, j]),
                    "null_p99": {str(tau): float(Q[i, j]) for i, tau in enumerate(TAUS) if tau > 1},
                    "align_null_p99": {
                        str(tau): float(QA[i, j]) for i, tau in enumerate(TAUS) if tau in ALIGN_TAUS
                    },
                    "smooth_r2": float(smooth[j]),
                    "rms": float(np.sqrt((F[:, j] ** 2).mean())),
                }
        print(f"L{b}: {len(rds)} readers", flush=True)

    index = {
        str(t): {str(tau): sorted((rid for rid, e in readers.items() if tau in e["positions"][str(t)]["periods"]),
                                  key=lambda s: (readers[s]["block"], readers[s]["kind"], readers[s]["component"]))
                 for tau in TAUS if tau > 1}
        for t in TS
    }  # fmt: skip
    doc = {
        "definitions": __doc__,
        "threshold": thresh,
        "nperm": NPERM,
        "align_taus": list(ALIGN_TAUS),
        "blocks": list(BLOCKS),
        "positions": {"1": "a", "3": "b"},
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
    for t in TS:
        print(f"t={t}:", {tau: len(v) for tau, v in index[str(t)].items()})


if __name__ == "__main__":
    main()
