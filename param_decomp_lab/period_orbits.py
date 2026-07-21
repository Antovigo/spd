"""2D-FFT frequency-orbit decomposition of (a, b) operand grids — the period-purity math.

Shared by the offline `scripts/validation/score_period_separation.py` scorer and the
`eval_metrics.PeriodSeparation` training-time metric. Numpy-only: no torch / lab imports,
so it is importable from anywhere without circularity.
"""

import math
from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


def orbit_label(ka: int, kb: int, n: int) -> tuple[str, int]:
    """Direction label and integer period of the frequency orbit `(ka, kb)` on an n×n grid.

    `ka`/`kb` are the a-axis / b-axis frequencies with `kb` folded into `(-n/2, n/2]`.
    """
    if kb == 0:
        return "a", round(n / ka)
    if ka == 0:
        return "b", round(n / abs(kb))
    if ka == kb:
        return "a+b", round(n / ka)
    if ka == -kb:
        return "a-b", round(n / ka)
    return "mixed2d", round(n / math.gcd(ka, abs(kb)))


class Orbit(NamedTuple):
    """One conjugate frequency orbit: human label + integer period + power share, plus the
    reduced direction `direction` and fundamental frequency `freq` (`(ka, kb) = freq *
    direction`) that let harmonics of a common fundamental be pooled."""

    label: str
    period: int
    share: float
    direction: tuple[str | int, int]
    freq: int

    def is_harmonic_of(self, other: "Orbit") -> bool:
        return self.direction == other.direction and self.freq % other.freq == 0


def orbit_powers_2d(grid: NDArray[np.floating]) -> list[Orbit]:
    """Non-DC power per conjugate orbit of a square grid, descending by share.
    `grid` is `[b, a]`, so axis 1 carries the a-frequency."""
    n_b, n_a = grid.shape
    assert n_a == n_b, f"period labels assume a square grid, got {grid.shape}"
    n = n_a
    f = np.fft.fft2(grid - grid.mean())
    power = np.abs(f) ** 2
    orbits: dict[tuple[int, int], float] = defaultdict(float)
    for kb_raw in range(n):
        for ka_raw in range(n):
            if ka_raw == 0 and kb_raw == 0:
                continue
            ka = ka_raw - n if ka_raw > n // 2 else ka_raw
            kb = kb_raw - n if kb_raw > n // 2 else kb_raw
            key = (ka, kb) if (ka > 0 or (ka == 0 and kb > 0)) else (-ka, -kb)
            orbits[key] += float(power[kb_raw, ka_raw])
    total = sum(orbits.values())
    if total < 1e-12:
        return []
    ranked = sorted(orbits.items(), key=lambda kv: kv[1], reverse=True)
    return [
        Orbit(*orbit_label(ka, kb, n), p / total, (ka // g, kb // g), g)
        for (ka, kb), p in ranked
        for g in (math.gcd(ka, abs(kb)),)
    ]


def orbit_powers_marginals(grid: NDArray[np.floating], displayed: NDArray[np.bool_]) -> list[Orbit]:
    """1D-FFT power orbits of the two displayed-cell marginals, pooled and normalised.

    Fallback for ops whose prompts don't cover the full grid (subtraction's triangle):
    `f(a)` = nan-mean of CI over the displayed b's, and vice versa. Diagonal (`a±b`)
    structure is invisible to marginals — `+` rows are the comparable ones.
    """
    masked = np.where(displayed, grid, np.nan)
    with np.errstate(invalid="ignore"):
        marginals = (("a", np.nanmean(masked, axis=0)), ("b", np.nanmean(masked, axis=1)))
    powers: list[Orbit] = []
    for axis_label, marginal in marginals:
        marginal = marginal[~np.isnan(marginal)]
        if marginal.size < 4:
            continue
        centered = marginal - marginal.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2
        for k in range(1, power.shape[0]):
            powers.append(
                Orbit(axis_label, round(marginal.size / k), float(power[k]), (axis_label, 0), k)
            )
    total = sum(o.share for o in powers)
    if total < 1e-12:
        return []
    return [
        o._replace(share=o.share / total)
        for o in sorted(powers, key=lambda o: o.share, reverse=True)
    ]


def n_orbits_to_share(orbits: list[Orbit], target_share: float) -> int:
    """Number of top orbits needed to accumulate `target_share` of the non-DC power."""
    assert orbits, "no orbits"
    cum = np.cumsum([o.share for o in orbits])
    return int(np.searchsorted(cum, target_share) + 1)


def canonical_fundamentals(n: int) -> tuple[int, ...]:
    """Fundamental frequencies of the canonical period classes on a 1..n grid: the
    divisors of `n` up to `n/2` (period `T = n/f`; f=n would be period 1 = constant)."""
    return tuple(f for f in range(1, n // 2 + 1) if n % f == 0)


def period_class_shares(grid: NDArray[np.floating]) -> dict[int, float]:
    """Share of non-DC 2D-FFT power per canonical period class, `{T: share}`.

    A bin (ka, kb) (frequencies folded to `(-n/2, n/2]`) belongs to class `T = n/f` when
    its nonzero frequency magnitudes all equal `f` — i.e. the bins a *linear* read of
    period-T Fourier features can produce: `(f, 0)` (a-feature), `(0, f)` (b-feature),
    `(f, ±f)` (result/both-operand diagonals). All other bins are unclassified
    (broadband / non-canonical) and only contribute to the normalising total, so a
    component mixing several canonical periods shows several large shares while an
    aperiodic component shows none.
    """
    n_b, n_a = grid.shape
    assert n_a == n_b, f"period classes assume a square grid, got {grid.shape}"
    n = n_a
    power = np.abs(np.fft.fft2(grid - grid.mean())) ** 2
    total = float(power.sum()) - float(power[0, 0])
    shares: dict[int, float] = {n // f: 0.0 for f in canonical_fundamentals(n)}
    if total < 1e-12:
        return shares
    for kb_raw in range(n):
        for ka_raw in range(n):
            if ka_raw == 0 and kb_raw == 0:
                continue
            ka = abs(ka_raw - n if ka_raw > n // 2 else ka_raw)
            kb = abs(kb_raw - n if kb_raw > n // 2 else kb_raw)
            f = max(ka, kb)
            if n % f != 0 or (ka != 0 and ka != f) or (kb != 0 and kb != f):
                continue
            shares[n // f] += float(power[kb_raw, ka_raw]) / total
    return shares


def count_periods(shares: dict[int, float], theta: float) -> int:
    """Number of canonical period classes whose share clears `theta`."""
    return sum(1 for share in shares.values() if share >= theta)
