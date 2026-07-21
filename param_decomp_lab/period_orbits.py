"""Canonical-period-class decomposition of (a, b) operand-grid spectra.

Shared by the `eval_metrics.PeriodSeparation` training-time metric and its offline twin
`scripts/validation/score_period_separation.py`. Numpy-only: no torch / lab imports, so
it is importable from anywhere without circularity.
"""

import numpy as np
from numpy.typing import NDArray


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
