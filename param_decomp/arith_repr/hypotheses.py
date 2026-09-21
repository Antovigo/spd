"""The preregistered hypothesis lattice of `notes/arith_representations/plan.md` (section 2).

A hypothesis is a subspace of functions on a prompt set: for a quantity `Q` and a period
`tau | 100`, the pure period-`tau` part `P_{Q,tau}` = functions of `Q mod tau` orthogonal to
every coarser period (and the constant); for a quantity whose range exceeds 100, the direct
part `D_Q` = functions of `Q` orthogonal to every periodic part. Every basis is built on the
prompts it is fitted on and re-evaluated on other prompts through its residue-weight matrix
`G` (`Phi = I G`, `I` the residue indicator matrix), so a fit generalises to prompts whose
values were never seen exactly as the hypothesis says it should."""

from dataclasses import dataclass

import numpy as np

DIVISORS: tuple[int, ...] = (2, 4, 5, 10, 20, 25, 50, 100)
"""The periods, in divisor order; `1` (the constant) is handled by centering."""

RANK_TOL = 1e-6


def divisor_order_check() -> None:
    for i, tau in enumerate(DIVISORS):
        for d in DIVISORS[:i]:
            assert d < tau, (d, tau)


@dataclass(frozen=True)
class Labels:
    """One prompt set's labels; `op` is 0 for add, 1 for sub."""

    op: np.ndarray
    a: np.ndarray
    b: np.ndarray

    @property
    def n(self) -> int:
        return int(self.a.size)

    def quantity(self, name: str) -> np.ndarray:
        match name:
            case "a":
                return self.a
            case "b":
                return self.b
            case "op":
                return self.op
            case "res":
                return np.where(self.op == 0, self.a + self.b, self.a - self.b)
            case "cross":
                return np.where(self.op == 0, self.a - self.b, self.a + self.b)
            case "sum":
                return self.a + self.b
            case "diff":
                return self.a - self.b
            case _:
                raise KeyError(name)

    def subset(self, mask: np.ndarray) -> "Labels":
        return Labels(self.op[mask], self.a[mask], self.b[mask])


QUANTITIES_BY_POSITION: dict[int, tuple[str, ...]] = {
    1: ("a",),
    2: ("a", "op"),
    3: ("a", "op", "b", "res", "cross"),
    4: ("a", "op", "b", "res", "cross"),
}
"""Which quantities can be present at each position (a quantity needs its tokens)."""


def indicator(values: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((values.size, n_classes), np.float64)
    out[np.arange(values.size), values] = 1.0
    return out


def _orth_complement(r_m: np.ndarray) -> np.ndarray:
    """Orthonormal basis of `col(r_m)` with the rank cut `RANK_TOL`."""
    if r_m.shape[1] == 0:
        return r_m[:, :0]
    u_m, s, _ = np.linalg.svd(r_m, full_matrices=False)
    keep = s > RANK_TOL * max(float(s[0]) if s.size else 0.0, 1e-300) * np.sqrt(r_m.shape[0])
    return u_m[:, keep]


@dataclass(frozen=True)
class Hypothesis:
    """`Phi` (n x m, orthonormal, centred) on the fitting prompts and the residue weights `G`
    (n_classes x m) that evaluate the same functions elsewhere. `classes(labels)` maps a
    prompt to its class index (its residue, or its value offset for a direct part)."""

    quantity: str
    period: int | None
    """`None` = the direct (non-periodic) part."""
    Phi: np.ndarray
    G: np.ndarray
    value_offset: int

    @property
    def name(self) -> str:
        return f"{self.quantity}:{'direct' if self.period is None else self.period}"

    @property
    def dim(self) -> int:
        return int(self.Phi.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.G.shape[0])

    def classes(self, labels: Labels) -> np.ndarray:
        values = labels.quantity(self.quantity)
        if self.period is None:
            return values - self.value_offset
        return values % self.period

    def evaluate(self, labels: Labels) -> np.ndarray:
        """The hypothesis functions on another prompt set (n' x m). A class never seen where
        the hypothesis was built (a value outside a direct part's range) evaluates to 0."""
        cls = self.classes(labels)
        valid = (cls >= 0) & (cls < self.n_classes)
        out = np.zeros((labels.n, self.dim))
        out[valid] = indicator(cls[valid], self.n_classes) @ self.G
        return out


def pure_parts(
    labels: Labels, quantity: str, periods: tuple[int, ...] = DIVISORS
) -> list[Hypothesis]:
    """`P_{Q,tau}` for every period, built by Gram-Schmidt in divisor order, plus `D_Q` when
    the quantity's range exceeds one period."""
    values = labels.quantity(quantity)
    n = values.size
    const = np.full((n, 1), 1.0 / np.sqrt(n))
    parts: dict[int, Hypothesis] = {}
    for tau in periods:
        i_m = indicator(values % tau, tau)
        lower = [const] + [parts[d].Phi for d in periods if tau % d == 0 and d < tau]
        l_m = np.concatenate(lower, axis=1)
        r_m = i_m - l_m @ (l_m.T @ i_m)
        phi_m = _orth_complement(r_m)
        g_m = np.linalg.lstsq(i_m, phi_m, rcond=None)[0] if phi_m.shape[1] else np.zeros((tau, 0))
        parts[tau] = Hypothesis(quantity, tau, phi_m, g_m, 0)
    out = list(parts.values())
    lo, hi = int(values.min()), int(values.max())
    if hi - lo + 1 > max(periods):
        i_m = indicator(values - lo, hi - lo + 1)
        l_m = np.concatenate([const] + [h.Phi for h in out], axis=1)
        r_m = i_m - l_m @ (l_m.T @ i_m)
        phi_m = _orth_complement(r_m)
        g_m = (
            np.linalg.lstsq(i_m, phi_m, rcond=None)[0]
            if phi_m.shape[1]
            else np.zeros((i_m.shape[1], 0))
        )
        out.append(Hypothesis(quantity, None, phi_m, g_m, lo))
    return out


def op_hypothesis(labels: Labels) -> Hypothesis:
    """The 1-D operation indicator, centred and normalised; classes 0/1."""
    i_m = indicator(labels.op, 2)
    n = labels.n
    const = np.full((n, 1), 1.0 / np.sqrt(n))
    r_m = i_m - const @ (const.T @ i_m)
    phi_m = _orth_complement(r_m)
    g_m = np.linalg.lstsq(i_m, phi_m, rcond=None)[0]
    return Hypothesis("op", 2, phi_m, g_m, 0)


def build_hypotheses(labels: Labels, quantities: tuple[str, ...]) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for q in quantities:
        if q == "op":
            if np.unique(labels.op).size > 1:
                out.append(op_hypothesis(labels))
            continue
        out.extend(pure_parts(labels, q))
    return [h for h in out if h.dim > 0]


def linear_direction(labels: Labels, quantity: str) -> np.ndarray:
    """The centred, unit-norm linear function of `Q` (the number-line probe)."""
    v = labels.quantity(quantity).astype(np.float64)
    v = v - v.mean()
    return v / max(np.linalg.norm(v), 1e-300)


@dataclass(frozen=True)
class ValueFolds:
    """Held-out `a` values and `b` values per fold (plan section 3)."""

    a_out: tuple[np.ndarray, ...]
    b_out: tuple[np.ndarray, ...]

    @staticmethod
    def make(n_folds: int, values: np.ndarray, seed: int) -> "ValueFolds":
        rng = np.random.default_rng(seed)
        pa, pb = rng.permutation(values), rng.permutation(values)
        return ValueFolds(
            tuple(np.sort(x) for x in np.array_split(pa, n_folds)),
            tuple(np.sort(x) for x in np.array_split(pb, n_folds)),
        )

    def train_mask(self, fold: int, labels: Labels) -> np.ndarray:
        return ~np.isin(labels.a, self.a_out[fold]) & ~np.isin(labels.b, self.b_out[fold])

    def test_mask(self, fold: int, labels: Labels, quantity: str) -> np.ndarray:
        """Where a hypothesis about `quantity` is scored: prompts whose relevant value was
        never seen in training (`a` for `a`, `b` for `b`, either for the rest)."""
        a_out = np.isin(labels.a, self.a_out[fold])
        b_out = np.isin(labels.b, self.b_out[fold])
        match quantity:
            case "a":
                return a_out
            case "b":
                return b_out
            case _:
                return a_out | b_out
