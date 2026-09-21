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

RANK_TOL = 1e-5


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
            case "b@add":
                return np.where(self.op == 0, self.b, -1)
            case "b@sub":
                return np.where(self.op == 1, self.b, -1)
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
"""Which quantities can be present at each position (a quantity needs its tokens). On the
pooled prompt set `b` is replaced by the op-conditional `b@add` and `b@sub` (`pooled_quantities`):
the second operand is identified separately per operation."""


def pooled_quantities(quantities: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for q in quantities:
        out.extend(("b@add", "b@sub") if q == "b" else (q,))
    return tuple(out)


def held_out_quantity(quantity: str) -> str:
    """Which value a hypothesis about `quantity` is held out on (`a`, `b`, or `either`)."""
    if quantity == "a":
        return "a"
    if quantity == "b" or quantity.startswith("b@"):
        return "b"
    return "either"


def indicator(values: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((values.size, n_classes), np.float64)
    out[np.arange(values.size), values] = 1.0
    return out


def _orth_complement(r_m: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Orthonormal basis of `col(r_m)` with the rank cut `RANK_TOL` relative to `scale`
    (default: the largest singular value; pass the pre-projection scale for a residual, or
    a single column that is entirely numerical noise would be kept)."""
    if r_m.shape[1] == 0:
        return r_m[:, :0]
    u_m, s, _ = np.linalg.svd(r_m, full_matrices=False)
    ref = float(s[0]) if scale is None else scale
    return u_m[:, s > RANK_TOL * max(ref, 1e-300)]


def _project_out(i_m: np.ndarray, lower: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """`i_m` minus its projection onto the SUM of the `lower` spans, and `i_m`'s own scale
    (largest singular value) for the rank cut. The lower spans are orthonormalised first:
    under a non-product prompt measure the pure parts of incomparable divisors (4 and 10,
    say) are not mutually orthogonal, so their concatenation is not a projector."""
    l_m = _orth_complement(np.concatenate(lower, axis=1))
    scale = float(np.linalg.norm(i_m, 2))
    return i_m - l_m @ (l_m.T @ i_m), scale


def _pure(i_m: np.ndarray, lower: list[np.ndarray]) -> np.ndarray:
    return _orth_complement(*_project_out(i_m, lower))


@dataclass(frozen=True)
class Hypothesis:
    """`Phi` (n x m, orthonormal, centred) on the fitting prompts, plus what re-evaluates the
    same functions on other prompts: for a periodic or direct part the residue weights `G`
    (`n_classes x m`, `Phi = I G` with `I` the class indicator), for the linear part the
    affine map `(mean, 1 / norm)` stored in `G`. `classes(labels)` maps a prompt to its class
    index (residue, or value offset for the direct and linear parts; -1 = not supported)."""

    quantity: str
    kind: str
    """`periodic`, `direct` or `linear`."""
    period: int | None
    Phi: np.ndarray
    G: np.ndarray
    value_offset: int
    n_classes: int

    @property
    def name(self) -> str:
        match self.kind:
            case "periodic":
                return f"{self.quantity}:{self.period}"
            case "linear":
                return f"{self.quantity}:lin"
            case _:
                return f"{self.quantity}:direct"

    @property
    def dim(self) -> int:
        return int(self.Phi.shape[1])

    def classes(self, labels: Labels) -> np.ndarray:
        values = labels.quantity(self.quantity)
        if self.kind == "periodic":
            assert self.period is not None
            return np.where(values >= 0, values % self.period, -1)
        return np.where(values >= 0, values - self.value_offset, -1)

    def evaluate(self, labels: Labels) -> np.ndarray:
        """The hypothesis functions on another prompt set (n' x m). A class never seen where
        the hypothesis was built (a value outside a direct part's range) evaluates to 0."""
        values = labels.quantity(self.quantity)
        support = values >= 0
        out = np.zeros((labels.n, self.dim))
        if self.kind == "linear":
            if self.dim:
                out[support, 0] = (values[support] - self.G[0, 0]) * self.G[1, 0]
            return out
        cls = self.classes(labels)
        valid = (cls >= 0) & (cls < self.n_classes)
        out[valid] = indicator(cls[valid], self.n_classes) @ self.G
        return out


def additive_space(labels: Labels) -> np.ndarray:
    """Functions of `a` alone plus functions of `b` alone (`b` per operation): what a
    result hypothesis must be orthogonal to in order to count as a computed quantity."""
    cols = [indicator(labels.a - labels.a.min(), int(labels.a.max() - labels.a.min() + 1))]
    b0 = labels.b - labels.b.min()
    nb = int(b0.max() + 1)
    for op in np.unique(labels.op):
        cols.append(indicator(b0, nb) * (labels.op == op)[:, None])
    return np.concatenate(cols, axis=1)


INTERACTION_QUANTITIES = ("res", "cross", "sum", "diff")


def pure_parts(
    labels: Labels,
    quantity: str,
    periods: tuple[int, ...] = DIVISORS,
    extra_lower: list[np.ndarray] | None = None,
) -> list[Hypothesis]:
    """The linear part, `P_{Q,tau}` for every period (Gram-Schmidt in divisor order, every
    part orthogonal to the linear one), and `D_Q` when the quantity's range exceeds one
    period. `extra_lower` spans are projected out of everything (the additive space, for a
    result quantity)."""
    values = labels.quantity(quantity)
    n = values.size
    support = values >= 0
    # The "constant" of an op-conditional quantity is its support indicator (the op itself),
    # so every pure part is orthogonal to the op hypothesis as well as to the constant.
    const = np.full((n, 1), 1.0 / np.sqrt(n))
    if not support.all():
        const = _orth_complement(np.stack([np.ones(n), support.astype(np.float64)], axis=1))
    base = [const] + (extra_lower or [])

    def masked_indicator(cls: np.ndarray, n_classes: int) -> np.ndarray:
        out = np.zeros((n, n_classes))
        out[support] = indicator(cls[support], n_classes)
        return out

    lo, hi = int(values[support].min()), int(values[support].max())
    out: list[Hypothesis] = []
    mean = float(values[support].mean())
    raw = np.where(support, values - mean, 0.0)[:, None]
    lin_phi = _pure(raw, base)
    scale = float(lin_phi[:, 0] @ raw[:, 0]) if lin_phi.shape[1] else 1.0
    linear = Hypothesis(
        quantity,
        "linear",
        None,
        lin_phi,
        np.array([[mean], [1.0 / scale if scale else 0.0]]),
        lo,
        hi - lo + 1,
    )
    out.append(linear)
    base = base + [linear.Phi]

    parts: dict[int, Hypothesis] = {}
    for tau in periods:
        i_m = masked_indicator(np.where(support, values % tau, 0), tau)
        lower = base + [parts[d].Phi for d in periods if tau % d == 0 and d < tau]
        phi_m = _pure(i_m, lower)
        g_m = np.linalg.lstsq(i_m, phi_m, rcond=None)[0] if phi_m.shape[1] else np.zeros((tau, 0))
        parts[tau] = Hypothesis(quantity, "periodic", tau, phi_m, g_m, 0, tau)
    out.extend(parts.values())
    if hi - lo + 1 > max(periods):
        i_m = masked_indicator(np.where(support, values - lo, 0), hi - lo + 1)
        phi_m = _pure(i_m, base + [h.Phi for h in parts.values()])
        g_m = (
            np.linalg.lstsq(i_m, phi_m, rcond=None)[0]
            if phi_m.shape[1]
            else np.zeros((i_m.shape[1], 0))
        )
        out.append(Hypothesis(quantity, "direct", None, phi_m, g_m, lo, hi - lo + 1))
    return out


def op_hypothesis(labels: Labels) -> Hypothesis:
    """The 1-D operation indicator, centred and normalised; classes 0/1."""
    i_m = indicator(labels.op, 2)
    n = labels.n
    const = np.full((n, 1), 1.0 / np.sqrt(n))
    phi_m = _pure(i_m, [const])
    g_m = np.linalg.lstsq(i_m, phi_m, rcond=None)[0]
    return Hypothesis("op", "periodic", 2, phi_m, g_m, 0, 2)


def build_hypotheses(labels: Labels, quantities: tuple[str, ...]) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for q in quantities:
        if q == "op":
            if np.unique(labels.op).size > 1:
                out.append(op_hypothesis(labels))
            continue
        extra = [additive_space(labels)] if q in INTERACTION_QUANTITIES else None
        out.extend(pure_parts(labels, q, extra_lower=extra))
    return [h for h in out if h.dim > 0]


def lattice_overlap(hyps: list[Hypothesis]) -> float:
    """Largest cosine between two hypothesis spaces of the SAME quantity (0 on a product
    measure; reported as the order-dependence diagnostic of plan section 2)."""
    worst = 0.0
    for i, h in enumerate(hyps):
        for g in hyps[i + 1 :]:
            if g.quantity == h.quantity and h.dim and g.dim:
                worst = max(worst, float(np.abs(h.Phi.T @ g.Phi).max()))
    return worst


def linear_direction(labels: Labels, quantity: str) -> np.ndarray:
    """The centred, unit-norm linear function of `Q` (the number-line probe)."""
    raw = labels.quantity(quantity)
    support = raw >= 0
    v = np.where(support, raw - raw[support].mean(), 0.0)
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
        match held_out_quantity(quantity):
            case "a":
                return a_out
            case "b":
                return b_out
            case _:
                return a_out | b_out
