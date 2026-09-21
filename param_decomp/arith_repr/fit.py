"""Fitting hypotheses to read-basis coordinates (plan section 3): marginal and unique
explained energy, the value-held-out direction test with its permutation null, and the
recovered subspace `S_H` (the directions that generalise).

Hypothesis bases are built once per fold (and once per null replicate, one shared label
permutation per replicate), then every hypothesis is scored against them."""

from dataclasses import dataclass, field

import numpy as np

from param_decomp.arith_repr.hypotheses import (
    Hypothesis,
    Labels,
    ValueFolds,
    build_hypotheses,
    lattice_overlap,
    linear_direction,
)

RANK_TOL = 1e-8


def orthonormal_union(bases: list[np.ndarray]) -> np.ndarray:
    """Orthonormal basis of the sum of the spans (columns), rank-cut."""
    bases = [B for B in bases if B.shape[1]]
    if not bases:
        return np.zeros((0, 0))
    A = np.concatenate(bases, axis=1)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > RANK_TOL * max(float(s[0]), 1e-300) * 10]


@dataclass
class HypothesisFit:
    hypothesis: Hypothesis
    marginal: float
    """`||Phi^T Y||^2 / ||Y||^2` on the full (centred) data."""
    unique: float
    """Energy lost when this hypothesis is dropped from the joint fit."""
    singular_values: np.ndarray
    """Of `M = Phi^T Y`, i.e. the energy `sigma_i^2` along each fitted direction."""
    directions: np.ndarray
    """`(k, m)` right singular vectors of `M`: the fitted directions in read coordinates."""
    heldout_r2: np.ndarray
    """Per direction: pooled over folds, `1 - ||(Y - Yhat) w||^2 / ||Y w||^2` on the test
    prompts of the hypothesis's quantity."""
    kept: np.ndarray
    """Per direction: held-out R^2 above the permutation null's 99th percentile."""
    linear_r2: float
    """Energy of the centred linear function of the quantity inside `col(Phi)`, over its
    total energy — how much of a number-line code this part carries."""
    total: float = field(default=1.0, repr=False)

    @property
    def S(self) -> np.ndarray:
        """`(k, dim S_H)` orthonormal: the kept directions."""
        return self.directions[:, self.kept]

    @property
    def generalising_energy(self) -> float:
        """Energy of the kept directions over `||Y||^2` (the presence score)."""
        return float(np.sum(self.singular_values[self.kept] ** 2) / self.total)


@dataclass(frozen=True)
class FoldFit:
    """One fold's training-side objects: the hypotheses built on the training prompts
    (possibly label-permuted) and their fitted means."""

    train: np.ndarray
    mean: np.ndarray
    by_name: dict[str, tuple[Hypothesis, np.ndarray]]
    """name -> (hypothesis on the training prompts, `M = Phi^T Y_train`)."""


def _fold_fit(
    Y: np.ndarray,
    labels: Labels,
    train: np.ndarray,
    quantities: tuple[str, ...],
    perm: np.ndarray | None,
) -> FoldFit:
    train_labels = labels.subset(train)
    if perm is not None:
        train_labels = Labels(train_labels.op[perm], train_labels.a[perm], train_labels.b[perm])
    mean = Y[train].mean(axis=0)
    Yt = Y[train] - mean
    by_name = {h.name: (h, h.Phi.T @ Yt) for h in build_hypotheses(train_labels, quantities)}
    return FoldFit(train, mean, by_name)


def _heldout_r2(
    Y: np.ndarray,
    labels: Labels,
    h: Hypothesis,
    W: np.ndarray,
    folds: ValueFolds,
    fold_fits: list[FoldFit],
) -> tuple[np.ndarray, np.ndarray]:
    """Per direction of `W`: the summed held-out residual and total energies over folds."""
    num = np.zeros(W.shape[1])
    den = np.zeros(W.shape[1])
    for f, ff in enumerate(fold_fits):
        test = folds.test_mask(f, labels, h.quantity)
        Ytest = Y[test] - ff.mean
        got = ff.by_name.get(h.name)
        if got is None:
            Yhat = np.zeros_like(Ytest)
        else:
            h_train, M = got
            Yhat = h_train.evaluate(labels.subset(test)) @ M
        num += np.sum(((Ytest - Yhat) @ W) ** 2, axis=0)
        den += np.sum((Ytest @ W) ** 2, axis=0)
    return num, den


def fit_hypotheses(
    Y_raw: np.ndarray,
    labels: Labels,
    quantities: tuple[str, ...],
    folds: ValueFolds,
    n_null: int,
    seed: int,
) -> tuple[list[HypothesisFit], dict[str, float]]:
    """Every hypothesis at one (read point, position, operation set). `Y` is `(n, k)`,
    centred here. Returns the fits and summary energies."""
    Y = Y_raw - Y_raw.mean(axis=0)
    total = float(np.sum(Y**2))
    hyps = build_hypotheses(labels, quantities)
    joint = orthonormal_union([h.Phi for h in hyps])
    joint_energy = float(np.sum((joint.T @ Y) ** 2)) if joint.size else 0.0
    rng = np.random.default_rng(seed)
    n_folds = len(folds.a_out)
    real_fits = [
        _fold_fit(Y, labels, folds.train_mask(f, labels), quantities, None) for f in range(n_folds)
    ]
    null_fits = [
        [
            _fold_fit(
                Y,
                labels,
                folds.train_mask(f, labels),
                quantities,
                rng.permutation(int(folds.train_mask(f, labels).sum())),
            )
            for f in range(n_folds)
        ]
        for _ in range(n_null)
    ]
    fits: list[HypothesisFit] = []
    null_pool: list[np.ndarray] = []
    for h in hyps:
        M = h.Phi.T @ Y
        _, s, Wt = np.linalg.svd(M, full_matrices=False)
        W = Wt.T
        others = orthonormal_union([g.Phi for g in hyps if g is not h])
        without = float(np.sum((others.T @ Y) ** 2)) if others.size else 0.0
        num, den = _heldout_r2(Y, labels, h, W, folds, real_fits)
        r2 = 1.0 - num / np.maximum(den, 1e-300)
        for replicate in null_fits:
            num_n, den_n = _heldout_r2(Y, labels, h, W, folds, replicate)
            null_pool.append(1.0 - num_n / np.maximum(den_n, 1e-300))
        lin_r2 = 0.0
        if h.quantity != "op":
            lin = linear_direction(labels, h.quantity)
            lin_r2 = float(np.sum((h.Phi.T @ lin) ** 2))
        fits.append(
            HypothesisFit(
                hypothesis=h,
                marginal=float(np.sum(M**2)) / total,
                unique=(joint_energy - without) / total,
                singular_values=s,
                directions=W,
                heldout_r2=r2,
                kept=np.zeros(s.size, bool),
                linear_r2=lin_r2,
                total=total,
            )
        )
    threshold = float(np.percentile(np.concatenate(null_pool), 99)) if null_pool else 0.0
    for fit in fits:
        fit.kept = fit.heldout_r2 > threshold
    summary = {
        "total_energy": total,
        "joint_energy": joint_energy / total if total else 0.0,
        "null_threshold": threshold,
        "n_hypotheses": len(fits),
        "lattice_overlap": lattice_overlap(hyps),
    }
    return fits, summary
