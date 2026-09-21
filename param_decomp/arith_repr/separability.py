"""Separability judged by the model's own reads (plan section 4): per alive component `c`
the overlap `s_A(c) = ||P_A v_c|| / ||v_c||` with each recovered subspace, the null of a random
direction in the read subspace, the separability graph and its connected components
(clusters), and the per-cluster geometry (union subspace, PCA of the class centroids)."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from param_decomp.arith_repr.fit import HypothesisFit, orthonormal_union
from param_decomp.arith_repr.hypotheses import Labels

READS_Q = 0.999
IGNORES_Q = 0.99


def overlap(S: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`(n_components,)`: `||S^T v_c|| / ||v_c||` for the columns of `v` (`k x n_components`)."""
    norms = np.linalg.norm(v, axis=0)
    return np.linalg.norm(S.T @ v, axis=0) / np.maximum(norms, 1e-12)


def null_quantile(dim_s: int, k: int, q: float) -> float:
    """`s` of a uniformly random direction in a `k`-dimensional space against an
    `m`-dimensional subspace: `s^2 ~ Beta(m/2, (k - m)/2)`."""
    if dim_s >= k:
        return 1.0
    return float(np.sqrt(beta.ppf(q, dim_s / 2, (k - dim_s) / 2)))


@dataclass
class SeparabilityResult:
    names: list[str]
    reads: np.ndarray
    """`(n_hyp, n_components)` bool."""
    ignores: np.ndarray
    separable: np.ndarray
    """`(n_hyp, n_hyp)` bool, symmetric; the diagonal is False."""
    clusters: list[list[int]]
    """Connected components of the NOT-separable graph over the READ hypotheses, as
    hypothesis indices; an unread hypothesis (no component reads it) is its own cluster
    and is listed in `unread` — nothing points at it, so it can neither separate nor merge."""
    unread: list[int]
    overlaps: np.ndarray
    """`(n_hyp, n_components)` the raw `s_A(c)`."""


def separability(
    fits: list[HypothesisFit], v: np.ndarray, k: int, component_mask: np.ndarray | None = None
) -> SeparabilityResult:
    """`v` is `(k, n_components)` (alive reads in read coordinates); `component_mask`
    restricts to one kind. Hypotheses with an empty `S_H` are absent from the graph."""
    live = [f for f in fits if f.S.shape[1] > 0]
    names = [f.hypothesis.name for f in live]
    if component_mask is not None:
        v = v[:, component_mask]
    n_h, n_c = len(live), v.shape[1]
    s = np.zeros((n_h, n_c))
    reads = np.zeros((n_h, n_c), bool)
    ignores = np.zeros((n_h, n_c), bool)
    for i, f in enumerate(live):
        s[i] = overlap(f.S, v)
        m = f.S.shape[1]
        reads[i] = s[i] > null_quantile(m, k, READS_Q)
        ignores[i] = s[i] < null_quantile(m, k, IGNORES_Q)
    sep = np.zeros((n_h, n_h), bool)
    for i in range(n_h):
        for j in range(i + 1, n_h):
            a_not_b = bool(np.any(reads[i] & ignores[j]))
            b_not_a = bool(np.any(reads[j] & ignores[i]))
            sep[i, j] = sep[j, i] = a_not_b and b_not_a
    # Connected components of the complement graph over the read hypotheses.
    unread = [i for i in range(n_h) if not reads[i].any()]
    seen = np.zeros(n_h, bool)
    clusters: list[list[int]] = [[i] for i in unread]
    seen[unread] = True
    for start in range(n_h):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in range(n_h):
                if not seen[j] and not sep[i, j]:
                    seen[j] = True
                    stack.append(j)
        clusters.append(sorted(comp))
    return SeparabilityResult(names, reads, ignores, sep, clusters, unread, s)


@dataclass
class ClusterGeometry:
    members: list[str]
    dim: int
    spectrum: np.ndarray
    """Singular values of the stacked class centroids in the cluster subspace."""
    axes: np.ndarray
    """`(k, n_axes)` the top principal axes of the centroids, in read coordinates."""
    centroids: dict[str, np.ndarray]
    """Per member: `(n_classes, n_axes)` class centroids in the principal axes."""
    counts: dict[str, np.ndarray]
    norms: dict[str, np.ndarray]
    """Per member: the full norm of each class centroid in the cluster subspace."""


def class_centroids(
    Y: np.ndarray, classes: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(classes, minlength=n_classes)
    sums = np.zeros((n_classes, Y.shape[1]))
    np.add.at(sums, classes, Y)
    return sums / np.maximum(counts, 1)[:, None], counts


def cluster_geometry(
    fits: list[HypothesisFit],
    members: list[int],
    Y: np.ndarray,
    labels: Labels,
    n_axes: int,
) -> ClusterGeometry:
    live = [f for f in fits if f.S.shape[1] > 0]
    chosen = [live[i] for i in members]
    S = orthonormal_union([f.S for f in chosen])
    Yc = Y - Y.mean(axis=0)
    stacked, per_member, counts_by, names = [], {}, {}, []
    for f in chosen:
        h = f.hypothesis
        cls = h.classes(labels)
        cent, counts = class_centroids(Yc @ S, cls, h.n_classes)
        stacked.append(cent[counts > 0])
        per_member[h.name] = cent
        counts_by[h.name] = counts
        names.append(h.name)
    stack = np.concatenate(stacked)
    _, s, Wt = np.linalg.svd(stack - stack.mean(axis=0), full_matrices=False)
    axes = Wt.T[:, :n_axes]
    return ClusterGeometry(
        members=names,
        dim=int(S.shape[1]),
        spectrum=s,
        axes=S @ axes,
        centroids={n: per_member[n] @ axes for n in names},
        counts=counts_by,
        norms={n: np.linalg.norm(per_member[n], axis=1) for n in names},
    )


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between two subspaces (orthonormal bases)."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.zeros(0)
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0, 1)
