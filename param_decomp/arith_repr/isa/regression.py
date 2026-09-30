"""Sphere pursuit without global whitening: quantities are scored after standardising them inside their
own frame, and each quantity found is removed from the readers by regression.

The readers are kept in their raw units, on the principal directions retained at the read point:
X (N, r), column j with variance lambda_j (not rescaled). A candidate quantity is z = X W for any
r x k matrix W. Its score is the norm-constancy J of z after standardising z to identity covariance
(so a circle read loudly on one axis and quietly on the other is not penalised for being an ellipse).
Once accepted, the quantity is regressed out of X: X <- X - z B with B = lstsq(z, X), which removes
from every reader everything correlated with z. Gradient steps move W in raw units, so directions of
small variance are explored slowly; nothing is ever divided by a small variance except inside the
k x k covariance of the candidate itself."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.pursuit import norm_var, split

jax.config.update("jax_enable_x64", True)
RIDGE = 1e-9


@dataclass
class RegQuantity:
    z: np.ndarray  # (N, k) standardised values on every prompt
    J: float  # norm-constancy on held-out prompts
    seed: int  # reader that seeded it


def _standardise(z: jax.Array) -> jax.Array:
    z = z - z.mean(0)
    k = z.shape[1]
    cov = z.T @ z / z.shape[0]
    L = jnp.linalg.cholesky(cov + RIDGE * jnp.trace(cov) / k * jnp.eye(k))
    return jax.scipy.linalg.solve_triangular(L, z.T, lower=True).T


def _score(W: jax.Array, X: jax.Array) -> jax.Array:
    zs = _standardise(X @ W)
    q = (zs**2).sum(1)
    return q.var() / (2 * W.shape[1])


_grad: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]] = jax.jit(
    jax.value_and_grad(_score)
)


def fit_frame(
    X: np.ndarray, W0: np.ndarray, iters: int = 300, lr: float = 0.05
) -> tuple[np.ndarray, float]:
    """Adam on the raw-unit frame W (columns rescaled to unit length after each step: J does not
    depend on the scale of W's columns)."""
    w = jnp.asarray(W0)
    Xj = jnp.asarray(X)
    m = jnp.zeros_like(w)
    v = jnp.zeros_like(w)
    for t in range(1, iters + 1):
        _, g = _grad(w, Xj)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        w = w - lr * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-12)
        w = w / jnp.linalg.norm(w, axis=0, keepdims=True)
    return np.asarray(w), float(_score(w, Xj))


def best_frame(X: np.ndarray, seed_dir: np.ndarray, kmax: int, rng: np.random.Generator,
               restarts: int = 4) -> tuple[np.ndarray, float]:  # fmt: skip
    r = X.shape[1]
    best_W = seed_dir[:, None] / np.linalg.norm(seed_dir)
    best_J = float(_score(jnp.asarray(best_W), jnp.asarray(X)))
    for k in range(1, min(kmax, r) + 1):
        for _ in range(restarts if k > 1 else 1):
            W0 = np.concatenate(
                [best_W[:, :1] if k > 1 else best_W, rng.standard_normal((r, k - 1))], 1
            )
            W, J = fit_frame(X, W0 / np.linalg.norm(W0, axis=0, keepdims=True))
            if best_J - 0.02 > J:  # a larger k must earn its keep
                best_W, best_J = W, J
    return best_W, best_J


def pursue(X: np.ndarray, reads: np.ndarray, order: np.ndarray, fit: np.ndarray, test: np.ndarray,
           tol: float = 0.5, kmax: int = 4, explained: float = 0.5, seed: int = 0) -> list[RegQuantity]:  # fmt: skip
    """X: (N, r) readers on the retained principal directions, raw units; reads: (n, r), reader c's
    activation is X @ reads[c]; order: readers to try as seeds; fit / test: prompt rows (distinct
    prompts) on which frames are fitted and accepted."""
    rng = np.random.default_rng(seed)
    Xd: np.ndarray = X.copy()
    found: list[RegQuantity] = []
    for c in order:
        u = reads[c]
        if (Xd[fit] @ u).var() < explained * (X[fit] @ u).var():
            continue  # this reader is already mostly explained
        W, _ = best_frame(Xd[fit], u, kmax, rng)
        z = Xd @ W
        z = z - z[fit].mean(0)
        cov = z[fit].T @ z[fit] / len(fit)
        k = W.shape[1]
        L = np.linalg.cholesky(cov + RIDGE * np.trace(cov) / k * np.eye(k))
        zs = np.linalg.solve(L, z.T).T
        if norm_var(zs[test]) >= tol:
            continue
        for part in split(zs[fit], np.eye(k), rng, tol):
            zp = zs @ part
            Jp = norm_var(zp[test])
            if Jp < tol:
                found.append(RegQuantity(zp, Jp, int(c)))
        B = np.linalg.lstsq(zs[fit], Xd[fit], rcond=None)[0]
        Xd = Xd - zs @ B
    return found
