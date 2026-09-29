"""How much rms error the -05 decomposition and the original model tolerate at the inner norms
(blocks 0-30: t = 0..61; the last block and the final norm are excluded).

    python -m param_decomp.arith_repr.rmsnorm.robust forced   # decomposed model forced to the original's rms on block windows
    python -m param_decomp.arith_repr.rmsnorm.robust noise    # random per-token rms noise, both models
    python -m param_decomp.arith_repr.rmsnorm.robust scale    # coherent rms scaling, all inner norms and one at a time

KL is against the SAME model run clean (own norms), so it is the model's sensitivity to the perturbation."""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.rmsnorm.model import NormHook, NormModel, forced
from param_decomp.arith_repr.rmsnorm.run import (
    Conds,
    HookFactory,
    N,
    load_base,
    log,
    run_conditions,
    subset,
)

INNER = set(range(62))
EQ = 4


def noise(sigma: float, only_eq: bool = False, seed: int = 0) -> HookFactory:
    def factory(rows: np.ndarray) -> NormHook:
        key = jax.random.PRNGKey(seed * 100003 + int(rows[0]))

        def hook(t: int, rms: jax.Array, _idx: np.ndarray) -> jax.Array:
            if t not in INNER:
                return rms
            z = jax.random.normal(jax.random.fold_in(key, t), rms.shape)
            if only_eq:
                z = z * (jnp.arange(rms.shape[1]) == EQ)[None]
            return rms * jnp.exp(sigma * z)

        return hook

    return factory


def scale(c: float, points: set[int]) -> HookFactory:
    def factory(_rows: np.ndarray) -> NormHook:
        return lambda t, rms, _idx: rms * c if t in points else rms

    return factory


def run_forced(M: NormModel) -> None:
    ro = load_base()["rms_o"]
    conds: Conds = {"dec_free": (True, None)}
    windows = {
        "b0-30": range(0, 31),
        "b0-7": range(0, 8),
        "b8-15": range(8, 16),
        "b16-23": range(16, 24),
        "b24-30": range(24, 31),
    }
    for name, blocks in windows.items():
        pts = {2 * b + o for b in blocks for o in (0, 1)}
        conds[f"dec_forced_orig_{name}"] = (
            True,
            lambda rows, pts=pts: forced(jnp.asarray(ro[:, rows]), pts),
        )
        conds[f"dec_forced_orig_{name}_eq"] = (
            True,
            lambda rows, pts=pts: forced(jnp.asarray(ro[:, rows]), pts, [EQ]),
        )
    run_conditions(M, np.arange(N)[::2], conds, "robust_forced")


def run_noise(M: NormModel) -> None:
    rows = subset(2000, seed=5)
    for dec in (False, True):
        conds: Conds = {}
        for s in (0.03, 0.1, 0.2):
            conds[f"sigma{s}_all"] = (dec, noise(s))
            conds[f"sigma{s}_eq"] = (dec, noise(s, only_eq=True))
        run_conditions(M, rows, conds, f"robust_noise_{'dec' if dec else 'orig'}", ref_dec=dec)


def run_scale(M: NormModel) -> None:
    rows = subset(2000, seed=5)
    for dec in (False, True):
        conds = {f"scale{c}_all": (dec, scale(c, INNER)) for c in (0.9, 0.95, 1.05, 1.1)}
        run_conditions(M, rows, conds, f"robust_scale_{'dec' if dec else 'orig'}", ref_dec=dec)
        singles = {
            f"scale{c}_t{t}": (dec, scale(c, {t})) for t in sorted(INNER) for c in (0.9, 1.1)
        }
        run_conditions(
            M, rows[:1000], singles, f"robust_single_{'dec' if dec else 'orig'}", ref_dec=dec
        )


if __name__ == "__main__":
    M = NormModel()
    log("model loaded")
    {"forced": run_forced, "noise": run_noise, "scale": run_scale}[sys.argv[1]](M)
