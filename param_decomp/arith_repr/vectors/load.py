"""Loaders for the vector-level products, with the linear part of a and b split off the a / b lines."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from param_decomp.arith_repr.vectors.common import LINES, OUT, comp_table, peak_value, period

W50 = np.r_[np.full(49, 2.0), 1.0]  # variance weight of harmonic k (k = 50 is real)
RAMP_VAR = 833.25  # variance of q - 50.5 over q = 1..100


def ramp_dft() -> np.ndarray:
    q = np.arange(1, 101)
    k = np.arange(1, 51)[:, None]
    return ((q - 50.5)[None] * np.exp(-2j * np.pi * k * q / 100)).mean(1)


@dataclass
class Reads:
    """Line spectra of every component's inner (A, 4 pos, 2 op, 4 lines, 50), linear part removed."""

    R: np.ndarray
    var: np.ndarray
    mean: np.ndarray
    slope_a: np.ndarray
    slope_b: np.ndarray

    @cached_property
    def energy(self) -> np.ndarray:
        """Share of the inner's grid variance on each (line, k)."""
        return np.abs(self.R) ** 2 * W50 / np.maximum(self.var[..., None, None], 1e-12)

    @property
    def lin(self) -> np.ndarray:
        """(A, 4, 2, 2): share of variance linear in a / in b."""
        v = np.maximum(self.var, 1e-12)
        return np.stack([self.slope_a**2 * RAMP_VAR / v, self.slope_b**2 * RAMP_VAR / v], -1)


def load_reads() -> Reads:
    z = np.load(OUT / "read_spec.npz")
    R = z["R"].copy()
    c = ramp_dft().astype(np.complex64)
    R[:, :, :, 0] -= z["slope_a"][..., None] * c
    R[:, :, :, 1] -= z["slope_b"][..., None] * c
    return Reads(R, z["var"], z["mean"], z["slope_a"], z["slope_b"])


@dataclass
class Writes:
    """`U_c . F` on the frames right after c's add, for residual writers (cols), linear part removed."""

    cols: np.ndarray
    W: np.ndarray  # (Aw, 4, 2, 4, 50) complex
    slope_a: np.ndarray
    slope_b: np.ndarray
    opdiff: np.ndarray
    unorm: np.ndarray


def load_writes() -> Writes:
    z = np.load(OUT / "write_spec.npz")
    W = z["W"].copy()
    c = ramp_dft().astype(np.complex64)
    W[:, :, :, 0] -= z["slope_a"][..., None] * c
    W[:, :, :, 1] -= z["slope_b"][..., None] * c
    return Writes(z["cols"], W, z["slope_a"], z["slope_b"], z["opdiff"], z["unorm"])


def describe(spec: np.ndarray, energy: np.ndarray, n: int = 3, thr: float = 0.0) -> str:
    """Top (line, k) entries of one (4, 50) spectrum: `sum k10 (T10) @7.2 0.41`."""
    idx = np.argsort(energy.ravel())[::-1][:n]
    out = []
    for i in idx:
        li, k0 = divmod(int(i), 50)
        k = k0 + 1
        if energy.ravel()[i] < thr:
            break
        out.append(
            f"{LINES[li]} k{k} (T{period(k)}) @{peak_value(spec[li, k0], k):.1f} {energy[li, k0]:.2f}"
        )
    return "; ".join(out)


__all__ = ["Reads", "Writes", "load_reads", "load_writes", "describe", "comp_table", "W50"]
