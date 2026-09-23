"""Shared paths, the (a, b) grid, component tables and the line DFT.

Grid convention: prompt i = op * 10000 + (a - 1) * 100 + (b - 1), a, b in 1..100 (checked against
`index.npz`). A quantity q in LINES is a+0b ("a"), 0a+b ("b"), a+b ("sum"), a-b ("diff"); its code at
frequency k (1..50) is the complex coefficient `F(k) = mean_i x_i exp(-2 pi i k q_i / 100)`, the
(k, 0) / (0, k) / (k, k) / (k, -k) entry of the 2-D DFT over the grid. A signal c(q) = Re(A e^{2 pi
i k q / 100}) has F(k) = A / 2; the value it peaks at is q* = -arg(F) * 100 / (2 pi k) (mod 100 / k)."""

from pathlib import Path

import numpy as np

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05")
FILTER = RUN / "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling"
DATASET = FILTER / "dataset"
AUTOINTERP = RUN / "analysis/arith_repr/autointerp"
OUT = RUN / "analysis/arith_repr/vectors"
RESID = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/original-resid/addsub1-100_llama31-8b")

LINES = ("a", "b", "sum", "diff")
LINE_COEF = {"a": (1, 0), "b": (0, 1), "sum": (1, 1), "diff": (1, -1)}
K = np.arange(1, 51)
N_LAYER, D = 32, 4096
N_HEAD, N_KV, HD = 32, 8, 128
POS_NAMES = ("BOS", "a", "op", "b", "=")


def period(k: int | np.ndarray) -> int | np.ndarray:
    return 100 // np.gcd(k, 100)


def point_names() -> list[str]:
    """The 65 raw-stream points of `resid.npy`: embed, then L<l>.attn / L<l>.mlp."""
    return ["embed"] + [f"L{li}.{s}" for li in range(N_LAYER) for s in ("attn", "mlp")]


def write_point(layer: int, kind: str) -> int:
    """Index in `point_names()` of the stream point right after a writer's add."""
    return 1 + 2 * layer + (0 if kind == "o" else 1)


def read_point(layer: int, kind: str) -> int:
    """Index of the raw point a reader's norm is applied to (q/k/v before the attn add, gate/up after)."""
    return 2 * layer + (0 if kind in ("q", "k", "v") else 1)


def line_dft(x: np.ndarray) -> np.ndarray:
    """x: (100, 100, ...) over (a, b) -> (4, 50, ...) complex line coefficients (mean-normalised)."""
    f = np.fft.fft2(x, axes=(0, 1)) / 1e4
    # value index is a-1 / b-1: shift the phase so that the coefficient refers to q, not q - 1
    ks = K
    out = np.empty((4, 50) + x.shape[2:], np.complex64)
    for li, line in enumerate(LINES):
        ca, cb = LINE_COEF[line]
        fa, fb = (ca * ks) % 100, (cb * ks) % 100
        ph = np.exp(-2j * np.pi * ks * (ca + cb) / 100)  # q_i = ca*a + cb*b, a = idx+1
        out[li] = f[fa, fb] * ph.reshape((-1,) + (1,) * (x.ndim - 2))
    return out


def peak_value(F: np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """Value of q, in [0, 100 / k), at which Re(F e^{2 pi i k q / 100}) peaks (k peaks per 100)."""
    k = np.asarray(k)
    return np.mod(-np.angle(F) * 100 / (2 * np.pi * k), 100 / k)


def comp_table() -> dict[str, np.ndarray]:
    ix = np.load(DATASET / "index.npz")
    kind = np.array([k.split(".")[-1].replace("_proj", "") for k in ix["comp_kind"]])
    return {
        "site": ix["comp_site"],
        "layer": ix["comp_layer"].astype(int),
        "kind": kind,
        "cidx": ix["comp_index"].astype(int),
    }


def load_uv(
    comps: dict[str, np.ndarray], cols: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """V (d_in,) and U (d_out,) of the given dataset columns."""
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    V: list[np.ndarray] = [np.empty(0)] * cols.size
    U: list[np.ndarray] = [np.empty(0)] * cols.size
    site = comps["site"][cols]
    for s in np.unique(site):
        ids = {int(i): j for j, i in enumerate(uv[s + ".ids"])}
        Vs, Us = uv[s + ".V"], uv[s + ".U"]
        for n in np.flatnonzero(site == s):
            j = ids[int(comps["cidx"][cols[n]])]
            V[n], U[n] = Vs[:, j].copy(), Us[j].copy()
    return V, U


def check_grid() -> None:
    ix = np.load(DATASET / "index.npz")
    i = np.arange(20000)
    assert (ix["op"] == i // 10000).all()
    assert (ix["a"] == (i % 10000) // 100 + 1).all()
    assert (ix["b"] == i % 100 + 1).all()


def name(comps: dict[str, np.ndarray], col: int) -> str:
    return f"L{comps['layer'][col]}.{comps['kind'][col]}.c{comps['cidx'][col]}"
