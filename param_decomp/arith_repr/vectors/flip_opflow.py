"""Per-prompt inner activations at `=` of every alive component, alive-only model (every alive component on
everywhere, no delta), for the account of where the operation goes after layer 15.

    python -m param_decomp.arith_repr.vectors.flip_opflow capture   # -> OUT/flip/opflow/h_eq.npz

At `=` every sublayer's write is h @ U of that position's inners, so the stream at `=` before any
sublayer is the `=` embedding plus the writes of the sublayers before it; `capture` checks this
against the captured stream entering layer 15's and layer 24's MLPs."""

import sys
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS
from param_decomp.arith_repr.isa.components_model import ComponentsModel, T
from param_decomp.arith_repr.vectors.flip_components import DIR

EQ = 4  # position of `=`
KINDS = ("q", "k", "v", "o", "gate", "up", "down")
OUT = DIR / "opflow"


def alive_model() -> ComponentsModel:
    cm = ComponentsModel()
    ncols = {
        li: int(np.load(ATLAS / "masks" / f"L{li}.npy", mmap_mode="r").shape[2]) for li in range(32)
    }
    cm.masks = lambda layer, rows: np.ones((len(rows), T, ncols[layer]), np.float32)  # type: ignore[method-assign]
    return cm


def stream_at(cm: ComponentsModel, H: dict[str, np.ndarray], t: int) -> np.ndarray:
    """Raw stream at `=` at point t (2l before block l's attention, 2l + 1 before its MLP), (N, d)."""
    x = np.asarray(cm.embed[int(cm.tokens[0, EQ])], np.float64)[None]
    for li in range(32):
        for off, kind in ((0, "o"), (1, "down")):
            if 2 * li + off >= t:
                return x
            x = x + H[f"{li}.{kind}"].astype(np.float64) @ np.asarray(
                cm.sites[li][kind].U, np.float64
            )
    return x


def capture() -> None:
    cm = alive_model()
    assert len(set(cm.tokens[:, EQ].tolist())) == 1, "`=` token differs across prompts"
    rows = np.arange(len(cm.op))
    H = {
        f"{li}.{k}": np.zeros((len(rows), cm.sites[li][k].V.shape[1]), np.float32)
        for li in range(32)
        for k in KINDS
    }

    def inner(li: int, kind: str, h: Any, _rms: Any, idx: np.ndarray) -> Any:
        H[f"{li}.{kind}"][idx] = np.asarray(h[:, EQ])
        return h

    _, caps = cm.forward(rows, inner=inner, capture={31, 49})
    for t, c in caps.items():
        rec = stream_at(cm, H, t)
        ref = c[:, EQ].astype(np.float64)
        err = np.linalg.norm(rec - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(
            f"stream at t={t}: relative reconstruction error median {np.median(err):.2e} max {err.max():.2e}",
            flush=True,
        )
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "h_eq.npz", **cast(dict[str, Any], H))
    print("saved", OUT / "h_eq.npz", flush=True)


if __name__ == "__main__":
    cast(Any, {"capture": capture})[sys.argv[1]]()
