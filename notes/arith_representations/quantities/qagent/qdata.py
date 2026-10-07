"""Per-position site data for the quantity agent (written by qprep.py).

A position's domain is the set of distinct inputs its activations can depend on (see qprep.py);
`dom_index(t, op, a, b)` maps prompts to domain rows. Each site holds Z (stream in reader-span
coordinates, D x k), W (readout, k x n; raw reads Y = Z W), CI (D x n), the readers' dataset
columns and Q (4096 x k, the reader-span basis).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RUN = Path.home() / "out/pod-backup/p-ba5a0c05"
DATA = RUN / "analysis/quantities/qagent"
POS_NAMES = {1: "a", 2: "op", 3: "b", 4: "="}


@dataclass
class Site:
    label: str  # "L<block>.<attn|mlp>"
    Z: np.ndarray
    W: np.ndarray
    CI: np.ndarray
    cols: np.ndarray
    Q: np.ndarray

    @property
    def Y(self) -> np.ndarray:
        return self.Z @ self.W

    @property
    def block(self) -> int:
        return int(self.label[1:].split(".")[0])

    @property
    def point(self) -> str:
        return self.label.split(".")[1]

    @property
    def stream_pos(self) -> int:
        return 2 * self.block + (self.point == "mlp")


@dataclass
class Position:
    t: int
    rows: np.ndarray  # prompt index of each domain row
    op: np.ndarray
    a: np.ndarray
    b: np.ndarray
    sites: dict[str, Site] = field(default_factory=dict)

    @property
    def D(self) -> int:
        return len(self.rows)


def load(t: int, sites: list[str] | None = None) -> Position:
    f = np.load(DATA / f"site_data_t{t}.npz")
    pos = Position(t, f["rows"], f["op"].astype(int), f["a"].astype(int), f["b"].astype(int))
    labels = sorted({k.split("/")[0] for k in f.files if "/" in k},
                    key=lambda s: (int(s[1:].split(".")[0]), s.endswith("mlp")))  # fmt: skip
    for lab in labels:
        if sites is None or lab in sites:
            pos.sites[lab] = Site(lab, f[lab + "/Z"].astype(np.float64), f[lab + "/W"].astype(np.float64),
                                  f[lab + "/CI"].astype(np.float64), f[lab + "/cols"], f[lab + "/Q"].astype(np.float64))  # fmt: skip
    return pos


def dom_index(t: int, op: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Domain row of each prompt (op 0/1, a, b in 1..100) at position t."""
    if t == 1:
        return a - 1
    if t == 2:
        return op * 100 + a - 1
    return op * 10000 + (a - 1) * 100 + (b - 1)
