"""Which components carry each block's MLP action on the a mod 20 plane at position 1 (alive-only model, blocks 0..16).

Nothing is centred. Notation as in mod20_track.py (a = integer operand 1..400; s = a mod 20; X = stream after block b's
attention, mid_b; P = its frequency-1 plane of a mod 20; Delta = MLP(X) - MLP(X - Pi_P X)). In addition:
  writer w     a down component of block b; td_w(a) its activation, U_w in R^4096 its write vector, so the MLP output
               is Y(a) = sum_w td_w(a) U_w
  Psi_1(f)     frequency-1 Fourier coefficient of a field f over a: sum_{s=0..19} (mean of f over a mod 20 = s) e^{-2 pi i s/20}
  out share_w  Re <Psi_1(Y), Psi_1(td_w) U_w> / ||Psi_1(Y)||^2: writer w's share of the MLP's period-20 output (sums to 1)
  delta share_w  ||U_w||^2 sum_a (td_w(X) - td_w(X - Pi_P X))^2, normalised to sum 1: its share of the plane-driven write
  readers      the reader list of mod20_track.py (share of sum_a ||Delta_c||^2), with each reader's odd share and best edge
               from mod20_readers.py (edge 0 = tens parity h, edge 5 = rounded-tens parity g)
Writes <run>/analysis/reader_writer/track_mod20/components.json.

    python mod20_components.py
"""

import json

import numpy as np
from mod20_track import OUT, S20, means, plane
from track_build import LAST, load_blocks
from track_build import OUT as SRC


def psi1(F: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * np.arange(20) / 20) @ means(F, S20, 20)


def main() -> None:
    res = json.loads((OUT / "alive" / "analysis.json").read_text())
    scan = np.load(OUT / "alive" / "readers_scan.npz")
    st = np.load(SRC / "stream.npz")
    blocks = load_blocks()[0]
    out = {}
    for b in range(LAST + 1):
        blk = blocks[b]
        X = st[f"mid_{b}"].astype(np.float64)
        P = plane(X)
        Xd = X - (X @ P) @ P.T
        _, td = blk.mlp_from_t(*blk.readers(X))
        _, tdd = blk.mlp_from_t(*blk.readers(Xd))
        U = blk.m["U_down"]
        Y = td @ U
        py = psi1(Y)
        pw = psi1(td)  # (n_writers,) complex
        contrib = pw[:, None] * U  # (n_writers, 4096)
        out_share = (contrib.conj() @ py).real / np.vdot(py, py).real
        dsh = (U**2).sum(1) * ((td - tdd) ** 2).sum(0)
        dsh = dsh / dsh.sum()
        wid = blk.index[blk.kind == "down"]
        order = np.argsort(-np.abs(out_share))
        writers = [{"writer": f"d{int(wid[i])}", "out_share": float(out_share[i]), "delta_share": float(dsh[i])}
                   for i in order[:6]]  # fmt: skip
        readers = []
        for t in res["sublayers"][f"mlp_{b}"]["mechanism"]["top"][:6]:
            kind = {"g": "gate", "u": "up"}[t["reader"][0]]
            cid = int(t["reader"][1:])
            j = list(scan[f"L{b}_{kind}_ids"]).index(cid)
            readers.append(t | {"odd_share": float(scan[f"L{b}_{kind}_odd"][j]), "edge": int(scan[f"L{b}_{kind}_edge"][j]),
                                "step_r2": float(scan[f"L{b}_{kind}_step_r2"][j]),
                                "circle_r2": float(scan[f"L{b}_{kind}_circle_r2"][j])})  # fmt: skip
        out[b] = {"writers": writers, "readers": readers,
                  "n_writers_90_out": int(np.searchsorted(np.cumsum(np.sort(np.abs(out_share))[::-1]) / np.abs(out_share).sum(), 0.9) + 1)}  # fmt: skip
        print(b, [(w["writer"], round(w["out_share"], 2), round(w["delta_share"], 2)) for w in writers[:4]],
              [(r["reader"], round(r["share"], 2), round(r["odd_share"], 2), r["edge"]) for r in readers[:4]], flush=True)  # fmt: skip
    (OUT / "components.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
