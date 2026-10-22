"""Which period-20 variable each MLP reader of blocks 0..LAST reads at position 1, in the alive-only model
(kept components; track_build.py) and in the original Llama-3.1-8B (every gate and up neuron; track_build_orig.py).

Nothing is centred. Notation (a = integer operand 1..400, s = a mod 20, r = a mod 10, as in mod20_track.py):
  t_c(a)     reader c's activation at the MLP input of its block (component: x . V_c; neuron: x . w_c, x the
             post-attention RMSNorm output)
  m_c(s)     mean of t_c over the 20 a with a mod 20 = s
  o_c(s)     = (m_c(s) - m_c(s + 10)) / 2, s = 0..9: the odd part, what tells s from s + 10 (o_c(s + 10) = -o_c(s))
  odd share  = 20 sum_{s=0..19} o_c(s)^2 / sum_a t_c(a)^2 (uncentred): the share of the reader's energy in the odd part
  edge e     integer 0..9; the binary variable B_e(a) = floor((a + 10 - e) / 10) mod 2 flips at s = e and s = e + 10.
             B_0 = h = floor(a / 10) mod 2 (tens parity); B_5 = g = floor((a + 5) / 10) mod 2 (parity of a rounded to the
             nearest ten). The odd part of +-1 coded B_e is the step sigma_e(s) = -1 for s < e, +1 for s >= e (s = 0..9)
  step r2_e  uncentred r2 of o_c ~= beta sigma_e (one coefficient beta) over s = 0..9; best edge = argmax_e
  circle r2  uncentred r2 of o_c ~= beta_1 cos(2 pi s / 20) + beta_2 sin(2 pi s / 20)
  quarter    the indicator of one quarter Q = floor(s / 5) (e.g. s in 10..14): its odd part is +-1/2 on five s and 0
             on the other five; quarter r2 = best uncentred r2 over the four quarters of o_c ~= beta * that odd part
Writes <run>/analysis/reader_writer/track_mod20/<model>/readers_scan.npz (per block arrays) and readers_scan.json
(per block counts).

    python mod20_readers.py --model alive|orig
"""

import argparse
import json

import numpy as np
from mod20_track import OUT, S20, means
from track_build import LAST

SS = np.arange(10)
STEPS = np.stack([np.where(e > SS, -1.0, 1.0) for e in range(10)])  # (10 edges, 10 s)
CIRC = np.stack([np.cos(2 * np.pi * SS / 20), np.sin(2 * np.pi * SS / 20)], 1)
QUART = np.stack([np.where(q % 2 == SS // 5, 0.5 if q < 2 else -0.5, 0.0) for q in range(4)])


def scan(T: np.ndarray) -> dict[str, np.ndarray]:
    m = means(T, S20, 20)
    o = (m[:10] - m[10:]) / 2  # (10, n)
    on = (o**2).sum(0)
    odd = 20 * on / (T**2).sum(0)
    step = (STEPS @ o) ** 2 / (STEPS**2).sum(1, keepdims=True) / on  # (10, n)
    beta = np.linalg.lstsq(CIRC, o, rcond=None)[0]
    circ = 1 - ((o - CIRC @ beta) ** 2).sum(0) / on
    quart = ((QUART @ o) ** 2 / (QUART**2).sum(1, keepdims=True) / on).max(0)
    return {"odd": odd, "edge": step.argmax(0), "step_r2": step.max(0), "step_r2_all": step.T,
            "circle_r2": circ, "quarter_r2": quart}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("alive", "orig"), required=True)
    ap.add_argument("--odd-min", type=float, default=0.1)
    args = ap.parse_args()
    if args.model == "alive":
        from track_build import OUT as SRC
        from track_build import load_blocks

        blocks = load_blocks()[0]
        get = lambda b: blocks[b]  # noqa: E731
    else:
        from track_build_orig import OUT as SRC
        from track_build_orig import load_block as get
    st = np.load(SRC / "stream.npz")
    save, summary = {}, {}
    for b in range(LAST + 1):
        blk = get(b)
        tg, tu = blk.readers(st[f"mid_{b}"].astype(np.float64))
        ids = {"gate": blk.index[blk.kind == "gate"], "up": blk.index[blk.kind == "up"]} if args.model == "alive" else {
            "gate": np.arange(tg.shape[1]), "up": np.arange(tu.shape[1])}  # fmt: skip
        summary[b] = {}
        for kind, T in (("gate", tg), ("up", tu)):
            sc = scan(T)
            for k, v in sc.items():
                save[f"L{b}_{kind}_{k}"] = v
            save[f"L{b}_{kind}_ids"] = ids[kind]
            sel = sc["odd"] >= args.odd_min
            summary[b][kind] = {
                "n": int(T.shape[1]),
                "n_odd": int(sel.sum()),
                "edge_counts_step_r2_ge_0.8": np.bincount(
                    sc["edge"][sel & (sc["step_r2"] >= 0.8)], minlength=10
                ).tolist(),
                "n_circle_r2_ge_0.8": int((sel & (sc["circle_r2"] >= 0.8)).sum()),
                "n_quarter_r2_ge_0.8": int((sel & (sc["quarter_r2"] >= 0.8)).sum()),
            }
        print(b, {k: (v["n_odd"], v["edge_counts_step_r2_ge_0.8"], v["n_circle_r2_ge_0.8"], v["n_quarter_r2_ge_0.8"])
                  for k, v in summary[b].items()}, flush=True)  # fmt: skip
        del blk
    out = OUT / args.model
    np.savez(out / "readers_scan.npz", **save)
    (out / "readers_scan.json").write_text(
        json.dumps({"odd_min": args.odd_min, "blocks": summary}, indent=1)
    )


if __name__ == "__main__":
    main()
