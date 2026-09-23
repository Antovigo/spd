"""How each sublayer changes each code, and which components' writes make the change.

    python -m param_decomp.arith_repr.vectors.transfer   # -> OUT/transfer.npz

For every sublayer (attention or MLP of block l; 64 steps t -> t+1 of the raw stream), position
1..4, op and (line, k): the change of the code `dF = F_{t+1} - F_t` (linear parts of a, b removed as
in `storage`) and, for every alive writer c of that sublayer, its own contribution to the code,
`C_c = R_c(line, k) U_c` (R = line coefficient of its inner, i.e. of what its V read, U = what it
writes). Stored:

* `share` (Aw, 4, 2, 4, 50): Re<C_c, dF> / |dF|^2 (the writers of a step sum to the alive share);
* `dF_energy` (64, 4, 2, 4, 50): |dF|^2 w / tvar (same units as `storage.energy`);
* `keep` (64, 4, 2, 4, 50): Re<dF, F_t> / |F_t|^2 (the part of the step along the existing code
  — amplification if > 0, erasure if < 0); `turn` the complex argument of <dF, F_t> (residue shift);
* `mirror` (64, 4, 4, 50) for sub: Re<dF_sub, conj(F_add,t) - F_sub,t> / |conj(F_add,t) - F_sub,t|^2
  — how much of the step moves the sub code to the mirror image of the add code (1 = all the way);
* `cols` writer columns (as in `write_spec`)."""

import numpy as np

from param_decomp.arith_repr.vectors.common import OUT, comp_table, load_uv, write_point
from param_decomp.arith_repr.vectors.load import W50, load_reads
from param_decomp.arith_repr.vectors.storage import load_frame


def main() -> None:
    comps = comp_table()
    cols = np.flatnonzero(np.isin(comps["kind"], ("o", "down")))
    _, U = load_uv(comps, cols)
    Um = np.stack(U)
    reads = load_reads()
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    wp = np.array([write_point(comps["layer"][c], comps["kind"][c]) for c in cols])
    n_step = fre.shape[0] - 1
    share = np.zeros((cols.size, 4, 2, 4, 50), np.float32)
    dF_energy = np.zeros((n_step, 4, 2, 4, 50), np.float32)
    keep = np.zeros_like(dF_energy)
    turn = np.zeros_like(dF_energy)
    mirror = np.zeros((n_step, 4, 4, 50), np.float32)
    F_prev = load_frame(fre, fim, st, 0)
    for t in range(1, n_step + 1):
        F = load_frame(fre, fim, st, t)
        dF = F - F_prev
        n2 = (np.abs(dF) ** 2).sum(-1)  # (4, 2, 4, 50)
        dF_energy[t - 1] = n2 * W50 / st["tvar"][t][:, :, None, None]
        ip = (dF * np.conj(F_prev)).sum(-1)
        keep[t - 1] = ip.real / np.maximum((np.abs(F_prev) ** 2).sum(-1), 1e-12)
        turn[t - 1] = np.angle(ip)
        target = np.conj(F_prev[:, 0]) - F_prev[:, 1]
        mirror[t - 1] = (dF[:, 1] * np.conj(target)).sum(-1).real / np.maximum(
            (np.abs(target) ** 2).sum(-1), 1e-12
        )
        sel = np.flatnonzero(wp == t)
        if sel.size:
            # Re<R U, dF> = Re(R * (U . conj dF))
            udf = np.einsum("cd,polkd->cpolk", Um[sel], np.conj(dF))
            share[sel] = (reads.R[cols[sel]] * udf).real / np.maximum(n2, 1e-12)
        F_prev = F
        print(t, sel.size, flush=True)
    np.savez(
        OUT / "transfer.npz", share=share, dF_energy=dF_energy, keep=keep, turn=turn,
        mirror=mirror, cols=cols,
    )  # fmt: skip


if __name__ == "__main__":
    main()
