"""Singular spectra of the decomposed matrices in a PD checkpoint's target model.

Reads the frozen `target_model.*` weights of every decomposed module straight from the
checkpoint, computes their singular values, and reports the numerical rank at a grid of
`rank_threshold` values (`sigma > tau * sigma_max`) — the `init_rank_threshold` knob of
the `span_proj` weight init.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_restriction.scan_spectra \
        <run>/model_24000.pth --out-dir=~/pd_scratch/subspace_restriction/spectra

Outputs in `out_dir`: `spectra.npz` (one array per module), `ranks.tsv`
(module, d_out, d_in, sigma_max, r at each tau), `spectra.png` (log10(sigma/sigma_max)).
"""

from pathlib import Path

import fire
import matplotlib
import numpy as np
import torch

from param_decomp.log import logger

_DEFAULT_TAUS = (1e-5, 1e-3, 1e-2, 3e-2, 1e-1)


def scan_spectra(
    model_path: str,
    out_dir: str,
    taus: tuple[float, ...] = _DEFAULT_TAUS,
) -> Path:
    sd = torch.load(Path(model_path).expanduser(), map_location="cpu", weights_only=True, mmap=True)
    modules = sorted({k.split(".")[1] for k in sd if k.startswith("_components.")})
    assert modules, "no _components.* keys in checkpoint"

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    spectra: dict[str, np.ndarray] = {}
    rows: list[list[str]] = []
    for mod in modules:
        w = sd["target_model." + mod.replace("-", ".") + ".weight"].float()
        s = torch.linalg.svdvals(w).numpy()
        spectra[mod] = s
        ranks = [int((s > tau * s[0]).sum()) for tau in taus]
        rows.append([mod, str(w.shape[0]), str(w.shape[1]), f"{s[0]:.4g}", *map(str, ranks)])
        logger.info(f"{mod}: shape {tuple(w.shape)}, ranks at {taus} = {ranks}")

    header = ["module", "d_out", "d_in", "sigma_max", *[f"r_tau{tau:g}" for tau in taus]]
    (out / "ranks.tsv").write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n")
    np.savez_compressed(out / "spectra.npz", **spectra)  # pyright: ignore[reportArgumentType]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for mod, s in spectra.items():
        ax.plot(np.arange(1, len(s) + 1) / len(s), np.log10(s / s[0]), lw=1, label=mod)
    for tau in taus:
        ax.axhline(np.log10(tau), color="gray", ls=":", lw=0.6)
    ax.set_xlabel("index / min(d_out, d_in)")
    ax.set_ylabel("log10(sigma / sigma_max)")
    ax.legend(fontsize=7)
    ax.set_title(f"singular spectra: {Path(model_path).expanduser().parent.name}")
    fig.tight_layout()
    fig.savefig(out / "spectra.png", dpi=150)
    logger.info(f"wrote {out}/ranks.tsv, spectra.npz, spectra.png")
    return out


if __name__ == "__main__":
    fire.Fire(scan_spectra)
