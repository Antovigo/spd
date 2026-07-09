"""Post-hoc Proposal-A retrofit: project a dense checkpoint's V/U into row/col spaces.

For every decomposed module, computes the target weight's economy SVD (kept rank
`sigma > rank_threshold * sigma_max`) and replaces `V <- Q_in Q_in^T V`,
`U <- U Q_out Q_out^T` in a copy of the run. The parameterization stays dense
(`LinearComponents`), so the whole analysis stack — including the subspace-filtering
battery — runs unchanged; the removed mass lands in the on-the-fly weight delta.
Full-rank sides are projected too (a no-op). Logs per-matrix relative change.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_restriction.retrofit_svd_projection \
        ~/out/runs/addsub-L18-04-hidden/model_24000.pth ~/out/runs/addsub-L18-04-hidden-Aretro
"""

import shutil
from pathlib import Path

import fire
import torch

from param_decomp.log import logger


def retrofit_svd_projection(
    model_path: str,
    dst_run_dir: str,
    rank_threshold: float = 1e-5,
) -> Path:
    src_ckpt = Path(model_path).expanduser()
    src_run = src_ckpt.parent
    dst_run = Path(dst_run_dir).expanduser()
    assert not dst_run.exists(), f"{dst_run} already exists"
    dst_run.mkdir(parents=True)

    config_text = (src_run / "experiment_config.yaml").read_text()
    config_text = config_text.replace(f"label: {src_run.name}", f"label: {dst_run.name}", 1)
    (dst_run / "experiment_config.yaml").write_text(config_text)
    if (src_run / "run_metadata.json").exists():
        shutil.copy(src_run / "run_metadata.json", dst_run / "run_metadata.json")

    sd = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    modules = sorted({k.split(".")[1] for k in sd if k.startswith("_components.")})
    assert modules, "no _components.* keys in checkpoint"

    for mod in modules:
        w = sd["target_model." + mod.replace("-", ".") + ".weight"].float()
        q_out, s, vh = torch.linalg.svd(w, full_matrices=False)
        r = int((s > rank_threshold * s[0]).sum().item())
        q_in = vh[:r].T

        v_key, u_key = f"_components.{mod}.V", f"_components.{mod}.U"
        v, u = sd[v_key].float(), sd[u_key].float()
        v_proj = q_in @ (q_in.T @ v)
        u_proj = (u @ q_out[:, :r]) @ q_out[:, :r].T
        logger.info(
            f"{mod}: r={r}/{min(w.shape)}, "
            f"|dV|/|V|={(v_proj - v).norm() / v.norm():.4f}, "
            f"|dU|/|U|={(u_proj - u).norm() / u.norm():.4f}"
        )
        sd[v_key] = v_proj.to(sd[v_key].dtype)
        sd[u_key] = u_proj.to(sd[u_key].dtype)

    torch.save(sd, dst_run / src_ckpt.name)
    logger.info(f"wrote {dst_run / src_ckpt.name}")
    return dst_run / src_ckpt.name


if __name__ == "__main__":
    fire.Fire(retrofit_svd_projection)
