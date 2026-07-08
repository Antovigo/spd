"""Find a minimal subset of L18 MLP neurons sufficient to reconstruct the model's output.

Neurons have no causal importance, so this does an actual greedy removal: neurons are
ordered by mean |post-SwiGLU activation| at the `=` position (ascending — least active
first) and removed in that order with an adaptive batch size, zeroing each removed
neuron's post-SwiGLU output (the down_proj input dim). A removal batch is accepted iff
the mean KL(target ‖ ablated) at the `=` position over a fixed random scoring subset of
the 0..200 add/sub grids stays ≤ `--kl-thr`; on rejection the batch is bisected, and a
neuron whose singleton removal is rejected stays **alive** permanently (one ranked pass,
no retries). The final removal set is then re-scored on the full grids.

The decomposition checkpoint only locates the frozen base model; nothing depends on the
decomposition itself, so outputs land in the shared `runs/neurons/` census dir.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.find_alive_neurons <model_path> \
        [--kl-thr=0.007] [--ops=add,sub] [--layer=18] [--batch-size=512] \
        [--score-prompts=4000] [--init-batch=512] [--seed=0] [--out-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=4:00:00 --slurm-mem=...]]

Outputs (default under `<PARAM_DECOMP_OUT_DIR>/runs/neurons/`):
- `alive_neurons.tsv` — the kept neurons: neuron, mean_abs_act.
- `alive_neurons_curve.tsv` — accepted-removal trajectory: n_removed, mean_kl.
- `alive_neurons.npz` — removal order, alive mask, full greedy trajectory (accepted and
  rejected), scoring indices, final full-grid KL stats, mean_abs_act, kl_thr.
- `alive_neurons_curve.png` — mean KL vs n_removed with the threshold line.
"""

import csv
from pathlib import Path
from typing import Any, cast

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp.torch_helpers import bf16_autocast  # noqa: E402
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    SlurmOptions,
    load_lm_run,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.neurons.common import (  # noqa: E402
    D_INT,
    NEURONS_DIR,
    tokenize_grid,
)

_MODULE = "param_decomp_lab.scripts.validation.neurons.find_alive_neurons"


def _plot_curve(n_removed: list[int], mean_kl: list[float], kl_thr: float, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_removed, mean_kl, "o-", color="tab:blue", ms=3)
    ax.axhline(kl_thr, ls="--", color="tab:green", label=f"kl_thr={kl_thr}")
    ax.set_xlabel("neurons removed")
    ax.set_ylabel("mean KL(target ‖ ablated)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Greedy neuron removal (accepted steps)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def find_alive_neurons(
    model_path: ModelPath,
    kl_thr: float = 0.007,
    ops: str | tuple[str, ...] = "add,sub",
    layer: int = 18,
    batch_size: int = 512,
    score_prompts: int = 4000,
    init_batch: int = 512,
    seed: int = 0,
    out_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "4:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Write the alive-neurons TSV + greedy-removal trajectory (TSV, npz, figure).

    Returns the alive TSV path, or `None` when `--slurm` submits the job instead.
    """
    ops_list = list(ops) if isinstance(ops, tuple) else ops.split(",")
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--kl-thr={kl_thr}",
            f"--ops={','.join(ops_list)}",
            f"--layer={layer}",
            f"--batch-size={batch_size}",
            f"--score-prompts={score_prompts}",
            f"--init-batch={init_batch}",
            f"--seed={seed}",
        ]
        if out_dir is not None:
            argv.append(f"--out-dir={Path(out_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-find-alive-neurons")
        return None

    run = load_lm_run(model_path)
    hf = run.model.target_model  # the bare, frozen Llama-3.1-8B
    device = run.device
    out_root = Path(out_dir).expanduser() if out_dir else NEURONS_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    pool = torch.cat([tokenize_grid(run.tokenizer, op) for op in ops_list]).to(device)
    n_pool = pool.shape[0]

    # The keep-mask hook: zeroes removed neurons' post-SwiGLU outputs in place; when
    # capturing, also accumulates last-position |activation| per neuron.
    dtype = next(hf.parameters()).dtype
    keep = torch.ones(D_INT, device=device, dtype=dtype)
    act_acc = torch.zeros(D_INT, device=device)
    capture = [False]

    def pre_hook(_m: Any, args: tuple[torch.Tensor, ...]) -> None:
        x = args[0]
        assert x.shape[-1] == D_INT
        x.mul_(keep)
        if capture[0]:
            act_acc.add_(x[:, -1].float().abs().sum(dim=0))

    down_proj = hf.get_submodule(f"model.layers.{layer}.mlp.down_proj")
    handle = down_proj.register_forward_pre_hook(pre_hook)

    rng = np.random.default_rng(seed)
    score_idx = (
        np.sort(rng.choice(n_pool, size=score_prompts, replace=False))
        if score_prompts < n_pool
        else np.arange(n_pool)
    )
    score_pool = pool[torch.from_numpy(score_idx).to(device)]
    n_score = score_pool.shape[0]

    def last_pos_logprobs(ids: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(hf(input_ids=ids).logits[:, -1].float(), dim=-1)

    # Phase 1: one full-pool pass for the per-neuron mean |activation| (removal order),
    # then the scoring subset's reference log-probs with the SAME batching the greedy
    # evals use — bf16 logits differ across batch shapes, so extracting reference rows
    # from full-pool batches would put a noise floor under every KL.
    capture[0] = True
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        for start in range(0, n_pool, batch_size):
            last_pos_logprobs(pool[start : start + batch_size])
    capture[0] = False
    mean_abs_act = (act_acc / n_pool).cpu().numpy()
    order = np.argsort(mean_abs_act, kind="stable")  # ascending: least active removed first

    vocab_size = int(cast(Any, hf).config.vocab_size)
    ref_logp = torch.zeros(n_score, vocab_size, device=device)
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        for start in range(0, n_score, batch_size):
            logp = last_pos_logprobs(score_pool[start : start + batch_size])
            ref_logp[start : start + logp.shape[0]] = logp

    ref_p = ref_logp.exp()

    def score_mean_kl() -> float:
        total = 0.0
        with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
            for start in range(0, n_score, batch_size):
                logq = last_pos_logprobs(score_pool[start : start + batch_size])
                sl = slice(start, start + logq.shape[0])
                total += float((ref_p[sl] * (ref_logp[sl] - logq)).sum(dim=-1).sum())
        return total / n_score

    # Wiring check: an all-ones keep-mask must be a no-op.
    noop_kl = score_mean_kl()
    assert noop_kl < 1e-4, f"all-ones keep-mask perturbs the model (KL {noop_kl}); hook bug"

    # Phase 2: greedy removal in `order` with adaptive batching. One ranked pass: a
    # rejected singleton is alive for good.
    alive = np.ones(D_INT, dtype=bool)
    traj_n_removed: list[int] = []
    traj_kl: list[float] = []
    traj_accepted: list[bool] = []
    i, trial_size, n_evals = 0, init_batch, 0
    while i < D_INT:
        trial = order[i : i + trial_size]
        trial_t = torch.from_numpy(trial).to(device)
        keep[trial_t] = 0.0
        kl = score_mean_kl()
        n_evals += 1
        accepted = kl <= kl_thr
        traj_n_removed.append(int((~alive).sum()) + (len(trial) if accepted else 0))
        traj_kl.append(kl)
        traj_accepted.append(accepted)
        if accepted:
            alive[trial] = False
            i += len(trial)
            trial_size *= 2
        else:
            keep[trial_t] = 1.0
            if len(trial) == 1:
                i += 1
            else:
                trial_size = max(1, trial_size // 2)
        if n_evals % 25 == 0:
            logger.info(
                f"eval {n_evals}: {int((~alive).sum())} removed, cursor {i}/{D_INT}, "
                f"trial {trial_size}, last KL {kl:.4g}"
            )

    n_alive = int(alive.sum())
    logger.info(f"greedy done in {n_evals} evals: {n_alive} alive / {D_INT - n_alive} removed")

    # Phase 3: re-score the final removal set on the full pool (streamed ref + ablated).
    keep_final = keep.clone()
    kl_full = np.zeros(n_pool, np.float32)
    agree_full = np.zeros(n_pool, bool)
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        for start in range(0, n_pool, batch_size):
            chunk = pool[start : start + batch_size]
            keep.fill_(1.0)
            logp = last_pos_logprobs(chunk)
            keep.copy_(keep_final)
            logq = last_pos_logprobs(chunk)
            sl = slice(start, start + chunk.shape[0])
            kl_full[sl] = (logp.exp() * (logp - logq)).sum(dim=-1).cpu().numpy()
            agree_full[sl] = (logq.argmax(dim=-1) == logp.argmax(dim=-1)).cpu().numpy()
    handle.remove()
    logger.info(
        f"full-grid check ({n_pool} prompts): mean KL {kl_full.mean():.4g} "
        f"(scoring thr {kl_thr}), q95 {np.percentile(kl_full, 95):.4g}, "
        f"max {kl_full.max():.4g}, argmax agree {agree_full.mean():.4f}"
    )

    alive_path = out_root / "alive_neurons.tsv"
    curve_path = out_root / "alive_neurons_curve.tsv"
    npz_path = out_root / "alive_neurons.npz"
    fig_path = out_root / "alive_neurons_curve.png"

    with alive_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["neuron", "mean_abs_act"], delimiter="\t")
        writer.writeheader()
        for neuron in np.flatnonzero(alive):
            writer.writerow({"neuron": int(neuron), "mean_abs_act": float(mean_abs_act[neuron])})

    acc_n = [n for n, a in zip(traj_n_removed, traj_accepted, strict=True) if a]
    acc_kl = [k for k, a in zip(traj_kl, traj_accepted, strict=True) if a]
    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_removed", "mean_kl"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"n_removed": n, "mean_kl": k} for n, k in zip(acc_n, acc_kl, strict=True))

    np.savez_compressed(
        npz_path,
        alive=alive,
        order=order,
        mean_abs_act=mean_abs_act,
        traj_n_removed=np.array(traj_n_removed),
        traj_kl=np.array(traj_kl, np.float32),
        traj_accepted=np.array(traj_accepted),
        score_idx=score_idx,
        kl_full=kl_full,
        agree_full=agree_full,
        kl_thr=np.array(kl_thr),
        layer=np.array(layer),
        ops=np.array(ops_list),
    )
    _plot_curve(acc_n, acc_kl, kl_thr, fig_path)

    logger.info(f"{n_alive}/{D_INT} neurons alive (mean KL <= {kl_thr}) → {alive_path}")
    return alive_path


if __name__ == "__main__":
    fire.Fire(find_alive_neurons)
