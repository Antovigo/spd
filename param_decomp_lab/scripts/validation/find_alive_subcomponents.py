"""Find a minimal subset of subcomponents sufficient to reconstruct the target output.

Ranks every subcomponent by its **max-over-positions mean lower-leaky CI** (per position,
mean over the target prompts; then max over positions — so a subcomponent that only fires
early ranks by its early-position strength), then sweeps top-k prefixes of that ranking:
for each k, the top-k subcomponents are enabled and all others are zeroed **at every
position** (the weight-delta component pinned **off** everywhere, matching
`TargetReconLoss` on target data: the components must do the work), and the masked
model's last-position output is compared to the raw target model's (KL + argmax
agreement). The KL is read at the last position because that is where the answer is read
— but masking acts everywhere, so a component matters iff masking it *anywhere* on the
prompt moves the `=` output. k=0 is the all-components-off floor; k=C_total is the full
component-sum reconstruction (a faithfulness check — it approximates, not equals, the
target). The **alive subcomponents** are the top-k for the smallest swept k whose mean
KL is ≤ the threshold — pass a dense `--ks` grid around the knee to tighten the
selection.

The default threshold (`--kl-thr=rounded`) anchors to the run's own achieved quality:
the sweep also evaluates the **rounded circuit** — per-(prompt, position) masks with CI >
`--rounding-thr` (0.01), delta off, matching `TargetReconLoss`'s `rounded` strategy — at
the last position on the same prompts, and cuts at its mean KL. The alive set is then the smallest static
top-k that reconstructs as well as the decomposition's own rounded circuit does, and the
anchor adapts across decompositions instead of hard-coding an absolute. Computing the
anchor in-sweep (not reading `eval/target_recon/rounded` from the training logs) keeps it
commensurable: the logged metric averages over all sequence positions, while both the
sweep and this anchor read KL at the last position only. Pass a float `--kl-thr` for an
explicit absolute cut.

The ranking is a proxy: CI measures per-subcomponent maskability with everything else
present, so redundant pairs can both rank low. If the curve degrades earlier than the AB
analyses suggest it should, fall back to greedy backward elimination in the transition
region.

An 8B forward pass needs a GPU; pass `--slurm` to submit this invocation as a single-GPU
SLURM job instead of running it on the (GPU-less) login node.

Usage:
    python -m param_decomp_lab.scripts.validation.find_alive_subcomponents <model_path> \
        [--kl-thr=rounded|0.008] [--rounding-thr=0.01] [--ci-thr=0.1] [--batch-size=256] \
        [--n-points=40] [--ks=0,8,64,...] \
        [--prompts=PATH] [--output=PATH] [--output-curve=PATH] [--output-npz=PATH] \
        [--output-fig=PATH] [--output-json=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Outputs (defaults in the run's `analysis/` layout):
- `datasets/alive_subcomponents.tsv` — the alive subset (the top-k_alive rows of the CI
  ranking): layer, matrix, component, rank, mean_ci, mean_ci_last, max_mean_ci. The
  reference alive list consumed by every downstream script.
- `datasets/alive_subcomponents_curve.tsv` — per swept k: k, max_mean_ci_at_k, mean_kl,
  q5_kl, q95_kl, max_kl, argmax_agree.
- `datasets/alive_subcomponents_kl.npz` — per-(k, prompt) KL + argmax agreement, the
  rounded-circuit per-prompt KL (`rounded_kl`), and the full CI ranking, for per-prompt
  analysis / re-thresholding without a GPU.
- `datasets/alive_subcomponents_per_position.json` — per (prompt, position), the alive
  subcomponents with lower-leaky CI > `--ci-thr` at that position, organised as
  prompt > position > matrix (full module path) > [{component, ci}]. Consumed by
  `plot_ci_heatmaps` / `plot_ab_heatmaps` / `build_addition_explorer` /
  `build_neuron_connection_explorer`.
- `alive_subcomponents/recon_vs_k.png` — the recon-vs-k curve with the alive cut marked.
"""

import csv
import json
from pathlib import Path
from typing import Any, cast

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp.torch_helpers import bf16_autocast  # noqa: E402
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset  # noqa: E402
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    SlurmOptions,
    analysis_datasets_dir,
    analysis_dir,
    load_lm_run,
    parse_module_name,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.find_alive_subcomponents"
_RANK_FIELDS = ["layer", "matrix", "component", "rank", "mean_ci", "mean_ci_last", "max_mean_ci"]
_CURVE_FIELDS = ["k", "max_mean_ci_at_k", "mean_kl", "q5_kl", "q95_kl", "max_kl", "argmax_agree"]


def _k_grid(ks: Any, n_points: int, n_total: int) -> list[int]:
    if ks is not None:
        parts = ks.split(",") if isinstance(ks, str) else ks
        k_list = sorted({int(k) for k in parts})
    else:
        log_grid = np.geomspace(1, n_total, n_points).round().astype(int)
        k_list = sorted({0, *log_grid.tolist(), n_total})
    assert all(0 <= k <= n_total for k in k_list), f"ks must lie in [0, {n_total}]: {k_list}"
    return k_list


def _plot_curve(
    k_list: list[int],
    kl: np.ndarray,
    agree: np.ndarray,
    k_alive: int,
    kl_thr: float,
    rounded_mean: float,
    fig_path: Path,
) -> None:
    mean_kl = kl.mean(axis=1)
    q5, q95 = np.percentile(kl, [5, 95], axis=1)
    max_kl = kl.max(axis=1)
    floor = 1e-7  # log-scale plot floor: q5 can round to exactly 0 in float32
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = [i for i, k in enumerate(k_list) if k > 0]
    kx = [k_list[i] for i in pos]
    ax.fill_between(
        kx, np.maximum(q5[pos], floor), np.maximum(q95[pos], floor),
        color="tab:blue", alpha=0.2, lw=0, label="q5–q95",
    )  # fmt: skip
    ax.plot(kx, np.maximum(max_kl[pos], floor), ":", color="tab:blue", lw=1, label="max")
    ax.plot(kx, np.maximum(mean_kl[pos], floor), "o-", color="tab:blue", label="mean")
    if 0 in k_list:
        ax.axhline(
            mean_kl[k_list.index(0)], ls="--", color="grey", label="all components off (k=0)"
        )
    ax.axhline(rounded_mean, ls=":", color="tab:red", lw=1, label="rounded circuit (anchor)")
    ax.axvline(
        k_alive, ls="--", color="tab:green", lw=1, label=f"alive: k={k_alive} @ KL≤{kl_thr:.4g}"
    )
    ax.legend(loc="center left", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (top-k subcomponents by max-over-positions mean CI)")
    ax.set_ylabel("last-pos KL(target ‖ top-k)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(kx, agree[pos], "s-", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("argmax agreement", color="tab:orange")
    ax2.set_ylim(0, 1.02)
    ax.set_title("Reconstruction vs subset size (CI-ranked)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def find_alive_subcomponents(
    model_path: ModelPath,
    kl_thr: float | str = "rounded",
    rounding_thr: float = 0.01,
    ci_thr: float = 0.1,
    batch_size: int = 256,
    n_points: int = 40,
    ks: str | None = None,
    prompts: str | None = None,
    output: str | None = None,
    output_curve: str | None = None,
    output_npz: str | None = None,
    output_fig: str | None = None,
    output_json: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> tuple[Path, Path] | None:
    """Write the alive subset + top-k sufficiency curve + per-position CI JSON.

    Returns `(alive_tsv, curve_tsv)`, or `None` when `--slurm` submits the job instead.
    """
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--kl-thr={kl_thr}",
            f"--rounding-thr={rounding_thr}",
            f"--ci-thr={ci_thr}",
            f"--batch-size={batch_size}",
            f"--n-points={n_points}",
        ]
        if ks is not None:
            argv.append(f"--ks={ks}")
        if prompts is not None:
            argv.append(f"--prompts={Path(prompts).expanduser()}")
        for flag, val in [
            ("--output", output),
            ("--output-curve", output_curve),
            ("--output-npz", output_npz),
            ("--output-fig", output_fig),
            ("--output-json", output_json),
        ]:
            if val is not None:
                argv.append(f"{flag}={Path(val).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-find-alive-subcomponents")
        return None

    assert kl_thr == "rounded" or isinstance(kl_thr, int | float), (
        f"--kl-thr must be 'rounded' or a float, got {kl_thr!r}"
    )

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer

    prompts_path = prompts if prompts is not None else cfg.data.prompts_file
    assert prompts_path is not None, (
        "find_alive_subcomponents requires prompts-based target data (or pass --prompts)"
    )
    prompt_texts = [
        ln.strip() for ln in Path(prompts_path).expanduser().read_text().splitlines() if ln.strip()
    ]
    assert len(set(prompt_texts)) == len(prompt_texts), (
        "duplicate prompts in the prompts file would collide as JSON keys"
    )
    pool = load_prompts_dataset(str(Path(prompts_path).expanduser()), cast(Any, tokenizer))
    pool = pool.to(device)
    assert pool.shape[0] == len(prompt_texts)
    n_prompts, seq = pool.shape

    modules = sorted(model.components.keys(), key=parse_module_name)
    n_comp = {m: model.components[m].V.shape[1] for m in modules}
    n_total = sum(n_comp.values())
    weight_deltas = model.calc_weight_deltas()
    dtype = next(model.parameters()).dtype

    # Phase 1: per-(position, subcomponent) mean lower-leaky CI, the sparse
    # per-(prompt, position) record of components with CI > ci_thr (filtered to the alive
    # set before writing the JSON), and the rounded-circuit masks (CI > rounding_thr) that
    # phase 2 evaluates as the kl_thr anchor.
    ci_pos_sum = {m: torch.zeros(seq, n_comp[m], device=device) for m in modules}
    rounded_np = {m: np.zeros((n_prompts, seq, n_comp[m]), dtype=bool) for m in modules}
    # prompt -> position -> module -> [{component, ci}]
    per_position: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            for m in modules:
                ci_pos_sum[m] += ci.lower_leaky[m].float().sum(dim=0)  # [seq, C]

            # Move CI to CPU/numpy once per chunk to avoid per-element syncs.
            ci_np = {m: ci.lower_leaky[m].float().cpu().numpy() for m in modules}
            for m in modules:
                rounded_np[m][start : start + chunk.shape[0]] = ci_np[m] > rounding_thr
            for i in range(chunk.shape[0]):
                pos_entry: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for pos in range(seq):
                    per_module: dict[str, list[dict[str, Any]]] = {}
                    for m, arr in ci_np.items():
                        ci_vec = arr[i, pos]  # [C]
                        active_idx = (ci_vec > ci_thr).nonzero()[0]
                        if active_idx.size > 0:
                            per_module[m] = sorted(
                                (
                                    {"component": int(comp), "ci": round(float(ci_vec[comp]), 3)}
                                    for comp in active_idx
                                ),
                                key=lambda d: d["ci"],
                                reverse=True,
                            )
                    pos_entry[str(pos)] = per_module
                per_position[prompt_texts[start + i]] = pos_entry

    # (module, component, mean_ci, mean_ci_last, max_mean_ci)
    ranking: list[tuple[str, int, float, float, float]] = []
    for m in modules:
        pos_mean = ci_pos_sum[m] / n_prompts  # [seq, C]
        mean_ci = pos_mean.mean(dim=0).tolist()
        mean_ci_last = pos_mean[-1].tolist()
        max_mean_ci = pos_mean.amax(dim=0).tolist()
        ranking.extend(
            (m, c, mean_ci[c], mean_ci_last[c], max_mean_ci[c]) for c in range(n_comp[m])
        )
    ranking.sort(key=lambda r: r[4], reverse=True)  # by the position where each fires hardest

    k_list = _k_grid(ks, n_points, n_total)

    # Phase 2: sweep top-k subsets. Outer loop over chunks (one target-reference forward
    # + one rounded-circuit forward each), inner loop over ks ascending, growing the
    # enabled set incrementally.
    kl = np.zeros((len(k_list), n_prompts), np.float32)
    agree = np.zeros((len(k_list), n_prompts), bool)
    rounded_kl = np.zeros(n_prompts, np.float32)
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            logP = torch.log_softmax(model(chunk)[:, -1].float(), dim=-1)  # [b, vocab]
            P = logP.exp()
            ref_token = logP.argmax(dim=-1)

            # Delta pinned off everywhere: on target data the components must do the
            # work (TargetReconLoss semantics). A ones mask here silently makes every
            # subset look sufficient — the delta absorbs what masked components drop.
            delta_mask = torch.zeros(b, seq, device=device, dtype=dtype)

            # The rounded circuit — per-(prompt, position) masks, delta off — read at the
            # last position like the sweep, so its mean KL is a commensurable kl_thr anchor.
            rounded_masks = {
                m: torch.from_numpy(rounded_np[m][start : start + b]).to(device=device, dtype=dtype)
                for m in modules
            }
            infos = make_mask_infos(
                rounded_masks,
                weight_deltas_and_masks={m: (weight_deltas[m], delta_mask) for m in modules},
            )
            logR = torch.log_softmax(model(chunk, mask_infos=infos)[:, -1].float(), dim=-1)
            rounded_kl[start : start + b] = (P * (logP - logR)).sum(dim=-1).cpu().numpy()
            enabled = {m: torch.zeros(n_comp[m], device=device, dtype=dtype) for m in modules}
            prev_k = 0
            for ki, k in enumerate(k_list):
                for module, component, *_ in ranking[prev_k:k]:
                    enabled[module][component] = 1.0
                prev_k = k
                # ablate outside the top-k at every position (the delta stays fully on),
                # so a component important anywhere on the prompt registers at the `=` read
                masks = {m: enabled[m].expand(b, seq, n_comp[m]) for m in modules}
                infos = make_mask_infos(
                    masks,
                    weight_deltas_and_masks={m: (weight_deltas[m], delta_mask) for m in modules},
                )
                out = model(chunk, mask_infos=infos)
                logQ = torch.log_softmax(out[:, -1].float(), dim=-1)
                kl[ki, start : start + b] = (P * (logP - logQ)).sum(dim=-1).cpu().numpy()
                agree[ki, start : start + b] = (logQ.argmax(dim=-1) == ref_token).cpu().numpy()

            if start == 0 and k_list[-1] == n_total:
                # All components on, delta off = the full component-sum reconstruction.
                # A faithful decomposition keeps this small; a blow-up means the curve
                # is meaningless (mask wiring bug or an unfaithful decomposition).
                full_kl = float(kl[-1, :b].mean())
                assert full_kl < 0.5, (
                    f"component-sum (all-on, delta off) far from target (KL {full_kl}); "
                    "mask wiring bug or unfaithful decomposition"
                )
                logger.info(f"faithfulness check: mean KL(target||all_on_no_delta)={full_kl:.4g}")

    mean_kl = kl.mean(axis=1)
    agree_mean = agree.mean(axis=1)
    rounded_mean = float(rounded_kl.mean())
    kl_thr_val = rounded_mean if kl_thr == "rounded" else float(kl_thr)
    logger.info(
        f"rounded-circuit anchor: mean last-pos KL={rounded_mean:.4g} "
        f"(kl_thr={'anchor' if kl_thr == 'rounded' else kl_thr_val})"
    )
    under_thr = [k for ki, k in enumerate(k_list) if mean_kl[ki] <= kl_thr_val]
    assert under_thr, (
        f"no swept k reaches mean KL <= {kl_thr_val} (best: {mean_kl.min():.4g} at "
        f"k={k_list[-1]}); raise --kl-thr or check the decomposition"
    )
    k_alive = under_thr[0]

    data_dir = analysis_datasets_dir(run.run_dir)
    alive_path = Path(output).expanduser() if output else data_dir / "alive_subcomponents.tsv"
    curve_path = (
        Path(output_curve).expanduser()
        if output_curve
        else data_dir / "alive_subcomponents_curve.tsv"
    )
    npz_path = (
        Path(output_npz).expanduser() if output_npz else data_dir / "alive_subcomponents_kl.npz"
    )
    fig_path = (
        Path(output_fig).expanduser()
        if output_fig
        else analysis_dir(run.run_dir) / "alive_subcomponents" / "recon_vs_k.png"
    )
    json_path = (
        Path(output_json).expanduser()
        if output_json
        else data_dir / "alive_subcomponents_per_position.json"
    )
    for p in (alive_path, curve_path, npz_path, fig_path, json_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with alive_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RANK_FIELDS, delimiter="\t")
        writer.writeheader()
        for rank, (module, component, mean_ci, mean_ci_last, max_mean_ci) in enumerate(
            ranking[:k_alive]
        ):
            layer, matrix = parse_module_name(module)
            writer.writerow(
                {
                    "layer": layer,
                    "matrix": matrix,
                    "component": component,
                    "rank": rank,
                    "mean_ci": mean_ci,
                    "mean_ci_last": mean_ci_last,
                    "max_mean_ci": max_mean_ci,
                }
            )

    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CURVE_FIELDS, delimiter="\t")
        writer.writeheader()
        for ki, k in enumerate(k_list):
            writer.writerow(
                {
                    "k": k,
                    "max_mean_ci_at_k": ranking[k - 1][4] if k > 0 else "",
                    "mean_kl": float(mean_kl[ki]),
                    "q5_kl": float(np.percentile(kl[ki], 5)),
                    "q95_kl": float(np.percentile(kl[ki], 95)),
                    "max_kl": float(kl[ki].max()),
                    "argmax_agree": float(agree_mean[ki]),
                }
            )

    np.savez_compressed(
        npz_path,
        ks=np.array(k_list),
        kl=kl,
        agree=agree,
        prompts=np.array(prompt_texts),
        rank_module=np.array([r[0] for r in ranking]),
        rank_component=np.array([r[1] for r in ranking]),
        rank_mean_ci=np.array([r[2] for r in ranking]),
        rank_mean_ci_last=np.array([r[3] for r in ranking]),
        rank_max_mean_ci=np.array([r[4] for r in ranking]),
        k_alive=np.array(k_alive),
        rounded_kl=rounded_kl,
        kl_thr=np.array(kl_thr_val),
    )
    _plot_curve(k_list, kl, agree_mean, k_alive, kl_thr_val, rounded_mean, fig_path)

    alive_set = {(module, component) for module, component, *_ in ranking[:k_alive]}
    for pos_entry in per_position.values():
        for pos, per_module in pos_entry.items():
            pos_entry[pos] = {
                m: kept
                for m, comps in per_module.items()
                if (kept := [e for e in comps if (m, e["component"]) in alive_set])
            }
    json_path.write_text(json.dumps(per_position, separators=(",", ":")))

    logger.info(
        f"{k_alive}/{n_total} subcomponents alive (mean KL <= {kl_thr_val:.4g} at k={k_alive}) over "
        f"{n_prompts} prompts → {alive_path}, {curve_path}, {json_path}, {fig_path}"
    )
    return alive_path, curve_path


if __name__ == "__main__":
    fire.Fire(find_alive_subcomponents)
