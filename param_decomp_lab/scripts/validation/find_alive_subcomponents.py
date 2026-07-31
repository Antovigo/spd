"""Find minimal subsets of subcomponents sufficient to reconstruct each CI objective.

Ranks every subcomponent by its **max-over-positions mean lower-leaky CI** (per position,
mean over the target prompts; then max over positions — so a subcomponent that only fires
early ranks by its early-position strength), then sweeps top-k prefixes of that ranking:
for each k, the top-k subcomponents are enabled and all others are zeroed **at every
position** (the weight-delta component pinned **off** everywhere, matching
`TargetReconLoss` on target data: the components must do the work). k=0 is the
all-components-off floor; k=C_total is the full component-sum reconstruction (a
faithfulness check — it approximates, not equals, the target). The **alive subcomponents**
are the top-k for the smallest swept k whose reconstruction error is ≤ the threshold —
pass a dense `--ks` grid around the knee to tighten the selection.

On a `dual_hidden_ci` checkpoint (`model.ci_fn_hidden is not None`) this ranks and sweeps
**twice**, once per CI net:

- `role="output"` — the masked model's last-position output is compared to the raw target
  model's (KL + argmax agreement). Read at the last position because that is where the
  answer is read — but masking acts everywhere, so a component matters iff masking it
  *anywhere* on the prompt moves the `=` output.
- `role="hidden"` — the masked model's **decomposed-site activations** (via the
  early-exit `ComponentModel.site_outputs`, which stops the forward once every site is
  cached) are compared to the frozen target model's own site outputs, as the relative
  error `Σ(out−tgt)²/Σtgt²` averaged over sites — the same quantity
  `CIHiddenActsReconLoss` / `StochasticHiddenReconSubsetLoss` report, evaluated over a
  top-k CI-ranked mask instead of raw CI / a stochastic mask.

Both sweeps share the same prompt pool and the same `cache_type="input"` forward per
chunk (the hidden objective's clean targets and the output objective's clean logits come
from one cached pass; ranking both CI nets from that one cache costs no extra model
forward), so a dual-CI run costs roughly one extra *cheap* early-exit forward per swept k
on top of the single-CI cost, not a second full pass over the pool.

**Output-alive is assumed to be a subset of hidden-alive** — a subcomponent load-bearing
for the final output must also be load-bearing for reconstructing the sites that feed it,
since it does not do its work by accident. The hidden-role sweep therefore only needs to
find what output-alive *misses*: `alive_hidden = (hidden sweep's own top-k cut) ∪
alive_output`, a cheap set union rather than a second constrained sweep.

The default thresholds (`--kl-thr=rounded`, `--hidden-thr=rounded`) each anchor to the
run's own achieved quality on their objective: the sweep also evaluates the **rounded
circuit** — per-(prompt, position) masks with CI > `--rounding-thr` (0.01), delta off,
matching `TargetReconLoss`'s `rounded` strategy — on the same metric, and cuts at its
value. The alive set is then the smallest static top-k that reconstructs as well as the
decomposition's own rounded circuit does, so the anchor adapts across decompositions
instead of hard-coding an absolute. Pass a float `--kl-thr` / `--hidden-thr` for an
explicit cut instead.

The ranking is a proxy: CI measures per-subcomponent maskability with everything else
present, so redundant pairs can both rank low. If a curve degrades earlier than the AB
analyses suggest it should, fall back to greedy backward elimination in the transition
region.

An 8B forward pass needs a GPU; pass `--slurm` to submit this invocation as a single-GPU
SLURM job instead of running it on the (GPU-less) login node.

Usage:
    python -m param_decomp_lab.scripts.validation.find_alive_subcomponents <model_path> \
        [--kl-thr=rounded|0.008] [--hidden-thr=rounded|0.05] [--rounding-thr=0.01] \
        [--ci-thr=0.1] [--batch-size=256] [--n-points=40] [--ks=0,8,64,...] \
        [--prompts=PATH] \
        [--output=PATH] [--output-curve=PATH] [--output-npz=PATH] \
        [--output-fig=PATH] [--output-json=PATH] \
        [--output-hidden=PATH] [--output-curve-hidden=PATH] [--output-npz-hidden=PATH] \
        [--output-fig-hidden=PATH] [--output-json-hidden=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Outputs (defaults in the run's `analysis/` layout). The unsuffixed (`output`-role) files
are written unconditionally and are unchanged from the single-CI-net script; the
`_hidden`-suffixed files are written only when the checkpoint has a hidden CI net:

- `datasets/alive_subcomponents[_hidden].tsv` — the alive subset: layer, matrix,
  component, rank, mean_ci, mean_ci_last, max_mean_ci. `rank` is the component's position
  in that role's own CI ranking, so a hidden-alive row pulled in only via the
  output-alive union can have `rank >= ` the hidden sweep's own cut. The reference alive
  lists consumed by every downstream script.
- `datasets/alive_subcomponents_curve[_hidden].tsv` — per swept k. Output: k,
  max_mean_ci_at_k, mean_kl, q5_kl, q95_kl, max_kl, argmax_agree. Hidden: k,
  max_mean_ci_at_k, mean_relerr (a ratio of pooled sums over the whole prompt pool, so it
  has no per-prompt spread to report — see `hidden_acts.reduced_relative_errors`).
- `datasets/alive_subcomponents_kl.npz` / `datasets/alive_subcomponents_hidden.npz` —
  per-(k, prompt) KL + argmax agreement (output) or per-k pooled relative error (hidden),
  plus the full CI ranking, for per-prompt analysis / re-thresholding without a GPU.
- `datasets/alive_subcomponents_per_position[_hidden].json` — per (prompt, position), the
  alive subcomponents with that role's lower-leaky CI > `--ci-thr` at that position,
  organised as prompt > position > matrix (full module path) > [{component, ci}].
  Consumed by `plot_ci_heatmaps` / `plot_ab_heatmaps` / `build_addition_explorer` /
  `build_neuron_connection_explorer`.
- `alive_subcomponents/recon_vs_k[_hidden].png` — the recon-vs-k curve with the alive cut
  marked.
"""

import csv
import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from param_decomp.ci_fns import CIRole  # noqa: E402
from param_decomp.log import logger  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp.metrics.hidden_acts import (  # noqa: E402
    SiteErrors,
    add_site_errors,
    clean_site_outputs,
    reduced_relative_errors,
    site_squared_errors,
)
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
_CURVE_FIELDS_OUTPUT = [
    "k", "max_mean_ci_at_k", "mean_kl", "q5_kl", "q95_kl", "max_kl", "argmax_agree",
]  # fmt: skip
_CURVE_FIELDS_HIDDEN = ["k", "max_mean_ci_at_k", "mean_relerr"]

# (module, component, mean_ci, mean_ci_last, max_mean_ci), sorted by max_mean_ci desc.
Ranking = list[tuple[str, int, float, float, float]]


class AliveSubcomponentsPaths(NamedTuple):
    alive: Path
    curve: Path
    alive_hidden: Path | None
    curve_hidden: Path | None


def _k_grid(ks: Any, n_points: int, n_total: int) -> list[int]:
    if ks is not None:
        parts = ks.split(",") if isinstance(ks, str) else ks
        k_list = sorted({int(k) for k in parts})
    else:
        log_grid = np.geomspace(1, n_total, n_points).round().astype(int)
        k_list = sorted({0, *log_grid.tolist(), n_total})
    assert all(0 <= k <= n_total for k in k_list), f"ks must lie in [0, {n_total}]: {k_list}"
    return k_list


def _rank_by_ci(
    ci_pos_sum: dict[str, torch.Tensor], modules: list[str], n_comp: dict[str, int], n_prompts: int
) -> Ranking:
    ranking: Ranking = []
    for m in modules:
        pos_mean = ci_pos_sum[m] / n_prompts  # [seq, C]
        mean_ci = pos_mean.mean(dim=0).tolist()
        mean_ci_last = pos_mean[-1].tolist()
        max_mean_ci = pos_mean.amax(dim=0).tolist()
        ranking.extend(
            (m, c, mean_ci[c], mean_ci_last[c], max_mean_ci[c]) for c in range(n_comp[m])
        )
    ranking.sort(key=lambda r: r[4], reverse=True)  # by the position where each fires hardest
    return ranking


def _plot_output_curve(
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
    ax.set_title("Reconstruction vs subset size (output-CI-ranked)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_hidden_curve(
    k_list: list[int],
    relerr: np.ndarray,
    k_alive: int,
    hidden_thr: float,
    rounded_mean: float,
    fig_path: Path,
) -> None:
    floor = 1e-7
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pos = [i for i, k in enumerate(k_list) if k > 0]
    kx = [k_list[i] for i in pos]
    ax.plot(kx, np.maximum(relerr[pos], floor), "o-", color="tab:purple", label="mean relerr")
    if 0 in k_list:
        ax.axhline(relerr[k_list.index(0)], ls="--", color="grey", label="all components off (k=0)")
    ax.axhline(rounded_mean, ls=":", color="tab:red", lw=1, label="rounded circuit (anchor)")
    ax.axvline(
        k_alive,
        ls="--",
        color="tab:green",
        lw=1,
        label=f"hidden sweep cut: k={k_alive} @ relerr≤{hidden_thr:.4g}",
    )
    ax.legend(loc="center left", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k (top-k subcomponents by max-over-positions mean CI, hidden net)")
    ax.set_ylabel("mean over sites of Σ(out−tgt)²/Σtgt²")
    ax.set_title("Decomposed-site reconstruction vs subset size (hidden-CI-ranked)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def find_alive_subcomponents(
    model_path: ModelPath,
    kl_thr: float | str = "rounded",
    hidden_thr: float | str = "rounded",
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
    output_hidden: str | None = None,
    output_curve_hidden: str | None = None,
    output_npz_hidden: str | None = None,
    output_fig_hidden: str | None = None,
    output_json_hidden: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> AliveSubcomponentsPaths | None:
    """Write the alive subset(s) + top-k sufficiency curve(s) + per-position CI JSON(s).

    Returns paths to the written files (`alive_hidden`/`curve_hidden` are `None` on a
    single-CI-net checkpoint), or `None` when `--slurm` submits the job instead.
    """
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--kl-thr={kl_thr}",
            f"--hidden-thr={hidden_thr}",
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
            ("--output-hidden", output_hidden),
            ("--output-curve-hidden", output_curve_hidden),
            ("--output-npz-hidden", output_npz_hidden),
            ("--output-fig-hidden", output_fig_hidden),
            ("--output-json-hidden", output_json_hidden),
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
    assert hidden_thr == "rounded" or isinstance(hidden_thr, int | float), (
        f"--hidden-thr must be 'rounded' or a float, got {hidden_thr!r}"
    )

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    dual = model.ci_fn_hidden is not None
    roles: tuple[CIRole, ...] = ("output", "hidden") if dual else ("output",)

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

    # Phase 1: one cached forward per chunk, both CI nets scored off the same cache (no
    # extra model forward per role). Per role: per-(position, subcomponent) mean lower-leaky
    # CI, the sparse per-(prompt, position) record of components with CI > ci_thr (filtered
    # to that role's alive set before writing its JSON), and the rounded-circuit masks
    # (CI > rounding_thr) that phase 2 evaluates as that role's threshold anchor.
    ci_pos_sum = {
        role: {m: torch.zeros(seq, n_comp[m], device=device) for m in modules} for role in roles
    }
    rounded_np = {
        role: {m: np.zeros((n_prompts, seq, n_comp[m]), dtype=bool) for m in modules}
        for role in roles
    }
    # role -> prompt -> position -> module -> [{component, ci}]
    per_position: dict[CIRole, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]] = {
        role: {} for role in roles
    }
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            cached = model(chunk, cache_type="input")
            for role in roles:
                ci = model.calc_causal_importances(cached.cache, sampling="continuous", role=role)
                for m in modules:
                    ci_pos_sum[role][m] += ci.lower_leaky[m].float().sum(dim=0)  # [seq, C]

                # Move CI to CPU/numpy once per chunk to avoid per-element syncs.
                ci_np = {m: ci.lower_leaky[m].float().cpu().numpy() for m in modules}
                for m in modules:
                    rounded_np[role][m][start : start + b] = ci_np[m] > rounding_thr
                for i in range(b):
                    pos_entry: dict[str, dict[str, list[dict[str, Any]]]] = {}
                    for pos in range(seq):
                        per_module: dict[str, list[dict[str, Any]]] = {}
                        for m, arr in ci_np.items():
                            ci_vec = arr[i, pos]  # [C]
                            active_idx = (ci_vec > ci_thr).nonzero()[0]
                            if active_idx.size > 0:
                                per_module[m] = sorted(
                                    (
                                        {
                                            "component": int(comp),
                                            "ci": round(float(ci_vec[comp]), 3),
                                        }
                                        for comp in active_idx
                                    ),
                                    key=lambda d: d["ci"],
                                    reverse=True,
                                )
                        pos_entry[str(pos)] = per_module
                    per_position[role][prompt_texts[start + i]] = pos_entry

    ranking = {role: _rank_by_ci(ci_pos_sum[role], modules, n_comp, n_prompts) for role in roles}

    k_list = _k_grid(ks, n_points, n_total)

    # Phase 2: sweep top-k subsets. Outer loop over chunks (one clean forward + one
    # rounded-circuit forward per role + one sweep forward per (role, k), the hidden
    # role's forwards all being the cheap early-exit `site_outputs`), inner loop over ks
    # ascending, growing each role's enabled set incrementally.
    kl = np.zeros((len(k_list), n_prompts), np.float32)
    agree = np.zeros((len(k_list), n_prompts), bool)
    rounded_kl = np.zeros(n_prompts, np.float32)
    hidden_errors: list[SiteErrors] = [{} for _ in k_list]
    rounded_hidden_errors: SiteErrors = {}
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            cached = model(chunk, cache_type="input")
            logP = torch.log_softmax(cached.output[:, -1].float(), dim=-1)  # [b, vocab]
            P = logP.exp()
            ref_token = logP.argmax(dim=-1)
            clean_targets = clean_site_outputs(model, cached.cache, modules) if dual else None

            # Delta pinned off everywhere: on target data the components must do the
            # work (TargetReconLoss semantics). A ones mask here silently makes every
            # subset look sufficient — the delta absorbs what masked components drop.
            delta_mask = torch.zeros(b, seq, device=device, dtype=dtype)
            weight_deltas_and_masks = {m: (weight_deltas[m], delta_mask) for m in modules}

            # The rounded circuit (role=output) — per-(prompt, position) masks, delta off —
            # read at the last position like the sweep, so its mean KL is a commensurable
            # kl_thr anchor.
            rounded_masks_output = {
                m: torch.from_numpy(rounded_np["output"][m][start : start + b]).to(
                    device=device, dtype=dtype
                )
                for m in modules
            }
            infos = make_mask_infos(
                rounded_masks_output, weight_deltas_and_masks=weight_deltas_and_masks
            )
            logR = torch.log_softmax(model(chunk, mask_infos=infos)[:, -1].float(), dim=-1)
            rounded_kl[start : start + b] = (P * (logP - logR)).sum(dim=-1).cpu().numpy()

            if dual:
                # The rounded circuit (role=hidden), read via the cheap early-exit forward
                # against the same relative-error metric CIHiddenActsReconLoss reports.
                rounded_masks_hidden = {
                    m: torch.from_numpy(rounded_np["hidden"][m][start : start + b]).to(
                        device=device, dtype=dtype
                    )
                    for m in modules
                }
                infos_h = make_mask_infos(
                    rounded_masks_hidden, weight_deltas_and_masks=weight_deltas_and_masks
                )
                site_out_r = model.site_outputs(chunk, infos_h)
                add_site_errors(
                    rounded_hidden_errors,
                    site_squared_errors(
                        site_out_r, cast("dict[str, torch.Tensor]", clean_targets), infos_h
                    ),
                )

            # role=output sweep: full forward + last-position KL, unchanged from the
            # single-CI-net script.
            enabled_output = {
                m: torch.zeros(n_comp[m], device=device, dtype=dtype) for m in modules
            }
            prev_k = 0
            for ki, k in enumerate(k_list):
                for module, component, *_ in ranking["output"][prev_k:k]:
                    enabled_output[module][component] = 1.0
                prev_k = k
                # ablate outside the top-k at every position (the delta stays fully on),
                # so a component important anywhere on the prompt registers at the `=` read
                masks = {m: enabled_output[m].expand(b, seq, n_comp[m]) for m in modules}
                infos = make_mask_infos(masks, weight_deltas_and_masks=weight_deltas_and_masks)
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

            if dual:
                # role=hidden sweep: the truncated site_outputs forward, so this costs a
                # cheap forward per k on top of role=output's full forward per k.
                enabled_hidden = {
                    m: torch.zeros(n_comp[m], device=device, dtype=dtype) for m in modules
                }
                prev_k = 0
                for ki, k in enumerate(k_list):
                    for module, component, *_ in ranking["hidden"][prev_k:k]:
                        enabled_hidden[module][component] = 1.0
                    prev_k = k
                    masks = {m: enabled_hidden[m].expand(b, seq, n_comp[m]) for m in modules}
                    infos = make_mask_infos(masks, weight_deltas_and_masks=weight_deltas_and_masks)
                    site_out = model.site_outputs(chunk, infos)
                    add_site_errors(
                        hidden_errors[ki],
                        site_squared_errors(
                            site_out, cast("dict[str, torch.Tensor]", clean_targets), infos
                        ),
                    )

    mean_kl = kl.mean(axis=1)
    agree_mean = agree.mean(axis=1)
    rounded_mean_output = float(rounded_kl.mean())
    kl_thr_val = rounded_mean_output if kl_thr == "rounded" else float(kl_thr)
    logger.info(
        f"rounded-circuit anchor (output): mean last-pos KL={rounded_mean_output:.4g} "
        f"(kl_thr={'anchor' if kl_thr == 'rounded' else kl_thr_val})"
    )
    under_thr = [k for ki, k in enumerate(k_list) if mean_kl[ki] <= kl_thr_val]
    assert under_thr, (
        f"no swept k reaches mean KL <= {kl_thr_val} (best: {mean_kl.min():.4g} at "
        f"k={k_list[-1]}); raise --kl-thr or check the decomposition"
    )
    k_alive_output = under_thr[0]
    alive_set_output = {(m, c) for m, c, *_ in ranking["output"][:k_alive_output]}

    relerr = np.zeros(len(k_list), np.float32)
    rounded_mean_hidden = 0.0
    hidden_thr_val = 0.0
    k_alive_hidden_own = 0
    alive_set_hidden: set[tuple[str, int]] = set()
    if dual:
        if k_list[-1] == n_total:
            # The last swept k's mask is all-on/delta-off, the same faithfulness check as
            # the output role's — over the whole prompt pool rather than just the first
            # chunk, since a whole-pass ratio is the correct aggregation for this metric
            # (see hidden_acts.reduced_relative_errors), unlike per-example KL.
            full_relerr = float(reduced_relative_errors(hidden_errors[-1], "full")["full"])
            assert full_relerr < 1.0, (
                f"component-sum (all-on, delta off) far from target (hidden relerr {full_relerr}); "
                "mask wiring bug or unfaithful decomposition"
            )
            logger.info(
                f"faithfulness check (hidden): mean relerr(target||all_on_no_delta)={full_relerr:.4g}"
            )

        relerr = np.array(
            [
                float(reduced_relative_errors(hidden_errors[ki], "k")["k"])
                for ki in range(len(k_list))
            ],
            dtype=np.float32,
        )
        rounded_mean_hidden = float(
            reduced_relative_errors(rounded_hidden_errors, "rounded")["rounded"]
        )
        hidden_thr_val = rounded_mean_hidden if hidden_thr == "rounded" else float(hidden_thr)
        logger.info(
            f"rounded-circuit anchor (hidden): mean relerr={rounded_mean_hidden:.4g} "
            f"(hidden_thr={'anchor' if hidden_thr == 'rounded' else hidden_thr_val})"
        )
        under_thr_hidden = [k for ki, k in enumerate(k_list) if relerr[ki] <= hidden_thr_val]
        assert under_thr_hidden, (
            f"no swept k reaches mean relerr <= {hidden_thr_val} (best: {relerr.min():.4g} at "
            f"k={k_list[-1]}); raise --hidden-thr or check the decomposition"
        )
        k_alive_hidden_own = under_thr_hidden[0]

        # Output-alive is assumed alive-for-hidden too (see module docstring): union
        # rather than re-sweep under a constraint.
        alive_set_hidden_own = {(m, c) for m, c, *_ in ranking["hidden"][:k_alive_hidden_own]}
        alive_set_hidden = alive_set_hidden_own | alive_set_output
        if len(alive_set_hidden) > len(alive_set_hidden_own):
            logger.info(
                f"hidden alive set: {len(alive_set_hidden_own)} from its own sweep cut "
                f"(k={k_alive_hidden_own}) + {len(alive_set_hidden) - len(alive_set_hidden_own)} "
                "pulled in from the output-alive set"
            )

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
    out_paths = [alive_path, curve_path, npz_path, fig_path, json_path]

    alive_path_hidden = curve_path_hidden = None
    npz_path_hidden = fig_path_hidden = json_path_hidden = None
    if dual:
        alive_path_hidden = (
            Path(output_hidden).expanduser()
            if output_hidden
            else data_dir / "alive_subcomponents_hidden.tsv"
        )
        curve_path_hidden = (
            Path(output_curve_hidden).expanduser()
            if output_curve_hidden
            else data_dir / "alive_subcomponents_curve_hidden.tsv"
        )
        npz_path_hidden = (
            Path(output_npz_hidden).expanduser()
            if output_npz_hidden
            else data_dir / "alive_subcomponents_hidden.npz"
        )
        fig_path_hidden = (
            Path(output_fig_hidden).expanduser()
            if output_fig_hidden
            else analysis_dir(run.run_dir) / "alive_subcomponents" / "recon_vs_k_hidden.png"
        )
        json_path_hidden = (
            Path(output_json_hidden).expanduser()
            if output_json_hidden
            else data_dir / "alive_subcomponents_per_position_hidden.json"
        )
        out_paths += [
            alive_path_hidden,
            curve_path_hidden,
            npz_path_hidden,
            fig_path_hidden,
            json_path_hidden,
        ]

    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    with alive_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RANK_FIELDS, delimiter="\t")
        writer.writeheader()
        for rank, (module, component, mean_ci, mean_ci_last, max_mean_ci) in enumerate(
            ranking["output"][:k_alive_output]
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
        writer = csv.DictWriter(f, fieldnames=_CURVE_FIELDS_OUTPUT, delimiter="\t")
        writer.writeheader()
        for ki, k in enumerate(k_list):
            writer.writerow(
                {
                    "k": k,
                    "max_mean_ci_at_k": ranking["output"][k - 1][4] if k > 0 else "",
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
        rank_module=np.array([r[0] for r in ranking["output"]]),
        rank_component=np.array([r[1] for r in ranking["output"]]),
        rank_mean_ci=np.array([r[2] for r in ranking["output"]]),
        rank_mean_ci_last=np.array([r[3] for r in ranking["output"]]),
        rank_max_mean_ci=np.array([r[4] for r in ranking["output"]]),
        k_alive=np.array(k_alive_output),
        rounded_kl=rounded_kl,
        kl_thr=np.array(kl_thr_val),
    )
    _plot_output_curve(
        k_list, kl, agree_mean, k_alive_output, kl_thr_val, rounded_mean_output, fig_path
    )

    for pos_entry in per_position["output"].values():
        for pos, per_module in pos_entry.items():
            pos_entry[pos] = {
                m: kept
                for m, comps in per_module.items()
                if (kept := [e for e in comps if (m, e["component"]) in alive_set_output])
            }
    json_path.write_text(json.dumps(per_position["output"], separators=(",", ":")))

    if dual:
        assert alive_path_hidden and curve_path_hidden and npz_path_hidden
        assert fig_path_hidden and json_path_hidden
        rank_index_hidden = {(m, c): idx for idx, (m, c, *_) in enumerate(ranking["hidden"])}
        row_by_key_hidden = {(m, c): (mean_ci, mean_ci_last, max_mean_ci) for m, c, mean_ci, mean_ci_last, max_mean_ci in ranking["hidden"]}  # fmt: skip
        hidden_rank_order = sorted(alive_set_hidden, key=lambda mc: rank_index_hidden[mc])

        with alive_path_hidden.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_RANK_FIELDS, delimiter="\t")
            writer.writeheader()
            for module, component in hidden_rank_order:
                mean_ci, mean_ci_last, max_mean_ci = row_by_key_hidden[(module, component)]
                layer, matrix = parse_module_name(module)
                writer.writerow(
                    {
                        "layer": layer,
                        "matrix": matrix,
                        "component": component,
                        "rank": rank_index_hidden[(module, component)],
                        "mean_ci": mean_ci,
                        "mean_ci_last": mean_ci_last,
                        "max_mean_ci": max_mean_ci,
                    }
                )

        with curve_path_hidden.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CURVE_FIELDS_HIDDEN, delimiter="\t")
            writer.writeheader()
            for ki, k in enumerate(k_list):
                writer.writerow(
                    {
                        "k": k,
                        "max_mean_ci_at_k": ranking["hidden"][k - 1][4] if k > 0 else "",
                        "mean_relerr": float(relerr[ki]),
                    }
                )

        np.savez_compressed(
            npz_path_hidden,
            ks=np.array(k_list),
            relerr=relerr,
            rank_module=np.array([r[0] for r in ranking["hidden"]]),
            rank_component=np.array([r[1] for r in ranking["hidden"]]),
            rank_mean_ci=np.array([r[2] for r in ranking["hidden"]]),
            rank_mean_ci_last=np.array([r[3] for r in ranking["hidden"]]),
            rank_max_mean_ci=np.array([r[4] for r in ranking["hidden"]]),
            k_alive=np.array(k_alive_hidden_own),
            k_alive_union=np.array(len(alive_set_hidden)),
            rounded_relerr=np.array(rounded_mean_hidden),
            hidden_thr=np.array(hidden_thr_val),
        )
        _plot_hidden_curve(
            k_list, relerr, k_alive_hidden_own, hidden_thr_val, rounded_mean_hidden, fig_path_hidden
        )

        for pos_entry in per_position["hidden"].values():
            for pos, per_module in pos_entry.items():
                pos_entry[pos] = {
                    m: kept
                    for m, comps in per_module.items()
                    if (kept := [e for e in comps if (m, e["component"]) in alive_set_hidden])
                }
        json_path_hidden.write_text(json.dumps(per_position["hidden"], separators=(",", ":")))

    logger.info(
        f"output: {k_alive_output}/{n_total} subcomponents alive (mean KL <= {kl_thr_val:.4g} "
        f"at k={k_alive_output}) over {n_prompts} prompts → {alive_path}, {curve_path}, "
        f"{json_path}, {fig_path}"
    )
    if dual:
        logger.info(
            f"hidden: {len(alive_set_hidden)}/{n_total} subcomponents alive (own sweep cut "
            f"k={k_alive_hidden_own} at relerr <= {hidden_thr_val:.4g}, unioned with output-alive) "
            f"→ {alive_path_hidden}, {curve_path_hidden}, {json_path_hidden}, {fig_path_hidden}"
        )

    return AliveSubcomponentsPaths(
        alive=alive_path,
        curve=curve_path,
        alive_hidden=alive_path_hidden,
        curve_hidden=curve_path_hidden,
    )


if __name__ == "__main__":
    fire.Fire(find_alive_subcomponents)
