"""Completeness fine-tune: a second "complete" CI net learns each block's standalone
mechanisms, including the redundant copies the joint decomposition skipped.

Starting from a finished joint decomposition, duplicate the CI net (normal +
complete). Each step picks a random block `b` and optimizes two configurations,
summed into one update:

- A: block `b` masked by the NORMAL CI net, all other blocks left as the original
  matrices (their modules are simply not masked). This is the per-block regime —
  intact partners make every redundant copy in `b` unnecessary, so the normal CI
  converges to the in-context/marginal map, pruning ALL copies (including the
  credited ones).
- B: block `b` masked by the COMPLETE CI net, other blocks masked by their normal CI
  (detached — B trains the complete net and the components, not the backdrop). Once
  A has stripped the copies from the normal CI, no other block supplies the
  redundant computation, and reconstruction forces the complete CI to activate
  block `b`'s own copy.

Both configurations use the recipe losses (stochastic + layerwise + PPGD recon,
SmoothL0 minimality on the selected block's masking net only; independent PPGD
sources per configuration). The all-normal joint masking is NOT trained: the normal
net becomes a marginal-importance map, the complete net the standalone map.

Outputs two scoreable run dirs, `<out_dir>/normal` and `<out_dir>/complete`,
differing only in their `ci_fn.*` weights — run `plot_ci` on each.

Usage:
    python -m param_decomp_lab.experiments.toy_model_redundancy.complete <model_path> --out-dir=PATH
"""

import copy
import json
from pathlib import Path
from typing import Any, cast

import fire
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
from tqdm import tqdm

from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp.faithfulness_warmup import run_faithfulness_warmup
from param_decomp.log import logger
from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLoss,
    PersistentPGDReconLossConfig,
)
from param_decomp.metrics.persistent_pgd_state import AdamPGDConfig, PerBatchPerPositionScope
from param_decomp.metrics.smooth_l0_importance_minimality import (
    SmoothL0ImportanceMinimalityLoss,
    SmoothL0ImportanceMinimalityLossConfig,
)
from param_decomp.metrics.stochastic_recon import StochasticReconLoss, StochasticReconLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import (
    StochasticReconLayerwiseLoss,
    StochasticReconLayerwiseLossConfig,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl, run_batch_passthrough
from param_decomp_lab.distributed import get_device
from param_decomp_lab.experiments.toy_model_redundancy.ci_figure import plot_subcomponent_grid
from param_decomp_lab.experiments.toy_model_redundancy.run import (
    AnyRedundancyToy,
    SavedToyModelRedundancyRun,
    build_target,
)
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.seed import set_seed


def _block_index(module_path: str) -> int:
    parts = module_path.split(".")
    assert parts[0] == "blocks", f"unexpected module path: {module_path}"
    return int(parts[1])


def _calc_ci(
    model: ComponentModel, ci_fn: nn.Module, pre_weight_acts: dict[str, Tensor]
) -> CIOutputs:
    """CI values from an arbitrary CI net through `model`'s sigmoids (continuous sampling)."""
    raw: dict[str, Tensor] = ci_fn(pre_weight_acts)
    return CIOutputs(
        lower_leaky={m: model.lower_leaky_fn(t) for m, t in raw.items()},
        upper_leaky={m: model.upper_leaky_fn(t) for m, t in raw.items()},
        pre_sigmoid=raw,
    )


def _restrict(ci: CIOutputs, modules: list[str]) -> CIOutputs:
    return CIOutputs(
        lower_leaky={m: ci.lower_leaky[m] for m in modules},
        upper_leaky={m: ci.upper_leaky[m] for m in modules},
        pre_sigmoid={m: ci.pre_sigmoid[m] for m in modules},
    )


def _mix(
    selected_ci: CIOutputs, rest_ci: CIOutputs, selected: set[str], detach_rest: bool
) -> CIOutputs:
    def pick(field: dict[str, Tensor], rest_field: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            m: field[m]
            if m in selected
            else (rest_field[m].detach() if detach_rest else rest_field[m])
            for m in rest_field
        }

    return CIOutputs(
        lower_leaky=pick(selected_ci.lower_leaky, rest_ci.lower_leaky),
        upper_leaky=pick(selected_ci.upper_leaky, rest_ci.upper_leaky),
        pre_sigmoid=pick(selected_ci.pre_sigmoid, rest_ci.pre_sigmoid),
    )


def _ones(ci: CIOutputs) -> CIOutputs:
    """All-ones CI (fully-active components ≈ the original matrix, up to the delta).

    Used as the PPGD backdrop in configuration A: PPGD's sources cover every module,
    so bypassed blocks need a mask of 1 (zero source gradient) rather than no mask.
    """
    ones = {m: torch.ones_like(t) for m, t in ci.lower_leaky.items()}
    return CIOutputs(lower_leaky=ones, upper_leaky=ones, pre_sigmoid=ones)


def _recipe_metrics(
    model: ComponentModel,
    device: str,
    *,
    impmin_coeff: float,
    beta: float,
    gamma_final: float,
    gamma_anneal_start_frac: float,
) -> dict[str, Metric[Any]]:
    metrics: dict[str, Metric[Any]] = {
        "impmin": SmoothL0ImportanceMinimalityLoss(
            SmoothL0ImportanceMinimalityLossConfig(
                coeff=impmin_coeff,
                gamma=1.0,
                beta=beta,
                normalize_at_one=True,
                gamma_final=gamma_final,
                gamma_anneal_start_frac=gamma_anneal_start_frac,
            )
        ),
        "stoch": StochasticReconLoss(StochasticReconLossConfig(coeff=1.0)),
        "layerwise": StochasticReconLayerwiseLoss(StochasticReconLayerwiseLossConfig(coeff=1.0)),
        "ppgd": PersistentPGDReconLoss(
            PersistentPGDReconLossConfig(
                coeff=0.5,
                n_samples=1,
                optimizer=AdamPGDConfig(
                    beta1=0.5,
                    beta2=0.99,
                    lr_schedule=ScheduleConfig(
                        fn_type="constant", start_val=0.01, warmup_pct=0.025
                    ),
                ),
                scope=PerBatchPerPositionScope(),
            )
        ),
    }
    for m in metrics.values():
        m.bind(model=model, device=device)
    return metrics


def _make_ctx(
    model: ComponentModel,
    tokens: Tensor,
    target_out: Tensor,
    pre_weight_acts: dict[str, Tensor],
    weight_deltas: dict[str, Tensor],
    step: int,
    total_steps: int,
    ci: CIOutputs,
) -> MetricContext:
    return MetricContext(
        model=model,
        batch=tokens,
        target_out=target_out,
        pre_weight_acts=pre_weight_acts,
        ci=ci,
        weight_deltas=weight_deltas,
        step=step,
        total_steps=total_steps,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_kl,
        is_eval=False,
    )


def _save_ci_figures(
    model: ComponentModel,
    complete_ci_fn: nn.Module,
    toy: AnyRedundancyToy,
    figures_dir: Path,
    step: int,
    device: str,
) -> None:
    """One subcomponent-CI grid per CI net on the canonical probe batch."""
    tokens = toy.enumerate_inputs().to(device)
    with torch.no_grad():
        cached = model(tokens, cache_type="input")
        for tag, ci_fn in (("normal", model.ci_fn), ("complete", complete_ci_fn)):
            ci = _calc_ci(model, ci_fn, cached.cache)
            cis = {m: t.float().numpy(force=True) for m, t in sorted(ci.lower_leaky.items())}
            fig = plot_subcomponent_grid(cis)
            fig.savefig(figures_dir / f"{tag}_{step:05d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def _save_checkpoints(
    model: ComponentModel,
    complete_ci_fn: nn.Module,
    normal_dir: Path,
    complete_dir: Path,
    step: int,
) -> None:
    state = model.state_dict()
    torch.save(state, normal_dir / f"model_{step}.pth")
    complete_state = dict(state)
    for key, value in complete_ci_fn.state_dict().items():
        full_key = f"ci_fn.{key}"
        assert full_key in complete_state, f"missing CI key in model state: {full_key}"
        complete_state[full_key] = value
    torch.save(complete_state, complete_dir / f"model_{step}.pth")


def main(
    model_path: ModelPath,
    out_dir: str,
    steps: int = 10_000,
    batch_size: int = 512,
    lr: float = 2e-3,
    impmin_coeff: float = 1e-4,
    beta: float = 0.5,
    gamma_final: float = 0.1,
    gamma_anneal_start_frac: float = 0.75,
    save_every: int = 2500,
    plot_every: int = 500,
    subset_selection: bool = False,
    from_scratch: bool = False,
    seed: int = 0,
) -> None:
    """Fine-tune a finished decomposition into a (normal, complete) CI-net pair.

    With `subset_selection`, each step selects a random block subset instead of a
    single block: first the subset size `k` uniform in `[1, n_blocks - 1]`, then a
    uniform size-`k` subset. In configuration B the selected blocks are all masked
    by the complete net simultaneously, so their resurrected components must
    reconstruct jointly (never all blocks — configuration A needs at least one
    original block as backstop).

    With `from_scratch`, `model_path` supplies only the config (target + pd):
    components and CI net are freshly initialized and faithfulness-warmed, so the
    protocol runs ab initio instead of fine-tuning a finished decomposition.
    """
    set_seed(seed)
    device = get_device()
    run = SavedToyModelRedundancyRun.from_path(model_path)
    if from_scratch:
        target_model = build_target(run.cfg.target).to(device)
        target_model.requires_grad_(False)
        target_model.eval()
        model = ComponentModel(
            target_model=target_model,
            run_batch=run_batch_passthrough,
            decomposition_targets=resolve_decomposition_targets(
                target_model, run.cfg.pd.all_decomposition_target_configs
            ),
            ci_config=run.cfg.pd.ci_config,
            sigmoid_type=run.cfg.pd.sigmoid_type,
        ).to(device)
        warmup_params = [p for c in model.components.values() for p in c.parameters()]
        run_faithfulness_warmup(model, warmup_params, run.cfg.pd)
    else:
        model = run.load_model().to(device)
    toy = cast(AnyRedundancyToy, model.target_model)
    complete_ci_fn = copy.deepcopy(model.ci_fn)
    # The CI wrapper computes its input features (V^T x) through a plain-dict reference
    # to the components, which deepcopy clones into dead frozen copies (they are not in
    # the wrapper's state_dict, so a reloaded checkpoint rebinds to the *trained*
    # components — a pairing the live net never saw). Share the live components instead.
    assert hasattr(complete_ci_fn, "components")
    complete_ci_fn.components = model.components

    out = Path(out_dir).expanduser()
    normal_dir, complete_dir = out / "normal", out / "complete"
    for d in (normal_dir, complete_dir):
        d.mkdir(parents=True, exist_ok=True)
        run.cfg.to_file(d / EXPERIMENT_CONFIG_FILENAME)
    figures_dir = out / "figures"
    figures_dir.mkdir(exist_ok=True)

    component_params = [
        p for name in model.target_module_paths for p in model.components[name].parameters()
    ]
    components_opt = optim.AdamW(component_params, lr=lr, weight_decay=0.0)
    ci_opt = optim.AdamW(
        [*model.ci_fn.parameters(), *complete_ci_fn.parameters()], lr=lr, weight_decay=0.0
    )

    recipe = dict(
        impmin_coeff=impmin_coeff,
        beta=beta,
        gamma_final=gamma_final,
        gamma_anneal_start_frac=gamma_anneal_start_frac,
    )
    metrics_a = _recipe_metrics(model, device, **recipe)
    metrics_b = _recipe_metrics(model, device, **recipe)

    (out / "complete_config.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "steps": steps,
                "batch_size": batch_size,
                "lr": lr,
                "plot_every": plot_every,
                "save_every": save_every,
                "subset_selection": subset_selection,
                "from_scratch": from_scratch,
                "seed": seed,
                "losses": {name: m.cfg.model_dump() for name, m in metrics_a.items()},
            },
            indent=2,
        )
    )

    n_blocks = 1 + max(_block_index(m) for m in model.target_module_paths)
    logger.info(f"completeness fine-tune: {n_blocks} blocks, {steps} steps, device {device}")

    with (out / "metrics.jsonl").open("w") as metrics_file:
        for step in tqdm(range(steps + 1), ncols=0):
            components_opt.zero_grad()
            ci_opt.zero_grad()

            tokens = toy.sample_inputs(batch_size).to(device)
            target = model(tokens, cache_type="input")
            weight_deltas = model.calc_weight_deltas()
            normal_ci = _calc_ci(model, model.ci_fn, target.cache)
            complete_ci = _calc_ci(model, complete_ci_fn, target.cache)

            if subset_selection:
                k = int(torch.randint(1, n_blocks, (1,)).item())
                blocks = torch.randperm(n_blocks)[:k].tolist()
            else:
                blocks = [int(torch.randint(0, n_blocks, (1,)).item())]
            block_modules = [m for m in model.target_module_paths if _block_index(m) in blocks]
            selected = set(block_modules)

            ctx_args = (model, tokens, target.output, target.cache, weight_deltas, step, steps)
            ctx_a = _make_ctx(*ctx_args, _restrict(normal_ci, block_modules))
            ctx_a_ppgd = _make_ctx(
                *ctx_args, _mix(normal_ci, _ones(normal_ci), selected, detach_rest=False)
            )
            ctx_b = _make_ctx(*ctx_args, _mix(complete_ci, normal_ci, selected, detach_rest=True))
            ctx_b_min = _make_ctx(*ctx_args, _restrict(complete_ci, block_modules))

            plan: list[tuple[str, Metric[Any], MetricContext]] = [
                ("A/stoch", metrics_a["stoch"], ctx_a),
                ("A/layerwise", metrics_a["layerwise"], ctx_a),
                ("A/ppgd", metrics_a["ppgd"], ctx_a_ppgd),
                ("A/impmin", metrics_a["impmin"], ctx_a),
                ("B/stoch", metrics_b["stoch"], ctx_b),
                ("B/layerwise", metrics_b["layerwise"], ctx_b),
                ("B/ppgd", metrics_b["ppgd"], ctx_b),
                ("B/impmin", metrics_b["impmin"], ctx_b_min),
            ]

            total = torch.zeros((), device=device)
            live_losses: list[tuple[Metric[Any], Tensor | None]] = []
            log_data: dict[str, float] = {}
            for loss_name, metric, metric_ctx in plan:
                loss_val = metric.update(metric_ctx)
                live_losses.append((metric, loss_val))
                if loss_val is None:
                    continue
                assert torch.isfinite(loss_val).all(), f"non-finite {loss_name} at step {step}"
                coeff = cast(LossMetricConfig, metric.cfg).coeff
                assert coeff is not None
                total = total + coeff * loss_val
                log_data[f"loss/{loss_name}"] = loss_val.item()

            for metric, loss_val in live_losses:
                metric.before_backward(loss_val)
            total.backward()
            for metric, _ in live_losses:
                metric.after_backward()

            if step % 100 == 0:
                log_data["loss/total"] = total.item()
                metrics_file.write(
                    json.dumps({"step": step, "blocks": sorted(blocks), **log_data}) + "\n"
                )
                metrics_file.flush()

            if step == steps or step % plot_every == 0:
                _save_ci_figures(model, complete_ci_fn, toy, figures_dir, step, device)

            if step == steps or (step > 0 and step % save_every == 0):
                _save_checkpoints(model, complete_ci_fn, normal_dir, complete_dir, step)

            if step != steps:
                components_opt.step()
                ci_opt.step()

    logger.info(f"saved normal/complete checkpoints under {out}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
