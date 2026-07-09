"""Targeted-decomposition convergence tests on pretrained toy targets (slow, GPU).

Each test loads a pretrained target from W&B, runs a targeted decomposition on a
3-feature target distribution with a full-distribution nontarget pass (driven by the
checked-in `*_targeted_config.yaml`), then asserts the CI patterns: identity-like on
the target one-hot probes, (almost) no active components on target-feature-free
nontarget rows.

Run on a GPU node:
    pytest param_decomp_lab/tests/test_targeted_convergence.py --runslow
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.optimize import NontargetTrainPass, Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse
from param_decomp_lab.experiments.resid_mlp.run import (
    ResidMLPExperimentConfig,
    build_resid_mlp_loader,
)
from param_decomp_lab.experiments.resid_mlp.run import (
    _build_eval_loop as _build_resid_mlp_eval_loop,
)
from param_decomp_lab.experiments.resid_mlp.run import build_target as build_resid_mlp_target
from param_decomp_lab.experiments.resid_mlp.run import make_run_batch as make_resid_mlp_run_batch
from param_decomp_lab.experiments.tms.run import (
    TMSExperimentConfig,
    build_tms_loader,
)
from param_decomp_lab.experiments.tms.run import _build_eval_loop as _build_tms_eval_loop
from param_decomp_lab.experiments.tms.run import build_target as build_tms_target
from param_decomp_lab.experiments.tms.run import make_run_batch as make_tms_run_batch
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.targeted import build_nontarget_loss_configs
from param_decomp_lab.toy_models.target_ci import (
    DenseCIPattern,
    IdentityCIPattern,
    TargetCISolution,
)

TMS_YAML = (
    Path(__file__).parent.parent / "experiments" / "tms" / "tms_40-10-id_targeted_config.yaml"
)
RESID_MLP_YAML = (
    Path(__file__).parent.parent / "experiments" / "resid_mlp" / "resid_mlp1_targeted_config.yaml"
)

CI_ALIVE_THRESHOLD = 0.1
PATTERN_TOLERANCE = 0.2
INPUT_MAGNITUDE = 0.75


def _probe_ci(
    model: ComponentModel, active_indices: list[int], n_features: int, device: str
) -> CIOutputs:
    """CI on one-hot probes over `active_indices` (wrapped as an (inputs, labels) pair)."""
    probes = torch.zeros(len(active_indices), n_features, device=device)
    for row, feature_idx in enumerate(active_indices):
        probes[row, feature_idx] = INPUT_MAGNITUDE
    with torch.no_grad():
        pre_weight_acts = model((probes, probes), cache_type="input").cache
        return model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )


def _batch_ci(
    model: ComponentModel, loader: DataLoader[Any], device: str
) -> tuple[torch.Tensor, CIOutputs]:
    batch, labels = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        pre_weight_acts = model((batch, labels.to(device)), cache_type="input").cache
        return batch, model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )


MAX_SPURIOUS_ROW_FRAC = 0.05


def _assert_nontarget_rows_dead(
    batch: torch.Tensor, ci: CIOutputs, layer: str, active_indices: list[int]
) -> None:
    """(Almost) no alive CI on rows where no target feature fires.

    CI is a pure function of the input, so on full-distribution rows that do contain a
    target feature the target-specialized components fire by design — only
    target-feature-free rows are checked. Those can't be *exactly* dead either: for
    layers reading a superposed representation (e.g. TMS linear2's 10-dim hidden), some
    nontarget feature combinations produce hidden states the target model itself cannot
    distinguish from a target feature, and the target components fire there
    (empirically ~3% of rows, confined to the target components). Hence a small
    row-fraction bound rather than exact zero.
    """
    target_free = (batch[:, active_indices] == 0).all(dim=-1)
    n_rows = int(target_free.sum().item())
    assert n_rows > 0, "nontarget batch has no target-free rows"
    alive = ci.lower_leaky[layer] > CI_ALIVE_THRESHOLD
    rows_with_alive = int(alive[target_free].any(dim=-1).sum().item())
    frac = rows_with_alive / n_rows
    assert frac <= MAX_SPURIOUS_ROW_FRAC, (
        f"{layer}: {rows_with_alive}/{n_rows} target-feature-free nontarget rows have alive "
        f"CI ({frac:.1%} > {MAX_SPURIOUS_ROW_FRAC:.0%})"
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="convergence tests need a GPU")
class TestTargetedConvergence:
    def test_tms_40_10_id_three_feature_target(self, tmp_path: Path):
        device = "cuda"
        cfg = TMSExperimentConfig.from_file(TMS_YAML)
        try:
            target_model = build_tms_target(cfg.target).to(device)
        except Exception as e:  # noqa: BLE001 — skip-if-target-unavailable
            pytest.skip(f"pretrained TMS target unavailable: {e}")
        assert cfg.nontarget is not None and cfg.data.active_indices is not None
        active_indices = cfg.data.active_indices
        nontarget_cfg = cfg.nontarget

        set_seed(cfg.pd.seed)
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})
        train_loader = build_tms_loader(
            cfg.target, cfg.data, split="train", device=device, batch_size=cfg.pd.batch_size
        )
        nontarget_loader = build_tms_loader(
            cfg.target,
            nontarget_cfg.data,
            split="train",
            device=device,
            batch_size=nontarget_cfg.batch_size,
        )
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_tms_run_batch(cfg.target),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(
            train_loader,
            RunSink.local(tmp_path),
            cfg.cadence,
            _build_tms_eval_loop(cfg, device),
            nontarget=NontargetTrainPass(
                loader=nontarget_loader,
                loss_configs=build_nontarget_loss_configs(
                    cfg.pd.loss_metrics,
                    nontarget_cfg.impmin_coeff_ratio,
                    nontarget_batch_size=nontarget_cfg.batch_size,
                ),
            ),
        )

        model = trainer.component_model
        target_ci = _probe_ci(model, active_indices, n_features=40, device=device)
        solution = TargetCISolution(
            {
                "linear1": IdentityCIPattern(n_features=3),
                "linear2": IdentityCIPattern(n_features=3),
                "hidden_layers.0": DenseCIPattern(k=5, min_entries=0),
            }
        )
        distance = solution.distance_from(target_ci.lower_leaky, tolerance=PATTERN_TOLERANCE)
        per_layer = {
            layer: ci.max(dim=-1).values.tolist() for layer, ci in target_ci.lower_leaky.items()
        }
        assert distance == 0, f"target CI pattern distance {distance}; row maxima: {per_layer}"

        nontarget_batch, nontarget_ci = _batch_ci(model, nontarget_loader, device)
        for layer in ("linear1", "linear2"):
            _assert_nontarget_rows_dead(nontarget_batch, nontarget_ci, layer, active_indices)

    def test_resid_mlp1_three_feature_target(self, tmp_path: Path):
        device = "cuda"
        cfg = ResidMLPExperimentConfig.from_file(RESID_MLP_YAML)
        try:
            target_model = build_resid_mlp_target(cfg.target).to(device)
        except Exception as e:  # noqa: BLE001 — skip-if-target-unavailable
            pytest.skip(f"pretrained ResidMLP target unavailable: {e}")
        assert cfg.nontarget is not None and cfg.data.active_indices is not None
        active_indices = cfg.data.active_indices
        nontarget_cfg = cfg.nontarget

        set_seed(cfg.pd.seed)
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})
        train_loader = build_resid_mlp_loader(
            cfg.target, cfg.data, split="train", device=device, batch_size=cfg.pd.batch_size
        )
        nontarget_loader = build_resid_mlp_loader(
            cfg.target,
            nontarget_cfg.data,
            split="train",
            device=device,
            batch_size=nontarget_cfg.batch_size,
        )
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_resid_mlp_run_batch(cfg.target),
            reconstruction_loss=recon_loss_mse,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(
            train_loader,
            RunSink.local(tmp_path),
            cfg.cadence,
            _build_resid_mlp_eval_loop(cfg, device),
            nontarget=NontargetTrainPass(
                loader=nontarget_loader,
                loss_configs=build_nontarget_loss_configs(
                    cfg.pd.loss_metrics,
                    nontarget_cfg.impmin_coeff_ratio,
                    nontarget_batch_size=nontarget_cfg.batch_size,
                ),
            ),
        )

        model = trainer.component_model
        target_ci = _probe_ci(model, active_indices, n_features=100, device=device)
        solution = TargetCISolution(
            {
                "layers.*.mlp_in": IdentityCIPattern(n_features=3),
                "layers.*.mlp_out": DenseCIPattern(k=5, min_entries=0),
            }
        )
        distance = solution.distance_from(target_ci.lower_leaky, tolerance=PATTERN_TOLERANCE)
        per_layer = {
            layer: ci.max(dim=-1).values.tolist() for layer, ci in target_ci.lower_leaky.items()
        }
        assert distance == 0, f"target CI pattern distance {distance}; row maxima: {per_layer}"

        nontarget_batch, nontarget_ci = _batch_ci(model, nontarget_loader, device)
        _assert_nontarget_rows_dead(
            nontarget_batch, nontarget_ci, "layers.0.mlp_in", active_indices
        )
