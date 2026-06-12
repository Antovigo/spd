"""Lab tests for targeted decomposition: configs, datasets, eval wiring, train smoke."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import PersistentPGDReconLossConfig
from param_decomp.metrics.persistent_pgd_state import SignPGDConfig, SingleSourceScope
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLossConfig
from param_decomp.metrics.stochastic_recon import StochasticReconLossConfig
from param_decomp.metrics.unmasked_recon import UnmaskedReconLossConfig
from param_decomp.optimize import EvalLoop, NontargetEvalPass, NontargetTrainPass, Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp.targeted import get_delta_override
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.eval_metrics.nontarget_ci_mean_per_component import (
    NontargetCIMeanPerComponent,
    NontargetCIMeanPerComponentConfig,
)
from param_decomp_lab.eval_metrics.targeted_ci_heatmap import (
    TargetedCIHeatmap,
    TargetedCIHeatmapConfig,
)
from param_decomp_lab.eval_metrics.targeted_recon_loss import (
    NontargetReconLoss,
    NontargetReconLossConfig,
    TargetReconLoss,
    TargetReconLossConfig,
)
from param_decomp_lab.eval_metrics.weight_magnitude import WeightMagnitude, WeightMagnitudeConfig
from param_decomp_lab.experiments.lm.data import LMDataConfig
from param_decomp_lab.experiments.lm.prompts_dataset import StaticBatchLoader, load_prompts_dataset
from param_decomp_lab.experiments.tms.data import SparseFeatureDataset
from param_decomp_lab.experiments.tms.models import TMSModel, TMSModelConfig
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.targeted import build_nontarget_loss_configs, split_eval_metrics

# =========================== build_nontarget_loss_configs ===========================


def _ppgd_config() -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=1.0,
        optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.1)),
        scope=SingleSourceScope(),
    )


class TestBuildNontargetLossConfigs:
    def test_drops_excluded_and_scales_impmin(self):
        configs: list[Any] = [
            ImportanceMinimalityLossConfig(coeff=2.0, pnorm=2.0, beta=0.0),
            StochasticReconLossConfig(coeff=1.0),
            UnmaskedReconLossConfig(coeff=1.0),
            StochasticHiddenActsReconLossConfig(coeff=1.0),
            _ppgd_config(),
        ]
        out = build_nontarget_loss_configs(configs, 0.5, nontarget_batch_size=16)
        types = [type(c).__name__ for c in out]
        assert types == ["ImportanceMinimalityLossConfig", "StochasticReconLossConfig"]
        impmin = out[0]
        assert isinstance(impmin, ImportanceMinimalityLossConfig)
        assert impmin.coeff == 1.0
        # originals untouched
        assert configs[0].coeff == 2.0

    def test_keeps_stochastic_recon(self):
        out = build_nontarget_loss_configs(
            [StochasticReconLossConfig(coeff=3.0)], 1.0, nontarget_batch_size=4
        )
        assert len(out) == 1 and out[0].coeff == 3.0


# =========================== validators ===========================


def _minimal_pd_config(**overrides: Any) -> PDConfig:
    defaults: dict[str, Any] = dict(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        decomposition_targets=[DecompositionTargetConfig(module_pattern="linear1", C=8)],
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=2.0, beta=0.0),
            StochasticReconLossConfig(coeff=1.0),
        ],
        components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
        batch_size=8,
        steps=2,
    )
    defaults.update(overrides)
    return PDConfig(**defaults)


def _tms_experiment_config_dict(pd: PDConfig) -> dict[str, Any]:
    return {
        "pd": pd.model_dump(),
        "runtime": {"device": "cpu", "autocast_bf16": False},
        "cadence": {"train_log_every": 1},
        "target": {"run_path": "some/run/path"},
        "data": {"feature_probability": 0.05, "active_indices": [0, 1]},
        "nontarget": {
            "data": {"feature_probability": 0.05},
            "batch_size": 8,
            "eval_batch_size": 8,
            "impmin_coeff_ratio": 1.0,
        },
    }


class TestTargetedValidators:
    def _validate(self, cfg_dict: dict[str, Any]) -> None:
        from param_decomp_lab.experiments.tms.run import TMSExperimentConfig

        TMSExperimentConfig.model_validate(cfg_dict)

    def test_valid_targeted_config_passes(self):
        self._validate(_tms_experiment_config_dict(_minimal_pd_config()))

    def test_use_delta_component_false_raises(self):
        cfg = _tms_experiment_config_dict(_minimal_pd_config(use_delta_component=False))
        with pytest.raises(Exception, match="use_delta_component"):
            self._validate(cfg)

    def test_faithfulness_loss_raises(self):
        pd = _minimal_pd_config(
            loss_metrics=[
                FaithfulnessLossConfig(coeff=1.0),
                StochasticReconLossConfig(coeff=1.0),
            ]
        )
        with pytest.raises(Exception, match="FaithfulnessLoss"):
            self._validate(_tms_experiment_config_dict(pd))

    def test_faithfulness_warmup_raises(self):
        cfg = _tms_experiment_config_dict(_minimal_pd_config(faithfulness_warmup_steps=10))
        with pytest.raises(Exception, match="warmup"):
            self._validate(cfg)

    def test_missing_batch_size_is_schema_error(self):
        cfg = _tms_experiment_config_dict(_minimal_pd_config())
        del cfg["nontarget"]["batch_size"]
        with pytest.raises(Exception, match="batch_size"):
            self._validate(cfg)

    def test_lm_data_config_exactly_one_source(self):
        common: dict[str, Any] = {"tokenizer_name": "gpt2"}
        with pytest.raises(Exception, match="exactly one"):
            LMDataConfig(**common)
        with pytest.raises(Exception, match="exactly one"):
            LMDataConfig(dataset_name="d", prompts_file="p.txt", **common)
        LMDataConfig(dataset_name="d", **common)
        LMDataConfig(prompts_file="p.txt", **common)


# =========================== active_indices ===========================


class TestActiveIndices:
    def _dataset(self, **kwargs: Any) -> SparseFeatureDataset:
        defaults: dict[str, Any] = dict(
            n_features=10,
            feature_probability=0.5,
            device="cpu",
            batch_size=64,
        )
        defaults.update(kwargs)
        return SparseFeatureDataset(**defaults)

    @pytest.mark.parametrize(
        "data_generation_type", ["exactly_one_active", "exactly_two_active", "at_least_zero_active"]
    )
    def test_only_listed_columns_nonzero(self, data_generation_type: str):
        active = [1, 4, 7]
        ds = self._dataset(data_generation_type=data_generation_type, active_indices=active)
        batch, _ = ds.generate_batch(256)
        inactive = [i for i in range(10) if i not in active]
        assert (batch[:, inactive] == 0).all()
        assert (batch[:, active] != 0).any()

    def test_exactly_n_active_count_preserved(self):
        ds = self._dataset(data_generation_type="exactly_two_active", active_indices=[0, 3, 9])
        batch, _ = ds.generate_batch(128)
        assert ((batch != 0).sum(dim=-1) == 2).all()

    def test_out_of_range_raises(self):
        with pytest.raises(AssertionError):
            self._dataset(active_indices=[0, 10])

    def test_none_unchanged(self):
        ds = self._dataset(active_indices=None)
        batch, _ = ds.generate_batch(512)
        assert (batch != 0).any(dim=0).all()  # every column activates somewhere


# =========================== prompts_dataset ===========================


class FakeTokenizer:
    """Maps each character to its codepoint."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


@pytest.fixture
def prompts_file(tmp_path: Path) -> str:
    path = tmp_path / "prompts.txt"
    path.write_text("abc\n\ndef\nghi\n")
    return str(path)


class TestPromptsDataset:
    def test_loads_constant_length(self, prompts_file: str):
        pool = load_prompts_dataset(prompts_file, FakeTokenizer())  # pyright: ignore[reportArgumentType]
        assert pool.shape == (3, 3)  # blank line skipped
        assert pool[0].tolist() == [ord("a"), ord("b"), ord("c")]

    def test_raises_on_length_mismatch(self, tmp_path: Path):
        path = tmp_path / "mixed.txt"
        path.write_text("abc\nde\n")
        with pytest.raises(AssertionError, match="must share one length"):
            load_prompts_dataset(str(path), FakeTokenizer())  # pyright: ignore[reportArgumentType]

    def test_static_batch_loader_samples_pool(self, prompts_file: str):
        pool = load_prompts_dataset(prompts_file, FakeTokenizer())  # pyright: ignore[reportArgumentType]
        loader = StaticBatchLoader(pool, batch_size=2, seed=0)
        it = iter(loader)
        batches = [next(it) for _ in range(20)]
        assert all(b.shape == (2, 3) for b in batches)
        pool_rows = {tuple(r.tolist()) for r in pool}
        for b in batches:
            rows = [tuple(r.tolist()) for r in b]
            assert set(rows) <= pool_rows
            assert len(set(rows)) == 2  # without replacement within a batch
        assert len({tuple(map(tuple, (r.tolist() for r in b))) for b in batches}) > 1

    def test_static_batch_loader_reproducible(self, prompts_file: str):
        pool = load_prompts_dataset(prompts_file, FakeTokenizer())  # pyright: ignore[reportArgumentType]
        a = [next(iter(StaticBatchLoader(pool, batch_size=2, seed=7))) for _ in range(1)]
        b = [next(iter(StaticBatchLoader(pool, batch_size=2, seed=7))) for _ in range(1)]
        assert torch.equal(a[0], b[0])

    def test_yields_whole_pool_when_batch_exceeds(self, prompts_file: str):
        pool = load_prompts_dataset(prompts_file, FakeTokenizer())  # pyright: ignore[reportArgumentType]
        batch = next(iter(StaticBatchLoader(pool, batch_size=10, seed=0)))
        assert batch.shape == (3, 3)
        assert {tuple(r.tolist()) for r in batch} == {tuple(r.tolist()) for r in pool}


# =========================== eval wiring ===========================


class TestSplitEvalMetrics:
    def test_partitions_nontarget_metrics(self):
        metrics = [
            EVAL_METRIC_CLASSES["WeightMagnitude"](WeightMagnitudeConfig()),
            EVAL_METRIC_CLASSES["TargetReconLoss"](TargetReconLossConfig(rounding_threshold=0.5)),
            EVAL_METRIC_CLASSES["NontargetReconLoss"](
                NontargetReconLossConfig(rounding_threshold=0.5)
            ),
            EVAL_METRIC_CLASSES["NontargetCIMeanPerComponent"](NontargetCIMeanPerComponentConfig()),
            EVAL_METRIC_CLASSES["TargetedCIHeatmap"](
                TargetedCIHeatmapConfig(active_indices=[0, 1])
            ),
        ]
        target, nontarget = split_eval_metrics(metrics)
        assert [type(m).__name__ for m in target] == ["WeightMagnitude", "TargetReconLoss"]
        assert [type(m).__name__ for m in nontarget] == [
            "NontargetReconLoss",
            "NontargetCIMeanPerComponent",
            "TargetedCIHeatmap",
        ]


# =========================== train-loop smoke ===========================


def _tiny_tms_setup(
    tmp_path: Path, *, targeted: bool
) -> tuple[Trainer, Any, Any, EvalLoop, NontargetTrainPass | None]:
    set_seed(0)
    device = "cpu"
    active_indices = [0, 1, 2]

    target_model = TMSModel(
        config=TMSModelConfig(
            n_features=8,
            n_hidden=4,
            n_hidden_layers=0,
            tied_weights=False,
            init_bias_to_zero=False,
            device=device,
        )
    ).to(device)
    target_model.eval()

    pd_config = _minimal_pd_config(
        decomposition_targets=[
            DecompositionTargetConfig(module_pattern="linear1", C=10),
            DecompositionTargetConfig(module_pattern="linear2", C=10),
        ],
        batch_size=16,
        steps=3,
    )

    def loader(active: list[int] | None, batch_size: int) -> Any:
        from torch.utils.data import DataLoader

        return DataLoader(
            SparseFeatureDataset(
                n_features=8,
                feature_probability=0.2,
                device=device,
                batch_size=batch_size,
                active_indices=active,
            ),
            batch_size=None,
        )

    eval_metrics = [
        TargetReconLoss(TargetReconLossConfig(rounding_threshold=0.5)),
        WeightMagnitude(WeightMagnitudeConfig()),
    ]
    nontarget_eval = None
    nontarget_pass = None
    if targeted:
        nontarget_eval = NontargetEvalPass(
            loader=loader(None, 16),
            metrics=[
                NontargetReconLoss(NontargetReconLossConfig(rounding_threshold=0.5)),
                NontargetCIMeanPerComponent(NontargetCIMeanPerComponentConfig()),
                TargetedCIHeatmap(TargetedCIHeatmapConfig(active_indices=active_indices)),
            ],
        )
        nontarget_pass = NontargetTrainPass(
            loader=loader(None, 16),
            loss_configs=build_nontarget_loss_configs(
                pd_config.loss_metrics, 2.0, nontarget_batch_size=16
            ),
        )

    eval_loop = EvalLoop(
        loader=loader(active_indices if targeted else None, 16),
        metrics=list(eval_metrics),
        n_steps=2,
        every=2,
        slow_every=2,
        slow_on_first_step=True,
        nontarget=nontarget_eval,
    )
    trainer = Trainer(
        target_model=target_model,
        run_batch=run_batch_first_element,
        reconstruction_loss=recon_loss_mse,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(device=device, autocast_bf16=False),
    )
    train_loader = loader(active_indices if targeted else None, 16)
    sink = RunSink.local(tmp_path)
    return trainer, train_loader, sink, eval_loop, nontarget_pass


def _logged_keys(tmp_path: Path) -> set[str]:
    keys: set[str] = set()
    metrics_file = tmp_path / "metrics.jsonl"
    for line in metrics_file.read_text().splitlines():
        keys |= set(json.loads(line))
    return keys


class TestTargetedTrainSmoke:
    def test_targeted_run_emits_nontarget_keys(self, tmp_path: Path):
        trainer, train_loader, sink, eval_loop, nontarget = _tiny_tms_setup(tmp_path, targeted=True)
        trainer.run(train_loader, sink, Cadence(train_log_every=1), eval_loop, nontarget=nontarget)
        assert get_delta_override() is None, "delta override leaked out of the train loop"
        keys = _logged_keys(tmp_path)
        assert "train/nontarget/loss/total" in keys
        assert "train/nontarget/loss/StochasticReconLoss" in keys
        assert "train/nontarget/loss/ImportanceMinimalityLoss" in keys
        for strategy in ("stochastic", "ci_masked", "rounded", "delta_only", "total_l0"):
            assert f"eval/target_recon/{strategy}" in keys
            assert f"eval/nontarget_recon/{strategy}" in keys
        figures = {p.name for p in (tmp_path / "figures").iterdir()}
        assert any(name.startswith("eval_figures_weight_magnitude") for name in figures)
        assert any(name.startswith("eval_figures_targeted_ci_heatmap") for name in figures)
        assert any(
            name.startswith("eval_figures_nontarget_ci_mean_per_component") for name in figures
        )

    def test_nontargeted_run_has_no_nontarget_keys(self, tmp_path: Path):
        trainer, train_loader, sink, eval_loop, _ = _tiny_tms_setup(tmp_path, targeted=False)
        trainer.run(train_loader, sink, Cadence(train_log_every=1), eval_loop, nontarget=None)
        keys = _logged_keys(tmp_path)
        assert not any(k.startswith("train/nontarget") for k in keys)
        assert not any(k.startswith("eval/nontarget_recon") for k in keys)
