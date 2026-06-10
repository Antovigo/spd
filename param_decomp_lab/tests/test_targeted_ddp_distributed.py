"""2-rank DDP smoke for the targeted nontarget pass (gloo/CPU).

Exercises the full-parameter-coverage requirement of the nontarget backward under
DDP's default reducer (`find_unused_parameters=False`) — invisible in single-process
runs. A coverage violation fails loudly with "Expected to have finished reduction".

Run directly:
    torchrun --standalone --nproc_per_node=2 --master_port=29517 \
        param_decomp_lab/tests/test_targeted_ddp_distributed.py
or via pytest (spawns torchrun in a subprocess; `--runslow`).
"""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import DataLoader

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.stochastic_recon import StochasticReconLossConfig
from param_decomp.optimize import NontargetTrainPass, Trainer
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.distributed import cleanup_distributed, init_distributed
from param_decomp_lab.experiments.tms.data import SparseFeatureDataset
from param_decomp_lab.experiments.tms.models import TMSModel, TMSModelConfig
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.targeted import build_nontarget_loss_configs


def _loader(active_indices: list[int] | None, batch_size: int) -> DataLoader[Any]:
    return DataLoader(
        SparseFeatureDataset(
            n_features=8,
            feature_probability=0.2,
            device="cpu",
            batch_size=batch_size,
            active_indices=active_indices,
        ),
        batch_size=None,
    )


def _run_test() -> None:
    init_distributed()
    try:
        set_seed(0)
        target_model = TMSModel(
            config=TMSModelConfig(
                n_features=8,
                n_hidden=4,
                n_hidden_layers=0,
                tied_weights=False,
                init_bias_to_zero=False,
                device="cpu",
            )
        )
        target_model.eval()

        pd_config = PDConfig(
            seed=0,
            n_mask_samples=1,
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
            decomposition_targets=[
                DecompositionTargetConfig(module_pattern="linear1", C=10),
                DecompositionTargetConfig(module_pattern="linear2", C=10),
            ],
            loss_metrics=[
                ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=2.0, beta=0.0),
                StochasticReconLossConfig(coeff=1.0),
            ],
            components_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
            ci_fn_optimizer=OptimizerConfig(lr_schedule=ScheduleConfig(start_val=1e-3)),
            batch_size=16,
            steps=3,
        )
        trainer = Trainer(
            target_model=target_model,
            run_batch=run_batch_first_element,
            reconstruction_loss=recon_loss_mse,
            pd_config=pd_config,
            runtime_config=RuntimeConfig(device="cpu", autocast_bf16=False),
        )
        nontarget = NontargetTrainPass(
            loader=_loader(None, 8),
            loss_configs=build_nontarget_loss_configs(
                pd_config.loss_metrics, 1.0, nontarget_batch_size=8
            ),
        )
        trainer.run(
            _loader([0, 1, 2], 8),
            RunSink.silent(),
            Cadence(train_log_every=1),
            None,
            nontarget=nontarget,
        )
    finally:
        cleanup_distributed()


@pytest.mark.slow
class TestTargetedDDP:
    def test_targeted_nontarget_pass_2rank_ddp(self):
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "--master_port",
            "29517",
            str(Path(__file__).resolve()),
        ]
        new_env = os.environ.copy()
        new_env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(cmd, env=new_env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, (
            f"DDP targeted smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    _run_test()
