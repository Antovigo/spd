"""Run SPD on a RedundantCopyTransformer model."""

from pathlib import Path

import fire

from spd.configs import CompletenessTaskConfig
from spd.experiments.completeness.models import (
    CompletenessTargetRunInfo,
    CopyTaskDataset,
    RedundantCopyTransformer,
)
from spd.log import logger
from spd.run_spd import run_experiment
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import parse_config, parse_sweep_params


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    launch_id: str | None = None,
    sweep_params_json: str | None = None,
    run_id: str | None = None,
) -> None:
    config = parse_config(config_path, config_json)

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    task_config = config.task_config
    assert isinstance(task_config, CompletenessTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = CompletenessTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = RedundantCopyTransformer.from_run_info(target_run_info)
    target_model.to(device).eval()

    model_config = target_run_info.config.completeness_model_config
    dataset = CopyTaskDataset(
        vocab_size=model_config.vocab_size,
        eq_token=model_config.eq_token,
        device=device,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    run_experiment(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_tag="completeness",
        run_id=run_id,
        launch_id=launch_id,
        evals_id=evals_id,
        sweep_params=parse_sweep_params(sweep_params_json),
        target_model_train_config=target_run_info.config,
    )


if __name__ == "__main__":
    fire.Fire(main)
