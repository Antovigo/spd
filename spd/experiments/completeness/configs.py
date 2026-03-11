from pydantic import PositiveInt

from spd.base_config import BaseConfig
from spd.configs import ScheduleConfig
from spd.spd_types import Probability


class CompletenessModelConfig(BaseConfig):
    vocab_size: PositiveInt
    d_model: PositiveInt
    n_layers: PositiveInt
    seq_len: PositiveInt
    eq_token: int


class CompletenessTrainConfig(BaseConfig):
    wandb_project: str | None = None
    completeness_model_config: CompletenessModelConfig
    layer_dropout_p: Probability
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr_schedule: ScheduleConfig
