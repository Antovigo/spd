"""2-pool LM PD experiment: YAML -> ``TwoPoolTrainer`` glue.

The 2-pool sibling of ``three_pool_run``. Its config — ``TwoPoolLMExperimentConfig`` —
pairs a ``ThreePoolConstrainedPDConfig`` (same four-loss set + frozen algorithm scalars
the 2-pool also honours) with a ``TwoPoolRuntimeConfig`` (core ``RuntimeConfig`` scalars
+ an authored ``TwoPoolTopology``).

Checkpoint / resume / async consolidation are out of scope for this MVP (see
``three_pool/DESIGN.md``): ``cadence.save_every`` must be ``None``.

Run a local DDP smoke directly via
``torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.two_pool_run config.yaml``.
"""

from typing import Self

import fire
from pydantic import model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Cadence
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.distributed import (
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.experiments.lm.data import LMDataConfig
from param_decomp_lab.experiments.lm.run import (
    LMTargetConfig,
    _build_eval_loop,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.lm.three_pool_run import ThreePoolRuntimeConfig
from param_decomp_lab.experiments.utils import EvalConfig, WandbConfig, init_pd_run
from param_decomp_lab.run_sink import ThreePoolSink
from param_decomp_lab.seed import set_seed
from param_decomp_lab.three_pool.pd_config import ThreePoolConstrainedPDConfig
from param_decomp_lab.three_pool.two_pool_config import TwoPoolTopology
from param_decomp_lab.three_pool.two_pool_optimize import TwoPoolTrainer


class TwoPoolRuntimeConfig(ThreePoolRuntimeConfig):
    """Core's substrate scalars + a 2-pool ``topology``. Subclasses
    ``ThreePoolRuntimeConfig`` only to narrow the topology field's type."""

    topology: TwoPoolTopology  # pyright: ignore[reportIncompatibleVariableOverride]


class TwoPoolLMExperimentConfig(BaseConfig):
    """Full YAML schema for a 2-pool LM PD run."""

    pd: ThreePoolConstrainedPDConfig
    runtime: TwoPoolRuntimeConfig
    cadence: Cadence
    target: LMTargetConfig
    data: LMDataConfig
    eval: EvalConfig | None = None
    wandb: WandbConfig | None = None

    @model_validator(mode="after")
    def validate_pd_against_topology(self) -> Self:
        topology = self.runtime.topology
        bs = self.pd.batch_size
        for name, per_rank_batch in (
            ("pool_a", topology.pool_a.per_rank_batch),
            ("chunkwise", topology.chunkwise.per_rank_batch),
        ):
            assert bs % per_rank_batch == 0, (
                f"pd.batch_size ({bs}) must be divisible by topology.{name}.per_rank_batch "
                f"({per_rank_batch})"
            )
        assert self.cadence.save_every is None, (
            "2-pool MVP does not implement checkpointing; set cadence.save_every: null"
        )
        return self


@with_distributed_cleanup
def main(config_path: str, *, group: str | None = None, tags: str | None = None) -> None:
    """Run a 2-pool LM PD experiment from a YAML (local DDP via torchrun)."""
    cfg = TwoPoolLMExperimentConfig.from_file(config_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            )
        }
    )

    target_model = build_target(cfg.target)
    # 2-pool requires the full global batch on every rank — each pool slices it locally.
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=None,
        seed=cfg.pd.seed,
    )

    sink = init_pd_run(cfg, sink_class=ThreePoolSink, group=group, tags=tags)
    eval_loop = _build_eval_loop(cfg, device, dist_state=None, include_slow=False)
    try:
        trainer = TwoPoolTrainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
            two_pool_config=cfg.runtime.topology,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop=eval_loop)
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
