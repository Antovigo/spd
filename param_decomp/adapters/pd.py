from functools import cached_property
from pathlib import Path

from param_decomp.autointerp.schemas import ModelMetadata
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.experiments.lm.config import LMExperimentConfig
from param_decomp.experiments.lm.load_run import RunMetadata, run_metadata
from param_decomp.harvest.schemas import get_harvest_dir
from param_decomp.infra.dataset_store import (
    DatasetDir,
    DatasetMeta,
    NamedDataset,
    read_dataset_meta,
    resolve_dataset_ref,
)
from param_decomp.topology.path_schemas import path_schema_for_model_type


def is_jax_run(data_root: Path, decomposition_id: str) -> bool:
    """A JAX single-pool run dir pins its single self-contained run config as
    `launch_config.yaml` and checkpoints with orbax under `ckpts/`; a torch run instead has
    `model_*.pth` and no orbax `ckpts/`. The orbax `ckpts/` dir is the explicit marker."""
    run_dir = get_harvest_dir(data_root, decomposition_id).parent
    return (run_dir / LAUNCH_CONFIG_FILENAME).exists() and (run_dir / "ckpts").is_dir()


class PDAdapter:
    """Autointerp/clustering adapter for a JAX single-pool run, read torch-free from its
    pinned launch config. Autointerp consumes harvest output plus run metadata only — no trained
    components — so the target topology (`n_blocks`, vocab, per-site `(name, C)`) comes
    from `param_decomp.experiments.lm.load_run.run_metadata` (config + pretrain-cache `model_config`,
    no orbax restore); canonical layer descriptions render via the torch-free path schema."""

    def __init__(self, decomposition_id: str, data_root: Path):
        self._run_id = decomposition_id
        self._data_root = data_root

    @cached_property
    def _run_dir(self) -> Path:
        return get_harvest_dir(self._data_root, self._run_id).parent

    @cached_property
    def cfg(self) -> LMExperimentConfig:
        config_path = self._run_dir / LAUNCH_CONFIG_FILENAME
        assert config_path.exists(), f"config not found: {config_path}"
        return LMExperimentConfig.from_file(config_path)

    @cached_property
    def _metadata(self) -> RunMetadata:
        return run_metadata(self._run_dir, data_root=self._data_root)

    @property
    def decomposition_id(self) -> str:
        return self._run_id

    @property
    def vocab_size(self) -> int:
        return self._metadata.vocab_size

    @property
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        return self._metadata.layer_activation_sizes

    @cached_property
    def _dataset_meta(self) -> DatasetMeta:
        return read_dataset_meta(resolve_dataset_ref(self.cfg.data.train, self._data_root))

    @property
    def tokenizer_name(self) -> str:
        return self._dataset_meta.tokenizer_name

    @property
    def model_metadata(self) -> ModelMetadata:
        schema = path_schema_for_model_type(self._metadata.model_type)
        return ModelMetadata(
            n_blocks=self._metadata.n_blocks,
            dataset_name=self._dataset_name(),
            layer_descriptions={
                path: schema.parse_target_path(path).canonical_str()
                for path, _ in self._metadata.layer_activation_sizes
            },
            seq_len=self._dataset_meta.seq_len,
        )

    def _dataset_name(self) -> str:
        """The corpus identity for `DATASET_DESCRIPTIONS` (an ad-hoc dir's basename
        stands in for a name)."""
        match self.cfg.data.train:
            case NamedDataset(name=name):
                return name
            case DatasetDir(dir=dir):
                return dir.name
