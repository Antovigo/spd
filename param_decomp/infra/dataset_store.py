"""The dataset-store artifact contract: layout, the reference a config names shards by,
and the self-describing `meta.json`.

A store dataset is a directory of pre-tokenized `*.parquet` shards plus a `meta.json`
carrying the dataset's own facts. Composition roots and consumers read the meta here
and thread the values into the core loader as explicit parameters
(`ShardServer(seq_len=...)`).

`DatasetRef` is the one way any config names shards — the pretrainer and the
decomposition trainer share it. It lives beside the layout because resolving its `name`
arm IS the layout.
"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Discriminator, Field, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig

DATASET_META_FILENAME = "meta.json"


def dataset_dir(data_root: Path, name: str) -> Path:
    """The store layout: a named dataset's shards live at `<data_root>/datasets/<name>`."""
    return data_root / "datasets" / name


class NamedDataset(BaseConfig):
    """A dataset in the store: shards + `meta.json` at `<data_root>/datasets/<name>`.
    Names are immutable versions — a changed dataset is a new name."""

    kind: Literal["name"] = "name"
    name: str

    @model_validator(mode="after")
    def _flat_name(self) -> Self:
        assert self.name and "/" not in self.name and "*" not in self.name, (
            f"dataset names are flat store names: {self.name!r}"
        )
        return self


class DatasetDir(BaseConfig):
    """Ad-hoc escape hatch: an explicit directory of `*.parquet` shards. Machine-specific
    by nature, so the path must be absolute; a named store dataset is the portable form."""

    kind: Literal["dir"] = "dir"
    dir: Path

    @model_validator(mode="after")
    def _absolute(self) -> Self:
        assert self.dir.is_absolute(), f"ad-hoc shard dirs are absolute paths: {self.dir}"
        return self


DatasetRef = Annotated[NamedDataset | DatasetDir, Discriminator("kind")]


def resolve_dataset_ref(ref: DatasetRef, data_root: Path) -> Path:
    match ref:
        case NamedDataset(name=name):
            return dataset_dir(data_root, name)
        case DatasetDir(dir=dir):
            return dir


class DatasetMeta(BaseConfig):
    """`seq_len` is the training sequence length — rows may carry `seq_len` or
    `seq_len + 1` tokens (next-token staging), so the artifact is ambiguous without it.
    `tokenizer_name` is the tokenizer that produced the ids — the decode authority for
    consumers rendering harvested tokens."""

    seq_len: PositiveInt
    tokenizer_name: str = Field(min_length=1)


def read_dataset_meta(data_dir: Path) -> DatasetMeta:
    path = data_dir / DATASET_META_FILENAME
    assert path.exists(), (
        f"no {DATASET_META_FILENAME} in {data_dir}: dataset dirs are self-describing "
        "(prestage writes it; a pre-meta dir needs a one-time backfill)"
    )
    return DatasetMeta.model_validate_json(path.read_text())


def write_dataset_meta(data_dir: Path, meta: DatasetMeta) -> None:
    (data_dir / DATASET_META_FILENAME).write_text(meta.model_dump_json() + "\n")
