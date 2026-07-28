"""The dataset-store artifact contract: layout + the self-describing `meta.json`.

A store dataset is a directory of pre-tokenized `*.parquet` shards plus a `meta.json`
carrying the dataset's own facts. Composition roots and consumers read the meta here
and thread the values into the core loader as explicit parameters
(`ShardServer(seq_len=...)`).
"""

from pathlib import Path

from pydantic import Field, PositiveInt

from param_decomp.core.base_config import BaseConfig

DATASET_META_FILENAME = "meta.json"


def dataset_dir(data_root: Path, name: str) -> Path:
    """The store layout: a named dataset's shards live at `<data_root>/datasets/<name>`."""
    return data_root / "datasets" / name


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
