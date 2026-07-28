"""The `data: DatasetRef` schema: store names resolve under `data_root`, ad-hoc dirs are
absolute, datasets self-describe via `meta.json`, and stored pins predating the schema
(`data_files` globs) migrate on load."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from param_decomp.experiments.lm.config import DatasetDir, NamedDataset, migrate_glob_pin_data
from param_decomp.infra.dataset_store import (
    DatasetMeta,
    dataset_dir,
    read_dataset_meta,
    write_dataset_meta,
)


def test_store_name_resolves_under_data_root() -> None:
    assert dataset_dir(Path("/project"), "pile") == Path("/project/datasets/pile")


def test_dataset_names_are_flat() -> None:
    with pytest.raises(ValidationError, match="flat store names"):
        NamedDataset.model_validate({"kind": "name", "name": "datasets/pile"})


def test_ad_hoc_dirs_are_absolute() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        DatasetDir.model_validate({"kind": "dir", "dir": "relative/shards"})


def test_dataset_meta_round_trips(tmp_path: Path) -> None:
    meta = DatasetMeta(seq_len=512, tokenizer_name="EleutherAI/gpt-neox-20b")
    write_dataset_meta(tmp_path, meta)
    assert read_dataset_meta(tmp_path) == meta


def test_dataset_meta_missing_refuses(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="self-describing"):
        read_dataset_meta(tmp_path)


def _glob_pin_data(data_files: str) -> dict[str, object]:
    return {
        "dataset_name": "parquet",
        "data_files": data_files,
        "tokenizer_name": "t",
        "max_seq_len": 512,
        "revision": None,
        "column_name": "input_ids",
        "train_split": "train",
        "eval_split": "train",
        "is_tokenized": True,
        "streaming": False,
        "buffer_size": 1000,
        "shuffle_each_epoch": True,
    }


def test_stored_pin_glob_migrates_to_store_name() -> None:
    migrated = migrate_glob_pin_data(_glob_pin_data("datasets/pile_neox_tok_512/*.parquet"))
    assert migrated == {"kind": "name", "name": "pile_neox_tok_512"}


def test_stored_pin_absolute_glob_migrates_to_dir() -> None:
    migrated = migrate_glob_pin_data(
        _glob_pin_data("/abs/x/datasets/fineweb_llama_tok_2048/*.parquet")
    )
    assert migrated == {"kind": "dir", "dir": "/abs/x/datasets/fineweb_llama_tok_2048"}


def test_stored_pin_non_store_relative_glob_refuses() -> None:
    with pytest.raises(AssertionError, match="datasets/<name>/"):
        migrate_glob_pin_data(_glob_pin_data("sample/350BT/*.parquet"))
