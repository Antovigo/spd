"""Config parsing: the committed production config loads; unknown keys raise."""

from pathlib import Path

import pytest
import yaml

from jax_single_pool.config import load_config
from jax_single_pool.llama8b import mlp_family_site_cs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_production_config_loads():
    cfg = load_config(CONFIGS / "llama8b_l18_b512.yaml")
    assert cfg.data.global_batch == 512
    assert cfg.target.sites == mlp_family_site_cs(18, 18, 24576)
    assert cfg.losses.faith == 1e5 and cfg.imp_min.p_start == 2.0
    from jax_single_pool.train import SourceAdamConfig

    assert isinstance(cfg.adversary, SourceAdamConfig)
    assert cfg.adversary.n_warmup == 2 and cfg.vu_optimizer.grad_clip_norm == 0.01
    assert cfg.wandb is not None and cfg.wandb.project == "param-decomp-llama"
    assert cfg.run_dir == cfg.out_dir / cfg.run_name


def test_smoke_config_loads():
    cfg = load_config(CONFIGS / "llama8b_l18_smoke8.yaml")
    assert cfg.wandb is None
    assert cfg.data.global_batch == 32


def test_unknown_key_raises(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_smoke8.yaml").read_text())
    raw["ppgd"]["typo_key"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError, match="unknown keys"):
        load_config(bad)


def test_missing_key_raises(tmp_path: Path):
    raw = yaml.safe_load((CONFIGS / "llama8b_l18_smoke8.yaml").read_text())
    del raw["losses"]["ppgd"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError, match="missing keys"):
        load_config(bad)
