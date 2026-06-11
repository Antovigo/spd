"""torch `LMExperimentConfig` YAML → the JAX trainer's `ExperimentConfig`.

The shared `param-decomp-config` package (torch-free pydantic schema, same repo, branch
`refactor/shared-config-package`) validates the torch run YAML; this module maps the
subspace this trainer implements onto its knobs and ASSERTS loudly on anything else —
a torch config either converts exactly or refuses to run, never silently approximates.

Entry: a small wrapper YAML carrying what the torch schema cannot express —

    torch_config: <path, relative to the wrapper>   # the torch LMExperimentConfig yaml
    run_id: p-1a2b3c4d        # canonical id (generate: secrets.token_hex(4)); run dir
                              # name + wandb id — the torch runs/<id>/ convention
    run_name: my-run          # human-readable wandb display name
    out_dir: /mnt/data/.../param-decomp/runs
    remat_recon_forwards: false                     # jax-runtime memory/compute trade

`jsp-train` detects the `torch_config` key and routes here (`load_torch_wrapper`).

Knowingly ignored torch fields (runtime details with no JAX analog, or JAX-side
equivalents derived elsewhere): `runtime.device/dp` (GSPMD owns placement),
`target.activation_checkpointing` (the wrapper's `remat_recon_forwards` is the
explicit analog), `target.output_extract`, `data.buffer_size/shuffle_each_epoch/
train_split/eval_split` (the JAX data schedule is deterministic by construction),
`eval.slow_every/slow_on_first_step` (no slow in-loop metrics; plot/slow metrics run
offline via `jsp-export` + `pd-offline-eval`), `use_fused_kl` (torch impl detail).
"""

import re
from pathlib import Path
from typing import Any

import yaml
from param_decomp_config.eval_metrics import CEandKLLossesConfig, CI_L0Config
from param_decomp_config.lm import HFTarget, HFWeightsInVendored, LMExperimentConfig
from param_decomp_config.losses import (
    AdamPGDConfig,
    BroadcastAcrossBatchScope,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    StochasticReconSubsetLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp_config.pd import OptimizerConfig
from param_decomp_config.schedule import ScheduleConfig

from jax_single_pool.ci_fn import CIArch
from jax_single_pool.config import (
    CadenceConfig,
    CIOptimizerConfig,
    DataConfig,
    DenseLogPhase,
    EvalConfig,
    EvalPGDConfig,
    ExperimentConfig,
    FaithWarmupConfig,
    ReconConfig,
    TargetConfig,
    VUOptimizerConfig,
    WandbConfig,
    load_config,
)
from jax_single_pool.train import ImpMinConfig, LossCoeffs, SourceAdamConfig

OFFLINE_EVAL_METRIC_TYPES = frozenset(
    {
        "CIHistograms",
        "ComponentActivationDensity",
        "CIMeanPerComponent",
        "StochasticHiddenActsReconLoss",
        "CIHiddenActsReconLoss",
        "UVPlots",
        "PermutedCIPlots",
        "IdentityCIError",
        "CIMaskedAttnPatternsReconLoss",
        "StochasticAttnPatternsReconLoss",
        "AutointerpLabels",
    }
)

_SITE_PATTERN = re.compile(r"^(?:model\.)?layers\.(\d+)\.mlp\.(gate|up|down)_proj$")
"""Raw-HF specs name modules `model.layers.*`; the vendored class drops the prefix.
Same matrices either way."""


def _layer_range_and_c(cfg: LMExperimentConfig) -> tuple[int, int, int]:
    """Targets must be exactly the gate/up/down MLP projections of one contiguous
    layer range, all at the same C — the only site family this trainer implements."""
    assert cfg.pd.identity_decomposition_targets is None, "identity targets unsupported"
    per_layer_kinds: dict[int, set[str]] = {}
    c_values = set()
    for target in cfg.pd.decomposition_targets:
        match = _SITE_PATTERN.match(target.module_pattern)
        assert match, f"unsupported decomposition target {target.module_pattern!r}"
        per_layer_kinds.setdefault(int(match.group(1)), set()).add(match.group(2))
        c_values.add(target.C)
    assert len(c_values) == 1, f"per-site C values differ: {sorted(c_values)}"
    layers = sorted(per_layer_kinds)
    assert layers == list(range(layers[0], layers[-1] + 1)), f"non-contiguous layers {layers}"
    for layer, kinds in per_layer_kinds.items():
        assert kinds == {"gate", "up", "down"}, f"layer {layer} has partial sites {kinds}"
    return layers[0], layers[-1], c_values.pop()


def _ci_arch(cfg: LMExperimentConfig, seq_len: int) -> CIArch:
    ci = cfg.pd.ci_config
    assert ci.mode == "global" and ci.fn_type == "global_shared_transformer", ci
    transformer = ci.simple_transformer_ci_cfg
    assert transformer is not None
    assert transformer.mlp_hidden_dim is not None and len(transformer.mlp_hidden_dim) == 1, (
        f"CI MLP must be single-hidden-layer, got {transformer.mlp_hidden_dim}"
    )
    assert transformer.attn_config.rope_base == 10000.0, transformer.attn_config
    assert transformer.attn_config.max_len >= seq_len, (transformer.attn_config.max_len, seq_len)
    return CIArch(
        d_model=transformer.d_model,
        n_blocks=transformer.n_blocks,
        n_heads=transformer.attn_config.n_heads,
        mlp_hidden=transformer.mlp_hidden_dim[0],
    )


def _assert_cosine_to_tenth(schedule: ScheduleConfig, who: str) -> None:
    """The trainer hardcodes optax cosine decay to 0.1x with no warmup (SPEC S19/S20)."""
    assert schedule.fn_type == "cosine", f"{who}: only cosine lr supported, got {schedule}"
    assert schedule.warmup_pct == 0.0, f"{who}: lr warmup unsupported, got {schedule}"
    assert schedule.final_val_frac == 0.1, f"{who}: final_val_frac must be 0.1, got {schedule}"


def _assert_plain_adamw(optimizer: OptimizerConfig, who: str) -> None:
    assert optimizer.betas == (0.9, 0.999), f"{who}: betas must be (0.9, 0.999)"
    assert optimizer.weight_decay == 0.0, f"{who}: weight_decay must be 0"


def _losses(
    cfg: LMExperimentConfig, n_sites: int
) -> tuple[LossCoeffs, ImpMinConfig, SourceAdamConfig, int, int]:
    """Returns (coeffs, imp_min, ppgd, sites_per_chunk, n_recon_samples). The four
    production losses must each appear exactly once; any other loss metric refuses."""
    faith = imp = stoch = ppgd = None
    sites_per_chunk = n_recon_samples = None
    for metric in cfg.pd.loss_metrics:
        assert metric.coeff is not None
        match metric:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = metric.coeff
            case ImportanceMinimalityLossConfig():
                assert imp is None
                imp = metric
            case StochasticReconSubsetLossConfig():
                assert stoch is None and sites_per_chunk is None
                assert isinstance(metric.routing, UniformKSubsetRoutingConfig), metric.routing
                stoch = metric.coeff
                sites_per_chunk = n_sites
                n_recon_samples = cfg.pd.n_mask_samples
            case ChunkwiseSubsetReconLossConfig():
                assert stoch is None and sites_per_chunk is None
                assert isinstance(metric.routing, UniformKSubsetRoutingConfig), metric.routing
                stoch = metric.coeff
                sites_per_chunk = metric.sites_per_chunk
                n_recon_samples = metric.n_samples
            case PersistentPGDReconLossConfig():
                assert ppgd is None
                ppgd = metric
            case _:
                raise AssertionError(f"unsupported loss metric {metric.type!r}")
    assert faith is not None and imp is not None and stoch is not None and ppgd is not None, (
        f"need all four production losses, got {[m.type for m in cfg.pd.loss_metrics]}"
    )
    assert sites_per_chunk is not None and n_recon_samples is not None

    assert imp.coeff is not None and imp.p_anneal_final_p is not None
    imp_min = ImpMinConfig(
        beta=imp.beta,
        eps=imp.eps,
        p_start=imp.pnorm,
        p_final=imp.p_anneal_final_p,
        anneal_start_frac=imp.p_anneal_start_frac,
        anneal_end_frac=imp.p_anneal_end_frac,
    )

    assert isinstance(ppgd.scope, BroadcastAcrossBatchScope), ppgd.scope
    assert not ppgd.use_sigmoid_parameterization and ppgd.start_frac == 0.0, ppgd
    assert ppgd.n_samples == 1, ppgd
    adversary_optimizer = ppgd.optimizer
    assert isinstance(adversary_optimizer, AdamPGDConfig), adversary_optimizer
    source_schedule = adversary_optimizer.lr_schedule
    assert source_schedule.fn_type == "constant" and source_schedule.final_val_frac == 1.0, (
        source_schedule
    )
    assert ppgd.coeff is not None
    source_adam = SourceAdamConfig(
        lr=source_schedule.start_val,
        lr_warmup_frac=source_schedule.warmup_pct,
        beta1=adversary_optimizer.beta1,
        beta2=adversary_optimizer.beta2,
        eps=adversary_optimizer.eps,
        n_warmup=ppgd.n_warmup_steps,
    )
    coeffs = LossCoeffs(faith=faith, imp=imp.coeff, stoch=stoch, ppgd=ppgd.coeff)
    return coeffs, imp_min, source_adam, sites_per_chunk, n_recon_samples


def _data(cfg: LMExperimentConfig) -> DataConfig:
    data = cfg.data
    assert data.is_tokenized and not data.streaming, (
        "JAX trainer reads pre-tokenized parquet shards; tokenize offline first"
    )
    assert data.dataset_name == "parquet" and data.column_name == "input_ids", data
    assert data.data_files is not None
    shard_glob = Path(data.data_files)
    assert shard_glob.name == "*.parquet", f"expected a *.parquet glob, got {data.data_files}"
    return DataConfig(
        dir=shard_glob.parent, seq_len=data.max_seq_len, global_batch=cfg.pd.batch_size
    )


def _eval(cfg: LMExperimentConfig) -> EvalConfig | None:
    if cfg.eval is None:
        return None
    ce_kl = ci_l0 = pgd = None
    skipped_offline: list[str] = []
    for metric in cfg.eval.metrics:
        match metric:
            case CEandKLLossesConfig():
                ce_kl = metric
            case CI_L0Config():
                assert metric.groups is None, "CI_L0 groups unsupported in-loop"
                ci_l0 = metric
            case PGDReconLossConfig():
                assert metric.init == "random" and metric.mask_scope == "shared_across_batch", (
                    metric
                )
                pgd = EvalPGDConfig(n_steps=metric.n_steps, step_size=metric.step_size)
            case _ if metric.type in OFFLINE_EVAL_METRIC_TYPES:
                skipped_offline.append(metric.type)
            case _:
                raise AssertionError(f"unsupported eval metric {metric.type!r}")
    if skipped_offline:
        print(f"eval metrics deferred to the offline path: {sorted(skipped_offline)}", flush=True)
    assert ce_kl is not None and ci_l0 is not None, (
        "in-loop eval needs CEandKLLosses + CI_L0 in eval.metrics"
    )
    return EvalConfig(
        batch_size=cfg.eval.batch_size,
        every=cfg.eval.every,
        n_steps=cfg.eval.n_steps,
        rounding_threshold=ce_kl.rounding_threshold,
        ci_alive_threshold=ci_l0.ci_alive_threshold,
        pgd=pgd,
    )


def convert_torch_lm_config(
    torch_cfg: LMExperimentConfig,
    run_name: str,
    run_id: str | None,
    out_dir: Path,
    remat_recon_forwards: bool,
) -> ExperimentConfig:
    first_layer, last_layer, C = _layer_range_and_c(torch_cfg)
    n_sites = 3 * (last_layer - first_layer + 1)

    assert torch_cfg.pd.sampling == "continuous", torch_cfg.pd.sampling
    assert torch_cfg.pd.sigmoid_type == "leaky_hard", torch_cfg.pd.sigmoid_type
    assert torch_cfg.pd.use_delta_component and torch_cfg.pd.tied_weights is None
    assert torch_cfg.runtime.autocast_bf16, "JAX trainer computes in bf16 (autocast analog)"
    assert torch_cfg.pd.faithfulness_warmup_weight_decay == 0.0

    # Vendored and raw-HF Llama specs load the SAME meta-llama weights (the export
    # bridge round-trip verified vendored == HF numerics); both map to our HF loader.
    spec = torch_cfg.target.spec
    match spec:
        case HFWeightsInVendored():
            assert spec.model_class.rsplit(".", 1)[-1] == "VendoredLlama", spec.model_class
        case HFTarget():
            assert spec.model_class == "transformers.LlamaForCausalLM", spec.model_class
        case _:
            raise AssertionError(f"unsupported target spec {spec}")
    assert "Llama-3.1-8B" in spec.model_name, spec.model_name
    if torch_cfg.target.weights_dtype == "float32":
        print(
            "DIVERGENCE: torch config asks for an fp32 frozen target; the JAX trainer keeps "
            "the frozen target in bf16 (measured ~5e-4 nats KL on clean logits — negligible "
            "vs recon KLs, but not bit-parity).",
            flush=True,
        )
    else:
        assert torch_cfg.target.weights_dtype == "bfloat16", torch_cfg.target.weights_dtype

    vu_opt = torch_cfg.pd.components_optimizer
    ci_opt = torch_cfg.pd.ci_fn_optimizer
    _assert_cosine_to_tenth(vu_opt.lr_schedule, "components_optimizer")
    _assert_cosine_to_tenth(ci_opt.lr_schedule, "ci_fn_optimizer")
    _assert_plain_adamw(vu_opt, "components_optimizer")
    _assert_plain_adamw(ci_opt, "ci_fn_optimizer")
    assert vu_opt.grad_clip_norm is not None, "components grad clip is part of the method"
    assert ci_opt.grad_clip_norm is None, "CI-fn grad clip unsupported"

    coeffs, imp_min, source_adam, sites_per_chunk, n_recon_samples = _losses(torch_cfg, n_sites)
    data = _data(torch_cfg)

    cadence = torch_cfg.cadence
    assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None, cadence

    return ExperimentConfig(
        run_name=run_name,
        run_id=run_id,
        out_dir=out_dir,
        seed=torch_cfg.pd.seed,
        steps=torch_cfg.pd.steps,
        target=TargetConfig(
            model_name=spec.model_name, first_layer=first_layer, last_layer=last_layer, C=C
        ),
        data=data,
        losses=coeffs,
        imp_min=imp_min,
        ppgd=source_adam,
        recon=ReconConfig(
            sites_per_chunk=sites_per_chunk,
            n_samples=n_recon_samples,
            remat_forwards=remat_recon_forwards,
        ),
        vu_optimizer=VUOptimizerConfig(
            lr=vu_opt.lr_schedule.start_val, grad_clip_norm=vu_opt.grad_clip_norm
        ),
        ci_optimizer=CIOptimizerConfig(lr=ci_opt.lr_schedule.start_val),
        ci_fn=_ci_arch(torch_cfg, data.seq_len),
        faith_warmup=FaithWarmupConfig(
            steps=torch_cfg.pd.faithfulness_warmup_steps, lr=torch_cfg.pd.faithfulness_warmup_lr
        ),
        cadence=CadenceConfig(
            log_every=cadence.train_log_every,
            save_every=cadence.save_every,
            keep_last=cadence.keep_last_n_checkpoints,
            dense_log_phase=(
                DenseLogPhase(
                    every=cadence.dense_log_phase.every,
                    until_step=cadence.dense_log_phase.until_step,
                )
                if cadence.dense_log_phase is not None
                else None
            ),
        ),
        eval=_eval(torch_cfg),
        wandb=(
            WandbConfig(project=torch_cfg.wandb.project, entity=torch_cfg.wandb.entity)
            if torch_cfg.wandb is not None
            else None
        ),
    )


WRAPPER_KEYS = {"torch_config", "run_id", "run_name", "out_dir", "remat_recon_forwards"}
_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def load_torch_wrapper(wrapper_path: Path) -> tuple[ExperimentConfig, Path, dict[str, Any]]:
    """Parse a wrapper YAML (see module docstring) -> (config, torch yaml path, raw torch
    dict for wandb). The torch path is resolved relative to the wrapper file.

    `run_id` is the canonical `p-<8hex>` identity (torch `generate_run_id` format):
    run dir name + wandb run id, written into the wrapper at authoring time
    (`python -c "import secrets; print('p-' + secrets.token_hex(4))"`) so resumes
    derive the same identity and the byte-compare pins it. Wrappers WITHOUT the key
    predate the scheme (the live C49k run) — drop that arm once it migrates."""
    raw = yaml.safe_load(wrapper_path.read_text())
    assert set(raw) in (WRAPPER_KEYS, WRAPPER_KEYS - {"run_id"}), (
        f"{wrapper_path}: keys must be {sorted(WRAPPER_KEYS)} (run_id optional pre-migration)"
    )
    run_id = raw.get("run_id")
    if run_id is not None:
        assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    torch_yaml_path = (wrapper_path.parent / raw["torch_config"]).resolve()
    assert torch_yaml_path.exists(), f"torch config not found: {torch_yaml_path}"
    torch_raw = yaml.safe_load(torch_yaml_path.read_text())
    torch_cfg = LMExperimentConfig(**torch_raw)
    cfg = convert_torch_lm_config(
        torch_cfg,
        run_name=raw["run_name"],
        run_id=run_id,
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
    )
    return cfg, torch_yaml_path, torch_raw


def load_run_dir_config(run_dir: Path) -> ExperimentConfig:
    """Rebuild a run's `ExperimentConfig` from its pinned config copies (for tools
    that read finished/live run dirs, e.g. the exporter).

    Native runs pin only `config.yaml`. Torch-wrapper runs pin the wrapper as
    `config.yaml` AND the referenced torch yaml beside it as `torch_config.yaml`;
    the wrapper's own (launch-relative) path field is ignored — the pinned copy is
    the source of truth."""
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    if "torch_config" not in raw:
        return load_config(run_dir / "config.yaml")
    assert set(raw) in (WRAPPER_KEYS, WRAPPER_KEYS - {"run_id"}), (
        f"{run_dir}/config.yaml: keys must be {sorted(WRAPPER_KEYS)} (run_id optional pre-migration)"
    )
    torch_raw = yaml.safe_load((run_dir / "torch_config.yaml").read_text())
    return convert_torch_lm_config(
        LMExperimentConfig(**torch_raw),
        run_name=raw["run_name"],
        run_id=raw.get("run_id"),
        out_dir=Path(raw["out_dir"]),
        remat_recon_forwards=raw["remat_recon_forwards"],
    )
