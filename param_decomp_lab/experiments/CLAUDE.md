# `param_decomp_lab/experiments/`

Experiment glue, torch-free. Training is JAX through the generic core engine
(`jax_single_pool.run.run_decomposition_training`). LM runs go to SLURM via `jsp-train` /
`pd-jax-lm`; the toy domains (TMS, ResidMLP) run on CPU in-process via `pd-tms` /
`pd-resid-mlp`. The torch `build_target` bridge + the `pretrain/` dir were DELETED with the
rest of torch: autointerp/clustering read a run's target topology from
`jax_single_pool.load_run.run_metadata` (config + pretrain cache, no checkpoint restore) —
see `param_decomp_lab/adapters/jax_pd.py`.

## Toy domains (TMS, ResidMLP)

The TMS and ResidualMLP toys are LAB experiments that call the core engine as a library
(the core itself has zero toy-specific code). Each `experiments/{tms,resid_mlp}/` carries:

- `model.py` — the JAX `DecomposedModel` (sites, pure fns, MSE `recon_loss_fn`), the frozen
  target (`eqx.Module`), from-scratch in-process pretrain (`pretrain_*_target`), the
  ground-truth identity-CI eval (`identity_ci_error` + the single-feature probe), and the
  lab `*TargetConfig` dataclass carried on `ExperimentConfig.target` (satisfies the core
  `config.TargetSites` protocol).
- `run.py` — the `pd-tms` / `pd-resid-mlp` CLI: builds the `ExperimentConfig` from the
  canonical schema via the public shared helpers
  (`config.convert_shared_algorithm_config` / `run_instance` / `layerwise_mlp_ci_arch`),
  pretrains + builds the target, and calls `run_decomposition_training` with a synthetic
  `sample_batch` + an `identity_ci_error` `eval_fn`. CPU, synchronous, no SLURM.
- `configs/*.yaml` — the canonical `param_decomp_config.{tms,resid_mlp}` schema (TMS: 5-2 /
  40-10 / the `-id` deeper variants; ResidMLP: 1l/2l/3l + the global-CI variant).

TMS deeper variant (`n_hidden_layers>0`, the `-id` configs) + the ResidMLP `global` CI arch
(`fn_type=global_shared_mlp`) are restored and wired end-to-end (the global arch dispatches
through the core `init_train_state` via `config.toy_ci_arch`). Toy harvest / autointerp /
clustering is NOT yet wired (`load_run` is LM-only) — the remaining Phase-3 bucket.

## Layout

The `ExperimentConfig[T,D]` generic + `EvalConfig` + `WandbConfig` +
`ResumeProvenance` live in `param_decomp_config/experiment.py`; the LM schema
(`LMExperimentConfig`, `LMTargetConfig`, `LMDataConfig`, the `target.spec` union) in
`param_decomp_config/lm.py`; the toy schemas in `param_decomp_config/{tms,resid_mlp}.py`.

```
experiments/
├── utils.py                 # EXPERIMENT_CONFIG_FILENAME
├── lm/
│   ├── jax_launch.py        # pd-jax-lm: snapshot + shared-FS workspace + sbatch
│   ├── data.py              # tokenize_and_concatenate (offline helper for prestage)
│   └── prestage_tokenized.py  # HF text -> int32 parquet shards for the JAX trainer
├── tms/                     # pd-tms (CPU): model.py + run.py + configs/ + test_tms.py
└── resid_mlp/               # pd-resid-mlp (CPU): model.py + run.py + configs/ + test
```

## LM `target.spec`

The LM target is a discriminated union on `kind`:

```yaml
target:
  spec:
    kind: hf                            # HuggingFace model
    model_class: transformers.GPT2LMHeadModel
    model_name: openai-community/gpt2
  output_extract: logits

# or
target:
  spec:
    kind: pretrained                    # in-repo lab-pretrained model
    model_class: param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: goodfire/spd/runs/<run_id>
  output_extract: 0

# or
target:
  spec:
    kind: hf_weights_in_vendored        # HF weights loaded into a vendored, componentizable arch
    model_class: param_decomp_lab.experiments.lm.vendored.llama_3_1.model.VendoredLlama
    model_name: meta-llama/Llama-3.1-8B
  output_extract: 0
```

`output_extract` (default `"logits"`) is the key/index used to pull the prediction
tensor out of the model's forward output. The `model_class` strings are NOT imported by
the JAX trainer — `jax_single_pool.config` only asserts the class-name suffix and routes
to its own vendored JAX arch (`pretrained` LlamaSimpleMLP -> the pretrain-cache loader,
`hf_weights_in_vendored` Llama -> `vendored_jax`). They reference the deleted torch
`pretrain/` module only as identifiers.

The path schemas (`topology/path_schemas.py`) cover the GPT-2 and `LlamaSimple*` archs —
so `JaxPDAdapter`'s layer-description path is exercised by `kind: pretrained` runs (the
pile `LlamaSimpleMLP` decompositions), the production target.

## `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
