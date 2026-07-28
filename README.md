# Parameter Decomposition

Training tools for parameter decomposition on neural networks: a JAX implementation of
the VPD training loop, generic over vendored LM targets, plus the experiment
composition, the post-decomposition pipeline (harvest → autointerp / clustering), and
CPU toy testbeds. For a compact single-file implementation of the core method, see
[`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Published 4L-Pile decomposition: https://wandb.ai/goodfire/spd/runs/s-55ea3f9b.
  Current JAX reference config: [`param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml), validated by https://wandb.ai/goodfire/param-decomp/runs/p-76082aa1.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## What's here

ONE library, `param-decomp` (importing as `param_decomp`) — enumerated layers, each a
subpackage, importing only downward:

- [`param_decomp/core/`](param_decomp/core/) — the generic VPD trainer engine
  (`run.py::run_decomposition_training`, the `DecomposedModel` protocol, losses,
  adversaries, checkpointing) plus the pydantic config schema it reads directly. A pure
  library: no `main()`, no YAML reading — it takes built objects. Semantics are pinned
  by [`param_decomp/core/SPEC.md`](param_decomp/core/SPEC.md).
- [`param_decomp/targets/`](param_decomp/targets/) — every `DecomposedModel`
  implementation, one slice per architecture (the GLU-transformer family with
  Llama-3.1-8B / Qwen3-8B-Base loaders, `LlamaSimpleMLP`, and the TMS / ResidualMLP
  toys), with per-target parity/golden suites.
- [`param_decomp/vendored_jax/`](param_decomp/vendored_jax/) — bit-parity JAX ports of
  the vendored target architectures.
- [`param_decomp/pretrain/`](param_decomp/pretrain/) — the in-house target-LM
  pretrainer (`python -m param_decomp.pretrain.train`).
- The composition and consumer layers:
  [`experiments/`](param_decomp/experiments/) (per-domain composition roots + the YAML
  authoring schemas), [`harvest/`](param_decomp/harvest/),
  [`autointerp/`](param_decomp/autointerp/), [`clustering/`](param_decomp/clustering/),
  `topology/`, `adapters/`, `infra/`.

Also here: [`papers/`](papers/) — the APD / SPD / VPD paper sources and figures.

The library is deployment-agnostic: it reads no ambient environment for paths — every
entry point takes an explicit `--data-root` (default `./out`), the one root under which
runs, datasets, and caches live. We drive it on our own clusters through a thin private
wrapper that owns submission and storage fit; nothing in this repo depends on it.

## Install

```bash
make install-dev   # library + dev tooling + pre-commit hooks
make install       # library only
```

CPU jax is the base dependency; on a GPU box install a CUDA extra
(`uv sync --extra cuda`, or `--extra cuda13` for driver ≥ r580).

## Run

The toys train on CPU in seconds:

```bash
pd-tms       param_decomp/experiments/tms/configs/tms_5-2.yaml
pd-resid-mlp param_decomp/experiments/resid_mlp/configs/resid_mlp_1l.yaml
```

(The shipped configs log to wandb — `wandb login` first, or delete the `wandb:` block
from the config to run without it.)

An LM decomposition is one self-contained YAML driven through the composition root:

```bash
python -m param_decomp.experiments.lm.run <config.yaml> [--data-root out]
```

Process topology derives from the config — no launch flags: `runtime.dp` ≤
`runtime.gpus_per_node` runs one process over exactly `dp` local devices;
`dp > gpus_per_node` expects one process per node (`dp // gpus_per_node` nodes) brought
up via `jax.distributed`'s own cluster auto-detection. A scheduler submitter is a thin
wrapper around this module invocation.

## Datasets

An LM run's `data:` block names a dataset:

```yaml
data:
  kind: name
  name: pile_neox_tok_512
```

A name resolves to `<data_root>/datasets/<name>` — a directory of pre-tokenized parquet
shards plus a self-describing `meta.json` (`seq_len`, `tokenizer_name`). Populate the
store however you like;
`python -m param_decomp.experiments.lm.prestage_tokenized` writes shards + meta from any
HF text dataset. `data: {kind: dir, dir: /abs/path}` is the ad-hoc escape hatch for
shards outside the store.

## Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the pydantic
schema in core (`param_decomp.core.configs`) and computed by the JAX trainer
(`param_decomp/core/losses.py`, `param_decomp/core/slow_eval.py`).

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
