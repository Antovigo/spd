# Parameter Decomposition

Training tools for parameter decomposition on neural networks: a JAX implementation of
the VPD training loop, generic over vendored LM targets, plus the experiment
composition, the post-decomposition pipeline (harvest → autointerp / clustering), and
CPU toy testbeds. For a compact single-file implementation of the core method, see
[`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Published 4L-Pile decomposition: https://wandb.ai/goodfire/spd/runs/s-55ea3f9b.
  Current JAX reference config: [`param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml).
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

The public package is self-contained. Create the environment from the locked repository:

```bash
uv sync --frozen --no-dev                    # CPU
uv sync --frozen --no-dev --extra cuda       # NVIDIA driver r525-r579
uv sync --frozen --no-dev --extra cuda13     # NVIDIA driver r580+
```

**Blackwell GPUs require `--extra cuda13` and driver r580 or newer.** The CUDA-12 lock
contains cuBLAS older than 13.2, which can silently corrupt execution on Blackwell rather
than merely failing. Use `--extra cuda` only for Ampere, Ada, or Hopper hosts whose driver
cannot load the CUDA-13 wheels.

`make install` is the library-only shorthand; `make install-dev` adds the dev tooling and
pre-commit hooks.

## Run

The toys train on CPU in seconds:

```bash
pd-tms       param_decomp/experiments/tms/configs/tms_5-2.yaml
pd-resid-mlp param_decomp/experiments/resid_mlp/configs/resid_mlp_1l.yaml
```

(The shipped configs log to wandb — `wandb login` first, or delete the `wandb:` block
from the config to run without it.)

An LM decomposition is one self-contained YAML configuration. Start from a shipped
config, make a copy for the experiment, and run it inside the GPU allocation supplied by
your compute system:

```bash
uv run python -m param_decomp.experiments.lm.run <config.yaml> --data-root <data-root>
```

Process topology derives from the config — no launch flags, and the command does not
submit a job or choose a cluster. `runtime.dp` must equal the allocation's total GPU
count; `runtime.gpus_per_node` describes its node shape. `dp` ≤ `gpus_per_node` runs one
process over exactly `dp` local devices; `dp > gpus_per_node` expects one process per
node (`dp // gpus_per_node` nodes) brought up via `jax.distributed`'s own cluster
auto-detection. A scheduler submitter is a thin wrapper around this module invocation.
For example, the current JAX reference config
[`param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml)
sets `dp: 16` and therefore needs 16 GPUs.

## Datasets

LM training reads pre-tokenized parquet shards and never tokenizes or streams source text
at run time. An LM run's `data:` block names a dataset:

```yaml
data:
  kind: name
  name: pile_neox_tok_512
```

A name resolves to `<data-root>/datasets/<name>/` — a directory of pre-tokenized parquet
shards plus a self-describing `meta.json` (`seq_len`, `tokenizer_name`). Prepare an
immutable dataset with:

```bash
uv run python -m param_decomp.experiments.lm.prestage_tokenized \
  --out-dir <data-root>/datasets/<name> \
  --dataset-repo <huggingface-dataset> --subdir <parquet-subdir> \
  --column-name text --tokenizer-name <target-tokenizer> \
  --seq-len <sequence-length> --num-files <count>
```

The tokenizer must already be in the Hugging Face cache, and its identity and sequence
length must match the target. For an ad-hoc local dataset, use the explicit
`data: {kind: dir, dir: <path>}` escape arm.

## Pretrained target weights

For `target.spec.kind: pretrained`, `run_path` names a W&B pretrain run such as
`goodfire/spd/runs/t-9d2b8f02`; it is never a filesystem path. On first use, the library
fetches `model_config.yaml` and `model_step_<N>.safetensors` into
`<data-root>/pretrain_cache/<project>-<run-id>/`. Later runs read that cache without
network access. `python -m param_decomp.pretrain.train` writes the same layout directly
when training a target locally.

The reference run `goodfire/spd/runs/t-9d2b8f02` predates the JAX migration and contains
`model_step_99999.pt`, not safetensors. Its first fetch downloads that file and then stops
with the converter path and destination. Run that converter from git
tag `torch-oracle` in a torch environment; after `model_step_99999.safetensors` is beside
the downloaded files, rerun the same decomposition command.

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
