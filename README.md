# Parameter Decomposition

Training tools for parameter decomposition on neural networks: a JAX implementation of
the Variational Parameter Decomposition (VPD) training loop, generic over vendored LM
targets, plus the experiment harness that drives it. For a compact single-file
implementation of the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Canonical 4L-pile run: `goodfire/spd/runs/s-55ea3f9b`.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## What's here

Two Python distributions in one uv workspace:

- **`param-decomp`** (root) — the core library, importing as `param_decomp`:
  - [`param_decomp/`](param_decomp/) — the JAX trainer core: the four-term VPD loss as
    one `jax.jit` step, GSPMD-sharded, generic over targets. See its
    [README](param_decomp/README.md) for the module map and design notes.
    [`param_decomp/configs/`](param_decomp/configs/) ships self-contained run YAMLs as
    reference recipes.
  - [`pretrain/`](pretrain/) — pretraining for the in-house target LMs
    (`python -m pretrain.train`).
  - [`vendored_jax/`](vendored_jax/) — bit-parity JAX ports of the vendored target archs.
- **`param-decomp-lab`** — experiments, postprocessing, and CLI tooling, importing as
  [`param_decomp_lab`](param_decomp_lab/): the composition roots that turn a run YAML
  into a built run, the SLURM launchers, and the post-decomposition pipeline
  (harvest → autointerp / intruder / clustering).

Also here: [`papers/`](papers/) — the APD/SPD paper sources and figures.

## Install

```bash
uv sync            # core + lab + dev tooling into one .venv (make install-dev adds pre-commit hooks)
make install       # core package only, no dev dependencies
```

## Run experiments

The lab installs the `pd-*` CLIs. Decomposition runs are one self-contained YAML
(the `param_decomp_lab.experiments.config.ExperimentConfig` schema over the core
`param_decomp.configs` pieces); reference LM configs live in
[`param_decomp_lab/experiments/lm/`](param_decomp_lab/experiments/lm/), toy configs in
`param_decomp_lab/experiments/{tms,resid_mlp}/configs/`.

```bash
pd-lm param_decomp_lab/experiments/lm/jose-ish.yaml            # LM run via SLURM (or --local)
pd-tms param_decomp_lab/experiments/tms/configs/tms_5-2.yaml   # toy runs, CPU, in-process
```

| CLI | What it runs |
|---|---|
| `pd-lm` | LM decomposition: snapshot + sbatch `python -m param_decomp_lab.experiments.lm.run` (`--local` runs inline) |
| `pd-tms` / `pd-resid-mlp` | toy-domain decompositions (CPU, in-process pretrain → decompose) |
| `pd-pretrain` | target-LM pretraining launcher for `python -m pretrain.train` |
| `pd-harvest` | component-statistics harvest over training data (SLURM array + merge) |
| `pd-intruder` | label-free intruder eval of decomposition quality |
| `pd-autointerp` | LLM-based component interpretation over harvest output |
| `pd-clustering` / `pd-cluster-merge` / `pd-cluster-distances` | ensemble coactivation clustering: pipeline / single merge / consensus |
| `pd-investigate` | agent-driven investigation of a decomposition |
| `pd-postprocess` | the whole post-decomposition pipeline with SLURM dependency wiring |

The trainer engine (`param_decomp.run.run_decomposition_training`) is a pure library —
no `main()`, no YAML. Each domain's composition root lives lab-side
(`param_decomp_lab/experiments/`).

## Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the torch-free
pydantic schema (`param_decomp.configs`) and computed by the JAX trainer
(`param_decomp/losses.py`, `eval.py`, `slow_eval.py`).

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
