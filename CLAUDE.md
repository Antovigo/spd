# CLAUDE.md

Guidance for coding agents (and humans) working in this repo.

## Environment

One uv workspace, one `.venv` at the repo root:

```bash
uv sync                      # core + lab + dev tooling (the lab rides the root dev group)
source .venv/bin/activate
```

`make install-dev` is `uv sync` plus pre-commit hooks. A `.env` with WandB credentials
(see `.env.example`) is only needed for runs that log to wandb.

The repo is JAX; the one torch island is `nano_param_decomp/` (standalone, excluded from
type-checking, imported by nothing).

## Repo map

Two distributions, deliberately split — **core is a pure trainer library; lab is
composition / IO / CLI / experiment assembly**:

- **`param-decomp`** (root distribution):
  - `param_decomp/` — the core VPD trainer engine: `run.py`
    (`run_decomposition_training`), `model.py` (`DecomposedModel`), the loss terms,
    the pydantic config schema (`configs.py` / `base_config.py` / `schedule.py`), and
    the built-run bundle (`built_run.py`). A pure library — no `main()`, no YAML
    reading; it takes built objects.
    `param_decomp/configs/` ships self-contained run YAMLs as reference recipes.
  - `pretrain/` — the in-house target-LM pretrainer (`python -m pretrain.train`).
  - `vendored_jax/` — bit-parity JAX ports of the target architectures.
- **`param-decomp-lab`** (`param_decomp_lab/`) — the harness: per-domain composition
  roots (`experiments/{lm,tms,resid_mlp}/run.py` read a run YAML, build the target /
  data / `ExperimentConfig`, call the engine), launchers, run loading
  (`experiments/lm/load_run.py`), the post-decomposition pipeline
  (`harvest/`, `autointerp/`, `clustering/`, `investigate/`, `postprocess/`), and
  `infra/` (settings, slurm, wandb). Depends on core; the reverse edge
  (`param_decomp → param_decomp_lab`) is forbidden.

Also here: `nano_param_decomp/` — a single-file torch reference implementation of the
method for paper readers — and `papers/` — the APD/SPD paper sources and figures.
Module-level CLAUDE.md files under `param_decomp_lab/` carry the per-pipeline detail.

## CLI entry points

All console scripts live in `param_decomp_lab/pyproject.toml`; the root distribution
declares none (its trainers run as modules, which is what the launchers sbatch).

| Command | Purpose |
|---|---|
| `pd-lm` | Launch an LM decomposition run (SLURM or inline, per `runtime.launch`) |
| `pd-pretrain` | Launch a target-LM pretraining run (SLURM or inline, per `dp`) |
| `pd-tms` / `pd-resid-mlp` | Toy-domain decompositions (CPU, in-process, synchronous) |
| `pd-harvest` | Component-statistics harvest over training data (SLURM) |
| `pd-autointerp` | LLM-based component interpretation over harvest output (SLURM) |
| `pd-intruder` | Label-free intruder eval of decomposition quality (SLURM) |
| `pd-clustering` / `pd-cluster-merge` / `pd-cluster-distances` | Coactivation clustering: ensemble / merge / consensus distances |
| `pd-investigate` | Agent-driven investigation of a decomposition (SLURM) |
| `pd-postprocess` | The whole post-decomposition pipeline with SLURM dependency wiring |

## How a run works

A run is one self-contained YAML (the `param_decomp_lab.experiments.config.
ExperimentConfig` schema over the core `param_decomp.configs` pieces). `pd-lm
<config.yaml>` is config-driven: the launch mode is the config's `runtime.launch`,
and `runtime.dp` (also required) declares the world size — both are authored
decisions, never inferred from ambient env.

- `launch: inline` — mint a run id, pin the launch config, and run the trainer here,
  in the launching process's allocation (no SLURM submission); the trainer asserts it
  finds exactly `dp` local devices. `dp: 1` is the single-device smoke/debug run;
  larger `dp` fits a run inside an external scheduler's own job.
- `launch: slurm` (`dp` a multiple of 8) — mint a run id, snapshot the working tree to
  `refs/runs/snapshot/<id>`, stage the run dir (pinned `launch_config.yaml` + `.env`),
  and sbatch `python -m param_decomp_lab.experiments.lm.run` across `dp // 8` nodes,
  one trainer process per node owning all 8 local GPUs; each node builds its own
  workspace and CUDA venv at job start. SIGTERM → save → requeue → resume.

Artifacts land under `PARAM_DECOMP_OUT_DIR/runs/<run_id>/` (`launch_config.yaml`,
`ckpts/<step>/`, `metrics.jsonl`, then per-stage `harvest/`, `autointerp/`, …).
`PARAM_DECOMP_OUT_DIR` defaults to `./out`; set the env var to point at real storage.

## Development commands

| Command | Purpose |
|---|---|
| `make check` | ruff format/lint + basedpyright |
| `make type` | basedpyright over the whole workspace |
| `make format` | ruff lint + format |
| `make test` | tests excluding slow |
| `make test-all` | all tests (adds a simulated-multidevice pass) |

Single test: `python -m pytest path/to/test_file.py::test_name`.

## Coding guidelines

This is research code. Prioritize simplicity and fail-fast over defensive programming.

- **Fail fast.** If you hold an invariant, `assert` it. Don't write
  `if everything_is_ok: continue_happy_path()` — assert and proceed. No `try/except`
  for control flow; if the program isn't working as it should, it shouldn't be running.
- **Enumerate states.** Prefer `match` over `if/elif` chains for dispatch on a tag or
  kind; unknown cases die loudly, never fall through to a degraded mode.
- **Encode invariants in types.** Jointly-varying fields go in one optional bundle, not
  two independently optional fields. Avoid `| None` unless absence is meaningful. Typed
  pydantic configs, not bare dicts, for heterogeneous data. PEP 604 unions, lowercase
  generics. Type-checker is **basedpyright** — keep it green.
- **No legacy shims.** No old-format fallbacks; delete unused code; if an argument is
  always the same value, inline it. (Config schema fields and CLI flags are user-facing
  surface — their users may live outside this tree.)
- **Tensors:** einops for clarity, jaxtyping for shape documentation, assert shapes
  liberally.
- **Comments carry what the code can't** — a constraint, invariant, or gotcha. Never
  narrate changes ("now uses X"). Docstrings default to a single line; skip them when
  name + types say everything.
