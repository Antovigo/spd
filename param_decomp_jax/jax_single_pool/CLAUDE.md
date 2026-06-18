# jax_single_pool — agent notes

Single-pool VPD trainer in JAX, **generic over vendored LM targets**. The semantics
source of truth is `SPEC.md` (normative pseudocode + numbered invariants, grounded in
the stable torch `param_decomp` impl). See `README.md` for the file map.

Open items: persistent-source scopes `c`/`nsc` and sigmoid parameterization are
deliberately refused. The hidden-acts seam is now BUILT (SPEC S31 amended 2026-06-16):
`CIHiddenActsReconLoss` / `StochasticHiddenActsReconLoss` are standalone offline eval
metrics (`hidden_acts_eval.py`, via `jsp-slow-eval`) over a fifth model fn
`masked_site_outputs` — NOT recon-grid training terms (the recon loss stays
KL-on-final-logits). `sc` and `bsc` are supported (`bsc` is batch-sharded:
an independent source per batch element and position, no cross-replica sync — SPEC
S16/D1). Persistent `start_frac>0` is now implemented (SPEC S32, `term_active`
`where`-gating); SPEC S24's two torch-parity quirks (PPGD warmup route-all, fresh-PGD
single routing draw) are pinned pending a team decision. CI-fn numerics are unified
with the torch oracle (#624/#625/#730 resolved): GELU is exact-erf
(`approximate=False`) and RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`).

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`lm.py` defines `DecomposedModel` — ordered `sites` + `leading_axes` + five pure fns
(`clean_output`, `site_inputs`, `masked_output`, `weight_deltas`) plus a pluggable
`recon_loss_fn` (default `kl_per_position`), flat site-name-keyed dicts at the boundary,
frozen pytree always a runtime arg (never a jit closure constant — an 8B target becomes a
multi-GB HLO constant). The activation waist is GENERIC `[*leading, d]` (masks/CI
`[*leading, C]`), `leading = (batch,) + named position axes`: masking / routing / sources
/ imp-min all read an opaque `leading = residual.shape[:-1]`; reductions are
`math.prod(shape[:-1])` / `axis=tuple(range(ndim-1))`. CI is independent over every leading
axis (no per-axis CI semantics, only axis NAMES — see AXIS_SEMANTICS_DESIGN.md).
`DecomposedModel.leading_axes` names the position axes (`("sequence",)` for LM, `()` for
TMS); `CIFn.expects_axes` mirrors it, and `init_train_state` asserts they're equal (early
fail) so the CI fn stays per-domain (RoPE over `sequence`) without the core adapting. The
three EDGES are generic so non-LM (bio-style) targets fit (#828): the model INPUT
(`prefix_residual_fn(prefix, inputs)` in `run.py` takes `Any` — tokens for an LM, a dict
for bio), the model OUTPUT (`clean_output`/`masked_output` return `Any` — logits, a tuple
of heads, coords; field NAMES stay `*_logits` pending a deferred rename), and the recon
comparison (`recon_loss_fn(clean_output, masked_output) -> scalar`, default
`kl_per_position` so the LM path is byte-identical). The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_forward`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over a static tuple of recon loss TERMS (S10′ — the
torch loss-class cartesian product factored as chunking × routing × mask-source
strategy: a chunking helper (`one_chunk`/`per_site`/`into_groups`) feeds the single
`make_plan` constructor, built from the shared configs by `recon.build_recon_terms`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules) and `adversary.py` (persistent
vs fresh source machinery — semantically distinct adversaries sharing only
`source_masks`); `ci_fn.py` the shared CI transformer; `llama8b.py` + `llama8b_sharding.py` the first target. There is ONE
recon semantics: masks thread through the suffix forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (llama8b) vs `LlamaSimpleMLPTargetConfig` in
`config.py` (which also reads the canonical `param_decomp_config` schema DIRECTLY —
`build_experiment_config`/`load_config` — routing `kind: pretrained` specs + `h.*`
wildcards), target build in `run.py::main`. The slow plot metrics are computed
NATIVELY in JAX via `jsp-slow-eval` (`slow_eval.py`) — no torch export round-trip
(the torch offline-eval bridge `jsp-export` / `pd-offline-eval` was retired).

**The toys (TMS, ResidMLP) live in the lab, not the core.** The core trainer carries ZERO
toy-specific code (CI-fn arches are the one allowed exception — see `ci_fn_mlp.py`). The
generic engine is `run.py::run_decomposition_training(cfg, raw_cfg, lm, frozen,
sample_batch, eval_fn, eval_every, perf_tokens_per_step, mesh)` — the ONE train loop every
target runs through (init/restore/finetune/faith-warmup via `_init_or_restore_state`, the
recon-grid step factory, orbax checkpointing, schedules, SIGTERM-save). A target injects
exactly three seams: the data source (`sample_batch(step) -> residual`), the eval metric
(`eval_fn(state, now_step) -> dict`, run every `eval_every`), and (for the LM) the perf
token count. `run.py::train` is the thin LM caller (parquet `sample_batch` + the
CEandKL/CI-L0/PGD/attn-patterns `eval_fn` in `_make_lm_eval_fn`); `jsp-train` is LM-ONLY
(`config.build_from_schema` validates `LMExperimentConfig`; `main`'s `match cfg.target`
covers only `TargetConfig` / `LlamaSimpleMLPTargetConfig`). `cfg.target` is typed by the
`config.TargetSites` protocol (just `.sites`), `cfg.data` is `DataConfig | None` (None for a
toy run). The shared algorithm-config conversion is public for the lab toys to reuse:
`config.convert_shared_algorithm_config` / `run_instance` / `layerwise_mlp_ci_arch` (+
`SharedAlgorithmConfig`).

The TMS + ResidMLP targets now live under `param_decomp_lab/experiments/{tms,resid_mlp}/`
(`model.py` = the JAX `DecomposedModel` + frozen target + in-process pretrain + identity-CI
eval; `run.py` = the `pd-tms` / `pd-resid-mlp` CPU CLI that builds the `ExperimentConfig`
from the canonical schema and calls `run_decomposition_training`). They are positionless
(`leading_axes=()`) and use the layerwise per-site MLP CI fn. `ci_fn_mlp.py` (the second
CI-fn arch, the allowed exception) stays in the core: `LayerwiseMLPCIFn` (`fn_type=mlp`,
`expects_axes=()`, one independent MLP per site mapping `site_input [B,d_in] -> [B,C]`) plus
the new `GlobalMLPCIFn` (`fn_type=global_shared_mlp`, one shared MLP over all sites jointly,
concat/split in canonical site order). `run_state.init_train_state` dispatches CI-fn
construction on `cfg.ci_fn` (`CIArch` transformer / `MLPCIArch` layerwise / `GlobalMLPCIArch`
global) and uses replicated (not C-sharded) V/U + CI for the tiny toys; `config.CIFnArch`
admits all three and `config.toy_ci_arch` builds the layerwise / global arch from the toy
ci_config (validated end-to-end on CPU via `pd-resid-mlp`). Harvest / slow-eval / export over
the toys are NOT wired (`load_run.build_target` / `run_metadata` are LM-only).

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path forward (`clean_output`), never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph).
- **S13/S15**: source updates go through the persistent Adam AND project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- **S14**: the final source ascent reuses the main backward's source-grad
  (pre-update θ), unscaled by the ppgd coeff. No extra forward.
- **N1**: fp32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0; this was audit finding A7).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in `CIFn.__call__`.
- **S10/S11**: chunking is sequential `sites_per_chunk` groups in canonical site
  order; routing is uniform-k over the chunk's sites only.

## Validation stack (run all before claiming correctness)

1. `pytest jax_single_pool/tests/` — at the default device count AND
   `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.
2. `tests/equivalence/` — fixture-driven JAX-vs-frozen-golden per-term numeric
   equivalence (fp32, no RNG, zeroed attn). The torch references are FROZEN committed
   goldens (`torch_reference.json`, `simple_mlp_equivalence/*.npz`,
   `tools/export_fixtures/*`); the torch generators/verifier that produced them are
   deleted so `param_decomp_jax` imports no torch (push-1). Regenerate goldens only when
   the MATH changes: redraw fixtures JAX-side with `gen_fixtures.py`, then check out the
   `torch-oracle` git tag in a torch-venv worktree and run that revision's
   `torch_reference.py` / `gen_torch_fixtures.py` / `gen_export_fixture.py`, copying the
   emitted goldens back here.
3. `experiments/invariance_check.py` at 4 sim devices — trajectory invariant to
   device count up to float reassociation (SPEC D4).

`basedpyright jax_single_pool/` must be clean (run via `make check-jax` in the JAX
distribution's own venv); the package stays out of the main repo `[tool.pyright]` include
and is resolved there only via `extraPaths` for the lab consumers that import it.

## The training pipeline (`run.py`)

`jsp-train <config.yaml>` is the composition root and the only I/O layer; the step
stays pure. Data is a pre-tokenized parquet artifact under
`$DATA_MOUNT/artifacts/mechanisms/param-decomp/datasets/` (`fineweb_llama_tok_2048`
for Llama-8B, `pile_neox_tok_512` for `LlamaSimpleMLP`) — NEVER stream/tokenize from
HF at run time (the 80-rank thunderherd lesson). The batch schedule is a pure
function of `(seed, step)` (O(1) resume, no replay); checkpoints are orbax sharded
saves (no on-loop full-gather); SIGTERM → save → SLURM requeue → resume from latest.
Resume with a changed config is refused (byte-compare). Smokes before a long run
MUST exercise save AND resume at the production per-rank shape.

A run config is ONE self-contained yaml: the `param_decomp_config` experiment schema
(`pd`/`data`/`eval`/`cadence`/`runtime`/`target`/`wandb`) plus the run-instance fields
the schema now also carries — top-level `run_name`/`run_id`/`out_dir`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are absent in a hand-authored config; `pd-jax-lm` mints + stamps them.

**Fine-tune from a parent checkpoint** (`resume_provenance`, SPEC S33, LM-only). A fresh
run can initialize its trained decomposition (V/U + ci_fn) from a PARENT run's checkpoint
and continue under a DIFFERENT config (changed LR / coeffs / eps / seq / batch / steps —
NOT changed C / sites / ci-fn arch). Add to the config:

```yaml
resume_provenance:
  # ABSOLUTE path — jsp-train runs with cwd = <workspace>/param_decomp_jax, so a
  # relative path would resolve under the workspace, not the output runs dir.
  parent_run_dir: /mnt/data/artifacts/mechanisms/param-decomp/runs/p-bd3cd4d4
  parent_step: 175000
```

On the FIRST entry (own `ckpts/` empty) the trainer loads `parent_run_dir/ckpts/175000`
onto the fresh reference and keeps ONLY the components + ci_fn; the optimizer states,
persistent sources, and `step` are FRESH (`step = 0`, no faith warmup) so the new LR /
p-anneal schedule recomputes over the new `cfg.steps` from 0. A subsequent SLURM requeue
(own `ckpts/` now non-empty) resumes from the run's own dir and ignores provenance.
`run.py::assert_finetune_structural_compat` reads the parent's pinned `config.yaml` and
asserts matching sites (names + C) + ci-fn arch before the restore. Provenance flows into
`config.yaml` + `wandb.config`. Launch as usual via `pd-jax-lm <config.yaml> --nodes N`.

**Launch via `pd-jax-lm <config.yaml> --nodes N`** (lab-side, torch venv): mints the
`p-` run id, snapshots the tree to `refs/runs/snapshot/<id>`, materializes an
immutable shared-FS workspace (clone + both venvs) at
`$PARAM_DECOMP_OUT_DIR/workspaces/<id>`, stamps the id (+ out_dir / wandb group / tags)
into the workspace's single config yaml, and sbatches. Requeues re-enter the workspace,
never the live checkout. `--run_id` resubmits an existing workspace. Don't hand-write
sbatch files.

`main` enables JAX's persistent compilation cache
(`_enable_persistent_compilation_cache`) at `$PARAM_DECOMP_OUT_DIR/xla_compilation_cache`
— a SIBLING of `runs/` (derived from `out_dir.parent`), shared across all runs and all
8N ranks, NOT per-run. The ~24-min chunkwise-step compile is keyed by HLO + backend +
topology + jax/xla version, so a requeue/resume or a fresh run at the same config+topology
loads the executable from disk in seconds. Set after `init_distributed` (the write gate
reads the distributed state) and before the first compile; threshold 60s
(`jax_persistent_cache_min_compile_time_secs`) so only the big compiles cache. Multi-host
safe on jax 0.10.1: jax gates the cache WRITE on `process_id == 0` (`compiler.py` — "Only
write cache entries from the first process … contention for writes on some filesystems"),
so all ranks read but only rank 0 writes — no shared-FS race. Requires the cache dir on a
shared FS, which `$PARAM_DECOMP_OUT_DIR` already is.

## Gotchas

- **`shard_batch` topology** (`sharding.py`): uses `make_array_from_process_local_data`
  so it's correct for BOTH single-process-many-devices and multi-process-1-device.
  Do NOT revert to the per-`process_index()`-slice idiom — it silently replicates one
  slice on single-process multi-device CPU.
- **`vendored_jax` is part of this distribution** (moved from `jax_spike/`); no
  `sys.path` hacks anywhere. If an import fails, the install is broken — fix the env
  (`uv pip install -e .`), don't add a path shim. (The old `jax_spike` stage scripts
  that imported it by cwd are superseded by this package.)
- **Bench schedules**: `llama8b_real.py` anneals over `--total_steps` (default 100k),
  not the benched `--steps` — short benches measure start-of-training semantics.
