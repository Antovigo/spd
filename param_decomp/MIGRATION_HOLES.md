# Migration holes — known gaps in the JAX-primary stack

Living list of things deliberately left un-migrated / un-reimplemented during the
torch→JAX migration, so they're tracked rather than silently lost. Append as found.

## Eval metrics — all slow/plot metrics now run IN-LOOP

Slow/plot eval is IN-LOOP ONLY (SPEC S28; no offline CLI — `pd-slow-eval` and
`run_offline_slow_eval` were removed). `slow_eval.py` is a pure library the in-loop tier
calls. The base plot metrics (`CIHistograms` / `ComponentActivationDensity` /
`CIMeanPerComponent`), the hidden-acts scalars, and `UVPlots` / `PermutedCIPlots` /
`IdentityCIError` all run on cadence `eval.slow_every`. The permutation/UV/identity metrics
are config-gated off the run's `eval.metrics` (re-validated from `config.yaml` via
`resolve_permutation_metrics`), operate on the batch-mean `(position, C)` CI matrix from the
real LM eval batches (`make_position_ci_step` / `accumulate_position_ci`), and:

- **`PermutedCIPlots`** — per-site `(position, C)` CI heatmaps with columns permuted toward
  each site's target shape (identity via scipy `linear_sum_assignment`, or dense by column
  mass). Lower-leaky + upper-leaky views (`render_permutation_figures` →
  `figures/causal_importances{,_upper_leaky}`).
- **`UVPlots`** — per-site V/U heatmaps with the component axis reordered by the same
  permutation (`figures/uv_matrices`). A config-gated figure metric usable for ANY decomp:
  cheap for the toys (small on-host V/U, `toy_uv_eval`), a naive V/U host gather for the LM
  in-loop tier that breaks at production C by design (per Oli).
- **`IdentityCIError`** — the discrete CI-vs-target distance, generalizing the toy
  `tms`/`resid_mlp` `identity_ci_error` (`identity_ci_error` / `dense_ci_error` in
  `slow_eval.py`), keyed `IdentityCIError[/<site>]`.

The config classes live in torch-free `param_decomp_config.eval_metrics`. (`AutointerpLabels`
was a dead pre-migration entry — registered, zero configs, no driver — and is not in the
config union.)

## Deferred to a follow-up push (not in the current push)

- **#10 torch→jax run adapter** — loading OLD torch PD runs (`model_*.pth`) into the JAX
  consumers (autointerp/intruder/app). The torch-run-loading surface (`adapters/pd`,
  `component_model_io`, vendored Llama) was dropped (#872); re-add as a JAX-native loader
  off `open_jax_run`/orbax. Until then autointerp/intruder work on JAX runs only. The
  `DecompositionAdapter` ABC seam was also collapsed to the single `PDAdapter` impl (the
  multi-method comparison surface is off the table) — re-adding a second method (torch
  loader, CLT/transcoder, …) rebuilds that ABC + the `DecompositionMethod` discriminator.
- **App** (`param_decomp_lab/app/`) — temporarily removed (#868); slated for a JAX-native
  re-add. `pd-investigate` subprocess-launches it, so it's broken until the app returns.

## `pretrain/` is DELETED — reimplement in JAX when next needed

`param_decomp_lab/experiments/lm/pretrain/` (the in-house target model defs + training
loop + `pd-pretrain` CLI) was **deleted** with the rest of torch — the repo is now
torch-free (zero `import torch`). When we next need to pretrain a target, write it in JAX
from scratch. This costs nothing in the meantime: the base models we currently decompose
(Llama-3.1-8B, the pile `LlamaSimpleMLP` `t-9d2b8f02`) are already pretrained on disk, and
the trainer loads them through its OWN torch-free loaders
(`llama_simple_mlp.load_target_from_pretrain_cache` / `load_prefix_from_pretrain_cache`)
reading the on-disk weight cache — never through `pretrain/` code. The one-off torch
checkpoint→safetensors converter (`tools/convert_llama_simple_mlp_checkpoint.py`) was
deleted too; the existing caches already hold their converted safetensors.

## Review blind spots — dropped/changed with no prior tracking note

Surfaced by the `main`-vs-`feature/jax` recursive taxonomy (`MIGRATION_TAXONOMY.md`,
repo root). All are deliberate consequences of the torch shed; listed here so they're
tracked rather than silently absent. None block the squash.

Intentional drops (confirm scope, no re-add planned this push):

- **Component types: `EmbeddingComponents`, Radford-`Conv1D`, `Identity`** — the JAX stack
  decomposes `nn.Linear`-equivalents only (Llama MLP). The factory's conv1d/identity/
  embedding dispatch has no JAX path; folds into the deferred eqx-auto-decompose (#11).
- **`identity_insertion` / `identity_decomposition_targets`** — unsupported; the
  `identity_decomposition_targets` config field is REMOVED from `PDConfig`, so `extra=forbid`
  rejects any config that sets it (no longer an inert-but-present field).
- **LM-path component weight tying (`tie_component_weights`)** — NOT a lost capability;
  **obviated by the JAX design.** Torch tied two SEPARATE component decompositions
  post-init (`tgt.U/V = src.V.T/U.T`) *because* it decomposed tied target modules as
  independent sites. JAX carries the target's native tying inside the vendored arch
  (`wte`↔`lm_head`) and decomposes each UNIQUE matrix once as a single site — so there is
  nothing to re-tie. The `tied_weights` config field is REMOVED from `PDConfig`; `extra=forbid`
  rejects any config that sets it.
- **`ci_sigmoids` registry** — only `leaky_hard` (split lower/upper) survives; the
  `sigmoid_type` config field is REMOVED from `PDConfig` (the CI fns hardcode lower/upper
  leaky-hard), so `normal` / `hard` / `swish_hard` can no longer be requested — `extra=forbid`
  rejects the key.
- **`mlp_scalar` CI-fn arch** — torch's scalar `get_component_acts(x)=x@V` couples CI-fn
  input to trained components; doesn't fit the generic `ci_fn(site_inputs)` waist. Replaced
  by the vector-input `LayerwiseMLPCIFn`. (Rationale in `CLAUDE.md`.)
- **`PersistentPGDReconSubsetLoss`** — dropped from the config union; a future composition
  per `LOSS_PARITY_DESIGN.md`.
- **CLT/transcoder adapters + `_vendor` models (#863)** — comparison-method tooling; can't
  harvest CLT/transcoder runs until re-added. Re-adding any second method rebuilds the
  `DecompositionAdapter` ABC + `DecompositionMethod` discriminator collapsed in this push.
- **`editing/` + `generate_token_divergence.py`** — model-editing + token-divergence viz
  (also noted in `TRANSITION.md §1/§6`).
- **toy-models target-CI pattern framework** — `DenseCIPattern` / `TargetCISolution` /
  fnmatch expansion / greedy permutation gone; only per-target `identity_ci_error` survives.

Benign mechanism changes (no behavior risk):

- **`component_acts` cache-mode coupling removed** — autointerp/harvest recompute `x@V`
  inline; no per-component pre/post-detach cache.
- **`wandb.finish()` never called** — relies on process exit.
- **torch DDP `with_distributed_cleanup` / `ensure_cached_and_call`** — os._exit SIGABRT
  guard + download-once-per-node helper have no JAX analog.
- **imp-min `world_size` scaling** — explicit `log2(1+sum·world_size)` replaced by GSPMD's
  in-graph global sum (see `imp_min_world_size_noop` memory).

## Imp-min token-count reparameterization (deferred, Oli)

The imp-min entropy term carries a `log2(batch·seq)` coupling — a per-token-batch
artifact. A token-count-invariant reparameterization would remove the (currently
ignored) batch sensitivity. See `project_impmin_scaling` memory.

## PPGD source sigmoid parameterization (REMOVED — how to re-add)

The torch PersistentPGD adversary could read the adversarial mask from its sources two
ways: **clamp** (sources ARE the [0,1] mask, projected via clamp after each ascent — the
only implemented JAX path, SPEC S13/S15) or **sigmoid** (sources are unconstrained,
`mask = sigmoid(source)`). The `use_sigmoid_parameterization` config option for the sigmoid
form was never ported and has now been **removed** from the schema
(`PersistentPGDReconLossConfig`); a `model_validator(mode="before")` strips it from stored
run configs (all carried `false`) so they still load, and rejects `true`.

To re-add it (est. ~1 day), all in `adversary.py`:
1. init persistent sources **unconstrained** (not `U[0,1]`) — e.g. zeros (`sigmoid(0)=0.5`).
2. materialize `mask = jax.nn.sigmoid(source)` wherever sources become masks (today the
   source is used directly / clamped).
3. drop the `[0,1]` clamp-project in `sources_adam_ascend_project` (the sigmoid bounds the
   mask; ascent now moves unconstrained params, grad flows through the sigmoid).
4. handle the fused final source step (SPEC S14) + the trailing weight-delta channel under
   the new mapping; amend SPEC S13–S15 with the sigmoid variant; add a parity test.
Re-introduce the config field (or a `source_parameterization: clamp|sigmoid` enum) at that
point and delete the strip-on-load shim once no stored config carries the old field.
