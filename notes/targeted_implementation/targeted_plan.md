# Targeted Parameter Decomposition (tPD) — Conceptual Specification

An implementation-agnostic description of everything required to add targeted decomposition to a
parameter-decomposition (PD/SPD) codebase. It describes **what must hold**, not how a particular
codebase wires it. Given this document plus a specific codebase, a concrete implementation plan
should be derivable.

Assumed background (any PD codebase has these): a frozen **target model**; its weights decomposed
into **parameter components** plus a **delta component** (the residual `W_target − Σ components`);
per-datapoint **causal importance (CI)** values gating each component; **masks** applied to
components and to the delta during forward passes; and loss terms for **faithfulness** (delta → 0),
**reconstruction** (masked forward ≈ target output), and **importance-minimality** (CI sparsity).
Masks may be produced **stochastically** and/or via **adversarial** schemes (PGD / persistent PGD).

---

## 1. Core idea

Standard PD decomposes a model over its whole input distribution. tPD instead decomposes the
mechanism used on a narrow **target** distribution, while a broad **nontarget** distribution is used
only to keep components from absorbing general behavior. The intended outcome:

- On **target** inputs: components capture the mechanism — sparse, structured CI.
- On **nontarget** inputs: components are **inactive** (CI ≈ 0); the **delta component alone**
  carries the model's behavior.

This isolates the circuitry specific to the target distribution.

## 2. Two data distributions

The run is parameterized by two data sources of the **same task family** (same model input space):

- **Target distribution** — narrow. How it is specified is domain-dependent:
  - feature-based toy models: a restricted set of *active input features* (others always zero);
  - language models: a fixed set of prompts (or a narrow dataset).
- **Nontarget distribution** — broad/general (the model's normal data).

Both must be loadable as independent train and eval streams, with independently configurable batch
sizes. The target subset / prompts and the nontarget source are separate configuration inputs.

## 3. The delta-forcing mechanic (the heart of tPD)

On **nontarget** data the delta component must be **forced fully on** (delta mask = 1) so that
`Σ (component·mask) + delta` reconstructs the target model exactly *regardless of the component
masks*. Consequences that must be respected wherever masks are built:

- The delta mask is normally sampled (stochastically) or optimized (adversarially). tPD requires the
  ability to **override it to a constant** for a nontarget forward pass.
- This override must reach **every** place a delta mask is produced for a loss that uses the delta
  component (stochastic masking and adversarial/PGD masking alike). Any masking path that does **not**
  use the delta component is unaffected and out of scope.
- **Structural point for adversarial masking:** a forced (constant) delta must **not** also be an
  optimized adversarial variable. Where the delta normally occupies an optimized slot, forcing it must
  remove that slot — i.e. the override changes both the *value* and the *allocation*.
- The override must be **scoped**: it applies only during the nontarget pass (and the nontarget eval
  metrics that need it), and must not leak into target passes.

The same override is reused, with value 0 or 1, by some eval metrics (§7).

## 4. Training loop

Each optimization step performs two passes **sequentially**, accumulating gradients before a single
optimizer update:

1. **Target pass** — normal PD: forward target batch, compute CI, compute the configured losses,
   backward.
2. **Nontarget pass** — forward a nontarget batch, compute CI, recompute any cached quantities the
   target backward invalidated (e.g. the delta tensors, if the autograd graph was freed), then with
   the **delta forced on** compute a *restricted* set of losses (§5) and backward. Gradients
   **accumulate** with the target pass; a single optimizer step follows.

The two passes are run one after the other — back through each pass's graph and free it before the
next — rather than by fusing the two distributions into a single batched forward. This is a
requirement, not just an optimization:

- **Memory:** only one activation graph is ever resident, so peak memory is that of a single pass and
  each distribution may use the full memory budget independently. Accumulated gradients live in
  parameter-sized buffers and do **not** raise the peak — so sequential accumulation is the
  memory-cheap option, not a cost.
- **The delta-forcing override is per-pass, not per-row.** A fused batch would require the delta mask
  to be constant on nontarget rows yet sampled on target rows in the *same* forward — a per-row mask
  the scoped override (§3) deliberately cannot express — and would force every loss to slice the batch
  by distribution.
- **Different loss sets per distribution (§5)** fall out for free: each pass simply runs its own set.

Logging should mirror target logging for the nontarget pass (per-loss values; per-layer CI L0 — the
nontarget L0-throughout-training signal). The whole nontarget pass is gated on targeted mode being
enabled and is otherwise absent.

## 5. Loss handling on nontarget data

Not all losses are meaningful when the delta is forced on:

- **Exclude** losses that become trivial or ill-defined: a pure unmasked-reconstruction loss
  (trivially ~0 once delta=1), and persistent-adversarial losses whose state is coupled to the target
  batch. Regular (non-persistent) adversarial reconstruction *can* be kept.
- **Exclude faithfulness entirely in targeted mode** (see §6).
- **Activation-/hidden-state reconstruction losses are target-only** — exclude them from the
  nontarget set.
- **Importance-minimality** is kept but its coefficient is **scaled** by a configurable ratio (it is
  the main pressure pushing nontarget CI → 0).
- The informative nontarget signal comes from: reconstruction losses (with delta forced on, these
  penalize component contributions that change the output when masked) + scaled importance-minimality.

## 6. Configuration & validation constraints

New configuration inputs: the nontarget data source; nontarget train and eval batch sizes; the
importance-minimality scaling ratio; and the target-subset specifier (active feature set / prompts).

Validation (fail fast) when targeted mode is enabled:

- nontarget batch sizes are present;
- target and nontarget sources are the **same task family** (one model ⇒ one input space);
- **no faithfulness loss and no faithfulness warmup** are configured: both drive the delta → 0, but
  tPD needs the delta nonzero to carry nontarget behavior. A warmup that zeroes the delta immediately
  before targeted training would only force it to re-grow, so the warmup is disallowed alongside the
  ongoing loss.
- domain-specific source exclusivity (e.g. for LMs, exactly one of dataset vs prompts).

Target-subset indices must be range-checked against the model's feature count.

## 7. Evaluations

tPD-specific (require the nontarget eval stream):

- **Target reconstruction** under several masking strategies (e.g. CI-thresholded, raw-CI,
  stochastic, delta-only) with delta off for the component strategies and on for the delta-only
  strategy — measures how well components alone explain target behavior.
- **Nontarget reconstruction** under the same strategies with **delta forced on** for all — plus an
  aggregate nontarget CI L0.
- **CI comparison, target vs nontarget**: a per-component CI view (heatmap) and a mean-CI-per-component
  view, each computed on both distributions. The expectation is dense/structured on target, ~empty on
  nontarget.

Generic (useful but not tPD-specific): per-component **weight magnitude** (e.g. component norm) sorted
/ colored by CI.

L0 throughout training: the existing per-step CI-L0 logging exists for target; the nontarget pass
adds the analogous per-layer nontarget L0 (§4). Aggregate nontarget L0 also comes from the nontarget
reconstruction metric.

## 8. Correctness criteria (what tests assert)

On a converged toy-model run with a small random target feature set:

- **Target inputs** produce sparse, structured CI: for "encode/decode"-style layers, ~one distinct
  active component per target input; for mixing/hidden layers, a small number of active components per
  input.
- **Nontarget inputs** produce ~no active components in the layers that encode the target features
  (the delta carries behavior there).

Tests should also cover: the override semantics (value pinned in scope, no leak, no-op when unset, and
the structural allocation change in the adversarial path); the nontarget loss filtering and impmin
scaling; the validation constraints; target-subset data restriction; and an isolation check that
disabling targeted mode reproduces baseline behavior exactly. The fast invariant *all component masks
= 1 and delta = 1 ⇒ exact target output* exercises the delta path cheaply. Convergence tests are
inherently slower/flakier — seed everything, keep steps minimal, allow tolerance.

## 9. Cross-cutting requirements

- **Isolation / maintainability:** keep new behavior additive and default-off so the codebase behaves
  identically when targeted mode is disabled; prefer a delta-override mechanism that does **not**
  change the signatures of shared loss/mask code, so upstream files stay mergeable.
- **Graph/grad handling:** the nontarget pass must recompute anything the target backward freed.
- **Distributed / mixed precision:** nontarget forward/backward and nontarget eval metrics must
  reduce across ranks like their target counterparts and run under the same autocast policy; a
  process-local override is fine (each rank sets its own). Under data-parallel training the two
  backwards each all-reduce — two syncs per step. This is correct (averaging an already-averaged
  gradient with equal counts is idempotent) and is what the implementation does. Collapsing to a
  single all-reduce by deferring synchronization to the final backward (`no_sync` on the target pass)
  is **unsafe** unless every parameter receives a gradient on that final, restricted-loss pass — so it
  is deferred, not assumed.
- **Determinism:** the target and nontarget streams need independent, reproducible seeding.

## 10. Domain coverage

Toy models (feature-subset targets) and language models (prompt/dataset targets) must be handled by
the **same** training/eval code path; only the construction of the two data streams differs per
domain. The target-subset specifier and prompts loader are the domain-specific pieces.
