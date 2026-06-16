# JAX-first: standalone training + torch retirement

The pivot (decided 2026-06-16, with Oli) and the **push-1** scope. The JAX single-pool
trainer is the **production** trainer (faster); **torch remains the battle-tested
semantic oracle** for correctness, preserved in git history + frozen goldens. `SPEC.md`
(grounded in torch) stays the contract; JAX **conforms** to it. This doc is the
structural plan.

## 1. The decision

- **Torch is the oracle; JAX is production.** Torch is the battle-tested reference for
  *correctness* — the SPEC is grounded in it. JAX is faster and is the trainer we run,
  but it must **conform** to torch's semantics, not redefine them. "Faster" is not
  "canonical." The live torch trainer is removed from HEAD but **preserved via git** (tag
  `torch-oracle`) + the frozen parity goldens — resurrectable on demand to regenerate
  references.
- **Drop the hard features outright** (no JAX parity, *removed* not ported):
  - **Attribution graphs** — `dataset_attributions/`, `graph_interp/`,
    `topology/gradient_connectivity.py`, the app's graph/attribution/intervention/circuit
    routers. This was the *keystone* (the `cache_type='component_acts'` detach-boundary
    forward + vmapped vjp); ~7 of the catalog's hard items sat on it.
  - **Interactive app autodiff** — `optim_cis.optimize_ci_values`,
    `editing/editable_model.py`, app PGD intervention. The app becomes a **read-only
    viewer**.
  These two drops remove the *entire* genuinely-hard set surfaced by the transition-cost
  catalog (wf `88a402d8`).
- **No dual-backend generic interface.** We considered an `Arr`-generic protocol to keep
  torch alive behind a seam — unnecessary, because the hard consumers are *dropped*, not
  ported-to-torch. So `DecomposedLM` is simply THE interface (no `Arr` TypeVar).
- **Adjacent torch consumers are untouched in push 1.** Harvest / app (read-only) /
  eval-consumers / autointerp / clustering keep running on torch, fed by the JAX
  `export` → `offline_eval.py` bridge. They are ported (or dropped) in *later* pushes.

## 2. Canonical interface (JAX-only)

Promote what already exists — pure-functional, state-injected:

- **`DecomposedLM`** (`lm.py`): ordered `sites` + `clean_logits` / `site_inputs` /
  `masked_logits` / `weight_deltas` over `(frozen, vu)` pytrees.
- **`CIFn`** (`ci_fn.py`): `site_inputs -> (ci_lower, ci_upper)`. (SRP: model and CI are
  separate concerns; a run is the pair + frozen target.)
- **Dispatch lives only at the edges:** one `load(run) -> Decomposition` and one
  `export(state) -> safetensors`. Nothing else knows architecture/layout.

## 3. Push-1 scope — self-contained JAX single-pool LM training

**In:** the `param_decomp_jax` single-pool **LM training RUNTIME** (`jsp-train` + the
trainer/losses/eval) imports **nothing** from the adjacent submodules.
**Out:** TMS/ResidMLP trainers; every consumer; the dropped features; **the launcher**
(see below — it stays).

The boundary: "self-contained" constrains the **runtime** (the thing that runs on GPUs),
not the login-node launcher. `lab → param_decomp_jax` is a fine dependency direction;
only `param_decomp_jax → adjacent` is forbidden. `run.py` (`jsp-train`) is already
runtime-clean (no lab imports). The deps to address:

| bucket | today | action |
|---|---|---|
| **Config** | `param_decomp_config` (canonical, torch-free) **+ `torch_config.py` conversion** | **Keep `param_decomp_config`** as the shared foundation; **read the schema directly**; **delete `torch_config.py`** (the conversion layer existed only to translate the canonical schema into internal NamedTuples — gone). |
| **Launcher + infra** | `pd-jax-lm` (lab-side, login-node submission) + `param_decomp_lab.infra` (slurm / git-snapshot / wandb) | **KEEP as-is.** It's the submission *wrapper*, not the GPU runtime — runs in the lab venv and reuses battle-tested infra (the two-venv split is already the `pd-jax-lm` design). `lab → jax` is fine; nothing to do. |
| **Torch parity refs** | `param_decomp.*` + lab `vendored` / `three_pool` / `batch_and_loss_fns` — **only in test/export tooling** (golden generation, the rotted `verify_export_torch.py`) | **Freeze the goldens** as committed artifacts; **drop the torch-reference generation** and the rotted verifier. The fixtures stay; the torch code that produced them goes. |
| **Logging / sink** | in-package already (`run.py` `MetricsSink`) | keep |

Vendored targets: `vendored_jax/` (Llama, GPT-2) is already self-contained — keep; drop
the torch-vendored *parity* imports.

## 4. Retire the torch trainer from HEAD — preserved in git (push 1)

**First tag the pre-deletion commit `torch-oracle`** and document the regen recipe
(`git worktree` the tag → torch venv → run to regenerate a golden). The trainer is the
oracle; it lives in git, not in HEAD.

Removed from HEAD (training-only, nothing in push-1 imports them once §3 lands):
`param_decomp/optimize.py` (`Trainer`), `train_step.py`, `faithfulness_warmup.py`, the
whole `param_decomp/metrics/` tree (loss metrics incl. PPGD), and the torch training
drivers in `experiments/lm/run.py` / `offline_eval.py`'s train path.

**Stays (bridge substrate, do NOT delete in push 1):** `component_model.py`,
`ci_fns.py`, `ci_sigmoids.py`, `components.py`, `masks.py`, `decomposition_targets.py` —
the torch consumers reach these via `component_model_io.py` (`ComponentModel`,
`CIOutputs`, `OutputWithCache`) to load JAX exports. They retire with their consumers in
later pushes.

## 5. The 207 parity issues — reconcile

The "match-torch" premise is largely gone. Re-cut:
- **Re-default to match-torch:** the numeric-seam decisions (`#624/#625` CI numerics,
  `#644/#645` cosine/grad-clip) **default to MATCHING torch** (the oracle); diverge only
  deliberately, documented in `SPEC.md`. Decide per-seam — not auto-closed. (The wave-1
  SPEC amendments + tests already merged stay.)
- **Close as wontfix-dropped:** every attribution-graph / app-autodiff issue.
- **Keep / re-home:** consumer-port issues (harvest/clustering/eval) → a *later* push.
- **New push-1 issue set** (small, focused — launcher untouched):
  1. Read `param_decomp_config` directly; delete `torch_config.py`.
  2. Freeze parity goldens; drop the torch-reference generators + the verifier (severs
     the last `param_decomp_jax → torch` imports).
  3. Tag `torch-oracle`; retire the torch trainer from HEAD (`optimize` / `train_step` /
     `metrics/` / `faithfulness_warmup` + torch train drivers).
  4. `param_decomp_jax` runtime `pyproject` standalone (deps: jax/optax/orbax/… +
     `param_decomp_config` only); confirm `jsp-train` imports nothing adjacent.
  5. Validation: standalone JAX training smoke — save **and** resume at production
     per-rank shape.

## 6. Deferred (kept, not in push 1)

`AutointerpLabels`, non-vendored targets (TMS/ResidMLP are tiny ports; generic HF
`from_pretrained` is the one open-ended tail — production vendored Llama is covered),
hidden-acts/attn-pattern recon (already a refused spec item).
