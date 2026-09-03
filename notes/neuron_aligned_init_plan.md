# Neuron-aligned component init — implementation plan

Branch: `feature/neuron-aligned-init` (off `feature/dual_obj_jax`). Written 2026-09-03, before
implementation; the "open points" at the end are what to settle in the chat. Revised the same
day after a Codex plan-review (five findings, all folded in — marked **[rev]** below: restore
reference, writer-only blocks, exhaustive sweep, multi-host aggregation, capture memory).

Origin: torch commit `e74dd5fe4` ("Add neuron-aligned MLP component init baseline", 2026-04-22,
snapshot branches only — never merged, old `spd/` layout). There it was a frozen *baseline*
(V/U set from the target weights and never trained, C forced to the neuron count). Here it is
an *initialization*: trainable, C free, adapted to SwiGLU and to C ≪ n_neurons.

## 0. Decisions already taken

| question | decision |
|---|---|
| C per MLP site | free, as configured (SOTA: gate/up 456, down 512). Each site takes the top-C of the block's shared neuron ranking, so top-456 ⊂ top-512 — the 56 extra down neurons are exactly the unmatched ones. |
| neuron score | over the target prompt pool, all positions (BOS caveat, §2) — statistic in §2 |
| scope | any set of decomposed blocks; per-block ranking; anatomy-agnostic (gated `gate/up/down` and plain `fc/down`) |
| non-MLP sites (q/k/v/o, embed…) | `zero_u` values, bit-identical to a `weight_init: zero_u` run at the same seed |
| CI fn | `zero_init_readout: true` (already the SOTA setting) |
| config surface | one literal: `pd.weight_init: neuron_aligned` |
| freezing | none — V/U train as usual; no added noise (§1, "no degenerate point") |

## 1. The recipe

Notation: a site's frozen weight is `W ∈ [d_out, d_in]`; components are `V ∈ [d_in, C]`,
`U ∈ [C, d_out]`, and the site computes `x @ V @ U` + delta, with `delta = W − (V@U)ᵀ`
(so whatever the components do not carry is carried exactly by the delta — the same
mechanism `coupled`/`zero_u` already rely on).

For an MLP block with hidden width `n` (Llama-3.1-8B: 14336) and a block ranking
`S = (s_1, …, s_n)` of neurons by decreasing score, a site with C components takes
`S_C = S[:C]`. `E ∈ [C, n]` is the selection matrix `E[i, s_i] = 1`.

**Hidden writers** (`gate`, `up`; plain anatomy: `fc`) — neuron axis is `d_out = n`:

    V = W[S_C, :]ᵀ        ∈ [d_in, C]     (column i = the weights INTO neuron s_i)
    U = E                 ∈ [C, n]        (component i writes only neuron s_i)
    ⇒ (V@U)ᵀ = W with every non-selected row zeroed; delta = the other n−C rows.

**Reader** (`down`) — neuron axis is `d_in = n`:

    V = Eᵀ                ∈ [n, C]        (component i reads only neuron s_i)
    U = W[:, S_C]ᵀ        ∈ [C, d_out]    (row i = the weights OUT of neuron s_i)
    ⇒ (V@U)ᵀ = W with every non-selected column zeroed; delta = the other n−C columns.

Properties worth stating because they are what the run will inherit:

- `x @ V` for a writer is exactly the pre-activation of the selected neurons; for the
  reader it is exactly the selected post-nonlinearity activations `silu(g)·u`. Component
  activations start *meaningful*, not random.
- Per-subcomponent norm `‖v_i‖·‖u_i‖ = ‖W row/col s_i‖` — the neuron's true weight norm.
  T11's CI-scaled weight decay therefore acts on true-scale vectors from step 0, like `coupled`
  (unit norm on one side, W-image on the other; here the unit side is one-hot rather than a
  random unit vector).
- **No degenerate point** (the "exactly-degenerate" question): with `U = E` one-hot,
  `∂L/∂V = (∂L/∂Wᵀ)·Uᵀ` is the selected rows of the weight gradient and
  `∂L/∂U = Vᵀ·(∂L/∂Wᵀ)` is generically dense — both nonzero from step 0. Nothing needs
  noise to "break" anything; one-hot `U` is just the gauge in which each component IS one
  neuron. Keep the recipe exact.
- Gating is handled by construction: gate and up select the *same* neurons, so
  `silu(gate_i)·up_i` is a single-neuron product from step 0; down reads that same index.

Non-MLP sites get `_coupled_site_vu(W, key_site, C)` with `U` zeroed (`with_silenced_u`) —
i.e. the existing `zero_u` arm, site by site.

## 2. The neuron ranking pass

Run **before init**, on the frozen target, over the **whole target prompt pool** (finite:
`TargetPromptPool.tokens ∈ [n_prompts, prompt_len]`, ~2×10⁴ × 5 tokens for addsub 1..100) —
one exhaustive sweep, not the `pool_batch` sampler. Capture the post-nonlinearity hidden
activation `h_b ∈ [B, T, n]` of every decomposed block `b` in one `clean_forward` with
`capture_keys = {mlp_hidden_tap_key(b)}` — that tap is `_GLUTap.DOWN_INPUT = silu(gate)·up`
(`targets/transformer_taps.py:47`, `glu_transformer.py:758`), so the gated nonlinearity is
already folded in and the plain `fc` anatomy uses the same tap.

**Statistic (proposal — see open point A).** Per block, per neuron `i`:

    score_i = E_{tokens}[ h_i² ] · ‖W_down[:, i]‖²      ("write energy")

i.e. the mean squared norm of what neuron `i` writes into the residual stream on target
data. Two reasons for this over the raw variance of `h_i` that was floated:

1. *Variance kills task detectors.* On a narrow pool a neuron that fires the same on every
   prompt ("this is arithmetic") has ~zero variance on the pool but is exactly a neuron the
   decomposition must reconstruct. The uncentred second moment keeps it; centring only
   makes sense against a *broad* reference (target-vs-nontarget contrast — a later option).
2. *Down-column norm makes neurons comparable.* There is no normalisation between the hidden
   and `down_proj`, and column norms vary; `h_i` alone ranks in hidden-unit units, not by
   residual-stream effect. gate/up/down all use the same block score (the reader's norms are
   part of the score, not a separate ranking), which is what gives the nested top-C property.

Both `h²` and `h` sums are accumulated in fp32 so `variance` is a one-line alternative
(`E[h²] − E[h]²`, same weighting) if we want to A/B the ranking rule itself.

**Positions.** "All positions" — with one caveat to decide (open point B): position 0 is BOS,
identical across the pool, carries no task information, and in Llama-3 hosts massive-activation
neurons that would dominate `E[h²]` at some layers. Proposal: exclude position 0; everything
else counts.

**Mechanics.**

- Composition-side, not core: needs the anatomy and tap names. New module
  `param_decomp/targets/neuron_alignment.py` (targets import core only — layering intact):
  - `mlp_blocks_of(sites, anatomy) -> dict[int, MLPBlockSites]` — per decomposed block, its
    hidden-writer site names and (optional) reader site name; a block with only attention
    sites has no entry. **[rev]** Writer-only and reader-only blocks are both legal: the
    score needs `W_down` of every block that has *any* aligned site, decomposed or not, so it
    is read through a **target-owned accessor** `frozen_mlp_down_weight(model, block) ->
    [d_out, n]` on the concrete `GLUDecomposedModel` (narrowing via `PlacedModel.model`, the
    sanctioned route for target-specific surfaces) — never through `site_weight_delta`, which
    only knows decomposed sites. The column norms `‖W_down[:, i]‖²` are computed inside the
    same jit as the moments.
  - `accumulate_neuron_moments(model: PlacedModel, blocks, slices) -> dict[int, (Σh, Σh², n_tok)]`
    — jitted `clean_forward` + masked reduction per slice, accumulating in fp32.
    **[rev] Multi-host:** the reduction's `out_shardings` is declared **fully replicated**
    (`NamedSharding(mesh, P())`), so the batch collective happens inside the jit and every
    process holds the complete `[n]` vectors (the capture may arrive batch-sharded on the
    data axes and even TP-sharded on `n`; the declared output sharding forces the gather).
    Only then `np.asarray`. The final ranked indices are cross-checked with
    `multihost_utils.assert_equal` before init — a divergent selection across processes is a
    crash, not a silent split-brain decomposition.
  - `neuron_alignment(model, sites, moments, down_sq_norms, score) -> NeuronAlignment` — score,
    `argsort` descending with index tie-break on host numpy, per-site `top-C`.
- **[rev] Exhaustive sweep** in `experiments/lm/training_targeted.py` (`train_targeted`, after
  `pool` and `mesh` exist, before the engine call): a new `pool_slices(pool, global_batch) ->
  Iterator[(tokens [global_batch, T], row_mask [global_batch])]` that **slices `pool.tokens`
  row-contiguously** (`[start:stop]`), pads only the final slice and returns its row mask;
  each process takes its `per_process` share of the slice exactly as `pool_global_batch` does
  for the *distribution*, but the *row choice* never goes through `pool_batch`, which samples
  with replacement. Every prompt contributes exactly once (pinned by a test with uniquely
  identifiable rows).
- **[rev] Capture memory:** a capture is `B·T·n·4` ≈ 37 MB per block at `B=128, T=5`, and the
  forward retains one slot per *requested* block simultaneously, so a run decomposing many
  blocks must not request them all at once. Blocks are captured in **groups under a byte
  budget** (constant, 1 GiB ⇒ ≤ 27 blocks at this pool geometry; a longer-prompt pool shrinks
  the group), one compile per group. Peak = `group_size · B · T · n · 4` on top of the frozen
  model; state the actual number in the run log. ~160 forwards per group of a 5-token batch
  through 8B: seconds plus the compile.
- Diagnostics written once to `run_dir/neuron_alignment.json` and printed: per block the
  selected indices, their scores, and the **covered write-energy fraction**
  `Σ_{top-C} score / Σ_all score` for each site's C — the number that tells us whether 456
  is enough for this task. Also log it as a step-0 scalar in the sink.

## 3. Core changes (by file)

`param_decomp/core/components.py`
- `NeuronAxis = Literal["d_out", "d_in"]`; `@dataclass(frozen) SiteNeuronAlignment(neuron_axis, neurons: Int[Array, " C"])`;
  `NeuronAlignment = dict[str, SiteNeuronAlignment]` (MLP sites only; absent ⇒ zero_u).
  Core stays anatomy-blind: the target says *which axis of W is the neuron axis* and *which
  neurons*; core does the linear algebra.
- `_neuron_aligned_site_vu(W, alignment, C) -> (V, U)` — §1, asserting `len(neurons) == C`
  and `neurons < W.shape[axis]`, unique.
- `init_component_stacks_neuron_aligned(sites, target_weights, alignment, key)` — per site:
  aligned ⇒ `_neuron_aligned_site_vu`; else `with_silenced_u`-style `_coupled_site_vu` on the
  site's own key. **Key discipline:** split `len(sites)` keys exactly as
  `init_component_stacks_coupled` does and index by site position; aligned sites simply do not
  consume theirs, so every non-MLP site is bit-identical to `zero_u` at the same seed
  (pinned by a test).

`param_decomp/core/init_placed.py`
- `init_component_stacks_neuron_aligned_placed(model, key, rules, alignment)` — mirrors
  `init_component_stacks_coupled_placed`: `W` read inside the jit via the zero-stacks
  `weight_deltas` trick, indices passed as small int32 jit args, `out_shardings` from
  `component_stacks_shardings`.

`param_decomp/core/configs.py`
- `WeightInit = Literal["default", "coupled", "zero_u", "neuron_aligned"]` + docstring row.

`param_decomp/core/run_state.py`, `run.py`
- `init_decomposition(..., neuron_alignment: NeuronAlignment | None = None)`.
  **[rev] Reference vs. fresh init.** The engine always builds a fresh reference state
  (`init_train_state`, `run.py:598`) *before* trying `restore_latest` / `init_from_parent`, so
  a requeue or fine-tune of a `neuron_aligned` run would otherwise demand an alignment just
  to build a tree it is about to overwrite. Rule: the `neuron_aligned` arm with
  `neuron_alignment=None` builds the reference with the alignment-free `zero_u` values (same
  shapes, same seed discipline); the alignment is *required* exactly at the engine's
  fresh-init branch — `_init_or_restore_state`, after both restore paths declined —
  which asserts `pd.weight_init != "neuron_aligned" or neuron_alignment is not None`. A root
  that forgets the sweep on a fresh run therefore crashes there rather than silently training
  from `zero_u`. Tree structure is unchanged, so `load_run.py:152`'s
  `eval_shape(init_decomposition(...))` structure-only use keeps working.
- Thread `neuron_alignment` through `init_train_state` → `_init_or_restore_state` →
  `_prepare_run` → `run_targeted_decomposition_training(...)` (new keyword, default `None`).
- `run_decomposition_training` (plain PD) refuses `neuron_aligned` with a clear assertion:
  the statistic is defined on a target pool, which a plain run does not have. Fail closed.

`param_decomp/experiments/lm/training_targeted.py`
- If `built.pd.weight_init == "neuron_aligned"`: run §2, assert `built.ci_fn.zero_init_readout`
  (open point C), pass the alignment to the engine, write the diagnostics.
- **[rev]** Restore/requeue: the root runs the sweep only for a genuinely fresh run — own
  `ckpts/` empty (the same predicate `restore_latest` will use) and `resume_provenance is
  None` (S33 loads the parent's V/U). Otherwise it passes `neuron_alignment=None` and the
  engine builds the alignment-free reference (§3, `run_state.py`). The two predicates are
  evaluated once, root-side, and the engine's fresh-init assert is the backstop.

## 4. Tests

`param_decomp/core/tests/test_weight_init.py` (extend; uses `tiny_glu_decomposed_lm` and
`tiny_simple_mlp_decomposed_model`):
1. Writer/reader read-back: for a hand-chosen `S_C`, `(V@U)ᵀ` equals `W` on the selected
   rows/columns and is zero elsewhere; `weight_deltas` equals the complement.
2. Nested selection: with `C_down > C_gate`, the down neurons contain the gate/up neurons.
3. Non-MLP sites equal the `zero_u` values bit-for-bit at the same seed.
4. Placed == eager values (mirror `test_placed_init_matches_the_eager_values`).
5. Both anatomies (gated and plain) go through the same path.

`param_decomp/targets/tests/test_neuron_alignment.py`:
6. Score against a hand computation from `capture_clean(model, tokens, {mlp_hidden_tap_key})`
   on a tiny model, including BOS exclusion and last-chunk padding mask.
7. `mlp_blocks_of` on a mixed site set (attention-only block, hidden-writers-only block,
   reader-only block, full block) — **[rev]** the writer-only and reader-only blocks rank
   via `frozen_mlp_down_weight`, and the result equals the full block's ranking.
8. **[rev]** Exhaustive coverage: a pool whose rows are uniquely identifiable (one distinct
   token per row) — every row contributes exactly once to the moments, including the padded
   final slice, at a `global_batch` that does not divide `n_prompts`.
9. **[rev]** Multi-device (the 8-CPU-device test mesh, batch axis *and* `n` sharded via a
   placement table): the reduction lands fully replicated, `np.asarray` succeeds on every
   process, and the ranking equals the unplaced eager one. A single-device placed==eager
   test does not exercise this.
10. **[rev]** Requeue and fine-tune: a `neuron_aligned` run restored from its own checkpoint,
    and one initialized from a parent (S33), both start with `neuron_alignment=None`, run no
    sweep, and restore bit-for-bit; a fresh run with `None` hits the engine assert.

Config: the literal parses; the plain LM root refuses it.

## 5. SPEC / docs

- `core/SPEC.md`: a new T-row (targeted-only invariant) stating the statistic, the position
  rule, the per-block ranking and nested top-C, the §1 formulas, the `zero_u` fallback for
  other sites, and that nothing is frozen. Also a one-paragraph note that `weight_init` arms
  are currently undocumented in SPEC — add all four while there.
- `WeightInit` docstring, `core/CLAUDE.md` init pointer, `notes/dual_objective/README.md` row.

## 6. Open points to settle before implementing

A. **Score**: `E[h²]·‖W_down[:,i]‖²` (proposed) vs. the centred variance you suggested. My
   recommendation is the former as the default for the reason in §2.1; accumulating both
   moments makes the other a one-line switch for an A/B.
B. **BOS**: exclude position 0 from the statistic (proposed) or literally all positions.
C. **CI fn coupling**: assert `zero_init_readout: true` when `weight_init: neuron_aligned`
   (fail-closed, "one option means all of it"), vs. leaving the two fields independent.
D. **Ordering of components**: slot `i` = `i`-th highest score (proposed; makes component
   index interpretable in the app and lets a later `C` change be a prefix change) — or keep
   ascending neuron index.
E. **Coverage number**: is the covered-write-energy fraction (§2 diagnostics) the number you
   want at step 0, or also a per-position variant?

## 7. First experiment

Twin of `notes/dual_objective/addsub-L18-sota.yaml` with `weight_init: neuron_aligned`, same
seed. Read at step 0: coverage fraction, target-stream KL/PGD at step 0 (should start far
below the `coupled` run's), CI L0 at step 0 (expected high — every selected neuron starts
"important"; watch how fast imp-min prunes). Compare at 4k/20k against p-5b7fa697 on the
same panels as the campaign close-out.
