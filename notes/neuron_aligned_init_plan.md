# Neuron-aligned component init (`neuron_aligned_targeted`) — implementation plan

Branch: `feature/neuron-aligned-init` (off `feature/dual_obj_jax`). Written 2026-09-03, before
implementation. Revision 2: after a Codex plan-review (five findings, folded in) and the
decisions of the second discussion round — the ranking is now a **cached artifact produced by
a separate harvest script**, the literal is `neuron_aligned_targeted` (upstream may take
`neuron_aligned` for a full-data variant), BOS handling is an A/B, no CI-fn assert.

Origin: torch commit `e74dd5fe4` ("Add neuron-aligned MLP component init baseline", 2026-04-22,
snapshot branches only — never merged, old `spd/` layout). There it was a frozen *baseline*
(V/U set from the target weights and never trained, C forced to the neuron count). Here it is
an *initialization*: trainable, C free, adapted to SwiGLU and to C ≪ n_neurons.

## 0. Decisions taken

| question | decision |
|---|---|
| C per MLP site | free, as configured (SOTA: gate/up 456, down 512). Each site takes the top-C of the block's shared neuron ranking, so top-456 ⊂ top-512 — the 56 extra down neurons are exactly the unmatched ones. |
| neuron score | uncentred second moment of the post-nonlinearity activation, weighted by the down column norm (§2) — over the whole target prompt pool |
| BOS (position 0) | not decided by fiat: the harvest script has a `bos` flag, the A/B is two artifacts |
| scope | any set of decomposed blocks; per-block ranking; anatomy-agnostic (gated `gate/up/down` and plain `fc/down`) |
| non-MLP sites (q/k/v/o, embed…) | `zero_u` values, bit-identical to a `weight_init: zero_u` run at the same seed |
| CI fn | `zero_init_readout: true` is the recipe (already the SOTA setting) but is **not** asserted — the two fields stay independent |
| component slot order | slot `i` = `i`-th highest score (the artifact stores the full ranking; a C change is a prefix change) |
| config surface | `pd.weight_init: neuron_aligned_targeted` + the location of the ranking artifact (§2b); nothing else |
| freezing | none — V/U train as usual; no added noise (§1, "no degenerate point") |
| where the forward passes run | once, offline, in a harvest script; the trainer never sweeps |

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

Properties the run inherits:

- `x @ V` for a writer is exactly the pre-activation of the selected neurons; for the
  reader it is exactly the selected post-nonlinearity activations `silu(g)·u`. Component
  activations start *meaningful*, not random.
- Per-subcomponent norm `‖v_i‖·‖u_i‖ = ‖W row/col s_i‖` — the neuron's true weight norm.
  T11's CI-scaled weight decay therefore acts on true-scale vectors from step 0, like `coupled`
  (unit norm on one side, W-image on the other; here the unit side is one-hot rather than a
  random unit vector).
- **No degenerate point**: with `U = E` one-hot, `∂L/∂V = (∂L/∂Wᵀ)·Uᵀ` is the selected rows
  of the weight gradient and `∂L/∂U = Vᵀ·(∂L/∂Wᵀ)` is generically dense — both nonzero from
  step 0. One-hot `U` is just the gauge in which each component IS one neuron. Exact recipe,
  no noise.
- Gating is handled by construction: gate and up select the *same* neurons, so
  `silu(gate_i)·up_i` is a single-neuron product from step 0; down reads that same index.

Non-MLP sites get `_coupled_site_vu(W, key_site, C)` with `U` zeroed (`with_silenced_u`) —
i.e. the existing `zero_u` arm, site by site.

## 2. The neuron ranking

The ranking depends only on the **target model** and the **target prompt pool** — not on the
decomposition (C, seed, losses, blocks). It is therefore harvested once into an artifact and
loaded by every run that uses the init.

**Statistic.** Per block `b`, per neuron `i`, with `h ∈ [B, T, n]` the post-nonlinearity
hidden activation (`silu(gate)·up`; for the plain anatomy the post-GELU `fc` output):

    score_i = E_{tokens}[ h_i² ] · ‖W_down[:, i]‖²      ("write energy")

the mean squared norm of what neuron `i` writes into the residual stream on target data.
Uncentred, so a neuron that fires the same on every prompt ("this is arithmetic") — zero
variance on a narrow pool, yet exactly what the decomposition must reconstruct — is kept.
The down column norm converts hidden-unit scale into residual-stream effect (there is no
normalisation between the hidden and `down_proj`, and column norms vary). One block score
serves gate/up/down alike; that is what gives the nested top-C property. `Σh` is accumulated
alongside `Σh²` (free) so the centred variance stays a one-line alternative in the script.

**Positions.** All positions, with the `bos` flag deciding whether position 0 counts. It is
identical across the pool, carries no task information, and in Llama-3 hosts
massive-activation neurons that can dominate `E[h²]` — but whether that matters at L18 on
this pool is an empirical question, hence the A/B (§7).

**Exhaustive sweep.** Training draws target batches by *sampling the pool with replacement*
(`pool_batch`: `rng.integers(0, n_prompts, size=batch)` — a step's batch has duplicates and
omissions, by design, for O(1) resume). The ranking instead walks the pool **once, every
prompt exactly once**: contiguous slices `tokens[start:stop]`, the last one padded with a
row mask. The statistic is then the exact pool mean, not a seed-dependent Monte-Carlo
estimate — cheap because the pool is finite and small (addsub 1..100: ~2×10⁴ prompts × 5
tokens).

### 2a. The harvest script (new, composition-side)

`python -m param_decomp.experiments.lm.harvest_neuron_ranks --config <run.yaml>
--out_dir <abs> [--bos exclude|include] [--layers all|<list>] [--batch_size N]`

Mirrors `prestage_tokenized.py` (fire CLI, `--out_dir`, artifact + `meta.json`). Reads the run
config for the **target** (`decomposition.target`) and the **prompts** (`prompts`) only —
everything else in the YAML is ignored, so one artifact serves every decomposition config
over that (model, pool) pair.

- Loads the frozen target the way the trainer does, single process (`assert
  jax.process_count() == 1` — the sweep is a few seconds; multi-host aggregation is not worth
  specifying). Device sharding *within* the process is fine: the jitted per-slice
  `clean_forward` + masked reduction declares its output fully replicated
  (`NamedSharding(mesh, P())`), so the `[n]` vectors are complete on the host.
- Captures the `mlp_hidden` tap of the harvested blocks
  (`transformer_taps.mlp_hidden_tap_key(b)` = `_GLUTap.DOWN_INPUT`), in **groups under a
  byte budget** (a capture is `B·T·n·4` per block and the forward retains one slot per
  requested block; 1 GiB budget ⇒ all 32 layers in two groups at `B=128, T=5`).
  Default `--layers all`: 32 × 14336 × (int32 rank + fp32 score) ≈ 3.7 MB, so one artifact
  covers any block selection.
- `W_down` column norms are read through a target-owned accessor
  `frozen_mlp_down_weight(model, block)` on the concrete `GLUDecomposedModel` (narrowing via
  `PlacedModel.model`, the sanctioned route for target-specific surfaces) — not through
  `site_weight_delta`, which only knows decomposed sites. That is what makes writer-only or
  reader-only blocks rankable.
- Writes `neuron_ranks.npz` — per block `rank_{b}` (`int32[n]`, neuron indices by descending
  score, index tie-break) and `score_{b}` (`fp32[n]`, in that order) — and `meta.json`:
  target `model_name`, prompts spec (the grid parameters, or the file's sha256), tokenizer,
  `statistic: write_energy`, `bos`, `n_prompts`, `positions_counted`, layers, git commit.
  Plus, per block, the cumulative covered-energy curve `cum_{b}` (`fp32[n]`) so "how much of
  the target-task MLP write energy does C cover" is a lookup, not a recomputation.

### 2b. Loading at train time

Config: `pd.weight_init: neuron_aligned_targeted` and `pd.neuron_ranks: {kind: ref, name}
| {kind: dir, dir: <abs>}` — the same two shapes as `data` (`DatasetRef` resolved under
`data_root` via `infra.dataset_store.resolve_dataset_ref`, or an explicit location). Required
iff the literal is set (pydantic model validator; fail closed both ways).

`experiments/lm/training_targeted.py` (`train_targeted`, before the engine call):
- load the artifact; **assert provenance** — `model_name` equals the run's target, the prompts
  spec matches the run's `prompts` (grid params / file sha256), every decomposed MLP block is
  present. A mismatch is a crash with the two specs printed, never a silent init from the
  wrong pool.
- build `NeuronAlignment`: for each MLP site, `neurons = rank_b[:C]` and the site's neuron
  axis (`d_out` for writers, `d_in` for the reader — from the anatomy, root-side).
- write `run_dir/neuron_alignment.json` (artifact path + sha256, per-site C, covered-energy
  fraction `cum_b[C−1]`) and log the fractions as step-0 scalars.
- always pass the alignment to the engine — a file read is cheap, so requeue and fine-tune
  (S33) build the same aligned reference and then overwrite it; no `None` special case
  beyond the structure-only consumers (`load_run.py:152`, default arm).

## 3. Core changes (by file)

`param_decomp/core/components.py`
- `NeuronAxis = Literal["d_out", "d_in"]`; `@dataclass(frozen) SiteNeuronAlignment(neuron_axis, neurons: Int[Array, " C"])`;
  `NeuronAlignment = dict[str, SiteNeuronAlignment]` (MLP sites only; absent ⇒ zero_u).
  Core stays anatomy-blind: the target says *which axis of W is the neuron axis* and *which
  neurons*; core does the linear algebra.
- `_neuron_aligned_site_vu(W, alignment, C) -> (V, U)` — §1, asserting `len(neurons) == C`,
  `neurons < W.shape[axis]`, unique.
- `init_component_stacks_neuron_aligned(sites, target_weights, alignment, key)` — per site:
  aligned ⇒ `_neuron_aligned_site_vu`; else `_coupled_site_vu` + `with_silenced_u` on the
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
- `WeightInit = Literal["default", "coupled", "zero_u", "neuron_aligned_targeted"]` + docstring
  row; `PDConfigBase.neuron_ranks: NeuronRanksRef | NeuronRanksDir | None` with the
  iff-validator.

`param_decomp/core/run_state.py`, `run.py`
- `init_decomposition(..., neuron_alignment: NeuronAlignment | None = None)`; the
  `neuron_aligned_targeted` arm asserts it is present. Tree structure is unchanged.
- Thread `neuron_alignment` through `init_train_state` → `_init_or_restore_state` →
  `_prepare_run` → `run_targeted_decomposition_training(...)` (new keyword, default `None`).
- `run_decomposition_training` (plain PD) refuses the literal with a clear assertion: the
  statistic is defined on a target pool, which a plain run does not have. Fail closed.

`param_decomp/targets/neuron_alignment.py` (new)
- `mlp_blocks_of(sites, anatomy)`, `frozen_mlp_down_weight(model, block)`, the jitted
  `accumulate_neuron_moments`, `rank_neurons(moments, down_sq_norms)`, the artifact
  reader/writer with the provenance check. Targets import core only — layering intact.

`param_decomp/experiments/lm/harvest_neuron_ranks.py` (new) — §2a.
`param_decomp/experiments/lm/training_targeted.py` — §2b.

## 4. Tests

`param_decomp/core/tests/test_weight_init.py` (extend; `tiny_glu_decomposed_lm` and
`tiny_simple_mlp_decomposed_model`):
1. Writer/reader read-back: for a hand-chosen `S_C`, `(V@U)ᵀ` equals `W` on the selected
   rows/columns and is zero elsewhere; `weight_deltas` equals the complement.
2. Nested selection: with `C_down > C_gate`, the down neurons contain the gate/up neurons.
3. Non-MLP sites equal the `zero_u` values bit-for-bit at the same seed.
4. Placed == eager values (mirror `test_placed_init_matches_the_eager_values`).
5. Both anatomies (gated and plain) go through the same path.

`param_decomp/targets/tests/test_neuron_alignment.py`:
6. Score against a hand computation from `capture_clean(model, tokens, {mlp_hidden_tap_key})`
   on a tiny model, both `bos` settings, last-slice padding mask.
7. `mlp_blocks_of` on a mixed site set (attention-only, writer-only, reader-only, full block);
   the partial blocks rank via `frozen_mlp_down_weight` and equal the full block's ranking.
8. Exhaustive coverage: a pool whose rows are uniquely identifiable (one distinct token per
   row) — every row contributes exactly once, at a `batch_size` that does not divide
   `n_prompts`.
9. Multi-device (the 8-CPU-device test mesh, batch axis *and* `n` sharded): the reduction
   lands fully replicated and equals the unplaced eager ranking.
10. Artifact round-trip + provenance: write, read back, and each of {model name, grid params,
    file sha256, missing block} mismatches is a crash.

`param_decomp/experiments/lm/`:
11. Config: the literal parses; `neuron_ranks` is required iff the literal is set; the plain
    LM root refuses the literal.
12. Requeue and fine-tune (S33): a `neuron_aligned_targeted` run restores bit-for-bit from its
    own checkpoint / its parent; the harvest is never invoked by the trainer.

## 5. SPEC / docs

- `core/SPEC.md`: a new T-row (targeted-only invariant) stating the statistic, the `bos`
  flag, the exhaustive-pool rule, the per-block ranking and nested top-C, the §1 formulas,
  the `zero_u` fallback for other sites, that nothing is frozen, and that the ranking is an
  artifact with checked provenance. `weight_init` arms are currently undocumented in SPEC —
  add all four while there.
- `WeightInit` docstring, `core/CLAUDE.md` init pointer, `experiments/lm/CLAUDE.md` for the
  harvest script, `notes/dual_objective/README.md` row, `CONFIGS.md` if a seat gets the
  literal.

## 6. Still open (minor)

- Artifact naming under `data_root/neuron_ranks/<name>/` — proposal:
  `<pool>_<model>_<statistic>_bos-<excl|incl>` (e.g. `addsub1-100_llama31-8b_we_bos-excl`).
- Whether the harvest should also emit the centred-variance ranking as a second `rank_var_{b}`
  array in the same artifact (free; keeps the ranking-rule A/B one flag away). Proposal: yes.

## 7. First experiments

1. Harvest twice from the SOTA config: `--bos exclude` and `--bos include`. Read the two
   `cum_{18}` curves at C=456/512 — if the top-C sets differ by a handful of neurons the BOS
   question is moot and one run suffices.
2. Twin(s) of `notes/dual_objective/addsub-L18-sota.yaml` with `weight_init:
   neuron_aligned_targeted`, same seed. Read at step 0: covered-energy fraction, target-stream
   KL/PGD (should start far below the `coupled` run's), CI L0 (expected high — every selected
   neuron starts "important"; watch how fast imp-min prunes). Compare at 4k/20k against
   p-5b7fa697 on the campaign close-out panels.
