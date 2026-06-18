# Migration review guide — what to scrutinize

This session optimistically merged a large batch of torch-shed + JAX-migration PRs into
`feature/jax` (swarm model: agent-validated → merged; human review in hindsight before the
squash to `main`). This points the review at the highest-leverage spots so it doesn't take
ages.

## The safety net: equivalence goldens

Every trainer-touching change kept the LM training trajectory **bit-identical**
(`jax_single_pool/tests/equivalence/` — JAX vs frozen torch-oracle goldens, per loss
term). So the *training math is provably unchanged*. PRs that only touch trainer / loss /
eval semantics can be **skimmed** — the goldens are the proof: axis-semantics (#866),
attn-patterns→JAX (#862), hidden-acts, TMS (#861) / ResidMLP (#864), config-collapse
(#873), fine-tune (#874), caching (#875).

Review effort belongs on the **goldens-blind surface**: the checkpoint migration, the
torch-free consumer metadata, partial-state loads, config plumbing — where a silent
semantic bug wouldn't trip a golden.

## Review these (ranked by silent-corruption risk)

### 1. Checkpoint migration — `tools/migrate_c49k_checkpoint.py` (#870) — HIGHEST
Remaps the old frozen-clone layout (`Vg/Ug/Vu/Uu/Vd/Ud`, legacy 3-D V/U) → current
site-keyed `components.vu[<site>][0|1]` (2-D, leading singleton squeezed), + re-nests the
persistent sources. Verified STRUCTURE (shapes, `step==175000`, finiteness) but **not
value-equivalence** — a swapped `V`↔`U`, a mis-mapped `g/u/d`→`gate/up/down`, or a wrong
squeeze axis would pass every check and silently corrupt the 175k fine-tune base. A
leaf-value equivalence check (old leaves == migrated leaves under the remap) is being
added; **until it's green, treat the 175k fine-tune base as unproven.** Look at: the remap
table + the squeeze.

### 2. Delete-all-torch: `build_target` / `JaxPDAdapter.topology` swap (in flight)
Derives `TransformerTopology` dims from the JAX target / pinned config instead of
instantiating a torch model. If the derived dims are off, autointerp/intruder silently get
wrong topology — no golden covers consumers. Check the derived topology matches the old
torch path.

### 3. Fine-tune `ResumeProvenance` — `run.py` + `checkpoint.py::init_from_parent` (#874)
Loads V/U + ci_fn from the parent, **fresh optimizer + step 0**. Judgment calls, not bugs:
(a) the structural-compat assert (reads the parent's pinned config), (b) whether
fresh-optimizer/step-0 is the intended fine-tune semantics vs continuing Adam state.

### 4. Config-collapse — `is_jax_run` discriminator (#873)
Changed from "has `torch_config:` key" → "has an orbax `ckpts/` dir beside `config.yaml`".
A mis-classification silently mis-routes adapter/harvest. Also quietly fixed a latent bug
(`pile_ppgd_bsc` was missing `weights_dtype: bfloat16`) — confirm that fix.

## Skim (low-risk — enforced by validation, not correctness-subtle)

Pure deletions, gated by `make type` + the non-slow test suite + consumer import-smoke:
CLT delete (#863), dead-dir delete (#867), app strip (#868), the torch-run-loading cascade
(#872), and the delete-all-torch deletions. Review for *scope judgment* (did it cut the
right things), not correctness.

## Known gaps

See `MIGRATION_HOLES.md` — orphaned eval metrics (UVPlots / PermutedCIPlots /
general-IdentityCIError), deferred #10 (torch→jax run adapter) + app re-add, pretrain
reimplementation in JAX.
