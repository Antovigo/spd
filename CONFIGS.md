# Config policy

The repo maintains a small, named set of experiment configs — the **canonical
seats** — and nothing else. Every yaml the repo carries is a maintenance
obligation: each schema change must migrate it, forever. Seats are capped at
**10 LM configs**, each with a stated purpose.

## Why committing sweep configs adds nothing

A launched run's config provenance already lives in two places, neither of them
the repo tree:

1. the run dir's pinned `launch_config.yaml` (immutable; resume byte-compares it),
2. the wandb run config.

So a sweep/profile/one-off yaml committed "for the record" records nothing —
it only rots. Keep one-offs in your own workspace (the composition roots take a
config at any path); if the sweep matters, whatever you write up cites the run
ids, and the run dirs carry the exact configs.

## The canonical seats

| seat | file | purpose |
|---|---|---|
| llama8b L18 | `param_decomp/experiments/lm/configs/llama8b_l18_C49k_200k.yaml` | the L18-MLP decomposition flagship recipe |
| llama8b L18 addsub tPD | `param_decomp/experiments/lm/configs/addsub-L18-dual-obj.yaml` | the CURRENT STATE OF THE ART for the JAX targeted (dual-objective) addsub decomposition — tracks the best-known recipe, updated as it advances (lineage: torch addsub-L18-11-trunk-imp2x → jax-dual-01 → 2026-08-18 eval/batch/init/seq-passes revision → addsub-L18-14 imp-min/frequency warm-start scheduling) |
| llama8b full-model | `param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b64_dp64.yaml` | the full-32L production recipe (HSDP tp=1, dp64 — the tp>1/dp128 seats died with the tp=8 PPGD-source pathology; validated on H100s) |
| save-path smoke | `param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b32_dp32_SAVESMOKE.yaml` | cheap end-to-end save/resume smoke launch |
| config-suite fixture | `param_decomp/experiments/lm/configs/llama8b_l18_b128_cmp32.yaml` | the representative full config the core config/resume tests load (`test_config.py`, `test_finetune_resume.py`, `test_llama_simple_mlp.py`) |
| chunkwise fixture | `param_decomp/experiments/lm/configs/llama8b_l18-26_9layer_chunkwise.yaml` | the 27-site chunkwise CI-fn config `test_config.py` converts |
| ss 2L SimpleMLP | `param_decomp/experiments/lm/configs/ss_llama_simple_mlp-2L.yaml` | current JAX reference for the 2L SimpleStories VPD target (dp=1); reproduces [p-5926d125](https://wandb.ai/goodfire/param-decomp-ss2l-repro/runs/p-5926d125) |
| pile 4L VPD reference | `param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml` | current JAX reference for the VPD paper target; reproduces [p-76082aa1](https://wandb.ai/goodfire/param-decomp/runs/p-76082aa1) |

The toy testbeds (`param_decomp/experiments/tms/configs/`,
`param_decomp/experiments/resid_mlp/configs/`) and the pretrain configs
(`param_decomp/pretrain/configs/`) are separate small schemas, maintained with
their experiments; they are seats too, just not LM-schema ones.

## The pre-JAX archetypes

Three seats reached tip with a `pd.ci_config` asking for `mode: global` /
`fn_type: global_shared_transformer`, a CI function the JAX trainer never
gained (the name survives only in the torch reference,
`nano_param_decomp/run.py`) — not *unmigrated* but **not migratable as
written**. Two are now the JAX reference configs in the table above, each
rewritten onto a `chunkwise_transformer` CI function and revalidated by a
completed run: pile-4L on 2026-07-27, ss-2L on 2026-07-30. Neither reproduces
the paper's own PyTorch decomposition — its Lp importance-minimality objective
is gone from this codebase, so no config can express it.

The third, `jose.yaml` (the original gpt2-arch 4L flagship reference), was
evicted on 2026-07-23 ("Remove unused configs") along with `jose-ish.yaml`
(#917 — the deliberate rewrite of that recipe onto one `chunkwise_transformer`
chunk over all 4 blocks); git history keeps both.

## Rules

1. **Every LM config yaml in the tree parses at tip** — CI-enforced by
   `param_decomp/tests/test_repo_configs_parse.py` (schema parse + the placement
   gate). A schema PR that breaks one migrates
   it **in the same PR**, with an executed in-repo migration (the #966
   pattern) — never a script attached to a PR comment (#939 attached one; it
   never ran, and 97 of 104 stored runs became unopenable before anyone
   noticed).
2. **Sweep / profile / one-off configs are not committed.** Launch them from a
   workspace path. Deleting a config is cheap (git history keeps it; run dirs
   pin what actually ran) — un-rotting one is not.
3. **Adding a canonical seat is taking on a maintenance obligation**: add the
   file, a registry row above naming its purpose, and it's covered by the CI
   gate from that commit on. The cap is **10** (the table seats 8) — at cap,
   the next seat requires an eviction, which is the point. A config a test loads
   is a seat by definition — deleting it is a test change, so grep for the
   basename first.
4. **Stored-run pins are immutable.** Never migrate a run dir's
   `launch_config.yaml` in place (resume byte-compares it; a live old-code run
   whose pin is rewritten refuses its next requeue). Consumers reparse stored
   pins against the full canonical schema
   (`experiments/lm/config.py::load_config`, via `load_run.py`), so a pin from
   an older schema opens at its original revision or through an explicit
   external converter — tip does not migrate it (dataset-name case history
   below).
5. **Seats carry names, never locations** (the portability rule — root
   CLAUDE.md, "Configs are portable"). No absolute path appears in a committed
   config outside a tagged escape arm (`kind: dir`) — CI-enforced by the parse
   gate's `test_seats_carry_names_never_locations`.

## Case history

- **#939** (2026-07-03): ScheduleConfig unification; migration script attached
  as a PR comment, never executed → 7/104 stored runs parseable at tip.
- **#966**: the counter-example — carve migration shipped as an in-repo,
  executed tool covering every live repo yaml.
- **#982**: 25 sweep yamls in one PR — the accumulation pattern this policy
  ends. The sweeps' findings were written up outside the tree; the run dirs pin
  their configs.
- **pretrain dataset-name schema** (2026-07-31): the pretrain seats' absolute
  `data.dir` + duplicated `tokenizer_name` became the same `{kind: name, name}` reference
  the LM decomposition seats use. The caller supplies `data_root`; the dataset store's
  `meta.json` remains the tokenizer authority.
- **dataset-name schema** (2026-07-27): the whole HF-costume `data:` block →
  `data: {kind: name, name} | {kind: dir, dir}` — the block IS the dataset ref
  (2026-07-30: nested to `data: {train, eval}`, each a dataset ref; `eval` is the held-out
  split the eval pass reads, None = eval reads the training shards). The dataset's own facts
  (`seq_len`, `tokenizer_name`) moved into its `meta.json` (`infra.dataset_store.DatasetMeta`),
  read at load; every seat re-stamped in the same PR. Older immutable pins require
  their original revision or an explicit external converter; tip does not migrate them.
- **#23** (2026-07-16): SwiGLU / FFN-as-its-own-config migrated all 30
  `param_decomp/configs/` yamls in one commit — the same 8-line edit applied
  30 times, ~26 of them to launched one-offs nobody will open again. The tax
  this policy stops, paid once more while the policy sat in review.
