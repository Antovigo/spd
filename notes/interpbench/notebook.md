# InterpBench decomposition — working notebook

Running log of findings/observations. Newest objective last.

## Sources of truth
- **Models:** HF Hub `cybershiptrooper/InterpBench` — 86 model dirs, each with `ll_model.pth`
  (weights), `ll_model_cfg.pkl` (`HookedTransformerConfig`), `meta.json`, `edges.pkl`
  (ground-truth circuit edges). Released ids: 82 numeric cases + `ioi` + `ioi_next_token`.
- **Per-model metadata:** `benchmark_cases_metadata.csv` on the HF repo — authoritative; 86 rows,
  full `transformer_cfg.*` + `task_description` + `min/max_seq_len` + `training_args.*`.
- **RASP source:** GitHub `FlyingPumba/InterpBench` → `circuits_benchmark/benchmark/cases/case_N.py`
  (+ `common_programs.py`, `vocabs.py`). Shallow-cloned read-only; NOT installed (heavy
  fork-pinned deps: tracr/JAX, iit, transformer-lens==1.19, acdc, auto-circuit).

## Objective 0 — model catalog + initial pick

Deliverable: **`notes/interpbench/models_catalog.tsv`** (86 rows; columns: name, id, task,
input_format, output_format, n_layers, n_heads, d_model, params_total, params_nonembed,
features, circuit_uses_attn_and_mlp, onehot_in_and_dist_out, is_trivial, comments).
Param counts read directly from the downloaded `ll_model.pth` state dicts (embedding =
`embed.W_E`, `pos_embed.W_pos`, `unembed.W_U`, `unembed.b_U`; non-embedding = the rest).

### Architecture facts (Tracr-derived models, 84 of 86)
- **No LayerNorm.** `normalization_type` is None for all Tracr-derived models; only the 2 IOI
  models use `LNPre`. → conversion to plain `nn.Linear` transformer needs no LN handling.
- `attn_only=False` everywhere (all have MLP blocks). All `model_name="custom"`, `act_fn=gelu`.
- **Heads:** up to 4 per layer (SIIT adds non-circuit heads); residual = `d_head × n_heads`,
  `d_mlp = d_model × 4`. Most models are **2 layers**.
- **Q/K/V are already separate** params in HookedTransformer (`W_Q/W_K/W_V`, each
  `[n_heads, d_model, d_head]`) — not fused. So "split QKV" for Objective 1 just means emit
  three `nn.Linear`s; heads stay together in one matrix. `W_O` is `[n_heads, d_head, d_model]`.
- `d_vocab` (input) ≠ `d_vocab_out` (output) for 84/86 — output vocab is its own size.
- Attention direction split: 65 causal, 21 bidirectional (Tracr-native is bidirectional;
  the causal ones are the post-publication retrain batch). Irrelevant to PD.

### Categorical vs magnitude — classification method + key nuance
- "Magnitude" features = Tracr **numerical** s-ops (value stored as the magnitude of a 1-D
  subspace). Detected = program transitively uses `rasp.numerical(...)` **or** `SelectorWidth`
  (selector width / counts are numerical). Resolved through the call graph
  (`common_programs.py` helpers + case-local `make_*`).
- **Nuance that matters:** arithmetic tasks (sqrt, cube, exp, log, increment, …) are
  **categorical lookup tables, NOT magnitude** — an integer-token `rasp.Map` compiles to a
  one-hot MLP lookup unless explicitly wrapped in `rasp.numerical`. So "operates on numbers"
  ≠ "uses magnitude." These are categorical, but they are **single-`Map` → MLP-only circuits
  (no attention)**, so they fail the "uses both" test.
- Counts: **63 categorical-only**, 21 magnitude, 2 IOI. **81 usable** (one-hot in + vocab-dist
  out); 3 excluded as regression output (cases 3, 4, 39); 11 author-flagged `is_trivial`.

### Shortlist (categorical-only + circuit uses BOTH attention and MLP + non-trivial)
Only **3** models qualify:

| id  | task | L | d_model | params(total/non-emb) | notes |
|-----|------|---|---------|-----------------------|-------|
| 13  | Trend (increasing/decreasing/constant) of numeric tokens | 2 | 20 | 10525 / 10162 | eval-set(T10); bidir; circuit = `shift_by(-1)` (attn) + `SequenceMap` 3-class (MLP); vocab = 3 int digits |
| 19  | Remove consecutive duplicate tokens | 2 | 32 | 26343 / 25604 | eval-set(T10); causal; circuit = `shift_by(1)` (attn) + `SequenceMap` token-or-None (MLP); vocab = 3 ascii letters |
| 110 | Insert zeros between elements, drop latter half | 2 | 20 | 10853 / 10162 | bidir; positional; not in eval set |

Both 13 and 19 are clean 2-operation circuits (1 attention head to shift a neighbour + 1 MLP
to combine) — exactly "uses both, minimally, not trivial." Their non-circuit heads/MLPs (SIIT
keeps them inactive) are a bonus for PD: ground truth should mark them dead.

**Recommended initial model: case 19 (dedup)** — canonical task, in the eval set, largest of
the clean candidates (least degenerate), simple causal shift+compare circuit. case 13 is the
near-identical smaller alternative. **Awaiting user confirmation before Objective 1.**

## Objective 1 — case 19 conversion + data (in progress)

### Exact case-19 structure (from `ll_model_cfg.pkl` + state dict)
- d_model 32, d_head 8, n_heads 4, d_mlp 128, n_layers 2, d_vocab 5 (in), d_vocab_out 3 (out),
  n_ctx 15, act_fn `gelu`, attention_dir `causal`, `use_attn_scale=True`, **no LayerNorm**
  (`normalization_type=None`), `default_prepend_bos=True`, positional `standard`.
- State dict (no LN params): per block `attn.{W_Q,W_K,W_V}` `[4,32,8]` + `b_*` `[4,8]`,
  `attn.W_O` `[4,8,32]` + `b_O` `[32]`, `mlp.{W_in[32,128],b_in,W_out[128,32],b_out}`; plus
  `embed.W_E[5,32]`, `pos_embed.W_pos[15,32]`, `unembed.{W_U[32,3],b_U[3]}`. `mask`/`IGNORE`
  are buffers, not params.
- **Decompose:** the 12 weight matrices = per layer {q,k,v,o, mlp_in, mlp_out} × 2.
  **Do NOT decompose** embed / pos_embed / unembed (one-hot in, vocab-dist out).
- Q/K/V already separate in the state dict → emit separate `nn.Linear`s; unfold each
  `[n_heads, d_model, d_head]` into one `Linear(d_model, n_heads*d_head)` (heads concatenated),
  `W_O[n_heads,d_head,d_model]` into `Linear(n_heads*d_head, d_model)`.

### Token encoding — IMPORTANT deviation from the roadmap
- The saved artifacts have **no tokenizer** (`tokenizer_name=None`). The token→id map lives in
  the tracr-compiled encoder (`hl_model.map_tracr_input_to_tl_input`), which needs tracr/JAX —
  NOT recoverable from the HF files, and the external tracr fork install is blocked.
- **Resolved empirically** (no external code): implemented a faithful folded forward from the
  state dict and brute-forced the id assignment that maximises dedup accuracy. The model only
  behaves correctly under its true training encoding, so this recovers it uniquely.
- **case-19 maps (validated):** input `{BOS:1, PAD:0, a:3, b:2, c:4}`; output class→letter
  `{0:'b', 1:'a', 2:'c'}`. Interior content positions (2–9) decode at **100%**; only the
  BOS-adjacent position 1 is imperfect (~67%) — intrinsic model edge behavior (the tracr
  target at the first content token is ambiguous), not a bug.
- Consequence: per-case encoding maps must be obtained once (empirically here). Generalising to
  other cases needs either tracr (blocked) or the same brute-force per case (only tractable for
  small categorical vocabs). Flag for the user.

### Faithful forward (validated, pure torch, no TL/tracr)
resid = `W_E[ids] + W_pos[:T]`; per block: causal MHA (scale `1/sqrt(d_head)`, per-head bias
`b_*[:,None,:]`, softmax, `W_O` + `b_O`, residual) then MLP (`gelu(x@W_in+b_in)@W_out+b_out`,
residual); logits = `resid @ W_U + b_U`. No LN, no final norm. `F.gelu` (erf) — `gelu`/`gelu_new`
identical here. This is the conversion reference: assert nn.Linear-unfolded forward == this.

### Carry-forward for Objective 1
- Convert HookedTransformer → plain `nn.Linear` transformer: emit separate q/k/v/o Linears
  (heads together), MLP in/out Linears; **no LayerNorm** for Tracr cases; **do NOT decompose
  embed/unembed** (one-hot in, vocab-dist out). New target `kind` in `lm/run.py`.
- Data: vendor `circuits_benchmark` vocab + sampler only (no install); encode via the loaded
  model's own tokenizer; no labels needed (KL self-supervises). For case 19/13: `shift_by`
  uses `Select(indices,indices,q==k±offset)` + `Aggregate` → confirms a real attention head.

### Objective 1 — DONE (case 19)
New module `param_decomp_lab/experiments/lm/interpbench/`:
- `model.py` — `InterpBenchTransformer` (`nn.Linear` q/k/v/o + mlp, embed/pos/unembed kept
  whole) + `from_hf(case_id)` that downloads, unfolds folded HookedTransformer weights, freezes.
  General over Tracr-derived cases (asserts no LayerNorm → IOI unsupported, as intended).
- `data.py` — vendored vocab + `[BOS]+content+PAD` sampler + `CASE_SPECS` registry (case 19
  validated maps) + `InterpBenchDataConfig`. Decode/task helpers reused by the test.
- `interpbench_19.yaml` — reference decomposition config (12 matrices, C 32/96).
- `test_interpbench.py` (slow, downloads from HF) — (1) converted == folded reference fp64;
  (2) interior-position task accuracy ≥ 0.99.

`lm/run.py` edits (minimal): `InterpBenchTarget` added to `LMTargetSpec` (carries `case_id`,
`n_samples`, and a fixed `model_class` so existing `spec.model_class` consumers still
type-check); `build_target` dispatches to `InterpBenchTransformer.from_hf`; `build_lm_loader`
dispatches on the target kind to `make_interpbench_loader`. **`LMDataConfig` is left
unchanged** — interpbench YAMLs put placeholder `dataset_name`/`tokenizer_name` in the
`data:` block (the synthetic loader ignores them). A `model_validator` enforces
`output_extract: null` for interpbench. Run via
`pd-lm param_decomp_lab/experiments/lm/interpbench/interpbench_19.yaml`.

Why placeholders, not a data union (code-review finding): making `data` a discriminated
union broke reload of every existing config (old `data:` blocks have no `kind` tag) and
widened `cfg.data` / `cfg.target.spec` enough to fail repo-wide `basedpyright` in 5
unrelated files (adapters/app/editing). The placeholder approach keeps the shared types
intact → zero downstream churn, no reload break. Verified: repo-wide basedpyright 0 errors;
full suite (421) green; CPU smoke decomposition runs (12 decomposed modules).

Deviation noted: the HF artifacts carry NO tokenizer, so per-case encoding maps must be
supplied (recovered empirically here, not via the roadmap's "loaded model's tokenizer").
Generalising data-gen to other cases needs their maps (tracr, or brute-force per case).

## Objective 2 — hyperparameter tuning (in progress)

Goal: PGDReconLoss < 0.02 (eval, 20-step PGD adversary) with L0 as low as possible, in
< 15k steps. Recipe scaled down from `pile_llama_simple_mlp-4L.yaml`: C=100 on all 12
matrices, the four PGD losses (ImportanceMinimality / StochasticReconSubset /
PersistentPGDRecon / Faithfulness), layerwise `vector_mlp` CI fn. **Runs on GPU via SLURM**
(`~/pd_scratch/case19_*.yaml` + `tune_case19_driver.py` + `case19_run.sbatch`; driver streams
eval metrics through a capture sink, no wandb). The CEandKLLosses eval metric was dropped —
it does CE-vs-labels but case 19's output vocab (3 classes) ≠ input vocab (5), so label ids
3/4 trip a CUDA `t < n_classes` assert. Decomposition is self-supervised on the target's own
logits (KL), so labels are irrelevant. Also added `drop_last=True` to `make_interpbench_loader`
(repo): the finite dataset's partial last batch (20000 % 256 = 32) breaks PersistentPGD, whose
per-position mask state requires uniform batch size.

### Round 1 — impmin frontier (C=100, lr 1e-3, 10–12k steps)
Random-subset reconstruction is essentially solved everywhere (`StochasticReconSubsetLoss`
~0.01). The hard metric is the **adversarial** PGD recon. Floors reached:

| impmin   | PGDRecon floor | L0 total @ floor | note |
|----------|----------------|------------------|------|
| 3e-4 (baseline) | ~0.13 (osc 0.13–0.17) | ~470 | sparsity pressure *raises* PGDRecon |
| 1e-4     | ~0.10 (extrap)  | ~790 | (cancelled early — intermediate) |
| 5e-5     | ~0.08           | ~800 | lowest impmin, still 4× target |

**Key finding:** PGDRecon and L0 trade off (lower impmin → lower PGDRecon, higher L0), but the
PGDRecon floor is **~0.08 even at near-zero sparsity pressure** — structural, not set by impmin.
So impmin alone cannot reach 0.02. Mechanism: eval adversary optimises `source` in
`mask = ci + (1-ci)·source`; PGDRecon is low only when the CI fn is **sharp** (ci≈1 for needed
comps, ≈0 else). The 0.08 floor ⇒ CI fn under-confident. Prime suspect: lr 1e-3 is 20× the 4L
recipe's 5e-5, too coarse to sharpen the CI fn.

### Round 2 — sharpness levers (impmin 1e-4)
Tested three orthogonal levers against the ~0.05–0.08 adversarial floor. PGDRecon @ step 3000:

| variant | change | PGDRecon @3k | note |
|---------|--------|--------------|------|
| `lowlr` | lr+ci_lr 2e-4, 15k steps | 0.19 | low LR converges too slowly; not winning |
| `pgd2`  | PersistentPGDReconLoss coeff 0.5→2.0 | **0.076** | best single lever — stronger adversarial training |
| `bigci` | CI hidden_dims [32]→[64,64] | (cancelled) | folded into combo |

**`pgd2` is the decisive lever:** training the components harder against the persistent-PGD
adversary directly lowers the eval PGD-recon floor. lowlr's slow convergence isn't worth it at
this LR; bigci alone untested (folded into combo).

### Round 3 — strong-PGD frontier (jobs 2469 pgd2, 2476 pgd4, 2471 combo)
`pgd2` (impmin 1e-4, PersistentPGD coeff 2.0, lr 1e-3, 12k) reached PGDRecon **~0.025–0.028**,
L0 ~700 — *near* but not robustly under 0.02 (eval PGDRecon is noisy, ±0.005, random adversary
init). `pgd4` (coeff 4.0) reached ~0.03 by step 3k but with even **higher** L0 (~900): stronger
PersistentPGD lowers the floor yet keeps more components alive (they're needed to survive
ablation), so PGD-coeff trades against L0 just like impmin. With layerwise `vector_mlp` CI the
best achievable was ~0.025 floor at L0 ~700 — the L0 stayed far from the "~dozen/matrix" ideal.

**Root cause (user-flagged): wrong CI fn.** I had matched the 4L *loss* recipe but kept the
old interpbench `layerwise vector_mlp` CI fn instead of the 4L's `global_shared_transformer`.
The PGD-recon floor is gated by CI-mask sharpness (`mask = ci + (1-ci)·source`; the adversary
wins wherever ci is mushy), and a per-matrix vector_mlp is too weak to produce sharp,
context-dependent masks — hence the stuck floor and high L0.

### Round 4 (v2) — corrected recipe ⇒ breakthrough (jobs 2479/2480/2482)
Three user-suggested fixes, all on the CI side:
1. `global_shared_transformer` CI fn (d_model 256, n_blocks 4, n_heads 4, mlp [1024], max_len 16)
   — scaled-down 4L CI fn, replaces layerwise vector_mlp.
2. ImportanceMinimalityLoss `beta` 0.5 → **0**.
3. components LR (1e-3) **>** CI-fn LR (3e-4) — keeps masks from thrashing (cf. team note:
   CI-fn LR below components LR → sparser, cleaner components).
Kept: PersistentPGD coeff 2.0, 20k steps, C=100×12.

Result @ step 3000 (17k still to go): PGDRecon converges **~5× faster** and both land under 0.02.

| variant | impmin | PGDRecon @3k | L0 @3k | subset @3k |
|---------|--------|--------------|--------|------------|
| v2a | 1e-4 | 0.0192 | 916 | 0.0016 |
| v2b | 3e-4 | **0.0173** | 821 | 0.0035 |
| v2c | 1e-3 | (running) | — | — |

v2b (higher impmin) is under 0.02 *and* lower L0. Pushed impmin further (v2c=1e-3) to map the
L0 floor under the PGD constraint.

**Final floors (20k steps):**

| variant | impmin | final PGDRecon | L0 total | tail stability |
|---------|--------|----------------|----------|----------------|
| v2a | 1e-4 | ~0.007 | ~880 | (cancelled @6k to free GPU; recon already 0.007) |
| v2b | 3e-4 | **0.0068** | **684** | rock-solid (tail 0.006–0.009, never spikes) |
| v2c | 1e-3 | **0.0123** | **530** | mostly 0.012–0.015, one noisy spike to 0.030 @15k |

Higher impmin → lower L0 with the constraint still met, but L0 is **sticky**: 10× impmin (1e-4→1e-3)
only moved L0 880→530. The SIIT model genuinely spreads mass across many components (its
non-circuit heads/MLPs absorb some), so ~44/matrix is the realistic floor here, well above the
naive "~dozen/matrix" guess. L0 is dominated by layer 0 (442 of 684 for v2b) — the dedup circuit
(shift head + compare MLP) lives in layer 0; layer 1 is lighter (~242).

### Objective 2 — DONE
**Chosen recipe → `interpbench_19.yaml`: v2c (impmin 1e-3).** By the roadmap's criterion (final
PGDRecon < 0.02, then L0 as low as possible under it), v2c wins: **PGDRecon 0.012, L0 530** vs v2b's
0.007 / 684 (−22% L0). v2b (impmin 3e-4) is the conservative alternative if a bulletproof
sub-0.02 reproduction matters more than L0 — bump `coeff` 0.001→0.0003 to switch.

The corrected recipe vs the Objective-1 placeholder: **global_shared_transformer CI fn** (was
layerwise vector_mlp — the main fix), **ImportanceMinimality beta 0** (was 0.5),
**components LR 1e-3 > CI-fn LR 3e-4** (was equal), **PersistentPGDReconLoss coeff 2.0** (was 0.5),
20k steps. The CI-fn swap was decisive: PGDRecon floor 0.05–0.13 (layerwise) → <0.02 (global),
converging ~5× faster. Runs on GPU via SLURM (~15–35 min each depending on node co-location).

Repo code touched (minimal): `data.py` gained `drop_last=True` (finite dataset's partial last
batch breaks PersistentPGD's per-position mask state). CEandKLLosses is unusable as an eval metric
here (CE-vs-labels assumes output vocab = input vocab; case 19 has 3 vs 5 → CUDA index assert).

### Round 5 — sparsity investigation (PGD limit relaxed 0.02 → 0.1)
The v2 L0 (~530) was far above the "~dozen/matrix" expectation. Investigated by cranking impmin
(the v2 recipe otherwise fixed: global CI, beta 0, LR ratio, PersistentPGD 2.0, 20k). **Finding:
L0 is NOT structurally stuck — I just hadn't pushed impmin hard enough.** Frontier (@ step 6000,
still converging on the LR tail):

| impmin | L0 total | PGDRecon | under 0.1? |
|--------|----------|----------|------------|
| 1e-3 (v2c) | 530 | 0.012 | yes (strict 0.02) |
| 3e-3 | 504 | 0.052 | yes |
| 1e-2 | 289 | 0.071 | yes — safe sparsest |
| 2e-2 | (running) | — | testing sweet spot |
| 3e-2 | 126 | 0.136 | no (recon overshoots) |

Two orthogonal findings: (a) **weak PGD is worse** — impmin 1e-2 with PersistentPGD coeff 0.5
gave recon 0.94 vs 0.20 at the same L0, so keep coeff 2.0; (b) **L0 stays layer-0-dominant**
(~2:1 over layer 1) at every sparsity — consistent with the dedup circuit (shift head + compare
MLP) living in layer 0.

**Full frontier (near-final, step ~13–14k of 20k):**

| impmin | L0 total | PGDRecon | comp/matrix | verdict |
|--------|----------|----------|-------------|---------|
| 1e-3 (v2c) | 530 | 0.012 | 44 | strict-0.02 pick |
| 1e-2 | 152 | 0.050 | 13 | safe, robustly <0.1 |
| 2e-2 | ~110 | ~0.09 | 9 | middle |
| **3e-2** | **~55–66** | **~0.08** | **~5** | **chosen (sparsest <0.1)** |

**C is not the sparsity lever:** the `C=30` diagnostic (impmin 1e-3) plateaued at L0 ~220 (~18/matrix)
— capping C just bounds redundancy, it doesn't *force* it out. impmin does. So keep C=100 and let
impmin drive sparsity.

### Objective 2 — DONE (final)
**Answer to "why was L0 so high": impmin was simply too weak.** With the corrected CI fn, L0 is a
smooth function of impmin — 530 (1e-3) → 55 (3e-2) — bounded below only by the recon budget.

**`interpbench_19.yaml` set to impmin 3e-2:** L0 ~55–66 (~5 alive components/matrix, near the
"~dozen or fewer" ground-truth expectation), eval PGDReconLoss ~0.08 (< the relaxed 0.1 limit),
layer-0-dominant (the dedup circuit). Recipe is otherwise the v2 stack (global_shared_transformer
CI, beta 0, comp LR 1e-3 > CI LR 3e-4, PersistentPGD coeff 2.0, 20k steps, C=100).
**Alternative:** impmin 1e-3 for a strict recon < 0.02 at L0 ~530 (bump `coeff` back).

**Definitive run:** `pd-lm` on the repo config → run folder **`runs/p-d1e2260e/`** under
`PARAM_DECOMP_OUT_DIR` (checkpoint + metrics.jsonl + figures) for downstream pipelines.
**Final (step 20k): PGDReconLoss 0.058, L0 total 50.4** (layer 0 37.0, layer 1 13.4) — under the 0.1
limit with room to spare, reproducing the s_imp3e2 sweep. Checkpoint `model_20000.pth` (+
`training_20000.pth` for resume).

**Per-matrix L0 validates the ground truth.** The decomposition concentrates almost entirely in
layer 0 (the dedup circuit) and kills layer-1 attention:

| | q | k | v | o | mlp_in | mlp_out |
|---|---|---|---|---|---|---|
| **layer 0** | 5.8 | 6.9 | 5.3 | 4.1 | 7.5 | 8.1 |
| **layer 1** | 1.1 | 0.8 | 1.0 | 1.3 | 6.0 | 3.6 |

Layer-1 attention is ~1 component/matrix (effectively dead) — exactly what `edges.pkl` says (dedup
= one layer-0 head + one layer-0 MLP; layer 1 + non-circuit heads dead). This is the "~dozen or
fewer per matrix" the roadmap expected, and it maps cleanly onto the known circuit. Residual
layer-1 MLP mass (~6/3.6) is the only sizeable off-circuit component set — a candidate for the
ground-truth check to scrutinise.
