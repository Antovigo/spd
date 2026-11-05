# Distributed Alignment Search on the arithmetic decomposition's target: plan

Goal: for the operands and the result of `a+b=` and `a-b=`, find in Llama-3.1-8B (the "original model") which
positions and stream points hold each variable, how many dimensions each position needs, and the subspaces themselves.

Scope: original model only. Position flags, the operation and the integer-only model are out of scope for now. The
trained subspaces will be tested and fine-tuned on the integer-only model later, as a separate step.

## 1. Notation

| Symbol | Meaning | Type and range |
|---|---|---|
| `$RUN` | The run folder, `~/out/pod-backup/p-ba8b0c08` | path |
| `$REPO` | The worktree holding this plan and the scripts, `~/Code/param-decomp/das` (branch `experiment/das`) | path |
| `a`, `b` | Operands of the base prompt | integers, 1..400 |
| `a'`, `b'` | Operands of the source prompt | integers, 1..400 |
| `a''` | First operand of a third prompt, used only by the diagnostic in section 7.4 | integer, 1..400 |
| `r` | The result: `a+b` on the addition task, `a-b` on the subtraction task | integer, 1..800 |
| `v` | A variable: `a`, `b` or `r` | one of three names |
| `p` | Token position: 0 = `<BOS>`, 1 = `a`, 2 = operator, 3 = `b`, 4 = `=` | integer, 0..4 |
| `l` | Block index | integer, 0..31 |
| `t` | Stream point: `t = 2l` is the raw residual stream before block `l`'s attention norm, `t = 2l + 1` before its MLP norm | integer, 0..63 |
| `d` | Stream width, 4096 | integer |
| `P_v` | The positions patched for variable `v` (section 4) | set of positions |
| `k` | Subspace dimension | integer, 0..64 |
| `k_p` | Subspace dimension at position `p` | integer, 0..64 |
| `R_p` | Subspace basis at position `p`, a `d`×`k_p` matrix with orthonormal columns | real matrix |
| `h_p`, `h'_p` | Base and source stream vectors at position `p` and the current `t` | real vectors of length `d` |
| `D` | Source-minus-base matrix at one `(t, p)`, with one row `h'_p - h_p` per training pair | real matrix, pairs × `d` |
| `q` | Model output renormalised over the tokens of the integers 0..999 | probability vector of length 1000 |
| IIA | Interchange intervention accuracy: the fraction of test pairs whose patched top integer token, the argmax of `q`, is the expected answer | fraction, 0..1 |

`t = 37` is the paper's "pre-MLP" point at layer 18. `t = 38` is its "post-MLP" point at layer 18. The output of block
31 (the paper's post-MLP point at layer 31) is not run.

## 2. Decisions to check before implementation

These are the choices I made while drafting. Each value is a key in `$REPO/notes/das/configs/original.yaml` (section 9).

1. **Stopping rule for one position.** You asked to lower `k` until IIA no longer beats chance. That works when all
   positions share one `k` (section 7.2). It does not work when lowering one position in a whole-layer patch, because
   the other positions keep IIA above chance even after that position has lost most of its contribution. So the
   per-position search (section 7.3) stops when whole-layer IIA falls more than `position_margin` = 0.02 below the
   reference where every `k_p` = 64.
2. **The chance rule measures any signal, not the whole variable.** A low `k` can beat chance while recovering only a
   few percent of answers. So each site also reports `k_95`: the smallest `k` with IIA at least 95% of the IIA at
   `k` = 64.
3. **`min_iia_for_position_search` = 0.10.** Sites whose IIA at `k` = 64 beats chance but stays below this skip the
   per-position search. A 0.02 margin means little against a very low reference.
4. **One pair set per task, shared by the three variables.** It requires all four prompts to be correct, which costs
   sampling on subtraction (section 5) but gives one SVD per task and allows direct overlap tests.
5. **Objective:** the KL divergence in section 6.
6. **Queue patience:** `slurm_wait_minutes` = 30 before moving to a pod (section 10).

## 3. What the paper does

Feucht et al. 2026, "Arithmetic in the Wild" (https://arxiv.org/html/2605.01148v1, code at
https://github.com/goodfire-ai/arithmetic-wild):

| Item | Value |
|---|---|
| Intervention | `h = h_o + R Rᵀ (h_c - h_o)`, with `h_o` the original state, `h_c` the counterfactual state and `R` a `d`×`k` matrix with orthonormal columns |
| Sites | Last token position, every layer, after attention and after the MLP |
| Pairs | Two independently sampled prompts, both from prompts the model answers correctly: 3,584 train, 512 test |
| Hyperparameters | Learning rate 1e-4, batch 16, 8 epochs, chosen from learning rate {1e-4, 1e-3, 1e-2}, batch {16, 32, 64}, epochs {4, 8} |
| Dimensions | `k` from 1 to 8, extended to 16 if IIA never passes 90%. The best `k` is reported, usually the largest; no minimum is searched for |
| Initialization | Random within the top principal components explaining at least 90% of the variance |
| Objective | Cross-entropy against the causal model's answer |
| Addition at layer 18, operands 1..100 | IIA 61.7% for the first operand and 63.8% for the second (`k` = 16 and 15, pre-MLP), 96.6% for the sum (`k` = 16, post-MLP) |

Differences here: operands go to 400, subtraction is added, all positions are patched together, `k` starts at 64 and
descends, and the initialization and objective differ (sections 6 and 8).

## 4. Interchange interventions

Base `(a, b)` and source `(a', b')` are sampled independently, so the source differs in both operands. This is what
separates the variables: each one predicts a different answer.

| Variable | Addition: expected answer | Subtraction: expected answer | Expected-answer prompt | `P_v` |
|---|---|---|---|---|
| `a` | `a' + b` | `a' - b` | `(a', b)` | {1, 2, 3, 4} |
| `b` | `a + b'` | `a - b'` | `(a, b')` | {3, 4} |
| `r` | `a' + b'` | `a' - b'` | `(a', b')`, the source itself | {3, 4} |

`P_v` follows from causality. A position sees only tokens at or before it, so `b` cannot be at positions 1 or 2, and `r`
needs both operands. Position 0 is identical in every prompt.

The patch is applied at one `t`, at every position of `P_v` with `k_p` > 0, with a separate `R_p` per position:

`h_p <- h_p + R_p R_pᵀ (h'_p - h_p)`

Blocks after `t` then run normally. Positions that have no base-source difference at that `t` are removed from `P_v`.
For example, at `t = 0` the stream is the token embedding, so positions 2 and 4 never differ.

### Why this handles repeated copying

A patch at one position fails if later attention re-reads the base value from another position. Our earlier interventions
(`param_decomp/arith_repr/isa/interventions.py`) found later heads re-reading operands at their own positions until
about block 24. Everything after `t` reads only the stream at `t`, so patching all positions where a variable can exist
leaves no unpatched copy for a later block to re-read. Giving each position its own `R_p` keeps one shared subspace from
having to hold every copy, and it answers which positions matter: a position with minimal `k_p` = 0 is not needed.

### Limits of the minimal `k_p` reading

- **Redundant copies have more than one minimum.** If either of two positions is enough, each can be dropped alone but
  not both, so the result depends on the search order. Section 7.3 runs two orders and reports both when they disagree.
- **`k_p` = 0 means "not needed given the other positions".** It does not mean the variable is absent there. Section
  7.4 adds the single-position and leave-one-out patches that separate these cases.
- **`r` can be imitated by the operands.** Patching subspaces holding `a` and `b` together also yields `a' + b'`, so a
  high `r` IIA before the result is computed does not by itself show the result is stored. Restricting `P_r` to {3, 4}
  limits this because positions 1 and 2 stay unpatched, but copies at positions 3 and 4 remain possible. Section 7.4
  checks it with overlap and a double patch.

## 5. Pairs

Requirements, on the prompts of each task:

- Subtraction uses only `a > b`, and every expected answer must be at least 1.
- The base, the source and all three expected-answer prompts `(a', b)`, `(a, b')` and `(a', b')` must be answered
  correctly: the top integer token is the true answer.
- The four answers `r`, `a' + b`, `a + b'` and `a' + b'` (with `-` on subtraction) must be pairwise distinct.

Measured from `$RUN/analysis/pruning/step_40000/addsub-08-prune/references.npz`:

| | Addition | Subtraction |
|---|---|---|
| Correct prompts | 145,246 of 160,000 | 52,802 of 79,800 |
| Share of sampled correct base-source pairs meeting every requirement | 87.0% | 33.8% |
| Mean `a` after the requirements / among correct prompts | 220.3 / 212.6 | 277.1 / 261.5 |
| Mean `b` after the requirements / among correct prompts | 189.1 / 192.6 | 144.7 / 150.5 |

Share of accepted pairs by digit count:

| Quantity | Addition: 1 / 2 / 3 digits | Subtraction: 1 / 2 / 3 digits |
|---|---|---|
| `a` | 1.3% / 14.3% / 84.4% | 0.0% / 2.1% / 97.9% |
| `b` | 2.5% / 24.4% / 73.0% | 1.2% / 36.7% / 62.1% |
| `r` | 0.0% / 2.7% / 97.2% | 3.7% / 39.7% / 56.7% |

The split is on prompts: 20% of correct prompts are test prompts, and a pair belongs to a split only if its base, source
and expected-answer prompts all do. Training starts at 3,584 pairs, subject to the pilot.

The test set has 4,096 uniformly sampled pairs, which is 8 times the paper's 512 and needed for the chance test. It is
topped up so that each non-empty combination of digit counts of `a` and `b` has at least 256 pairs where the pool allows
it. Headline IIA uses the uniform part only. IIA is also reported by digit count of `a`, `b` and the expected answer.

## 6. Objective

Train on `KL(q_target || q_patched)`, where `q_target` is the model's own `q` on the expected-answer prompt and
`q_patched` is its `q` after the patch.

This is more principled than the paper's cross-entropy against the answer token:

- The true effect of setting `a` to `a'` is available: it is the run on prompt `(a', b)`. The KL asks the patch to
  reproduce that behaviour, and a perfect interchange reaches exactly 0.
- Cross-entropy cannot reach 0 even for a perfect interchange. The original model's own integer cross-entropy on these
  prompts is 0.80 on addition and 2.10 on subtraction, so its gradient keeps rewarding answers sharper than the model
  ever gives.
- The full distribution gives more signal per pair than a single label.

The restriction to integer tokens is the part of this choice most open to argument. Mean integer mass is 0.85 on
addition and 0.76 on subtraction, so a full-vocabulary KL is more faithful to the model as a whole. I chose the integer
version because the variables only say which integer comes out, and because the integer-only model was selected on the
same quantity (`kl_int`), so the later test uses one objective.

Metrics per fit, on test pairs: IIA (the selection metric), the same IIA over the full vocabulary, the KL, the paper's
cross-entropy, and mean integer mass. The pilot also trains the paper's cross-entropy at the pilot sites for a direct
comparison.

## 7. Procedure

### 7.0 Collection

Run the original model on all 320,000 prompts. Save `q` in log form, the top integer, and correctness. Check: per-prompt
correctness equals `references.npz` on its 239,800 prompts, with 145,246 and 52,802 correct.

Build the pairs (section 5). Then compute the SVD initialization: for each task, `t` and `p` in {1, 2, 3, 4}, take the
SVD of `D` over the first 3,584 training pairs, without centring, and keep the top 64 right singular vectors. Pairs are
shared across variables, so this is one initialization per task.

### 7.1 Tests and pilot

No GPU fit starts until these pass:

- Running from any `t` reproduces the full forward.
- `R_p` = identity at all of `P_a` makes the output equal the source's own at every `t`.
- With every `k_p` = 0 the output equals the base's own.
- Pair requirements hold on every saved row.

The pilot uses four sites at every `k_p` = 64: `a` at `t = 37` and `r` at `t = 38`, on each task.

| Pilot sweep | Values | Fits |
|---|---|---|
| Hyperparameters, at 3,584 pairs | Learning rate {1e-4, 1e-3, 1e-2}, batch {16, 32, 64}, epochs {4, 8} | 18 × 4 = 72 |
| Training pairs, at the chosen hyperparameters | 3,584, 14,336, 57,344 | 2 extra × 4 = 8 |
| Objective, at the chosen hyperparameters | Paper's cross-entropy | 1 × 4 = 4 |

The pilot also measures seconds per fit against `t` and the number of fits one GPU can run in parallel.

**Review point:** I report the pilot results and the projected cost of 7.2 and 7.3 before continuing. The upper bound if
every site passes is 384 screen fits, 4,224 uniform-descent fits and 25,600 per-position fits. If the projection is too
high, the options are stepping `t` by 2 or raising `min_iia_for_position_search`.

### 7.2 Screen and uniform descent

Ladder: 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1.

**Screen.** For each of the 6 task-variable combinations and all 64 values of `t`, train with every position of `P_v` at
`k_p` = 64. That is 384 fits.

**Chance.** Five random `R_p` draws per configuration on the same test pairs, untrained. A fit beats chance when the
lower 95% Wilson bound on its IIA exceeds both the upper 95% Wilson bound on the pooled random-draw IIA and 1/1000.

**Uniform descent.** At each site that beats chance, step down the ladder with one `k` shared by all positions, and stop
at the first `k` that does not beat chance. Each site reports its whole IIA curve, the lowest `k` that beats chance, and
`k_95`.

### 7.3 Per-position minimum

At each site with IIA at `k` = 64 of at least `min_iia_for_position_search`. The reference is whole-layer IIA at every
`k_p` = 64. Every fit retrains all remaining `R_p` from the SVD initialization.

1. **Order the positions.** Train with each position set to `k_p` = 0 in turn, and sort by IIA loss, smallest first.
2. **First order.** Go through the positions in that order. For each, step down the ladder and then to 0, holding the
   others at their current values, and keep the lowest `k_p` whose IIA is within `position_margin` of the reference.
3. **Second order.** Repeat in the reverse order.
4. **Report.** Both minimal tuples of `k_p`, and whether they agree.

### 7.4 Diagnostics at the minimal tuples

None of these needs training, except the 3 seeds.

- **Which positions suffice.** IIA with each `R_p` patched alone and with each left out, with no retraining.
- **Stability.** Refit each minimal tuple with 3 seeds. Initialization is deterministic, so seeds change pair order and
  batches. Report the IIA spread and the principal angles between seeds.
- **`r` against the operands.** At each `(t, p)` where both are found, the overlap of the `r` subspace with the span of
  the `a` and `b` subspaces, using the paper's measure: the mean singular value of `AᵀB`, with `A` and `B` orthonormal
  bases of the two subspaces, 0 for orthogonal and 1 for contained.
- **Double patch.** Patch `r` from `(a', b')` and then `a` from a third prompt with first operand `a''`, at the same `t`.
  Report the share of top answers equal to `a' + b'` (the `r` patch wins) and to `a'' + b'` (the operands are still used).
  Use `-` in place of `+` on subtraction.

## 8. Fit specification

| Setting | Value |
|---|---|
| Parametrisation | One unconstrained `d`×`k_p` matrix per position, with `R_p` its Q factor from QR at each step |
| Initialization | Top `k_p` right singular vectors of `D`, per section 7.0 |
| Optimizer | Adam. The paper's text does not name its optimizer |
| Starting hyperparameters | Learning rate 1e-4, batch 16, 8 epochs, 3,584 pairs, replaced by the pilot's choice |
| Model precision | `bf16` matmuls with fp32 accumulation, as in `param_decomp/pruning/model.py` |
| Model weights | Frozen |

## 9. Files

Each path below, other than the scripts, is a key in `$REPO/notes/das/configs/original.yaml`, so editing that file moves it. Scripts read paths only from there.
Paths written with `{task}` use `addition` and `subtraction`; `{variable}` uses `a`, `b` and `r`.

### 9.1 Specification files, to be written

| File | Contents |
|---|---|
| `$REPO/notes/das/configs/original.yaml` | Every path, hyperparameter, ladder, threshold and pilot grid |
| `$REPO/notes/das/configs/variables.yaml` | For each task and variable: the expected-answer prompt, the answer rule and `P_v` (section 4) |

### 9.2 Existing inputs, read only

| File | Used for |
|---|---|
| `$RUN/prompts/tokenized.npz` | Tokens, `a`, `b` and the operation for 320,000 prompts |
| `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/` | Weights, `config.json` and `tokenizer.json` |
| `$RUN/analysis/pruning/step_40000/addsub-08-prune/references.npz` | Check of per-prompt correctness only |

### 9.3 Collected datasets

Rows align with `$RUN/prompts/tokenized.npz`, as in the other `datasets/` folders.

| File | Contents |
|---|---|
| `$RUN/datasets/answers/README.md` | Description |
| `$RUN/datasets/answers/integer_log_probs.npy` | `(320000, 1000)` float32, 1.28 GB: log of `q` on each prompt, used as `q_target` |
| `$RUN/datasets/answers/predicted_integer.npy` | `(320000,)` int16: argmax of `q` |
| `$RUN/datasets/answers/correct.npy` | `(320000,)` bool: answer at least 1 and top integer correct |
| `$RUN/datasets/answers/metadata.json` | Commit, precision and check results |
| `$RUN/datasets/das/README.md` | Description of everything under `datasets/das` |
| `$RUN/datasets/das/pairs/{task}.npz` | Row indices of the base, source and three expected-answer prompts; the base and three expected answers; split; whether a test pair is in the uniform or top-up part |
| `$RUN/datasets/das/init/{task}.npz` | `singular_vectors` `(64, 4, 64, 4096)` float32, 268 MB: stream points, positions 1 to 4, rank, `d`. Also `singular_values` `(64, 4, 64)` |

### 9.4 Outputs

| File | Contents |
|---|---|
| `$RUN/datasets/das/fits/{task}_{variable}.jsonl` | One row per fit, including the chance draws: stage, `t`, each `k_p`, seed, hyperparameters, all metrics of section 6, IIA by digit count, seconds |
| `$RUN/datasets/das/subspaces/{task}_{variable}/t{t}_k{k1}-{k2}-{k3}-{k4}_s{seed}.npz` | `R_p` per position, with `t`, position labels and test metrics. `k1` to `k4` are `k_p` at positions 1 to 4. Saved for the screen, the uniform descent and the minimal tuples; intermediate per-position fits keep only their row in `fits` |
| `$RUN/datasets/das/summary/{task}_{variable}.json` | Per `t`: IIA at `k` = 64, chance level, lowest `k` beating chance, `k_95`, both minimal tuples and whether they agree |
| `$RUN/datasets/das/position_patches/{task}_{variable}.jsonl` | Single-position and leave-one-out IIA at the minimal tuples |
| `$RUN/datasets/das/overlap/{task}.npz` | `r` against operand overlap per `(t, p)`, and principal angles between seeds |
| `$RUN/datasets/das/double_patch/{task}.jsonl` | Double-patch shares per `(t, p)` |
| `$RUN/analysis/das/pilot/pilot.jsonl` | One row per pilot fit |
| `$RUN/analysis/das/pilot/cost.json` | Seconds per fit by `t`, parallel fits per GPU and projected totals |
| `$RUN/analysis/das/logs/` | SLURM or pod logs |
| `$REPO/notes/das/report.md` | Report |
| `$REPO/notes/das/figures/` | PNGs embedded in the report |

### 9.5 Scripts, all in the repo

The repo's `CLAUDE.md` keeps scheduler code and cluster paths out of `param_decomp/`, so the configurations and launch
scripts live under `notes/das/`, and the library code takes paths as required arguments. The new test folder must also
be added to a CI shard (`param_decomp/tests/infra/test_ci_shards.py`).

| File | Role |
|---|---|
| `$REPO/param_decomp/das/__init__.py` | Package marker |
| `$REPO/param_decomp/das/config.py` | Load and validate both YAML files |
| `$REPO/param_decomp/das/forward.py` | Original-model forward resumable at any `t`, reusing `DenseWeights` and the block code of `param_decomp/pruning/model.py`. `run_blocks` starts only at block boundaries, so this adds the odd `t`; written so the integer-only model can use it later |
| `$REPO/param_decomp/das/collect_answers.py` | Section 7.0 collection |
| `$REPO/param_decomp/das/pairs.py` | Section 5 |
| `$REPO/param_decomp/das/init_svd.py` | SVD initialization |
| `$REPO/param_decomp/das/fit.py` | One fit: patch, loss, QR, metrics, chance draws; runs many fits in parallel on one GPU |
| `$REPO/param_decomp/das/pilot.py` | Section 7.1 pilot |
| `$REPO/param_decomp/das/search.py` | Sections 7.2 and 7.3: work queue, one worker per GPU. Skips fits already in `fits`, so a preempted job resumes |
| `$REPO/param_decomp/das/diagnostics.py` | Section 7.4 |
| `$REPO/param_decomp/das/figures.py` | Report figures |
| `$REPO/notes/das/slurm/collect.sbatch` | Collection, SVD: 1 L40 |
| `$REPO/notes/das/slurm/pilot.sbatch` | Pilot: 1 L40 |
| `$REPO/notes/das/slurm/search.sbatch` | Search and diagnostics: 4 L40 |
| `$REPO/notes/das/pod/setup.sh` | Pod environment, and copying the inputs of section 9.2 and the datasets |
| `$REPO/notes/das/pod/sync.sh` | Copy outputs back to `$RUN` |
| `$REPO/param_decomp/tests/das/test_forward.py` | Forward tests of section 7.1 |
| `$REPO/param_decomp/tests/das/test_patch.py` | Patch tests of section 7.1 |
| `$REPO/param_decomp/tests/das/test_pairs.py` | Pair requirements |

## 10. Compute

- All of this runs through SLURM or on a pod; only editing and submitting happen on the dev node. Scratch goes to
  `/ephemeral/$USER`.
- Use `sbatch` on the free `compute` partition with 4 L40s if the job starts within `slurm_wait_minutes` = 30.
  Otherwise, cancel it and deploy a RunPod pod. Paid `elastic-*` partitions are not used.
- Jobs copy `param_decomp` to node-local storage before importing it, because NFS has served stale modules right after
  edits, and they log a line that only the new code prints.
- Because `search.py` resumes from `fits`, a run can move between SLURM and a pod.

## 11. Order of work

1. Write the configuration files, `forward.py` and the tests. Run the tests.
2. Collect answers, then build pairs and the SVD initialization.
3. Run the pilot. **Review point** (section 7.1).
4. Run the screen and uniform descent.
5. Run the per-position search and diagnostics.
6. Write the report in `notes/das/report.md`.
7. Separately, after your review of these results: integer-only model.
