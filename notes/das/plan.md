# Distributed Alignment Search on the arithmetic decomposition's target: plan

Goal: for the operands and the result of `a+b=` and `a-b=`, find in Llama-3.1-8B (the "original model") which
positions and stream points hold each variable, how many dimensions each position needs, and the subspaces themselves.

Scope: original model only. Position flags, the operation and the integer-only model are out of scope for now. The
trained subspaces will be tested and fine-tuned on the integer-only model later, as a separate step.

How each part of the question is answered:

| Question | Measurement | Where it is reported |
|---|---|---|
| At which stream points can a variable be swapped? | Whole-layer patch at every `k_p` = 64, at all 64 stream points (section 7.2) | `where_each_variable_is_stored.csv` |
| Which positions hold what the model uses? | Each position patched alone and left out (section 7.4), and each position's minimal `k_p` (section 7.3) | `iia_patching_each_position_alone_or_left_out.csv`, `where_each_variable_is_stored.csv` |
| How many dimensions are needed? | Pruning, with IIA recorded at every dimension reached (section 7.3) | `iia_during_pruning.csv`, `where_each_variable_is_stored.csv` |
| What are the subspaces? | The 64-dimension and minimal `R_p` | `trained_subspaces/` |
| Does later computation use `r`, or still the operands? | Double patch with competing predictions (section 7.4) | `result_patch_followed_by_operand_patch.csv` |

DAS measures what later computation uses. A copy of a variable that nothing reads gives IIA near 0, so "stored" in this
plan means "stored and used".

## 1. Notation

| Symbol | Meaning | Type and range |
|---|---|---|
| `$RUN` | The run folder, `~/out/pod-backup/p-ba8b0c08` | path |
| `$REPO` | The worktree holding this plan and the scripts, `~/Code/param-decomp/das` (branch `experiment/das`) | path |
| `a`, `b` | Operands of the base prompt | integers, 1..400 |
| `a'`, `b'` | Operands of the source prompt | integers, 1..400 |
| `a''`, `b''` | Operands of a third prompt, used only by the double patch in section 7.4 | integers, 1..400 |
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
| `S_p` | Right singular vectors of `D R_p`, ordered by decreasing singular value, used to rotate `R_p` | `k_p`×`k_p` orthogonal matrix |
| `q` | Model output renormalised over the tokens of the integers 0..999 | probability vector of length 1000 |
| IIA | Interchange intervention accuracy: the fraction of evaluated pairs whose patched top integer token, the argmax of `q`, is the expected answer. Validation pairs are used during the search and test pairs in the final evaluation (section 7.5) | fraction, 0..1 |
| Full fit | Training from the SVD initialization for the full budget: 28,672 pair passes at the starting 3,584 pairs and 8 epochs | training run |
| Fine-tune | Training an existing set of `R_p` for a short budget: 3,584 pair passes at the starting value of one epoch | training run |

`t = 37` is the paper's "pre-MLP" point at layer 18. `t = 38` is its "post-MLP" point at layer 18. The output of block
31 (the paper's post-MLP point at layer 31) is not run.

## 2. Decisions to check before implementation

Each value is a key in `$REPO/notes/das/das_experiment_config.yaml` (section 9).

1. **Stopping rule for one position.** A round passes when whole-layer validation IIA is at least the reference minus
   `position_margin` = 0.02, where the reference is the IIA at every `k_p` = 64. A round that improves IIA passes. Your "beats chance" rule cannot stop a per-position search, because the other
   positions keep IIA above chance after that position has lost most of its contribution.
2. **`min_iia_for_position_search` = 0.40 at `k_p` = 64.** Two risks, both checked in the pilot:
   - The paper's best operand IIA for addition with operands up to 100 was 61.7% and 63.8%, so with operands up to 400
     few operand sites may pass.
   - The initialization holds both operands, since the source differs in both. So IIA for `a` or `b` might rise as `k`
     falls, which would make `k_p` = 64 a poor screen and a poor reference. The pilot's fresh ladder measures this.
3. **One pair set per task, shared by the three variables.** All four prompts must be correct, which keeps 33.8% of
   correct subtraction pairs (section 5).
4. **Objective:** cross-entropy against the expected answer, as in the paper (section 6).
5. **Queue patience:** `slurm_wait_minutes` = 30 before moving to a pod (section 10).
6. **Pruning schedule:** `prune_schedule` = 64→32→16, then single steps to 0, with bisection after a failed large step
   (section 7.3).
7. **Pruning budgets:** `finetune_epochs` = 1, chosen by the pilot from {1, 4}, and `prune_score_pairs` = 256.
8. **Pilot pass rule for pruning** (section 7.1). If no arm passes, the fresh-fit search stays available.
9. **Split:** 60% train, 20% validation and 20% test (section 5).

### Removed in this revision

Each item was checked against the question above. Any of them can be restored.

| Removed | Why | What is lost |
|---|---|---|
| Uniform descent with one shared `k`, the lowest `k` beating chance, and `k_95` | It was the largest cost (4,224 full fits), and pruning already gives dimensions per position and an IIA curve | Your descent to chance, and a curve from fresh fits at every site. The pilot still has the fresh curve at 8 sites |
| Chance as a decision | With the 0.40 threshold it decided nothing | Nothing: random-subspace IIA is still reported per site |
| Rerun seeds | Seeds only change batches, and the paper does not use them as a control | The spread of `k_p` across seeds |
| Subspace overlap | The double patch tests the same concern causally | The paper's overlap measure |
| Test pair top-up by digit count | Added code and a file column for small bins | Small bins have wide intervals, for example about 53 test pairs with a one-digit `a` on addition. Pair counts are reported |
| Batch, epoch and 57,344-pair sweeps, and `finetune_epochs` = 0.25 | The fit quality question is the learning rate and whether 3,584 pairs overfit | The paper's batch and epochs are used without re-checking them |
| Cross-entropy and integer mass in the result tables | No part of the question uses them | They stay available during the pilot |

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
descends, per-position dimensions are found by pruning (section 7.3), a validation split makes the search decisions, and
the initialization differs (section 8).

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
  tests it.

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

The split is on prompts: 60% of correct prompts are train, 20% validation and 20% test. A pair belongs to a split only if
its base, source and expected-answer prompts all do. Training starts at 3,584 pairs, subject to the pilot.

Measured on one random split of the same correctness data, by sampling 4,000,000 prompt pairs within each split:

| Split | Addition: prompts / share of pairs accepted / approximate distinct pairs | Subtraction: prompts / share of pairs accepted / approximate distinct pairs |
|---|---|---|
| Train | 87,242 / 31.4% / 2.4e9 | 31,572 / 12.1% / 1.2e8 |
| Validation | 29,257 / 3.5% / 3.0e7 | 10,679 / 1.4% / 1.6e6 |
| Test | 28,747 / 3.4% / 2.8e7 | 10,551 / 1.4% / 1.5e6 |

**Validation** has 4,096 uniformly sampled pairs. Every search decision uses it: pilot choices, the screen threshold and
the pruning stop. Pruning checks IIA repeatedly and keeps what passes, so validation IIA
on the kept subspace is biased upward and is not reported as the result.

**Test** has 4,096 uniformly sampled pairs, which is 8 times the paper's 512. Test pairs are evaluated once, after every
selection is frozen (section 7.5). IIA is also reported by digit count of `a`, `b` and the expected answer, with the number
of pairs and a 95% Wilson interval. Some bins are small or empty: on subtraction, the sample above found 3 test pairs
with a one-digit `a`.

## 6. Objective

Train on the paper's objective (its Eq. 1): cross-entropy between the patched model's output and the token of the expected
answer in section 4. The softmax is over the full vocabulary. The paper's text does not restrict it, but its loss code is
in the `causalab` dependency, which I have not read.

A perfect interchange does not reach 0 under this loss. The original model's own full-vocabulary cross-entropy on prompts
with an answer of at least 1 is 0.97 on addition and 2.40 on subtraction.

Reported metrics: IIA, which makes the decisions, and the same IIA over the full vocabulary, for comparison with the paper.
They are computed on validation pairs during the search and on test pairs in section 7.5.

## 7. Procedure

### 7.0 Pairs and initialization

Build the pairs (section 5) from the per-prompt correctness already in `references.npz`, so no collection run is needed.
That file has 239,800 rows and no index, and `tokenized.npz` has 320,000. Its rows are the rows of `tokenized.npz` whose true
answer is at least 1, in the same order. `build_prompt_pairs.py` rebuilds the reference pool with `build_problem` from the
pruning run's `config.yaml`, maps through its `pool_rows`, and asserts that the tokens equal `tokenized.npz` on every row.
Saved pairs index the rows of `tokenized.npz`.

Before a site's first fit, compute its SVD initialization in memory: for each `p` in {1, 2, 3, 4}, take the SVD of `D` over
the first 3,584 training pairs, without centring, and keep the top 64 right singular vectors. Pairs are shared across
variables, so the initialization depends only on the task and `t`.

### 7.1 Tests and pilot

No GPU fit starts until these pass:

- Running from any `t` reproduces the full forward.
- The forward pass reproduces per-prompt correctness in `references.npz` on a sample of 1,024 prompts per task, which also
  checks that its rows align with `tokenized.npz`.
- `R_p` = identity at all of `P_a` makes the output equal the source's own at every `t`.
- With every `k_p` = 0 the output equals the base's own.
- Replacing `R_p` with `R_p S_p` leaves the patched output unchanged.
- Pair requirements hold on every saved row, and the three splits share no prompts.
- A pruning round that raises IIA above the reference passes.

The pilot uses eight sites: `a` and `r` at `t = 1` (block 0, before the MLP) and `t = 37` (block 18, before the MLP), on each
task. It runs on the cluster only, not on a pod. All its decisions use validation pairs. It
answers three questions: does a 64-dimension fit train well, does IIA depend on `k` as the search assumes, and does
pruning reach what fresh fits reach?

| Pilot sweep | Values | Training runs |
|---|---|---|
| Fit quality, at every `k_p` = 64, batch 16 and 8 epochs | Learning rate {1e-4, 1e-3, 1e-2}, training pairs {3,584, 14,336} | 6 × 8 = 48 full fits |
| Fresh-fit search, first position order only | Each position steps down 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1, then 0, with a full fit at every step | At most 288 full fits |
| Pruning, `prune_schedule` | `finetune_epochs` {1, 4} | At most 864 fine-tunes |
| Pruning, single steps from 64, addition sites only | `finetune_epochs` = 1. Tests the shortcuts in the schedule | At most 768 fine-tunes |

The pruning arms use the same position order as the fresh-fit search, so the comparison isolates the method. For each
arm, the pilot records the `k_p` tuple, the sum of `k_p`, the IIA gap from the reference, and seconds.

**Pass rule for pruning.** An arm passes at a site when its sum of `k_p` is at most the fresh-fit sum plus the larger of 2
and 10% of that sum.

The default for section 7.3 is the cheapest `prune_schedule` arm that passes at all eight sites. If none passes, I report
that and the choice between the fresh-fit search and a changed pruning arm is yours.

The pilot also measures seconds per fit and fine-tune against `t`. Runs go one at a time per GPU unless the projected
cost requires running several in parallel. That would add code, since pruning paths diverge in array shape.

**Review point:** I report the pilot results and the projected cost before continuing, including whether IIA rises as `k`
falls (section 2, item 2). If every site passes, the upper bounds are:

| Stage | Upper bound | Full-fit equivalents at `finetune_epochs` = 1 |
|---|---|---|
| Screen | 384 full fits | 384 |
| Per-position pruning | 36,864 fine-tunes | 4,608 |
| Total | | 4,992 |

The fresh-fit search would need 25,600 full fits in place of the pruning row. These equivalents count training only;
section 7.3 lists the scoring and validation passes. If the projection is too high, the options are stepping `t` by 2 or
raising `min_iia_for_position_search`.

### 7.2 Screen

For each of the 6 task-variable combinations and all 64 values of `t`, train with every position of `P_v` at `k_p` = 64.
That is 384 fits. Each site's validation IIA decides whether it enters section 7.3.

### 7.3 Per-position minimum by iterative pruning

This approach is untested, and the pilot compares it with fresh fits (section 7.1).

It runs at each site whose validation IIA at `k` = 64 is at least `min_iia_for_position_search`. The starting point and
reference is the site's screen fit from section 7.2: whole-layer, every `k_p` = 64, from the SVD initialization. No other
fit in this search uses that initialization. Test pairs are not used. Validation IIA is recorded at every round, which
gives the curve of IIA against dimensions.

**One pruning round at position `p`:**

1. **Rotate.** Replace `R_p` with `R_p S_p`. The patch depends only on the span of `R_p`, so this leaves the output
   unchanged. It gives the columns an order by energy, which is not importance: since the source differs in both
   operands, a high-energy direction can carry the variable that is not being swapped. So step 2 tests every column.
2. **Score.** For each column of `R_p`, remove it without training and compute the training objective on
   `prune_score_pairs` = 256 training pairs.
3. **Remove.** Remove the columns whose removal raises the objective least, as many as the schedule asks for. Columns are
   never removed at random: before this scoring they have no special meaning, so a random removal can drop an important
   direction.
4. **Fine-tune.** Train all remaining `R_p`, at every position, jointly for `finetune_epochs` = 1. Adam's state restarts
   at each fine-tune, because the parameter shapes change.
5. **Check.** Compute whole-layer validation IIA. The round passes if it is at least the reference minus
   `position_margin` = 0.02, so an improvement passes. A pass keeps an in-memory checkpoint. A failure restores the previous checkpoint.

**Schedule.** `k_p` goes 64→32→16, then down by one to 0.

- After a failed large step, bisect between the last passing and the failing `k_p`, always removing from the last passing
  checkpoint.
- After a failed single step, or when bisection closes, the position keeps its last passing subspace and the search moves
  on.
- A large step removes several columns from one scoring round, which assumes their harms add. Bisection also assumes
  that passing is monotonic in `k_p`, which a path-dependent search does not guarantee. Both are shortcuts for cost, and
  the pilot's single-step arm tests them.

**Position orders.** Pruning does not remove the search-order problem: if two positions hold redundant copies, the one
pruned first loses its copy and the other must keep it.

1. **Rank.** From the `k_p` = 64 fit, set each position to `k_p` = 0 in turn without training, and sort by the increase
   in the training objective, smallest first.
2. **First order.** Prune the positions in that order, each starting from where the previous one ended.
3. **Second order.** Restart from the `k_p` = 64 fit and prune in the reverse order.
4. **Report.** Both final tuples of `k_p`, and whether they agree.

**Limits.** The pilot's fresh-fit search compares these at its 8 sites only; the full run has no such check:

- Greedy removal can reach a different minimum than a fresh fit.
- Candidates are only the columns of one basis, so the least important direction in the span may not be removable.
- A fine-tune may be too short to recover from a removal, which would stop the search early.

**Cost per position, in one order, if the path reaches `k_p` = 0.** Pair passes are at the starting hyperparameters:

| Method | Training runs | Pair passes, training (forward and backward) | Pair passes, scoring (forward) | Pair passes, validation (forward) |
|---|---|---|---|---|
| Fresh fits on the ladder, then 0 | 12 full fits | 344,064 | 0 | 49,152 |
| Pruning, single steps from 64 | 64 fine-tunes | 229,376 | 532,480 | 262,144 |
| Pruning, `prune_schedule` | 18 fine-tunes | 64,512 | 59,392 | 73,728 |

Compared with fresh fits, single steps from 64 use 33% fewer training passes but add 532,480 scoring passes and 212,992
validation passes. `prune_schedule` uses 81% fewer training passes, and adds 59,392 scoring passes and 24,576
validation passes. Seconds per pass are not yet measured; the pilot measures them. Over all sites, the bound for `prune_schedule` is 18 fine-tunes × 4 positions × 2 orders for `a` and 18 × 2 × 2 for each of
`b` and `r`. That is 288 per `t` and task, or 36,864 over 64 values of `t` and 2 tasks. Single steps would need 131,072.

### 7.4 Diagnostics

None of these needs training.

**Which positions matter, at every site of the screen.** Using the `k_p` = 64 fit with no retraining, IIA with each `R_p`
patched alone and with each left out. The fit is joint, so positions may have split the work: a low "alone" IIA does not
show a position could not do it alone.

**Whether later computation uses `r` or the operands (double patch).** High `r` IIA alone cannot separate these. This runs
only at stream points where the `k_p` = 64 fits of `r` and of the operand both reach validation IIA of at least 0.40, so the
second patch is known to work alone. With no retraining, at the same `t`, patch `r` from `(a', b')`, then patch `a` from a
third prompt with first operand `a''`. Separately, do the same with `b` and a third prompt with second operand `b''`.

| Top answer, for the `a` version on addition | Reading |
|---|---|
| `a' + b'` | Later computation follows the `r` patch, and the operand patch does not change it |
| `a'' + b'` | Later computation still uses the operand |
| Anything else | Neither prediction holds |

The `b` version predicts `a' + b'` and `a' + b''`. On subtraction, `-` replaces `+`. All these prompts come from the test
split, must be correct, and must have distinct predicted answers. Each row reports the three shares and both fits' IIA alone.

### 7.5 Final test evaluation

After every selection is frozen, evaluate these once on test pairs:

- the `k_p` = 64 fit at every site, with random-subspace IIA pooled over 5 draws of the same dimensions as a control;
- both final tuples at each searched site, with the gap from the same site's `k_p` = 64 test IIA;
- the single-position and left-out patches and the double patch of section 7.4.

Pruning rounds are not stored, so they report validation IIA only.

The test gap can exceed `position_margin`, because the tuple was selected on validation. It is reported as measured, and
no tuple, threshold or hyperparameter is changed after this step.

## 8. Fit specification

| Setting | Value |
|---|---|
| Parametrisation | One unconstrained `d`×`k_p` matrix per position, with `R_p` its Q factor from QR at each step |
| Initialization | Full fits: top `k_p` right singular vectors of `D`, per section 7.0. Fine-tunes: the kept columns of the current `R_p` |
| Optimizer | Adam, restarted at each fine-tune. The paper's text does not name its optimizer |
| Starting hyperparameters | Learning rate 1e-4, batch 16, 8 epochs, 3,584 pairs, replaced by the pilot's choice |
| Fine-tune budget | `finetune_epochs` = 1, on the same training pairs and at the same learning rate as full fits |
| Model precision | `bf16` matmuls with fp32 accumulation, as in `param_decomp/pruning/model.py` |
| Model weights | Frozen |

## 9. Files

Only three kinds of thing are stored: what is very expensive to compute, what will be re-used, and results that directly
answer the scientific question. All paths are keys in `$REPO/notes/das/das_experiment_config.yaml`, so editing that file
moves them. Only section 9.5 lists paths outside `$RUN/analysis/das/` that hold data, and they are temporary.

Placeholders in the names:

| Placeholder | Values |
|---|---|
| `{task}` | `addition`, `subtraction` |
| `{variable}` | `first_operand` (`a`), `second_operand` (`b`), `result` (`r`) |
| `{block}` | The block index `l`, an integer from 0 to 31 |
| `{stream_point}` | `before_attention` (`t = 2l`), `before_mlp` (`t = 2l + 1`) |

### 9.1 Experiment settings, to be written

| File | Contents |
|---|---|
| `$REPO/notes/das/das_experiment_config.yaml` | Every path and setting, and for each variable its expected-answer rule and `P_v` (section 4) |

### 9.2 Existing inputs, read only

| File | Used for |
|---|---|
| `$RUN/prompts/tokenized.npz` | Tokens, `a`, `b` and the operation for 320,000 prompts |
| `$RUN/analysis/pruning/step_40000/addsub-08-prune/references.npz` | The original model's per-prompt correctness, used to build the pairs |
| `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/` | Weights, `config.json` and `tokenizer.json` |

### 9.3 Stored data, under `$RUN/analysis/das/`

| File | Contents |
|---|---|
| `README.md` | What each folder and column is |
| `prompt_pairs/{task}_prompt_pairs.npz` | Prompt rows of each pair's base and source, and its split (train, validation or test). Expected answers follow from these rows |
| `trained_subspaces/{task}_{variable}/block_{block}_{stream_point}.npz` | The `R_p` matrices of one site, under the keys `start_64_dims_position_{p}`, `minimal_first_order_position_{p}` and `minimal_reverse_order_position_{p}`, where `{p}` is the position, an integer from 1 to 4. A key is absent if `k_p` = 0 or the site skipped the search. At most about 13 MB per file and 3.2 GB in total |

### 9.4 Results, under `$RUN/analysis/das/results/`

| File | One row per | Contents |
|---|---|---|
| `where_each_variable_is_stored.csv` | Task, variable and stream point | Validation and test IIA at `k_p` = 64, full-vocabulary test IIA, random-subspace IIA, each position's minimal `k_p` in both orders, their test IIA and gap, and whether the orders agree |
| `iia_during_pruning.csv` | Pruning round: task, variable, stream point and order | `k_p` tuple and validation IIA |
| `iia_patching_each_position_alone_or_left_out.csv` | Task, variable, stream point and position | Test IIA at `k_p` = 64 with `R_p` patched alone and with it left out |
| `iia_by_digit_count.csv` | Stored subspace, quantity and digit count | Test IIA by digit count of `a`, `b` and the expected answer, the number of pairs and the interval |
| `result_patch_followed_by_operand_patch.csv` | Task, stream point and operand | Shares of the two predicted answers and of neither, and each fit's IIA alone |

### 9.5 Temporary files, deleted after use

| Path | Contents |
|---|---|
| `~/pd_scratch/das_pilot/hyperparameter_and_pair_count_sweep.csv` | One row per pilot sweep fit |
| `~/pd_scratch/das_pilot/pruning_compared_with_fresh_fits.csv` | Per site and arm: `k_p` tuple, sum of `k_p`, IIA gap, seconds and pass rule outcome |
| `~/pd_scratch/das_pilot/training_time_and_projected_cost.csv` | Seconds per fit and fine-tune by `t`, parallel runs per GPU and projected totals |
| `~/pd_scratch/das_job_logs/` | SLURM or pod logs |

`~/pd_scratch/das_pilot/` is removed after the review point. The chosen settings go into the config file, and the
comparison that justified them goes into the report. The logs are removed when the report is done.

### 9.6 Not stored

| Item | Why it is safe to drop | Consequence |
|---|---|---|
| SVD initialization | Recomputed from `prompt_pairs` | Each site runs one forward pass over 3,584 pairs before its first fit |
| Answers on all prompts | `references.npz` already has correctness | One test checks the forward pass against it on a sample (section 7.1) |
| Subspaces of intermediate pruning rounds | Not re-used. Only their results are needed | They report validation IIA, not test IIA |
| Pruning checkpoints | Only the final tuple is used | An interrupted search restarts that site from its 64-dimension subspaces |

### 9.7 Scripts, all in the repo

The repo's `CLAUDE.md` keeps scheduler code and cluster paths out of `param_decomp/`, so the settings and launch scripts
live under `notes/das/`, and the library code takes paths as required arguments. The new test folder must also be added
to a CI shard (`param_decomp/tests/infra/test_ci_shards.py`).

| File | Role |
|---|---|
| `$REPO/param_decomp/das/__init__.py` | Package marker |
| `$REPO/param_decomp/das/load_das_experiment_config.py` | Load and validate the settings |
| `$REPO/param_decomp/das/build_prompt_pairs.py` | Section 5. Light enough for the dev node |
| `$REPO/param_decomp/das/patched_llama_forward.py` | Original-model forward resumable at any `t`, with the patch, reusing `DenseWeights` and the block code of `param_decomp/pruning/model.py`. `run_blocks` starts only at block boundaries, so this adds the odd `t`; written so the integer-only model can use it later |
| `$REPO/param_decomp/das/train_das_subspaces.py` | One full fit or fine-tune: SVD initialization, loss, QR and metrics |
| `$REPO/param_decomp/das/prune_das_subspaces.py` | One pruning search: rotate, score, remove, fine-tune, check, schedule, bisection and position orders |
| `$REPO/param_decomp/das/run_das_pilot.py` | Section 7.1 pilot |
| `$REPO/param_decomp/das/run_das_search.py` | Screen and pruning. One process drives all GPUs and is the only writer, and it skips sites already finished in `results`, so a preempted job continues |
| `$REPO/param_decomp/das/evaluate_das_on_test_pairs.py` | Section 7.5, including the no-training diagnostics of section 7.4 |
| `$REPO/param_decomp/das/make_das_report_figures.py` | Report figures |
| `$REPO/notes/das/launch/run_das_pilot.sbatch` | Pilot: 1 L40 |
| `$REPO/notes/das/launch/run_das_search.sbatch` | Search: 4 L40 |
| `$REPO/notes/das/launch/evaluate_das_on_test_pairs.sbatch` | Final evaluation: 1 L40 |
| `$REPO/notes/das/launch/set_up_runpod_pod.sh` | Pod environment, and copying the inputs of section 9.2 and the data of section 9.3 |
| `$REPO/notes/das/launch/copy_results_from_runpod_pod.sh` | Copy `$RUN/analysis/das/` back from the pod |
| `$REPO/param_decomp/tests/das/test_patched_llama_forward.py` | Forward and patch tests of section 7.1 |
| `$REPO/param_decomp/tests/das/test_prune_das_subspaces.py` | Rotation invariance, and the pass rule when IIA improves |
| `$REPO/param_decomp/tests/das/test_build_prompt_pairs.py` | Pair requirements and split separation |
| `$REPO/notes/das/report.md` | Report |
| `$REPO/notes/das/report_figures/` | PNGs embedded in the report |

## 10. Compute

- All of this runs through SLURM or on a pod; only editing and submitting happen on the dev node. Scratch goes to
  `/ephemeral/$USER`.
- Use `sbatch` on the free `compute` partition with 4 L40s if the job starts within `slurm_wait_minutes` = 30.
  Otherwise, cancel it and deploy a RunPod pod. Paid `elastic-*` partitions are not used.
- Jobs copy `param_decomp` to node-local storage before importing it, because NFS has served stale modules right after
  edits, and they log a line that only the new code prints.
- Because `run_das_search.py` skips sites already finished in `results`, a run can move between SLURM and a pod.

## 11. Order of work

1. Write the settings file, `patched_llama_forward.py` and the tests. Run the tests.
2. Build the prompt pairs.
3. Run the pilot, including pruning against fresh fits. **Review point** (section 7.1), then remove the pilot folder.
4. Run the screen.
5. Run the per-position search.
6. Freeze all selections, then run the final test evaluation (section 7.5).
7. Write the report in `notes/das/report.md`.
8. Separately, after your review of these results: integer-only model.
