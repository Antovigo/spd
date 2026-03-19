# Rich Examples Variant Plan

Baseline experiment uses Jose subset:
- subset file: `component_subsets/jose_coherent_100_seed0.txt`
- harvest subrun: `h-20260318_223737`

## Branch Structure

These branch names are intended for SLURM `snapshot_branch` runs.

Important:
- `snapshot_branch` uses the committed tip of the named branch.
- Local uncommitted changes in the current worktree will not be visible to jobs.
- So these branches are placeholders until each variant's code is committed on that branch.

## Proposed Fan-Out

### 1. Baseline

- branch: `exp/rich-base-jose100`
- purpose: current `rich_examples` baseline after no-regrets cleanup

### 2. High-Confidence Union

- branch: `exp/rich-hiconf-union-jose100`
- purpose: combine the next batch of high-confidence prompt changes in one variant
- candidate changes:
  - slightly tighter skepticism / mixed-evidence wording
  - modest cleanup of punctuation-heavy highlighted examples
  - any small prompt clarifications that are clearly beneficial and low-risk

### 3. Delimiter Sweep

- branch: `exp/rich-delim-brackets-jose100`
- purpose: current `[[[token]]] (ci, act)` style

- branch: `exp/rich-delim-angle-jose100`
- purpose: legacy angle-bracket-style delimiter for direct comparison

- branch: `exp/rich-delim-inline-jose100`
- purpose: lighter inline annotation style without heavy wrappers, e.g. `token{ci,act}`

- branch: `exp/rich-delim-prefix-jose100`
- purpose: prefix marker style, e.g. `^token (ci, act)` or similar

## Recommended Execution Order

1. Run `exp/rich-base-jose100`.
2. Run delimiter sweep variants in parallel.
3. Run `exp/rich-hiconf-union-jose100`.
4. Inspect compare view and evals before fanning further.

## Launch Record

Final clean 100-key run:
- date: `2026-03-19`
- decomposition id: `s-55ea3f9b`
- harvest subrun: `h-20260318_223737` (`activation_threshold: 0.5`)
- subset file: `component_subsets/jose_coherent_100_seed0.txt`
- interpret/eval model: `google/gemini-3-flash-preview`
- initial shared config: `scratch/autointerp_rich_jose100_qwen_base.yaml`
- eval retry config used to recover coverage: `scratch/autointerp_rich_jose100_eval_slow.yaml`

Final subruns:

- branch: `exp/rich-delim-angle-jose100`
  - subrun: `a-20260319_165443_381153-exp-rich-delim-angle-jose100`
  - interpret: `358375`
  - slow retry evals: detection `358462`, fuzzing `358463`

- branch: `exp/rich-hiconf-union-jose100`
  - subrun: `a-20260319_165443_381197-exp-rich-hiconf-union-jose100`
  - interpret: `358376`
  - slow retry evals: detection `358464`, fuzzing `358465`

- branch: `exp/rich-delim-inline-jose100`
  - subrun: `a-20260319_165443_381151-exp-rich-delim-inline-jose100`
  - interpret: `358377`
  - slow retry evals: detection `358466`, fuzzing `358467`

- branch: `exp/rich-delim-prefix-jose100`
  - subrun: `a-20260319_165443_397378-exp-rich-delim-prefix-jose100`
  - interpret: `358378`
  - slow retry evals: detection `358468`, fuzzing `358469`

- branch: `exp/rich-base-jose100`
  - subrun: `a-20260319_165443_395787-exp-rich-base-jose100`
  - interpret: `358381`
  - slow retry evals: detection `358470`, fuzzing `358471`

Final 100-key summary:

- `angle`
  - detection: `100`, mean `0.7700`
  - fuzzing: `100`, mean `0.7306`

- `hiconf`
  - detection: `100`, mean `0.7726`
  - fuzzing: `100`, mean `0.7301`

- `inline`
  - detection: `100`, mean `0.7704`
  - fuzzing: `100`, mean `0.7177`

- `prefix`
  - detection: `100`, mean `0.7711`
  - fuzzing: `99`, mean `0.7259`

- `base`
  - detection: `100`, mean `0.7596`
  - fuzzing: `100`, mean `0.7242`

Readout:
- differences are real but small
- `base` looks mildly worse on detection
- `inline` looks mildly worse on fuzzing
- `angle`, `hiconf`, and `prefix` are clustered near the front

Prepared next subset:
- `component_subsets/jose_coherent_500_seed0.txt`

Logs:
- SLURM logs live under `/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/`
