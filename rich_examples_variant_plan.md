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
