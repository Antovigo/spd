# Rich Examples Open Threads

Working notes for the current `spd/autointerp/strategies/rich_examples.py` iteration.

## Agreed Direction

- Keep `rich_examples` centered on activation examples rather than rebuilding the full compact/dual-view prompt.
- Keep the uniformly sampled reservoir examples. Do not rebalance or curate the sample.
- Keep the prompt relatively elegant; avoid piling on extra summary sections unless they clearly earn their keep.
- Bring back output-side evidence, but not input token correlations.

## Open Threads

### 1. Output Token Evidence

- Add output token correlations back into `rich_examples`.
- Current leaning: output-only, not input.
- Open question: use output precision, output PMI, or both.

Notes:
- `TokenPRLift.top_precision` is actually `P(component fires | output token)`, not `P(output token | component fires)`.
- That naming / prompt wording mismatch should be handled carefully if we surface this metric.
- PMI may be useful, but rare-token artifacts are a concern.

### 2. Rare-Token / Low-Count Filtering

- Before using PMI heavily, decide whether to filter low-support tokens.
- Counts do exist in harvest token stats, but the prompt-facing PMI objects currently do not carry counts through.

Possible implementation directions:
- Filter inside token-stat computation using thresholds like:
  - minimum token marginal count
  - minimum co-occurrence count
- Or extend prompt-facing stat objects to include counts.

Current preference:
- If we keep PMI, filter on support rather than trusting raw top PMI.

Jose eval notes (`s-55ea3f9b`, 40 random active components, top-10 output PMI):
- Output support filter by output `counts` has a meaningful effect.
- `counts >= 2`:
  - 8/40 components changed
  - average top-10 set Jaccard: `0.907`
- `counts >= 5`:
  - 12/40 components changed
  - average top-10 set Jaccard: `0.865`
- Adding an output `totals` floor on top of `counts >= 5` changed little.
- Filtering by `input_totals` as a proxy is much weaker:
  - `input_total >= 2`: 1/40 components changed
  - `input_total >= 5`: 3/40 components changed
  - `input_total >= 20`: 5/40 components changed
  - `input_total >= 100`: 11/40 components changed

Decision for current iteration:
- Use output PMI with an output-support filter.
- Do not rely on `input_totals` as the main support proxy.

Terminology reminder:
- For input stats, `counts` / `totals` are actual counts.
- For output stats, `counts` / `totals` are probability-mass analogs, not hard counts.

### 3. Example Rendering Format

- The current XML-style `<raw>` / `<highlighted>` format is likely too bulky.
- The more serious issue is that some rendered examples are hard to read because token spans can be pathological.

Current direction:
- Replace XML blocks with one annotated line per example, probably as a numbered list.
- Keep inline annotations on firing tokens.

Potential rendering changes:
- Use `AppTokenizer.get_spans()` instead of `get_raw_spans()` so control characters become visible markers.
- Add a fallback for empty decoded spans, since the current prompt can produce `<<<>` artifacts.
- Keep enough context to preserve boundary information.

### 4. Boundary / Truncation Cues

- We want the model to better notice when a firing occurs at the beginning of the sampled window.
- This is especially important for BOS / prefix / boundary components.

Current direction:
- Reuse the shared "Data" framing.
- Add explicit wording that left-edge or right-edge truncation can itself be evidence about the feature.

### 5. Context Section

- The current `## Context` block is too noisy.
- The model-class string is probably not useful in its current fully qualified form.

Current direction:
- Thin context to only the parts the model can actually use.
- Likely keep:
  - component location
  - firing rate
  - dataset description (optional)
- Likely drop or shorten:
  - full model class path
  - extra prose that does not inform interpretation

### 6. Task Framing / Skepticism

- `rich_examples` should keep the functional framing, but be more explicit about uncertainty / abstention.
- We do not want to reintroduce obviously overfit wording.

Current direction:
- Tighten the task instructions so the model is willing to be uncertain when examples are noisy or heterogeneous.
- Keep the style simpler than `compact_skeptical`, but borrow its skepticism where helpful.

### 7. Sign / Polarity Guidance

- The SPD explanation currently says magnitude is what matters.
- The annotation legend also says sign may separate distinct patterns within a component.

Open question:
- Keep both ideas, but make the distinction cleaner:
  - sign is not "suppression"
  - sign may still matter within-component as a split in behavior

### 8. Prompt Size vs Readability

- We want to improve evidence quality without regressing into a long, busy prompt.
- The prompt should stay cheaper and simpler than the token-stat-heavy strategies.

Decision boundary:
- Favor changes that improve scanability and interpretability of examples.
- Be cautious about adding new sections unless they supply clearly distinct evidence.

## Candidate Next Patch

If we do a first implementation pass, the highest-value bundle is:

1. Add output-only token correlations.
2. Replace XML examples with one-line annotated examples.
3. Improve boundary/truncation wording in the data section.
4. Slim down the context section.
