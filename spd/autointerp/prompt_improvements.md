# Autointerp prompt improvements tracker

## Status as of 2026-03-20 18:05

The "canon" strategy has been codified and submitted for evaluation against all 3 CI thresholds on the same 200-component Jose subset used in the earlier sweep. Results pending.

### Running jobs

| Threshold | Harvest | Interpret | Detection | Fuzzing | Subrun ID |
|-----------|---------|-----------|-----------|---------|-----------|
| 0.0 | h-20260227_010249 | 361230 | 361231 | 361232 | `a-20260320_180347_247339` |
| 0.1 | h-20260319_121635 | 361233 | 361234 | 361235 | `a-20260320_180357_271631` |
| 0.5 | h-20260318_223737 | 361236 | 361237 | 361238 | `a-20260320_180407_787644` |

### Baselines to beat (earlier sweep fuzzing scores)

| Threshold | Best variant | Score |
|-----------|-------------|-------|
| ci0.0 | rich-xml-brackets | 0.7099 |
| ci0.1 | compact-legacy | 0.7315 |
| ci0.5 | rich-xml-brackets | 0.7468 |

Full earlier results: `http://goodfre-login:8081/spd//autointerp/jose_threshold_matrix_v2_200_results.html`

## What changed in the canon strategy vs rich-xml-brackets

The canon strategy (`spd/autointerp/strategies/canon.py`) is a ground-up rewrite of the prompt, not a rendering tweak. Key differences from `rich_examples`:

1. **Full SPD method explanation** — 3 losses (faithfulness, minimality, reconstruction), CI network, rank-1 decomposition with U/V read/write directions. `rich_examples` had a shorter blurb.
2. **Explicit CI-vs-act guidance** — "When CI and act diverge, trust CI." The Gemini probe showed models getting confused when these values diverged.
3. **Sign convention rewrite** — Explains that relative sign is meaningful (opposite output contributions) but absolute sign is arbitrary. Tells the model to describe both clusters if they exist but not assign meaning to which is "positive". Previous versions had a contradiction between "check sign clusters" and "don't conclude anything about sign".
4. **Dynamic CI threshold** — Reads from harvest config via `HarvestDB.get_activation_threshold()` and injects into the prompt. No more hardcoded "0.0".
5. **Next-token prediction clarity** — PMI section explicitly says "the model's next-token prediction distribution" rather than ambiguous "output tokens".
6. **Rendering is fixed** — XML format, bracket delimiters, activation annotations, no sanitization. Not configurable — this is the one right way.
7. **No input token stats** — Dropped input PMI/precision (noisy, doesn't account for base rates).
8. **No model class name** — Doesn't show `LlamaForCausalLM` or similar — just block count, dataset, layer description.
9. **No layer position editorial** — Doesn't say "this is in the final block so its output directly influences predictions" (was in dual_view). Just states the facts.
10. **`<|endoftext|>` explained** — Training data section explains document concatenation and the separator token.

## Files changed

### New
- `spd/autointerp/strategies/canon.py` — Canon strategy implementation
- `spd/autointerp/prompt_draft.md` — Baked markdown reference (human-readable version of the prompt)
- `spd/autointerp/prompt_improvements.md` — This file
- `scratch/autointerp_sweeps/canon_v1/` — Config YAMLs for the 3-threshold evaluation

### Modified
- `spd/autointerp/config.py` — Added `CanonConfig`, updated `StrategyConfig` union, `resolve_example_rendering`
- `spd/autointerp/strategies/dispatch.py` — Added `CanonConfig` case, threaded `activation_threshold` parameter
- `spd/autointerp/interpret.py` — Reads `activation_threshold` from harvest config, routes `CanonConfig` stats correctly
- `spd/autointerp/prompt_helpers.py` — Renamed `<highlighted>` → `<annotated>`, added `build_separated_examples()`

## Completed prompt improvements

- Renamed `<highlighted>` → `<annotated>` in draft + code
- Fixed "logit mass" → "next-token prediction distribution"
- Removed "Pythia fashion" jargon
- Fixed `<|endoftext|>` spelling consistency
- Named both "read direction" (V) and "write direction" (U)
- Added `build_separated_examples()` to prompt_helpers.py
- Fixed duplicate "### Context" → "### This component"
- Capitalised "we" at start of sentence
- Added "(natural log)" after "nats"
- Made CI threshold dynamic from harvest config
- Sign convention rewrite — removed contradiction, clarified relative vs absolute
- CI emphasis — explicit "when CI and act diverge, trust CI"
- Added consecutive-firing grouping format to annotation legend

## Position analysis (side investigation)

Ran a GPU analysis of firing position distributions across all components (500 batches × 32 = 16K sequences).

**Results page**: `http://goodfre-login:8081/spd/autointerp/debug/position_distributions.html`

**Script**: `spd/autointerp/scripts/analyze_position_distributions.py`

**Findings**:
- Mean entropy: 7.71/9.00 bits — most components fire fairly uniformly across positions
- ~1.2% of components fire predominantly at absolute position 0 (6/7 labeled "unclear" by autointerp)
- Document-boundary components (~0.8%) are relative-position patterns visible in snippets
- For mid-sequence positions (20–491), current harvest data can't distinguish absolute position
- Position-0 components are concentrated in layer 0

## Still to do

### Prompt text (minor)
- [ ] **Dense bracket readability**: Consecutive firings produce hard-to-read brackets. Grouped format documented but not yet implemented in `_delimit_annotated`.
- [ ] **CI mask mechanism**: One Gemini respondent asked if CI is a soft multiplier or hard gate. Currently unexplained. Low priority.

### Harvest changes (requires re-harvest)
- [ ] **`seq_position` on examples**: Store the absolute sequence position (0–511) of the central firing token. Currently lost after windowing. Affects ~460 position-0 components. Files: `schemas.py`, `reservoir.py`, `harvester.py`, `db.py`. Prompt usage: `<example seq_position=123>`.
- [ ] **Activation normalization**: Inner activations not normalized across components (`act = v^T @ x * ||u||`). Simplest approach: normalize per-component at prompt-build time using reservoir p99. Better: accumulate `activation_sum_of_squares` in harvester for proper std normalization.

### Evaluation
- [ ] **Compare canon results** against sweep baselines once jobs complete
- [ ] **Detection eval quality**: Current detection negatives may be too easy. Consider harder negatives.
- [ ] **Update sweep results dashboard** to include canon runs
