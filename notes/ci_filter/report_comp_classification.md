# Classifying components: which help the arithmetic answer, which impair it

Companion to [`report.md`](report.md), which covers the filtering runs themselves. This one is
about the per-component classification: the dataset, what it says, and — importantly — what it
cannot yet answer.

Everything is measured on `p-ba5a0c05` step 40000 over the `a+b=` / `a-b=` pool
(`a, b` in 1..100, 20,000 prompts). The score of a prompt is `log p(correct result's first
token)` under the distribution restricted to integer tokens and renormalized.

## Sign convention (read this first)

Every effect column is **the effect of ABLATING the component**:

- **positive effect / negative `d_answer_ce`** → removing it makes the answer better, i.e. the
  component **impairs** arithmetic;
- **negative effect / positive `d_answer_ce`** → removing it makes the answer worse, i.e. the
  component **helps**.

`impairs_frac` is the *share of prompts* on which ablating helps, magnitude ignored. An earlier
version of this column was labelled "helps-on", meaning "ablating helps"; it is the same number
with a clearer name.

## The dataset

`<run_dir>/analysis/ci_filter/step_40000/components.tsv` — 12,553 rows (every component alive in
the decomposition on this pool) x 46 columns, built by
`python -m param_decomp.ci_filter.scripts.component_table --run_dir <run>`:

| group | columns |
|---|---|
| address | `site`, `layer`, `kind`, `component` |
| survival | `alive_<f>`, `max_ci_<f>` for each of the five filters |
| screen | `effect_<s>`, `effect_<s>_add/_sub` (mean first-order effect of ablating) |
| frequency | `impairs_frac_<s>[_add/_sub]`, `active_frac_<s>` |
| causal | `swept_d{kl,integer_kl,answer_ce,accuracy}_run` (every component, 512 prompts) |
| causal (extremes) | `ablated_dscore_<s>`, `ablated_dacc_<s>` (200 per filter, 1,000 prompts) |

Sources `<s>` are `run` (the decomposition's own CI), `integers` and `ce` (the two narrow
filters). The sweeps for the two filters are still running; their columns arrive on a rebuild.

## Method

1. **First-order screen** — one forward/backward per prompt batch gives every component's
   `-sum_t m * d(score)/d(m)`, plus the per-prompt sign counts. 123 s for all 20,000 prompts.
2. **Causal sweep** — each component's mask zeroed at every position, one masked forward per
   component over a fixed 512-prompt subset, recording full-vocabulary KL, integer KL, answer CE
   and accuracy. 0.29 s per component, 33 min for all 12,553.

Screen versus sweep: **Pearson 0.93** on the effect size, so the cheap pass ranks well. Sign
agreement is only 59%, but that is dominated by the 83% of components whose true effect is
~0 — among components with a real effect the signs agree (see `attribution.png` in `report.md`).

## What the classification says

Single-component ablation, 12,553 alive components, threshold `|d_answer_ce| > 0.001`:

| class | count | share |
|---|---|---|
| **helps** arithmetic (ablating hurts) | 1,640 | 13.1% |
| **impairs** arithmetic (ablating helps) | 454 | 3.6% |
| inert on arithmetic | 10,459 | 83.3% |

By depth, as a share of each band's alive components:

| layers | n | impairs | helps | inert |
|---|---|---|---|---|
| 0–7 | 2,652 | 5.2% | 8.7% | 86.1% |
| 8–15 | 2,185 | 4.4% | 12.2% | 83.3% |
| 16–21 | 2,587 | 1.7% | 16.1% | 82.3% |
| 22–27 | 2,566 | 1.4% | 14.1% | 84.5% |
| 28–31 | 2,563 | 5.5% | 14.2% | 80.3% |

Impairing components are **bimodal in depth** — common early (0–15) and again at 28–31, rarest
at 16–27. Attention sites impair slightly more often than MLP sites (5.8% vs 3.2%) despite being
16% of the population.

### Before layer 22, does anything impair on *most* prompts?

Yes. By per-prompt sign over all 20,000 prompts: **427 components in layers 0–21 impair on more
than half the prompts**, 37 on more than 80%, 14 on more than 90%, none on more than 99%. All 427
are also active on more than half the prompts, so they are consistently present, not rare-case.

The 14 above 90% split into two populations:

| component | impairs_frac | measured `d_answer_ce` | `d_accuracy` |
|---|---|---|---|
| L17 `mlp.down_proj` c3 | 0.988 | −0.109 | +1.2 pts |
| L17 `mlp.up_proj` c9 | 0.979 | −0.092 | +1.0 pts |
| L13 `mlp.down_proj` c0 | 0.955 | −0.051 | +1.0 pts |
| L14 `mlp.down_proj` c6 | 0.945 | −0.047 | +1.2 pts |
| L14 `mlp.gate_proj` c79 | 0.937 | −0.048 | +1.2 pts |
| L0–L1 MLPs (7 components), L7 `mlp.down_proj` c2 | 0.90–0.93 | −0.002 to −0.022 | ~0 |

**Layers 13–17** carry few but strong impairers — removing L17 c3 alone raises accuracy by 1.2
points. **Layers 0–1** contribute a broad, systematic, but tiny drag. The strongest impairers
overall live in layer 31 (`mlp.down_proj` c9 at `d_answer_ce` −1.03), and the strongest helpers in
layer 30 (`mlp.gate_proj` c124: ablating it costs 10 accuracy points).

Most early impairers hurt **subtraction** about twice as much as addition; L21 `mlp.down_proj` c38
is the clean exception (addition-specific).

## What this cannot answer yet

The motivating question — *if the network were trained only on arithmetic, which mechanisms would
be dropped?* — is **not** answered by this table, for three reasons:

1. **83% of components are individually inert.** One component out of 12,553 rarely moves any
   metric, so "no individual effect" does not mean "not needed".
2. **Effects are not additive.** The single-component `d_answer_ce` values sum to +7.16 against a
   baseline CE of 3.87 — removing them jointly cannot cost what the sum suggests. We already saw
   this directly: the alive components as a *joint* mask score KL 0.086 while all-on scores 0.024,
   because dead components were cancelling live ones.
3. **The cross-tab that would define "droppable" is empty.** Components that are inert on
   arithmetic but matter for faithfulness (`d_kl > 0.001`) number **8** — not because such
   mechanisms don't exist, but because single ablations barely move KL either.

Also note the 454 impairers are **all still alive** in both narrow filters. The filters gate them
per prompt rather than removing them, which is another sign that "alive" is too weak a criterion
for the droppable set.

## What would answer it

1. **Minimal arithmetic subset.** Sweep the imp-min strength on the answer-CE filter (4 arms,
   3–30x) to trace accuracy versus subset size, and take the smallest subset holding ~95%.
2. **Validate the complement jointly** — ablate the whole dropped set at once and check arithmetic
   is intact while the full-vocabulary KL moves a lot.
3. **Find what the dropped set is for** — run the model with that set ablated over broad data
   (fineweb is already wired as the non-target stream), rank tokens by KL against clean, and read
   the top contexts.

Step 3 is the "why are these mechanisms still here" question, and it is inference-only.

## Reproduce

```bash
python -m param_decomp.ci_filter.scripts.run_attribution  --config <filter>/config.yaml \
    --data_root <root> --source run --verify_top 0        # screen + sign counts
python -m param_decomp.ci_filter.scripts.ablation_sweep   --config <filter>/config.yaml \
    --data_root <root> --source run --prompts 512 \
    --alive_npz <...>/addsub-05-filter-last-pos-alive/alive/kept.npz
python -m param_decomp.ci_filter.scripts.component_table  --run_dir <run>
```

On a 45 GB L40 the filter sources need `--prompts 256` (they carry one or two extra frozen
ceiling CI functions); the decomposition itself fits at 512.
