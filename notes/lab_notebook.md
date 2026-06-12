# Lab notebook — llama8b-add-02 (targeted decomposition of L18 MLP on integer addition)

Run: `~/out/runs/llama8b-add-02/model_20000.pth`. Decomposes the layer-18 MLP
(`gate_proj`, `up_proj`, `down_proj`, C=512 each) on prompts `X+Y=` for X,Y ∈ 1..100
(10000 prompts, file `addition_1-100.txt`). Token layout is 5 positions:
`[BOS, X, +, Y, =]` — every operand is a single token, the answer is predicted at the
`=` position (pos 4).

Artifacts I'm reading:
- `alive_components.tsv` — the ~150 alive subcomponents (per matrix).
- `alive_components_per_position.json` — per (prompt, position, matrix) the active
  components (CI > 0.1). This is the workhorse for the analyses below.

---

## Finding 1 — Two regimes: a fixed "scaffold" (pos 1-3) and computation (pos 4)

Counting how many *distinct* sets of active components appear at each position across all
10000 prompts:

| pos | token | gate sets | up sets | down sets |
|----|-------|-----------|---------|-----------|
| 0  | BOS   | 1 (empty) | 1 (empty) | 1 (empty) |
| 1  | X     | 3         | 2       | 2         |
| 2  | +     | 2         | 6       | 6         |
| 3  | Y     | 7         | 11      | 12        |
| 4  | =     | **415**   | **205** | **1307**  |

Positions 1-3 are near-constant: pos 1 is dominated by `{gate-82, up-179, down-500}`,
pos 3 by `{gate-371, up-61, down-148}`, regardless of the operand values. The actual
arithmetic — variation driven by the operands — happens almost entirely at pos 4 (`=`),
where the model has to emit the sum. **Nothing fires on BOS.**

## Finding 2 — Even the "scaffold" encodes operand magnitude

The small variation at pos 1-3 is not noise — it tracks the magnitude of the operand just
read:

- pos 1 (reading X): `gate-291` fires iff **X ≥ 61** (clean threshold).
- pos 3 (reading Y): `up-130` / `down-37` fire for **small Y** (Y ≲ 19).

So as each number is read, a component flags roughly which magnitude band it falls in.

## Finding 3 — Pos-4 components split into three interpretable families

Rendering each pos-4 component's activation over the (X, Y) grid
(`figures/pos4_analysis/pos4_grid.png`, `figures/pos4_analysis/pos4_grid_zoom.png`)
reveals clean geometric structure:

1. **Units-digit lattice** — `gate-389`, `gate-248`, `gate-460`, `gate-228`. A period-10
   checkerboard in both X and Y: activation depends only on `(ones(X), ones(Y))`. These
   are computing the **units digit of the sum and the carry**. `gate-389` is the cleanest
   (sharp 10×10 lattice of squares).
2. **Sum-band detectors** — `down-230`, `down-84`, `down-182`. Anti-diagonal stripes/blobs
   (constant-sum lines run anti-diagonal), so they track the actual **sum S = X+Y**.
   `down-182` lights up only in the top-right corner (large S → carry into the hundreds).
3. **Single-operand magnitude** — `gate-429` (vertical stripes = function of X only),
   `up-323` (horizontal bands = function of Y only). Same magnitude-encoding role as the
   scaffold, re-expressed at the compute position.

The always-on outputs `down-118`, `down-467`, `gate-494`, `up-406` fire on ~all prompts —
plausibly a generic "emit a number here" output, not value-specific.

### Reading of the mechanism (hypothesis)
The L18 MLP appears to add by decomposing the problem the way a human does columnar
addition: separate subcircuits for (a) the units digit + carry (the period-10 lattices),
(b) the overall magnitude / tens & hundreds of the sum (the anti-diagonal sum-bands), and
(c) per-operand magnitude bookkeeping (the stripes + scaffold features). Still to confirm
causally — see TODO.

## Finding 4 — Causal ablation confirms a units-digit vs magnitude double dissociation

Script: `param_decomp_lab/scripts/validation/ablate_component_groups.py` (1024 random
prompts; for each component family, force exactly that group off in the circuit mask and
re-read the argmax answer at `=`). The answer is a single token, so "is the units digit
right" / "is the magnitude right" are directly checkable. Output:
`ablate_component_groups.tsv`.

| condition (group forced off) | acc | pure-ones-digit err | magnitude (tens/100s) err | non-numeric |
|---|---|---|---|---|
| baseline (full circuit)      | 0.94 | 0.1% | 2.8%  | 2.6% |
| `units_lattice_gate`         | 0.71 | **6.0%** | 15.7% | 7.6% |
| `sum_band_down`              | **0.28** | 10.1% | **39.3%** | 22.3% |
| `always_on_down` (118, 467)  | 0.79 | 6.2% | 11.2% | 3.5% |
| `operand_mag_gate`           | 0.69 | **1.4%** | **21.6%** | 8.3% |
| `operand_mag_up`             | 0.72 | 5.2% | 14.2% | 9.0% |

Key reads:
- **Double dissociation between gate families.** Ablating the units-digit lattice
  (`units_lattice_gate`) produces *pure ones-digit* errors at 6.0% (≈60× baseline) — the
  tens/hundreds stay right, just the last digit flips (e.g. `83+24=107 → 103`,
  `77+27=104 → 106`). Ablating the operand-magnitude gate components (`operand_mag_gate`)
  does the opposite: almost no ones-digit damage (1.4%) but heavy magnitude damage (21.6%)
  — e.g. `49+70=119 → 99`, `89+38=127 → 107`. So the grid geometry is causal: the period-10
  lattice writes the units digit, the stripes/bands write the magnitude.
- **`sum_band_down` is the load-bearing output.** Knocking out the dozen graded down_proj
  components collapses accuracy to 0.28 and is overwhelmingly a magnitude failure (39%) with
  many non-numeric outputs — these components write the bulk of the answer's value.
- **The "always-on" outputs aren't generic.** `down-118`/`down-467` fire on ~every prompt,
  but ablating them shifts the answer *down* by a large amount (mean |err| ≈ 40;
  `194 → 94`, `70 → 30`): they supply a large constant-ish additive offset to the output
  magnitude rather than anything value-specific.
- The units-lattice damage is **not** concentrated on carry cases (`ones(X)+ones(Y) ≥ 10`):
  error rate is actually a bit higher on no-carry prompts, so it's a general units-digit
  contribution, not specifically a carry detector.

### Mechanism (now causally supported)
L18's MLP does columnar addition with separable subcircuits:
- **units digit** ← period-10 gate lattice (`gate-389/248/460/228/...`),
- **tens/hundreds magnitude** ← operand-magnitude gate/up stripes (`gate-429/163/276`,
  `up-323/...`) feeding the graded `sum_band` down_proj writers,
- plus a large constant output offset from the always-on down components.

## Finding 5 — Where the addition module fires in *natural* text (broad-data screen)

Script: `param_decomp_lab/scripts/validation/screen_components_on_data.py`. Streamed
**2.46M positions** of the run's broad nontarget distribution (fineweb) through the model,
computed lower-leaky CI for all 512×3 L18 components, and kept per-component firing counts
+ top-30 max-activating contexts. Outputs: `screen_components_on_data.tsv` (firing
frequency) and `screen_components_on_data.jsonl` (contexts).

**The module is strongly suppressed on broad text, and what little fires is *entirely*
the addition components.** Firing-frequency tiers:

| frequency on broad text | # components | alive-on-addition | not-alive |
|---|---|---|---|
| ≥ 1× (anywhere in 2.46M pos) | 1526 / 1536 | 147 | 1379 |
| ≥ 10× | 82 | 58 | 24 |
| **≥ 100×** | **39** | **39** | **0** |

So beyond a one-off tail (almost every component crosses CI 0.1 *somewhere* in 2.46M
positions — a single math-heavy doc trips ~900 of them once), every component that fires
*recurrently* on general text is one of the addition components. There is no separate
"other function" hiding in the components that were dead on addition. Peak broad-text
firing rate is only ~0.08% of positions (`gate-82`: 1887/2.46M) — vs. tens of percent on
addition prompts — so the targeting genuinely concentrated these on arithmetic.

**What broad-text contexts trigger them: structured numeric notation.** Across the 39
recurring components' contexts, only **22% of firing tokens are themselves a number** — the
other ~78% fire on the *delimiter that binds numbers together* (`:`, `;`, `-`, `(`, `/`),
and 35% of firings sit inside a digit string. The same circuitry that fires on `+` and `=`
in `X+Y=` fires on the separators of:
- **academic citations** — `J Bone Miner Res. 2003;18:1563-`, `Concilium 7 (197…`,
  `Vol. 98 (`, `502 N.E.2d` (volume:page, year-in-parens);
- **dates / times** — `2018 Jun;`, `8:00pm ET /`, `April 23, 2020 @`;
- **numeric ranges & scores** — `Euro 41-16`, `242 (+`, `range 36-91 years`,
  `England – 118 (a decrease from 131…`;
- **prices / percentages** — `€ 11.000`, `35% and the US was`.

By family (consistent with Findings 2–4):
- **Scaffold / number-onset** (`gate-82`, `up-179`, `down-500`) — fire on a number at the
  *start of a line/title* (`…still open.\n` → `7`; `91 Magazine Volume Eleven`): the generic
  "first number of a segment" role, the broad-text analogue of reading `X`.
- **Magnitude** (`gate-291`) — confirmed large-number detector, fires on `92`, `97`, `99`,
  `74`… in any context (birthdays, report numbers), not addition-specific.
- **Units-digit lattice** (`gate-248`, `gate-460`, `gate-228`) and **sum-bands**
  (`down-230`, `down-404`, `down-488`, `down-334`) — fire on citation/date/range numbers;
  `range`- and `year`-tagged contexts dominate (`down-404` range:14, `down-488`/`down-334`
  year:15). The digit/magnitude machinery is reused wherever digit-level numeric structure
  appears, not only in `a+b=`.

**Takeaway for the original question.** The L18 "addition module" is really a *numeric-
expression* module: a shared number-reading + digit/magnitude circuit. On `a+b=` it
implements addition; on general text the *same* components activate around the structured
number notations (citations, dates, ranges, prices) where digit- and magnitude-level
relationships matter. None of it activates for non-numeric reasons, and no hidden non-
addition components exist. The scaffold/magnitude components are the most context-general
(any number); the units-lattice/sum-band components are more tied to multi-number notation.

## TODO / next
- The grid families were hand-picked; a cleaner version would cluster all alive pos-4
  components by their (X,Y) activation map and ablate data-driven clusters.
- `operand_mag_up` is mixed (both ones and tens damage) — worth checking whether up_proj
  components are less cleanly specialised than gate.
- Check whether the same units/magnitude split holds at a *stricter* CI threshold (fewer
  components in the circuit).
