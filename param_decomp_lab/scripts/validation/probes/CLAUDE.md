# Fourier probes (Feucht-faithful)

A self-contained re-implementation of Feucht et al.'s (2026, "Arithmetic in the Wild",
github.com/goodfire-ai/arithmetic-wild) circular-feature probing for Llama-3.1-8B, kept
separate from the older `find_fourier_features` / `build_fourier_scatter` in the parent
folder (which probe the MLP in/out spaces with a closed-form fit). Everything here targets
**their** setup exactly.

## Pipeline

```
collect_resid_activations.py  (GPU/--slurm)  → resid_activations.npz  (resid_<site> grids)
fit_fourier_probes.py --site  (GPU/--slurm)  → probes_<site>.json     (run once per site)
build_probe_scatter.py        (CPU)          → probe_scatter/{index.html,data.js}
build_direction_scatter.py    (CPU)          → <run>/analysis/direction_scatter/{...}
```

Default artifact dir for the shared probes: `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`.
`build_probe_scatter` takes the **npz** and picks up every `probes_<site>.json` beside it — one
tab per site. `build_direction_scatter` additionally takes a **decomposition checkpoint** and drops
its applet in that run's `analysis/` folder (below).

## Ridge-CV probes (multi-layer, both ops, null-gated)

A second pipeline, independent of the Feucht-faithful one, answering *which variables
(`a`, `b`, `a+b`, `a-b`) are represented at which period, at each residual-stream
position* — with a protocol that a linearly-separable-but-not-circular code cannot pass:

```
collect_resid_stream.py   (GPU/--slurm)  → resid_stream_{add,sub}.npz   (resid_L14..L20 grids)
fit_ridge_cv_probes.py    (GPU/--slurm)  → ridge_cv_probes_<op>.json    (one job per op)
plot_ridge_cv_probes.py   (CPU, --slurm --gpus=0) → ridge_cv_heatmap.png + ridge_cv_summary.tsv
export_ridge_cv_planes.py (CPU, local)   → ridge_cv_planes_<op>/probes_L<i>.json  (Feucht format)
plot_probe_projections.py (CPU, --slurm --gpus=0) → probe_projections_<op>.png  (--all-layers: per-variable layer x period grids)
build_final_plane_scatter.py (CPU, --slurm --gpus=0) → final_plane_scatter/{index.html,data.js}
```

`build_final_plane_scatter` (template `final_plane_scatter_app.html`) fixes the probe to
`--probe-layer` (default 20) and projects every stream position onto that plane — an
applet for watching the final state get constructed: position slider + play, color by
`value mod T` or by per-point movement under a chosen block (in-plane, or full-resid
norm precomputed as `resid_delta`), displacement arrows, hover tooltips. Vanilla-JS
canvas, `file://`, smoke-test with the parent folder's `headless_check.py`.

### `build_alive_plane_scatter.py` — real model vs. alive-only circuits

Decomposition-aware variant of `build_final_plane_scatter`, dropped in
`<run>/analysis/alive_plane_scatter/` (template `alive_plane_scatter_app.html`). Requires
`find_alive_subcomponents` already run (reads `alive_subcomponents.tsv`, plus `_hidden.tsv`
when the checkpoint has a hidden CI net). Takes the checkpoint **and** the
`ridge_cv_probes_<op>.json`(s) (positional, after the checkpoint) — the probe planes are
decomposition-independent and reused as-is.

Recomputes fresh activations for **two or three sources** (three on a `dual_hidden_ci`
checkpoint, two otherwise) over one sampled `a<op>b=` grid — the real model, and one circuit
per alive list with only that role's components active everywhere (delta off —
`find_alive_subcomponents`'s masking) — so every source is exactly point-aligned. Running it
on both a baseline and a dual run of the same decomposition is how you see what adding the
hidden-CI net actually changed. The `<result> @ layer` own-probe row is dropped (own rows:
`a`, `b` only; the result row is fixed to one probe layer at a time). Extra colorby modes
beyond `final_plane_scatter`'s: **distortion** (plane-projected distance from the real
model; disabled while viewing the real model itself) and **causal importance** (role +
matrix + component pickers, gated by that role's alive list; CI is one value per prompt
from the real-model forward, so it colors every panel identically regardless of which
source is shown).

**frame** (client-side only) sets what a panel's origin means, and is the difference between
a picture the arrows are meaningful in and one they are not.

- **activation plane** (default) — points are `(x·e1, x·e2)` on the plane's *orthonormal*
  basis (`common.probe_plane_basis`), with the probe bias **dropped**. The origin is then the
  true activation-space zero, so a ring that looks off-centre *is* off-centre in the model.
  Because the target is bias-free (Llama has no `attention_bias`/`mlp_bias`, and RMSNorm
  rescales but never recentres), a component's read `x·V` vanishes on a hyperplane through
  that zero — so an arrow drawn from it makes the honest angle with every point, and since
  the default frame is square in data units on a square panel, screen angles *are* in-plane
  angles.
- **probe prediction** — the old view: `(x·w_cos + b_cos, x·w_sin + b_sin)`. Its origin is
  the *mean* activation (ridge's unpenalised intercept pins mean prediction to the mean
  target, which is ~0 on a period-commensurate value grid), and its axes are skewed, since
  `w_cos`/`w_sin` are neither orthogonal nor equal-norm. Angles there are not in-plane
  angles. Keeps the ring near unit radius (the dashed unit circle is drawn only here) and
  is the better view for judging probe fit.

Both come from **one** shipped projection: `ops[op].xf` carries the per-plane transform
(`pred_cos = k·p1 + bc`, `pred_sin = m1·p1 + m2·p2 + bs`, exact since `w_cos = k·e1` and
`w_sin` lies in the plane), so `pred` is rebuilt in-browser rather than doubling a ~100 MB
`data.js`. Those coefficients are shipped at full precision — rounding them to 4 significant
figures like the calibration scalars puts a ~1e-3 relative error into the reconstructed
frame, because `m1·p1` can cancel against `m2·p2`. Arrow lengths differ by orders of
magnitude between frames, so `meta.arrows.calib` holds a `norm_hi`/`norm_default`/
`mult_default` triple per frame and the threshold slider rescales on switch.

Empirically the two origins are *far* apart: measured against L18's own probes, the ring's
centroid sits **0.17–1.07 ring radii** from the true zero depending on the cell (`a+b` T=10
is 1.07 — the true zero lies roughly on the ring). So this is not a cosmetic change.

**Component arrows** overlay each alive subcomponent's own direction on the panel's plane,
drawn from the origin as the displacement it contributes (`dir·e1, dir·e2`; the probe bias
never applies to an arrow in either frame, since a panel maps `x → x·w + b` and adding `dir`
to `x` moves the point by `dir·w` regardless of `b`). A dropdown picks the **U** vectors of the components that *write* to the stream or
the **V** vectors of those that *read* from it. Directions use the same gauge-invariant
product form as `build_direction_scatter` (`V[:,c]·‖U[c]‖` read, `U[c]·‖V[:,c]‖` write), and a
read absorbs its RMSNorm gain (`γ ⊙ V`) because the component sees the normalised stream while
the panels plot the raw one. Candidates are the **union** of the output- and hidden-alive
lists; the tooltip says which lists each came from.

Each arrow is drawn **only in the column whose stream position that component actually
touches** — `_arrow_site` derives that from `(layer, matrix)`, so a decomposition spanning
many layers works without changes:

| matrix | role | site |
|---|---|---|
| `mlp.down_proj` | write (U) | `L{ℓ}` |
| `self_attn.o_proj` | write (U) | `L{ℓ}att` |
| `mlp.{gate,up}_proj` | read (V) | `L{ℓ}att` |
| `self_attn.{q,k,v}_proj` | read (V) | `L{ℓ-1}` |

A component whose site isn't among the captured positions has no panel to draw on and is
dropped; for a single-layer decomposition most columns stay empty. Two sliders: **min |proj|**
(a floor on the in-plane 2D norm, in data units, defaulting to the 90th percentile of the
shipped norms) and **length** (a shared multiplier, ±2 decades around `mult_default`, which
puts the 99th-percentile arrow at one data unit — a probe maps a `d_model` activation to a
~unit cosine, so `‖w‖` is tiny and raw `dir·w` projections are orders of magnitude shorter
than the cloud). Relative lengths are always preserved, so arrows stay comparable. Hovering an
arrowhead — the outer half of the shaft is the hit target, since every arrow starts at the
origin and the inner half is an unresolvable pile-up — names the subcomponent and gives
`|proj|` plus the angle to the plane (0° = it lies in the plane, 90° = orthogonal), taken
against the plane's orthonormal basis via the shared `common.probe_plane_basis`.

**view** (client-side only, no data regeneration needed — everything is already in
`data.js`) toggles the grid's column axis: **all layers** (default) fixes one period and
sweeps stream positions, matching `final_plane_scatter`; **all periods** fixes one layer
(a new `layer` picker replaces the `period` picker) and sweeps periods instead, useful for
comparing a single position across every period at a glance. The Δ-from-previous-position
colorby modes (`plane`/`resid`) and displacement arrows are layers-view-only — there's no
"previous" position when columns are periods — and get disabled (auto-falling back to
`value`) when switching to the periods view.

**`--probe-layers`** (default `18`, the decomposed layer; comma-separated for more — e.g.
`18,20`) picks
which layer(s)' probe the result row is fixed to. The ridge-CV fit already has a probe per
layer, so adding more probe layers costs only extra CPU-side projection of the *already
captured* activations, not extra GPU forward passes. The applet always shows a `probe
layer` dropdown (defaulting to L20 when it's among the prepared layers), populated from
whatever `data.js` was actually built with — a single-option dropdown when only one probe
layer was prepared, several when more were.

**Axes and zoom** (client-side only): every panel draws dashed zero-lines plus tick labels
giving each axis's scale and sign. The *default* frame is centred on the origin and sized
to fit the **real model's own** point cloud specifically (not whichever source is
currently displayed, so panels don't jump when switching `data`) with 15% slack — in the
activation-plane frame that deliberately leaves the cloud sitting off-centre by however far
it really is from zero, with the origin at the panel's middle. Computed
from `op`/`view`/`period`/`layer`/`probe-layer` alone, never from the sampled points'
own asymmetric range or which decomposition produced the run, so two separately generated
applets on the same base model + probes default to the same framing (their `n_show`
samples are close enough, especially at the shared default seed, that the real model's
cloud extent barely varies run to run). Scroll zooms toward the cursor, drag pans,
double-click resets — one row's panels share a frame (as they already did for comparing
`data` sources), so any interaction on one panel moves the whole row.

Batches are grouped by each sampled prompt's natural (non-zero-padded) token length rather
than padded, since `ComponentModel`'s masked forward has no attention-mask support — this
also keeps tokenization identical to what `collect_resid_stream.py` used to fit the probes,
so reusing their weights on freshly captured activations stays valid. Only 2D/1D projected
points + scalar deltas + CI (base64 uint8) + the arrows' 2D projections and angles (base64
float16) are ever serialized, never raw `d_model` activations. Site gating keeps the arrow
block small — a component only needs the planes its own column can display (that position's
own `a`/`b` probes plus each prepared probe layer's result probe), not one per stream
position.

```bash
uv run python -m param_decomp_lab.scripts.validation.probes.build_alive_plane_scatter \
    "$MODEL_PATH" ~/out/runs/fourier_probes/ridge_cv_probes_{add,sub}.json --slurm
```

`export_ridge_cv_planes` is a pure format transform (the shipped weights are the fit's
full-range refit; `r2_cos = r2_sin = cv_r2`, with `cv_r2` / `p_value` / `lambda_rel` /
`accepted` riding along). `plot_probe_projections` scatters a prompt subsample on each
accepted probe's predicted (cos, sin) plane — best accepted layer per (variable, period)
cell, colored by `v mod T` on a cyclic colormap, gated cells left grey.

- **Collection**: decoder-block outputs of layers 14..20 (`resid_L<i>` = stream *after*
  block `i` = input to block `i+1`) at the `=` token, full `a<op>b=` grid, `a, b ∈ 1..200`,
  for `+` and `-`.
- **Fit** (`fit_ridge_cv_probes`): closed-form ridge per (layer, variable, period) on raw
  per-prompt activations. λ is selected by CV over **rotating contiguous value blocks**
  (5 folds × two contiguous deciles of the variable's range held out), the reported probe
  is **refit on the full range** at the selected λ (homogeneous accuracy; `block_r2`
  verifies), and every cell is **gated by a permutation null** — the identical pipeline
  rerun with values shuffled across prompts (`p_value`; `n_perm=20` default). Solved in
  the train-Gram eigenbasis (one eigh per fold/permutation), fp32 GPU.
- **Plot**: small-multiples heatmap ops × variables, layer × period per panel; cells with
  `p > alpha` *or* `cv_r2 ≤ 0` are greyed (beating the null with a negative held-out R²
  is not a probe). The TSV carries `cv_r2`, `p_value`, `lambda_rel`, `cv_angerr_deg`,
  `full_r2`, `min_block_r2`, `max_null_cv_r2` per cell.

All three accept `--dependency=<jobid[:jobid...]>` so the chain can be submitted in one
go with `afterok` dependencies. Caveat for reading results: for T comparable to the value
range (T=100 on operands' 1..200), a contiguous held-out block removes a whole arc of the
circle from one of the two cycles, so `cv_r2` demands genuine angular extrapolation there
— calibrate expectations against the per-T null, not across Ts.

## Four sites (around the MLP)

The collector captures the layer-18 residual stream at four points around the MLP, in one
forward pass (`RESID_SITES` in `common.py`, computation order); `fit_fourier_probes --site`
fits the same probes on each, and the applet shows one tab per site:

- **`pre`** — input to `post_attention_layernorm`: residual after attention, **before the MLP**.
- **`norm`** — the MLP's actual input: `post_attention_layernorm` output (after RMSNorm).
- **`mlp_out`** — the MLP output: the **Δ** the MLP writes into the residual stream.
- **`post`** — the **decoder-block output** (`out[0] = pre + mlp_out`): after attention *and*
  MLP (Feucht's `source="resid"`). This is what Feucht ship.

Comparing sites isolates the MLP's role: the operand (`a`/`b`) curves are ~identical across
sites (operands sit in the residual already), while `a+b`'s peak R² is higher post-MLP than
pre-MLP — the `mlp_out` tab shows what the MLP itself writes for the sum.

## Fidelity to Feucht

- **Site**: Feucht's is `post` (the block output); the other three are our added controls.
- **Data**: `{a}+{b}=` for `a, b ∈ 1..200` (40 000 prompts), **last token** (`=`), left-padded
  so position `-1` is always the `=`.
- **Probe**: per period `T` and variable `v ∈ {a, b, a+b}` (their input / offset / output), two
  `nn.Linear(d_model, 1)` (bias) probes for `cos(2πv/T)` / `sin(2πv/T)`, Adam(lr 1e-3) + MSE,
  **500 epochs**, on **raw** (un-standardised) activations. `train_test_split(test_size=0.2,
  random_state=42)`; run order `a` outer / `b` inner matches their prompt order, so the split
  coincides. `r2_*` is the held-out (test) R². Period 2 has `sin≡0` → its sin probe is skipped.
- **Default = full per-variable sweep**: every `T ∈ 2..min(v.max()//2, max_period)` (`max_period`
  150; `a`,`b` → 2..100, `a+b` → 2..150), ~700 probes. `--periods` fits an explicit shared list.
- Closed-form least squares would give a higher R² / tighter circles; we deliberately keep their
  Adam-500-epochs fit so the results match theirs.

## Applet — the control

`build_probe_scatter` leads with an **R²-vs-period curve** (one line per variable), under a
**site tab** (before / after the MLP). This is the control: R² should spike only at the model's
true periods. Empirically `a+b` spikes sharply at periods 2/5/10 (+a broad 50–100 harmonic hump)
and sits at ~0 elsewhere — clicking a spike period draws a **circle**, a valley period a **blob**.
(`a`/`b` stay high at every period because a single operand is ~directly decodable, so any function
of it is too; `a+b` is the discriminating variable.)

Clicking the curve picks the period **on the current basis** (the **basis variable** dropdown
chooses which of `a`/`b`/`a+b`), and sets the colour **mod** to that period so the residue classes
line up. It draws the scatter: activations projected onto that probe's predicted `(cos, sin)` plane.
Other controls: **colour by** `a`/`b`/`a+b` with **mod** + **offset** (`(value − offset) mod m`,
scale `0..m-1`), zoom/pan, hover. A fixed `n_show` random subset of points is shipped per plot,
**per site**, to bound `data.js`. Vanilla-JS canvas, no CDN — smoke-test with the parent folder's
`headless_check.py`.

## Direction scatter — Feucht Fig 9c overlay (`build_direction_scatter`)

Same point clouds, but tied to a **decomposition run** (its checkpoint) and dropped in
`<run>/analysis/direction_scatter/`. Over each cloud it draws **arrows** for the directions of MLP
**neurons** or the run's **subcomponents** (a dropdown), reproducing Feucht Fig 9c (down_proj rows
projected onto the T-Fourier plane) and generalising it to reads and to the decomposition:

- **read** arrows on the operand planes (`a`, `b`): the residual read direction — neuron `gate`/`up`
  row, or subcomponent `gate`/`up` `V[:,c] · ‖U[c]‖`. Tagged `g` / `u` (both shown).
- **write** arrows on the sum plane (`a+b`): the residual write direction — neuron `down_proj[:,n]`,
  or subcomponent `down_proj.U[c] · ‖down_proj.V[:,c]‖`. Tagged `d`.

Subcomponents scale the residual-space vector by the norm of the component's *other* (14336-d)
vector — symmetric across read (`V·‖U‖`) and write (`U·‖V‖`). A rank-1 `u vᵀ` is gauge-free
(`u→αu, v→v/α`); this product form is gauge-invariant and equals the residual move per one std of
the unit's activation — the same quantity a raw neuron row/column is, so both are neuron-comparable.

**Fig 9c scale**: everything is projected onto the **unit-normalised** probe directions `d = w/‖w‖`,
and the cloud is recomputed in that same normalised, mean-centred frame — so `v·d` (arrow, from the
ring centre as an increment) and `x·d − mean` (activation) share one scale. Only the top-`top_k`
units by projected 2D norm are shipped per plane; the **threshold** slider filters those by `|v·d|`.

**Selection panel** (ported from `build_fourier_scatter`): click an arrowhead to open a right panel.

- **Angle** to the current Fourier plane (0° = the direction lies in the plane, 90° = orthogonal),
  from the orthonormal in-plane fraction of the *unit* direction. Shipped per shown arrow (`ang`), so
  it tracks the current site/period; both neurons and subcomponents.
- **`dir · activation(a, b)` heatmap** — subcomponents only (its inner activation for reads,
  write-projection for writes). Too many neuron directions to ship a grid each, so it's reconstructed
  in-browser: each site's activation grid gets a rank-`heat_rank` factorisation (`randomized_svd`),
  shipped as a shared spatial basis `P` plus a per-subcomponent `qd = Vᵀ·dir` and `m = dir·mean`;
  `dir·act(a,b) = m + P·qd`. Exact within the top-`heat_rank` subspace — on real component directions
  (`R=48`) the reconstruction correlates ~0.93+ with the exact grid. Uses the **current site**.
- **Colour by CI**: tints points by the selected subcomponent's causal importance, read from the run's
  `analysis/datasets/inner_activations_add.tsv` `ci` column (a, b ≤ 100; points outside are grey).
  Only the subcomponents that harvest covered are present; others fall back to grey.
- **Show fit vs. held-out** (checkbox; appears when the probes JSON carries a `block_ranges`
  field — per-variable held-out value ranges, e.g. probes fit with contiguous value blocks
  excluded): held-out points render at full opacity, in-fit points fade to 25%, while colours
  keep following the colour-by mode (a/b/a+b/CI). A visual generalization test: held-out
  points land on the ring in phase only for real periods.
- **`--alive-tsv <run>/analysis/datasets/alive_subcomponents.tsv`**: restricts the subcomponent
  arrows and heatmap basis to the run's alive components (original indices kept; neurons
  unaffected). The hint line reports the alive counts per tag.

Arrowheads render at Fig-9c-legible size. `n_show` defaults to 8000 to keep `data.js` ~85 MB with the
added heatmap basis + CI grids.

The target model must be the same base model the checkpoint decomposes (it always is — L18 MLP
decompositions freeze base Llama-3.1-8B), so the shared `resid_activations.npz` + `probes_<site>.json`
apply to any run. Sanity check baked into the generator: on `post`/`a+b`/`T=10` the top-8 write
neurons reproduce Feucht's hard-coded mod-10 addition-neuron list exactly.
