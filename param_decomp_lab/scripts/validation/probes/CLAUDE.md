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
