# Roadmap: addition analysis — work log

Concise log of work on `roadmap_addition_analysis.md`. Problems noted inline.

## Reference run

`addmult-L18-03` — L18 MLP (gate/up/down) + attn decomposed, trained on add+mult prompts.
- checkpoint: `~/out/runs/addmult-L18-03/model_28000.pth`
- target: Llama-3.1-8B, `d_model=4096`, `d_int=14336`, `C=456` per matrix.
- per-op prompt files already exist in `param_decomp_lab/experiments/lm/prompts/`
  (`addition_1-100.txt`, `subtraction_1-100.txt`, `multiplication_1-100.txt`); grid is
  `1..100 × 1..100` and prompts are unpadded constant-length, so the **last token = index
  -1 (the `=`)** uniformly.

## Clarifications resolved (before coding)

- Operators: add `+`, sub `-`, mult `×`. Per-op data files suffixed `_add` / `_sub` / `_mult`.
- New scripts scope to the **three L18 MLP matrices only** (ignore decomposed attn).
- "alive" = flagged by existing `find_alive_components` (run per-op) **AND** mean CI > thr
  (default 0.1). Mean CI taken at the **last token** over the 100×100 grid.
- Obj 1 stored as **npz** (not JSON — full hidden states are ~410M floats, JSON would be
  multi-GB and unloadable by the Obj 5 browser applet).
- Obj 4: gate+up share both spaces, so one figure (V|U side by side); down_proj is a
  separate figure (its U/V live in different dims from gate/up).

## Pipeline / file chain

0. (existing) `find_alive_components --prompts=<op file> --output=alive_components_<op>.tsv
   --output-json=alive_components_per_position_<op>.json`  [GPU]
1. `collect_hidden_activations` → `hidden_activations_<op>.npz`  [GPU]
2. `collect_inner_activations` → `inner_activations_<op>.tsv` + `alive_filtered_<op>.tsv`
   (mean-CI filter applied here)  [GPU]
3. `compute_subcomp_periods` → `subcomp_periods_<op>.tsv`  [CPU]
4. `plot_subcomp_cosine` → `figures/.../cosine_gate_up_<op>.png`, `cosine_down_<op>.png`  [CPU]
5. `build_neuron_connection_explorer` → `figures/neuron_explorer_<op>/{index.html,data.js}` [CPU]

## Progress

### Obj 1 — `collect_hidden_activations.py` ✅
Forward hooks on L18 `post_attention_layernorm` (pre=resid, post=mlp_input), `mlp.{gate,up,down}_proj`
outputs; plain forward (no masks) so captures are the true target in/outputs. Stores five
`[100,100,dim]` float16 grids at the last token.
- Ran on `add` (job 1582): `hidden_activations_add.npz`, 595 MB. Shapes verified
  (resid/mlp_input/mlp_output 4096; gate/up 14336). No problems.
- Setup: ran existing `find_alive_components --prompts=addition_1-100.txt` (job 1583) →
  `alive_components_add.tsv` (822 alive; 652 in MLP: 219 gate / 197 up / 236 down) +
  `alive_components_per_position_add.json` (consumed later by Obj 5).

### Obj 2 — `collect_inner_activations.py`
Normalized inner act `(x·V_c)/||V_c||` at last token via einsum on the cached module input
(input is already post-RMSNorm for gate/up, post-SwiGLU for down). Mean-CI filter over the
grid applied here → `alive_filtered_add.tsv`; full grid → `inner_activations_add.tsv`.
- **Problem (job 1584):** `einsum().cpu().numpy()` raised `unsupported ScalarType BFloat16` —
  under bf16 autocast the matmul returns bf16 even though both inputs were `.float()`. Fixed
  by `.float()`-ing the einsum *result* before numpy (same gotcha noted in collect_ablation_kl).
  Resubmitted as job 1585 → OK.
- Result: **38/652** MLP components pass mean-CI > 0.1 (16 down / 7 gate / 15 up). Note the
  0.1 mean-CI default is fairly strict (mean over the whole 100×100 grid); lower `--mean-ci-thr`
  to widen the set.

### Obj 3 — `compute_subcomp_periods.py` ✅
CPU. Reads `inner_activations_<op>.tsv` (+ `alive_filtered_<op>.tsv` for layer/full-matrix),
rebuilds each `[N,N]` grid, measures periodicity of the `f(a)`/`f(b)` marginals via
autocorrelation (best lag in `1..N//2`) and FFT (peak frequency → period). Representative
`period`/`period_axis` taken from the stronger FFT peak.
- Detected periods cluster at 2/5/10/20/50/100 — the expected modular structure.
- **Note:** autocorrelation often returns lag 1 with high score for smooth/monotone marginals
  (lag-1 correlation dominates); FFT is the cleaner periodicity signal, hence used for the
  representative period. Both metrics stored per spec.

### Obj 4 — `plot_subcomp_cosine.py` ✅
CPU, mmap U/V. Two figures: `cosine_gate_up_add.png` (22 comps, V|U side by side) and
`cosine_down_add.png` (16 comps), sorted by period with thick separators, `RdBu_r`
(positive=red), vmin/vmax ±1. Labels `{g,u,d}{comp}·p{period}`.
- Decision: used `RdBu_r` (not literal `RdBu`) so positive=red / negative=blue, matching
  Obj 5's stated colour semantics.
- Result: clear period-block structure in the gate/up V vectors (esp. the p50 group);
  U vectors near-orthogonal (14336-dim neuron space).

### Obj 5 — `build_neuron_connection_explorer.py` + `neuron_connection_explorer_app.html` ✅
CPU generator + dependency-free vanilla-JS/SVG applet. Connection strength = `U[c,j]*||V_c||`
(gate/up write) or `V[j,c]/||V_c||` (down read). Left=gate/up active subcomps (up on top,
period-sorted), center=neurons above the conn threshold sorted by their strongest gate/up
driver, right=down subcomps. Lines red/blue by sign; hover subcomp → CI (a,b) heatmap, hover
neuron → up/gate/silu(gate)·up. Neuron up/gate grids shipped as fp16 base64 (decoded in JS).
- **Sizing:** connection strengths are small in 14336-dim space (median |w|≈0.10, p90≈0.14,
  max 0.51), so floor 0.05 + UI thr 0.10 flooded the view (360 neurons / 64.9 MB data.js).
  Raised `conn_floor` default to 0.1 and UI default threshold to 0.15 → 613-neuron universe,
  **33 MB**, ~60 neurons shown. Limitation logged in spec: UI threshold below `conn_floor`
  can't surface new neurons (lower `--conn-floor` to widen, at the cost of file size).
- Playwright (headless Chromium via `headless_check.py`): no JS errors; verified node/line
  counts, subcomp-hover CI panel, neuron-hover up/gate/output, and operand re-render.

### Docs ✅
All five scripts documented in `spec.md` (new "Arithmetic analysis" section) and runnable
per-op examples added to `commands.md`. `common.py` helpers noted in the validation `CLAUDE.md`.

## Post-review changes

Code review (`/code-review high`) surfaced findings; user direction applied:
1. **Alive-set architecture redesigned.** `find_alive_components` stays op-agnostic and
   unsuffixed (ever-alive on the run's *original* data); downstream scripts read its
   `alive_components.tsv` / `alive_components_per_position.json` and do the per-op +
   last-position + mean-CI filtering themselves. `collect_inner_activations` default
   `--alive-tsv` → unsuffixed `alive_components.tsv`. Re-ran: candidate pool 652→1207 MLP,
   still **38** pass mean-CI>0.1 (mult-only-alive comps don't clear the addition filter).
2. **Explorer op-regex fix.** `_ci_grids` now matches `op_symbol(op)` exactly (not `\D`) and
   asserts ≥1 prompt hit — the unsuffixed JSON holds both `+` and `×`, so `\D` would have
   mis-binned `a×b=` into the addition grid. Reads unsuffixed `alive_components_per_position.json`.
3. **Grid-coverage assert** added via `common.square_grid_size` (full unique `n×n` or raise),
   used by both collectors.
4. **Hoisted** `_load_uv`→`load_component_uv` and `_read_periods`→`read_subcomp_periods` into
   `common.py`; both new files import them.
5. `_sparse_grid` duplication left as-is (per user).

**New feature:** explorer "hover shows" toggle — subcomponent heatmap switches between causal
importance and signed normalized inner activation (diverging colormap). Inner grids read from
`inner_activations_<op>.tsv`, shipped fp16 base64. Re-ran the full chain (inner re-run on GPU
job 1586; periods/cosine/explorer on CPU). Headless Chromium: no JS errors; toggle verified
(CI 1.000 ↔ inner 2.766 at (2,3)).

### Obj 6 — `reduce_dimensionality.py` + `dimensionality_explorer_app.html` ✅ (run: llama8b-add-02)
Projects the stored `mlp_input` / `mlp_output` onto the alive subcomponents' read/write
subspaces (orthonormal `Q = Dᵀ E Λ^(-1/2)` from the direction Gram), giving `z = Qᵀx`. Reports
geometric rank, PCA participation ratio, TwoNN (scikit-dimension), and the variance-captured
completeness fraction. Plotly self-contained applet (scree + threshold, raw-z & PCA 3D views
with floor shadows, a/b/a+b colour).
- Added `plotly` + `scikit-dimension` to `param_decomp_lab` deps; needs `uv sync --all-packages`
  (plain `uv sync` only audits the root and silently skips workspace-member deps).
- **Bug:** forgot the `window.PD_DATA = /*__PD_DATA__*/` injection line in the template, so the
  data marker matched nothing and the applet loaded with `PD_DATA` undefined (plotly itself was
  fine). Added the line + a fail-fast assert that both injection markers exist in the template.
- **Headless note:** `scatter3d` is WebGL; headless Chromium can't render it to a screenshot
  ("CONTEXT_LOST_WEBGL"), so only the colorbar shows. No JS errors; scree/stats/controls verified.
- Results (llama8b-add-02): input rank 20 → PR 10.9, TwoNN 6.3, 16.9% variance captured; output
  rank 14 → PR 6.6, TwoNN 6.6, 18.6%. Redundancy shows as PR ≪ rank (directions correlated, not
  collinear); only ~17–19% of total activation variance lives in the alive subcomponent subspace.

### Obj 7 — `find_independent_subspaces.py` + `isa_explorer_app.html` ✅ (run: llama8b-add-02)
ISA on the reduced `z`: PCA-reduce to `--var-keep` variance, FastICA, then group components
into subspaces by energy (magnitude) correlation; near-orthogonality checked via principal
angles. Plotly applet: per-subspace 3D scatter (a/b/a+b colour) + energy-correlation and
principal-angle heatmaps.
- **Tuning:** first run (`var_keep=0.99`) kept 17/12 components (≫ the ~11/6.6 effective dims),
  FastICA didn't converge, and grouping was nearly all singletons. Lowered default `var_keep`
  to 0.9 and raised `max_iter` → converged; input 11 ICs → 9 subspaces (two 2-D blocks + seven
  1-D), output 8 ICs → 8 1-D. Added a non-convergence warning.
- The energy-correlation heatmap clearly shows the two 2-D input blocks (IC3–IC8, IC1–IC4 at
  ~0.4–0.5 off-diagonal); principal angles between all subspaces are 80–90° (near-orthogonal).
- Same WebGL caveat for the 3D scatter under headless; heatmaps render fine, no JS errors.

### Obj 8 — `build_subspace_scatter.py` + `subspace_scatter_app.html` ✅ (run: llama8b-add-02)
Plotly applet: pick up to 3 alive subcomponents from a thumbnail grid (each = its inner-act
(a,b) heatmap, matplotlib PNG) → 3D scatter of the activation projected onto those 3 unit
directions (input mlp_input·V̂ of up/gate; output mlp_output·Û of down), colour a/b/a+b, with a
dark-grey floor shadow. All from the npz + checkpoint; no GPU.
- **Shadow caveat (flagged to user):** a screen-fixed shadow that ignores 3D rotation isn't
  possible in one Plotly gl3d scene (the scene orbits as a whole), so it's the standard floor
  projection (points flattened to z=min, grey #555).
- Headless: no JS errors; thumbnails render, 3-pick cap + side-reset verified. 3D is WebGL so
  the scatter doesn't capture in headless screenshots (empty until a pick, as designed).

### Obj 9 — logarithmic period detection ✅ (run: addmult-L18-03, mult)
Log-periodic detector added to `compute_subcomp_periods`: fit a sinusoid in `log(operand)` over
`operand > threshold`, pick the multiplicative ratio with the most **cross-validated** evidence
(held-out R²) at the lowest threshold clearing `--log-bar` (0.45); cluster detected ratios
(log-ratio space) into canonical periods. New `period_type` (additive/log/none) by comparing
additive vs log CV-R² (log wins near-ties — linear/log sinusoids are degenerate for long
periods). Schema gains log columns; `common.read_subcomp_period_groups` exposes the grouping.
- **Candidate set:** the 6 named test components are CI-alive (max CI=1) but active on only
  2.5–7% of prompts, so mean CI < 0.1 dropped them. Per user, **lowered the mean-CI threshold**:
  re-ran `collect_inner_activations --op=mult --mean-ci-thr=0.02` → 166-component mult set
  (includes all 6). (My initial "not CI-alive" claim was a zsh word-splitting bug in a check.)
- **Validation:** all 6 detected as log with correct axes; down#112 & gate#104 correctly a-only;
  clusters ×1.27 (29), ×2.0 (22), ×2.4 (8), ×3.0 (5), ×3.7 (2), ×6.5 (8) — a clean handful.
- **Applet:** `build_subspace_scatter` now groups picks by `read_subcomp_period_groups`
  (`period N` / `×r` / `no period`); mult log components appear by ratio. data.js ~40 MB.
- **Note:** add still uses mean-CI 0.1 (38 comps); lower it similarly if add needs its sparse
  components too. Full-scan log detection is ~2.5 min for 166 comps (CV over period×threshold).

## Status: Obj 1–9 complete. Obj 9 (log periods) run on addmult-L18-03 mult; add re-run for the new schema.

Artifacts in `~/out/runs/addmult-L18-03/`: `hidden_activations_add.npz`,
`inner_activations_add.tsv`, `alive_filtered_add.tsv` (38 comps), `subcomp_periods_add.tsv`,
`figures/subcomp_cosine/`, `figures/neuron_explorer_add/`.

# Objective 10 — neuron investigator applet

New `build_neuron_investigator.py` + `neuron_investigator_app.html`. **Coefficient of
interaction** (unit-normalized, ≥0): write (gate/up) `|U_c[j]|/||U_c||`, read (down)
`|V_c[j]|/||V_c||` — simpler than Obj 5's V-unit *connection strength* (no norm transfer
between U/V; just normalize the relevant vector). Two-panel applet: left = neuron ×
subcomponent heatmap (cols sorted period→mean CI; rows = top-K neurons by total coeff,
paged; write blue / read red via sign-flip + RdBu); click a cell → right panel shows the
subcomponent inner-act `(a,b)` heatmap + the neuron's up/gate/output (`silu(gate)·up`)
heatmaps, output computed in JS from embedded gate/up grids.
- **Size:** per-neuron up/gate grids dominate the payload, so capped to `--top-neurons`
  (default 512) by total coefficient → data.js ~28 MB. (Per user: top-K, not all 14336.)
- **Ran on llama8b-add-02** (`model_20000.pth`): 35 subcomponents, top 512/14336 neurons.
  Headless-checked clean (click `#matrix` → 4 right-panel canvases render, no JS errors).
- Artifact: `~/out/runs/llama8b-add-02/analysis/neuron_investigator_add/{index.html,data.js}`.
- **Iteration (per user):** column sort selector (matrix→period default, with period band
  labels + thick matrix / thin period delimiters; or total coeff overall / per-frequency /
  per-matrix), 50 neurons/page, right panel 3× larger + vertically stacked/scrollable with
  per-heatmap colour bars. Added `total_coeff` per subcomp to the payload for the coeff sorts.

# Folder reorg (orthogonal)

Reserved `<run>/figures/` for training; all validation scripts now write datasets to
`<run>/analysis/datasets/` and figures/applets to `<run>/analysis/<name>/`. New `common.py`
helpers `analysis_dir` / `analysis_datasets_dir` / `run_dir_of_dataset`. Migrated + de-staled
both `llama8b-add-02` and `addmult-L18-03` on disk. See commit `cf6464f70`.
