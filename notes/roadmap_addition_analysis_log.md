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

## Status: all 6 objectives complete (addition). sub/mult supported via `--op`; not yet run.

Artifacts in `~/out/runs/addmult-L18-03/`: `hidden_activations_add.npz`,
`inner_activations_add.tsv`, `alive_filtered_add.tsv` (38 comps), `subcomp_periods_add.tsv`,
`figures/subcomp_cosine/`, `figures/neuron_explorer_add/`.
</content>
