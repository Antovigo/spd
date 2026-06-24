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
</content>
