# Fourier probes (ridge-CV, null-gated)

Answers *which variables (`a`, `b`, `a+b`, `a-b`) are represented at which period, at each
residual-stream position* around L18's MLP in Llama-3.1-8B — with a protocol that a
linearly-separable-but-not-circular code cannot pass:

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

