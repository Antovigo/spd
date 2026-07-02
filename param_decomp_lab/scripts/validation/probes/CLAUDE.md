# Fourier probes (Feucht-faithful)

A self-contained re-implementation of Feucht et al.'s (2026, "Arithmetic in the Wild",
github.com/goodfire-ai/arithmetic-wild) circular-feature probing for Llama-3.1-8B, kept
separate from the older `find_fourier_features` / `build_fourier_scatter` in the parent
folder (which probe the MLP in/out spaces with a closed-form fit). Everything here targets
**their** setup exactly.

## Pipeline

```
collect_resid_activations.py  (GPU/--slurm)  → resid_activations.npz
fit_fourier_probes.py         (GPU/--slurm)  → probes.json
build_probe_scatter.py        (CPU)          → probe_scatter/{index.html,data.js}
```

Default artifact dir: `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`.

## Fidelity to Feucht

- **Site**: the layer-18 **decoder-block output** (`out[0]` of `model.layers[18]`) = the
  residual stream after that layer's attention *and* MLP have written (their `source="resid"`).
- **Data**: `{a}+{b}=` for `a, b ∈ 1..200` (40 000 prompts), **last token** (`=`), left-padded
  so position `-1` is always the `=`.
- **Probe**: per period `T` and variable `v ∈ {a, b, a+b}` (their input / offset / output), two
  `nn.Linear(d_model, 1)` (bias) probes for `cos(2πv/T)` / `sin(2πv/T)`, Adam(lr 1e-3) + MSE,
  **500 epochs**, on **raw** (un-standardised) activations. `train_test_split(test_size=0.2,
  random_state=42)`; run order `a` outer / `b` inner matches their prompt order, so the split
  coincides. `r2_*` is the held-out (test) R². Period 2 has `sin≡0` → its sin probe is skipped.
- Closed-form least squares would give a higher R² / tighter circles; we deliberately keep their
  Adam-500-epochs fit so the results match theirs (e.g. the a+b feature is a fuzzy ring with the
  characteristic period-20 dip to R²≈0.44–0.50; operands fit much tighter, R²≈0.98).

## Applet

`build_probe_scatter` projects the collected activations onto each probe's predicted `(cos, sin)`
plane (a clean feature traces the unit circle). Controls: **basis variable** (which probe defines
the plane), **colour by** `a` / `b` / `a+b` with **mod** + **offset** (`(value − offset) mod m`,
scale `0..m-1`), zoom/pan, hover. Vanilla-JS canvas, no CDN — smoke-test with the parent folder's
`headless_check.py`.
