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
- **Default = full per-variable sweep**: every `T ∈ 2..min(v.max()//2, max_period)` (`max_period`
  150; `a`,`b` → 2..100, `a+b` → 2..150), ~700 probes. `--periods` fits an explicit shared list.
- Closed-form least squares would give a higher R² / tighter circles; we deliberately keep their
  Adam-500-epochs fit so the results match theirs.

## Applet — the control

`build_probe_scatter` leads with an **R²-vs-period curve** (one line per variable). This is the
control: R² should spike only at the model's true periods. Empirically `a+b` spikes sharply at
periods 2/5/10 (+a broad 50–100 harmonic hump) and sits at ~0 elsewhere — clicking a spike period
draws a **circle**, a valley period a **blob**. (`a`/`b` stay high at every period because a single
operand is ~directly decodable, so any function of it is too; `a+b` is the discriminating variable.)

Clicking a period draws its scatter: the activations projected onto that probe's predicted
`(cos, sin)` plane. Controls: **basis variable**, **colour by** `a`/`b`/`a+b` with **mod** +
**offset** (`(value − offset) mod m`, scale `0..m-1`), zoom/pan, hover. A fixed `n_show` random
subset of points is shipped per plot to bound `data.js`. Vanilla-JS canvas, no CDN — smoke-test
with the parent folder's `headless_check.py`.
