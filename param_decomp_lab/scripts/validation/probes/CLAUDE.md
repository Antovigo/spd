# Fourier probes (Feucht-faithful)

A self-contained re-implementation of Feucht et al.'s (2026, "Arithmetic in the Wild",
github.com/goodfire-ai/arithmetic-wild) circular-feature probing for Llama-3.1-8B, kept
separate from the older `find_fourier_features` / `build_fourier_scatter` in the parent
folder (which probe the MLP in/out spaces with a closed-form fit). Everything here targets
**their** setup exactly.

## Pipeline

```
collect_resid_activations.py  (GPU/--slurm)  → resid_activations.npz  (resid_post + resid_pre)
fit_fourier_probes.py --site  (GPU/--slurm)  → probes_<site>.json     (run once per site)
build_probe_scatter.py        (CPU)          → probe_scatter/{index.html,data.js}
```

Default artifact dir: `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`. `build_probe_scatter`
takes the **npz** and picks up every `probes_<site>.json` beside it — one tab per site.

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
