# Shared spec — what every variant implements

Reference impl: `/mnt/home/oli/param-decomp/nano_param_decomp/run.py` (1219 lines, PyTorch).
You're porting a strict subset of this to JAX.

## The algorithm

Given a target model with weights `W ∈ R^{d_in × d_out}` at certain "sites":

1. **Decompose** each target `W` into rank-1 components:
   - `V ∈ R^{d_in × C}` (trainable)
   - `U ∈ R^{C × d_out}` (trainable)
   - `W_delta = W_target − V @ U` (frozen residual; updated to track `W_target − V@U` at every step? NO — frozen at init, only V, U change. Verify against nano reference.)

2. **CI function** — a small MLP per site (or one global) that takes the
   pre-weight activations `x ∈ R^{..., d_in}` and outputs `ci ∈ [0, 1]^{..., C}`.
   Use `lower_leaky_hard_sigmoid` (see nano reference + `param_decomp/ci_sigmoids.py`).

3. **Stochastic mask**: `m = ci + (1 − ci) · u` where `u ∼ U[0, 1]` per (b, t, c).
   Fresh sample every forward.

4. **Decomposed forward** at each site: `y = ((x @ V) * m) @ U + x @ W_delta`
   (transposed to match your convention; verify).

5. **Loss = α·faithfulness + β·importance_minimality + γ·stochastic_recon**
   where:
   - `faithfulness = mean((W_target − V@U)^2)` over all decomposed sites
   - `importance_minimality = mean(ci^p)` with `p = 0.9` (a simple sparsity penalty)
   - `stochastic_recon = mean((y_decomposed − y_target)^2)` (MSE on outputs)

   Skip PGD, layerwise variants, persistent PGD, hidden-acts recon for v1.

6. **Two optimizers**:
   - Optimizer A: V and U at every decomposed site (lr = 1e-3)
   - Optimizer B: CI fn params (lr = 1e-3)
   - Update both every step. (Same effect as one combined optimizer for now,
     but the structural separation must be there so it's easy to add separate
     LR schedules later.)

7. **Train for 5000 steps**. Log loss every 100. Show losses going down.

## Targets to implement

Each variant must demonstrate two targets:

### Target A: TMS (Toy Model of Superposition) — 5 features in 2 dims

A single linear layer that reconstructs 5 sparse binary features through a
2-dim bottleneck:

```python
# target_model: x → W2 @ relu(W1 @ x + b1) + b2
# where W1: [5 → 2], W2: [2 → 5], b1: [2], b2: [5]
# data: x ∈ {0, 1}^5, sparse (p_active = 0.1 per feature, iid)
```

Decompose `W1` AND `W2`, each with `C = 5` components.

### Target B: Toy 2-layer MLP — multi-site decomposition

A 2-layer MLP on synthetic data:

```python
# target: y = MLP_2layer(x)  with d_model=64, d_ff=128, output_dim=32
# 4 decomposable sites: layer1.up, layer1.down, layer2.up, layer2.down
# data: random gaussian inputs; teacher = a randomly-init MLP_2layer
```

Decompose all 4 sites with `C = 16` each. Demonstrates multi-site composition.

## Deliverables per variant

Inside your variant directory:

- `train_tms.py` — runs TMS target. `python train_tms.py` should train and print loss.
- `train_toy_mlp.py` — runs 2-layer MLP target. Same shape.
- `README.md` — 200-400 words explaining the architectural pattern you used.
  How does the trainer access the model? How are masks threaded? How is the
  two-optimizer pattern expressed? What's the user-facing API to add a new
  target model?
- `REFLECTION.md` — your honest 200-400 word assessment after building it.
  What worked elegantly? What was friction? What would you change if you
  ported this all the way up to a transformer? Is this approach worth
  scaling to the full library?
- `requirements.txt` — pinned versions of jax, equinox, optax, jaxtyping.

## Constraints — keep it small

- ~500-1000 LOC total per variant. Don't reproduce the whole nano reference.
- No PGD. No persistent state. No DDP. No autocast. No fancy LR schedules.
- No checkpointing. No wandb. No CLI configs. Just hard-coded hyperparams.
- The point is to demonstrate the *architectural pattern* clearly, in code
  small enough to read in 15 minutes.

## How to set up the environment

```bash
cd /mnt/home/oli/param-decomp-jax/variant-N
uv venv .venv
source .venv/bin/activate
uv pip install "jax[cuda13]" equinox optax jaxtyping einops numpy
# If cuda13 doesn't work, fall back to CPU JAX: uv pip install jax
python train_tms.py
```

CPU JAX is fine for this scale — both targets train in under a minute.
