# Completeness Toy Model

## Purpose

Tests whether SPD's importance minimality loss can detect **redundant** components. The model has N attention layers all learning the same copy task via residual sum. Since any single layer suffices, SPD should suppress the redundant layer(s) (low CI) while keeping the non-redundant `linear` projection active (high CI).

## Architecture: RedundantCopyTransformer

```
Input: [X, eq_token]     (seq_len=2, X ∈ {1..vocab_size-1})
Target: predict X at position 1

token_embed + pos_embed
  → layers.0 (attention, residual)   [REDUNDANT]
  → layers.1 (attention, residual)   [REDUNDANT]
  → linear                           [NON-REDUNDANT]
  → unembed → logits[:, 1, :]
```

- Each `SingleHeadAttention` layer is single-head causal self-attention (W_Q, W_K, W_V, W_O, no MLP/LayerNorm). Output is added to the residual stream.
- `linear` is a non-residual `Linear(d_model, d_model)` projection.
- During training, **layer dropout** (p=0.4) randomly drops layer residual contributions so each layer independently learns to solve the task. If all would be dropped, one is kept.

Default config: `vocab_size=16, d_model=64, n_layers=2, seq_len=2, eq_token=0`.

## Usage

```bash
# 1. Train the target model
python spd/experiments/completeness/train_model.py

# 2. Set pretrained_model_path in completeness_config.yaml

# 3. Run SPD
spd-local completeness
```

## Expected SPD Results

- One attention layer's components get high CI, the other's get suppressed (low CI) — demonstrating incompleteness.
- `linear` components always have high CI since the projection is non-redundant.
