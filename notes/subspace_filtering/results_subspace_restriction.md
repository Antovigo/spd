# Subspace restriction — results log

Running results for the experiments in `plan.md`. Reference:
`addsub-L18-04-hidden` (dense, 24k steps). Battery numbers are **raw**-flavor mean KL
over the active subset, add+sub pooled, unless stated.

## E0 — singular spectra (2026-07-09)

All 7 L18 matrices are numerically full rank (r = min dim at τ = 1e-2, except q/o
losing a handful of directions). Spectra are flat; meaningful truncation starts around
τ = 0.1. Ranks at candidate thresholds:

| module | full | τ=0.1 | τ=0.2 | τ=0.3 |
|---|---|---|---|---|
| gate_proj | 4096 | 2760 | 297 | 17 |
| up_proj | 4096 | 4065 | 3580 | 2334 |
| down_proj | 4096 | 4066 | 3221 | 1797 |
| q_proj | 4096 | 1451 | 366 | 95 |
| k_proj | 1024 | 787 | 411 | 176 |
| v_proj | 1024 | 1022 | 942 | 801 |
| o_proj | 4096 | 2420 | 1279 | 490 |

Consequence: at τ = 0 the parameterization constrains only the wide sides (gate/up
write, down/k/v read) — exactly where the battery's legality tests bite. E3 uses
τ ∈ {0.1, 0.2}. Artifacts: `~/pd_scratch/subspace_restriction/spectra/`.

## E1 — retrofit of the dense checkpoint (2026-07-09)

`retrofit_svd_projection` at τ = 1e-5 → `addsub-L18-04-hidden-Aretro`. Relative mass
removed by the projection, `‖ΔV‖/‖V‖` resp. `‖ΔU‖/‖U‖` on the constrained sides:
**0.84 (down V), 0.81 (gate U), 0.82 (up U), 0.85 (k V), 0.83 (v V)** — i.e. at the
level a random vector would have (√(1 − 4096/14336) ≈ 0.845). Training never removed
the illegal mass present at init. q/o: ~0.02 (full-rank square, no-op).

Battery, raw flavor, mean KL (mlp block):

| experiment | dense ref | retrofit |
|---|---|---|
| circuit_baseline | 0.0117 | **0.5210** |
| circuit_in_row:down | 0.1741 | 0.5212 (= baseline) |
| circuit_out_col:gate | 0.0469 | 0.5210 (= baseline) |
| circuit_out_col:up | 0.0617 | 0.5210 (= baseline) |
| orig_in_span:down | 0.1871 | 0.1265 |
| orig_in_span:gate/up | 0.148 | 0.148 |
| orig_out_span:gate | 0.1058 | 0.2160 |
| orig_out_span:up | 0.3089 | 0.2493 |
| orig_out_span:down | 0.0978 | 0.0978 |

Attention block: every number unchanged (ratio 1.00) — q/o are no-ops and k/v's
illegal read mass is dormant (orthogonal to on-distribution activations).

Conclusions:

1. Legality (F3) becomes structural, as designed: the circuit-side projections are
   exact no-ops after retrofit.
2. **The illegal mass was functionally load-bearing for the MLP circuit**: removing
   it degrades the circuit baseline 44× (0.012 → 0.52 nats). It cancels in the full
   sum but does real work in the masked circuit. Post-hoc projection without
   fine-tuning is not viable; the interesting question is E2 (training under the
   constraint from scratch).
3. F1 (orig-span tests) barely moves — no free win from projection alone.

## E2 — addsub-L18-05-svd-tau0 (in flight)

Memory probe (steps=3): peak 44.7 GiB / 46 — fits with the ~1 GB of fp32 Q buffers.
Found + fixed a real bug: DDP's default per-forward buffer broadcast in-place
clobbers Q_in/Q_out saved for backward → `broadcast_buffers=False` (all buffers are
static and rank-identical).

Full run: SLURM job 4157, 24k steps, 2×L40, exact reference recipe +
`svd_rank_threshold: 0.0`.
