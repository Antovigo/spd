# CI filter data snapshot — `p-ba5a0c05` step 40000

A copy of the small, derived outputs behind `../report.md` and `../report_comp_classification.md`,
taken on 2026-09-21. The canonical copy is the run folder,
`<run_dir>/analysis/` (`~/out/pod-backup/p-ba5a0c05/analysis/` on the lab cluster); this tree
mirrors its layout so paths in the reports resolve against either.

Left out on purpose: the fine-tuned CI fns (3.4 GB each), the `.npz` arrays (max CI, keep masks,
the decomposition's alive set), the grid applets, the per-sweep raw TSVs (their numbers are all in
`ablations/step_40000/components.tsv`), and the retired runs in `ci_filter/step_40000/Trash/`.

| path | what |
|---|---|
| `ci_filter/step_40000/<filter-id>/` | the three kept filters (all ceiling-capped; `-integers` and `-answer-ce` also pruned): `config.yaml`, `config.original.yaml` (as pinned when run, with the retired `ci_ceiling`/`prune_dead` switches — `-last-pos-ceiling` ran with `prune_dead: false`), `eval/{pool_evals.jsonl, summary.json, pgd_recon.json}`, `alive/alive.json` |
| `ci_filter/step_40000/pruning_by_layer.tsv` | alive components per kind and layer, per stage (`figures/pruning_by_layer.png`) |
| `ci_filter/step_40000/l0_by_position.tsv` | per-token L0 per site and position, per stage (`figures/l0_position_*.png`) |
| `ablations/step_40000/components.tsv` | one row per component alive in the decomposition: survival per filter, screen, sign rates, the three causal sweeps |
| `ablations/step_40000/<source>/` | screen summary and verified extremes, per CI source (`run` = the decomposition's own CI) |
| `ablations/step_40000/model_ablation.tsv` | the L13–17 candidates subtracted from the MODEL (weight delta on) |
| `ablations/step_40000/nontarget_probe/` | the million-token fineweb probe: `summary.tsv`, `examples.jsonl`, and `app/index.html` (open over `file://`) |
| `ablations/step_40000/mask_ablations/run.json` | global-mask ablations of the decomposition |
