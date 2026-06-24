# Validation scripts

Standalone analysis scripts that probe a *trained* decomposition (a saved
`ComponentModel` checkpoint) and write inspectable artifacts. Each script answers one
question — "which components are alive?", "what happens when I ablate X?", "how similar
are two decompositions?" — and is run by hand, not as part of the training loop or the
postprocess pipeline.

This file defines the CLI design every script in this folder must follow, so the set
stays consistent and composable. The design is ported from the SPD repo's
`spd/scripts/validation/`; keep parity with it when adding scripts.

## One script, one analysis, one entrypoint

- Each script exposes a **single public function named after the file**, wired at the
  bottom with `fire.Fire(<function>)` under `if __name__ == "__main__":`. No argparse.
- The function returns the `Path`(s) it wrote (so it's importable and testable), and
  logs a one-line summary via `param_decomp.log.logger` (or the lab logger).
- Run with the module path after activating the venv:

  ```bash
  source .venv/bin/activate
  python -m param_decomp_lab.scripts.validation.<name> <model_path> [args] [--flags]
  ```

- Module docstring: lead with one line on *what question it answers*, then a `Usage:`
  block and a list of the output files with their default filenames. Heavier scripts
  earn a paragraph on method/semantics — that's the one place a longer docstring is
  warranted here.

## Arguments

**Positional, in order:**

1. The path to a decomposed model (the checkpoint, e.g. `.../runs/<run_id>/model_<step>.pth`).
   Always first.
2. Any further *required* inputs — usually a TSV produced by an earlier script (e.g. an
   alive-components list), or a free-form spec string. Scripts chain by consuming each
   other's TSVs.

**Flags (optional, with defaults):** use `fire`'s `--kebab-case` form. Reuse these names
and meanings exactly across scripts — don't invent a synonym:

| Flag | Meaning |
|---|---|
| `--nontarget` | Evaluate on the nontarget distribution instead of the target. Assert the run's config actually has a nontarget config; error loudly if not. |
| `--n-batches` | Number of data batches to run. **Ignored** for prompt-based LM tasks, which always run one batch containing every prompt. |
| `--prompts` | Override the LM `prompts_file`. LM-only. |
| `--split` / `--batch-size` | Override the dataset split / batch size for dataset-based LM tasks. LM-only. |
| `--ci-thr` (or `--ci-threshold`) | CI value above which a component counts as active. |
| `--output` | Override the output path. Add `--output-fig` / `--output-tsv` / `--output-<role>` when a script writes more than one file — one flag per file, never a single dir-mangling flag. |

Defaults live in the function signature (`fire` reads them), and should be the value the
common case wants — keep them high in the call stack, don't re-default downstream.

## Output

- **Default location: the decomposed model's run folder** (the checkpoint's parent dir),
  under a fixed, script-specific filename. `--output*` overrides it. This keeps every
  artifact for a run co-located (see the "Saved-run layout" in the root `CLAUDE.md`).
- **Target vs nontarget filenames differ by a `_nontarget` suffix** (`effect_of_ablation.tsv`
  → `effect_of_ablation_nontarget.tsv`) so both can coexist. When a script compares two
  runs, suffix the output with the *other* run's folder name so several can coexist.
- **TSV is the default format** (`csv.DictWriter`, `delimiter="\t"`, header row). Use
  **JSONL** only when rows carry variable-length lists / nested dicts that don't fit a
  cell. Figures are `.png` written alongside the TSV.
- **Stable leading columns.** Per-component rows start with `layer, matrix, component`
  (block number; matrix like `attn.q_proj` or `mlp.c_fc`; component index). Per-position
  rows add `prompt, pos`. Downstream scripts rely on these names.
- One row per logical unit; don't duplicate a wide block across N rows when it's constant
  — split it into a second file keyed by the shared dimension (e.g. orig-predictions kept
  separate from per-component KL).

## GPU scripts run via SLURM

The login node has no GPU, so any script that runs a model forward pass must offer a
`--slurm` flag that re-submits the *same invocation* as a single-GPU job instead of
running locally. Use the shared `common.py` helpers — don't hand-roll `sbatch`:

- Add these flat params to the script function (fire needs them flat):
  `slurm=False`, `partition=DEFAULT_PARTITION_NAME`, `gpus=1`, `slurm_time="1:00:00"`,
  `slurm_mem=None`.
- First thing in the function, when `slurm` is set: build the forwarded `argv` (the
  positional model path **expanded to absolute** — the job runs from `REPO_ROOT`, not the
  caller's cwd — plus every non-SLURM flag), then call
  `submit_self_to_slurm(_MODULE, argv, SlurmOptions(...), job_name="val-<name>")` and
  return `None`. Define `_MODULE` as the script's dotted module path.

`submit_self_to_slurm` runs from `REPO_ROOT` with the live `.venv` (no git snapshot —
validation is interactive and throwaway). Outputs land wherever the model-path-derived
run dir points, so no env vars need forwarding when an absolute checkpoint path is passed.
The return type becomes `Path | None` (None on the submit path).

## HTML applets + headless testing

Some scripts emit a self-contained interactive HTML applet instead of a TSV/PNG
(`build_addition_explorer`: `index.html` + `data.js`, opened from `file://`, no
server/CDN/GPU). The big template lives as a sibling asset (`*_app.html`) the generator
copies + injects data into — not a Python string. Keep the applet dependency-free (vanilla
JS, inline canvas charts) so it runs offline on a laptop.

`headless_check.py` smoke-tests any such applet in headless Chromium (renders canvas in
software — no display needed), failing on any JS console error / uncaught exception. It is
the exception to the "run via the project venv" rule: it imports only `playwright` + `fire`
and runs under the isolated toolchain venv built by `headless_setup.sh` (Chromium's
system-lib needs are kept out of `make install-dev` / CI). The bug it most often catches is
JS-only and invisible to Python — a missing `<script src>`, an undefined global, a throwing
render.

## Shared helpers — put them in `common.py`, don't duplicate

Cross-cutting logic lives in `common.py`; scripts import from it rather than
reimplementing. Implemented so far:

- **`load_lm_run(model_path) -> LoadedRun`** — wraps `SavedLMRun.from_path(...).load_model()`
  to return an eval-moded `ComponentModel` on the compute device, plus `cfg`, `run_dir`,
  and `tokenizer` in one call.
- **`escape_tsv_value`** — reversible backslash-escaping for TSV cells.
- **`SlurmOptions` + `submit_self_to_slurm`** — the `--slurm` self-resubmission path (see
  above).
- **Arithmetic-analysis helpers** (`roadmap_addition_analysis`): `op_symbol` /
  `op_prompts_file` / `parse_operands` resolve an operation (`add`/`sub`/`mult`) to its infix
  symbol, `1..100` prompt file, and `(a, b)` parser; `MLP_MATRICES` is the L18 MLP proj
  triple; `read_alive_components` reads an alive-components TSV into `AliveComponent`s
  (optionally filtered to given proj names).

Add to `common.py` when a second script needs it (mirroring SPD's `common.py`): task
resolution (target vs nontarget, `--prompts` / `--split` overrides), a single-epoch
`input_ids` iterator, `(layer, matrix)` ↔ module-path parsing, `--task-*` JSON parsing.
If two scripts need the same thing, it belongs here.

## Composition & sample commands

- Scripts form pipelines by reading each other's TSVs (e.g.
  `find_alive_components` → `effect_of_ablation` → `summarize_*` → `find_swap_candidates`).
  Decouple an expensive full pass from cheap re-filtering by writing an intermediate
  summary TSV, so tweaking a threshold doesn't force another pass over the big file.
- Keep a **`commands.md`** (or `commands.sh`) in this folder with self-contained, runnable
  example invocations — a setup block that defines `$MODEL_PATH` / `$RUN_DIR`, then one
  pasteable block per script with real paths. It doubles as living documentation of how
  the scripts chain.

## Style

Follow the root `CLAUDE.md` coding guidelines — fail-fast asserts over defensive
branches, encode invariants in types, einops + jaxtyping + liberal shape asserts for
tensor work. Assert LM-only / `LinearComponents`-only restrictions up front rather than
silently producing wrong output.
