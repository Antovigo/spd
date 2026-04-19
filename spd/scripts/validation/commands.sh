# Swap-test pipeline for SPD run `s-0c454b30`, assumed to live at
# `~/spd_out/spd/s-0c454b30/` on the cluster. Each command below is
# self-contained — paste any one into a shell and run it. Run the setup block
# first (once per shell) so $RUN_DIR and $MODEL_PATH are defined.

# --- Setup (run first) -------------------------------------------------------

srun --time=2:00:00 --pty bash

################
# Prompt-based #
################

RUN_DIR=~/spd_out/spd/s-0c454b30 # 4L
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR=~/spd_out/spd/s-23733db9 # 4L_naive
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR=~/spd_out/spd/s-74b94cad # 12L
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR=~/spd_out/spd/s-bd04bd99 # 12L naive
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

cd ~/SPD/spd
echo $MODEL_PATH

PROMPTS=~/SPD/batch_commands/numpy/reference_4L/prompts/numpy_and_pandas.txt

# --- 1. Alive components on target (prompt-based) data -----------------------
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH" --prompts="$PROMPTS"

# --- 2. Ablate each alive component on target data ---------------------------
uv run python -m spd.scripts.validation.effect_of_ablation "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" --prompts="$PROMPTS"

# --- 3. Same on nontarget data (side-effect reference) -----------------------
uv run python -m spd.scripts.validation.effect_of_ablation "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" --nontarget --n-batches=7

# --- 4. Summarise nontarget side-effects per component -----------------------
uv run python -m spd.scripts.validation.summarize_nontarget \
    "$MODEL_PATH" \
    "$RUN_DIR/effect_of_ablation_nontarget.tsv" \
    "$RUN_DIR/orig_predictions_nontarget.tsv" \
    --task-a='{"prompt": "import numpy as", "target": " np"}' \
    --task-b='{"prompt": "import pandas as", "target": " pd"}' \
    --quantile=0.99 \
    --prompts="$PROMPTS"

# --- 5. Find candidate (A, B) pairs for swapping -----------------------------
uv run python -m spd.scripts.validation.find_swap_candidates \
    "$MODEL_PATH" \
    "$RUN_DIR/effect_of_ablation.tsv" \
    "$RUN_DIR/orig_predictions.tsv" \
    "$RUN_DIR/nontarget_summary.tsv" \
    --task-a='{"prompt": "import numpy as", "target": " np"}' \
    --task-b='{"prompt": "import pandas as", "target": " pd"}' \
    --high-kl=0.5 --low-kl=0.1 \
    --prompts="$PROMPTS"

# --- 6. Swap test ------------------------------------------------------------
# Pick one or more swaps from $RUN_DIR/swap_candidates.tsv and list them below.
# Each swap is one string of the form <layer>:<matrix>:<a_comp>/<b_comp>.
# Multiple swaps are applied simultaneously in a single forward pass.

#SWAPS=(9:attn.v_proj:52/35)
SWAPS=(9:attn.v_proj:52/35 9:mlp.down_proj:47/48)

# Target data (one batch of prompts):
uv run python -m spd.scripts.validation.swap_test \
    "$MODEL_PATH" \
    "$RUN_DIR/alive_components.tsv" \
    "${SWAPS[@]}" \
    --target-a=" np" --target-b=" pd" \
    --prompts="$PROMPTS"

# Nontarget data:
uv run python -m spd.scripts.validation.swap_test \
    "$MODEL_PATH" \
    "$RUN_DIR/alive_components.tsv" \
    "${SWAPS[@]}" \
    --target-a=" np" --target-b=" pd" \
    --nontarget --n-batches=20

####################
# Seed comparisons #
# ##################

# --- 7. Compare matched components across two seeds --------------------------
# Hungarian-match alive components between two decompositions of the same target model
# trained with different seeds (or hyperparams). Run `find_alive_components` on each
# model first so that `<run_dir>/alive_components.tsv` exists on both sides.

RUN_DIR=~/spd_out/spd/s-74b94cad # 12L seed 0
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR_B=~/spd_out/spd/s-a77c1728 # 12L seed 1
MODEL_PATH_B=$(ls -t "$RUN_DIR_B"/model_*.pth | head -n 1)

# Find alive components for model B (model A already has it from step 1).
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_B" --prompts="$PROMPTS"

# Compare matched components
uv run python -m spd.scripts.validation.compare_matched_components "$MODEL_PATH" "$MODEL_PATH_B"

# Max-cosine comparison (each A component -> best-match B component)
uv run python -m spd.scripts.validation.compare_components "$MODEL_PATH" "$MODEL_PATH_B"

# Random-init control: model B's components reinitialised; candidate pool per matrix
# is restricted to `len(alive_b)` random indices, averaged over 10 draws.
uv run python -m spd.scripts.validation.compare_components "$MODEL_PATH" "$MODEL_PATH_B" \
    --random-b --n-random-samples=10

###################
# CSS decomposition #
###################

# Goal: find components that have a large effect on CSS but no effect on non-CSS
# data, so removing them selectively destroys the model's CSS ability.
# Target data here is a large dataset (not a prompts file), so `find_alive_components`
# needs `--n-batches` instead of `--prompts`.

RUN_DIR_CSS=~/spd_out/spd/s-429ea112 # CSS seed 0 (reference)
MODEL_PATH_CSS=$(ls -t "$RUN_DIR_CSS"/model_*.pth | head -n 1)

RUN_DIR_CSS_1=~/spd_out/spd/s-705a9887 # CSS seed 1
MODEL_PATH_CSS_1=$(ls -t "$RUN_DIR_CSS_1"/model_*.pth | head -n 1)

# --- 8. Alive components on the CSS target dataset (both seeds) --------------
# `--split=train` — the target's `eval_data_split` (`validation`) is ~1k rows;
# use the train split for a bigger sample. Both seeds are needed for step 10.
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_CSS" --split=train --n-batches=200
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_CSS_1" --split=train --n-batches=200

# --- 9. Shortlist components with low off-target activity --------------------
# Full nontarget ablation (~hundreds of components × many batches) is too slow.
# Instead, use harvest data (cheap, no forward passes) to rank alive components
# by how often they fire on general text, then keep only those that rarely do.
# Requires `spd-harvest` to have run on the decomposition beforehand.
# Column 4 of nontarget_activity.tsv is `firing_density`; tune the threshold.
uv run python -m spd.scripts.validation.nontarget_activity "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_components.tsv"

awk -F'\t' 'NR==1 || ($4+0) < 5e-5' "$RUN_DIR_CSS/nontarget_activity.tsv" > "$RUN_DIR_CSS/shortlist.tsv"

# --- 10. Ablation summaries on the shortlisted components -------------------
# Target: measure how ablating each shortlisted component affects CSS output.
# Nontarget: sanity-check the harvest-based shortlist against actual ablation
# KL on general text (firing density is only a proxy). Components to remove
# are those with high target `kl_q99` and low nontarget `kl_q99`.
# `--checkpoint-every=N` persists streaming state every N batches to
# `<summary>.ckpt` next to the summary TSV; rerunning the same command resumes
# from that checkpoint. The checkpoint is removed once the final TSV is written.

# Target (CSS):
uv run python -m spd.scripts.validation.effect_of_ablation \
    "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_components.tsv" \
    --split=train --summary-only --n-batches=20 --quantile=0.99 --batch-size=32 --checkpoint-every=1

# Nontarget (general distribution):
uv run python -m spd.scripts.validation.effect_of_ablation \
    "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_components.tsv" \
    --nontarget --summary-only --n-batches=20 --quantile=0.99 --batch-size=32 --checkpoint-every=1

# original batch size was 64 but somehow it OOMs on nontarget

# --- 11. Compare components across the two CSS seeds -------------------------
# Max-cosine match alive components of seed 0 to seed 1, then to a random-init
# control with pool size matched per-matrix to seed 1's alive counts.

uv run python -m spd.scripts.validation.compare_components \
    "$MODEL_PATH_CSS" "$MODEL_PATH_CSS_1"

uv run python -m spd.scripts.validation.compare_components \
    "$MODEL_PATH_CSS" "$MODEL_PATH_CSS_1" \
    --random-b --n-random-samples=10

# --- 12. Multi-language eval with only the CSS-alive components active ------
# Keep only the components from alive_components.tsv on (plus or minus the
# delta via --invert), disable everything else, and measure per-position KL
# across several programming languages. Converts the TSV into the plain-text
# `<layer>:<matrix>:<component>` format that multilang_ablation expects.

awk -F'\t' 'NR>1 {print $1":"$2":"$3}' \
    "$RUN_DIR_CSS/alive_components.tsv" > "$RUN_DIR_CSS/alive_ablation_list.txt"

uv run huggingface-cli login

uv run python -m spd.scripts.validation.multilang_ablation \
    "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_ablation_list.txt" --invert \
    --batch-size=32 --tokens-per-lang=500000\
    --output="$RUN_DIR_CSS/css_only_multilang_ablations.tsv" \
    --output-sequences="$RUN_DIR_CSS/css_only_per_sequence_loss.tsv"
