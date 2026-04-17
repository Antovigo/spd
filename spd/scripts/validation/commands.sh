# Swap-test pipeline for SPD run `s-0c454b30`, assumed to live at
# `~/spd_out/spd/s-0c454b30/` on the cluster. Each command below is
# self-contained — paste any one into a shell and run it. Run the setup block
# first (once per shell) so $RUN_DIR and $MODEL_PATH are defined.

# --- Setup (run first) -------------------------------------------------------

srun --time=2:00:00 --pty bash

RUN_DIR=~/spd_out/spd/s-0c454b30 # 4L
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR=~/spd_out/spd/s-23733db9 # 4L_naive
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR=~/spd_out/spd/s-74b94cad # 12L
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
# Pick a pair from $RUN_DIR/swap_candidates.tsv and edit the four fields below.

LAYER=3
MATRIX=attn.o_proj
A_COMP=66
B_COMP=79

# Target data (one batch of prompts):
uv run python -m spd.scripts.validation.swap_test \
    "$MODEL_PATH" \
    "$RUN_DIR/alive_components.tsv" \
    --layer=$LAYER --matrix=$MATRIX --a-component=$A_COMP --b-component=$B_COMP \
    --target-a=" np" --target-b=" pd" \
    --prompts="$PROMPTS"

# Nontarget data:
uv run python -m spd.scripts.validation.swap_test \
    "$MODEL_PATH" \
    "$RUN_DIR/alive_components.tsv" \
    --layer=$LAYER --matrix=$MATRIX --a-component=$A_COMP --b-component=$B_COMP \
    --target-a=" np" --target-b=" pd" \
    --nontarget --n-batches=10

# --- 7. Compare matched components across two seeds --------------------------
# Hungarian-match alive components between two decompositions of the same target model
# trained with different seeds (or hyperparams). Run `find_alive_components` on each
# model first so that `<run_dir>/alive_components.tsv` exists on both sides.

RUN_DIR_B=~/spd_out/spd/s-a77c1728 # 12L seed 1

MODEL_PATH_B=$(ls -t "$RUN_DIR_B"/model_*.pth | head -n 1)

# Find alive components for model B (model A already has it from step 1).
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_B" --prompts="$PROMPTS"

# Compare matched components
uv run python -m spd.scripts.validation.compare_matched_components "$MODEL_PATH" "$MODEL_PATH_B"
