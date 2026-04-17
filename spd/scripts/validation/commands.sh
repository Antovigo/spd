# Swap-test pipeline for SPD run `s-0c454b30`, assumed to live at
# `~/spd_out/spd/s-0c454b30/` on the cluster. Each command below is
# self-contained — paste any one into a shell and run it. Run the setup block
# first (once per shell) so $RUN_DIR and $MODEL_PATH are defined.

# --- Setup (run first) -------------------------------------------------------
cd ~/SPD/spd
RUN_DIR=~/spd_out/spd/s-0c454b30
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)
echo $MODEL_PATH

# --- 1. Alive components on target (prompt-based) data -----------------------
uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH" --n-batches=1

# --- 2. Ablate each alive component on target data ---------------------------
uv run python -m spd.scripts.validation.effect_of_ablation "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" --n-batches=1

# --- 3. Same on nontarget data (side-effect reference) -----------------------
uv run python -m spd.scripts.validation.effect_of_ablation "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" --nontarget --n-batches=20

# --- 4. Rank candidate (A, B) pairs for swapping -----------------------------
uv run python -m spd.scripts.validation.find_swap_candidates "$MODEL_PATH" "$RUN_DIR/effect_of_ablation.tsv" "$RUN_DIR/effect_of_ablation_nontarget.tsv" --task-a='{"prompt": "import numpy as", "target": " np"}' --task-b='{"prompt": "import pandas as", "target": " pd"}' --top-k=20 --quantile=0.99

# --- 5. Swap test ------------------------------------------------------------
# Pick a pair from $RUN_DIR/swap_candidates.tsv and edit the four fields below.
uv run python -m spd.scripts.validation.swap_test "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" --layer=1 --matrix=attn.q_proj --a-component=279 --b-component=177 --target-a=" np" --target-b=" pd" --n-nontarget-batches=10
