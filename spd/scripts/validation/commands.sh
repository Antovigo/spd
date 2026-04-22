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

# --- 1b. CI heatmap of alive components (assumes $MODEL_PATH = Numpy 4L s-0c454b30) -----
PROMPTS_LAPTOP=~/Code/SPD/batch_commands/numpy/reference_4L/prompts/numpy_and_pandas.txt
RUN_DIR_LAPTOP=~/Documents/MATS/spd_out/spd/s-0c454b30 # 4L
MODEL_PATH_LAPTOP=$(ls -t "$RUN_DIR_LAPTOP"/model_*.pth | head -n 1)
uv run python -m spd.scripts.validation.plot_alive_components "$MODEL_PATH_LAPTOP" --prompts="$PROMPTS_LAPTOP"

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

### 4L ###
PROMPTS=~/SPD/batch_commands/numpy/reference_4L/prompts/numpy_and_pandas.txt

RUN_DIR=~/spd_out/spd/s-0c454b30 # 4L seed 0
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)

RUN_DIR_B=~/spd_out/spd/s-30310cbf # 12L seed 1
MODEL_PATH_B=$(ls -t "$RUN_DIR_B"/model_*.pth | head -n 1)

### 12L ###
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

# --- 7b. All pairwise cos sims between alive components of two decompositions ----
# Compares the numpy+pandas 4L run (s-0c454b30) against each of the single-task
# runs (numpy-only s-aeab077c, pandas-only s-61fe71f0), and the two single-task
# runs against each other. Assumes `find_alive_components` has been run on each.
RUN_DIR_BOTH=~/spd_out/spd/s-0c454b30   # numpy + pandas (4L)
MODEL_PATH_BOTH=$(ls -t "$RUN_DIR_BOTH"/model_*.pth | head -n 1)

RUN_DIR_NUMPY=~/spd_out/spd/s-aeab077c  # numpy only
MODEL_PATH_NUMPY=$(ls -t "$RUN_DIR_NUMPY"/model_*.pth | head -n 1)

RUN_DIR_PANDAS=~/spd_out/spd/s-61fe71f0 # pandas only
MODEL_PATH_PANDAS=$(ls -t "$RUN_DIR_PANDAS"/model_*.pth | head -n 1)

uv run python -m spd.scripts.validation.all_cosine_similarities \
    "$MODEL_PATH_BOTH" "$MODEL_PATH_NUMPY"

uv run python -m spd.scripts.validation.all_cosine_similarities \
    "$MODEL_PATH_BOTH" "$MODEL_PATH_PANDAS"

uv run python -m spd.scripts.validation.all_cosine_similarities \
    "$MODEL_PATH_NUMPY" "$MODEL_PATH_PANDAS"

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
    --batch-size=32 --tokens-per-lang=500000 \
    --output="$RUN_DIR_CSS/css_only_multilang_ablations.tsv" \
    --output-sequences="$RUN_DIR_CSS/css_only_per_sequence_loss.tsv"

# --- 13. Per-component multilang ablation (CSS shortlist) -------------------
# Loop over components in $RUN_DIR_CSS/component_shortlist.txt (one
# `<layer>:<matrix>:<component>` per line; blank and `#` lines ignored),
# ablating each one individually. Outputs land in
# $RUN_DIR_CSS/component_shortlist/. Extra flags after the 3rd arg are
# forwarded to multilang_ablation.

bash spd/scripts/validation/multilang_per_component.sh \
    "$MODEL_PATH_CSS" \
    "$RUN_DIR_CSS/component_shortlist.txt" \
    "$RUN_DIR_CSS/component_shortlist_output" \
    --batch-size=32 --tokens-per-lang=500000

# --- 14. Compare targeted decomposition against a larger decomposition -------
# For each alive component in the targeted model, find the component in the
# larger model with max |cos sim| searched across ALL larger-model components
# (not restricted to alive). Random baseline re-inits the targeted model's V/U
# and repeats the match n times. Writes two TSVs next to the targeted run,
# each suffixed with the larger run's folder name.

RUN_DIR_LARGER=~/spd_out/spd/jose
MODEL_PATH_LARGER=$(ls -t "$RUN_DIR_LARGER"/model_*.pth | head -n 1)

# CSS
uv run python -m spd.scripts.validation.compare_to_larger \
    "$MODEL_PATH_CSS" "$MODEL_PATH_LARGER" \
    --n-random-samples=10

# --- 15. Completeness check --------------------------------------------------
# For each alive component X, run two ablations and log per-sequence mean KL
# against the original model:
#  - mean_kl_circuit: only alive components on, delta OFF, X off
#  - mean_kl_all:     every component on, delta ON, X off
# Plus mean_kl_circuit_baseline (circuit, no X ablated) and max_ci (max over
# positions). Rows are filtered to sequences where max_ci > --ci-threshold,
# and components silent over an entire batch are skipped. Rows where
# mean_kl_all is small but mean_kl_circuit is large mean something outside the
# alive set (delta or an inactive component) is doing X's job in parallel.
# Run `find_alive_components` first so alive_components.tsv exists.

# Toy model (CompletenessTaskConfig — no prompts/split, just n-batches)
RUN_DIR_COMPL=~/spd_out/spd/s-afa9c6ad
MODEL_PATH_COMPL=$(ls -t "$RUN_DIR_COMPL"/model_*.pth | head -n 1)

uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_COMPL" --n-batches=10

uv run python -m spd.scripts.validation.completeness \
    "$MODEL_PATH_COMPL" "$RUN_DIR_COMPL/alive_components.tsv" \
    --n-batches=10 --batch-size=32

# numpy 12L (prompt-based target data):
RUN_DIR=~/spd_out/spd/s-74b94cad
MODEL_PATH=$(ls -t "$RUN_DIR"/model_*.pth | head -n 1)
PROMPTS=~/SPD/batch_commands/numpy/reference_4L/prompts/numpy_and_pandas.txt

uv run python -m spd.scripts.validation.completeness \
    "$MODEL_PATH" "$RUN_DIR/alive_components.tsv" \
    --prompts="$PROMPTS"



# CSS (dataset-based target data):
RUN_DIR_CSS=~/spd_out/spd/s-429ea112 # CSS seed 0 (reference)
MODEL_PATH_CSS=$(ls -t "$RUN_DIR_CSS"/model_*.pth | head -n 1)

uv run python -m spd.scripts.validation.completeness \
    "$MODEL_PATH_CSS" "$RUN_DIR_CSS/alive_components.tsv" \
    --split=train --n-batches=10 --batch-size=32



# jose (larger decomposition, dataset-based):
RUN_DIR_JOSE=~/spd_out/spd/jose
MODEL_PATH_JOSE=$(ls -t "$RUN_DIR_JOSE"/model_*.pth | head -n 1)

uv run python -m spd.scripts.validation.find_alive_components "$MODEL_PATH_JOSE" --split=train --n-batches=200

uv run python -m spd.scripts.validation.completeness \
    "$MODEL_PATH_JOSE" "$RUN_DIR_JOSE/alive_components.tsv" \
    --split=train --n-batches=10 --batch-size=8
