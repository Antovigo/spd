#!/usr/bin/env bash
# Run multilang_ablation once per component from a shortlist file.
#
# Usage:
#   bash multilang_per_component.sh <model_path> <shortlist_file> <output_dir> [extra-flags...]
#
# Shortlist file: one `<layer>:<matrix>:<component>` per line.
# Blank lines and lines starting with `#` are ignored; inline `# comment`
# after a spec is stripped. For each component the script writes:
#   <output_dir>/<safe>.tsv               (per-position)
#   <output_dir>/<safe>_per_sequence.tsv  (per-sequence CE/KL)
# where `<safe>` replaces `:` with `_` in the component spec.
#
# Any extra flags after the three positional args are forwarded verbatim to
# `multilang_ablation` (e.g. --batch-size=32 --tokens-per-lang=50000).

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_path> <shortlist_file> <output_dir> [extra-flags...]" >&2
    exit 1
fi

MODEL_PATH="$1"
SHORTLIST_FILE="$2"
OUTPUT_DIR="$3"
shift 3

mkdir -p "$OUTPUT_DIR"

while IFS= read -r COMP || [ -n "$COMP" ]; do
    COMP="${COMP%%#*}"                       # strip inline comment
    COMP="$(echo "$COMP" | xargs)"           # trim whitespace
    [ -z "$COMP" ] && continue
    SAFE="${COMP//:/_}"
    uv run python -m spd.scripts.validation.multilang_ablation \
        "$MODEL_PATH" "$COMP" \
        --output="$OUTPUT_DIR/${SAFE}.tsv" \
        --output-sequences="$OUTPUT_DIR/${SAFE}_per_sequence.tsv" \
        "$@"
done < "$SHORTLIST_FILE"
