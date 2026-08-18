#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PYTHON_BIN:?Set PYTHON_BIN to the xhquant Python executable.}"
: "${OBSERVER_ARTIFACT:?Set OBSERVER_ARTIFACT to one frozen official REAP observer artifact.}"
: "${CALIBRATION_CACHE:?Set CALIBRATION_CACHE to the shared train-only token artifact.}"
: "${CHANNEL_CACHE:?Set CHANNEL_CACHE to the matching channel topology cache.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for derived REAP profiles.}"

export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

declare -A PRUNE_COUNTS=(
  [25pct]=32
  [50pct]=64
  [stress]=76
)

for budget in 25pct 50pct stress; do
  "$PYTHON_BIN" code/scripts/build_reap_profile_from_observer.py \
    --observer-artifact "$OBSERVER_ARTIFACT" \
    --calibration-cache "$CALIBRATION_CACHE" \
    --channel-cache "$CHANNEL_CACHE" \
    --output-profile "$OUTPUT_DIR/reap_${budget}.pt" \
    --experts-to-prune-per-layer "${PRUNE_COUNTS[$budget]}" \
    --num-blocks 12 \
    --top-k 8
done