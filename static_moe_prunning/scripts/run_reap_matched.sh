#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PYTHON_BIN:?Set PYTHON_BIN to the xhquant Python executable.}"
: "${MODEL_PATH:?Set MODEL_PATH to the base MoE checkpoint.}"
: "${OFFICIAL_REAP_ROOT:?Set OFFICIAL_REAP_ROOT to the clean official REAP checkout.}"
: "${OFFICIAL_REAP_COMMIT:?Set OFFICIAL_REAP_COMMIT to the frozen official commit.}"
: "${CALIBRATION_CACHE:?Set CALIBRATION_CACHE to the shared train-only token artifact.}"
: "${CHANNEL_CACHE:?Set CHANNEL_CACHE to the matching train-only channel topology cache.}"
: "${OUTPUT_OBSERVER:?Set OUTPUT_OBSERVER for the official observer artifact.}"
: "${OUTPUT_PROFILE:?Set OUTPUT_PROFILE for the frozen REAP profile.}"
: "${EXPERTS_TO_PRUNE_PER_LAYER:?Set EXPERTS_TO_PRUNE_PER_LAYER.}"
: "${GPU:?Set GPU to one physical GPU in 4,5,6,7.}"

case "$GPU" in
  4|5|6|7) ;;
  *) echo "GPU must be one of physical devices 4,5,6,7." >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" code/scripts/build_official_reap_profile.py \
  --official-reap-root "$OFFICIAL_REAP_ROOT" \
  --official-reap-commit "$OFFICIAL_REAP_COMMIT" \
  --model-path "$MODEL_PATH" \
  --model-family "${MODEL_FAMILY:-qwen3}" \
  --calibration-cache "$CALIBRATION_CACHE" \
  --channel-cache "$CHANNEL_CACHE" \
  --output-observer "$OUTPUT_OBSERVER" \
  --output-profile "$OUTPUT_PROFILE" \
  --experts-to-prune-per-layer "$EXPERTS_TO_PRUNE_PER_LAYER" \
  --sequence-length 2048 \
  --batch-group-size "${BATCH_GROUP_SIZE:-8}" \
  --device-map cpu