#!/usr/bin/env bash

# Build CSP rankings and export one supported model family.
# Usage: MODEL_PATH=/path/to/model bash CSP/run_prepare.sh MODEL RETAINED_CHANNELS OUTPUT_ROOT
set -euo pipefail

CSP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CSP_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${1:-}"
RETAINED_CHANNELS="${2:-}"
OUTPUT_ROOT="${3:-$CSP_ROOT/experiments}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$RETAINED_CHANNELS" ]] || die "Usage: $0 MODEL RETAINED_CHANNELS OUTPUT_ROOT"
case "$MODEL" in
    qwen3) DEFAULT_MODEL_PATH="${QWEN3_MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}" ;;
    qwen36) DEFAULT_MODEL_PATH="${QWEN36_MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}" ;;
    gemma4) DEFAULT_MODEL_PATH="${GEMMA4_MODEL_PATH:-/data01/datasets/gemma-4-26B-A4B-it}" ;;
    deepseek) DEFAULT_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data01/datasets/DeepSeek-V2-Lite-Chat}" ;;
    all) die "Use one model at a time because model paths can be on different servers." ;;
    *) die "MODEL must be qwen3, qwen36, gemma4, or deepseek." ;;
esac
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
[[ -n "$MODEL_PATH" ]] || die "Set the corresponding model path environment variable."
[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"

ARTIFACT_ROOT="$OUTPUT_ROOT/$MODEL"
CACHE="$ARTIFACT_ROOT/csp_rankings.pt"
PROFILE="$ARTIFACT_ROOT/csp_${RETAINED_CHANNELS}ch.pt"
CHECKPOINT="$ARTIFACT_ROOT/checkpoint_${RETAINED_CHANNELS}ch"
mkdir -p "$ARTIFACT_ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m CSP.build_csp_artifacts \
    --model-path "$MODEL_PATH" \
    --output-channel-cache "$CACHE" \
    --output-profile "$PROFILE" \
    --retained-channels "$RETAINED_CHANNELS" \
    ${CSP_CANONICALIZE:+--canonicalize}
if [[ -f "$CHECKPOINT/pruning_export_manifest.json" ]]; then
    printf 'checkpoint=%s\n' "$CHECKPOINT"
    exit 0
fi
"$PYTHON_BIN" -m CSP.export_csp_checkpoint \
    --model-path "$MODEL_PATH" \
    --profile "$PROFILE" \
    --channel-cache "$CACHE" \
    --output-dir "$CHECKPOINT"
printf 'checkpoint=%s\n' "$CHECKPOINT"
