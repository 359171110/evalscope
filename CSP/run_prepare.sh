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
HETEROGENEOUS_WIDTHS="${CSP_HETEROGENEOUS_WIDTHS:-}"
BUDGET_WIDTH="${CSP_BUDGET_WIDTH:-}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

if [[ -n "$HETEROGENEOUS_WIDTHS" || -n "$BUDGET_WIDTH" ]]; then
    [[ "$MODEL" != "all" ]] || die "Use one model at a time in heterogeneous mode."
    [[ -n "$HETEROGENEOUS_WIDTHS" && -n "$BUDGET_WIDTH" ]] || die "Set CSP_HETEROGENEOUS_WIDTHS and CSP_BUDGET_WIDTH together."
    [[ -z "$RETAINED_CHANNELS" ]] || die "Do not pass RETAINED_CHANNELS in heterogeneous mode."
    [[ -z "${CSP_CANONICALIZE:-}" ]] || die "HSP-Hetero uses raw Expert-SP and raw Channel-SP; do not set CSP_CANONICALIZE."
else
    [[ -n "$MODEL" && -n "$RETAINED_CHANNELS" ]] || die "Usage: $0 MODEL RETAINED_CHANNELS OUTPUT_ROOT"
fi
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

if [[ -n "$HETEROGENEOUS_WIDTHS" ]]; then
    case "$MODEL:$BUDGET_WIDTH:$HETEROGENEOUS_WIDTHS" in
        qwen3:384:"320 384 448") ;;
        qwen36:256:"192 256 320") ;;
        gemma4:352:"288 352 416") ;;
        deepseek:704:"576 704 832") ;;
        *) die "Unsupported HSP-Hetero configuration for $MODEL: widths='$HETEROGENEOUS_WIDTHS', budget='$BUDGET_WIDTH'." ;;
    esac
    ARTIFACT_ROOT="$OUTPUT_ROOT/${MODEL}_heterogeneous_${HETEROGENEOUS_WIDTHS// /_}_budget${BUDGET_WIDTH}"
else
    ARTIFACT_ROOT="$OUTPUT_ROOT/$MODEL"
fi
CACHE="$ARTIFACT_ROOT/csp_rankings.pt"
if [[ -n "$HETEROGENEOUS_WIDTHS" ]]; then
    PROFILE="$ARTIFACT_ROOT/csp_heterogeneous_budget${BUDGET_WIDTH}ch.pt"
    CHECKPOINT="$ARTIFACT_ROOT/checkpoint_heterogeneous_${HETEROGENEOUS_WIDTHS// /_}_budget${BUDGET_WIDTH}"
else
    PROFILE="$ARTIFACT_ROOT/csp_${RETAINED_CHANNELS}ch.pt"
    CHECKPOINT="$ARTIFACT_ROOT/checkpoint_${RETAINED_CHANNELS}ch"
fi
mkdir -p "$ARTIFACT_ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
build_args=(
    "$PYTHON_BIN" -m CSP.build_csp_artifacts
    --model-path "$MODEL_PATH" \
    --output-channel-cache "$CACHE" \
    --output-profile "$PROFILE"
)
if [[ -n "$HETEROGENEOUS_WIDTHS" ]]; then
    read -r -a width_options <<< "$HETEROGENEOUS_WIDTHS"
    build_args+=(--heterogeneous-widths "${width_options[@]}" --budget-width "$BUDGET_WIDTH")
else
    build_args+=(--retained-channels "$RETAINED_CHANNELS")
fi
[[ -n "${CSP_CANONICALIZE:-}" ]] && build_args+=(--canonicalize)
"${build_args[@]}"
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
