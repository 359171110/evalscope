#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$DEFAULT_ROOT/result}"
MODEL="${MODEL_NAME:-Qwen330BA3BInstruct}"
PRUNING_RATIO_LABEL="50"
PRUNING_RATIO_PERCENT="50"
INFERENCE=""
CALIBRATION=""
PROTOCOL="quick9"
METHOD=""
TIMESTAMP=""
SEED="42"
DRY_RUN="false"
PRUNING_RATIO_SHORTHAND="false"

usage() {
    cat <<EOF
Usage:
  create_result_dir.sh \
    --inference vllm|transformer \
    --calibration WikiText128x2048|Mixed512x1024|CalibrationFree \
        [--model NAME] \
        [--protocol quick9|full6_v1|full8_v1] \
    --method NAME \
        [--pruning-ratio 0|25|50] \
        [--pruning-ratio-label NAME] \
        [--pruning-ratio-percent NUMBER] \
    [--timestamp YYYYMMDDHHMM] \
    [--dry-run]

The result directory is created under \$RESULT_ROOT
with this frozen name:
    <model>_<pruning>_<inference>_<calibration>_<protocol>_<method>_<timestamp>_42

Method names may contain only letters, numbers, and hyphens. Underscores are
forbidden because they separate experiment identity fields.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "Missing value for $1."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            require_value "$@"
            MODEL="$2"
            shift 2
            ;;
        --inference)
            require_value "$@"
            INFERENCE="$2"
            shift 2
            ;;
        --calibration)
            require_value "$@"
            CALIBRATION="$2"
            shift 2
            ;;
        --protocol)
            require_value "$@"
            PROTOCOL="$2"
            shift 2
            ;;
        --method)
            require_value "$@"
            METHOD="$2"
            shift 2
            ;;
        --pruning-ratio)
            require_value "$@"
            PRUNING_RATIO_SHORTHAND="true"
            PRUNING_RATIO_LABEL="$2"
            PRUNING_RATIO_PERCENT="$2"
            shift 2
            ;;
        --pruning-ratio-label)
            require_value "$@"
            PRUNING_RATIO_LABEL="$2"
            shift 2
            ;;
        --pruning-ratio-percent)
            require_value "$@"
            PRUNING_RATIO_PERCENT="$2"
            shift 2
            ;;
        --timestamp)
            require_value "$@"
            TIMESTAMP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option '$1'."
            ;;
    esac
done

[[ "$INFERENCE" == "vllm" || "$INFERENCE" == "transformer" ]] ||
    die "Inference must be vllm or transformer."
case "$CALIBRATION" in
    WikiText128x2048|Mixed512x1024|CalibrationFree) ;;
    *) die "Unknown calibration identity '$CALIBRATION'." ;;
esac
[[ "$PROTOCOL" == "quick9" || "$PROTOCOL" == "full6_v1" || "$PROTOCOL" == "full8_v1" ]] ||
    die "Protocol must be quick9, full6_v1, or full8_v1."
[[ "$MODEL" =~ ^[A-Za-z0-9]+([A-Za-z0-9.-]*[A-Za-z0-9])?$ ]] ||
    die "Model must contain only letters, numbers, dots, and hyphens."
[[ "$METHOD" =~ ^[A-Za-z0-9]+([A-Za-z0-9-]*[A-Za-z0-9])?$ ]] ||
    die "Method must contain only letters, numbers, and internal hyphens."
[[ "$PRUNING_RATIO_LABEL" =~ ^[A-Za-z0-9]+([A-Za-z0-9-]*[A-Za-z0-9])?$ ]] ||
    die "Pruning ratio label must contain only letters, numbers, and internal hyphens."
[[ "$PRUNING_RATIO_PERCENT" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "Pruning ratio percent must be a non-negative number."
if [[ "$PRUNING_RATIO_SHORTHAND" == "true" ]]; then
    [[ "$PRUNING_RATIO_LABEL" == "0" || "$PRUNING_RATIO_LABEL" == "25" || "$PRUNING_RATIO_LABEL" == "50" ]] ||
        die "Pruning ratio must be 0, 25 or 50."
fi

if [[ -z "$TIMESTAMP" ]]; then
    TIMESTAMP="$(date +%Y%m%d%H%M)"
fi
[[ "$TIMESTAMP" =~ ^[0-9]{12}$ ]] || die "Timestamp must use YYYYMMDDHHMM."
date -d "${TIMESTAMP:0:8} ${TIMESTAMP:8:2}:${TIMESTAMP:10:2}" +%Y%m%d%H%M >/dev/null 2>&1 ||
    die "Timestamp is not a valid calendar minute: $TIMESTAMP."

EXPERIMENT_NAME="${MODEL}_${PRUNING_RATIO_LABEL}_${INFERENCE}_${CALIBRATION}_${PROTOCOL}_${METHOD}_${TIMESTAMP}_${SEED}"
EXPERIMENT_DIR="$RESULT_ROOT/$EXPERIMENT_NAME"

if [[ "$DRY_RUN" == "true" ]]; then
    printf '%s\n' "$EXPERIMENT_DIR"
    exit 0
fi

[[ ! -e "$EXPERIMENT_DIR" ]] || die "Result directory already exists: $EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/checkpoints" "$EXPERIMENT_DIR/server_logs"

cat >"$EXPERIMENT_DIR/experiment_manifest.json" <<EOF
{
  "schema_version": 1,
  "target_model": "$MODEL",
    "pruning_ratio_label": "$PRUNING_RATIO_LABEL",
    "pruning_ratio_percent": $PRUNING_RATIO_PERCENT,
  "inference": "$INFERENCE",
  "calibration": "$CALIBRATION",
  "evaluation_protocol": "$PROTOCOL",
  "method": "$METHOD",
  "started_at_minute": "$TIMESTAMP",
  "seed": 42,
  "result_root": "$RESULT_ROOT",
  "experiment_name": "$EXPERIMENT_NAME"
}
EOF

printf '%s\n' "$EXPERIMENT_DIR"