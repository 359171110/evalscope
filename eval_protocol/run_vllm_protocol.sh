#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 4 ]]; then
    cat <<EOF >&2
Usage: $0 MODEL_ID API_URL METHOD EXPERIMENT_DIR

Required environment:
  PROTOCOL=quick9|full6_v1|full6_unlimited
  PYTHON_BIN, ARC_PATH, HELLASWAG_PATH, WINOGRANDE_PATH, GSM8K_PATH, MATH_500_PATH, MMLU_PATH

Optional:
  DATASETS=arc,hellaswag
  DRY_RUN=true
EOF
    exit 2
fi

MODEL_ID=$1
API_URL=$2
METHOD=$3
EXPERIMENT_DIR=$4
PROTOCOL="${PROTOCOL:-quick9}"

ARGS=(
    "$PYTHON_BIN" "$ROOT/eval_protocol/run_vllm_protocol.py"
    --protocol "$PROTOCOL"
    --model-id "$MODEL_ID"
    --api-url "$API_URL"
    --method "$METHOD"
    --experiment-dir "$EXPERIMENT_DIR"
    --python-bin "$PYTHON_BIN"
)
if [[ -n "${DATASETS:-}" ]]; then
    ARGS+=(--datasets "$DATASETS")
fi
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    ARGS+=(--dry-run)
fi

exec "${ARGS[@]}"
