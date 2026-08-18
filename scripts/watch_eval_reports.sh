#!/usr/bin/env bash
set -Eeuo pipefail

# Fill in experiment directories to compare. Leave empty to discover every experiment under RESULT_ROOT;
# default discovery rejects non-compliant/unfinished experiments and compares only manual-compliant results.
EXPERIMENT_PATHS=(
    # "/data01/home/xinpei.gao/evalscope/result/Qwen330BA3BInstruct_50_vllm_CalibrationFree_quick9_PurePseudo-K8-Q4_202608062216_42"
    # "/data01/home/xinpei.gao/evalscope/result/Qwen330BA3BInstruct_50_vllm_WikiText128x2048_quick9_ENP_202608081200_42"
)

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
WATCH_SECONDS="${WATCH_SECONDS:-0}"  # 0 = one snapshot; positive value = refresh interval
SHOW_DETAILS="${SHOW_DETAILS:-false}"
ALLOW_INVALID="${ALLOW_INVALID:-false}"

REPORTER="$ROOT/scripts/watch_eval_reports.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f "$REPORTER" ]] || die "Reporter not found: $REPORTER"
[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -d "$RESULT_ROOT" ]] || die "RESULT_ROOT does not exist: $RESULT_ROOT"
[[ "$SHOW_DETAILS" == "true" || "$SHOW_DETAILS" == "false" ]] || die "SHOW_DETAILS must be true or false."
[[ "$ALLOW_INVALID" == "true" || "$ALLOW_INVALID" == "false" ]] || die "ALLOW_INVALID must be true or false."
"$PYTHON_BIN" -c 'import yaml' || die "PYTHON_BIN must provide PyYAML for task_config.yaml validation."

ARGS=("$REPORTER" "--result-root" "$RESULT_ROOT")
if [[ "$WATCH_SECONDS" != "0" ]]; then
    ARGS+=("--watch" "$WATCH_SECONDS")
fi
if [[ "$SHOW_DETAILS" == "true" ]]; then
    ARGS+=("--details")
fi
if [[ "$ALLOW_INVALID" == "true" ]]; then
    ARGS+=("--allow-invalid")
fi
if ((${#EXPERIMENT_PATHS[@]} > 0)); then
    ARGS+=("${EXPERIMENT_PATHS[@]}")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"