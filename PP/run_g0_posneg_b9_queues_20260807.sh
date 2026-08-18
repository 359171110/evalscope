#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/PP/experiments/profiles/g0_posneg_b9_20260807}"
QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-202608071314}"
QUEUE="${1:?Usage: $0 1|2|3|5}"

PP_POSITIVE_CACHE="$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4/pure_pseudo_rankings.pt"
PP_POSNEG_CACHE="$ROOT/PP/experiments/profiles/requested_quick9_20260807/PurePseudo-K8-Q4-PosNeg-B8of12/rankings.pt"
AIMER_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
RANDOM_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_random_fixed_20260806/random_rankings.pt"
PROFILE_BUILDER="$ROOT/PP/build_protected_rankings.py"
EXPORTER="$ROOT/WICK/export_uniform_qwen3_moe.py"
CREATE_RESULT_DIR="$CODE_ROOT/scripts/create_result_dir.sh"
QUICK9_RUNNER="$ROOT/PP/run_vllm_quick9.sh"

export PYTHONPATH="$ROOT:$CODE_ROOT"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

pruning_identity() {
    local retained_blocks=$1
    local pruned_blocks=$((12 - retained_blocks))
    local percent
    percent=$("$PYTHON_BIN" - "$pruned_blocks" <<'PY'
import sys

print(f"{int(sys.argv[1]) / 12 * 100:.4f}".rstrip("0").rstrip("."))
PY
)
    printf 'Prune%sof12 %s\n' "$pruned_blocks" "$percent"
}

ensure_experiment_dir() {
    local method=$1
    local retained_blocks=$2
    local label percent expected created
    read -r label percent < <(pruning_identity "$retained_blocks")
    expected=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
        --inference vllm \
        --calibration CalibrationFree \
        --method "$method" \
        --pruning-ratio-label "$label" \
        --pruning-ratio-percent "$percent" \
        --timestamp "$QUEUE_TIMESTAMP" \
        --dry-run)
    if [[ ! -d "$expected" ]]; then
        created=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
            --inference vllm \
            --calibration CalibrationFree \
            --method "$method" \
            --pruning-ratio-label "$label" \
            --pruning-ratio-percent "$percent" \
            --timestamp "$QUEUE_TIMESTAMP")
        [[ "$created" == "$expected" ]] || die "Unexpected result directory: $created"
    fi
    printf '%s\n' "$expected"
}

quick9_complete() {
    local experiment_dir=$1
    local method=$2
    local report_count=0
    if [[ -d "$experiment_dir/$method" ]]; then
        report_count=$(find "$experiment_dir/$method" -type f -path '*/reports/*/*.json' | wc -l)
    fi
    ((report_count >= 6))
}

prepare_output_dir() {
    local output_dir=$1
    if [[ -d "$output_dir" && ! -f "$output_dir/pruning_export_manifest.json" ]]; then
        mv "$output_dir" "${output_dir}.incomplete.$(date +%Y%m%d%H%M%S)"
    fi
}

run_experiment() {
    local method=$1
    local backbone=$2
    local protection_ratio=$3
    local retained_blocks=$4
    local pseudo_cache=$5
    local gpu_id=$6
    local port=$7
    local backbone_cache variant_root profile rankings experiment_dir checkpoint_dir

    case "$backbone" in
        aimer) backbone_cache="$AIMER_CACHE" ;;
        random) backbone_cache="$RANDOM_CACHE" ;;
        *) die "Unknown backbone: $backbone" ;;
    esac

    variant_root="$PROFILE_ROOT/$method"
    profile="$variant_root/profile.pt"
    rankings="$variant_root/rankings.pt"
    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"

    echo "$(date --iso-8601=seconds) starting method=$method gpu=$gpu_id port=$port"
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi

    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$variant_root"
        "$PYTHON_BIN" "$PROFILE_BUILDER" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$backbone_cache" \
            --pseudo-cache "$pseudo_cache" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --backbone "$backbone" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio "$protection_ratio"
    fi

    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi

    CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$gpu_id" \
        PORT="$port" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_POSITIVE_CACHE"
require_file "$PP_POSNEG_CACHE"
require_file "$AIMER_CACHE"
require_file "$RANDOM_CACHE"
require_file "$PROFILE_BUILDER"
require_file "$EXPORTER"
require_file "$CREATE_RESULT_DIR"
require_file "$QUICK9_RUNNER"

case "$QUEUE" in
    1)
        run_experiment AIMER-G0-B8of12 aimer 0.00 8 "$PP_POSITIVE_CACHE" 1 18111
        run_experiment AIMER-PP-PosNeg-G5-B8of12 aimer 0.05 8 "$PP_POSNEG_CACHE" 1 18111
        run_experiment AIMER-PP-G5-B9of12 aimer 0.05 9 "$PP_POSITIVE_CACHE" 1 18111
        ;;
    2)
        run_experiment Random-G0-B8of12 random 0.00 8 "$PP_POSITIVE_CACHE" 2 18112
        run_experiment AIMER-PP-PosNeg-G10-B8of12 aimer 0.10 8 "$PP_POSNEG_CACHE" 2 18112
        ;;
    3)
        run_experiment AIMER-G0-B6of12 aimer 0.00 6 "$PP_POSITIVE_CACHE" 3 18113
        run_experiment AIMER-G0-B9of12 aimer 0.00 9 "$PP_POSITIVE_CACHE" 3 18113
        ;;
    5)
        run_experiment Random-G0-B6of12 random 0.00 6 "$PP_POSITIVE_CACHE" 5 18115
        run_experiment AIMER-PP-G10-B9of12 aimer 0.10 9 "$PP_POSITIVE_CACHE" 5 18115
        ;;
    *) die "Usage: $0 1|2|3|5" ;;
esac