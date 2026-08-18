#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/PP/experiments/profiles/aimer_pp_norm_g5_g20_b6_20260807}"
QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-202608072100}"
QUEUE="${1:?Usage: $0 1|3|5}"

PP_WITH_NORM_CACHE="$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4/pure_pseudo_rankings.pt"
PP_NO_NORM_CACHE="$ROOT/PP/experiments/profiles/down_proj_norm_ablation_20260807/PurePseudo-K8-Q4-NoDownNorm/rankings.pt"
AIMER_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
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

ensure_experiment_dir() {
    local method=$1
    local expected created
    expected=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
        --inference vllm \
        --calibration CalibrationFree \
        --method "$method" \
        --pruning-ratio-label Prune6of12 \
        --pruning-ratio-percent 50 \
        --timestamp "$QUEUE_TIMESTAMP" \
        --dry-run)
    if [[ ! -d "$expected" ]]; then
        created=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
            --inference vllm \
            --calibration CalibrationFree \
            --method "$method" \
            --pruning-ratio-label Prune6of12 \
            --pruning-ratio-percent 50 \
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
    local protection_ratio=$2
    local pseudo_cache=$3
    local gpu_id=$4
    local port=$5
    local variant_root profile rankings experiment_dir checkpoint_dir

    variant_root="$PROFILE_ROOT/$method"
    profile="$variant_root/profile.pt"
    rankings="$variant_root/rankings.pt"
    experiment_dir=$(ensure_experiment_dir "$method")
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
            --backbone-cache "$AIMER_CACHE" \
            --pseudo-cache "$pseudo_cache" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --backbone aimer \
            --retained-blocks 6 \
            --protection-ratio "$protection_ratio"
    fi

    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels 384
    fi

    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-384ch-$method" \
        GPU_ID="$gpu_id" \
        PORT="$port" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_WITH_NORM_CACHE"
require_file "$PP_NO_NORM_CACHE"
require_file "$AIMER_CACHE"
require_file "$PROFILE_BUILDER"
require_file "$EXPORTER"
require_file "$CREATE_RESULT_DIR"
require_file "$QUICK9_RUNNER"

case "$QUEUE" in
    1)
        run_experiment AIMER-PP-WithNorm-G5-B6of12 0.05 "$PP_WITH_NORM_CACHE" 1 18111
        run_experiment AIMER-PP-NoDownNorm-G5-B6of12 0.05 "$PP_NO_NORM_CACHE" 1 18111
        ;;
    3) run_experiment AIMER-PP-WithNorm-G20-B6of12 0.20 "$PP_WITH_NORM_CACHE" 3 18113 ;;
    5) run_experiment AIMER-PP-NoDownNorm-G20-B6of12 0.20 "$PP_NO_NORM_CACHE" 5 18115 ;;
    *) die "Usage: $0 1|3|5" ;;
esac