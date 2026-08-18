#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/PP/experiments/profiles/requested_quick9_20260807}"
QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-202608070214}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-3600}"
GPU_MEMORY_LIMIT_MB="${GPU_MEMORY_LIMIT_MB:-2048}"
GPU_UTIL_LIMIT="${GPU_UTIL_LIMIT:-5}"
DRY_RUN="${DRY_RUN:-false}"
GROUP="${1:-all}"

PP_CACHE="$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4/pure_pseudo_rankings.pt"
AIMER_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
RANDOM_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_random_fixed_20260806/random_rankings.pt"
CREATE_RESULT_DIR="$CODE_ROOT/scripts/create_result_dir.sh"
PP_BUILDER="$ROOT/PP/build_pure_pseudo_profile.py"
PROTECTION_BUILDER="$ROOT/PP/build_protected_rankings.py"
PP_EXPORTER="$ROOT/PP/export_uniform_qwen3_moe.py"
GENERIC_EXPORTER="$ROOT/WICK/export_uniform_qwen3_moe.py"
QUICK9_RUNNER="$ROOT/PP/run_vllm_quick9.sh"

export PYTHONPATH="$ROOT:$CODE_ROOT"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

gpu_is_idle() {
    local gpu_id=$1
    local used util
    IFS=, read -r used util < <(
        nvidia-smi --id="$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    used=${used//[[:space:]]/}
    util=${util//[[:space:]]/}
    [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] || return 1
    (( used <= GPU_MEMORY_LIMIT_MB && util <= GPU_UTIL_LIMIT ))
}

wait_for_gpu() {
    local gpu_id=$1
    while ! gpu_is_idle "$gpu_id"; do
        echo "$(date --iso-8601=seconds) GPU $gpu_id is busy; retrying in ${GPU_WAIT_SECONDS}s" >&2
        sleep "$GPU_WAIT_SECONDS"
    done
    echo "$(date --iso-8601=seconds) GPU $gpu_id is idle" >&2
}

wait_for_port() {
    local port=$1
    while ! "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
    try:
        handle.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
    do
        echo "$(date --iso-8601=seconds) port $port is busy; retrying in ${GPU_WAIT_SECONDS}s" >&2
        sleep "$GPU_WAIT_SECONDS"
    done
}

pruning_identity() {
    local retained_blocks=$1
    case "$retained_blocks" in
        8) printf '%s %s\n' "Prune4of12" "33.3333" ;;
        6) printf '%s %s\n' "Prune6of12" "50" ;;
        *) die "Unsupported retained block count: $retained_blocks" ;;
    esac
}

target_pruning_ratio() {
    local retained_blocks=$1
    case "$retained_blocks" in
        8) printf '%s\n' "0.3333333333333333" ;;
        6) printf '%s\n' "0.5" ;;
        *) die "Unsupported retained block count: $retained_blocks" ;;
    esac
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
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%s\n' "$expected"
        return
    fi
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
    (( report_count >= 6 ))
}

prepare_output_dir() {
    local output_dir=$1
    if [[ -d "$output_dir" && ! -f "$output_dir/pruning_export_manifest.json" ]]; then
        mv "$output_dir" "${output_dir}.incomplete.$(date +%Y%m%d%H%M%S)"
    fi
}

run_quick9() {
    local method=$1
    local retained_blocks=$2
    local gpu_id=$3
    local port=$4
    local checkpoint_dir=$5
    local experiment_dir=$6
    local retained_channels=$((retained_blocks * 64))

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN quick9 method=$method blocks=$retained_blocks gpu=$gpu_id port=$port checkpoint=$checkpoint_dir result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    wait_for_gpu "$gpu_id"
    wait_for_port "$port"
    CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-${retained_channels}ch-$method" \
        GPU_ID="$gpu_id" \
        PORT="$port" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_pure_pseudo_variant() {
    local method=$1
    local neighbors=$2
    local top_q=$3
    local probe_signs=$4
    local retained_blocks=$5
    local gpu_id=$6
    local port=$7
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local cache="$variant_root/rankings.pt"
    local experiment_dir checkpoint_dir pruning_ratio

    echo "$(date --iso-8601=seconds) starting $method on preferred GPU $gpu_id"
    pruning_ratio=$(target_pruning_ratio "$retained_blocks")
    if [[ "$DRY_RUN" == "true" ]]; then
        experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
        checkpoint_dir="$experiment_dir/checkpoints/$method"
        echo "DRY_RUN pure-pseudo method=$method K=$neighbors Q=$top_q signs=$probe_signs blocks=$retained_blocks ratio=$pruning_ratio"
        run_quick9 "$method" "$retained_blocks" "$gpu_id" "$port" "$checkpoint_dir" "$experiment_dir"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$cache" ]]; then
        wait_for_gpu "$gpu_id"
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$PP_BUILDER" \
            --model-path "$MODEL_PATH" \
            --output-profile "$profile" \
            --output-channel-cache "$cache" \
            --target-pruning-ratio "$pruning_ratio" \
            --router-neighbors "$neighbors" \
            --top-q "$top_q" \
            --probe-signs "$probe_signs" \
            --channel-block-size 64 \
            --device cuda:0
    fi

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$PP_EXPORTER" \
            --model-path "$MODEL_PATH" \
            --profile "$profile" \
            --channel-cache "$cache" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    run_quick9 "$method" "$retained_blocks" "$gpu_id" "$port" "$checkpoint_dir" "$experiment_dir"
}

run_protection_variant() {
    local backbone=$1
    local gamma_label=$2
    local gamma=$3
    local retained_blocks=$4
    local gpu_id=$5
    local port=$6
    local method backbone_cache variant_root profile cache experiment_dir checkpoint_dir

    case "$backbone" in
        aimer)
            method="AIMER-PP-G${gamma_label}-B${retained_blocks}of12"
            backbone_cache="$AIMER_CACHE"
            ;;
        random)
            method="Random-PP-G${gamma_label}-B${retained_blocks}of12"
            backbone_cache="$RANDOM_CACHE"
            ;;
        *) die "Unknown protection backbone: $backbone" ;;
    esac
    variant_root="$PROFILE_ROOT/$method"
    profile="$variant_root/profile.pt"
    cache="$variant_root/rankings.pt"

    echo "$(date --iso-8601=seconds) starting $method on preferred GPU $gpu_id"
    if [[ "$DRY_RUN" == "true" ]]; then
        experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
        checkpoint_dir="$experiment_dir/checkpoints/$method"
        echo "DRY_RUN protection method=$method backbone=$backbone gamma=$gamma blocks=$retained_blocks"
        run_quick9 "$method" "$retained_blocks" "$gpu_id" "$port" "$checkpoint_dir" "$experiment_dir"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$cache" ]]; then
        mkdir -p "$variant_root"
        "$PYTHON_BIN" "$PROTECTION_BUILDER" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$backbone_cache" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$cache" \
            --method "${backbone}_pp_g${gamma_label}_b${retained_blocks}of12" \
            --backbone "$backbone" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio "$gamma"
    fi

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$GENERIC_EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$cache" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    run_quick9 "$method" "$retained_blocks" "$gpu_id" "$port" "$checkpoint_dir" "$experiment_dir"
}

run_group_1() {
    run_pure_pseudo_variant PurePseudo-K8-Q9-B8of12 8 9 positive 8 1 18101
    run_pure_pseudo_variant PurePseudo-K16-Q17-B8of12 16 17 positive 8 1 18101
    run_pure_pseudo_variant PurePseudo-K16-Q4-B8of12 16 4 positive 8 1 18101
    run_pure_pseudo_variant PurePseudo-K0-Q1-B8of12 0 1 positive 8 1 18101
}

run_group_2() {
    run_pure_pseudo_variant PurePseudo-K8-Q4-PosNeg-B8of12 8 4 positive-negative 8 2 18102
}

run_group_3() {
    local backbone retained gamma_label gamma
    for backbone in aimer random; do
        for retained in 6 8; do
            for specification in 5:0.05 10:0.10 20:0.20; do
                gamma_label=${specification%%:*}
                gamma=${specification#*:}
                run_protection_variant "$backbone" "$gamma_label" "$gamma" "$retained" 3 18103
            done
        done
    done
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_CACHE"
require_file "$AIMER_CACHE"
require_file "$RANDOM_CACHE"
require_file "$CREATE_RESULT_DIR"

case "$GROUP" in
    1) run_group_1 ;;
    2) run_group_2 ;;
    3) run_group_3 ;;
    all)
        echo "Use groups 1, 2, and 3 as separate processes so they run concurrently." >&2
        exit 2
        ;;
    *) die "Usage: $0 1|2|3" ;;
esac