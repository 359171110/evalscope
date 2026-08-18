#!/usr/bin/env bash

set -euo pipefail

TENP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TENP_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-$ROOT/static_moe_prunning/experiments/calibration/reap_50pct_screening/c1_wikitext_train_128x2048.pt}"
EXPECTED_CACHE_SHA256="${EXPECTED_CACHE_SHA256:-11324347a87608d294c47a157a5f3791a72e75b0ab7c5752fae096473da5ffb1}"
EXPECTED_PROTOCOL_NAME="${EXPECTED_PROTOCOL_NAME:-c1_wikitext_train_128x2048_seed42_screening_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/static_moe_prunning/experiments/calibration/qwen3_wikitext128x2048_enp}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/static_moe_prunning/experiments/profiles/qwen3_wikitext128x2048_enp}"
STATISTICS_CACHE="$ARTIFACT_ROOT/enp_statistics.pt"
CHANNEL_CACHE="$ARTIFACT_ROOT/enp_signed_projection_channels_b64.pt"
BUILD_GPU="${BUILD_GPU:-1}"
EVAL_GPU_25="${EVAL_GPU_25:-1}"
EVAL_GPU_50="${EVAL_GPU_50:-5}"
PORT_25="${PORT_25:-18225}"
PORT_50="${PORT_50:-18250}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
ACTION="${1:-dry-run}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

validate_gpu() {
    [[ "$1" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer; got '$1'."
}

profile_path() {
    case "$1" in
        25) printf '%s/enp_25pct_per_layer.pt\n' "$PROFILE_ROOT" ;;
        50) printf '%s/enp_50pct_per_layer.pt\n' "$PROFILE_ROOT" ;;
        *) die "Pruning ratio must be 25 or 50." ;;
    esac
}

retained_channels() {
    case "$1" in
        25) printf '576\n' ;;
        50) printf '384\n' ;;
        *) die "Pruning ratio must be 25 or 50." ;;
    esac
}

experiment_dir() {
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" "$CODE_ROOT/scripts/create_result_dir.sh" \
        --inference vllm \
        --calibration WikiText128x2048 \
        --method ENP \
        --protocol full6_v1 \
        --pruning-ratio "$1" \
        --timestamp "$TIMESTAMP" \
        --dry-run
}

verify_inputs() {
    [[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
    [[ -x "$VLLM_PYTHON" ]] || die "vLLM Python executable is not executable: $VLLM_PYTHON"
    [[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
    [[ -f "$CALIBRATION_CACHE" ]] || die "Calibration cache does not exist: $CALIBRATION_CACHE"
    local actual_sha256
    actual_sha256="$(sha256sum "$CALIBRATION_CACHE" | awk '{print $1}')"
    [[ "$actual_sha256" == "$EXPECTED_CACHE_SHA256" ]] ||
        die "Calibration cache SHA256 mismatch: expected $EXPECTED_CACHE_SHA256, got $actual_sha256"
    validate_gpu "$BUILD_GPU"
    validate_gpu "$EVAL_GPU_25"
    validate_gpu "$EVAL_GPU_50"
}

build_profiles() {
    mkdir -p "$ARTIFACT_ROOT" "$PROFILE_ROOT"
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$BUILD_GPU"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        "$CODE_ROOT/scripts/build_enp_tenp_profiles.py"
        --model-path "$MODEL_PATH"
        --model-family qwen3
        --calibration-cache "$CALIBRATION_CACHE"
        --output-statistics "$STATISTICS_CACHE"
        --output-channel-cache "$CHANNEL_CACHE"
        --output-profile-dir "$PROFILE_ROOT"
        --routed-param-retention 0.75 0.50
        --important-expert-ratio 0.30
        --shallow-weight 1.0
        --deep-weight 2.0
        --channel-block-size 64
        --sequence-length 2048
        --calibration-sequences 128
        --min-tokens-per-expert 32
        --allow-undercovered-experts
        --zero-token-policy prune_uniform
        --device-map cuda:0
    )
    print_command "${command[@]}"
    "${command[@]}"
}

ensure_experiment_dir() {
    local ratio="$1"
    local directory
    directory="$(experiment_dir "$ratio")"
    if [[ ! -f "$directory/experiment_manifest.json" ]]; then
        RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" "$CODE_ROOT/scripts/create_result_dir.sh" \
            --inference vllm \
            --calibration WikiText128x2048 \
            --method ENP \
            --protocol full6_v1 \
            --pruning-ratio "$ratio" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$directory"
}

export_checkpoint() {
    local ratio="$1"
    local directory profile checkpoint channels
    directory="$(ensure_experiment_dir "$ratio")"
    profile="$(profile_path "$ratio")"
    checkpoint="$directory/checkpoints/ENP"
    channels="$(retained_channels "$ratio")"
    [[ -f "$profile" ]] || die "ENP profile does not exist: $profile"
    [[ -f "$CHANNEL_CACHE" ]] || die "ENP channel cache does not exist: $CHANNEL_CACHE"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        printf '%s\n' "$checkpoint"
        return
    fi
    mkdir -p "$checkpoint"
    "$PYTHON_BIN" "$CODE_ROOT/scripts/export_uniform_enp_qwen3_moe.py" \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$CHANNEL_CACHE" \
        --output-dir "$checkpoint" \
        --retained-channels "$channels" \
        --expected-protocol-name "$EXPECTED_PROTOCOL_NAME"
    printf '%s\n' "$checkpoint"
}

run_final6() {
    local ratio="$1"
    local gpu="$2"
    local port="$3"
    local directory checkpoint model_id server_log server_pid
    directory="$(ensure_experiment_dir "$ratio")"
    checkpoint="$directory/checkpoints/ENP"
    model_id="Qwen330BA3BInstruct-${ratio}-ENP-final6-v1"
    [[ -f "$checkpoint/pruning_export_manifest.json" ]] || die "Checkpoint is not exported: $checkpoint"
    server_log="$directory/server_logs/ENP.log"
    local -a server_command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
        --model "$checkpoint"
        --served-model-name "$model_id"
        --host 127.0.0.1
        --port "$port"
        --dtype bfloat16
        --seed 42
        --max-model-len 8192
        --max-num-seqs 16
        --gpu-memory-utilization 0.90
        --generation-config vllm
        --default-chat-template-kwargs '{"enable_thinking":false}'
    )
    "${server_command[@]}" >"$server_log" 2>&1 &
    server_pid=$!
    cleanup_server() {
        kill -TERM "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server EXIT INT TERM
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -100 "$server_log" >&2
            die "vLLM server exited before becoming healthy: $server_log"
        fi
        sleep 1
    done
    curl --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null || die "vLLM server health check failed."
    RESULTS_ROOT="$directory" \
    DATASETS="${DATASETS:-arc,hellaswag,winogrande,gsm8k,math_500,mmlu}" \
    bash "$ROOT/WICK/run_vllm_full6.sh" "$model_id" "http://127.0.0.1:$port" ENP "$directory"
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
    trap - EXIT INT TERM
}

dry_run() {
    echo "Calibration cache: $CALIBRATION_CACHE"
    echo "Calibration SHA256: $EXPECTED_CACHE_SHA256"
    echo "Calibration protocol: $EXPECTED_PROTOCOL_NAME"
    echo "Build GPU: $BUILD_GPU"
    echo "ENP-25 profile: $(profile_path 25); retained channels: $(retained_channels 25)"
    echo "ENP-50 profile: $(profile_path 50); retained channels: $(retained_channels 50)"
    echo "ENP-25 experiment: $(experiment_dir 25)"
    echo "ENP-50 experiment: $(experiment_dir 50)"
    echo "Final6/full6_v1: ARC 600, HellaSwag 1000, WinoGrande 400, GSM8K 128, MATH-500 5x20, MMLU 570"
    echo "MATH-500 max_tokens: 4096; seed: 42; Dense: skipped"
}

verify_inputs
case "$ACTION" in
    dry-run) dry_run ;;
    build) build_profiles ;;
    export)
        export_checkpoint 25
        export_checkpoint 50
        ;;
    eval-25) run_final6 25 "$EVAL_GPU_25" "$PORT_25" ;;
    eval-50) run_final6 50 "$EVAL_GPU_50" "$PORT_50" ;;
    all)
        build_profiles
        export_checkpoint 25
        export_checkpoint 50
        run_final6 25 "$EVAL_GPU_25" "$PORT_25"
        run_final6 50 "$EVAL_GPU_50" "$PORT_50"
        ;;
    *) die "Usage: $0 dry-run|build|export|eval-25|eval-50|all" ;;
esac