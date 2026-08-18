#!/usr/bin/env bash

set -euo pipefail

TENP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TENP_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/gemma-4-26B-A4B-it}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-$ROOT/static_moe_prunning/experiments/calibration/gemma4_wikitext128x2048_v1/calibration.pt}"
EXPECTED_CACHE_SHA256="${EXPECTED_CACHE_SHA256:-4ef0fe7aa02c4d2825c85cd8d6b6a38d66907b761580e8316cd9d022f32649e2}"
EXPECTED_PROTOCOL_NAME="${EXPECTED_PROTOCOL_NAME:-gemma4_wikitext128x2048_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/static_moe_prunning/experiments/calibration/gemma4_wikitext128x2048_enp}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/static_moe_prunning/experiments/profiles/gemma4_wikitext128x2048_enp}"
STATISTICS_CACHE="$ARTIFACT_ROOT/enp_statistics.pt"
CHANNEL_CACHE="$ARTIFACT_ROOT/enp_signed_projection_channels_b64.pt"
BUILD_GPU="${BUILD_GPU:-6}"
EVAL_GPU_25="${EVAL_GPU_25:-6}"
EVAL_GPU_50="${EVAL_GPU_50:-6}"
PORT_25="${PORT_25:-18261}"
PORT_50="${PORT_50:-18262}"
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
        25) printf '512\n' ;;
        50) printf '384\n' ;;
        *) die "Pruning ratio must be 25 or 50." ;;
    esac
}

experiment_dir() {
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" MODEL_NAME="Gemma4-26B-A4B" \
        "$CODE_ROOT/scripts/create_result_dir.sh" \
        --model Gemma4-26B-A4B \
        --inference vllm \
        --calibration WikiText128x2048 \
        --method ENP \
        --protocol full6_v1 \
        --pruning-ratio-label "$1" \
        --pruning-ratio-percent "$1" \
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
    if [[ -f "$(profile_path 25)" && -f "$(profile_path 50)" && -f "$CHANNEL_CACHE" ]]; then
        echo "ENP profiles already exist under $PROFILE_ROOT"
        return
    fi
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$BUILD_GPU"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$VLLM_PYTHON"
        "$CODE_ROOT/scripts/build_enp_tenp_profiles.py"
        --model-path "$MODEL_PATH"
        --model-family gemma4
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
        --device-map none
        --device cuda
    )
    print_command "${command[@]}"
    "${command[@]}"
}

ensure_experiment_dir() {
    local ratio="$1"
    local directory
    directory="$(experiment_dir "$ratio")"
    if [[ ! -f "$directory/experiment_manifest.json" ]]; then
        RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" MODEL_NAME="Gemma4-26B-A4B" \
            "$CODE_ROOT/scripts/create_result_dir.sh" \
            --model Gemma4-26B-A4B \
            --inference vllm \
            --calibration WikiText128x2048 \
            --method ENP \
            --protocol full6_v1 \
            --pruning-ratio-label "$ratio" \
            --pruning-ratio-percent "$ratio" \
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
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        "$CODE_ROOT/scripts/export_uniform_enp_qwen3_moe.py"
        --model-path "$MODEL_PATH"
        --profile "$profile"
        --channel-cache "$CHANNEL_CACHE"
        --output-dir "$checkpoint"
        --retained-channels "$channels"
        --expected-protocol-name "$EXPECTED_PROTOCOL_NAME"
    )
    print_command "${command[@]}"
    "${command[@]}"
    printf '%s\n' "$checkpoint"
}

run_final6() {
    local ratio="$1"
    local gpu="$2"
    local port="$3"
    local directory checkpoint model_id server_log server_pid
    directory="$(ensure_experiment_dir "$ratio")"
    checkpoint="$directory/checkpoints/ENP"
    model_id="Gemma4-26B-A4B-${ratio}-ENP"
    [[ -f "$checkpoint/pruning_export_manifest.json" ]] || die "Checkpoint is not exported: $checkpoint"
    mkdir -p "$directory/server_logs"
    server_log="$directory/server_logs/ENP.log"
    local vllm_bin_dir
    vllm_bin_dir="$(dirname "$VLLM_PYTHON")"
    local -a server_command=(
        env
        -u LD_LIBRARY_PATH
        "PATH=$vllm_bin_dir:${PATH:-}"
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
    local healthy="false"
    for _ in $(seq 1 300); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            healthy="true"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -100 "$server_log" >&2
            die "vLLM server exited before becoming healthy: $server_log"
        fi
        sleep 2
    done
    [[ "$healthy" == "true" ]] || die "Timed out waiting for vLLM server: $server_log"
    env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null ||
        die "vLLM /v1/models check failed."
    PROTOCOL=full6_v1 \
    PYTHON_BIN="$PYTHON_BIN" \
    ARC_PATH="${ARC_PATH:-/data01/datasets/evalscope_benchmarks/arc}" \
    HELLASWAG_PATH="${HELLASWAG_PATH:-/data01/datasets/evalscope_benchmarks/hellaswag}" \
    WINOGRANDE_PATH="${WINOGRANDE_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip}" \
    GSM8K_PATH="${GSM8K_PATH:-/data01/datasets/evalscope_benchmarks/gsm8k}" \
    MATH_500_PATH="${MATH_500_PATH:-/data01/datasets/evalscope_benchmarks/math_500}" \
    MMLU_PATH="${MMLU_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu}" \
        bash "$ROOT/eval_protocol/run_vllm_protocol.sh" \
        "$model_id" \
        "http://127.0.0.1:$port" \
        ENP \
        "$directory"
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
    trap - EXIT INT TERM
}

dry_run() {
    echo "model=gemma4"
    echo "target_model=Gemma4-26B-A4B"
    echo "model_path=$MODEL_PATH"
    echo "collect_python=$VLLM_PYTHON"
    echo "build_gpu=$BUILD_GPU"
    echo "calibration=WikiText128x2048"
    echo "protocol=full6_v1"
    echo "method=ENP"
    echo "cache=$CALIBRATION_CACHE"
    echo "calibration_sha256=$EXPECTED_CACHE_SHA256"
    echo "calibration_protocol=$EXPECTED_PROTOCOL_NAME"
    echo "ENP-25 profile: $(profile_path 25); retained channels: $(retained_channels 25)"
    echo "ENP-50 profile: $(profile_path 50); retained channels: $(retained_channels 50)"
    echo "ENP-25 experiment: $(experiment_dir 25)"
    echo "ENP-50 experiment: $(experiment_dir 50)"
    echo "source_expert_width: 704"
    echo "channel_block_size: 64"
    echo "Dense: skipped"
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
