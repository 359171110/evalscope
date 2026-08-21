#!/usr/bin/env bash

set -euo pipefail

MAGNITUDE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$MAGNITUDE_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RATIO="${RATIO:-50}"
SEED="${SEED:-42}"
MODEL="${1:-}"
ACTION="${2:-dry-run}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

validate_gpu() {
    local gpu="$1"
    local part
    IFS=',' read -r -a parts <<< "$gpu"
    [[ "${#parts[@]}" -ge 1 ]] || die "GPU list must not be empty."
    for part in "${parts[@]}"; do
        [[ "$part" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer; got '$part'."
    done
}

model_field() {
    local model="$1"
    local field="$2"
    case "$model:$field" in
        qwen3:name) printf 'Qwen330BA3BInstruct\n' ;;
        qwen3:path) printf '%s\n' "${QWEN3_MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}" ;;
        qwen3:gpu) printf '%s\n' "${QWEN3_GPU:-1}" ;;
        qwen3:port) printf '%s\n' "${QWEN3_PORT:-18280}" ;;
        qwen3:width)
            case "$RATIO" in
                50) printf '384\n' ;;
                25) printf '576\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen3." ;;
            esac
            ;;
        qwen3:vllm_extra) printf '\n' ;;
        gemma4:name) printf 'Gemma4-26B-A4B\n' ;;
        gemma4:path) printf '%s\n' "${GEMMA4_MODEL_PATH:-/data01/datasets/gemma-4-26B-A4B-it}" ;;
        gemma4:gpu) printf '%s\n' "${GEMMA4_GPU:-2}" ;;
        gemma4:port) printf '%s\n' "${GEMMA4_PORT:-18281}" ;;
        gemma4:width)
            case "$RATIO" in
                50) printf '352\n' ;;
                25) printf '512\n' ;;
                *) die "Unsupported RATIO=$RATIO for gemma4." ;;
            esac
            ;;
        gemma4:vllm_extra) printf '\n' ;;
        qwen36:name) printf 'Qwen3.6-35B-A3B\n' ;;
        qwen36:path) printf '%s\n' "${QWEN36_MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}" ;;
        qwen36:gpu) printf '%s\n' "${QWEN36_GPU:-4}" ;;
        qwen36:port) printf '%s\n' "${QWEN36_PORT:-18282}" ;;
        qwen36:width)
            case "$RATIO" in
                50) printf '256\n' ;;
                25) printf '384\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen36." ;;
            esac
            ;;
        qwen36:vllm_extra) printf '\n' ;;
        deepseek:name) printf 'DeepSeek-V2-Lite-Chat\n' ;;
        deepseek:path) printf '%s\n' "${DEEPSEEK_MODEL_PATH:-/data01/datasets/DeepSeek-V2-Lite-Chat}" ;;
        deepseek:gpu) printf '%s\n' "${DEEPSEEK_GPU:-3}" ;;
        deepseek:port) printf '%s\n' "${DEEPSEEK_PORT:-18283}" ;;
        deepseek:width)
            case "$RATIO" in
                50) printf '704\n' ;;
                25) printf '1056\n' ;;
                *) die "Unsupported RATIO=$RATIO for deepseek." ;;
            esac
            ;;
        deepseek:vllm_extra) printf '%s\n' "--trust-remote-code" ;;
        *:artifact) printf '%s\n' "$MAGNITUDE_ROOT/experiments/profiles/${model}_magnitude" ;;
        *) die "Unknown model/field '$model:$field'." ;;
    esac
}

experiment_dir() {
    local model="$1"
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" MODEL_NAME="$(model_field "$model" name)" \
        "$CODE_ROOT/scripts/create_result_dir.sh" \
        --model "$(model_field "$model" name)" \
        --inference vllm \
        --calibration CalibrationFree \
        --protocol full8_v1 \
        --method Magnitude \
        --pruning-ratio-label "$RATIO" \
        --pruning-ratio-percent "$RATIO" \
        --timestamp "$TIMESTAMP" \
        --dry-run
}

tensor_parallel_size() {
    local gpu="$1"
    local count
    IFS=',' read -r -a parts <<< "$gpu"
    count="${#parts[@]}"
    printf '%s\n' "$count"
}

ensure_experiment_dir() {
    local model="$1"
    local directory
    directory="$(experiment_dir "$model")"
    if [[ ! -f "$directory/experiment_manifest.json" ]]; then
        RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" \
            "$CODE_ROOT/scripts/create_result_dir.sh" \
            --model "$(model_field "$model" name)" \
            --inference vllm \
            --calibration CalibrationFree \
            --protocol full8_v1 \
            --method Magnitude \
            --pruning-ratio-label "$RATIO" \
            --pruning-ratio-percent "$RATIO" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$directory"
}

build_artifacts() {
    local model="$1"
    local artifact profile
    artifact="$(model_field "$model" artifact)"
    profile="$artifact/magnitude_${RATIO}pct_per_layer.pt"
    [[ -d "$(model_field "$model" path)" ]] || die "Model path does not exist: $(model_field "$model" path)"
    mkdir -p "$artifact"
    if [[ -f "$profile" && -f "$artifact/magnitude_rankings.pt" ]]; then
        echo "artifacts already exist: $profile"
        return
    fi
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        -m Magnitude.build_magnitude_artifacts
        --model-path "$(model_field "$model" path)"
        --output-channel-cache "$artifact/magnitude_rankings.pt"
        --output-profile "$profile"
        --retained-channels "$(model_field "$model" width)"
    )
    print_command "${command[@]}"
    "${command[@]}"
}

export_checkpoint() {
    local model="$1"
    local artifact directory checkpoint
    artifact="$(model_field "$model" artifact)"
    directory="$(ensure_experiment_dir "$model")"
    checkpoint="$directory/checkpoints/Magnitude"
    [[ -f "$artifact/magnitude_${RATIO}pct_per_layer.pt" ]] || die "Magnitude profile does not exist: $artifact/magnitude_${RATIO}pct_per_layer.pt"
    [[ -f "$artifact/magnitude_rankings.pt" ]] || die "Magnitude channel cache does not exist: $artifact/magnitude_rankings.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        printf '%s\n' "$checkpoint"
        return
    fi
    mkdir -p "$checkpoint"
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        -m Magnitude.export_magnitude_checkpoint
        --model-path "$(model_field "$model" path)"
        --profile "$artifact/magnitude_${RATIO}pct_per_layer.pt"
        --channel-cache "$artifact/magnitude_rankings.pt"
        --output-dir "$checkpoint"
    )
    print_command "${command[@]}"
    "${command[@]}"
    printf '%s\n' "$checkpoint"
}

run_eval() {
    local model="$1"
    local directory checkpoint model_id gpu port server_log server_pid extra tp
    directory="$(ensure_experiment_dir "$model")"
    checkpoint="$directory/checkpoints/Magnitude"
    model_id="$(model_field "$model" name)-${RATIO}-Magnitude"
    gpu="$(model_field "$model" gpu)"
    port="$(model_field "$model" port)"
    extra="$(model_field "$model" vllm_extra)"
    tp="$(tensor_parallel_size "$gpu")"
    server_log="$directory/server_logs/Magnitude.log"
    [[ -f "$checkpoint/pruning_export_manifest.json" ]] || die "Checkpoint is not exported: $checkpoint"
    validate_gpu "$gpu"
    mkdir -p "$directory/server_logs"
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
        --tensor-parallel-size "$tp"
    )
    if [[ -n "$extra" ]]; then
        # shellcheck disable=SC2206
        server_command+=($extra)
    fi
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
    env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null || die "vLLM /v1/models check failed."
    PROTOCOL=full8_v1 \
    PYTHON_BIN="$PYTHON_BIN" \
    ARC_PATH="${ARC_PATH:-/data01/datasets/evalscope_benchmarks/arc}" \
    HELLASWAG_PATH="${HELLASWAG_PATH:-/data01/datasets/evalscope_benchmarks/hellaswag}" \
    WINOGRANDE_PATH="${WINOGRANDE_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip}" \
    GSM8K_PATH="${GSM8K_PATH:-/data01/datasets/evalscope_benchmarks/gsm8k}" \
    MATH_500_PATH="${MATH_500_PATH:-/data01/datasets/evalscope_benchmarks/math_500}" \
    MMLU_PATH="${MMLU_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu}" \
    HUMANEVAL_PATH="${HUMANEVAL_PATH:-/data01/datasets/evalscope_benchmarks/humaneval}" \
    MBPP_PATH="${MBPP_PATH:-/data01/datasets/evalscope_benchmarks/mbpp}" \
        bash "$ROOT/eval_protocol/run_vllm_protocol.sh" \
        "$model_id" \
        "http://127.0.0.1:$port" \
        Magnitude \
        "$directory"
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
    trap - EXIT INT TERM
}

dry_run_one() {
    local model="$1"
    echo "model=$model"
    echo "target_model=$(model_field "$model" name)"
    echo "model_path=$(model_field "$model" path)"
    echo "gpu=$(model_field "$model" gpu)"
    echo "port=$(model_field "$model" port)"
    echo "ratio=$RATIO"
    echo "seed=$SEED"
    echo "retained_channels=$(model_field "$model" width)"
    echo "calibration=CalibrationFree"
    echo "protocol=full8_v1"
    echo "method=Magnitude"
    echo "artifact_root=$(model_field "$model" artifact)"
    echo "experiment=$(experiment_dir "$model")"
}

run_one() {
    local model="$1"
    local action="$2"
    case "$action" in
        dry-run) dry_run_one "$model" ;;
        build) build_artifacts "$model" ;;
        export) export_checkpoint "$model" ;;
        eval) run_eval "$model" ;;
        prepare)
            build_artifacts "$model"
            export_checkpoint "$model"
            ;;
        *) die "Usage: $0 qwen3|gemma4|qwen36|deepseek|all dry-run|build|export|eval|prepare" ;;
    esac
}

[[ -n "$MODEL" ]] || die "Usage: RATIO=25|50 $0 qwen3|gemma4|qwen36|deepseek|all dry-run|build|export|eval|prepare"
[[ "$RATIO" == "25" || "$RATIO" == "50" ]] || die "RATIO must be 25 or 50; got '$RATIO'."
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer; got '$SEED'."
[[ -x "$CODE_ROOT/scripts/create_result_dir.sh" ]] || die "Framework result helper is missing."

if [[ "$MODEL" == all ]]; then
    for item in qwen3 gemma4 qwen36 deepseek; do
        run_one "$item" "$ACTION"
    done
    exit 0
fi

[[ "$MODEL" == qwen3 || "$MODEL" == gemma4 || "$MODEL" == qwen36 || "$MODEL" == deepseek ]] ||
    die "Model must be qwen3, gemma4, qwen36, deepseek, or all."
run_one "$MODEL" "$ACTION"
