#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
GPU_25="${GPU_25:-1}"
GPU_50="${GPU_50:-2}"
PORT_25="${PORT_25:-18325}"
PORT_50="${PORT_50:-18350}"

RANDOM_BUILDER="$ROOT/WICK/build_random_profiles.py"
EXPORTER="$ROOT/WICK/export_uniform_qwen3_moe.py"
CREATE_RESULT_DIR="$ROOT/static_moe_prunning/code/scripts/create_result_dir.sh"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6.sh"
PSEUDO_CACHE="$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4/pure_pseudo_rankings.pt"
ARTIFACT_ROOT="$ROOT/WICK/experiments/profiles/qwen3_random_full6_${TIMESTAMP}"

export PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code${PYTHONPATH:+:$PYTHONPATH}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

wait_for_server() {
    local server_pid="$1"
    local port="$2"
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            die "vLLM server exited before becoming healthy."
        fi
        sleep 1
    done
    die "Timed out waiting for vLLM server on port $port."
}

ensure_experiment_dir() {
    local pruning_ratio="$1"
    local method="$2"
    local experiment_dir
    experiment_dir=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
        --inference vllm \
        --calibration CalibrationFree \
        --protocol full6_v1 \
        --method "$method" \
        --pruning-ratio "$pruning_ratio" \
        --timestamp "$TIMESTAMP" \
        --dry-run)
    if [[ ! -d "$experiment_dir" ]]; then
        RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
            --inference vllm \
            --calibration CalibrationFree \
            --protocol full6_v1 \
            --method "$method" \
            --pruning-ratio "$pruning_ratio" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$experiment_dir"
}

run_variant() {
    local pruning_ratio="$1"
    local gpu="$2"
    local port="$3"
    local retained_channels
    local artifact_dir
    local channel_cache
    local experiment_dir
    local checkpoint_dir

    retained_channels=$((768 * (100 - pruning_ratio) / 100))
    artifact_dir="$ARTIFACT_ROOT/random_${pruning_ratio}pct"
    channel_cache="$artifact_dir/random_rankings.pt"
    experiment_dir=$(ensure_experiment_dir "$pruning_ratio" random)
    checkpoint_dir="$experiment_dir/checkpoints/random"

    if [[ ! -f "$channel_cache" ]]; then
        mkdir -p "$artifact_dir"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$RANDOM_BUILDER" \
            --model-path "$MODEL_PATH" \
            --pseudo-ranking-cache "$PSEUDO_CACHE" \
            --output-dir "$artifact_dir" \
            --target-pruning-ratio "0.$(printf '%02d' "$pruning_ratio")" \
            --protection-ratio 0 \
            --channel-block-size 64 \
            --seed 42
        mv "$artifact_dir/random_50pct_per_expert.pt" "$artifact_dir/random_profile.pt"
    fi

    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        mkdir -p "$checkpoint_dir"
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$channel_cache" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$retained_channels"
    fi

    local server_log="$experiment_dir/server_logs/random.log"
    local server_command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
        --model "$checkpoint_dir"
        --served-model-name "Qwen330BA3BInstruct-${retained_channels}ch-random"
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
    mkdir -p "$experiment_dir/server_logs"
    "${server_command[@]}" >"$server_log" 2>&1 &
    local server_pid=$!
    cleanup_variant() {
        kill -TERM "$server_pid" 2>/dev/null || true
    }
    trap cleanup_variant RETURN
    wait_for_server "$server_pid" "$port"

    CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD=random \
        MODEL_ID="Qwen330BA3BInstruct-${retained_channels}ch-random" \
        GPU_ID="$gpu" \
        PORT="$port" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$FULL6_RUNNER" \
            "Qwen330BA3BInstruct-${retained_channels}ch-random" \
            "http://127.0.0.1:$port" \
            random \
            "$experiment_dir"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PSEUDO_CACHE"
require_file "$RANDOM_BUILDER"
require_file "$EXPORTER"
require_file "$CREATE_RESULT_DIR"
require_file "$FULL6_RUNNER"

case "${1:-all}" in
    25)
        run_variant 25 "$GPU_25" "$PORT_25"
        ;;
    50)
        run_variant 50 "$GPU_50" "$PORT_50"
        ;;
    all)
        run_variant 25 "$GPU_25" "$PORT_25" &
        pid_25=$!
        run_variant 50 "$GPU_50" "$PORT_50" &
        pid_50=$!
        wait "$pid_25"
        wait "$pid_50"
        ;;
    *)
        die "Usage: $0 [25|50|all]"
        ;;
esac