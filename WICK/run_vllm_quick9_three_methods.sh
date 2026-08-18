#!/usr/bin/env bash

set -euo pipefail

ROOT=/data01/home/xinpei.gao/evalscope
VLLM_PYTHON=${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/WICK/experiments/results/qwen3_wick_vllm_quick9_20260806}
LOG_ROOT="$RESULTS_ROOT/server_logs"
RUNNER="$ROOT/WICK/run_vllm_quick9.sh"

mkdir -p "$LOG_ROOT"

declare -A MODEL_IDS=(
    [random]=qwen3-random-50pct-seed42
    [random_wick_protect]=qwen3-random-wick-protect-50pct-seed42
    [aimer_wick_protect]=qwen3-aimer-wick-protect-50pct
)
declare -A MODEL_PATHS=(
    [random]="$ROOT/WICK/experiments/exported_models/qwen3_random_50pct_seed42"
    [random_wick_protect]="$ROOT/WICK/experiments/exported_models/qwen3_random_wick_protect_50pct_seed42"
    [aimer_wick_protect]="$ROOT/WICK/experiments/exported_models/qwen3_aimer_wick_protect_50pct"
)

wait_for_server() {
    local server_pid=$1
    local port=$2
    local log_path=$3
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM server exited before becoming healthy: $log_path" >&2
            tail -100 "$log_path" >&2
            exit 1
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM server: $log_path" >&2
    exit 1
}

run_method() {
    local method=$1
    local gpu_id=$2
    local port=$3
    local model_id=${MODEL_IDS[$method]}
    local model_path=${MODEL_PATHS[$method]}
    local log_path="$LOG_ROOT/$method.log"

    [[ -f "$model_path/config.json" ]] || {
        echo "Missing compact checkpoint: $model_path" >&2
        exit 2
    }

    echo "Starting method=$method model=$model_id gpu=$gpu_id port=$port"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$model_id" \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype bfloat16 \
        --seed 42 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --generation-config vllm \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        >"$log_path" 2>&1 &
    local server_pid=$!
    trap 'kill -TERM "$server_pid" 2>/dev/null || true' RETURN
    wait_for_server "$server_pid" "$port" "$log_path"

    bash "$RUNNER" "$model_id" "http://127.0.0.1:$port" "$method" "$RESULTS_ROOT"
    kill -TERM "$server_pid"
    wait "$server_pid" || true
    trap - RETURN
    echo "Completed method=$method"
}

run_gpu2() {
    run_method random 2 18052
}

run_gpu5() {
    run_method random_wick_protect 5 18055
    run_method aimer_wick_protect 5 18055
}

run_gpu2 &
gpu2_worker=$!
run_gpu5 &
gpu5_worker=$!

cleanup_workers() {
    kill -TERM "$gpu2_worker" "$gpu5_worker" 2>/dev/null || true
}

trap cleanup_workers EXIT INT TERM
wait "$gpu2_worker"
wait "$gpu5_worker"
trap - EXIT INT TERM

echo "Completed all methods. Results: $RESULTS_ROOT"