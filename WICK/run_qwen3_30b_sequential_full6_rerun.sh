#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
GPU_ID="${GPU_ID:-2}"
PORT="${PORT:-18430}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
FULL6_RUNNER="${FULL6_RUNNER:-$ROOT/WICK/run_vllm_full6.sh}"
QUEUE_ROOT="$RESULT_ROOT/Qwen330BA3BInstruct_sequential_full6_newtokens_${TIMESTAMP}_42"
START_FROM="${START_FROM:-Original}"
STARTED=false

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

wait_for_server() {
    local server_pid="$1"
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM exited before becoming healthy; see the current server log." >&2
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $PORT." >&2
    return 1
}

run_eval() {
    local label="$1"
    local model_id="$2"
    local model_dir="$3"
    local work_dir="$QUEUE_ROOT/$label"
    local server_log="$work_dir/server_logs/vllm.log"

    require_file "$model_dir/config.json"
    require_file "$model_dir/model.safetensors.index.json"
    mkdir -p "$work_dir/server_logs"

    echo "[$(date -Is)] START $label model=$model_dir" | tee -a "$QUEUE_ROOT/queue.log"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_dir" \
        --served-model-name "$model_id" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --dtype bfloat16 \
        --seed 42 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --generation-config vllm \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        >"$server_log" 2>&1 &
    local server_pid=$!

    cleanup_server() {
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server RETURN

    wait_for_server "$server_pid"
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$PORT" "$label" "$work_dir"

    cleanup_server
    trap - RETURN
    echo "[$(date -Is)] DONE $label" | tee -a "$QUEUE_ROOT/queue.log"
}

run_from() {
    local label="$1"
    shift

    if [[ "$label" == "$START_FROM" ]]; then
        STARTED=true
    fi
    if [[ "$STARTED" == "true" ]]; then
        run_eval "$label" "$@"
    fi
}

require_file "$FULL6_RUNNER"
mkdir -p "$QUEUE_ROOT"

run_from Original "Qwen330BA3BInstruct-Original-newtokens-$TIMESTAMP" "$MODEL_PATH"
run_from PPFrozenV1-B9 "Qwen330BA3BInstruct-PPFrozenV1-B9-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMER-PPFv1-G10-B9of12_202608071705_42/checkpoints/AIMER-PPFv1-G10-B9of12"
run_from PPFrozenV1-B6 "Qwen330BA3BInstruct-PPFrozenV1-B6-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMER-PPFv1-G10-B6of12_202608071705_42/checkpoints/AIMER-PPFv1-G10-B6of12"
run_from random-25 "Qwen330BA3BInstruct-random-25-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_25_vllm_CalibrationFree_full6_v1_random_202608082255_42/checkpoints/random"
run_from random-50 "Qwen330BA3BInstruct-random-50-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_50_vllm_CalibrationFree_full6_v1_random_202608082255_42/checkpoints/random"
run_from ENP-25 "Qwen330BA3BInstruct-ENP-25-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_25_vllm_WikiText128x2048_full6_v1_ENP_202608081610_42/checkpoints/ENP"
run_from ENP-50 "Qwen330BA3BInstruct-ENP-50-newtokens-$TIMESTAMP" \
    "$ROOT/result/Qwen330BA3BInstruct_50_vllm_WikiText128x2048_full6_v1_ENP_202608081610_42/checkpoints/ENP"

if [[ "$STARTED" != "true" ]]; then
    echo "Unknown START_FROM value: $START_FROM" >&2
    exit 2
fi

echo "[$(date -Is)] ALL DONE" | tee -a "$QUEUE_ROOT/queue.log"