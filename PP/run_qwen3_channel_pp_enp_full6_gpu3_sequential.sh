#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
GPU_ID="${GPU_ID:-3}"
PORT="${PORT:-18432}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
CREATE_RESULT_DIR="$ROOT/static_moe_prunning/code/scripts/create_result_dir.sh"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"
SOURCE_TIMESTAMP="${SOURCE_TIMESTAMP:-202608091953}"
LOG_ROOT="$RESULT_ROOT/Qwen330BA3BInstruct_channel_pp_enp_full6_gpu3_${TIMESTAMP}_42_logs"

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

wait_for_server() {
    local server_pid=$1
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
            curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            return 1
        fi
        sleep 5
    done
    return 1
}

create_experiment() {
    local pruning=$1
    local calibration=$2
    local method=$3
    local experiment
    experiment=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
        --inference vllm \
        --calibration "$calibration" \
        --protocol full6_v1 \
        --method "$method" \
        --pruning-ratio-label "$pruning" \
        --pruning-ratio-percent "$pruning" \
        --timestamp "$TIMESTAMP" \
        --dry-run)
    if [[ ! -d "$experiment" ]]; then
        RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
            --inference vllm \
            --calibration "$calibration" \
            --protocol full6_v1 \
            --method "$method" \
            --pruning-ratio-label "$pruning" \
            --pruning-ratio-percent "$pruning" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$experiment"
}

run_one() {
    local method=$1
    local pruning=$2
    local calibration=$3
    local source_experiment=$4
    local checkpoint_dir="$source_experiment/checkpoints/$method"
    local experiment model_id server_log server_pid

    require_file "$checkpoint_dir/config.json"
    require_file "$checkpoint_dir/model.safetensors.index.json"
    experiment=$(create_experiment "$pruning" "$calibration" "$method")
    model_id="Qwen330BA3BInstruct-$method-$pruning-full6-$TIMESTAMP"
    server_log="$experiment/server_logs/$method.log"

    echo "[$(date -Is)] START $method pruning=$pruning gpu=$GPU_ID" | tee -a "$LOG_ROOT/queue.log"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$checkpoint_dir" \
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
    server_pid=$!
    cleanup_server() {
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server RETURN

    wait_for_server "$server_pid" || {
        tail -100 "$server_log" >&2
        return 1
    }
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$PORT" "$method" "$experiment"
    cleanup_server
    trap - RETURN
    echo "[$(date -Is)] DONE $method pruning=$pruning" | tee -a "$LOG_ROOT/queue.log"
}

mkdir -p "$LOG_ROOT"
require_file "$CREATE_RESULT_DIR"
require_file "$FULL6_RUNNER"

echo "[$(date -Is)] QUEUE START gpu=$GPU_ID port=$PORT timestamp=$TIMESTAMP" | tee "$LOG_ROOT/queue.log"
run_one AIMER 25 CalibrationFree \
    "$RESULT_ROOT/Qwen330BA3BInstruct_25_vllm_CalibrationFree_quick9_AIMER_${SOURCE_TIMESTAMP}_42"
run_one AIMER 50 CalibrationFree \
    "$RESULT_ROOT/Qwen330BA3BInstruct_50_vllm_CalibrationFree_quick9_AIMER_${SOURCE_TIMESTAMP}_42"
run_one Random-PP-G10 25 CalibrationFree \
    "$RESULT_ROOT/Qwen330BA3BInstruct_25_vllm_CalibrationFree_quick9_Random-PP-G10_${SOURCE_TIMESTAMP}_42"
run_one Random-PP-G10 50 CalibrationFree \
    "$RESULT_ROOT/Qwen330BA3BInstruct_50_vllm_CalibrationFree_quick9_Random-PP-G10_${SOURCE_TIMESTAMP}_42"
run_one ENP-PP-G10 25 WikiText128x2048 \
    "$RESULT_ROOT/Qwen330BA3BInstruct_25_vllm_WikiText128x2048_quick9_ENP-PP-G10_${SOURCE_TIMESTAMP}_42"
run_one ENP-PP-G10 50 WikiText128x2048 \
    "$RESULT_ROOT/Qwen330BA3BInstruct_50_vllm_WikiText128x2048_quick9_ENP-PP-G10_${SOURCE_TIMESTAMP}_42"
echo "[$(date -Is)] QUEUE DONE" | tee -a "$LOG_ROOT/queue.log"