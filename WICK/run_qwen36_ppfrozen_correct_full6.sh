#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
GPU_B9="${GPU_B9:-0}"
GPU_B6="${GPU_B6:-1}"
PORT_B9="${PORT_B9:-18525}"
PORT_B6="${PORT_B6:-18526}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result/Qwen36_PPFrozenV1_correct_full6_${TIMESTAMP}_42}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/WICK/experiments/profiles/qwen36_ppfrozen_correct_${TIMESTAMP}}"
AIMER_ROOT="${AIMER_ROOT:-$ROOT/WICK/experiments/profiles/qwen36_gpu5_channel_pp_202608091945}"
PP_CACHE="${PP_CACHE:-$ROOT/PP/experiments/profiles/qwen36_35b_a3b_pp_frozen_v1_20260808/rankings.pt}"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"

export PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }

wait_for_server() {
    local pid="$1" port="$2"
    for _ in $(seq 1 240); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        kill -0 "$pid" 2>/dev/null || { echo "vLLM exited on port $port" >&2; return 1; }
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $port" >&2
    return 1
}

build_stage() {
    local label="$1" ratio="$2" retained="$3"
    local source="$AIMER_ROOT/aimer_${ratio}pct/aimer_channel_rankings.pt"
    local output="$ARTIFACT_ROOT/$label"
    require_file "$source"
    mkdir -p "$output"
    if [[ ! -f "$output/checkpoint/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$ROOT/PP/build_protected_rankings.py" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$source" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$output/profile.pt" \
            --output-channel-cache "$output/rankings.pt" \
            --method aimer_pp \
            --backbone aimer \
            --retained-blocks "$((retained / 64))" \
            --protection-ratio 0.10
    fi
    if [[ ! -f "$output/checkpoint/pruning_export_manifest.json" ]]; then
        mkdir -p "$output/checkpoint"
        "$PYTHON_BIN" "$ROOT/PP/export_uniform_moe.py" \
            --model-path "$MODEL_PATH" \
            --profile "$output/profile.pt" \
            --channel-cache "$output/rankings.pt" \
            --output-dir "$output/checkpoint" \
            --retained-channels "$retained"
    fi
}

run_stage() {
    local label="$1" gpu="$2" port="$3"
    local model_id="Qwen3.6-35B-A3B-PPFrozenV1-${label}-correct-${TIMESTAMP}"
    local checkpoint="$ARTIFACT_ROOT/$label/checkpoint"
    local work_root="$RESULT_ROOT/$label"
    local server_log="$work_root/server_logs/vllm.log"
    mkdir -p "$work_root/server_logs"
    echo "[$(date -Is)] START $label GPU=$gpu PORT=$port" | tee -a "$RESULT_ROOT/queue.log"
    env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$checkpoint" \
        --served-model-name "$model_id" \
        --host 127.0.0.1 \
        --port "$port" \
        --dtype bfloat16 \
        --seed 42 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --generation-config vllm \
        --default-chat-template-kwargs '{"enable_thinking":false}' >"$server_log" 2>&1 &
    local server_pid=$!
    trap 'kill -TERM "$server_pid" 2>/dev/null || true' RETURN
    wait_for_server "$server_pid" "$port"
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$port" "$label" "$work_root"
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    trap - RETURN
    echo "[$(date -Is)] DONE $label" | tee -a "$RESULT_ROOT/queue.log"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_CACHE"
require_file "$FULL6_RUNNER"
mkdir -p "$ARTIFACT_ROOT" "$RESULT_ROOT"

build_stage B9 25 384
build_stage B6 50 256

run_stage B9 "$GPU_B9" "$PORT_B9" &
pid_b9=$!
run_stage B6 "$GPU_B6" "$PORT_B6" &
pid_b6=$!

status=0
wait "$pid_b9" || status=$?
wait "$pid_b6" || status=$?
exit "$status"