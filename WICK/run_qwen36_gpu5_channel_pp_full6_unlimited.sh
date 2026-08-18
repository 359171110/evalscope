#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
GPU_ID="${GPU_ID:-5}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
PORT_BASE="${PORT_BASE:-18511}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"
ARTIFACT_ROOT="$ROOT/WICK/experiments/profiles/qwen36_gpu5_channel_pp_202608091945"
QUEUE_LOG="$ROOT/WICK/experiments/qwen36_gpu5_full6_unlimited_${TIMESTAMP}.log"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }

wait_for_server() {
    local pid="$1" port="$2"
    for _ in $(seq 1 180); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            return
        fi
        kill -0 "$pid" 2>/dev/null || { echo "vLLM exited on port $port" >&2; return 1; }
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $port" >&2
    return 1
}

run_eval() {
    local label="$1" pruning="$2" calibration="$3" model_dir="$4" port="$5"
    local experiment="Qwen36_35B_A3B_${pruning}_vllm_${calibration}_full6_v1_${label}_${TIMESTAMP}_42"
    local experiment_dir="$RESULT_ROOT/$experiment"
    local server_log="$experiment_dir/server_logs/$label.log"
    local model_id="Qwen3.6-35B-A3B-${label}-full6-${TIMESTAMP}"
    mkdir -p "$experiment_dir/server_logs"
    echo "[$(date -Is)] START $label $experiment_dir" | tee -a "$QUEUE_LOG"
    env "CUDA_VISIBLE_DEVICES=$GPU_ID" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_dir" --served-model-name "$model_id" --host 127.0.0.1 --port "$port" \
        --dtype bfloat16 --seed 42 --max-model-len 8192 --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 --generation-config vllm \
        --default-chat-template-kwargs '{"enable_thinking":false}' >"$server_log" 2>&1 &
    local pid=$!
    trap 'kill -TERM "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' RETURN
    wait_for_server "$pid" "$port"
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$port" "$label" "$experiment_dir"
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - RETURN
    echo "[$(date -Is)] DONE $label" | tee -a "$QUEUE_LOG"
}

require_file "$FULL6_RUNNER"
require_file "$ARTIFACT_ROOT/aimer_25pct/checkpoint/pruning_export_manifest.json"
require_file "$ARTIFACT_ROOT/aimer_50pct/checkpoint/pruning_export_manifest.json"
require_file "$ARTIFACT_ROOT/random_pp_25pct/checkpoint/pruning_export_manifest.json"
require_file "$ARTIFACT_ROOT/random_pp_50pct/checkpoint/pruning_export_manifest.json"
require_file "$ARTIFACT_ROOT/enp_pp_25pct/checkpoint/pruning_export_manifest.json"
require_file "$ARTIFACT_ROOT/enp_pp_50pct/checkpoint/pruning_export_manifest.json"
mkdir -p "$(dirname "$QUEUE_LOG")"

run_eval AIMER-channel-25 25 CalibrationFree "$ARTIFACT_ROOT/aimer_25pct/checkpoint" "$PORT_BASE"
run_eval AIMER-channel-50 50 CalibrationFree "$ARTIFACT_ROOT/aimer_50pct/checkpoint" "$((PORT_BASE + 1))"
run_eval Random-PP-25 25 CalibrationFree "$ARTIFACT_ROOT/random_pp_25pct/checkpoint" "$((PORT_BASE + 2))"
run_eval Random-PP-50 50 CalibrationFree "$ARTIFACT_ROOT/random_pp_50pct/checkpoint" "$((PORT_BASE + 3))"
run_eval ENP-PP-25 25 WikiText128x2048 "$ARTIFACT_ROOT/enp_pp_25pct/checkpoint" "$((PORT_BASE + 4))"
run_eval ENP-PP-50 50 WikiText128x2048 "$ARTIFACT_ROOT/enp_pp_50pct/checkpoint" "$((PORT_BASE + 5))"