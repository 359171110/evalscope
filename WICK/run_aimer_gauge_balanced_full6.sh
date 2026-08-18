#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3}"
MODEL_PATH="${MODEL_PATH:-}"
GPU_ID="${GPU_ID:-3}"
PORT_BASE="${PORT_BASE:-18625}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"
PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/WICK/experiments/profiles/${MODEL_FAMILY}_aimer_gauge_balanced_${TIMESTAMP}}"
LOG_ROOT="$RESULT_ROOT/${MODEL_FAMILY}_aimer_gauge_balanced_full6_${TIMESTAMP}_42_logs"

case "$MODEL_FAMILY" in
    qwen3)
        MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
        MODEL_NAME="Qwen330BA3BInstruct"
        RETAINED_25=576
        RETAINED_50=384
        ;;
    qwen36)
        MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
        MODEL_NAME="Qwen36_35B_A3B"
        RETAINED_25=384
        RETAINED_50=256
        ;;
    *)
        echo "MODEL_FAMILY must be qwen3 or qwen36" >&2
        exit 2
        ;;
esac

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }

wait_for_server() {
    local pid="$1" port="$2"
    for _ in $(seq 1 180); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then return 0; fi
        kill -0 "$pid" 2>/dev/null || { echo "vLLM exited on port $port" >&2; return 1; }
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $port" >&2
    return 1
}

build_artifacts() {
    local ratio="$1" retained="$2"
    local out="$ARTIFACT_ROOT/aimer_${ratio}pct"
    if [[ -f "$out/checkpoint/pruning_export_manifest.json" ]]; then return; fi
    mkdir -p "$out"
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/static_moe_prunning/code/scripts/build_aimer_channel_profile.py" \
        --model-path "$MODEL_PATH" --aimer-root "$ROOT/static_moe_prunning" \
        --output-profile "$out/profile.pt" --output-channel-cache "$out/rankings.pt" \
        --target-pruning-ratio "0.$(printf '%02d' "$ratio")" --channel-block-size 64 \
        --score-variant gauge_balanced
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/PP/build_protected_rankings.py" \
        --model-path "$MODEL_PATH" --backbone-cache "$out/rankings.pt" --pseudo-cache "$out/rankings.pt" \
        --output-profile "$out/fixed_profile.pt" --output-channel-cache "$out/fixed_rankings.pt" \
        --method aimer_gauge_balanced --backbone gauge_balanced_aimer \
        --retained-blocks "$((retained / 64))" --protection-ratio 0
    if [[ "$MODEL_FAMILY" == "qwen3" ]]; then
        PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/WICK/export_uniform_qwen3_moe.py" \
            --model-path "$MODEL_PATH" --channel-cache "$out/fixed_rankings.pt" \
            --output-dir "$out/checkpoint" --retained-channels "$retained"
    else
        PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/PP/export_uniform_moe.py" \
            --model-path "$MODEL_PATH" --profile "$out/fixed_profile.pt" --channel-cache "$out/fixed_rankings.pt" \
            --output-dir "$out/checkpoint" --retained-channels "$retained"
    fi
}

run_eval() {
    local ratio="$1" retained="$2" port="$3"
    local label="AIMER-GaugeBalanced-${ratio}"
    local experiment="${MODEL_NAME}_${ratio}_vllm_CalibrationFree_full6_v1_${label}_${TIMESTAMP}_42"
    local experiment_dir="$RESULT_ROOT/$experiment"
    local model_id="${MODEL_NAME}-${label}-full6-${TIMESTAMP}"
    mkdir -p "$experiment_dir/server_logs"
    echo "[$(date -Is)] START $label gpu=$GPU_ID" | tee -a "$LOG_ROOT/queue.log"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$ARTIFACT_ROOT/aimer_${ratio}pct/checkpoint" --served-model-name "$model_id" \
        --host 127.0.0.1 --port "$port" --dtype bfloat16 --seed 42 --max-model-len 8192 \
        --max-num-seqs 16 --gpu-memory-utilization 0.90 --generation-config vllm \
        --default-chat-template-kwargs '{"enable_thinking":false}' \
        >"$experiment_dir/server_logs/$label.log" 2>&1 &
    local pid=$!
    trap "kill -TERM '$pid' 2>/dev/null || true; wait '$pid' 2>/dev/null || true" RETURN
    wait_for_server "$pid" "$port"
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$port" "$label" "$experiment_dir"
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - RETURN
    echo "[$(date -Is)] DONE $label" | tee -a "$LOG_ROOT/queue.log"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$FULL6_RUNNER"
mkdir -p "$LOG_ROOT"
build_artifacts 25 "$RETAINED_25"
build_artifacts 50 "$RETAINED_50"
run_eval 25 "$RETAINED_25" "$PORT_BASE"
run_eval 50 "$RETAINED_50" "$((PORT_BASE + 1))"
