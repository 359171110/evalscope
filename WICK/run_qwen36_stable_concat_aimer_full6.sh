#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
GPU_B9="${GPU_B9:-3}"
GPU_B6="${GPU_B6:-5}"
PORT_B9="${PORT_B9:-18825}"
PORT_B6="${PORT_B6:-18826}"
THRESHOLD="${THRESHOLD:-1e-12}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"
PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/WICK/experiments/profiles/qwen36_stable_concat_aimer_${TIMESTAMP}}"
BASE_ROOT="$ARTIFACT_ROOT/base"

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

run_python() {
    env -u LD_LIBRARY_PATH PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$@"
}

build_artifacts() {
    mkdir -p "$BASE_ROOT"
    if [[ ! -f "$BASE_ROOT/rankings.pt" ]]; then
        run_python "$ROOT/static_moe_prunning/code/scripts/build_aimer_channel_profile.py" \
            --model-path "$MODEL_PATH" --aimer-root "$ROOT/static_moe_prunning" \
            --output-profile "$BASE_ROOT/profile.pt" --output-channel-cache "$BASE_ROOT/rankings.pt" \
            --target-pruning-ratio 0.5 --channel-block-size 64 \
            --score-variant stable_concat --effective-zero-threshold "$THRESHOLD"
    fi
    build_ratio 25 384
    build_ratio 50 256
}

build_ratio() {
    local ratio="$1" retained="$2"
    local out="$ARTIFACT_ROOT/stable_concat_${ratio}pct"
    if [[ -f "$out/checkpoint/pruning_export_manifest.json" ]]; then return; fi
    mkdir -p "$out"
    if [[ ! -f "$out/fixed_profile.pt" || ! -f "$out/fixed_rankings.pt" ]]; then
        run_python "$ROOT/PP/build_protected_rankings.py" \
            --model-path "$MODEL_PATH" --backbone-cache "$BASE_ROOT/rankings.pt" \
            --pseudo-cache "$BASE_ROOT/rankings.pt" --output-profile "$out/fixed_profile.pt" \
            --output-channel-cache "$out/fixed_rankings.pt" --method stable_concat_aimer \
            --backbone stable_concat_aimer --retained-blocks "$((retained / 64))" --protection-ratio 0
    fi
    run_python "$ROOT/PP/export_uniform_moe.py" \
        --model-path "$MODEL_PATH" --profile "$out/fixed_profile.pt" \
        --channel-cache "$out/fixed_rankings.pt" --output-dir "$out/checkpoint" \
        --retained-channels "$retained"
}

run_eval() {
    local ratio="$1" gpu="$2" port="$3"
    local label="Stable-Concat-AIMER-${ratio}"
    local experiment="Qwen36_35B_A3B_${ratio}_vllm_CalibrationFree_full6_v1_${label}_${TIMESTAMP}_42"
    local experiment_dir="$RESULT_ROOT/$experiment"
    local model_id="Qwen36_35B_A3B-${label}-full6-${TIMESTAMP}"
    local checkpoint="$ARTIFACT_ROOT/stable_concat_${ratio}pct/checkpoint"
    require_file "$checkpoint/pruning_export_manifest.json"
    mkdir -p "$experiment_dir/server_logs"
    env CUDA_VISIBLE_DEVICES="$gpu" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$checkpoint" --served-model-name "$model_id" --host 127.0.0.1 --port "$port" \
        --dtype bfloat16 --seed 42 --max-model-len 8192 --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 --generation-config vllm \
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
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$FULL6_RUNNER"

case "$STAGE" in
    build) build_artifacts ;;
    eval-b9) run_eval 25 "$GPU_B9" "$PORT_B9" ;;
    eval-b6) run_eval 50 "$GPU_B6" "$PORT_B6" ;;
    all)
        build_artifacts
        run_eval 25 "$GPU_B9" "$PORT_B9" &
        pid_b9=$!
        run_eval 50 "$GPU_B6" "$PORT_B6" &
        pid_b6=$!
        wait "$pid_b9"
        wait "$pid_b6"
        ;;
    *)
        echo "Usage: $0 {build|eval-b9|eval-b6|all}" >&2
        exit 2
        ;;
esac