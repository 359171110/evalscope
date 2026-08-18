#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
ENP_PYTHON_BIN="${ENP_PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xh2/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
GPU_ID="${GPU_ID:-5}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PORT_BASE="${PORT_BASE:-18501}"
PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code"
export LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
ARTIFACT_ROOT="$ROOT/WICK/experiments/profiles/qwen36_gpu5_channel_pp_${TIMESTAMP}"
ENP_ROOT="$ARTIFACT_ROOT/enp_calibration"
LOG_ROOT="$RESULT_ROOT/Qwen36_gpu5_channel_pp_full6_${TIMESTAMP}_42"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6.sh"
PP_CACHE="${PP_CACHE:-$ROOT/PP/experiments/profiles/qwen36_35b_a3b_pp_frozen_v1_20260808/rankings.pt}"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }

wait_for_server() {
    local pid="$1" port="$2"
    for _ in $(seq 1 180); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then return; fi
        kill -0 "$pid" 2>/dev/null || { echo "vLLM exited on port $port" >&2; return 1; }
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $port" >&2
    return 1
}

run_eval() {
    local label="$1" model_id="$2" model_dir="$3" port="$4"
    local work_dir="$LOG_ROOT/$label"
    mkdir -p "$work_dir/server_logs"
    local log="$work_dir/server_logs/vllm.log"
    local -a cmd=(env "CUDA_VISIBLE_DEVICES=$GPU_ID" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
        --model "$model_dir" --served-model-name "$model_id" --host 127.0.0.1 --port "$port"
        --dtype bfloat16 --seed 42 --max-model-len 8192 --max-num-seqs 16
        --gpu-memory-utilization 0.90 --generation-config vllm
        --default-chat-template-kwargs '{"enable_thinking":false}')
    echo "[$(date -Is)] START $label" | tee -a "$LOG_ROOT/queue.log"
    "${cmd[@]}" >"$log" 2>&1 &
    local pid=$!
    if ! wait_for_server "$pid" "$port"; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi
    if ! DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu bash "$FULL6_RUNNER" \
        "$model_id" "http://127.0.0.1:$port" "$label" "$work_dir"; then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
    fi
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "[$(date -Is)] DONE $label" | tee -a "$LOG_ROOT/queue.log"
}

build_aimer() {
    local ratio="$1"
    local out="$ARTIFACT_ROOT/aimer_${ratio}pct"
    [[ -f "$out/aimer_channel_rankings.pt" ]] && return
    mkdir -p "$out"
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/static_moe_prunning/code/scripts/build_aimer_channel_profile.py" \
        --model-path "$MODEL_PATH" --aimer-root "$ROOT/static_moe_prunning" \
        --output-profile "$out/aimer_channel_${ratio}pct_per_layer.pt" \
        --output-channel-cache "$out/aimer_channel_rankings.pt" \
        --target-pruning-ratio "0.$(printf '%02d' "$ratio")" --channel-block-size 64
}

export_aimer() {
    local ratio="$1"
    local retained="$2"
    local out="$ARTIFACT_ROOT/aimer_${ratio}pct"
    [[ -f "$out/checkpoint/pruning_export_manifest.json" ]] && return
    if [[ ! -f "$out/fixed_profile.pt" ]]; then
        PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/PP/build_protected_rankings.py" \
            --model-path "$MODEL_PATH" --backbone-cache "$out/aimer_channel_rankings.pt" \
            --pseudo-cache "$out/aimer_channel_rankings.pt" --output-profile "$out/fixed_profile.pt" \
            --output-channel-cache "$out/fixed_rankings.pt" --method aimer_channel --backbone aimer \
            --retained-blocks "$((retained / 64))" --protection-ratio 0
    fi
    mkdir -p "$out/checkpoint"
    PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/PP/export_uniform_moe.py" \
        --model-path "$MODEL_PATH" --profile "$out/fixed_profile.pt" \
        --channel-cache "$out/fixed_rankings.pt" --output-dir "$out/checkpoint" \
        --retained-channels "$retained"
}

build_random() {
    local ratio="$1"
    local out="$ARTIFACT_ROOT/random_${ratio}pct"
    [[ -f "$out/random_rankings.pt" ]] && return
    mkdir -p "$out"
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/WICK/build_random_profiles.py" \
        --model-path "$MODEL_PATH" --pseudo-ranking-cache "$PP_CACHE" --output-dir "$out" \
        --target-pruning-ratio "0.$(printf '%02d' "$ratio")" --protection-ratio 0 --channel-block-size 64 --seed 42
}

build_enp() {
    local calibration="$ENP_ROOT/calibration.pt"
    if [[ ! -f "$calibration" ]]; then
        mkdir -p "$ENP_ROOT"
        PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/static_moe_prunning/code/scripts/build_shared_calibration_token_cache.py" \
            --model-path "$MODEL_PATH" --output-cache "$calibration" --dataset wikitext \
            --config wikitext-2-raw-v1 --split train --text-field text --sequence-length 2048 \
            --calibration-sequences 128 --token-offset 0 --protocol-name "qwen36_gpu5_${TIMESTAMP}"
    fi
    [[ -f "$ENP_ROOT/enp_50pct_per_layer.pt" ]] && return
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$PYTHONPATH" "$ENP_PYTHON_BIN" "$ROOT/static_moe_prunning/code/scripts/build_enp_tenp_profiles.py" \
        --model-path "$MODEL_PATH" --model-family qwen3.6 --calibration-cache "$calibration" \
        --output-statistics "$ENP_ROOT/statistics.pt" --output-channel-cache "$ENP_ROOT/enp_channels.pt" \
        --output-profile-dir "$ENP_ROOT" --routed-param-retention 0.75 0.50 \
        --important-expert-ratio 0.30 --shallow-weight 1.0 --deep-weight 2.0 --channel-block-size 64 \
        --sequence-length 2048 --calibration-sequences 128 --min-tokens-per-expert 32 \
        --allow-undercovered-experts --zero-token-policy prune_uniform --device-map cuda:0
}

build_pp() {
    local backbone="$1"
    local ratio="$2"
    local retained="$3"
    local out="$ARTIFACT_ROOT/${backbone}_pp_${ratio}pct"
    [[ -f "$out/checkpoint/pruning_export_manifest.json" ]] && return
    local backbone_cache
    case "$backbone" in
        aimer) backbone_cache="$ARTIFACT_ROOT/aimer_${ratio}pct/aimer_channel_rankings.pt" ;;
        random) backbone_cache="$ARTIFACT_ROOT/random_${ratio}pct/random_rankings.pt" ;;
        enp) backbone_cache="$ENP_ROOT/enp_channels.pt" ;;
    esac
    mkdir -p "$out"
    PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" "$ROOT/PP/build_protected_rankings.py" \
        --model-path "$MODEL_PATH" --backbone-cache "$backbone_cache" --pseudo-cache "$PP_CACHE" \
        --output-profile "$out/profile.pt" --output-channel-cache "$out/rankings.pt" \
        --method "${backbone}_pp" --backbone "$backbone" \
        --retained-blocks "$((retained / 64))" --protection-ratio 0.10
    PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/PP/export_uniform_moe.py" \
        --model-path "$MODEL_PATH" --profile "$out/profile.pt" --channel-cache "$out/rankings.pt" \
        --output-dir "$out/checkpoint" --retained-channels "$retained"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_CACHE"
mkdir -p "$LOG_ROOT"

build_aimer 25; build_aimer 50
export_aimer 25 384; export_aimer 50 256
build_random 25; build_random 50
build_enp
build_pp random 25 384; build_pp random 50 256
build_pp enp 25 384; build_pp enp 50 256

run_eval AIMER-channel-25 "Qwen3.6-35B-A3B-AIMER-channel-25-$TIMESTAMP" "$ARTIFACT_ROOT/aimer_25pct/checkpoint" "$PORT_BASE"
run_eval AIMER-channel-50 "Qwen3.6-35B-A3B-AIMER-channel-50-$TIMESTAMP" "$ARTIFACT_ROOT/aimer_50pct/checkpoint" "$((PORT_BASE + 1))"
run_eval random-PP-25 "Qwen3.6-35B-A3B-random-PP-25-$TIMESTAMP" "$ARTIFACT_ROOT/random_pp_25pct/checkpoint" "$((PORT_BASE + 2))"
run_eval random-PP-50 "Qwen3.6-35B-A3B-random-PP-50-$TIMESTAMP" "$ARTIFACT_ROOT/random_pp_50pct/checkpoint" "$((PORT_BASE + 3))"
run_eval ENP-PP-25 "Qwen3.6-35B-A3B-ENP-PP-25-$TIMESTAMP" "$ARTIFACT_ROOT/enp_pp_25pct/checkpoint" "$((PORT_BASE + 4))"
run_eval ENP-PP-50 "Qwen3.6-35B-A3B-ENP-PP-50-$TIMESTAMP" "$ARTIFACT_ROOT/enp_pp_50pct/checkpoint" "$((PORT_BASE + 5))"