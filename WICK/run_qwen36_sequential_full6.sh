#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
ENP_DATA_PYTHON="${ENP_DATA_PYTHON:-$VLLM_PYTHON}"
ENP_PYTHON="${ENP_PYTHON:-/data01/home/xinpei.gao/.conda/envs/gemma4-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
GPU_ID="${GPU_ID:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
ARTIFACT_TIMESTAMP="${ARTIFACT_TIMESTAMP:-$TIMESTAMP}"
RUN_STAGES="${RUN_STAGES:-Original,PPFrozenV1-B9,PPFrozenV1-B6,random-25,random-50,ENP-25,ENP-50}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6_unlimited.sh"
RANKINGS="$ROOT/PP/experiments/profiles/qwen36_35b_a3b_pp_frozen_v1_20260808/rankings.pt"
ARTIFACT_ROOT="$ROOT/WICK/experiments/profiles/qwen36_random_full6_${ARTIFACT_TIMESTAMP}"
ENP_CALIBRATION="$ROOT/static_moe_prunning/experiments/calibration/qwen36_wikitext128x2048_${ARTIFACT_TIMESTAMP}/calibration.pt"
ENP_ARTIFACT_ROOT="$ROOT/static_moe_prunning/experiments/calibration/qwen36_wikitext128x2048_${ARTIFACT_TIMESTAMP}"
ENP_PROFILE_ROOT="$ROOT/static_moe_prunning/experiments/profiles/qwen36_wikitext128x2048_enp_${ARTIFACT_TIMESTAMP}"
LOG_ROOT="$RESULT_ROOT/Qwen36_sequential_full6_${TIMESTAMP}_42"

require_file() { [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }; }

stage_complete() {
    local label="$1" report_root="$LOG_ROOT/$label/$label"
    local dataset expected report actual
    for spec in arc:3548 hellaswag:10042 winogrande:1267 gsm8k:1319 math_500:500 mmlu:14042; do
        dataset="${spec%%:*}"
        expected="${spec##*:}"
        report="$(find "$report_root/$dataset/reports" -type f -name "${dataset}.json" -print -quit 2>/dev/null || true)"
        [[ -n "$report" ]] || return 1
        actual="$("$PYTHON_BIN" - "$report" <<'PY'
import json
import sys


def leaf_counts(value):
    if isinstance(value, dict):
        child_counts = [count for item in value.values() if isinstance(item, (dict, list))
                        for count in leaf_counts(item)]
        if child_counts:
            return child_counts
        count = value.get('num')
        return [count] if isinstance(count, int) else []
    if isinstance(value, list):
        return [count for item in value for count in leaf_counts(item)]
    return []


with open(sys.argv[1], encoding='utf-8') as report_file:
    print(sum(leaf_counts(json.load(report_file))))
PY
)"
        [[ "$actual" == "$expected" ]] || return 1
    done
}

should_run() { [[ ",$RUN_STAGES," == *",$1,"* ]]; }

stop_server() {
    local pid="$1"
    kill -TERM "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

wait_for_server() {
    local pid="$1" port="$2"
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then return; fi
        kill -0 "$pid" 2>/dev/null || { echo "vLLM exited on port $port" >&2; return 1; }
        sleep 1
    done
    echo "Timed out waiting for vLLM on port $port" >&2
    return 1
}

run_eval() {
    local label="$1" model_id="$2" model_dir="$3" port="$4"
    local work_dir="$LOG_ROOT/$label"
    if stage_complete "$label"; then
        echo "[$(date -Is)] SKIP $label (complete)" | tee -a "$LOG_ROOT/queue.log"
        return
    fi
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
        stop_server "$pid"
        return 1
    fi
    local eval_status=0
    set +e
    DATASETS=arc,hellaswag,winogrande,gsm8k,math_500,mmlu \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$port" "$label" "$work_dir"
    eval_status=$?
    set -e
    stop_server "$pid"
    if ! stage_complete "$label"; then
        echo "[$(date -Is)] FAILED $label (eval exit $eval_status, incomplete reports)" | tee -a "$LOG_ROOT/queue.log" >&2
        [[ "$eval_status" -ne 0 ]] && return "$eval_status"
        return 1
    fi
    echo "[$(date -Is)] DONE $label" | tee -a "$LOG_ROOT/queue.log"
}

build_random() {
    mkdir -p "$ARTIFACT_ROOT"
    for ratio in 25 50; do
        local out="$ARTIFACT_ROOT/random_${ratio}pct"
        if [[ ! -f "$out/random_profile.pt" ]]; then
            mkdir -p "$out"
            PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code" "$PYTHON_BIN" "$ROOT/WICK/build_random_profiles.py" \
                --model-path "$MODEL_PATH" --pseudo-ranking-cache "$RANKINGS" --output-dir "$out" \
                --target-pruning-ratio "0.$(printf '%02d' "$ratio")" --protection-ratio 0 --channel-block-size 64 --seed 42
            mv "$out/random_50pct_per_expert.pt" "$out/random_profile.pt"
        fi
        local retained=$((512 * (100 - ratio) / 100))
        if [[ ! -f "$out/checkpoint/pruning_export_manifest.json" ]]; then
            mkdir -p "$out/checkpoint"
            PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/PP/export_uniform_moe.py" --model-path "$MODEL_PATH" \
                --profile "$out/random_profile.pt" --channel-cache "$out/random_rankings.pt" \
                --output-dir "$out/checkpoint" --retained-channels "$retained"
        fi
    done
}

build_enp() {
    if [[ ! -f "$ENP_CALIBRATION" ]]; then
        mkdir -p "$ENP_ARTIFACT_ROOT"
        PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code" "$ENP_DATA_PYTHON" \
            "$ROOT/static_moe_prunning/code/scripts/build_shared_calibration_token_cache.py" \
            --model-path "$MODEL_PATH" --output-cache "$ENP_CALIBRATION" --dataset wikitext \
            --config wikitext-2-raw-v1 --split train --text-field text --sequence-length 2048 \
            --calibration-sequences 128 --token-offset 0 --protocol-name "qwen36_wikitext128x2048_${TIMESTAMP}"
    fi
    if [[ ! -f "$ENP_PROFILE_ROOT/enp_25pct_per_layer.pt" ]]; then
        mkdir -p "$ENP_PROFILE_ROOT"
        CUDA_VISIBLE_DEVICES="$GPU_ID" \
            MOE_EXTRA_SITE_PACKAGES="/data01/home/xuzk/anaconda3/envs/vllm/lib/python3.10/site-packages" \
            PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code" "$ENP_PYTHON" \
            "$ROOT/static_moe_prunning/code/scripts/build_enp_tenp_profiles.py" \
            --model-path "$MODEL_PATH" --model-family qwen3.6 --calibration-cache "$ENP_CALIBRATION" \
            --output-statistics "$ENP_ARTIFACT_ROOT/enp_statistics.pt" \
            --output-channel-cache "$ENP_ARTIFACT_ROOT/enp_channels.pt" --output-profile-dir "$ENP_PROFILE_ROOT" \
            --routed-param-retention 0.75 0.50 --important-expert-ratio 0.30 --shallow-weight 1.0 --deep-weight 2.0 \
            --channel-block-size 64 --sequence-length 2048 --calibration-sequences 128 \
            --min-tokens-per-expert 32 --allow-undercovered-experts --zero-token-policy prune_uniform --device-map cuda:0
    fi
}

export_enp() {
    local ratio="$1" retained checkpoint
    if [[ "$ratio" == 25 ]]; then
        retained=384
    else
        retained=256
    fi
    checkpoint="$ENP_ARTIFACT_ROOT/checkpoint_${ratio}pct"
    if [[ ! -f "$checkpoint/pruning_export_manifest.json" ]]; then
        mkdir -p "$checkpoint"
        PYTHONPATH="$ROOT" "$ENP_PYTHON" "$ROOT/PP/export_uniform_moe.py" \
            --model-path "$MODEL_PATH" --profile "$ENP_PROFILE_ROOT/enp_${ratio}pct_per_layer.pt" \
            --channel-cache "$ENP_ARTIFACT_ROOT/enp_channels.pt" --output-dir "$checkpoint" \
            --retained-channels "$retained" >"$checkpoint/export.log" 2>&1
    fi
    printf '%s\n' "$checkpoint"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$RANKINGS"
mkdir -p "$LOG_ROOT"

should_run Original && run_eval Original "Qwen3.6-35B-A3B-Original-rerun-$TIMESTAMP" "$MODEL_PATH" 18401
should_run PPFrozenV1-B9 && run_eval PPFrozenV1-B9 "Qwen3.6-35B-A3B-B9-rerun-$TIMESTAMP" "$ROOT/result/Qwen36_35B_A3B_25_vllm_CalibrationFree_full6_v1_PPFrozenV1-B9_202608081430_42/checkpoints/PPFrozenV1-B9" 18402
should_run PPFrozenV1-B6 && run_eval PPFrozenV1-B6 "Qwen3.6-35B-A3B-B6-rerun-$TIMESTAMP" "$ROOT/result/Qwen36_35B_A3B_50_vllm_CalibrationFree_full6_v1_PPFrozenV1-B6_202608081430_42/checkpoints/PPFrozenV1-B6" 18403
if should_run random-25 || should_run random-50; then build_random; fi
should_run random-25 && run_eval random-25 "Qwen3.6-35B-A3B-random-25-$TIMESTAMP" "$ARTIFACT_ROOT/random_25pct/checkpoint" 18404
should_run random-50 && run_eval random-50 "Qwen3.6-35B-A3B-random-50-$TIMESTAMP" "$ARTIFACT_ROOT/random_50pct/checkpoint" 18405
if should_run ENP-25 || should_run ENP-50; then build_enp; fi
if should_run ENP-25; then
    ENP_25_CHECKPOINT="$(export_enp 25)"
    run_eval ENP-25 "Qwen3.6-35B-A3B-ENP-25-$TIMESTAMP" "$ENP_25_CHECKPOINT" 18406
fi
if should_run ENP-50; then
    ENP_50_CHECKPOINT="$(export_enp 50)"
    run_eval ENP-50 "Qwen3.6-35B-A3B-ENP-50-$TIMESTAMP" "$ENP_50_CHECKPOINT" 18407
fi
