#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
GPU_ID="${GPU_ID:-3}"
PORT="${PORT:-18431}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/PP/experiments/profiles/qwen3_channel_pp_enp_gpu3_${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-$RESULT_ROOT/Qwen330BA3BInstruct_channel_pp_enp_gpu3_${TIMESTAMP}_42_logs}"

AIMER_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
RANDOM_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_random_fixed_20260806/random_rankings.pt"
PP_CACHE="$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4/pure_pseudo_rankings.pt"
ENP_CACHE="$ROOT/static_moe_prunning/experiments/calibration/qwen3_wikitext128x2048_enp/enp_signed_projection_channels_b64.pt"
PROTECTION_BUILDER="$ROOT/PP/build_protected_rankings.py"
EXPORTER="$ROOT/WICK/export_uniform_qwen3_moe.py"
RESULT_DIR_SCRIPT="$CODE_ROOT/scripts/create_result_dir.sh"
FULL6_RUNNER="$ROOT/WICK/run_vllm_full6.sh"

export PYTHONPATH="$ROOT:$CODE_ROOT"

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

experiment_dir() {
    local pruning="$1"
    local calibration="$2"
    local method="$3"
    RESULT_ROOT="$RESULT_ROOT" "$RESULT_DIR_SCRIPT" \
        --inference vllm \
        --calibration "$calibration" \
        --method "$method" \
        --pruning-ratio-label "$pruning" \
        --pruning-ratio-percent "$pruning" \
        --timestamp "$TIMESTAMP" \
        --dry-run
}

wait_for_server() {
    local server_pid="$1"
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

build_and_export() {
    local method="$1"
    local backbone="$2"
    local retained_blocks="$3"
    local protection_ratio="$4"
    local backbone_cache="$5"
    local calibration="$6"
    local pruning="$7"
    local variant_root="$PROFILE_ROOT/$method-$pruning"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local experiment checkpoint_dir

    mkdir -p "$variant_root"
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        env LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}" \
            "$PYTHON_BIN" "$PROTECTION_BUILDER" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$backbone_cache" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}-${pruning}" \
            --backbone "$backbone" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio "$protection_ratio" \
            >>"$LOG_ROOT/build.log" 2>&1
    fi

    experiment="$(experiment_dir "$pruning" "$calibration" "$method")"
    if [[ ! -f "$experiment/experiment_manifest.json" ]]; then
        RESULT_ROOT="$RESULT_ROOT" "$RESULT_DIR_SCRIPT" \
            --inference vllm \
            --calibration "$calibration" \
            --method "$method" \
            --pruning-ratio-label "$pruning" \
            --pruning-ratio-percent "$pruning" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    checkpoint_dir="$experiment/checkpoints/$method"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        env LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}" \
            "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))" \
            >>"$LOG_ROOT/export.log" 2>&1
    fi
    printf '%s\n' "$experiment|$checkpoint_dir"
}

run_one() {
    local method="$1"
    local backbone="$2"
    local retained_blocks="$3"
    local protection_ratio="$4"
    local backbone_cache="$5"
    local calibration="$6"
    local pruning="$7"
    local model_id="Qwen330BA3BInstruct-${method}-${pruning}-${TIMESTAMP}"
    local experiment checkpoint_dir server_log server_pid

    IFS='|' read -r experiment checkpoint_dir < <(
        build_and_export "$method" "$backbone" "$retained_blocks" "$protection_ratio" \
            "$backbone_cache" "$calibration" "$pruning"
    )
    server_log="$experiment/server_logs/$method.log"
    mkdir -p "$(dirname "$server_log")"
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
        RESULTS_ROOT="$experiment" \
        bash "$FULL6_RUNNER" "$model_id" "http://127.0.0.1:$PORT" "$method" "$experiment"
    cleanup_server
    trap - RETURN
    echo "[$(date -Is)] DONE $method pruning=$pruning" | tee -a "$LOG_ROOT/queue.log"
}

mkdir -p "$LOG_ROOT"
for required in "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors.index.json" \
    "$AIMER_CACHE" "$RANDOM_CACHE" "$PP_CACHE" "$ENP_CACHE" "$PROTECTION_BUILDER" \
    "$EXPORTER" "$RESULT_DIR_SCRIPT" "$FULL6_RUNNER"; do
    require_file "$required"
done

echo "[$(date -Is)] QUEUE START gpu=$GPU_ID port=$PORT timestamp=$TIMESTAMP" | tee "$LOG_ROOT/queue.log"
run_one AIMER aimer 9 0.00 "$AIMER_CACHE" CalibrationFree 25
run_one AIMER aimer 6 0.00 "$AIMER_CACHE" CalibrationFree 50
run_one Random-PP-G10 random 9 0.10 "$RANDOM_CACHE" CalibrationFree 25
run_one Random-PP-G10 random 6 0.10 "$RANDOM_CACHE" CalibrationFree 50
run_one ENP-PP-G10 enp 9 0.10 "$ENP_CACHE" WikiText128x2048 25
run_one ENP-PP-G10 enp 6 0.10 "$ENP_CACHE" WikiText128x2048 50
echo "[$(date -Is)] QUEUE DONE" | tee -a "$LOG_ROOT/queue.log"