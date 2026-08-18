#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the exported Pure-Pseudo checkpoint.}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to a directory created by create_result_dir.sh.}"
METHOD="${METHOD:-PurePseudo-K8-Q4}"
MODEL_ID="${MODEL_ID:-Qwen330BA3BInstruct-50-$METHOD}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-18080}"
DRY_RUN="${DRY_RUN:-false}"
SERVER_LOG="$EXPERIMENT_DIR/server_logs/$METHOD.log"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

wait_for_server() {
    local server_pid="$1"
    for _ in $(seq 1 180); do
        if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
            return
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "vLLM server exited before becoming healthy: $SERVER_LOG" >&2
            tail -100 "$SERVER_LOG" >&2
            exit 1
        fi
        sleep 1
    done
    die "Timed out waiting for vLLM server: $SERVER_LOG"
}

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be a non-negative physical GPU index."
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer."
[[ -x "$VLLM_PYTHON" ]] || die "vLLM Python executable is not executable: $VLLM_PYTHON"
[[ -f "$CHECKPOINT_DIR/config.json" ]] || die "Exported checkpoint is missing config.json: $CHECKPOINT_DIR"
[[ -f "$CHECKPOINT_DIR/pruning_export_manifest.json" ]] || die "Export manifest is missing: $CHECKPOINT_DIR"
[[ -f "$EXPERIMENT_DIR/experiment_manifest.json" ]] || die "Invalid experiment directory: $EXPERIMENT_DIR"
[[ -x "$CODE_ROOT/scripts/create_result_dir.sh" ]] || die "Framework result helper is missing."
mkdir -p "$EXPERIMENT_DIR/server_logs"

server_command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPU_ID"
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
    --model "$CHECKPOINT_DIR"
    --served-model-name "$MODEL_ID"
    --host 127.0.0.1
    --port "$PORT"
    --dtype bfloat16
    --seed 42
    --max-model-len 8192
    --max-num-seqs 16
    --gpu-memory-utilization 0.90
    --generation-config vllm
    --default-chat-template-kwargs '{"enable_thinking":false}'
)

if [[ "$DRY_RUN" == "true" ]]; then
    printf '%q ' "${server_command[@]}"
    printf '> %q 2>&1 &\n' "$SERVER_LOG"
    DRY_RUN=true bash "$ROOT/WICK/run_vllm_quick9.sh" "$MODEL_ID" "http://127.0.0.1:$PORT" "$METHOD" "$EXPERIMENT_DIR"
    exit 0
fi

"${server_command[@]}" >"$SERVER_LOG" 2>&1 &
server_pid=$!
cleanup() {
    kill -TERM "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait_for_server "$server_pid"
curl --silent --fail "http://127.0.0.1:$PORT/v1/models" >/dev/null

bash "$ROOT/WICK/run_vllm_quick9.sh" "$MODEL_ID" "http://127.0.0.1:$PORT" "$METHOD" "$EXPERIMENT_DIR"

kill -TERM "$server_pid"
wait "$server_pid" || true
trap - EXIT INT TERM