#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 MODEL_TAG RATIO VARIANT CHECKPOINT GPU PORT TIMESTAMP RESULT_ROOT" >&2
    exit 2
fi

MODEL_TAG=$1
RATIO=$2
VARIANT=$3
CHECKPOINT=$4
GPU=$5
PORT=$6
TIMESTAMP=$7
RESULT_ROOT=$8
ROOT=/data01/home/xinpei.gao/evalscope
VLLM_PYTHON=${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}

case "$MODEL_TAG" in
    qwen3) MODEL_NAME=Qwen330BA3BInstruct ;;
    qwen36) MODEL_NAME=Qwen36-35B-A3B ;;
    *) echo "Unsupported model tag: $MODEL_TAG" >&2; exit 2 ;;
esac
case "$VARIANT" in
    mask) METHOD=NAPS-Mask ;;
    merge) METHOD=NAPS-Bounded-Merge ;;
    *) echo "Unsupported variant: $VARIANT" >&2; exit 2 ;;
esac

CHECKPOINT=$(realpath "$CHECKPOINT")
EXPERIMENT="${MODEL_NAME}_${RATIO}_vllm_CalibrationFree_full6_v1_${METHOD}_${TIMESTAMP}_42"
EXPERIMENT_DIR="$RESULT_ROOT/$EXPERIMENT"
MODEL_ID="${MODEL_NAME}-${RATIO}-${METHOD}-full6-v1-${TIMESTAMP}"
LOG="$EXPERIMENT_DIR/server_logs/$METHOD.log"
mkdir -p "$(dirname "$LOG")"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf 'EXPERIMENT_DIR=%q\n' "$EXPERIMENT_DIR"
    printf 'MODEL_ID=%q\n' "$MODEL_ID"
    exit 0
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$CHECKPOINT" \
    --served-model-name "$MODEL_ID" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --seed 42 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    >"$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
    if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -n 80 "$LOG" >&2
        exit 1
    fi
    sleep 2
done
env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null

bash "$ROOT/NAPS/run_full6_v1.sh" "$MODEL_ID" "http://127.0.0.1:$PORT" "$METHOD" "$EXPERIMENT_DIR"
printf '%s\n' "$EXPERIMENT_DIR"