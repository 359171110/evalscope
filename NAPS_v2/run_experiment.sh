#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
    echo "Usage: $0 MODEL_TAG RATIO VARIANT CHECKPOINT GPU PORT TIMESTAMP RESULT_ROOT PROTOCOL" >&2
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
PROTOCOL=$9
ROOT=/data01/home/xinpei.gao/evalscope

case "$MODEL_TAG" in
    qwen3)
        MODEL_NAME=Qwen330BA3BInstruct
        DEFAULT_VLLM_PYTHON=/data01/home/xuzk/anaconda3/envs/vllm/bin/python
        ;;
    qwen36)
        MODEL_NAME=Qwen36-35B-A3B
        DEFAULT_VLLM_PYTHON=/data01/home/xuzk/anaconda3/envs/vllm/bin/python
        ;;
    gemma4)
        MODEL_NAME=Gemma4-26B-A4B-it
        DEFAULT_VLLM_PYTHON=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python
        VLLM_BIN_DIR=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin
        ;;
    *) echo "Unsupported model tag: $MODEL_TAG" >&2; exit 2 ;;
esac
VLLM_PYTHON=${VLLM_PYTHON:-$DEFAULT_VLLM_PYTHON}
VLLM_BIN_DIR=${VLLM_BIN_DIR:-$(dirname "$VLLM_PYTHON")}
CALIBRATION_TAG=CalibrationFree
case "$VARIANT" in
    dense) METHOD=Dense ;;
    mask) METHOD=NAPS-v2-Mask ;;
    aimer_gateup50_pp) METHOD=AIMER-GateUp50-PP ;;
    expertcomp) METHOD=NAPS-v2-ExpertComp ;;
    heterogeneous) METHOD=NAPS-v2-Heterogeneous-Mask ;;
    heterogeneous_aimer) METHOD=NAPS-v2-Heterogeneous-AIMER-only ;;
    uniform_medium_aimer) METHOD=NAPS-v2-Uniform-Medium-AIMER ;;
    uniform_large_aimer) METHOD=NAPS-v2-Uniform-Large-AIMER ;;
    heterogeneous_adaptive) METHOD=NAPS-v2-Heterogeneous-Adaptive-Mask ;;
    heterogeneous_aimer_pp) METHOD=NAPS-v2-Heterogeneous-AIMER-PP ;;
    heterogeneous_gaussian) METHOD=NAPS-v2-Heterogeneous-Gaussian ;;
    channel_uniform)
        METHOD=CHANNEL-Uniform-Nested
        CALIBRATION_TAG=LabelFree128
        ;;
    channel_sparse_merge)
        METHOD=CHANNEL-Uniform-SparseMerge
        CALIBRATION_TAG=LabelFree128
        ;;
    channel_puzzle)
        METHOD=CHANNEL-Puzzle-Materialized
        CALIBRATION_TAG=LabelFree128
        ;;
    channel_heterogeneous)
        METHOD=CHANNEL-Heterogeneous-FitUtility
        CALIBRATION_TAG=LabelFree128
        ;;
    *) echo "Unsupported variant: $VARIANT" >&2; exit 2 ;;
esac
case "$PROTOCOL" in
    full6) RUNNER="$ROOT/NAPS_v2/run_full6_v1.sh" ;;
    first5) RUNNER="$ROOT/NAPS_v2/run_first5_v1.sh" ;;
    diagnostic2) RUNNER="$ROOT/NAPS_v2/run_diagnostic2_v1.sh" ;;
    *) echo "Unsupported protocol: $PROTOCOL" >&2; exit 2 ;;
esac

CHECKPOINT=$(realpath "$CHECKPOINT")
EXPERIMENT="${MODEL_NAME}_${RATIO}_vllm_${CALIBRATION_TAG}_${PROTOCOL}_v1_${METHOD}_${TIMESTAMP}_42"
EXPERIMENT_DIR="$RESULT_ROOT/$EXPERIMENT"
MODEL_ID="${MODEL_NAME}-${RATIO}-${METHOD}-${PROTOCOL}-v1-${TIMESTAMP}"
LOG="$EXPERIMENT_DIR/server_logs/$METHOD.log"
mkdir -p "$(dirname "$LOG")"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

env -u LD_LIBRARY_PATH PATH="$VLLM_BIN_DIR:$PATH" CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
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

bash "$RUNNER" "$MODEL_ID" "http://127.0.0.1:$PORT" "$METHOD" "$EXPERIMENT_DIR"
printf '%s\n' "$EXPERIMENT_DIR"