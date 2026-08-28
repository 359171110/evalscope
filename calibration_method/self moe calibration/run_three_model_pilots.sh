#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/home/xinpei.gao/evalscope
METHOD_DIR="$ROOT/calibration_method/self moe calibration"
PY=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python
OUTPUT_DIR=${OUTPUT_DIR:-"$METHOD_DIR/pilot_outputs"}

MODELS=(
  /data01/datasets/Qwen3-30B-A3B-Instruct-2507
  /data01/datasets/Qwen3.6-35B-A3B
  /data01/datasets/gemma-4-26B-A4B-it
)
NAMES=(qwen3 qwen36 gemma4)
GPUS=(${GPUS:-0 1 2})
MEMORY=(0.85 0.96 0.85)
EAGER=(0 1 0)
MODES=(user_role_continuation user_role_continuation assistant_bootstrap)

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "GPUS must contain exactly three GPU indexes" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
pids=()
for i in 0 1 2; do
  gpu=${GPUS[$i]}
  active=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
  if [[ -n "$active" ]]; then
    echo "GPU $gpu is occupied by compute PIDs: $active" >&2
    exit 1
  fi
done

for i in 0 1 2; do
  args=(
    "$PY" "$METHOD_DIR/build_native_calibration.py"
    --model-path "${MODELS[$i]}"
    --output "$OUTPUT_DIR/${NAMES[$i]}_cn_moe_sc_pilot.pt"
    --blocks 2
    --block-length 256
    --pilot-episodes 16
    --max-attempts 128
    --episode-batch-size 4
    --max-user-tokens 64
    --max-assistant-tokens 128
    --user-generation-mode "${MODES[$i]}"
    --gpu-memory-utilization "${MEMORY[$i]}"
    --seed 42
    --force
  )
  [[ ${EAGER[$i]} -eq 1 ]] && args+=(--enforce-eager)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "${args[@]}" >"$OUTPUT_DIR/${NAMES[$i]}_generation.log" 2>&1 &
  pids+=("$!")
done

status=0
for i in 0 1 2; do
  if ! wait "${pids[$i]}"; then
    status=1
    tail -100 "$OUTPUT_DIR/${NAMES[$i]}_generation.log" >&2 || true
    continue
  fi
  "$PY" "$METHOD_DIR/inspect_native_calibration.py" \
    --cache "$OUTPUT_DIR/${NAMES[$i]}_cn_moe_sc_pilot.pt" \
    --model-path "${MODELS[$i]}" \
    --sample-count 4 >"$OUTPUT_DIR/${NAMES[$i]}_inspection.json"
done
exit "$status"