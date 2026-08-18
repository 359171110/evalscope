#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
GPU="${GPU:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${PROFILE:-$ROOT/experiments/profiles/frontier_committee_regret_20260728/compute_tail_anchor_q995_min.pt}"
CHANNEL_CACHE="${CHANNEL_CACHE:-$ROOT/experiments/calibration/tail_risk_channels_20260728/qwen3_channels_b64_tail_0p50.pt}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/results/frontier_committee_regret_full_rerun_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/experiments/logs/frontier_committee_regret_full_rerun_$(date +%Y%m%d_%H%M%S)}"
case "$GPU" in 4|5|6|7) ;; *) echo "GPU must be one of physical 4,5,6,7; got $GPU" >&2; exit 2;; esac
[[ -n "$MODEL_PATH" ]] || { echo "MODEL_PATH must point to a local Qwen3 MoE checkpoint" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROFILE" ]] || { echo "Profile not found: $PROFILE" >&2; exit 2; }
[[ -f "$CHANNEL_CACHE" ]] || { echo "Channel cache not found: $CHANNEL_CACHE" >&2; exit 2; }
mkdir -p "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
echo "physical_gpu=$GPU"
echo "model=$MODEL_PATH"
echo "profile=$PROFILE"
echo "protocol=WikiText-2 test, 114 windows, 233368 tokens, sequence_length=2048"
"$PYTHON_BIN" "$ROOT/code/scripts/static_expert_ppl_eval.py" \
  --model-path "$MODEL_PATH" --profile "$PROFILE" --channel-cache "$CHANNEL_CACHE" \
  --output-dir "$OUT_DIR" --correction-modes none \
  --sequence-length 2048 --moe-backend torch_index_add 2>&1 | tee "$LOG_DIR/full_ppl.stdout.log"
echo "result=$OUT_DIR/static_expert_wikitext_ppl.json"
