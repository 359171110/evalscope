#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-}"
GPU="${GPU:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE_DIR="${PROFILE_DIR:-$ROOT/experiments/profiles/frontier_committee_regret_rebuild_$RUN_ID}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/results/frontier_committee_regret_e2e_$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ROOT/experiments/logs/frontier_committee_regret_e2e_$RUN_ID}"
case "$GPU" in 4|5|6|7) ;; *) echo "GPU must be one of physical 4,5,6,7; got $GPU" >&2; exit 2;; esac
[[ -n "$MODEL_PATH" ]] || { echo "MODEL_PATH must point to a local Qwen3 MoE checkpoint" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$PROFILE_DIR" "$OUT_DIR" "$LOG_DIR"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

STRUCTURAL="$PROFILE_DIR/structural_q995_min.pt"
FINAL_PROFILE="$PROFILE_DIR/compute_tail_anchor_q995_min.pt"
TAIL="$ROOT/experiments/calibration/tail_risk_channels_20260728/qwen3_channels_b64_tail_0p50.pt"

"$PYTHON_BIN" "$ROOT/code/scripts/build_tail_risk_profile.py" \
  --teacher-cache "$ROOT/experiments/calibration/parent_component_dual_050_20260728/teacher.pt" \
  --reference-channel-cache "$ROOT/experiments/calibration/static_expert_rms_20260728/qwen3_channels_b64_rms.pt" \
  --tail-channel-cache "$TAIL" --risk-floor-cache "$TAIL" \
  --output-profile "$STRUCTURAL" --target-pruning-ratio 0.60 \
  --risk-floor-min-width 2 --risk-floor-early-layers 48 \
  --risk-floor-quantile 0.995 --risk-floor-relative-max 0.10 \
  --frontier-reference-profile "$ROOT/experiments/profiles/qwen3_compute_frontier_tail_risk_at_tail_anchor_20260728/profile.pt" \
  --frontier-regret-cache "$ROOT/experiments/calibration/block_committee_regret_folds16_20260728/offset_0/teacher.pt" \
  --frontier-regret-cache "$ROOT/experiments/calibration/block_committee_regret_folds16_20260728/offset_262144/teacher.pt" \
  --frontier-regret-cache "$ROOT/experiments/calibration/block_committee_regret_folds16_20260728/offset_524288/teacher.pt" \
  --frontier-regret-cache "$ROOT/experiments/calibration/block_committee_regret_folds16_20260728/offset_786432/teacher.pt" \
  --frontier-regret-floor-quantile 0.995 --frontier-regret-width-increment 1 \
  --frontier-regret-fold-aggregation minimum 2>&1 | tee "$LOG_DIR/build_structural.log"

"$PYTHON_BIN" "$ROOT/code/scripts/build_compute_calibrated_profile.py" \
  --source-profile "$STRUCTURAL" --output-profile "$FINAL_PROFILE" \
  --target-routed-pruning-ratio 0.1523270722892549 \
  --compute-route-cache "$TAIL" --search-iterations 64 \
  2>&1 | tee "$LOG_DIR/build_compute.log"

if [[ "${SKIP_PPL:-0}" == "1" ]]; then echo "profile=$FINAL_PROFILE"; exit 0; fi
"$PYTHON_BIN" "$ROOT/code/scripts/static_expert_ppl_eval.py" \
  --model-path "$MODEL_PATH" --profile "$FINAL_PROFILE" --channel-cache "$TAIL" \
  --output-dir "$OUT_DIR" --correction-modes none --sequence-length 2048 \
  --moe-backend torch_index_add 2>&1 | tee "$LOG_DIR/full_ppl.log"
echo "profile=$FINAL_PROFILE"
echo "result=$OUT_DIR/static_expert_wikitext_ppl.json"
