#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/home/xinpei.gao/evalscope
PY=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python

build_one() {
    local model=$1
    local model_path=$2
    local widths=$3
    local budget=$4
    local out=$5
    local log="$out/build_export.log"
    mkdir -p "$out"
    if [[ -f "$out/checkpoint_hsp_hetero_${widths// /_}_budget${budget}/pruning_export_manifest.json" ]]; then
        echo "skip completed $model"
        return 0
    fi
    {
        echo "build_start model=$model widths=$widths budget=$budget"
        "$PY" -u -m CSP.build_csp_artifacts \
            --model-path "$model_path" \
            --output-channel-cache "$out/csp_rankings.pt" \
            --output-profile "$out/hsp_hetero_${widths// /_}_budget${budget}.pt" \
            --heterogeneous-widths $widths \
            --budget-width "$budget" \
            --apply-input-scale never
        "$PY" -u -m CSP.export_csp_checkpoint \
            --model-path "$model_path" \
            --profile "$out/hsp_hetero_${widths// /_}_budget${budget}.pt" \
            --channel-cache "$out/csp_rankings.pt" \
            --output-dir "$out/checkpoint_hsp_hetero_${widths// /_}_budget${budget}"
        echo "build_done model=$model"
    } >"$log" 2>&1
}

export PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/CSP/experiments/qwen36_hsp_hetero_20260828" "$ROOT/CSP/experiments/gemma4_hsp_hetero_20260828"

build_one qwen36 /data01/datasets/Qwen3.6-35B-A3B "192 256 320" 256 "$ROOT/CSP/experiments/qwen36_hsp_hetero_20260828" &
pid_qwen36=$!
build_one gemma4 /data01/datasets/gemma-4-26B-A4B-it "288 352 416" 352 "$ROOT/CSP/experiments/gemma4_hsp_hetero_20260828" &
pid_gemma4=$!

status=0
wait "$pid_qwen36" || status=1
wait "$pid_gemma4" || status=1
exit "$status"
