#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
GPUS_CSV="${GPUS_CSV:-0,2,3}"
DRY_RUN="${DRY_RUN:-false}"

CALIBRATION_ROOT="$EXPERIMENT_ROOT/experiments/calibration/qwen3_mixed_512x1024_all_experts_20260804"
PROFILE_ROOT="$EXPERIMENT_ROOT/experiments/profiles/qwen3_mixed_512x1024_all_experts_20260804"
RESULTS_ROOT="$EXPERIMENT_ROOT/experiments/results/qwen3_mixed_512x1024_all_experts_quick9_max4096_20260804"
TOKEN_CACHE="$EXPERIMENT_ROOT/experiments/calibration/qwen3_mixed_train_wikitext256_mbpp128_gsm8k64_math64_20260802/mixed_train_512x1024_code_augmented.pt"
RMS_CACHE="$CALIBRATION_ROOT/channels_rms_all_experts_512x1024.pt"
TAIL_CACHE="$CALIBRATION_ROOT/tail_channels/qwen3_channels_b64_tail_0p50.pt"
TEACHER_CACHE="$EXPERIMENT_ROOT/experiments/calibration/qwen3_mixed_512x1024_code_augmented_20260802/conditional_dual_teacher_512x1024_50pct.pt"
AMP_CACHE="$EXPERIMENT_ROOT/experiments/calibration/static_expert_priors_20260728/amp_scores.pt"
AIMER_CACHE="$EXPERIMENT_ROOT/experiments/calibration/static_expert_priors_20260728/aimer_scores.pt"
ROUTE_PROFILE="$PROFILE_ROOT/route_tail_all_experts_50pct_global.pt"
TAIL_PROFILE="$PROFILE_ROOT/tail_risk_all_experts_50pct_global.pt"
RUNNER="$SCRIPT_DIR/run_evalscope_static_profile.py"

DATASET_ARGS='{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge","ARC-Easy"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}'
DATASETS=(arc hellaswag winogrande gsm8k math_500 mmlu)
LIMITS=(300 1000 400 128 20 10)
METHODS=(route_tail_all_experts tail_risk_all_experts)
if [[ -n "${METHODS_CSV:-}" ]]; then
    IFS=',' read -r -a METHODS <<< "$METHODS_CSV"
fi
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

if [[ ${#GPUS[@]} -eq 0 || ${#GPUS[@]} -gt 3 ]]; then
    echo "ERROR: GPUS_CSV must contain between one and three GPU IDs." >&2
    exit 2
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne ${#GPUS[@]} ]]; then
    echo "ERROR: GPUS_CSV must not contain duplicate GPU IDs." >&2
    exit 2
fi

export LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
export NUMEXPR_MAX_THREADS=16
export PYTHONPATH="$ROOT:$EXPERIMENT_ROOT/code"
mkdir -p "$CALIBRATION_ROOT" "$PROFILE_ROOT" "$RESULTS_ROOT/parallel_shards" "$RESULTS_ROOT/launcher_logs"

validate_all_expert_cache() {
    "$PYTHON_BIN" - "$RMS_CACHE" "$TAIL_CACHE" <<'PY'
import sys
from pathlib import Path

import torch

expected_tokens = 512 * 1024
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("expert_coverage") != "all":
        raise ValueError(f"cache is not all-expert calibrated: {path}")
    counts = payload.get("expert_activation_counts")
    if not isinstance(counts, dict) or len(counts) != 48:
        raise ValueError(f"cache must contain activation counts for 48 layers: {path}")
    for layer_id, values in counts.items():
        if tuple(values.shape) != (128,):
            raise ValueError(f"layer {layer_id} must contain 128 expert counts")
        if not bool((values == expected_tokens).all()):
            raise ValueError(f"layer {layer_id} did not send every token to every expert")
    if payload.get("calibration_input_ids_sha256") != "588f0e45bc49601c3fb951828c0b1bb78bf15809e193ff9c5a854ef10483c03a":
        raise ValueError(f"calibration token SHA mismatch: {path}")
    print(f"validated_all_expert_cache={path}")
PY
}

build_profiles() {
    validate_all_expert_cache

    if [[ ! -s "$ROUTE_PROFILE" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/build_static_expert_profiles.py" \
            --channel-cache "$TAIL_CACHE" \
            --output-profile "$ROUTE_PROFILE" \
            --mode route_rms \
            --target-pruning-ratio 0.50 \
            --allocation-scope global
    fi

    [[ -s "$TEACHER_CACHE" ]] || {
        echo "ERROR: frozen historical Conditional-Dual teacher is missing: $TEACHER_CACHE" >&2
        exit 2
    }

    if [[ ! -s "$TAIL_PROFILE" ]]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/build_tail_risk_profile.py" \
            --teacher-cache "$TEACHER_CACHE" \
            --reference-channel-cache "$RMS_CACHE" \
            --tail-channel-cache "$TAIL_CACHE" \
            --output-profile "$TAIL_PROFILE" \
            --target-pruning-ratio 0.50 \
            --allocation-scope global \
            --risk-floor-min-width 2 \
            --risk-floor-early-layers 48 \
            --risk-floor-quantile 0.995 \
            --risk-floor-relative-max 0.10
    fi

    "$PYTHON_BIN" - "$ROUTE_PROFILE" "$TAIL_PROFILE" <<'PY'
import sys
import torch

for raw_path in sys.argv[1:]:
    payload = torch.load(raw_path, map_location="cpu", weights_only=True)
    if int(payload["total_blocks"]) != 36864:
        raise ValueError(f"profile does not retain exactly 36,864 blocks: {raw_path}")
    if int(payload["maximum_blocks"]) != 73728:
        raise ValueError(f"profile maximum block count is not 73,728: {raw_path}")
    print(f"validated_profile={raw_path} profile_sha256={payload['profile_sha256']}")
PY
}

profile_for_method() {
    case "$1" in
        route_tail_all_experts) printf '%s\n' "$ROUTE_PROFILE" ;;
        tail_risk_all_experts) printf '%s\n' "$TAIL_PROFILE" ;;
        *) return 2 ;;
    esac
}

run_job() {
    local gpu="$1"
    local method="$2"
    local dataset="$3"
    local limit="$4"
    local profile
    local profile_sha
    local channel_sha
    local work_dir="$RESULTS_ROOT/parallel_shards/${method}_${dataset}"
    local log_path="$RESULTS_ROOT/launcher_logs/${method}_${dataset}.log"
    local -a cache_args=()

    profile="$(profile_for_method "$method")"
    profile_sha="$(sha256sum "$profile" | awk '{print $1}')"
    channel_sha="$(sha256sum "$TAIL_CACHE" | awk '{print $1}')"
    if find "$work_dir/reports" -name "${dataset}.json" -type f -print -quit 2>/dev/null | grep -q .; then
        echo "SKIP completed: $method $dataset"
        return
    fi
    if find "$work_dir/predictions" -name '*.jsonl' -type f -print -quit 2>/dev/null | grep -q .; then
        cache_args=(--use-cache "$work_dir")
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "GPU $gpu method: $method dataset: $dataset limit: $limit max_tokens: 4096"
        return
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$RUNNER" \
        --model-path "$MODEL_PATH" \
        --model-id "qwen3-mixed-512x1024-all-experts-50pct-${method}" \
        --model-family qwen3 \
        --profile "$profile" \
        --channel-cache "$TAIL_CACHE" \
        --expected-profile-file-sha256 "$profile_sha" \
        --expected-channel-file-sha256 "$channel_sha" \
        --work-dir "$work_dir" \
        --datasets "$dataset" \
        --dataset-args "$DATASET_ARGS" \
        --dataset-limits "{\"${dataset}\":${limit}}" \
        --generation-config '{"max_tokens":4096}' \
        --eval-batch-size 1 \
        --seed 42 \
        --correction-mode none \
        --max-correction-ratio 0.20 \
        --moe-backend torch_index_add \
        --no-enable-thinking \
        --no-timestamp \
        "${cache_args[@]}" >"$log_path" 2>&1
}

run_worker() {
    local worker_index="$1"
    local gpu="${GPUS[$worker_index]}"
    local job_index=0
    local method
    local dataset_index
    for method in "${METHODS[@]}"; do
        for dataset_index in "${!DATASETS[@]}"; do
            if (( job_index % ${#GPUS[@]} == worker_index )); then
                run_job "$gpu" "$method" "${DATASETS[$dataset_index]}" "${LIMITS[$dataset_index]}"
            fi
            job_index=$((job_index + 1))
        done
    done
}

if [[ "$DRY_RUN" != "true" ]]; then
    while [[ ! -s "$RMS_CACHE" || ! -s "$TAIL_CACHE" ]]; do
        echo "waiting_for_all_expert_calibration"
        read -r -t 60 _ </dev/null || true
    done
fi
build_profiles

pids=()
for worker_index in "${!GPUS[@]}"; do
    run_worker "$worker_index" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"