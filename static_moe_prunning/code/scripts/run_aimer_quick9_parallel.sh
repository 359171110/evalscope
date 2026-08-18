#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
PROFILE_ROOT="${PROFILE_ROOT:-$EXPERIMENT_ROOT/experiments/profiles/qwen3_mixed_512x1024_code_augmented_20260802}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/experiments/results/qwen3_mixed_512x1024_global_quick9_20260803}"
GPUS_CSV="${GPUS_CSV:-4,5,6,7}"
DRY_RUN="${DRY_RUN:-false}"
RUNNER="$SCRIPT_DIR/run_evalscope_static_profile.py"
PROFILE="$PROFILE_ROOT/aimer_50pct_per_layer.pt"
CHANNEL_CACHE="$PROFILE_ROOT/aimer_topology_channels.pt"
MODEL_ID="qwen3-mixed-512x1024-global-quick9-50pct-aimer"
PROFILE_SHA256="93092a6200ce6f8427664404bac8c1aa06e1f368f8c9253bc9943dda9c9755b8"
CHANNEL_SHA256="2c31a313d8f049f94dc475b3a2bde044842afcd8f558f31d24c7b397a17c6148"
DATASET_ARGS='{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge","ARC-Easy"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}'

DATASETS=(arc hellaswag winogrande gsm8k math_500 mmlu)
LIMITS=(300 1000 400 128 20 10)
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "ERROR: GPUS_CSV must contain at least one GPU ID." >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[4-7]$ ]]; then
        echo "ERROR: only physical GPUs 4-7 are allowed; got '$gpu'." >&2
        exit 2
    fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne ${#GPUS[@]} ]]; then
    echo "ERROR: GPUS_CSV must not contain duplicate GPU IDs." >&2
    exit 2
fi
if [[ "$(sha256sum "$PROFILE" | awk '{print $1}')" != "$PROFILE_SHA256" ]]; then
    echo "ERROR: AIMER profile SHA256 does not match the frozen quick9 artifact." >&2
    exit 2
fi
if [[ "$(sha256sum "$CHANNEL_CACHE" | awk '{print $1}')" != "$CHANNEL_SHA256" ]]; then
    echo "ERROR: AIMER topology channel SHA256 does not match the frozen quick9 artifact." >&2
    exit 2
fi

SHARD_ROOT="$RESULTS_ROOT/parallel_shards"
LOG_ROOT="$RESULTS_ROOT/launcher_logs"
mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

run_shard() {
    local gpu="$1"
    local dataset="$2"
    local limit="$3"
    local work_dir="$SHARD_ROOT/aimer_${dataset}_gpu${gpu}"
    local log_path="$LOG_ROOT/aimer_${dataset}_gpu${gpu}.log"

    env \
        "CUDA_VISIBLE_DEVICES=$gpu" \
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}" \
        "NUMEXPR_MAX_THREADS=16" \
        "PYTHONPATH=$ROOT:$EXPERIMENT_ROOT/code" \
        "$PYTHON_BIN" "$RUNNER" \
        --model-path "$MODEL_PATH" \
        --model-id "$MODEL_ID" \
        --model-family qwen3 \
        --profile "$PROFILE" \
        --channel-cache "$CHANNEL_CACHE" \
        --expected-profile-file-sha256 "$PROFILE_SHA256" \
        --expected-channel-file-sha256 "$CHANNEL_SHA256" \
        --work-dir "$work_dir" \
        --datasets "$dataset" \
        --dataset-args "$DATASET_ARGS" \
        --dataset-limits "{\"${dataset}\":${limit}}" \
        --generation-config '{"max_tokens":1024}' \
        --eval-batch-size 1 \
        --seed 42 \
        --correction-mode none \
        --max-correction-ratio 0.20 \
        --moe-backend torch_index_add \
        --no-enable-thinking \
        --no-timestamp >"$log_path" 2>&1
}

run_worker() {
    local gpu_index="$1"
    local gpu="${GPUS[$gpu_index]}"
    local dataset_index
    for ((dataset_index = gpu_index; dataset_index < ${#DATASETS[@]}; dataset_index += ${#GPUS[@]})); do
        run_shard "$gpu" "${DATASETS[$dataset_index]}" "${LIMITS[$dataset_index]}"
    done
}

if [[ "$DRY_RUN" == "true" ]]; then
    for index in "${!DATASETS[@]}"; do
        gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
        echo "GPU $gpu dataset: ${DATASETS[$index]} limit: ${LIMITS[$index]}"
    done
    exit 0
fi

pids=()
for gpu_index in "${!GPUS[@]}"; do
    run_worker "$gpu_index" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"