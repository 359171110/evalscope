#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="$ROOT/static_moe_prunning"
RUNNER="$EXPERIMENT_ROOT/code/scripts/run_evalscope_static_profile.py"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/WICK/experiments/profiles/qwen3_wick_gram_protect_20260806}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/WICK/experiments/results/qwen3_wick_gram_protect_quick9_20260806}"
GPU_ID="${GPU_ID:-2}"
DRY_RUN="${DRY_RUN:-false}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

PROFILE="$PROFILE_ROOT/wick_gram_protect_50pct_per_expert.pt"
CHANNEL_CACHE="$PROFILE_ROOT/wick_gram_protect_rankings.pt"
EXPECTED_PROFILE_SHA256="5eaff87e5fc0c35e6675bfd6f75cdfd1dc990a05c6081c4d224c13ee675ea674"
EXPECTED_CHANNEL_SHA256="7f85bd613c1af0e1f9fff7bd89c04b7ab64ee8836856542b35d9ca8a03d8cbf0"
MODEL_ID="qwen3-wick-gram-protect-50pct-quick9"
DATASET_ARGS='{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge","ARC-Easy"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}'

DATASETS=(arc hellaswag winogrande gsm8k math_500 mmlu)
LIMITS=(300 1000 10 400 128 20)
MAX_TOKENS=(64 32 1536 32 1024 4096)

die() {
    echo "ERROR: $*" >&2
    exit 2
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

[[ "$GPU_ID" =~ ^[0-7]$ ]] || die "GPU_ID must be a physical GPU from 0 to 7."
[[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
[[ -f "$RUNNER" ]] || die "EvalScope runner does not exist: $RUNNER"
[[ -f "$PROFILE" ]] || die "WICK profile does not exist: $PROFILE"
[[ -f "$CHANNEL_CACHE" ]] || die "WICK channel cache does not exist: $CHANNEL_CACHE"
[[ "$(file_sha256 "$PROFILE")" == "$EXPECTED_PROFILE_SHA256" ]] || die "WICK profile SHA256 changed."
[[ "$(file_sha256 "$CHANNEL_CACHE")" == "$EXPECTED_CHANNEL_SHA256" ]] || die "WICK channel cache SHA256 changed."

SHARD_ROOT="$RESULTS_ROOT/parallel_shards"
LOG_ROOT="$RESULTS_ROOT/launcher_logs"
mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

run_shard() {
    local dataset="$1"
    local limit="$2"
    local max_tokens="$3"
    local work_dir="$SHARD_ROOT/wick_gram_protect_${dataset}"
    local log_path="$LOG_ROOT/wick_gram_protect_${dataset}.log"
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$GPU_ID"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "NUMEXPR_MAX_THREADS=16"
        "PYTHONPATH=$ROOT:$EXPERIMENT_ROOT/code"
        "$PYTHON_BIN"
        "$RUNNER"
        --model-path "$MODEL_PATH"
        --model-id "$MODEL_ID"
        --model-family qwen3
        --profile "$PROFILE"
        --channel-cache "$CHANNEL_CACHE"
        --expected-profile-file-sha256 "$EXPECTED_PROFILE_SHA256"
        --expected-channel-file-sha256 "$EXPECTED_CHANNEL_SHA256"
        --work-dir "$work_dir"
        --datasets "$dataset"
        --dataset-args "$DATASET_ARGS"
        --dataset-limits "{\"${dataset}\":${limit}}"
        --generation-config "{\"max_tokens\":${max_tokens}}"
        --eval-batch-size 1
        --seed 42
        --correction-mode none
        --max-correction-ratio 0.20
        --moe-backend torch_index_add
        --no-enable-thinking
        --no-timestamp
    )
    if [[ -d "$work_dir/predictions" ]]; then
        command+=(--use-cache "$work_dir")
    fi
    if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
        command+=(--preflight-only)
    fi

    printf 'Starting dataset=%s limit=%s max_tokens=%s gpu=%s\n' "$dataset" "$limit" "$max_tokens" "$GPU_ID"
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    "${command[@]}" >"$log_path" 2>&1
    printf 'Completed dataset=%s log=%s\n' "$dataset" "$log_path"
}

for index in "${!DATASETS[@]}"; do
    run_shard "${DATASETS[$index]}" "${LIMITS[$index]}" "${MAX_TOKENS[$index]}"
done
