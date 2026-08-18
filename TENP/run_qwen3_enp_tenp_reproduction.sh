#!/usr/bin/env bash

set -euo pipefail

TENP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$TENP_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
CALIBRATION_CACHE="${CALIBRATION_CACHE:-$ROOT/static_moe_prunning/experiments/calibration/qwen3_mixed_train_wikitext256_mbpp128_gsm8k64_math64_20260802/mixed_train_512x1024_code_augmented.pt}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT/static_moe_prunning/experiments/calibration/qwen3_mixed_512x1024_enp_tenp_20260804}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/static_moe_prunning/experiments/profiles/qwen3_mixed_512x1024_enp_tenp_20260804}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/static_moe_prunning/experiments/results/qwen3_mixed_512x1024_enp_tenp_quick9_20260804}"
STATISTICS_CACHE="${STATISTICS_CACHE:-$ARTIFACT_ROOT/enp_tenp_statistics.pt}"
CHANNEL_CACHE="${CHANNEL_CACHE:-$ARTIFACT_ROOT/enp_tenp_signed_projection_channels_b64.pt}"
ROUTED_PARAM_RETENTION="${ROUTED_PARAM_RETENTION:-0.60}"
IMPORTANT_EXPERT_RATIO="${IMPORTANT_EXPERT_RATIO:-0.30}"
BUILD_GPU="${BUILD_GPU:-1}"
GPUS_CSV="${GPUS_CSV:-1,3}"
METHODS_CSV="${METHODS_CSV:-enp,tenp}"
ACTION="${1:-dry-run}"

DATASET_ARGS='{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge","ARC-Easy"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}'
DATASET_LIMITS='{"arc":300,"hellaswag":1000,"mmlu":10,"winogrande":400,"gsm8k":128,"math_500":20}'
GENERATION_CONFIG='{"max_tokens":4096}'
DATASETS=(arc hellaswag winogrande gsm8k math_500 mmlu)
IFS=',' read -r -a METHODS <<< "$METHODS_CSV"

usage() {
    cat <<'EOF'
Usage: run_qwen3_enp_tenp_reproduction.sh [dry-run|build|preflight|eval|all]

Environment overrides:
  PYTHON_BIN, MODEL_PATH, CALIBRATION_CACHE, ARTIFACT_ROOT, PROFILE_ROOT,
  RESULTS_ROOT, ROUTED_PARAM_RETENTION, IMPORTANT_EXPERT_RATIO, BUILD_GPU,
    GPUS_CSV, METHODS_CSV.

The evaluation protocol runs ARC 600, HellaSwag 1000, WinoGrande 400,
GSM8K 128, MATH-500 20 examples per Level (100 total), and MMLU 570 in this order,
with max_tokens=4096, seed=42, thinking disabled, and batch size 1.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

ratio_tag() {
    "$PYTHON_BIN" - "$ROUTED_PARAM_RETENTION" <<'PY'
import sys

retention = float(sys.argv[1])
pruning = (1.0 - retention) * 100.0
rounded = round(pruning)
if abs(pruning - rounded) < 1.0e-9:
    print(f"{rounded}pct")
else:
    print(f"{pruning:g}pct".replace(".", "p"))
PY
}

profile_path() {
    local method="$1"
    local tag="$2"
    case "$method" in
        dense)
            printf '%s/dense_full_width.pt\n' "$PROFILE_ROOT"
            ;;
        enp)
            printf '%s/enp_%s_per_layer.pt\n' "$PROFILE_ROOT" "$tag"
            ;;
        tenp)
            printf '%s/tenp_%s_trapezoid.pt\n' "$PROFILE_ROOT" "$tag"
            ;;
        *)
            die "Unknown method '$method'."
            ;;
    esac
}

validate_gpu() {
    local gpu="$1"
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer; got '$gpu'."
}

build_artifacts() {
    [[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
    [[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
    [[ -f "$CALIBRATION_CACHE" ]] || die "Calibration cache does not exist: $CALIBRATION_CACHE"
    validate_gpu "$BUILD_GPU"
    mkdir -p "$ARTIFACT_ROOT" "$PROFILE_ROOT"
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$BUILD_GPU"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        "$CODE_ROOT/scripts/build_enp_tenp_profiles.py"
        --model-path "$MODEL_PATH"
        --model-family qwen3
        --calibration-cache "$CALIBRATION_CACHE"
        --output-statistics "$STATISTICS_CACHE"
        --output-channel-cache "$CHANNEL_CACHE"
        --output-profile-dir "$PROFILE_ROOT"
        --routed-param-retention "$ROUTED_PARAM_RETENTION"
        --important-expert-ratio "$IMPORTANT_EXPERT_RATIO"
        --shallow-weight 1.0
        --deep-weight 2.0
        --channel-block-size 64
        --sequence-length 1024
        --calibration-sequences 512
        --min-tokens-per-expert 32
        --allow-undercovered-experts
        --zero-token-policy keep_full
        --device-map cuda:0
    )
    print_command "${command[@]}"
    "${command[@]}"
}

run_method() {
    local method="$1"
    local gpu="$2"
    local mode="$3"
    local tag="$4"
    local profile
    profile="$(profile_path "$method" "$tag")"
    [[ -f "$profile" ]] || die "Profile does not exist for '$method': $profile"
    [[ -f "$CHANNEL_CACHE" ]] || die "Channel cache does not exist: $CHANNEL_CACHE"
    mkdir -p "$RESULTS_ROOT/$method"
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "NUMEXPR_MAX_THREADS=16"
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "$PYTHON_BIN"
        "$CODE_ROOT/scripts/run_evalscope_static_profile.py"
        --model-path "$MODEL_PATH"
        --model-family qwen3
        --model-id "qwen3-enp-tenp-${tag}-${method}"
        --profile "$profile"
        --channel-cache "$CHANNEL_CACHE"
        --expected-profile-file-sha256 "$(sha256sum "$profile" | awk '{print $1}')"
        --expected-channel-file-sha256 "$(sha256sum "$CHANNEL_CACHE" | awk '{print $1}')"
        --work-dir "$RESULTS_ROOT/$method"
        --datasets "${DATASETS[@]}"
        --dataset-args "$DATASET_ARGS"
        --dataset-limits "$DATASET_LIMITS"
        --generation-config "$GENERATION_CONFIG"
        --eval-batch-size 1
        --seed 42
        --correction-mode none
        --max-correction-ratio 0.20
        --moe-backend torch_index_add
        --no-enable-thinking
        --no-timestamp
    )
    if [[ "$mode" == "preflight" ]]; then
        command+=(--preflight-only)
    elif find "$RESULTS_ROOT/$method/predictions" -name '*.jsonl' -type f -print -quit 2>/dev/null | grep -q .; then
        command+=(--use-cache "$RESULTS_ROOT/$method")
    fi
    print_command "${command[@]}"
    "${command[@]}"
}

run_gpu_worker() {
    local worker_index="$1"
    local gpu="$2"
    local mode="$3"
    local tag="$4"
    local worker_count="$5"
    local method_index method log_path
    for ((method_index = worker_index; method_index < ${#METHODS[@]}; method_index += worker_count)); do
        method="${METHODS[$method_index]}"
        log_path="$RESULTS_ROOT/launcher_logs/${mode}_${method}_gpu${gpu}.log"
        run_method "$method" "$gpu" "$mode" "$tag" >> "$log_path" 2>&1
    done
}

run_evaluation() {
    local mode="$1"
    local tag
    tag="$(ratio_tag)"
    IFS=',' read -r -a gpus <<< "$GPUS_CSV"
    [[ ${#gpus[@]} -gt 0 ]] || die "GPUS_CSV must not be empty."
    declare -A seen=()
    local gpu
    for gpu in "${gpus[@]}"; do
        validate_gpu "$gpu"
        [[ -z "${seen[$gpu]:-}" ]] || die "GPU '$gpu' was specified more than once."
        seen[$gpu]=1
    done
    mkdir -p "$RESULTS_ROOT/launcher_logs"
    local -a pids=()
    local index
    for index in "${!gpus[@]}"; do
        gpu="${gpus[$index]}"
        run_gpu_worker "$index" "$gpu" "$mode" "$tag" "${#gpus[@]}" &
        pids+=("$!")
    done
    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

dry_run() {
    local tag
    tag="$(ratio_tag)"
    echo "Build GPU: $BUILD_GPU"
    echo "Evaluation GPUs: $GPUS_CSV"
    echo "Retention: $ROUTED_PARAM_RETENTION; TENP full-expert ratio: $IMPORTANT_EXPERT_RATIO"
    echo "Datasets: ${DATASETS[*]}"
    echo "Dataset limits: $DATASET_LIMITS"
    echo "Generation config: $GENERATION_CONFIG"
    echo "Statistics: $STATISTICS_CACHE"
    echo "Channel cache: $CHANNEL_CACHE"
    local method
    for method in "${METHODS[@]}"; do
        echo "$method profile: $(profile_path "$method" "$tag")"
    done
}

case "$ACTION" in
    dry-run)
        dry_run
        ;;
    build)
        build_artifacts
        ;;
    preflight)
        run_evaluation preflight
        ;;
    eval)
        run_evaluation eval
        ;;
    all)
        build_artifacts
        run_evaluation preflight
        run_evaluation eval
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        die "Unknown action '$ACTION'."
        ;;
esac