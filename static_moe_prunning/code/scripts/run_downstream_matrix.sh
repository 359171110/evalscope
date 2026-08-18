#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
RUNNER="$EXPERIMENT_ROOT/code/scripts/run_evalscope_static_profile.py"
PYTHON_DEFAULT="/data01/home/xuzk/anaconda3/envs/xhquant/bin/python"
DEFAULT_DATASET_ARGS='{"mmlu_pro":{"local_path":"/data01/datasets/evalscope_benchmarks/mmlu_pro"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"},"ifeval":{"local_path":"/data01/datasets/evalscope_benchmarks/ifeval"},"boolq":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/super_glue/boolq"},"openbookqa":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/openbookqa","subset_list":["main"]},"rte":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/glue/RTE","subset_list":["rte"]},"humaneval_plus":{"local_path":"/data01/datasets/evalscope_benchmarks/humaneval_plus"},"mbpp_plus":{"local_path":"/data01/datasets/evalscope_benchmarks/mbpp_plus"},"live_code_bench":{"local_path":"/data01/datasets/evalscope_benchmarks/live_code_bench/v1","subset_list":["v1"]}}'

MODEL_PATH=""
MODEL_ID=""
MODEL_FAMILY=""
PRUNING_RATIO=""
GPUS_CSV=""
DATASETS_CSV=""
METHODS_CSV=""
PROFILE_ROOT=""
CHANNEL_CACHE=""
RESULTS_ROOT=""
DATASET_ARGS="$DEFAULT_DATASET_ARGS"
DATASET_LIMITS=""
SANDBOX_CONFIG=""
GENERATION_CONFIG='{"max_tokens":1024}'
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
LIMIT=""
EVAL_BATCH_SIZE="1"
SEED="42"
CORRECTION_MODE="none"
MAX_CORRECTION_RATIO="0.20"
MOE_BACKEND="torch_index_add"
PRELIGHT_ONLY="false"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Usage:
  run_downstream_matrix.sh \
    --model-path PATH \
    --model-id ID \
    --pruning-ratio RATIO \
    --gpus GPU[,GPU...] \
    --datasets DATASET[,DATASET...] \
    --methods METHOD[,METHOD...] \
    --channel-cache PATH \
    [options]

Required options:
  --model-path PATH       Local model path.
  --model-id ID           Base model id used in result model ids.
  --pruning-ratio RATIO   Profile ratio tag, e.g. 50pct, 0.5, or 50.
    --gpus GPU[,GPU...]     Physical GPUs. Only 0,1,2,3,4,5 are allowed.
  --datasets LIST         Ordered datasets, comma-separated.
  --methods LIST          Ordered methods, comma-separated.
  --channel-cache PATH    Shared channel cache used by all profiles.

Optional options:
  --model-family NAME     Model family passed to EvalScope.
  --profile-root PATH     Profile directory. Defaults to experiments/profiles/reap_${RATIO_TAG}_screening.
  --results-root PATH     Result directory. Defaults to a timestamped directory under experiments/results.
  --dataset-args JSON     EvalScope dataset args. Defaults to local benchmark paths on this machine.
    --dataset-limits JSON   Per-dataset limits, e.g. '{"arc":400,"mmlu_pro":100}'.
    --sandbox JSON           Sandbox configuration passed to EvalScope for code benchmarks.
  --generation-config JSON Generation config. Defaults to {"max_tokens":1024}.
  --limit VALUE            Per-subset sample limit, e.g. 100 or 0.1.
  --eval-batch-size N     EvalScope batch size. Defaults to 1.
  --seed N                 Random seed. Defaults to 42.
  --correction-mode NAME  Static profile correction mode. Defaults to none.
  --max-correction-ratio N Maximum correction ratio. Defaults to 0.20.
  --moe-backend NAME       torch or torch_index_add. Defaults to torch_index_add.
  --python PATH            Python executable. Defaults to the xhquant environment.
  --preflight-only         Validate and prepare each method without evaluation.
  --dry-run                Print assignments and commands without running them.
  -h, --help               Show this help.

Methods:
    dense, enp, tenp, aimer, pure_pseudo, wick_kernel, wick_kernel_merge,
    wick_pseudo_protect, wick_pseudo_protect_merge, official_reap, route_tail_global, route_tail_per_layer,
  tail_risk_global, tail_risk_per_layer

Scheduling:
  Methods are assigned round-robin to the selected GPUs. Methods assigned to
  the same GPU run sequentially. Each method receives datasets in the supplied
  order, and EvalScope evaluates those datasets sequentially.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "Missing value for $1."
}

normalize_ratio_tag() {
    local value="$1"
    if [[ "$value" =~ ^[0-9]+pct$ ]]; then
        printf '%s\n' "$value"
        return
    fi
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%spct\n' "$value"
        return
    fi
    if [[ "$value" =~ ^0\.[0-9]+$ ]]; then
        local percent
        percent="$(awk -v ratio="$value" 'BEGIN { printf "%g", ratio * 100 }')"
        printf '%spct\n' "$percent"
        return
    fi
    die "Invalid pruning ratio '$value'; use values such as 50pct, 0.5, or 50."
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

split_csv() {
    local value="$1"
    local -n output_array="$2"
    local item
    IFS=',' read -r -a output_array <<< "$value"
    [[ ${#output_array[@]} -gt 0 ]] || die "Comma-separated value cannot be empty."
    for index in "${!output_array[@]}"; do
        item="$(trim "${output_array[$index]}")"
        [[ -n "$item" ]] || die "Comma-separated value contains an empty item."
        output_array[$index]="$item"
    done
}

profile_path_for_method() {
    local method="$1"
    case "$method" in
        dense)
            printf '%s/dense_full_width.pt\n' "$PROFILE_ROOT"
            ;;
        enp)
            printf '%s/enp_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        tenp)
            printf '%s/tenp_%s_trapezoid.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        aimer)
            printf '%s/aimer_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        pure_pseudo)
            printf '%s/pure_pseudo_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        wick_kernel|wick_kernel_merge|wick_pseudo_protect|wick_pseudo_protect_merge)
            printf '%s/%s_%s_per_layer.pt\n' "$PROFILE_ROOT" "$method" "$RATIO_TAG"
            ;;
        official_reap)
            printf '%s/reap_official_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        route_tail_global)
            printf '%s/route_tail_%s_global.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        route_tail_per_layer)
            printf '%s/route_tail_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        tail_risk_global)
            printf '%s/tail_risk_%s_global.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        tail_risk_per_layer)
            printf '%s/tail_risk_%s_per_layer.pt\n' "$PROFILE_ROOT" "$RATIO_TAG"
            ;;
        *)
            die "Unknown method '$method'."
            ;;
    esac
}

merge_plan_path_for_method() {
    local method="$1"
    case "$method" in
        wick_kernel_merge)
            printf '%s/wick_kernel_merge_plan.pt\n' "$PROFILE_ROOT"
            ;;
        wick_pseudo_protect_merge)
            printf '%s/wick_pseudo_protect_merge_plan.pt\n' "$PROFILE_ROOT"
            ;;
        *)
            printf '\n'
            ;;
    esac
}

file_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

build_command() {
    local gpu="$1"
    local method="$2"
    local profile="$3"
    local profile_sha256="$4"
    local channel_sha256="$5"
    local merge_plan
    local merge_plan_sha256
    local model_instance_id="${MODEL_ID}-${RATIO_TAG}-${method}"
    local work_dir="$RESULTS_ROOT/$method"
    local -n output_command="$6"

    output_command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "LD_LIBRARY_PATH=$EXTRA_LD_LIBRARY_PATH"
        "NUMEXPR_MAX_THREADS=16"
        "PYTHONPATH=$ROOT:$EXPERIMENT_ROOT/code"
        "$PYTHON_BIN"
        "$RUNNER"
        --model-path "$MODEL_PATH"
        --model-id "$model_instance_id"
        --profile "$profile"
        --channel-cache "$CHANNEL_CACHE"
        --expected-profile-file-sha256 "$profile_sha256"
        --expected-channel-file-sha256 "$channel_sha256"
        --work-dir "$work_dir"
        --datasets "${DATASETS[@]}"
        --dataset-args "$DATASET_ARGS"
        --generation-config "$GENERATION_CONFIG"
        --eval-batch-size "$EVAL_BATCH_SIZE"
        --seed "$SEED"
        --correction-mode "$CORRECTION_MODE"
        --max-correction-ratio "$MAX_CORRECTION_RATIO"
        --moe-backend "$MOE_BACKEND"
        --no-enable-thinking
        --no-timestamp
    )
    merge_plan="$(merge_plan_path_for_method "$method")"
    if [[ -n "$merge_plan" ]]; then
        [[ -f "$merge_plan" ]] || die "Merge plan does not exist for method '$method': $merge_plan"
        merge_plan_sha256="$(file_sha256 "$merge_plan")"
        output_command+=(
            --merge-plan "$merge_plan"
            --expected-merge-plan-file-sha256 "$merge_plan_sha256"
        )
    fi
    if [[ -n "$MODEL_FAMILY" ]]; then
        output_command+=(--model-family "$MODEL_FAMILY")
    fi
    if [[ -n "$LIMIT" ]]; then
        output_command+=(--limit "$LIMIT")
    fi
    if [[ -n "$DATASET_LIMITS" ]]; then
        output_command+=(--dataset-limits "$DATASET_LIMITS")
    fi
    if [[ -n "$SANDBOX_CONFIG" ]]; then
        output_command+=(--sandbox "$SANDBOX_CONFIG")
    fi
    if [[ "$PRELIGHT_ONLY" == "true" ]]; then
        output_command+=(--preflight-only)
    fi
}

print_command() {
    local -n command_to_print="$1"
    printf '%q ' "${command_to_print[@]}"
    printf '\n'
}

run_method() {
    local gpu="$1"
    local method="$2"
    local profile
    local profile_sha256
    local channel_sha256
    local -a command_array

    profile="$(profile_path_for_method "$method")"
    [[ -f "$profile" ]] || die "Profile does not exist for method '$method': $profile"
    profile_sha256="$(file_sha256 "$profile")"
    channel_sha256="$(file_sha256 "$CHANNEL_CACHE")"
    build_command "$gpu" "$method" "$profile" "$profile_sha256" "$channel_sha256" command_array
    echo "Starting method '$method' on physical GPU $gpu."
    print_command command_array
    "${command_array[@]}"
}

run_worker() {
    local gpu="$1"
    shift
    local method
    for method in "$@"; do
        run_method "$gpu" "$method"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            require_value "$1" "${2:-}"
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-id)
            require_value "$1" "${2:-}"
            MODEL_ID="$2"
            shift 2
            ;;
        --model-family)
            require_value "$1" "${2:-}"
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --pruning-ratio)
            require_value "$1" "${2:-}"
            PRUNING_RATIO="$2"
            shift 2
            ;;
        --gpus)
            require_value "$1" "${2:-}"
            GPUS_CSV="$2"
            shift 2
            ;;
        --datasets)
            require_value "$1" "${2:-}"
            DATASETS_CSV="$2"
            shift 2
            ;;
        --methods)
            require_value "$1" "${2:-}"
            METHODS_CSV="$2"
            shift 2
            ;;
        --profile-root)
            require_value "$1" "${2:-}"
            PROFILE_ROOT="$2"
            shift 2
            ;;
        --channel-cache)
            require_value "$1" "${2:-}"
            CHANNEL_CACHE="$2"
            shift 2
            ;;
        --results-root)
            require_value "$1" "${2:-}"
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --dataset-args)
            require_value "$1" "${2:-}"
            DATASET_ARGS="$2"
            shift 2
            ;;
        --dataset-limits)
            require_value "$1" "${2:-}"
            DATASET_LIMITS="$2"
            shift 2
            ;;
        --sandbox)
            require_value "$1" "${2:-}"
            SANDBOX_CONFIG="$2"
            shift 2
            ;;
        --generation-config)
            require_value "$1" "${2:-}"
            GENERATION_CONFIG="$2"
            shift 2
            ;;
        --limit)
            require_value "$1" "${2:-}"
            LIMIT="$2"
            shift 2
            ;;
        --eval-batch-size)
            require_value "$1" "${2:-}"
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2:-}"
            SEED="$2"
            shift 2
            ;;
        --correction-mode)
            require_value "$1" "${2:-}"
            CORRECTION_MODE="$2"
            shift 2
            ;;
        --max-correction-ratio)
            require_value "$1" "${2:-}"
            MAX_CORRECTION_RATIO="$2"
            shift 2
            ;;
        --moe-backend)
            require_value "$1" "${2:-}"
            MOE_BACKEND="$2"
            shift 2
            ;;
        --python)
            require_value "$1" "${2:-}"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --preflight-only)
            PRELIGHT_ONLY="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option '$1'. Use --help for usage."
            ;;
    esac
done

[[ -n "$MODEL_PATH" ]] || die "--model-path is required."
[[ -n "$MODEL_ID" ]] || die "--model-id is required."
[[ -n "$PRUNING_RATIO" ]] || die "--pruning-ratio is required."
[[ -n "$GPUS_CSV" ]] || die "--gpus is required."
[[ -n "$DATASETS_CSV" ]] || die "--datasets is required."
[[ -n "$METHODS_CSV" ]] || die "--methods is required."
[[ -n "$CHANNEL_CACHE" ]] || die "--channel-cache is required."

RATIO_TAG="$(normalize_ratio_tag "$PRUNING_RATIO")"
PROFILE_ROOT="${PROFILE_ROOT:-$EXPERIMENT_ROOT/experiments/profiles/reap_${RATIO_TAG}_screening}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/experiments/results/${MODEL_ID}_${RATIO_TAG}_downstream_matrix_$(date +%Y%m%d_%H%M%S)}"
CHANNEL_CACHE="$(cd "$(dirname "$CHANNEL_CACHE")" && pwd)/$(basename "$CHANNEL_CACHE")"
[[ -f "$CHANNEL_CACHE" ]] || die "Channel cache does not exist: $CHANNEL_CACHE"
EXTRA_LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"

split_csv "$GPUS_CSV" GPUS
split_csv "$DATASETS_CSV" DATASETS
split_csv "$METHODS_CSV" METHODS

declare -A seen_gpus=()
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-5]$ ]] || die "Only physical GPU 0-5 are allowed for this downstream matrix; got '$gpu'."
    [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU '$gpu' was specified more than once."
    seen_gpus[$gpu]=1
done

[[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
[[ -f "$RUNNER" ]] || die "EvalScope runner does not exist: $RUNNER"
[[ "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "--eval-batch-size must be a positive integer."
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer."

declare -a worker_methods=()
for index in "${!GPUS[@]}"; do
    worker_methods[$index]=""
done
for index in "${!METHODS[@]}"; do
    method="${METHODS[$index]}"
    profile_path_for_method "$method" >/dev/null
    worker_index=$((index % ${#GPUS[@]}))
    if [[ -n "${worker_methods[$worker_index]}" ]]; then
        worker_methods[$worker_index]+=" "
    fi
    worker_methods[$worker_index]+="$method"
done

echo "Model: $MODEL_PATH"
echo "Ratio: $RATIO_TAG"
echo "Datasets (sequential): ${DATASETS[*]}"
echo "Methods: ${METHODS[*]}"
echo "GPUs (parallel workers): ${GPUS[*]}"
echo "Results root: $RESULTS_ROOT"

if [[ "$DRY_RUN" == "true" ]]; then
    for index in "${!GPUS[@]}"; do
        gpu="${GPUS[$index]}"
        IFS=' ' read -r -a assigned_methods <<< "${worker_methods[$index]}"
        echo "GPU $gpu methods: ${assigned_methods[*]}"
        for method in "${assigned_methods[@]}"; do
            profile="$(profile_path_for_method "$method")"
            [[ -f "$profile" ]] || die "Profile does not exist for method '$method': $profile"
            profile_sha256="$(file_sha256 "$profile")"
            channel_sha256="$(file_sha256 "$CHANNEL_CACHE")"
            build_command "$gpu" "$method" "$profile" "$profile_sha256" "$channel_sha256" command_array
            print_command command_array
        done
    done
    exit 0
fi

mkdir -p "$RESULTS_ROOT/launcher_logs"
declare -a worker_pids=()
for index in "${!GPUS[@]}"; do
    gpu="${GPUS[$index]}"
    IFS=' ' read -r -a assigned_methods <<< "${worker_methods[$index]}"
    run_worker "$gpu" "${assigned_methods[@]}" 2>&1 | tee "$RESULTS_ROOT/launcher_logs/gpu_${gpu}.log" &
    worker_pids+=("$!")
done

failed=0
for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

exit "$failed"