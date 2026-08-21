#!/usr/bin/env bash

set -euo pipefail

WANDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$WANDA_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data/xinpeigao/conda_envs/gemma4-vllm-cu128/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data/xinpeigao/conda_envs/gemma4-vllm-cu128/bin/python}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RATIO="${RATIO:-50}"
SEED="${SEED:-42}"
MODEL="${1:-}"
ACTION="${2:-dry-run}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

validate_gpu() {
    local gpu="$1"
    local part
    IFS=',' read -r -a parts <<< "$gpu"
    [[ "${#parts[@]}" -ge 1 ]] || die "GPU list must not be empty."
    for part in "${parts[@]}"; do
        [[ "$part" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer; got '$part'."
    done
}

model_field() {
    local model="$1"
    local field="$2"
    case "$model:$field" in
        qwen3:name) printf 'Qwen330BA3BInstruct\n' ;;
        qwen3:path) printf '%s\n' "${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}" ;;
        qwen3:gpu) printf '%s\n' "${QWEN3_GPU:-2}" ;;
        qwen3:port) printf '%s\n' "${QWEN3_PORT:-18580}" ;;
        qwen3:width)
            case "$RATIO" in
                50) printf '384\n' ;;
                25) printf '576\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen3." ;;
            esac
            ;;
        qwen3:protocol) printf 'qwen3_wikitext128x2048_v1\n' ;;
        gemma4:name) printf 'Gemma4-26B-A4B\n' ;;
        gemma4:path) printf '%s\n' "${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}" ;;
        gemma4:gpu) printf '%s\n' "${GEMMA4_GPU:-3}" ;;
        gemma4:port) printf '%s\n' "${GEMMA4_PORT:-18581}" ;;
        gemma4:width)
            case "$RATIO" in
                50) printf '352\n' ;;
                25) printf '512\n' ;;
                *) die "Unsupported RATIO=$RATIO for gemma4." ;;
            esac
            ;;
        gemma4:protocol) printf 'gemma4_wikitext128x2048_v1\n' ;;
        qwen36:name) printf 'Qwen3.6-35B-A3B\n' ;;
        qwen36:path) printf '%s\n' "${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}" ;;
        qwen36:gpu) printf '%s\n' "${QWEN36_GPU:-4}" ;;
        qwen36:port) printf '%s\n' "${QWEN36_PORT:-18582}" ;;
        qwen36:width)
            case "$RATIO" in
                50) printf '256\n' ;;
                25) printf '384\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen36." ;;
            esac
            ;;
        qwen36:protocol) printf 'qwen36_wikitext128x2048_v1\n' ;;
        deepseek:name) printf 'DeepSeek-V2-Lite-Chat\n' ;;
        deepseek:path) printf '%s\n' "${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}" ;;
        deepseek:gpu) printf '%s\n' "${DEEPSEEK_GPU:-5}" ;;
        deepseek:port) printf '%s\n' "${DEEPSEEK_PORT:-18583}" ;;
        deepseek:width)
            case "$RATIO" in
                50) printf '704\n' ;;
                25) printf '1056\n' ;;
                *) die "Unsupported RATIO=$RATIO for deepseek." ;;
            esac
            ;;
        deepseek:protocol) printf 'deepseek_v2_lite_chat_wikitext128x2048_v1\n' ;;
        *:cache) printf '%s\n' "/data/xinpeigao/evalscope_results/_artifacts/wanda/${model}/calibration.pt" ;;
        *:artifact) printf '%s\n' "/data/xinpeigao/evalscope_results/_artifacts/wanda/${model}" ;;
        *) die "Unknown model/field '$model:$field'." ;;
    esac
}

experiment_dir() {
    local model="$1"
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}" MODEL_NAME="$(model_field "$model" name)" \
        "$CODE_ROOT/scripts/create_result_dir.sh" \
        --model "$(model_field "$model" name)" \
        --inference vllm \
        --calibration WikiText128x2048 \
        --protocol full8_v1 \
        --method Wanda \
        --pruning-ratio-label "$RATIO" \
        --pruning-ratio-percent "$RATIO" \
        --timestamp "$TIMESTAMP" \
        --dry-run
}

tensor_parallel_size() {
    local gpu="$1"
    local count
    IFS=',' read -r -a parts <<< "$gpu"
    count="${#parts[@]}"
    printf '%s\n' "$count"
}

calibration_cache_for() {
    local model="$1"
    local cache
    cache="$(model_field "$model" cache)"
    if [[ "$model" == deepseek && -f /data/xinpeigao/evalscope_results/_calibration/deepseek_v2_lite_chat_wikitext_train_128x2048.pt ]]; then
        printf '%s\n' "/data/xinpeigao/evalscope_results/_calibration/deepseek_v2_lite_chat_wikitext_train_128x2048.pt"
        return
    fi
    printf '%s\n' "$cache"
}

build_cache() {
    local model="$1"
    local cache protocol model_path
    cache="$(calibration_cache_for "$model")"
    protocol="$(model_field "$model" protocol)"
    model_path="$(model_field "$model" path)"
    if [[ -f "$cache" ]]; then
        echo "cache already exists: $cache"
        return
    fi
    [[ -d "$model_path" ]] || die "Model path does not exist: $model_path"
    [[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
    mkdir -p "$(dirname "$cache")"
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "PYTHONNOUSERSITE=1"
        "HF_HOME=${HF_HOME:-/data1/xinpeigao/caches/huggingface}"
        "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data1/xinpeigao/caches/huggingface/datasets}"
        "HF_DATASETS_OFFLINE=1"
        "$PYTHON_BIN"
        "$CODE_ROOT/scripts/build_shared_calibration_token_cache.py"
        --model-path "$model_path"
        --output-cache "$cache"
        --dataset Salesforce/wikitext
        --config wikitext-2-raw-v1
        --split train
        --text-field text
        --sequence-length 2048
        --calibration-sequences 128
        --token-offset 0
        --protocol-name "$protocol"
    )
    if [[ -n "${WIKITEXT_TRAIN_ARROW:-}" ]]; then
        [[ -f "$WIKITEXT_TRAIN_ARROW" ]] || die "WikiText train Arrow file is missing: $WIKITEXT_TRAIN_ARROW"
        command+=(--arrow-file "$WIKITEXT_TRAIN_ARROW")
    fi
    print_command "${command[@]}"
    "${command[@]}"
}

collect_statistics() {
    local model="$1"
    local artifact cache gpu log
    artifact="$(model_field "$model" artifact)"
    cache="$(calibration_cache_for "$model")"
    gpu="$(model_field "$model" gpu)"
    log="$artifact/collect.log"
    validate_gpu "$gpu"
    [[ -x "$PYTHON_BIN" ]] || die "Python executable is not executable: $PYTHON_BIN"
    [[ -d "$(model_field "$model" path)" ]] || die "Model path does not exist: $(model_field "$model" path)"
    build_cache "$model"
    mkdir -p "$artifact"
    if [[ -f "$artifact/statistics.pt" ]]; then
        echo "statistics already exist: $artifact/statistics.pt"
        return
    fi
    local -a device_args=(--device-map none --device cuda)
    if [[ "$gpu" == *,* ]]; then
        device_args=(--device-map auto)
    fi
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "PYTHONNOUSERSITE=1"
        "HF_HOME=${HF_HOME:-/data1/xinpeigao/caches/huggingface}"
        "TMPDIR=${TMPDIR:-/data1/xinpeigao/tmp}"
        "$PYTHON_BIN"
        -m Wanda.collect_wanda_statistics
        --model-path "$(model_field "$model" path)"
        --calibration-cache "$cache"
        --output "$artifact/statistics.pt"
        --sequence-length 2048
        --calibration-sequences 128
        --route-weighting mass
        "${device_args[@]}"
    )
    print_command "${command[@]}"
    if ! "${command[@]}" >"$log" 2>&1; then
        tail -100 "$log" >&2
        die "Wanda collect failed: $log"
    fi
    echo "statistics=$artifact/statistics.pt"
}

build_artifacts() {
    local model="$1"
    local artifact profile
    artifact="$(model_field "$model" artifact)"
    profile="$artifact/wanda_${RATIO}pct_per_layer.pt"
    if [[ "$RATIO" == "25" ]]; then
        local source="$artifact/wanda_50pct_per_layer.pt"
        [[ -f "$artifact/channels.pt" ]] || die "Wanda channel cache does not exist: $artifact/channels.pt"
        [[ -f "$source" ]] || die "Need the 50% Wanda profile to clone 25% widths: $source"
        if [[ -f "$profile" ]]; then
            echo "artifacts already exist: $profile"
            return
        fi
        local -a clone_command=(
            env
            "PYTHONPATH=$ROOT:$CODE_ROOT"
            "PYTHONNOUSERSITE=1"
            "$PYTHON_BIN"
            -m Wanda.build_wanda_artifacts
            --from-profile "$source"
            --channel-cache "$artifact/channels.pt"
            --output-profile "$profile"
            --retained-channels "$(model_field "$model" width)"
            --target-pruning-ratio 0.25
        )
        print_command "${clone_command[@]}"
        "${clone_command[@]}"
        return
    fi
    [[ -f "$artifact/statistics.pt" ]] || die "Wanda statistics do not exist: $artifact/statistics.pt"
    if [[ -f "$profile" && -f "$artifact/channels.pt" ]]; then
        echo "artifacts already exist: $profile"
        return
    fi
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "PYTHONNOUSERSITE=1"
        "$PYTHON_BIN"
        -m Wanda.build_wanda_artifacts
        --model-path "$(model_field "$model" path)"
        --statistics "$artifact/statistics.pt"
        --output-channel-cache "$artifact/channels.pt"
        --output-profile "$profile"
        --retained-channels "$(model_field "$model" width)"
    )
    print_command "${command[@]}"
    "${command[@]}"
}

ensure_experiment_dir() {
    local model="$1"
    local directory
    directory="$(experiment_dir "$model")"
    if [[ ! -f "$directory/experiment_manifest.json" ]]; then
        RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}" \
            "$CODE_ROOT/scripts/create_result_dir.sh" \
            --model "$(model_field "$model" name)" \
            --inference vllm \
            --calibration WikiText128x2048 \
            --protocol full8_v1 \
            --method Wanda \
            --pruning-ratio-label "$RATIO" \
            --pruning-ratio-percent "$RATIO" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$directory"
}

export_checkpoint() {
    local model="$1"
    local artifact checkpoint
    artifact="$(model_field "$model" artifact)"
    checkpoint="$artifact/checkpoint_$RATIO"
    [[ -f "$artifact/wanda_${RATIO}pct_per_layer.pt" ]] || die "Wanda profile does not exist: $artifact/wanda_${RATIO}pct_per_layer.pt"
    [[ -f "$artifact/channels.pt" ]] || die "Wanda channel cache does not exist: $artifact/channels.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        printf '%s\n' "$checkpoint"
        return
    fi
    mkdir -p "$checkpoint"
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "PYTHONNOUSERSITE=1"
        "$PYTHON_BIN"
        -m Wanda.export_wanda_checkpoint
        --model-path "$(model_field "$model" path)"
        --profile "$artifact/wanda_${RATIO}pct_per_layer.pt"
        --channel-cache "$artifact/channels.pt"
        --output-dir "$checkpoint"
    )
    print_command "${command[@]}"
    "${command[@]}"
    printf '%s\n' "$checkpoint"
}

run_eval() {
    local model="$1"
    local artifact checkpoint gpu port
    artifact="$(model_field "$model" artifact)"
    checkpoint="$artifact/checkpoint_$RATIO"
    gpu="$(model_field "$model" gpu)"
    port="$(model_field "$model" port)"
    [[ -f "$checkpoint/pruning_export_manifest.json" ]] || die "Checkpoint is not exported: $checkpoint"
    validate_gpu "$gpu"
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}" TIMESTAMP="$TIMESTAMP" \
        bash "${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}" \
        "$(model_field "$model" name)" \
        "$checkpoint" \
        "$gpu" \
        "$port" \
        "$(tensor_parallel_size "$gpu")" \
        Wanda \
        "$RATIO" \
        WikiText128x2048 \
        full8_v1
}

dry_run_one() {
    local model="$1"
    echo "model=$model"
    echo "target_model=$(model_field "$model" name)"
    echo "model_path=$(model_field "$model" path)"
    echo "collect_python=$PYTHON_BIN"
    echo "gpu=$(model_field "$model" gpu)"
    echo "port=$(model_field "$model" port)"
    echo "ratio=$RATIO"
    echo "seed=$SEED"
    echo "retained_channels=$(model_field "$model" width)"
    echo "calibration=WikiText128x2048"
    echo "protocol=full8_v1"
    echo "method=Wanda"
    echo "cache=$(calibration_cache_for "$model")"
    echo "artifact_root=$(model_field "$model" artifact)"
    echo "experiment=$(experiment_dir "$model")"
}

run_one() {
    local model="$1"
    local action="$2"
    case "$action" in
        dry-run) dry_run_one "$model" ;;
        cache) build_cache "$model" ;;
        collect) collect_statistics "$model" ;;
        build) build_artifacts "$model" ;;
        export) export_checkpoint "$model" ;;
        eval) run_eval "$model" ;;
        prepare)
            collect_statistics "$model"
            build_artifacts "$model"
            export_checkpoint "$model"
            ;;
        *) die "Usage: $0 qwen3|gemma4|qwen36|deepseek|all dry-run|cache|collect|build|export|eval|prepare" ;;
    esac
}

[[ -n "$MODEL" ]] || die "Usage: RATIO=25|50 $0 qwen3|gemma4|qwen36|deepseek|all dry-run|cache|collect|build|export|eval|prepare"
[[ "$RATIO" == "25" || "$RATIO" == "50" ]] || die "RATIO must be 25 or 50; got '$RATIO'."
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer; got '$SEED'."
[[ -x "$CODE_ROOT/scripts/create_result_dir.sh" ]] || die "Framework result helper is missing."

if [[ "$MODEL" == all ]]; then
    for item in qwen3 gemma4 qwen36 deepseek; do
        run_one "$item" "$ACTION"
    done
    exit 0
fi

[[ "$MODEL" == qwen3 || "$MODEL" == gemma4 || "$MODEL" == qwen36 || "$MODEL" == deepseek ]] ||
    die "Model must be qwen3, gemma4, qwen36, deepseek, or all."
run_one "$MODEL" "$ACTION"
