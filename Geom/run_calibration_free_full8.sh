#!/usr/bin/env bash

set -euo pipefail

GEOM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$GEOM_ROOT/.." && pwd)"
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
        qwen3:port) printf '%s\n' "${QWEN3_PORT:-18980}" ;;
        qwen3:width)
            case "$RATIO" in
                50) printf '384\n' ;;
                25) printf '576\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen3." ;;
            esac
            ;;
        qwen3:vllm_extra) printf '\n' ;;
        gemma4:name) printf 'Gemma4-26B-A4B\n' ;;
        gemma4:path) printf '%s\n' "${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}" ;;
        gemma4:gpu) printf '%s\n' "${GEMMA4_GPU:-3}" ;;
        gemma4:port) printf '%s\n' "${GEMMA4_PORT:-18981}" ;;
        gemma4:width)
            case "$RATIO" in
                50) printf '352\n' ;;
                25) printf '512\n' ;;
                *) die "Unsupported RATIO=$RATIO for gemma4." ;;
            esac
            ;;
        gemma4:vllm_extra) printf '\n' ;;
        qwen36:name) printf 'Qwen3.6-35B-A3B\n' ;;
        qwen36:path) printf '%s\n' "${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}" ;;
        qwen36:gpu) printf '%s\n' "${QWEN36_GPU:-4}" ;;
        qwen36:port) printf '%s\n' "${QWEN36_PORT:-18982}" ;;
        qwen36:width)
            case "$RATIO" in
                50) printf '256\n' ;;
                25) printf '384\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen36." ;;
            esac
            ;;
        qwen36:vllm_extra) printf '\n' ;;
        deepseek:name) printf 'DeepSeek-V2-Lite-Chat\n' ;;
        deepseek:path) printf '%s\n' "${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}" ;;
        deepseek:gpu) printf '%s\n' "${DEEPSEEK_GPU:-5}" ;;
        deepseek:port) printf '%s\n' "${DEEPSEEK_PORT:-18983}" ;;
        deepseek:width)
            case "$RATIO" in
                50) printf '704\n' ;;
                25) printf '1056\n' ;;
                *) die "Unsupported RATIO=$RATIO for deepseek." ;;
            esac
            ;;
        deepseek:vllm_extra) printf '%s\n' "--trust-remote-code" ;;
        *:artifact) printf '%s\n' "/data/xinpeigao/evalscope_results/_artifacts/geom/${model}" ;;
        *) die "Unknown model/field '$model:$field'." ;;
    esac
}

experiment_dir() {
    local model="$1"
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}" MODEL_NAME="$(model_field "$model" name)" \
        "$CODE_ROOT/scripts/create_result_dir.sh" \
        --model "$(model_field "$model" name)" \
        --inference vllm \
        --calibration CalibrationFree \
        --protocol full8_v1 \
        --method Geom \
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

build_artifacts() {
    local model="$1"
    local artifact profile
    artifact="$(model_field "$model" artifact)"
    profile="$artifact/geom_${RATIO}pct_per_layer.pt"
    [[ -d "$(model_field "$model" path)" ]] || die "Model path does not exist: $(model_field "$model" path)"
    mkdir -p "$artifact"
    if [[ -f "$profile" && -f "$artifact/geom_rankings.pt" ]]; then
        echo "artifacts already exist: $profile"
        return
    fi
    local -a command=(
        env
        "PYTHONPATH=$ROOT:$CODE_ROOT"
        "PYTHONNOUSERSITE=1"
        "$PYTHON_BIN"
        -m Geom.build_geom_artifacts
        --model-path "$(model_field "$model" path)"
        --output-channel-cache "$artifact/geom_rankings.pt"
        --output-profile "$profile"
        --retained-channels "$(model_field "$model" width)"
    )
    print_command "${command[@]}"
    "${command[@]}"
}

export_checkpoint() {
    local model="$1"
    local artifact checkpoint
    artifact="$(model_field "$model" artifact)"
    checkpoint="$artifact/checkpoint_$RATIO"
    [[ -f "$artifact/geom_${RATIO}pct_per_layer.pt" ]] || die "Geom profile does not exist: $artifact/geom_${RATIO}pct_per_layer.pt"
    [[ -f "$artifact/geom_rankings.pt" ]] || die "Geom channel cache does not exist: $artifact/geom_rankings.pt"
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
        -m Geom.export_geom_checkpoint
        --model-path "$(model_field "$model" path)"
        --profile "$artifact/geom_${RATIO}pct_per_layer.pt"
        --channel-cache "$artifact/geom_rankings.pt"
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
        MASTER_PORT="$((29200 + ${gpu%%,*}))" \
        bash "${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}" \
        "$(model_field "$model" name)" \
        "$checkpoint" \
        "$gpu" \
        "$port" \
        "$(tensor_parallel_size "$gpu")" \
        Geom \
        "$RATIO" \
        CalibrationFree \
        full8_v1
}

dry_run_one() {
    local model="$1"
    echo "model=$model"
    echo "target_model=$(model_field "$model" name)"
    echo "model_path=$(model_field "$model" path)"
    echo "gpu=$(model_field "$model" gpu)"
    echo "port=$(model_field "$model" port)"
    echo "ratio=$RATIO"
    echo "seed=$SEED"
    echo "retained_channels=$(model_field "$model" width)"
    echo "calibration=CalibrationFree"
    echo "protocol=full8_v1"
    echo "method=Geom"
    echo "artifact_root=$(model_field "$model" artifact)"
    echo "experiment=$(experiment_dir "$model")"
}

run_one() {
    local model="$1"
    local action="$2"
    case "$action" in
        dry-run) dry_run_one "$model" ;;
        build) build_artifacts "$model" ;;
        export) export_checkpoint "$model" ;;
        eval) run_eval "$model" ;;
        prepare)
            build_artifacts "$model"
            export_checkpoint "$model"
            ;;
        *) die "Usage: $0 qwen3|gemma4|qwen36|deepseek|all dry-run|build|export|eval|prepare" ;;
    esac
}

[[ -n "$MODEL" ]] || die "Usage: RATIO=25|50 $0 qwen3|gemma4|qwen36|deepseek|all dry-run|build|export|eval|prepare"
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
