#!/usr/bin/env bash
# Collect WikiText Wanda statistics, export, and evaluate one model at 50% then 25%.
# Usage: run_one_model_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
# Keep the caller RESULT_ROOT (or this-server eval tree) after env.sh, which defaults to /data.
_WANDA_RESULT_ROOT="${RESULT_ROOT:-/home/xinpeigao/evalscope/results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_WANDA_RESULT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
SEED="${SEED:-42}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$GPU" && -n "$PORT" ]] || die "Usage: $0 qwen3|gemma4|qwen36|deepseek GPU PORT"

case "$MODEL" in
    qwen3)
        NAME="Qwen330BA3BInstruct"
        MODEL_PATH="${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}"
        WIDTH_50=384
        WIDTH_25=576
        PROTOCOL_NAME="qwen3_wikitext128x2048_v1"
        ;;
    gemma4)
        NAME="Gemma4-26B-A4B"
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        WIDTH_50=352
        WIDTH_25=512
        PROTOCOL_NAME="gemma4_wikitext128x2048_v1"
        ;;
    qwen36)
        NAME="Qwen3.6-35B-A3B"
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        WIDTH_50=256
        WIDTH_25=384
        PROTOCOL_NAME="qwen36_wikitext128x2048_v1"
        ;;
    deepseek)
        NAME="DeepSeek-V2-Lite-Chat"
        MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}"
        WIDTH_50=704
        WIDTH_25=1056
        PROTOCOL_NAME="deepseek_v2_lite_chat_wikitext128x2048_v1"
        ;;
    *) die "Unknown model '$MODEL'." ;;
esac

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/wanda/$MODEL}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
LEGACY_DEEPSEEK_WANDA="/data/xinpeigao/evalscope_results/_artifacts/deepseek_v2_lite_chat/wanda"
LEGACY_DEEPSEEK_CACHE="/data/xinpeigao/evalscope_results/_calibration/deepseek_v2_lite_chat_wikitext_train_128x2048.pt"
mkdir -p "$ARTIFACT_ROOT"

if [[ "$MODEL" == qwen36 ]]; then
    # H20 + this vLLM build: FlashInfer GDN TMA crashes with cudaErrorUnknown (999).
    # shellcheck disable=SC1091
    source /data/xinpeigao/evalscope_results/_launchers/qwen36.env
fi

ENV_PREFIX="${VLLM_ENV:-/data/xinpeigao/conda_envs/gemma4-vllm-cu128}"
NVIDIA_LIB="$(find "$ENV_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB}${ENV_PREFIX}/lib64:${ENV_PREFIX}/lib:${ENV_PREFIX}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

width_for() {
    case "$1" in
        50) printf '%s\n' "$WIDTH_50" ;;
        25) printf '%s\n' "$WIDTH_25" ;;
        *) die "RATIO must be 25 or 50." ;;
    esac
}

calibration_cache() {
    if [[ "$MODEL" == deepseek && -f "$LEGACY_DEEPSEEK_CACHE" ]]; then
        printf '%s\n' "$LEGACY_DEEPSEEK_CACHE"
        return
    fi
    printf '%s\n' "$ARTIFACT_ROOT/calibration.pt"
}

seed_existing_deepseek_artifacts() {
    [[ "$MODEL" == deepseek ]] || return 0
    [[ -d "$LEGACY_DEEPSEEK_WANDA" ]] || return 0
    local name
    for name in statistics.pt channels.pt wanda_50pct_per_layer.pt wanda_25pct_per_layer.pt \
        wanda_50pct_per_layer.json wanda_25pct_per_layer.json; do
        if [[ ! -f "$ARTIFACT_ROOT/$name" && -f "$LEGACY_DEEPSEEK_WANDA/$name" ]]; then
            cp -a "$LEGACY_DEEPSEEK_WANDA/$name" "$ARTIFACT_ROOT/$name"
            echo "reused DeepSeek Wanda artifact $name"
        fi
    done
}

build_cache() {
    local cache
    cache="$(calibration_cache)"
    if [[ -f "$cache" ]]; then
        echo "skip cache $MODEL"
        return
    fi
    echo "[$(date -Is)] CACHE $MODEL -> $cache"
    mkdir -p "$(dirname "$cache")"
    local -a command=(
        env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1
        HF_HOME="${HF_HOME:-/data1/xinpeigao/caches/huggingface}"
        HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data1/xinpeigao/caches/huggingface/datasets}"
        HF_DATASETS_OFFLINE=1
        "$VLLM_PYTHON"
        "$CODE_ROOT/scripts/build_shared_calibration_token_cache.py"
        --model-path "$MODEL_PATH"
        --output-cache "$cache"
        --dataset Salesforce/wikitext
        --config wikitext-2-raw-v1
        --split train
        --text-field text
        --sequence-length 2048
        --calibration-sequences 128
        --token-offset 0
        --protocol-name "$PROTOCOL_NAME"
    )
    if [[ -n "${WIKITEXT_TRAIN_ARROW:-}" ]]; then
        [[ -f "$WIKITEXT_TRAIN_ARROW" ]] || die "WikiText train Arrow file is missing: $WIKITEXT_TRAIN_ARROW"
        command+=(--arrow-file "$WIKITEXT_TRAIN_ARROW")
    fi
    "${command[@]}"
}

collect_statistics() {
    local cache
    cache="$(calibration_cache)"
    if [[ -f "$ARTIFACT_ROOT/statistics.pt" ]]; then
        echo "skip collect $MODEL"
        return
    fi
    echo "[$(date -Is)] COLLECT $MODEL gpu=$GPU"
    local -a device_args=(--device-map none --device cuda)
    if [[ "$GPU" == *,* ]]; then
        device_args=(--device-map auto)
    fi
    if ! env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        HF_HOME="${HF_HOME:-/data1/xinpeigao/caches/huggingface}" TMPDIR="$TMPDIR" \
        "$VLLM_PYTHON" -m Wanda.collect_wanda_statistics \
        --model-path "$MODEL_PATH" \
        --calibration-cache "$cache" \
        --output "$ARTIFACT_ROOT/statistics.pt" \
        --sequence-length 2048 \
        --calibration-sequences 128 \
        --route-weighting mass \
        "${device_args[@]}" \
        >"$ARTIFACT_ROOT/collect.log" 2>&1; then
        tail -100 "$ARTIFACT_ROOT/collect.log" >&2
        die "Wanda collect failed: $ARTIFACT_ROOT/collect.log"
    fi
}

build_profile() {
    local ratio="$1"
    local width profile
    width="$(width_for "$ratio")"
    profile="$ARTIFACT_ROOT/wanda_${ratio}pct_per_layer.pt"
    if [[ -f "$profile" && -f "$ARTIFACT_ROOT/channels.pt" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio width=$width"
    if [[ "$ratio" == "25" ]]; then
        [[ -f "$ARTIFACT_ROOT/wanda_50pct_per_layer.pt" ]] || die "Need the 50% Wanda profile to clone 25% widths."
        env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
            "$VLLM_PYTHON" -m Wanda.build_wanda_artifacts \
            --from-profile "$ARTIFACT_ROOT/wanda_50pct_per_layer.pt" \
            --channel-cache "$ARTIFACT_ROOT/channels.pt" \
            --output-profile "$profile" \
            --retained-channels "$width" \
            --target-pruning-ratio 0.25
        return
    fi
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Wanda.build_wanda_artifacts \
        --model-path "$MODEL_PATH" \
        --statistics "$ARTIFACT_ROOT/statistics.pt" \
        --output-channel-cache "$ARTIFACT_ROOT/channels.pt" \
        --output-profile "$profile" \
        --retained-channels "$width"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/wanda_${ratio}pct_per_layer.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        printf '%s\n' "$checkpoint"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Wanda.export_wanda_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$ARTIFACT_ROOT/channels.pt" \
        --output-dir "$checkpoint"
    printf '%s\n' "$checkpoint"
}

run_ratio() {
    local ratio="$1"
    local checkpoint experiment_dir
    build_profile "$ratio"
    checkpoint="$(export_checkpoint "$ratio" | tail -n 1)"
    experiment_dir="$RESULT_ROOT/${NAME}_${ratio}_vllm_WikiText128x2048_full8_v1_Wanda_${TIMESTAMP}_${SEED}"
    echo "[$(date -Is)] EVAL $MODEL ratio=$ratio gpu=$GPU port=$PORT"
    if [[ -d "$experiment_dir" ]]; then
        echo "resume existing $experiment_dir"
        RESULT_ROOT="$RESULT_ROOT" METHOD=Wanda \
            bash "$RESUME_LAUNCHER" \
            "$experiment_dir" \
            "$checkpoint" \
            "$GPU" \
            "$PORT" \
            1
        return
    fi
    RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" \
        bash "$EVAL_LAUNCHER" \
        "$NAME" \
        "$checkpoint" \
        "$GPU" \
        "$PORT" \
        1 \
        Wanda \
        "$ratio" \
        WikiText128x2048 \
        full8_v1
}

seed_existing_deepseek_artifacts
echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP result_root=$RESULT_ROOT"
status=0
build_cache
collect_statistics
IFS=' ' read -r -a ratio_list <<< "${RATIOS:-50 25}"
for ratio in "${ratio_list[@]}"; do
    if ! run_ratio "$ratio"; then
        echo "[$(date -Is)] FAILED $MODEL ratio=$ratio; continuing to remaining ratios" >&2
        status=1
    fi
done
echo "[$(date -Is)] ALL DONE $MODEL status=$status"
exit "$status"
