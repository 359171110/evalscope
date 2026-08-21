#!/usr/bin/env bash
# Build, export, and evaluate Geom pruning for one model at 50% then 25%.
# Usage: run_one_model_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
_GEOM_RESULT_ROOT="${RESULT_ROOT:-/home/xinpeigao/evalscope/results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_GEOM_RESULT_ROOT"
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
        ;;
    gemma4)
        NAME="Gemma4-26B-A4B"
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        WIDTH_50=352
        WIDTH_25=512
        ;;
    qwen36)
        NAME="Qwen3.6-35B-A3B"
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        WIDTH_50=256
        WIDTH_25=384
        ;;
    deepseek)
        NAME="DeepSeek-V2-Lite-Chat"
        MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}"
        WIDTH_50=704
        WIDTH_25=1056
        ;;
    *) die "Unknown model '$MODEL'." ;;
esac

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/geom/$MODEL}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
mkdir -p "$ARTIFACT_ROOT"

if [[ "$MODEL" == qwen36 ]]; then
    # H20 + this vLLM build: FlashInfer GDN TMA crashes with cudaErrorUnknown (999).
    # shellcheck disable=SC1091
    source /data/xinpeigao/evalscope_results/_launchers/qwen36.env
fi

width_for() {
    case "$1" in
        50) printf '%s\n' "$WIDTH_50" ;;
        25) printf '%s\n' "$WIDTH_25" ;;
        *) die "RATIO must be 25 or 50." ;;
    esac
}

build_profile() {
    local ratio="$1"
    local width profile
    width="$(width_for "$ratio")"
    profile="$ARTIFACT_ROOT/geom_${ratio}pct_per_layer.pt"
    if [[ -f "$profile" && -f "$ARTIFACT_ROOT/geom_rankings.pt" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio width=$width"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Geom.build_geom_artifacts \
        --model-path "$MODEL_PATH" \
        --output-channel-cache "$ARTIFACT_ROOT/geom_rankings.pt" \
        --output-profile "$profile" \
        --retained-channels "$width"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/geom_${ratio}pct_per_layer.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        printf '%s\n' "$checkpoint"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Geom.export_geom_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$ARTIFACT_ROOT/geom_rankings.pt" \
        --output-dir "$checkpoint"
    printf '%s\n' "$checkpoint"
}

run_ratio() {
    local ratio="$1"
    local checkpoint experiment_dir
    build_profile "$ratio"
    checkpoint="$(export_checkpoint "$ratio" | tail -n 1)"
    experiment_dir="$RESULT_ROOT/${NAME}_${ratio}_vllm_CalibrationFree_full8_v1_Geom_${TIMESTAMP}_${SEED}"
    echo "[$(date -Is)] EVAL $MODEL ratio=$ratio gpu=$GPU port=$PORT"
    if [[ -d "$experiment_dir" ]]; then
        echo "resume existing $experiment_dir"
        RESULT_ROOT="$RESULT_ROOT" METHOD=Geom MASTER_PORT="$((29200 + ${GPU%%,*}))" \
            bash "$RESUME_LAUNCHER" \
            "$experiment_dir" \
            "$checkpoint" \
            "$GPU" \
            "$PORT" \
            1
        return
    fi
    RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" MASTER_PORT="$((29200 + ${GPU%%,*}))" \
        bash "$EVAL_LAUNCHER" \
        "$NAME" \
        "$checkpoint" \
        "$GPU" \
        "$PORT" \
        1 \
        Geom \
        "$ratio" \
        CalibrationFree \
        full8_v1
}

echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP result_root=$RESULT_ROOT"
status=0
IFS=' ' read -r -a ratio_list <<< "${RATIOS:-50 25}"
for ratio in "${ratio_list[@]}"; do
    if ! run_ratio "$ratio"; then
        echo "[$(date -Is)] FAILED $MODEL ratio=$ratio; continuing to remaining ratios" >&2
        status=1
    fi
done
echo "[$(date -Is)] ALL DONE $MODEL status=$status"
exit "$status"
