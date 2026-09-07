#!/usr/bin/env bash
# Build, export, and evaluate HARP-v2 (combo search) for one model at 50% then 25%.
# Reuses v1 ranking caches; writes combo profiles/checkpoints under harp_combo/.
# Usage: run_one_model_harp_v2_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
_HARP_RESULT_ROOT="${RESULT_ROOT:-/data/xinpeigao/evalscope_results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_HARP_RESULT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
SEED="${SEED:-42}"
METHOD_TOKEN="${HARPV2_METHOD_TOKEN:-HARPv2}"
ALLOCATOR="${HARP_ALLOCATOR:-combo}"
GAMMA="${HARP_GAMMA:-2}"
MIN_FRACTION="${HARP_MIN_FRACTION:-0.15}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$GPU" && -n "$PORT" ]] || die "Usage: $0 qwen3|gemma4|qwen36|deepseek|olmoe GPU PORT"
[[ -z "${CSP_CANONICALIZE:-}" ]] || die "HARP-v2 uses raw Layer/Expert/Channel-SP; do not set CSP_CANONICALIZE."
[[ "$ALLOCATOR" == combo ]] || die "HARP-v2 launcher requires ALLOCATOR=combo."
IFS=',' read -r -a _GPU_PARTS <<< "$GPU"
TP="${TP:-${#_GPU_PARTS[@]}}"

case "$MODEL" in
    qwen3)
        NAME="Qwen330BA3BInstruct"
        MODEL_PATH="${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}"
        LOW_50=320
        BUDGET_50=384
        HIGH_50=448
        LOW_25=512
        BUDGET_25=576
        HIGH_25=640
        ;;
    gemma4)
        NAME="Gemma4-26B-A4B"
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        LOW_50=288
        BUDGET_50=352
        HIGH_50=416
        LOW_25=448
        BUDGET_25=512
        HIGH_25=576
        ;;
    qwen36)
        NAME="Qwen3.6-35B-A3B"
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        LOW_50=192
        BUDGET_50=256
        HIGH_50=320
        LOW_25=320
        BUDGET_25=384
        HIGH_25=448
        ;;
    deepseek)
        NAME="DeepSeek-V2-Lite-Chat"
        MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}"
        LOW_50=640
        BUDGET_50=704
        HIGH_50=768
        LOW_25=992
        BUDGET_25=1056
        HIGH_25=1120
        ;;
    olmoe)
        NAME="OLMoE-1B-7B-Instruct"
        MODEL_PATH="${OLMOE_MODEL_PATH:-/data1/xinpeigao/caches/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e}"
        LOW_50=448
        BUDGET_50=512
        HIGH_50=576
        LOW_25=704
        BUDGET_25=768
        HIGH_25=832
        ;;
    *) die "Unknown model '$MODEL'." ;;
esac

low_for() {
    case "$1" in
        50) printf '%s\n' "$LOW_50" ;;
        25) printf '%s\n' "$LOW_25" ;;
        *) die "RATIO must be 25 or 50." ;;
    esac
}

budget_for() {
    case "$1" in
        50) printf '%s\n' "$BUDGET_50" ;;
        25) printf '%s\n' "$BUDGET_25" ;;
        *) die "RATIO must be 25 or 50." ;;
    esac
}

high_for() {
    case "$1" in
        50) printf '%s\n' "$HIGH_50" ;;
        25) printf '%s\n' "$HIGH_25" ;;
        *) die "RATIO must be 25 or 50." ;;
    esac
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "calibration=CalibrationFree"
    echo "protocol=full8_v1"
    echo "method=$METHOD_TOKEN"
    echo "allocator=$ALLOCATOR"
    echo "gamma=$GAMMA"
    echo "min_fraction=$MIN_FRACTION"
    IFS=' ' read -r -a ratio_list <<< "${RATIOS:-50 25}"
    for ratio in "${ratio_list[@]}"; do
        echo "ratio=$ratio"
        echo "low_width=$(low_for "$ratio")"
        echo "budget_width=$(budget_for "$ratio")"
        echo "high_width=$(high_for "$ratio")"
        echo "$RESULT_ROOT/${NAME}_${ratio}_vllm_CalibrationFree_full8_v1_${METHOD_TOKEN}_${TIMESTAMP}_${SEED}"
    done
    exit 0
fi

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/harp_combo/$MODEL}"
RANKING_CACHE="${RANKING_CACHE:-/data/xinpeigao/evalscope_results/_artifacts/harp/$MODEL/harp_rankings.pt}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
SCORE_LOCK="${SCORE_LOCK:-/data/xinpeigao/evalscope_results/_launchers/harp_v2.score.lock}"
mkdir -p "$ARTIFACT_ROOT" "$(dirname "$SCORE_LOCK")"

if [[ "$MODEL" == qwen36 ]]; then
    # shellcheck disable=SC1091
    source /data/xinpeigao/evalscope_results/_launchers/qwen36.env
fi
if [[ "$MODEL" == olmoe ]]; then
    export MAX_MODEL_LEN=4096
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
fi
if [[ "$TP" -gt 1 && -z "${VLLM_EXTRA_ARGS:-}" ]]; then
    export VLLM_EXTRA_ARGS='--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}'
fi

ENV_PREFIX="${VLLM_ENV:-/data/xinpeigao/conda_envs/gemma4-vllm-cu128}"
NVIDIA_LIB="$(find "$ENV_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB}${ENV_PREFIX}/lib64:${ENV_PREFIX}/lib:${ENV_PREFIX}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build_profile() {
    local ratio="$1"
    local low_width budget high_width profile
    low_width="$(low_for "$ratio")"
    budget="$(budget_for "$ratio")"
    high_width="$(high_for "$ratio")"
    profile="$ARTIFACT_ROOT/harp_combo_${ratio}pct.pt"
    if [[ -f "$profile" && -f "$RANKING_CACHE" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio allocator=$ALLOCATOR low=$low_width budget=$budget high=$high_width"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m HARP.build_harp_artifacts \
        --model-path "$MODEL_PATH" \
        --output-channel-cache "$RANKING_CACHE" \
        --output-profile "$profile" \
        --low-width "$low_width" \
        --budget-width "$budget" \
        --high-width "$high_width" \
        --allocator "$ALLOCATOR" \
        --gamma "$GAMMA" \
        --min-fraction "$MIN_FRACTION"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/harp_combo_${ratio}pct.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        printf '%s\n' "$checkpoint"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m HARP.export_harp_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$RANKING_CACHE" \
        --output-dir "$checkpoint"
    printf '%s\n' "$checkpoint"
}

run_ratio() {
    local ratio="$1"
    local checkpoint experiment_dir attempt
    echo "[$(date -Is)] WAIT score lock $MODEL ratio=$ratio lock=$SCORE_LOCK"
    exec 8>"$SCORE_LOCK"
    flock 8
    echo "[$(date -Is)] GOT score lock $MODEL ratio=$ratio"
    build_profile "$ratio"
    checkpoint="$(export_checkpoint "$ratio" | tail -n 1)"
    flock -u 8
    experiment_dir="$RESULT_ROOT/${NAME}_${ratio}_vllm_CalibrationFree_full8_v1_${METHOD_TOKEN}_${TIMESTAMP}_${SEED}"
    echo "[$(date -Is)] EVAL $MODEL ratio=$ratio gpu=$GPU port=$PORT"
    for attempt in 1 2 3; do
        echo "[$(date -Is)] EVAL attempt ${attempt}/3 $MODEL ratio=$ratio"
        if [[ -d "$experiment_dir" ]]; then
            echo "resume existing $experiment_dir"
            if RESULT_ROOT="$RESULT_ROOT" METHOD="$METHOD_TOKEN" MASTER_PORT="$((29700 + ${GPU%%,*}))" \
                bash "$RESUME_LAUNCHER" \
                "$experiment_dir" \
                "$checkpoint" \
                "$GPU" \
                "$PORT" \
                "$TP"; then
                return 0
            fi
        elif RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" MASTER_PORT="$((29700 + ${GPU%%,*}))" \
            bash "$EVAL_LAUNCHER" \
            "$NAME" \
            "$checkpoint" \
            "$GPU" \
            "$PORT" \
            "$TP" \
            "$METHOD_TOKEN" \
            "$ratio" \
            CalibrationFree \
            full8_v1; then
            return 0
        fi
        echo "[$(date -Is)] EVAL failed attempt ${attempt}/3 $MODEL ratio=$ratio" >&2
        sleep 20
    done
    return 1
}

echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP result_root=$RESULT_ROOT method=$METHOD_TOKEN allocator=$ALLOCATOR"
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
