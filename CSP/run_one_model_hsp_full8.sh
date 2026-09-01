#!/usr/bin/env bash
# Build, export, and evaluate HSP-Hetero for one model at 50% then 25%.
# Widths are {k0-64, k0, k0+64} around the uniform keep width of that sparsity.
# Scoring/export take a global flock so concurrent model jobs score sequentially.
# Usage: run_one_model_hsp_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
_HSP_RESULT_ROOT="${RESULT_ROOT:-/data/xinpeigao/evalscope_results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_HSP_RESULT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
SEED="${SEED:-42}"
# Do not inherit METHOD_TOKEN=CSP from a shared eval shell.
METHOD_TOKEN="${HSP_METHOD_TOKEN:-HSP}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$GPU" && -n "$PORT" ]] || die "Usage: $0 qwen3|gemma4|qwen36|deepseek|olmoe|mixtral GPU PORT"
[[ -z "${CSP_CANONICALIZE:-}" ]] || die "HSP-Hetero uses raw Expert-SP and raw Channel-SP; do not set CSP_CANONICALIZE."
IFS=',' read -r -a _GPU_PARTS <<< "$GPU"
TP="${TP:-${#_GPU_PARTS[@]}}"

case "$MODEL" in
    qwen3)
        NAME="Qwen330BA3BInstruct"
        MODEL_PATH="${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}"
        WIDTHS_50="320 384 448"
        BUDGET_50=384
        WIDTHS_25="512 576 640"
        BUDGET_25=576
        ;;
    gemma4)
        NAME="Gemma4-26B-A4B"
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        WIDTHS_50="288 352 416"
        BUDGET_50=352
        WIDTHS_25="448 512 576"
        BUDGET_25=512
        ;;
    qwen36)
        NAME="Qwen3.6-35B-A3B"
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        WIDTHS_50="192 256 320"
        BUDGET_50=256
        WIDTHS_25="320 384 448"
        BUDGET_25=384
        ;;
    deepseek)
        NAME="DeepSeek-V2-Lite-Chat"
        MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}"
        WIDTHS_50="640 704 768"
        BUDGET_50=704
        WIDTHS_25="992 1056 1120"
        BUDGET_25=1056
        ;;
    olmoe)
        NAME="OLMoE-1B-7B-Instruct"
        MODEL_PATH="${OLMOE_MODEL_PATH:-/data1/xinpeigao/caches/huggingface/hub/models--allenai--OLMoE-1B-7B-0125-Instruct/snapshots/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e}"
        WIDTHS_50="448 512 576"
        BUDGET_50=512
        WIDTHS_25="704 768 832"
        BUDGET_25=768
        ;;
    mixtral)
        NAME="Mixtral-8x7B-Instruct"
        MODEL_PATH="${MIXTRAL_MODEL_PATH:-/data1/xinpeigao/caches/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/eba92302a2861cdc0098cc54bc9f17cb2c47eb61}"
        WIDTHS_50="7104 7168 7232"
        BUDGET_50=7168
        WIDTHS_25="10688 10752 10816"
        BUDGET_25=10752
        ;;
    *) die "Unknown model '$MODEL'." ;;
esac

widths_for() {
    case "$1" in
        50) printf '%s\n' "$WIDTHS_50" ;;
        25) printf '%s\n' "$WIDTHS_25" ;;
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

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "calibration=CalibrationFree"
    echo "protocol=full8_v1"
    echo "method=$METHOD_TOKEN"
    echo "apply_input_scale=never"
    IFS=' ' read -r -a ratio_list <<< "${RATIOS:-50 25}"
    for ratio in "${ratio_list[@]}"; do
        echo "ratio=$ratio"
        echo "heterogeneous_widths=$(widths_for "$ratio")"
        echo "budget_width=$(budget_for "$ratio")"
        echo "$RESULT_ROOT/${NAME}_${ratio}_vllm_CalibrationFree_full8_v1_${METHOD_TOKEN}_${TIMESTAMP}_${SEED}"
    done
    exit 0
fi

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/hsp/$MODEL}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
SCORE_LOCK="${SCORE_LOCK:-/data/xinpeigao/evalscope_results/_launchers/hsp.score.lock}"
mkdir -p "$ARTIFACT_ROOT" "$(dirname "$SCORE_LOCK")"

if [[ "$MODEL" == qwen36 ]]; then
    # H20 + this vLLM build: FlashInfer GDN TMA crashes with cudaErrorUnknown (999).
    # shellcheck disable=SC1091
    source /data/xinpeigao/evalscope_results/_launchers/qwen36.env
fi
if [[ "$MODEL" == olmoe ]]; then
    # OLMoE max_position_embeddings is 4096; vLLM rejects the default 8192.
    # full8 MATH-500 max_tokens=4096 is also rejected (prompt + max_tokens > 4096).
    export MAX_MODEL_LEN=4096
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
fi
if [[ "$TP" -gt 1 && -z "${VLLM_EXTRA_ARGS:-}" ]]; then
    # Do not use ${VAR:-json}: bash closes ${} at the first } and appends leftover braces.
    export VLLM_EXTRA_ARGS='--compilation-config {"pass_config":{"fuse_allreduce_rms":false}}'
fi

ENV_PREFIX="${VLLM_ENV:-/data/xinpeigao/conda_envs/gemma4-vllm-cu128}"
NVIDIA_LIB="$(find "$ENV_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIB}${ENV_PREFIX}/lib64:${ENV_PREFIX}/lib:${ENV_PREFIX}/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build_profile() {
    local ratio="$1"
    local widths budget profile
    widths="$(widths_for "$ratio")"
    budget="$(budget_for "$ratio")"
    profile="$ARTIFACT_ROOT/hsp_${ratio}pct_hetero.pt"
    if [[ -f "$profile" && -f "$ARTIFACT_ROOT/csp_rankings.pt" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio widths=[$widths] budget=$budget"
    # shellcheck disable=SC2086
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m CSP.build_csp_artifacts \
        --model-path "$MODEL_PATH" \
        --output-channel-cache "$ARTIFACT_ROOT/csp_rankings.pt" \
        --output-profile "$profile" \
        --heterogeneous-widths $widths \
        --budget-width "$budget" \
        --apply-input-scale never
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/hsp_${ratio}pct_hetero.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        printf '%s\n' "$checkpoint"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m CSP.export_csp_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$ARTIFACT_ROOT/csp_rankings.pt" \
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

echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP result_root=$RESULT_ROOT method=$METHOD_TOKEN"
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
