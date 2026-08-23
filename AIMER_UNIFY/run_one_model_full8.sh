#!/usr/bin/env bash
# Build agreement-gated Unify rankings from Mix + LayerProp + PRP caches, export, eval 50% then 25%.
# Usage: run_one_model_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
_UNIFY_RESULT_ROOT="${RESULT_ROOT:-/home/xinpeigao/evalscope/results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_UNIFY_RESULT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
SEED="${SEED:-42}"
LAYERPROP_TAU="${LAYERPROP_TAU:-8}"
METHOD_TOKEN="${METHOD_TOKEN:-AIMERUnify}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$GPU" && -n "$PORT" ]] || die "Usage: $0 qwen3|gemma4|qwen36 GPU PORT"
[[ "$METHOD_TOKEN" =~ ^[A-Za-z0-9]+([A-Za-z0-9-]*[A-Za-z0-9])?$ ]] || die "Invalid METHOD_TOKEN '$METHOD_TOKEN'."

MASTER_PORT_BASE="${MASTER_PORT_BASE:-19890}"
case "$MODEL" in
    qwen3)
        NAME="Qwen330BA3BInstruct"
        MODEL_PATH="${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}"
        WIDTH_50=384
        WIDTH_25=576
        DEFAULT_LAYERPROP="/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus_lp/qwen3/layerprop.pt"
        ;;
    gemma4)
        NAME="Gemma4-26B-A4B"
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        WIDTH_50=352
        WIDTH_25=512
        DEFAULT_LAYERPROP="/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus_lpxa/gemma4/layerprop.pt"
        ;;
    qwen36)
        NAME="Qwen3.6-35B-A3B"
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        WIDTH_50=256
        WIDTH_25=384
        DEFAULT_LAYERPROP="/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus_lp/qwen36/layerprop.pt"
        ;;
    *) die "Unknown model '$MODEL'. Unify currently evaluates qwen3, gemma4, and qwen36." ;;
esac

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

MIX_CACHE="${MIX_CACHE:-/data/xinpeigao/evalscope_results/_artifacts/aimer_mix/$MODEL/aimer_mix_rankings.pt}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus/$MODEL}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/aimer_unify/$MODEL}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
LAYERPROP_SOURCE="${LAYERPROP_SOURCE:-$DEFAULT_LAYERPROP}"
mkdir -p "$ARTIFACT_ROOT"
[[ -f "$MIX_CACHE" ]] || die "AIMER-Mix ranking cache is missing: $MIX_CACHE"
[[ -f "$SOURCE_ROOT/prp.pt" ]] || die "PRP cache is missing: $SOURCE_ROOT/prp.pt"
[[ -f "$LAYERPROP_SOURCE" ]] || die "LayerProp cache is missing: $LAYERPROP_SOURCE"

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

reuse_named_cache() {
    local name="$1"
    local source="$2"
    local dest="$ARTIFACT_ROOT/${name}.pt"
    if [[ -f "$dest" ]]; then
        echo "skip source $MODEL $name"
        return
    fi
    [[ -f "$source" ]] || die "missing $name cache at $source"
    cp -a "$source" "$dest"
    if [[ -f "${source%.pt}.json" ]]; then
        cp -a "${source%.pt}.json" "$ARTIFACT_ROOT/${name}.json"
    fi
    echo "copied $name cache from $source"
}

build_profile() {
    local ratio="$1"
    local width profile rankings
    width="$(width_for "$ratio")"
    profile="$ARTIFACT_ROOT/aimer_unify_${ratio}pct_per_layer.pt"
    rankings="$ARTIFACT_ROOT/aimer_unify_${ratio}pct_rankings.pt"
    if [[ -f "$profile" && -f "$rankings" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    cat >"$ARTIFACT_ROOT/fusion_${ratio}.json" <<EOF
{
  "model": "$MODEL",
  "method": "aimer_unify",
  "ratio": $ratio,
  "retained_channels": $width,
  "sources": ["prp", "layerprop"],
  "gate": "mix_keepset_overlap_with_ffn_pseudo",
  "layerprop_tau": $LAYERPROP_TAU,
  "ignore_base": false,
  "use_pp": false
}
EOF
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio width=$width tau=$LAYERPROP_TAU gate=keepset-overlap"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m AIMER_UNIFY.build_unify_artifacts \
        --model-path "$MODEL_PATH" \
        --aimer-mix-cache "$MIX_CACHE" \
        --source "prp=$ARTIFACT_ROOT/prp.pt" \
        --source "layerprop=$ARTIFACT_ROOT/layerprop.pt" \
        --output-channel-cache "$rankings" \
        --output-profile "$profile" \
        --retained-channels "$width" \
        --layerprop-tau "$LAYERPROP_TAU"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile rankings
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/aimer_unify_${ratio}pct_per_layer.pt"
    rankings="$ARTIFACT_ROOT/aimer_unify_${ratio}pct_rankings.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        printf '%s\n' "$checkpoint"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    if [[ -d "$checkpoint" ]]; then
        rm -rf "$checkpoint"
    fi
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m AIMER_UNIFY.export_unify_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$rankings" \
        --output-dir "$checkpoint"
    printf '%s\n' "$checkpoint"
}

run_ratio() {
    local ratio="$1"
    local checkpoint experiment_dir
    build_profile "$ratio"
    checkpoint="$(export_checkpoint "$ratio" | tail -n 1)"
    experiment_dir="$RESULT_ROOT/${NAME}_${ratio}_vllm_CalibrationFree_full8_v1_${METHOD_TOKEN}_${TIMESTAMP}_${SEED}"
    echo "[$(date -Is)] EVAL $MODEL ratio=$ratio gpu=$GPU port=$PORT method=$METHOD_TOKEN"
    local dist_port="$(( MASTER_PORT_BASE + ${GPU%%,*} ))"
    if [[ -d "$experiment_dir" ]]; then
        echo "resume existing $experiment_dir"
        RESULT_ROOT="$RESULT_ROOT" METHOD="$METHOD_TOKEN" MASTER_PORT="$dist_port" VLLM_PORT="$dist_port" \
            bash "$RESUME_LAUNCHER" \
            "$experiment_dir" \
            "$checkpoint" \
            "$GPU" \
            "$PORT" \
            1
        return
    fi
    RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" MASTER_PORT="$dist_port" VLLM_PORT="$dist_port" \
        bash "$EVAL_LAUNCHER" \
        "$NAME" \
        "$checkpoint" \
        "$GPU" \
        "$PORT" \
        1 \
        "$METHOD_TOKEN" \
        "$ratio" \
        CalibrationFree \
        full8_v1
}

echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP method=$METHOD_TOKEN tau=$LAYERPROP_TAU result_root=$RESULT_ROOT"
reuse_named_cache prp "$SOURCE_ROOT/prp.pt"
reuse_named_cache layerprop "$LAYERPROP_SOURCE"
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
