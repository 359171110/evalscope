#!/usr/bin/env bash
# Build PP/PRP sources, width-specific AMP rankings, export, and evaluate 50% then 25%.
# Usage: run_one_model_full8.sh MODEL GPU PORT
set -euo pipefail

MODEL="${1:-}"
GPU="${2:-}"
PORT="${3:-}"

ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
_AMP_RESULT_ROOT="${RESULT_ROOT:-/home/xinpeigao/evalscope/results}"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export RESULT_ROOT="$_AMP_RESULT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
SEED="${SEED:-42}"
SCORE_MODE="${SCORE_MODE:-output}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" && -n "$GPU" && -n "$PORT" ]] || die "Usage: $0 qwen3|gemma4|qwen36|deepseek GPU PORT"

MASTER_PORT_BASE="${MASTER_PORT_BASE:-29400}"
USE_PP=1
USE_PRP=1
USE_LAYERPROP=0
LAYERPROP_WEIGHT=0.0
DEFAULT_METHOD_TOKEN="AIMERMixPlusV2"
DEFAULT_ARTIFACT_KIND="aimer_mix_plus_v2"
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
        # Gemma4 PP is a poor proxy: router RMS/scale is not the expert FFN input.
        USE_PP=0
        USE_LAYERPROP=1
        DEFAULT_METHOD_TOKEN="AIMERMixPlusLP"
        DEFAULT_ARTIFACT_KIND="aimer_mix_plus_lp"
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
if [[ "${ENABLE_LAYERPROP:-0}" == 1 && "$MODEL" != gemma4 && -z "${SOURCE_SET:-}" ]]; then
    USE_LAYERPROP=1
    DEFAULT_METHOD_TOKEN="AIMERMixPlusLP"
    DEFAULT_ARTIFACT_KIND="aimer_mix_plus_lp"
fi
if [[ -n "${SOURCE_SET:-}" ]]; then
    USE_PP=0
    USE_PRP=0
    USE_LAYERPROP=0
    LAYERPROP_WEIGHT=0.0
    case "$SOURCE_SET" in
        pp)
            USE_PP=1
            DEFAULT_METHOD_TOKEN="AIMERMixPlusPP"
            DEFAULT_ARTIFACT_KIND="aimer_mix_plus_pp"
            ;;
        prp)
            USE_PRP=1
            DEFAULT_METHOD_TOKEN="AIMERMixPlusPRP"
            DEFAULT_ARTIFACT_KIND="aimer_mix_plus_prp"
            ;;
        layerprop)
            USE_LAYERPROP=1
            DEFAULT_METHOD_TOKEN="AIMERMixPlusLPOnly"
            DEFAULT_ARTIFACT_KIND="aimer_mix_plus_lponly"
            ;;
        *) die "SOURCE_SET must be pp, prp, or layerprop (got '$SOURCE_SET')." ;;
    esac
    # Pure pseudo ranking: Mix core size 0 (same ignore_base as LPx, one source).
    if [[ "${IGNORE_BASE:-0}" == 1 ]]; then
        case "$SOURCE_SET" in
            pp)
                DEFAULT_METHOD_TOKEN="AIMERMixPlusPPnb"
                DEFAULT_ARTIFACT_KIND="aimer_mix_plus_ppnb"
                ;;
            prp)
                DEFAULT_METHOD_TOKEN="AIMERMixPlusPRPnb"
                DEFAULT_ARTIFACT_KIND="aimer_mix_plus_prpnb"
                ;;
            layerprop)
                DEFAULT_METHOD_TOKEN="AIMERMixPlusLPnb"
                DEFAULT_ARTIFACT_KIND="aimer_mix_plus_lpnb"
                ;;
        esac
    fi
fi
if [[ "${FUSION_PRESET:-}" == lpxa ]]; then
    USE_PP=0
    USE_PRP=1
    USE_LAYERPROP=1
    DEFAULT_METHOD_TOKEN="AIMERMixPlusLPxa"
    DEFAULT_ARTIFACT_KIND="aimer_mix_plus_lpxa"
fi
METHOD_TOKEN="${METHOD_TOKEN:-$DEFAULT_METHOD_TOKEN}"

# Per-model, per-width fusion.
# Gemma4 uses LayerProp + PRP (no PP): router space is not the expert FFN input.
# Qwen3/Qwen3.6 keep a small PP-leaning rescue. DeepSeek-50 shrinks PRP toward Mix.
apply_ablation_fusion() {
    local ratio="$1"
    local want_ignore="${IGNORE_BASE:-0}"
    PP_WEIGHT=0.0
    PRP_WEIGHT=0.0
    LAYERPROP_WEIGHT=0.0
    IGNORE_BASE=0
    MINIMUM_BOUNDARY_CHANNELS=32
    case "$SOURCE_SET" in
        pp) PP_WEIGHT=1.0 ;;
        prp) PRP_WEIGHT=1.0 ;;
        layerprop) LAYERPROP_WEIGHT=1.0 ;;
        *) die "Unknown SOURCE_SET='$SOURCE_SET'." ;;
    esac
    case "$MODEL:$ratio" in
        qwen3:50)
            BOUNDARY_FRACTION=0.16
            MAXIMUM_BOUNDARY_FRACTION=0.28
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.05
            ;;
        qwen3:25)
            BOUNDARY_FRACTION=0.08
            MAXIMUM_BOUNDARY_FRACTION=0.14
            BASE_BOUNDARY_WEIGHT=1.00
            PSEUDO_WEIGHT=0.70
            ;;
        gemma4:50)
            BOUNDARY_FRACTION=0.55
            MAXIMUM_BOUNDARY_FRACTION=0.65
            BASE_BOUNDARY_WEIGHT=0.25
            PSEUDO_WEIGHT=1.50
            ;;
        gemma4:25)
            BOUNDARY_FRACTION=0.20
            MAXIMUM_BOUNDARY_FRACTION=0.35
            BASE_BOUNDARY_WEIGHT=0.75
            PSEUDO_WEIGHT=1.00
            ;;
        qwen36:50)
            BOUNDARY_FRACTION=0.14
            MAXIMUM_BOUNDARY_FRACTION=0.24
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.05
            ;;
        qwen36:25)
            BOUNDARY_FRACTION=0.12
            MAXIMUM_BOUNDARY_FRACTION=0.20
            BASE_BOUNDARY_WEIGHT=0.85
            PSEUDO_WEIGHT=0.95
            ;;
        deepseek:50)
            BOUNDARY_FRACTION=0.08
            MAXIMUM_BOUNDARY_FRACTION=0.12
            MINIMUM_BOUNDARY_CHANNELS=16
            BASE_BOUNDARY_WEIGHT=1.15
            PSEUDO_WEIGHT=0.60
            ;;
        deepseek:25)
            BOUNDARY_FRACTION=0.18
            MAXIMUM_BOUNDARY_FRACTION=0.28
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.10
            ;;
        *) die "No ablation fusion for $MODEL ratio=$ratio." ;;
    esac
    if [[ "$want_ignore" == 1 ]]; then
        IGNORE_BASE=1
        BOUNDARY_FRACTION=1.00
        MAXIMUM_BOUNDARY_FRACTION=1.00
        BASE_BOUNDARY_WEIGHT=0.00
    fi
}

apply_fusion() {
    local ratio="$1"
    ADAPTIVE_LP_PRP=0
    LAYERPROP_TAU="${LAYERPROP_TAU:-8}"
    if [[ -n "${SOURCE_SET:-}" ]]; then
        apply_ablation_fusion "$ratio"
        return
    fi
    if [[ "${FUSION_PRESET:-}" == lpxa ]]; then
        PP_WEIGHT=0.0
        PRP_WEIGHT=1.0
        LAYERPROP_WEIGHT=1.0
        BOUNDARY_FRACTION=1.00
        MAXIMUM_BOUNDARY_FRACTION=1.00
        MINIMUM_BOUNDARY_CHANNELS=32
        BASE_BOUNDARY_WEIGHT=0.00
        PSEUDO_WEIGHT=1.00
        IGNORE_BASE=1
        ADAPTIVE_LP_PRP=1
        return
    fi
    local ratio="$1"
    PP_WEIGHT=1.0
    PRP_WEIGHT=0.5
    LAYERPROP_WEIGHT=0.0
    BOUNDARY_FRACTION=0.15
    MAXIMUM_BOUNDARY_FRACTION=0.30
    MINIMUM_BOUNDARY_CHANNELS=32
    BASE_BOUNDARY_WEIGHT=0.75
    PSEUDO_WEIGHT=1.0
    IGNORE_BASE=0
    case "$MODEL:$ratio" in
        qwen3:50)
            PRP_WEIGHT=0.55
            BOUNDARY_FRACTION=0.16
            MAXIMUM_BOUNDARY_FRACTION=0.28
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.05
            ;;
        qwen3:25)
            PRP_WEIGHT=0.25
            BOUNDARY_FRACTION=0.08
            MAXIMUM_BOUNDARY_FRACTION=0.14
            BASE_BOUNDARY_WEIGHT=1.00
            PSEUDO_WEIGHT=0.70
            ;;
        gemma4:50)
            PP_WEIGHT=0.0
            case "${FUSION_PRESET:-lp}" in
                lp)
                    PRP_WEIGHT=1.00
                    LAYERPROP_WEIGHT=1.25
                    BOUNDARY_FRACTION=0.55
                    MAXIMUM_BOUNDARY_FRACTION=0.65
                    BASE_BOUNDARY_WEIGHT=0.25
                    PSEUDO_WEIGHT=1.50
                    ;;
                lpa)
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.60
                    BOUNDARY_FRACTION=0.70
                    MAXIMUM_BOUNDARY_FRACTION=0.80
                    BASE_BOUNDARY_WEIGHT=0.15
                    PSEUDO_WEIGHT=1.80
                    ;;
                lpb)
                    PRP_WEIGHT=1.20
                    LAYERPROP_WEIGHT=1.10
                    BOUNDARY_FRACTION=0.35
                    MAXIMUM_BOUNDARY_FRACTION=0.45
                    BASE_BOUNDARY_WEIGHT=0.55
                    PSEUDO_WEIGHT=1.20
                    ;;
                lpx)
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.60
                    BOUNDARY_FRACTION=1.00
                    MAXIMUM_BOUNDARY_FRACTION=1.00
                    BASE_BOUNDARY_WEIGHT=0.00
                    PSEUDO_WEIGHT=1.80
                    IGNORE_BASE=1
                    ;;
                lph)
                    # Same sources as LPa, even smaller Mix core.
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.60
                    BOUNDARY_FRACTION=0.88
                    MAXIMUM_BOUNDARY_FRACTION=0.95
                    BASE_BOUNDARY_WEIGHT=0.05
                    PSEUDO_WEIGHT=2.20
                    ;;
                *) die "Unknown FUSION_PRESET='$FUSION_PRESET' for gemma4." ;;
            esac
            ;;
        gemma4:25)
            PP_WEIGHT=0.0
            case "${FUSION_PRESET:-lp}" in
                lp)
                    PRP_WEIGHT=1.00
                    LAYERPROP_WEIGHT=1.25
                    BOUNDARY_FRACTION=0.20
                    MAXIMUM_BOUNDARY_FRACTION=0.35
                    BASE_BOUNDARY_WEIGHT=0.75
                    PSEUDO_WEIGHT=1.00
                    ;;
                lpa)
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.50
                    BOUNDARY_FRACTION=0.30
                    MAXIMUM_BOUNDARY_FRACTION=0.40
                    BASE_BOUNDARY_WEIGHT=0.45
                    PSEUDO_WEIGHT=1.40
                    ;;
                lpb)
                    PRP_WEIGHT=1.10
                    LAYERPROP_WEIGHT=1.20
                    BOUNDARY_FRACTION=0.12
                    MAXIMUM_BOUNDARY_FRACTION=0.22
                    BASE_BOUNDARY_WEIGHT=0.90
                    PSEUDO_WEIGHT=0.90
                    ;;
                lpx)
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.50
                    BOUNDARY_FRACTION=1.00
                    MAXIMUM_BOUNDARY_FRACTION=1.00
                    BASE_BOUNDARY_WEIGHT=0.00
                    PSEUDO_WEIGHT=1.40
                    IGNORE_BASE=1
                    ;;
                lph)
                    PRP_WEIGHT=0.50
                    LAYERPROP_WEIGHT=1.50
                    BOUNDARY_FRACTION=0.40
                    MAXIMUM_BOUNDARY_FRACTION=0.50
                    BASE_BOUNDARY_WEIGHT=0.20
                    PSEUDO_WEIGHT=1.70
                    ;;
                *) die "Unknown FUSION_PRESET='$FUSION_PRESET' for gemma4." ;;
            esac
            ;;
        qwen36:50)
            PP_WEIGHT=1.20
            PRP_WEIGHT=0.40
            BOUNDARY_FRACTION=0.14
            MAXIMUM_BOUNDARY_FRACTION=0.24
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.05
            ;;
        qwen36:25)
            PRP_WEIGHT=0.35
            BOUNDARY_FRACTION=0.12
            MAXIMUM_BOUNDARY_FRACTION=0.20
            BASE_BOUNDARY_WEIGHT=0.85
            PSEUDO_WEIGHT=0.95
            ;;
        deepseek:50)
            PP_WEIGHT=0.50
            PRP_WEIGHT=0.15
            BOUNDARY_FRACTION=0.08
            MAXIMUM_BOUNDARY_FRACTION=0.12
            MINIMUM_BOUNDARY_CHANNELS=16
            BASE_BOUNDARY_WEIGHT=1.15
            PSEUDO_WEIGHT=0.60
            ;;
        deepseek:25)
            PP_WEIGHT=0.45
            PRP_WEIGHT=0.55
            BOUNDARY_FRACTION=0.18
            MAXIMUM_BOUNDARY_FRACTION=0.28
            BASE_BOUNDARY_WEIGHT=0.70
            PSEUDO_WEIGHT=1.10
            ;;
    *) die "No fusion preset for $MODEL ratio=$ratio." ;;
    esac
    if [[ "$USE_LAYERPROP" == 1 && "$LAYERPROP_WEIGHT" == "0.0" ]]; then
        case "$MODEL:$ratio" in
            qwen3:50) LAYERPROP_WEIGHT=1.00 ;;
            qwen3:25) LAYERPROP_WEIGHT=0.70 ;;
            qwen36:50) LAYERPROP_WEIGHT=1.00 ;;
            qwen36:25) LAYERPROP_WEIGHT=0.80 ;;
            deepseek:50) LAYERPROP_WEIGHT=0.80 ;;
            deepseek:25) LAYERPROP_WEIGHT=0.90 ;;
        esac
    fi
}

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
[[ -x "$VLLM_PYTHON" ]] || die "VLLM_PYTHON is not executable: $VLLM_PYTHON"
mkdir -p "$RESULT_ROOT"

MIX_CACHE="${MIX_CACHE:-/data/xinpeigao/evalscope_results/_artifacts/aimer_mix/$MODEL/aimer_mix_rankings.pt}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus/$MODEL}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/${DEFAULT_ARTIFACT_KIND}/$MODEL}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/run_one_vllm_eval.sh}"
RESUME_LAUNCHER="${RESUME_LAUNCHER:-/data/xinpeigao/evalscope_results/_launchers/resume_one_full8.sh}"
mkdir -p "$ARTIFACT_ROOT"
[[ -f "$MIX_CACHE" ]] || die "AIMER-Mix ranking cache is missing: $MIX_CACHE"

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

build_source() {
    local method="$1"
    local cache="$ARTIFACT_ROOT/${method}.pt"
    if [[ -f "$cache" ]]; then
        echo "skip source $MODEL $method"
        return
    fi
    echo "[$(date -Is)] SOURCE $MODEL method=$method score_mode=$SCORE_MODE device=cuda gpu=$GPU"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU" \
        "$VLLM_PYTHON" -m AIMER_MIX_PLUS.build_pseudo_source_artifacts \
        --model-path "$MODEL_PATH" \
        --method "$method" \
        --output-cache "$cache" \
        --score-mode "$SCORE_MODE" \
        --device cuda
}

build_layerprop() {
    local cache="$ARTIFACT_ROOT/layerprop.pt"
    local default_gemma="/data/xinpeigao/evalscope_results/_artifacts/aimer_mix_plus_lp/gemma4/layerprop.pt"
    local source="${LAYERPROP_SOURCE:-}"
    if [[ -z "$source" && "$MODEL" == gemma4 && "${FUSION_PRESET:-}" != lpxa ]]; then
        source="$default_gemma"
    fi
    if [[ -f "$cache" ]]; then
        echo "skip source $MODEL layerprop"
        return
    fi
    if [[ -n "$source" && -f "$source" ]]; then
        cp -a "$source" "$cache"
        if [[ -f "${source%.pt}.json" ]]; then
            cp -a "${source%.pt}.json" "$ARTIFACT_ROOT/"
        fi
        echo "copied layerprop cache from $source"
        return
    fi
    echo "[$(date -Is)] SOURCE $MODEL method=layerprop score_mode=$SCORE_MODE device=cuda gpu=$GPU"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU" \
        "$VLLM_PYTHON" -m AIMER_MIX_PLUS.build_layerprop_source_artifacts \
        --model-path "$MODEL_PATH" \
        --output-cache "$cache" \
        --score-mode "$SCORE_MODE" \
        --device cuda
}

build_profile() {
    local ratio="$1"
    local width profile rankings
    width="$(width_for "$ratio")"
    profile="$ARTIFACT_ROOT/aimer_mix_plus_${ratio}pct_per_layer.pt"
    rankings="$ARTIFACT_ROOT/aimer_mix_plus_${ratio}pct_rankings.pt"
    if [[ -f "$profile" && -f "$rankings" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    apply_fusion "$ratio"
    cat >"$ARTIFACT_ROOT/fusion_${ratio}.json" <<EOF
{
  "model": "$MODEL",
  "fusion_preset": "${FUSION_PRESET:-lp}",
  "source_set": "${SOURCE_SET:-mixed}",
  "ratio": $ratio,
  "retained_channels": $width,
  "pp_weight": $PP_WEIGHT,
  "prp_weight": $PRP_WEIGHT,
  "layerprop_weight": $LAYERPROP_WEIGHT,
  "use_pp": $USE_PP,
  "use_prp": $USE_PRP,
  "use_layerprop": $USE_LAYERPROP,
  "boundary_fraction": $BOUNDARY_FRACTION,
  "maximum_boundary_fraction": $MAXIMUM_BOUNDARY_FRACTION,
  "minimum_boundary_channels": $MINIMUM_BOUNDARY_CHANNELS,
  "base_boundary_weight": $BASE_BOUNDARY_WEIGHT,
  "pseudo_weight": $PSEUDO_WEIGHT,
  "ignore_base": $IGNORE_BASE,
  "adaptive_lp_prp": ${ADAPTIVE_LP_PRP:-0},
  "layerprop_tau": ${LAYERPROP_TAU:-8}
}
EOF
    local source_args=()
    if [[ "$USE_PP" == 1 ]]; then
        source_args+=(--source "pp=$ARTIFACT_ROOT/pp.pt")
    fi
    if [[ "$USE_PRP" == 1 ]]; then
        source_args+=(--source "prp=$ARTIFACT_ROOT/prp.pt")
    fi
    if [[ "$USE_LAYERPROP" == 1 ]]; then
        source_args+=(--source "layerprop=$ARTIFACT_ROOT/layerprop.pt")
    fi
    [[ ${#source_args[@]} -ge 1 ]] || die "No AMP pseudo sources selected for $MODEL."
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio width=$width pp=$PP_WEIGHT prp=$PRP_WEIGHT layerprop=$LAYERPROP_WEIGHT bound=$BOUNDARY_FRACTION/$MAXIMUM_BOUNDARY_FRACTION min=$MINIMUM_BOUNDARY_CHANNELS base_bw=$BASE_BOUNDARY_WEIGHT pseudo_w=$PSEUDO_WEIGHT ignore_base=$IGNORE_BASE adaptive_lp_prp=${ADAPTIVE_LP_PRP:-0} tau=${LAYERPROP_TAU:-8}"
    local extra_args=()
    if [[ "$IGNORE_BASE" == 1 ]]; then
        extra_args+=(--ignore-base)
    fi
    if [[ "${ADAPTIVE_LP_PRP:-0}" == 1 ]]; then
        extra_args+=(--adaptive-lp-prp --layerprop-tau "${LAYERPROP_TAU:-8}")
    fi
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m AIMER_MIX_PLUS.build_aimer_mix_plus_artifacts \
        --model-path "$MODEL_PATH" \
        --aimer-mix-cache "$MIX_CACHE" \
        "${source_args[@]}" \
        --output-channel-cache "$rankings" \
        --output-profile "$profile" \
        --retained-channels "$width" \
        --boundary-fraction "$BOUNDARY_FRACTION" \
        --maximum-boundary-fraction "$MAXIMUM_BOUNDARY_FRACTION" \
        --minimum-boundary-channels "$MINIMUM_BOUNDARY_CHANNELS" \
        --base-boundary-weight "$BASE_BOUNDARY_WEIGHT" \
        --pseudo-weight "$PSEUDO_WEIGHT" \
        --pp-weight "$PP_WEIGHT" \
        --prp-weight "$PRP_WEIGHT" \
        --layerprop-weight "$LAYERPROP_WEIGHT" \
        "${extra_args[@]}"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile rankings
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/aimer_mix_plus_${ratio}pct_per_layer.pt"
    rankings="$ARTIFACT_ROOT/aimer_mix_plus_${ratio}pct_rankings.pt"
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
        "$VLLM_PYTHON" -m AIMER_MIX_PLUS.export_aimer_mix_plus_checkpoint \
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
    local dist_port="$((MASTER_PORT_BASE + ${GPU%%,*}))"
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

reuse_source_cache() {
    local method="$1"
    if [[ -f "$ARTIFACT_ROOT/${method}.pt" ]]; then
        return
    fi
    [[ -f "$SOURCE_ROOT/${method}.pt" ]] || die "missing $method cache at $SOURCE_ROOT/${method}.pt"
    cp -a "$SOURCE_ROOT/${method}.pt" "$ARTIFACT_ROOT/"
    if [[ -f "$SOURCE_ROOT/${method}.json" ]]; then
        cp -a "$SOURCE_ROOT/${method}.json" "$ARTIFACT_ROOT/"
    fi
    echo "copied $method cache from $SOURCE_ROOT"
}

echo "[$(date -Is)] START $MODEL gpu=$GPU port=$PORT timestamp=$TIMESTAMP method=$METHOD_TOKEN preset=${FUSION_PRESET:-lp} source_set=${SOURCE_SET:-mixed} result_root=$RESULT_ROOT pp=$USE_PP prp=$USE_PRP layerprop=$USE_LAYERPROP"
if [[ "$USE_PP" == 1 ]]; then
    reuse_source_cache pp
    build_source pp
fi
if [[ "$USE_PRP" == 1 ]]; then
    reuse_source_cache prp
    build_source prp
fi
if [[ "$USE_LAYERPROP" == 1 ]]; then
    build_layerprop
fi
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
