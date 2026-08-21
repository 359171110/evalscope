#!/usr/bin/env bash
# Export Magnitude checkpoints for one model at 50% then 25%.
# Usage: export_one_model.sh MODEL
set -euo pipefail

MODEL="${1:-}"
ROOT="/home/xinpeigao/evalscope"
CODE_ROOT="$ROOT/static_moe_prunning/code"
# shellcheck disable=SC1091
source "$ROOT/eval_protocol/env.sh"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/data1/xinpeigao/tmp}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$MODEL" ]] || die "Usage: $0 qwen3|gemma4|qwen36|deepseek"

case "$MODEL" in
    qwen3)
        MODEL_PATH="${QWEN3_MODEL_PATH:-/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507}"
        WIDTH_50=384
        WIDTH_25=576
        ;;
    gemma4)
        MODEL_PATH="${GEMMA4_MODEL_PATH:-/data/xinpeigao/models/gemma-4-26B-A4B-it}"
        WIDTH_50=352
        WIDTH_25=512
        ;;
    qwen36)
        MODEL_PATH="${QWEN36_MODEL_PATH:-/data/xinpeigao/models/Qwen3.6-35B-A3B}"
        WIDTH_50=256
        WIDTH_25=384
        ;;
    deepseek)
        MODEL_PATH="${DEEPSEEK_MODEL_PATH:-/data/xinpeigao/models/DeepSeek-V2-Lite-Chat}"
        WIDTH_50=704
        WIDTH_25=1056
        ;;
    *) die "Unknown model '$MODEL'." ;;
esac

[[ -d "$MODEL_PATH" ]] || die "Model path does not exist: $MODEL_PATH"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/xinpeigao/evalscope_results/_artifacts/magnitude/$MODEL}"
mkdir -p "$ARTIFACT_ROOT"

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
    profile="$ARTIFACT_ROOT/magnitude_${ratio}pct_per_layer.pt"
    if [[ -f "$profile" && -f "$ARTIFACT_ROOT/magnitude_rankings.pt" ]]; then
        echo "skip build $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] BUILD $MODEL ratio=$ratio width=$width"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Magnitude.build_magnitude_artifacts \
        --model-path "$MODEL_PATH" \
        --output-channel-cache "$ARTIFACT_ROOT/magnitude_rankings.pt" \
        --output-profile "$profile" \
        --retained-channels "$width"
}

export_checkpoint() {
    local ratio="$1"
    local checkpoint profile
    checkpoint="$ARTIFACT_ROOT/checkpoint_$ratio"
    profile="$ARTIFACT_ROOT/magnitude_${ratio}pct_per_layer.pt"
    if [[ -f "$checkpoint/pruning_export_manifest.json" ]]; then
        echo "skip export $MODEL $ratio"
        return
    fi
    echo "[$(date -Is)] EXPORT $MODEL ratio=$ratio -> $checkpoint"
    rm -rf "$checkpoint"
    mkdir -p "$checkpoint"
    env PYTHONPATH="$ROOT:$CODE_ROOT" PYTHONNOUSERSITE=1 \
        "$VLLM_PYTHON" -m Magnitude.export_magnitude_checkpoint \
        --model-path "$MODEL_PATH" \
        --profile "$profile" \
        --channel-cache "$ARTIFACT_ROOT/magnitude_rankings.pt" \
        --output-dir "$checkpoint"
}

echo "[$(date -Is)] START export $MODEL"
build_profile 50
export_checkpoint 50
build_profile 25
export_checkpoint 25
echo "[$(date -Is)] ALL EXPORTED $MODEL"
ls -ld "$ARTIFACT_ROOT"/checkpoint_50 "$ARTIFACT_ROOT"/checkpoint_25
python3 -c "import json; from pathlib import Path
for r in (50,25):
 p=Path('$ARTIFACT_ROOT')/f'checkpoint_{r}'/'pruning_export_manifest.json'
 m=json.loads(p.read_text())
 print(r, m['retained_channels'], m['export_layout'], m.get('exported_shared_expert_intermediate_size'))"
