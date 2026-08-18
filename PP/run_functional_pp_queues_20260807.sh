#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xuzk/anaconda3/envs/vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
PROFILE_ROOT="${PROFILE_ROOT:-$ROOT/PP/experiments/profiles/functional_pp_frozen_v1_20260807}"
QUEUE_TIMESTAMP="${QUEUE_TIMESTAMP:-202608071705}"
QUEUE="${1:?Usage: $0 aimer|mfe|weight-moment|bfc|bfc-b6|bfc-b9|local-bfc|local-bfc-b6|local-bfc-b9|tre|tre-b6|tre-b9|recon|recon-b6|recon-b9|response-coverage|gate|hybrid|esp|pwrp|mix-pp39-pwrp38|mix-pp39-esp38|mix-pp27-esp25-pwrp25|mix-pp34-esp31-pwrp31|mix-pp40-esp38-pwrp37}"
GPU_ID="${GPU_ID:?Set GPU_ID to a truly idle GPU.}"
PORT="${PORT:?Set PORT to an unused local port.}"
DRY_RUN="${DRY_RUN:-false}"
IMPORTANCE_BACKBONE="${IMPORTANCE_BACKBONE:-}"

PP_CACHE="$ROOT/PP/experiments/profiles/down_proj_norm_ablation_20260807/PurePseudo-K8-Q4-NoDownNorm/rankings.pt"
AIMER_CACHE="$ROOT/WICK/experiments/profiles/qwen3_wick_aimer_fixed_diagnostics_20260806/aimer_fixed_rankings.pt"
FUNCTIONAL_BUILDER="$ROOT/PP/build_functional_backbone.py"
PROTECTION_BUILDER="$ROOT/PP/build_protected_rankings.py"
COVERAGE_BUILDER="$ROOT/PP/build_response_coverage.py"
BFC_BUILDER="$ROOT/PP/build_bilinear_functional_coverage.py"
TRE_BUILDER="$ROOT/PP/build_triad_removal_energy.py"
GATE_HYBRID_BUILDER="$ROOT/PP/build_gate_hybrid_protection.py"
MODEL_DERIVED_PROBE_BUILDER="$ROOT/PP/build_model_derived_probe_rankings.py"
MULTI_SOURCE_PROTECTION_BUILDER="$ROOT/PP/build_multi_source_protection.py"
EXPORTER="$ROOT/WICK/export_uniform_qwen3_moe.py"
RECON_EXPORTER="$ROOT/PP/export_reconstructed_qwen3_moe.py"
CREATE_RESULT_DIR="$CODE_ROOT/scripts/create_result_dir.sh"
QUICK9_RUNNER="$ROOT/PP/run_vllm_quick9.sh"

export PYTHONPATH="$ROOT:$CODE_ROOT"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1"
}

pruning_identity() {
    local retained_blocks=$1
    case "$retained_blocks" in
        6) printf '%s %s\n' "Prune6of12" "50" ;;
        9) printf '%s %s\n' "Prune3of12" "25" ;;
        *) die "Unsupported retained block count: $retained_blocks" ;;
    esac
}

ensure_experiment_dir() {
    local method=$1
    local retained_blocks=$2
    local label percent expected created
    read -r label percent < <(pruning_identity "$retained_blocks")
    expected=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
        --inference vllm \
        --calibration CalibrationFree \
        --method "$method" \
        --pruning-ratio-label "$label" \
        --pruning-ratio-percent "$percent" \
        --timestamp "$QUEUE_TIMESTAMP" \
        --dry-run)
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%s\n' "$expected"
        return
    fi
    if [[ ! -d "$expected" ]]; then
        created=$(RESULT_ROOT="$RESULT_ROOT" "$CREATE_RESULT_DIR" \
            --inference vllm \
            --calibration CalibrationFree \
            --method "$method" \
            --pruning-ratio-label "$label" \
            --pruning-ratio-percent "$percent" \
            --timestamp "$QUEUE_TIMESTAMP")
        [[ "$created" == "$expected" ]] || die "Unexpected result directory: $created"
    fi
    printf '%s\n' "$expected"
}

quick9_complete() {
    local experiment_dir=$1
    local method=$2
    local report_count=0
    if [[ -d "$experiment_dir/$method" ]]; then
        report_count=$(find "$experiment_dir/$method" -type f -path '*/reports/*/*.json' | wc -l)
    fi
    ((report_count >= 6))
}

prepare_output_dir() {
    local output_dir=$1
    if [[ -d "$output_dir" && ! -f "$output_dir/pruning_export_manifest.json" ]]; then
        mv "$output_dir" "${output_dir}.incomplete.$(date +%Y%m%d%H%M%S)"
    fi
}

build_backbone() {
    local importance=$1
    local rankings=$2
    local backbone_root
    backbone_root=$(dirname "$rankings")
    local profile="$backbone_root/profile.pt"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN build-backbone importance=$importance rankings=$rankings"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$backbone_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$FUNCTIONAL_BUILDER" \
            --model-path "$MODEL_PATH" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --importance "$importance" \
            --target-pruning-ratio 0.5 \
            --router-neighbors 8 \
            --channel-block-size 64 \
            --device cuda:0
    fi
}

run_experiment() {
    local importance=$1
    local method_prefix=$2
    local retained_blocks=$3
    local backbone_cache=$4
    local method="${method_prefix}-PPFv1-G10-B${retained_blocks}of12"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN importance=$importance backbone=$backbone_cache result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$variant_root"
        "$PYTHON_BIN" "$PROTECTION_BUILDER" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$backbone_cache" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --backbone "$importance" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_gate_hybrid_experiment() {
    local method_kind=$1
    local retained_blocks=$2
    local method
    case "$method_kind" in
        GateGA) method="GateGA-PPFv1-G10-B${retained_blocks}of12" ;;
        Hybrid) method="HybridPP5GA5-PPFv1-G10-B${retained_blocks}of12" ;;
        *) die "Unsupported Gate/Hybrid method: $method_kind" ;;
    esac
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local diagnostics="$variant_root/diagnostics.json"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN aimer=$AIMER_CACHE pseudo=$PP_CACHE result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" || ! -f "$diagnostics" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$GATE_HYBRID_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --diagnostics-output "$diagnostics" \
            --method "$method_kind" \
            --retained-blocks "$retained_blocks" \
            --router-neighbors 8 \
            --top-q 4 \
            --protection-ratio 0.10 \
            --channel-block-size 64 \
            --device cuda:0
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

build_model_derived_probe_cache() {
    local method_kind=$1
    local cache_name=${method_kind,,}
    local cache_root="$PROFILE_ROOT/backbones/$cache_name"
    local profile="$cache_root/profile.pt"
    local rankings="$cache_root/rankings.pt"
    local diagnostics="$cache_root/diagnostics.json"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN build-model-derived-probes method=$method_kind rankings=$rankings"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" || ! -f "$diagnostics" ]]; then
        mkdir -p "$cache_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$MODEL_DERIVED_PROBE_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            --pp-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --diagnostics-output "$diagnostics" \
            --method "$method_kind" \
            --probe-count 9 \
            --top-q 4 \
            --protection-ratio 0.10 \
            --retained-blocks 6 \
            --channel-block-size 64 \
            --device cuda:0
    fi
}

run_model_derived_probe_experiment() {
    local method_kind=$1
    local retained_blocks=$2
    local cache_name=${method_kind,,}
    local method_prefix
    case "$method_kind" in
        ESP) method_prefix="AIMERESP" ;;
        PWRP) method_prefix="AIMERPWRP" ;;
        *) die "Unsupported model-derived probe method: $method_kind" ;;
    esac
    local method="${method_prefix}-PPFv1-G10-B${retained_blocks}of12"
    local source_root="$PROFILE_ROOT/backbones/$cache_name"
    local source_rankings="$source_root/rankings.pt"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local experiment_dir checkpoint_dir

    build_model_derived_probe_cache "$method_kind"
    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN method=$method_kind source=$source_rankings result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    require_file "$source_rankings"
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$variant_root"
        "$PYTHON_BIN" "$PROTECTION_BUILDER" \
            --model-path "$MODEL_PATH" \
            --backbone-cache "$AIMER_CACHE" \
            --pseudo-cache "$source_rankings" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --backbone "${cache_name}_protection" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_multi_source_experiment() {
    local mix_kind=$1
    local retained_blocks=$2
    local method protection_ratio=0.10 source_args
    case "$mix_kind" in
        pp39-pwrp38)
            method="AIMERMix-PP39-PWRP38-B${retained_blocks}of12"
            source_args=(
                --source "PP=39=$PP_CACHE"
                --source "PWRP=38=$PROFILE_ROOT/backbones/pwrp/rankings.pt"
            )
            ;;
        pp39-esp38)
            method="AIMERMix-PP39-ESP38-B${retained_blocks}of12"
            source_args=(
                --source "PP=39=$PP_CACHE"
                --source "ESP=38=$PROFILE_ROOT/backbones/esp/rankings.pt"
            )
            ;;
        pp27-esp25-pwrp25)
            method="AIMERMix-PP27-ESP25-PWRP25-B${retained_blocks}of12"
            source_args=(
                --source "PP=27=$PP_CACHE"
                --source "ESP=25=$PROFILE_ROOT/backbones/esp/rankings.pt"
                --source "PWRP=25=$PROFILE_ROOT/backbones/pwrp/rankings.pt"
            )
            ;;
        pp34-esp31-pwrp31)
            method="AIMERMix-PP34-ESP31-PWRP31-B${retained_blocks}of12"
            protection_ratio=0.125
            source_args=(
                --source "PP=34=$PP_CACHE"
                --source "ESP=31=$PROFILE_ROOT/backbones/esp/rankings.pt"
                --source "PWRP=31=$PROFILE_ROOT/backbones/pwrp/rankings.pt"
            )
            ;;
        pp40-esp38-pwrp37)
            method="AIMERMix-PP40-ESP38-PWRP37-B${retained_blocks}of12"
            protection_ratio=0.15
            source_args=(
                --source "PP=40=$PP_CACHE"
                --source "ESP=38=$PROFILE_ROOT/backbones/esp/rankings.pt"
                --source "PWRP=37=$PROFILE_ROOT/backbones/pwrp/rankings.pt"
            )
            ;;
        *) die "Unsupported multi-source mix: $mix_kind" ;;
    esac
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local diagnostics="$variant_root/diagnostics.json"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN mix=$mix_kind result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    require_file "$PROFILE_ROOT/backbones/esp/rankings.pt"
    require_file "$PROFILE_ROOT/backbones/pwrp/rankings.pt"
    if [[ ! -f "$profile" || ! -f "$rankings" || ! -f "$diagnostics" ]]; then
        mkdir -p "$variant_root"
        "$PYTHON_BIN" "$MULTI_SOURCE_PROTECTION_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            "${source_args[@]}" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --diagnostics-output "$diagnostics" \
            --method "${method,,}" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio "$protection_ratio" \
            --channel-block-size 64
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_coverage_experiment() {
    local importance=$1
    local method_prefix=$2
    local retained_blocks=$3
    local backbone_cache=$4
    local method="${method_prefix}-PPFv1-G10-B${retained_blocks}of12"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN importance=$importance backbone=$backbone_cache result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$COVERAGE_BUILDER" \
            --model-path "$MODEL_PATH" \
            --importance-cache "$backbone_cache" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --importance-name "$importance" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10 \
            --candidate-multiplier 2.0 \
            --router-neighbors 8 \
            --channel-block-size 64 \
            --device cuda:0
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_bfc_experiment() {
    local retained_blocks=$1
    local method="AIMERBFC-PPFv1-G10-B${retained_blocks}of12"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN aimer=$AIMER_CACHE pseudo=$PP_CACHE result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$BFC_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --method "${method,,}" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10 \
            --candidate-extra-ratio 0.5 \
            --channel-block-size 64 \
            --device cuda:0
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_local_bfc_experiment() {
    local retained_blocks=$1
    local method="AIMERLocalBFC-PPFv1-G10-B${retained_blocks}of12"
    local global_method="AIMERBFC-PPFv1-G10-B${retained_blocks}of12"
    local global_cache="$PROFILE_ROOT/$global_method/rankings.pt"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local diagnostics="$variant_root/diagnostics.json"
    local experiment_dir checkpoint_dir

    require_file "$global_cache"
    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN aimer=$AIMER_CACHE pseudo=$PP_CACHE global_bfc=$global_cache result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" || ! -f "$diagnostics" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$BFC_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            --pseudo-cache "$PP_CACHE" \
            --global-bfc-cache "$global_cache" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --diagnostics-output "$diagnostics" \
            --method "${method,,}" \
            --selection-mode local \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10 \
            --channel-block-size 64 \
            --device cuda:0
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_tre_experiment() {
    local retained_blocks=$1
    local method="AIMERLocalTRE-PPFv1-G10-T5-B${retained_blocks}of12"
    local variant_root="$PROFILE_ROOT/$method"
    local profile="$variant_root/profile.pt"
    local rankings="$variant_root/rankings.pt"
    local diagnostics="$variant_root/diagnostics.json"
    local experiment_dir checkpoint_dir

    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN aimer=$AIMER_CACHE pseudo=$PP_CACHE result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    if [[ ! -f "$profile" || ! -f "$rankings" || ! -f "$diagnostics" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$TRE_BUILDER" \
            --model-path "$MODEL_PATH" \
            --aimer-cache "$AIMER_CACHE" \
            --pseudo-cache "$PP_CACHE" \
            --output-profile "$profile" \
            --output-channel-cache "$rankings" \
            --diagnostics-output "$diagnostics" \
            --method "${method,,}" \
            --retained-blocks "$retained_blocks" \
            --protection-ratio 0.10 \
            --boundary-ratio 0.05 \
            --channel-block-size 64 \
            --device cuda:0
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" ]]; then
        "$PYTHON_BIN" "$EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --retained-channels "$((retained_blocks * 64))"
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

run_recon_experiment() {
    local retained_blocks=$1
    local method="AIMERRecon-PPFv1-G10-B${retained_blocks}of12"
    local baseline_method="AIMER-PPFv1-G10-B${retained_blocks}of12"
    local rankings="$PROFILE_ROOT/$baseline_method/rankings.pt"
    local variant_root="$PROFILE_ROOT/$method"
    local diagnostics="$variant_root/diagnostics.json"
    local experiment_dir checkpoint_dir

    require_file "$rankings"
    experiment_dir=$(ensure_experiment_dir "$method" "$retained_blocks")
    checkpoint_dir="$experiment_dir/checkpoints/$method"
    echo "$(date --iso-8601=seconds) starting method=$method gpu=$GPU_ID port=$PORT"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN rankings=$rankings diagnostics=$diagnostics result=$experiment_dir"
        return
    fi
    if quick9_complete "$experiment_dir" "$method"; then
        echo "Quick9 already complete: $method"
        return
    fi
    prepare_output_dir "$checkpoint_dir"
    if [[ ! -f "$checkpoint_dir/pruning_export_manifest.json" || ! -f "$diagnostics" ]]; then
        mkdir -p "$variant_root"
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$RECON_EXPORTER" \
            --model-path "$MODEL_PATH" \
            --channel-cache "$rankings" \
            --output-dir "$checkpoint_dir" \
            --diagnostics-output "$diagnostics" \
            --retained-channels "$((retained_blocks * 64))" \
            --ridge-relative 1.0e-4 \
            --device cuda:0
    fi
    env -u LD_LIBRARY_PATH \
        CHECKPOINT_DIR="$checkpoint_dir" \
        EXPERIMENT_DIR="$experiment_dir" \
        METHOD="$method" \
        MODEL_ID="Qwen330BA3BInstruct-$((retained_blocks * 64))ch-$method" \
        GPU_ID="$GPU_ID" \
        PORT="$PORT" \
        VLLM_PYTHON="$VLLM_PYTHON" \
        bash "$QUICK9_RUNNER"
}

require_file "$MODEL_PATH/config.json"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$PP_CACHE"
require_file "$AIMER_CACHE"
require_file "$FUNCTIONAL_BUILDER"
require_file "$PROTECTION_BUILDER"
require_file "$COVERAGE_BUILDER"
require_file "$BFC_BUILDER"
require_file "$TRE_BUILDER"
require_file "$GATE_HYBRID_BUILDER"
require_file "$MODEL_DERIVED_PROBE_BUILDER"
require_file "$MULTI_SOURCE_PROTECTION_BUILDER"
require_file "$EXPORTER"
require_file "$RECON_EXPORTER"
require_file "$CREATE_RESULT_DIR"
require_file "$QUICK9_RUNNER"

case "$QUEUE" in
    aimer)
        method_prefix=AIMER
        backbone_cache="$AIMER_CACHE"
        run_experiment aimer "$method_prefix" 9 "$backbone_cache"
        run_experiment aimer "$method_prefix" 6 "$backbone_cache"
        exit 0
        ;;
    mfe) method_prefix=MFE ;;
    weight-moment) method_prefix=WeightMoment ;;
    bfc)
        run_bfc_experiment 6
        run_bfc_experiment 9
        exit 0
        ;;
    bfc-b6)
        run_bfc_experiment 6
        exit 0
        ;;
    bfc-b9)
        run_bfc_experiment 9
        exit 0
        ;;
    local-bfc)
        run_local_bfc_experiment 6
        run_local_bfc_experiment 9
        exit 0
        ;;
    local-bfc-b6)
        run_local_bfc_experiment 6
        exit 0
        ;;
    local-bfc-b9)
        run_local_bfc_experiment 9
        exit 0
        ;;
    tre)
        run_tre_experiment 6
        run_tre_experiment 9
        exit 0
        ;;
    tre-b6)
        run_tre_experiment 6
        exit 0
        ;;
    tre-b9)
        run_tre_experiment 9
        exit 0
        ;;
    recon)
        run_recon_experiment 6
        run_recon_experiment 9
        exit 0
        ;;
    recon-b6)
        run_recon_experiment 6
        exit 0
        ;;
    recon-b9)
        run_recon_experiment 9
        exit 0
        ;;
    response-coverage)
        case "$IMPORTANCE_BACKBONE" in
            mfe) method_prefix=MFEResponseCoverage ;;
            weight-moment) method_prefix=WeightMomentResponseCoverage ;;
            *) die "Set IMPORTANCE_BACKBONE to mfe or weight-moment for response-coverage." ;;
        esac
        backbone_cache="$PROFILE_ROOT/backbones/$IMPORTANCE_BACKBONE/rankings.pt"
        require_file "$backbone_cache"
        run_coverage_experiment "$IMPORTANCE_BACKBONE" "$method_prefix" 6 "$backbone_cache"
        run_coverage_experiment "$IMPORTANCE_BACKBONE" "$method_prefix" 9 "$backbone_cache"
        exit 0
        ;;
    gate)
        run_gate_hybrid_experiment GateGA 6
        run_gate_hybrid_experiment GateGA 9
        exit 0
        ;;
    hybrid)
        run_gate_hybrid_experiment Hybrid 6
        run_gate_hybrid_experiment Hybrid 9
        exit 0
        ;;
    esp)
        run_model_derived_probe_experiment ESP 6
        run_model_derived_probe_experiment ESP 9
        exit 0
        ;;
    pwrp)
        run_model_derived_probe_experiment PWRP 6
        run_model_derived_probe_experiment PWRP 9
        exit 0
        ;;
    mix-pp39-pwrp38)
        run_multi_source_experiment pp39-pwrp38 6
        run_multi_source_experiment pp39-pwrp38 9
        exit 0
        ;;
    mix-pp39-esp38)
        run_multi_source_experiment pp39-esp38 6
        run_multi_source_experiment pp39-esp38 9
        exit 0
        ;;
    mix-pp27-esp25-pwrp25)
        run_multi_source_experiment pp27-esp25-pwrp25 6
        run_multi_source_experiment pp27-esp25-pwrp25 9
        exit 0
        ;;
    mix-pp34-esp31-pwrp31)
        run_multi_source_experiment pp34-esp31-pwrp31 6
        run_multi_source_experiment pp34-esp31-pwrp31 9
        exit 0
        ;;
    mix-pp40-esp38-pwrp37)
        run_multi_source_experiment pp40-esp38-pwrp37 6
        run_multi_source_experiment pp40-esp38-pwrp37 9
        exit 0
        ;;
    *) die "Usage: $0 aimer|mfe|weight-moment|bfc|bfc-b6|bfc-b9|local-bfc|local-bfc-b6|local-bfc-b9|tre|tre-b6|tre-b9|recon|recon-b6|recon-b9|response-coverage|gate|hybrid|esp|pwrp|mix-pp39-pwrp38|mix-pp39-esp38|mix-pp27-esp25-pwrp25|mix-pp34-esp31-pwrp31|mix-pp40-esp38-pwrp37" ;;
esac

backbone_cache="$PROFILE_ROOT/backbones/$QUEUE/rankings.pt"
build_backbone "$QUEUE" "$backbone_cache"
run_experiment "$QUEUE" "$method_prefix" 6 "$backbone_cache"
run_experiment "$QUEUE" "$method_prefix" 9 "$backbone_cache"