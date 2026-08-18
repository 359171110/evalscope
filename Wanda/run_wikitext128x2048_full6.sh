#!/usr/bin/env bash

set -euo pipefail

WANDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$WANDA_ROOT/.." && pwd)"
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python}"
XHQUANT_SITE="/data01/home/xuzk/anaconda3/envs/xhquant/lib/python3.10/site-packages"
WIKITEXT_TRAIN_ARROW="${WIKITEXT_TRAIN_ARROW:-/data01/home/xinpei.gao/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
RATIO="${RATIO:-50}"
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
    [[ "$1" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer; got '$1'."
}

model_field() {
    local model="$1"
    local field="$2"
    case "$model:$field" in
        qwen3:name) printf 'Qwen330BA3BInstruct\n' ;;
        qwen3:path) printf '%s\n' "${QWEN3_MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}" ;;
        qwen3:python) printf '%s\n' "$PYTHON_BIN" ;;
        qwen3:gpu) printf '%s\n' "${QWEN3_GPU:-1}" ;;
        qwen3:port) printf '%s\n' "${QWEN3_PORT:-18180}" ;;
        qwen3:width)
            case "$RATIO" in
                50) printf '384\n' ;;
                25) printf '576\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen3." ;;
            esac
            ;;
        qwen3:cache) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/reap_50pct_screening/c1_wikitext_train_128x2048.pt" ;;
        qwen3:cache_sha) printf '11324347a87608d294c47a157a5f3791a72e75b0ab7c5752fae096473da5ffb1\n' ;;
        qwen3:protocol) printf 'c1_wikitext_train_128x2048_seed42_screening_v1\n' ;;
        qwen3:artifact) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/qwen3_wikitext128x2048_wanda" ;;
        gemma4:name) printf 'Gemma4-26B-A4B\n' ;;
        gemma4:path) printf '%s\n' "${GEMMA4_MODEL_PATH:-/data01/datasets/gemma-4-26B-A4B-it}" ;;
        gemma4:python) printf '%s\n' "$VLLM_PYTHON" ;;
        gemma4:gpu) printf '%s\n' "${GEMMA4_GPU:-2}" ;;
        gemma4:port) printf '%s\n' "${GEMMA4_PORT:-18181}" ;;
        gemma4:width)
            case "$RATIO" in
                50) printf '352\n' ;;
                25) printf '512\n' ;;
                *) die "Unsupported RATIO=$RATIO for gemma4." ;;
            esac
            ;;
        gemma4:cache) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/gemma4_wikitext128x2048_v1/calibration.pt" ;;
        gemma4:cache_sha) printf '%s\n' "${GEMMA4_CACHE_SHA256:-4ef0fe7aa02c4d2825c85cd8d6b6a38d66907b761580e8316cd9d022f32649e2}" ;;
        gemma4:protocol) printf 'gemma4_wikitext128x2048_v1\n' ;;
        gemma4:artifact) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/gemma4_wikitext128x2048_wanda" ;;
        qwen36:name) printf 'Qwen3.6-35B-A3B\n' ;;
        qwen36:path) printf '%s\n' "${QWEN36_MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}" ;;
        qwen36:python) printf '%s\n' "$VLLM_PYTHON" ;;
        qwen36:gpu) printf '%s\n' "${QWEN36_GPU:-4}" ;;
        qwen36:port) printf '%s\n' "${QWEN36_PORT:-18182}" ;;
        qwen36:width)
            case "$RATIO" in
                50) printf '256\n' ;;
                25) printf '384\n' ;;
                *) die "Unsupported RATIO=$RATIO for qwen36." ;;
            esac
            ;;
        qwen36:cache) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/qwen36_wikitext128x2048_202608091930/calibration.pt" ;;
        qwen36:cache_sha) printf '730ed2e9edc65c1da1332e628bc7b0fad7a2d4b1f0ba1b81edd1ee8e358479fe\n' ;;
        qwen36:protocol) printf 'qwen36_wikitext128x2048_202608091930\n' ;;
        qwen36:artifact) printf '%s\n' "$ROOT/static_moe_prunning/experiments/calibration/qwen36_wikitext128x2048_wanda" ;;
        *) die "Unknown model/field '$model:$field'." ;;
    esac
}

experiment_dir() {
    local model="$1"
    RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" MODEL_NAME="$(model_field "$model" name)" \
        "$CODE_ROOT/scripts/create_result_dir.sh" \
        --model "$(model_field "$model" name)" \
        --inference vllm \
        --calibration WikiText128x2048 \
        --protocol full6_v1 \
        --method Wanda \
        --pruning-ratio-label "$RATIO" \
        --pruning-ratio-percent "$RATIO" \
        --timestamp "$TIMESTAMP" \
        --dry-run
}

pythonpath_for() {
    printf '%s:%s\n' "$ROOT" "$CODE_ROOT"
}

verify_cache() {
    local model="$1"
    local cache protocol expected actual
    cache="$(model_field "$model" cache)"
    protocol="$(model_field "$model" protocol)"
    expected="$(model_field "$model" cache_sha)"
    [[ -f "$cache" ]] || die "Calibration cache does not exist: $cache"
    actual="$(sha256sum "$cache" | awk '{print $1}')"
    if [[ -n "$expected" && "$actual" != "$expected" ]]; then
        die "Calibration cache SHA256 mismatch for $model: expected $expected, got $actual"
    fi
    echo "cache=$cache"
    echo "cache_sha256=$actual"
    echo "protocol_name=$protocol"
}

build_gemma4_cache() {
    local cache protocol
    cache="$(model_field gemma4 cache)"
    protocol="$(model_field gemma4 protocol)"
    if [[ -f "$cache" ]]; then
        verify_cache gemma4
        return
    fi
    [[ -f "$WIKITEXT_TRAIN_ARROW" ]] || die "WikiText train Arrow file is missing: $WIKITEXT_TRAIN_ARROW"
    [[ -x "$VLLM_PYTHON" ]] || die "Gemma4 Python is not executable: $VLLM_PYTHON"
    mkdir -p "$(dirname "$cache")"
    local builder="$CODE_ROOT/scripts/build_shared_calibration_token_cache.py"
    local model_path
    model_path="$(model_field gemma4 path)"
    echo "build_gemma4_cache $cache using $WIKITEXT_TRAIN_ARROW"
    env PYTHONPATH="$ROOT:$CODE_ROOT" HF_DATASETS_OFFLINE=1 "$VLLM_PYTHON" - "$builder" "$XHQUANT_SITE" \
        --model-path "$model_path" \
        --output-cache "$cache" \
        --dataset wikitext \
        --config wikitext-2-raw-v1 \
        --split train \
        --text-field text \
        --arrow-file "$WIKITEXT_TRAIN_ARROW" \
        --sequence-length 2048 \
        --calibration-sequences 128 \
        --token-offset 0 \
        --protocol-name "$protocol" <<'PY'
import runpy
import sys

builder = sys.argv[1]
sys.path.append(sys.argv[2])
sys.argv = [builder, *sys.argv[3:]]
runpy.run_path(builder, run_name="__main__")
PY
    verify_cache gemma4
}

collect_statistics() {
    local model="$1"
    local artifact cache python_bin gpu log
    artifact="$(model_field "$model" artifact)"
    cache="$(model_field "$model" cache)"
    python_bin="$(model_field "$model" python)"
    gpu="$(model_field "$model" gpu)"
    log="$artifact/collect.log"
    validate_gpu "$gpu"
    [[ -x "$python_bin" ]] || die "Python executable is not executable: $python_bin"
    [[ -d "$(model_field "$model" path)" ]] || die "Model path does not exist: $(model_field "$model" path)"
    if [[ "$model" == gemma4 ]]; then
        build_gemma4_cache
    else
        verify_cache "$model"
    fi
    mkdir -p "$artifact"
    if [[ -f "$artifact/statistics.pt" ]]; then
        echo "statistics already exist: $artifact/statistics.pt"
        return
    fi
    local -a device_args=(--device-map cuda:0)
    if [[ "$python_bin" == "$VLLM_PYTHON" ]]; then
        device_args=(--device-map none --device cuda)
    fi
    local -a command=(
        env
        "CUDA_VISIBLE_DEVICES=$gpu"
        "LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
        "PYTHONPATH=$(pythonpath_for "$model")"
        "$python_bin"
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
    "${command[@]}" >"$log" 2>&1
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
        RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}" \
            "$CODE_ROOT/scripts/create_result_dir.sh" \
            --model "$(model_field "$model" name)" \
            --inference vllm \
            --calibration WikiText128x2048 \
            --protocol full6_v1 \
            --method Wanda \
            --pruning-ratio-label "$RATIO" \
            --pruning-ratio-percent "$RATIO" \
            --timestamp "$TIMESTAMP" >/dev/null
    fi
    printf '%s\n' "$directory"
}

export_checkpoint() {
    local model="$1"
    local artifact directory checkpoint
    artifact="$(model_field "$model" artifact)"
    directory="$(ensure_experiment_dir "$model")"
    checkpoint="$directory/checkpoints/Wanda"
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
    local directory checkpoint model_id gpu port server_log server_pid
    directory="$(ensure_experiment_dir "$model")"
    checkpoint="$directory/checkpoints/Wanda"
    model_id="$(model_field "$model" name)-${RATIO}-Wanda"
    gpu="$(model_field "$model" gpu)"
    port="$(model_field "$model" port)"
    server_log="$directory/server_logs/Wanda.log"
    [[ -f "$checkpoint/pruning_export_manifest.json" ]] || die "Checkpoint is not exported: $checkpoint"
    validate_gpu "$gpu"
    mkdir -p "$directory/server_logs"
    local vllm_bin_dir
    vllm_bin_dir="$(dirname "$VLLM_PYTHON")"
    local -a server_command=(
        env
        -u LD_LIBRARY_PATH
        "PATH=$vllm_bin_dir:${PATH:-}"
        "CUDA_VISIBLE_DEVICES=$gpu"
        "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server
        --model "$checkpoint"
        --served-model-name "$model_id"
        --host 127.0.0.1
        --port "$port"
        --dtype bfloat16
        --seed 42
        --max-model-len 8192
        --max-num-seqs 16
        --gpu-memory-utilization 0.90
        --generation-config vllm
        --default-chat-template-kwargs '{"enable_thinking":false}'
    )
    "${server_command[@]}" >"$server_log" 2>&1 &
    server_pid=$!
    cleanup_server() {
        kill -TERM "$server_pid" 2>/dev/null || true
    }
    trap cleanup_server EXIT INT TERM
    local healthy="false"
    for _ in $(seq 1 300); do
        if env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
            healthy="true"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -100 "$server_log" >&2
            die "vLLM server exited before becoming healthy: $server_log"
        fi
        sleep 2
    done
    [[ "$healthy" == "true" ]] || die "Timed out waiting for vLLM server: $server_log"
    env -u LD_LIBRARY_PATH curl --silent --fail "http://127.0.0.1:$port/v1/models" >/dev/null || die "vLLM /v1/models check failed."
    PROTOCOL=full6_v1 \
    PYTHON_BIN="$PYTHON_BIN" \
    ARC_PATH="${ARC_PATH:-/data01/datasets/evalscope_benchmarks/arc}" \
    HELLASWAG_PATH="${HELLASWAG_PATH:-/data01/datasets/evalscope_benchmarks/hellaswag}" \
    WINOGRANDE_PATH="${WINOGRANDE_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip}" \
    GSM8K_PATH="${GSM8K_PATH:-/data01/datasets/evalscope_benchmarks/gsm8k}" \
    MATH_500_PATH="${MATH_500_PATH:-/data01/datasets/evalscope_benchmarks/math_500}" \
    MMLU_PATH="${MMLU_PATH:-/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu}" \
        bash "$ROOT/eval_protocol/run_vllm_protocol.sh" \
        "$model_id" \
        "http://127.0.0.1:$port" \
        Wanda \
        "$directory"
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" || true
    trap - EXIT INT TERM
}

dry_run_one() {
    local model="$1"
    echo "model=$model"
    echo "target_model=$(model_field "$model" name)"
    echo "model_path=$(model_field "$model" path)"
    echo "collect_python=$(model_field "$model" python)"
    echo "gpu=$(model_field "$model" gpu)"
    echo "port=$(model_field "$model" port)"
    echo "ratio=$RATIO"
    echo "retained_channels=$(model_field "$model" width)"
    echo "calibration=WikiText128x2048"
    echo "protocol=full6_v1"
    echo "method=Wanda"
    echo "cache=$(model_field "$model" cache)"
    echo "artifact_root=$(model_field "$model" artifact)"
    echo "experiment=$(experiment_dir "$model")"
}

run_one() {
    local model="$1"
    local action="$2"
    case "$action" in
        dry-run) dry_run_one "$model" ;;
        cache)
            [[ "$model" == gemma4 ]] || die "cache is only required for gemma4; $model already has a frozen WikiText cache."
            build_gemma4_cache
            ;;
        collect) collect_statistics "$model" ;;
        build) build_artifacts "$model" ;;
        export) export_checkpoint "$model" ;;
        eval) run_eval "$model" ;;
        prepare)
            collect_statistics "$model"
            build_artifacts "$model"
            export_checkpoint "$model"
            ;;
        *) die "Usage: $0 qwen3|gemma4|qwen36|all dry-run|cache|collect|build|export|eval|prepare" ;;
    esac
}

[[ -n "$MODEL" ]] || die "Usage: RATIO=25|50 $0 qwen3|gemma4|qwen36|all dry-run|cache|collect|build|export|eval|prepare"
[[ "$RATIO" == "25" || "$RATIO" == "50" ]] || die "RATIO must be 25 or 50; got '$RATIO'."
[[ -x "$CODE_ROOT/scripts/create_result_dir.sh" ]] || die "Framework result helper is missing."

if [[ "$MODEL" == all ]]; then
    for item in qwen3 gemma4 qwen36; do
        run_one "$item" "$ACTION"
    done
    exit 0
fi

[[ "$MODEL" == qwen3 || "$MODEL" == gemma4 || "$MODEL" == qwen36 ]] ||
    die "Model must be qwen3, gemma4, qwen36, or all."
run_one "$MODEL" "$ACTION"
