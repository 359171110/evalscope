#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3-30B-A3B-Instruct-2507}"
PROFILE_ROOT="${PROFILE_ROOT:-$EXPERIMENT_ROOT/experiments/profiles/qwen3_aimer_channel_50pct_20260804}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXPERIMENT_ROOT/experiments/results/qwen3_aimer_channel_50pct_quick9_max4096_20260804}"
GPUS_CSV="${GPUS_CSV:-4,5}"
DRY_RUN="${DRY_RUN:-false}"
PROFILE="${PROFILE:-$PROFILE_ROOT/aimer_channel_50pct_per_layer.pt}"
CHANNEL_CACHE="${CHANNEL_CACHE:-$PROFILE_ROOT/aimer_channel_rankings.pt}"
RUNNER="$SCRIPT_DIR/run_evalscope_static_profile.py"
MODEL_ID="qwen3-aimer-channel-50pct-quick9-max4096"
MMLU_PATH="/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"
GSM8K_PATH="/data01/datasets/evalscope_benchmarks/gsm8k"
MATH_PATH="/data01/datasets/evalscope_benchmarks/math_500"

MMLU_SUBJECTS=(
    abstract_algebra anatomy astronomy business_ethics clinical_knowledge
    college_biology college_chemistry college_computer_science college_mathematics
    college_medicine college_physics computer_security conceptual_physics econometrics
    electrical_engineering elementary_mathematics formal_logic global_facts high_school_biology
    high_school_chemistry high_school_computer_science high_school_european_history
    high_school_geography high_school_government_and_politics high_school_macroeconomics
    high_school_mathematics high_school_microeconomics high_school_physics high_school_psychology
    high_school_statistics high_school_us_history high_school_world_history human_aging
    human_sexuality international_law jurisprudence logical_fallacies machine_learning management
    marketing medical_genetics miscellaneous moral_disputes moral_scenarios nutrition philosophy
    prehistory professional_accounting professional_law professional_medicine professional_psychology
    public_relations security_studies sociology us_foreign_policy virology world_religions
)
MATH_LEVELS=("Level 1" "Level 2" "Level 3" "Level 4" "Level 5")
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

if [[ ${#GPUS[@]} -ne 2 ]]; then
    echo "ERROR: GPUS_CSV must contain exactly two physical GPU IDs." >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[4-7]$ ]]; then
        echo "ERROR: physical GPU IDs must be one of 4,5,6,7; got '$gpu'." >&2
        exit 2
    fi
done
if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
    echo "ERROR: GPUS_CSV must not contain duplicate GPU IDs." >&2
    exit 2
fi
if [[ ! -f "$PROFILE" || ! -f "$CHANNEL_CACHE" ]]; then
    echo "ERROR: channel-wise AIMER profile and channel cache must exist." >&2
    exit 2
fi

PROFILE_SHA256="$(sha256sum "$PROFILE" | awk '{print $1}')"
CHANNEL_SHA256="$(sha256sum "$CHANNEL_CACHE" | awk '{print $1}')"
SHARD_ROOT="$RESULTS_ROOT/parallel_shards"
LOG_ROOT="$RESULTS_ROOT/launcher_logs/recovery"
mkdir -p "$SHARD_ROOT" "$LOG_ROOT"

has_complete_prediction() {
    local filename="$1"
    local expected="$2"
    local path
    while IFS= read -r -d '' path; do
        if [[ "$(wc -l < "$path")" -eq "$expected" ]]; then
            return 0
        fi
    done < <(find "$SHARD_ROOT" -path "*/predictions/$MODEL_ID/$filename" -type f -print0 2>/dev/null)
    return 1
}

run_eval() {
    local gpu="$1"
    local dataset="$2"
    local limit="$3"
    local dataset_args="$4"
    local work_dir="$5"
    local log_path="$6"

    env CUDA_VISIBLE_DEVICES="$gpu" \
        LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}" \
        NUMEXPR_MAX_THREADS=16 \
        PYTHONPATH="$ROOT:$EXPERIMENT_ROOT/code" \
        "$PYTHON_BIN" "$RUNNER" \
        --model-path "$MODEL_PATH" \
        --model-id "$MODEL_ID" \
        --model-family qwen3 \
        --profile "$PROFILE" \
        --channel-cache "$CHANNEL_CACHE" \
        --expected-profile-file-sha256 "$PROFILE_SHA256" \
        --expected-channel-file-sha256 "$CHANNEL_SHA256" \
        --work-dir "$work_dir" \
        --datasets "$dataset" \
        --dataset-args "$dataset_args" \
        --dataset-limits "{\"${dataset}\":${limit}}" \
        --generation-config '{"max_tokens":4096}' \
        --eval-batch-size 1 \
        --seed 42 \
        --correction-mode none \
        --max-correction-ratio 0.20 \
        --moe-backend torch_index_add \
        --no-enable-thinking \
        --no-timestamp >"$log_path" 2>&1
}

run_mmlu_queue() {
    local gpu="$1"
    local start_index="$2"
    local end_index="$3"
    local subject_index
    local subject
    for ((subject_index = start_index; subject_index < end_index; subject_index++)); do
        subject="${MMLU_SUBJECTS[$subject_index]}"
        if has_complete_prediction "mmlu_${subject}.jsonl" 10; then
            echo "SKIP mmlu subject: $subject (10 predictions already complete)"
            continue
        fi
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "GPU $gpu mmlu subject: $subject limit: 10 max_tokens: 4096"
            continue
        fi
        run_eval \
            "$gpu" \
            mmlu \
            10 \
            "{\"mmlu\":{\"local_path\":\"$MMLU_PATH\",\"subset_list\":[\"$subject\"]}}" \
            "$SHARD_ROOT/aimer_channel_recovery_mmlu_${subject}_gpu${gpu}" \
            "$LOG_ROOT/mmlu_${subject}_gpu${gpu}.log"
    done
}

run_gsm8k_math_queue() {
    local gpu="$1"
    local level
    if has_complete_prediction "gsm8k_main.jsonl" 128; then
        echo "SKIP gsm8k (128 predictions already complete)"
    elif [[ "$DRY_RUN" == "true" ]]; then
        echo "GPU $gpu gsm8k limit: 128 max_tokens: 4096"
    else
        run_eval \
            "$gpu" \
            gsm8k \
            128 \
            "{\"gsm8k\":{\"local_path\":\"$GSM8K_PATH\",\"few_shot_num\":0}}" \
            "$SHARD_ROOT/aimer_channel_recovery_gsm8k_gpu${gpu}" \
            "$LOG_ROOT/gsm8k_gpu${gpu}.log"
    fi

    for level in "${MATH_LEVELS[@]}"; do
        if has_complete_prediction "math_500_${level}.jsonl" 20; then
            echo "SKIP math_500 subset: $level (20 predictions already complete)"
            continue
        fi
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "GPU $gpu math_500 subset: $level limit: 20 max_tokens: 4096"
            continue
        fi
        run_eval \
            "$gpu" \
            math_500 \
            20 \
            "{\"math_500\":{\"local_path\":\"$MATH_PATH\",\"subset_list\":[\"$level\"]}}" \
            "$SHARD_ROOT/aimer_channel_recovery_math_${level// /_}_gpu${gpu}" \
            "$LOG_ROOT/math_${level// /_}_gpu${gpu}.log"
    done
}

MMLU_SPLIT_INDEX=36
run_mmlu_queue "${GPUS[0]}" 0 "$MMLU_SPLIT_INDEX" &
mmlu_pid="$!"
(
    run_gsm8k_math_queue "${GPUS[1]}"
    run_mmlu_queue "${GPUS[1]}" "$MMLU_SPLIT_INDEX" "${#MMLU_SUBJECTS[@]}"
) &
math_pid="$!"

status=0
wait "$mmlu_pid" || status=1
wait "$math_pid" || status=1
exit "$status"