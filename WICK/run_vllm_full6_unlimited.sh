#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 MODEL_ID API_URL METHOD RESULTS_ROOT" >&2
    exit 2
fi

MODEL_ID=$1
API_URL=${2%/}
METHOD=$3
RESULTS_ROOT=$4
PYTHON_BIN=${PYTHON_BIN:-/data01/home/xuzk/anaconda3/envs/xhquant/bin/python}
REPO_ROOT=/data01/home/xinpei.gao/evalscope
DATASETS=${DATASETS:-arc,hellaswag,winogrande,gsm8k,math_500,mmlu}
DRY_RUN=${DRY_RUN:-false}

export PYTHONPATH="$REPO_ROOT"
export NUMEXPR_MAX_THREADS=16

run_dataset() {
    local dataset=$1
    local max_tokens=$2
    local dataset_args=$3
    local work_dir="$RESULTS_ROOT/$METHOD/$dataset"
    local -a command=(
        "$PYTHON_BIN" -m evalscope.cli.cli eval
        --model "$MODEL_ID"
        --model-id "$MODEL_ID-$dataset"
        --eval-type openai_api
        --api-url "$API_URL/v1/chat/completions"
        --api-key EMPTY
        --datasets "$dataset"
        --dataset-args "$dataset_args"
        --generation-config "{\"max_tokens\":$max_tokens,\"temperature\":0.0,\"do_sample\":false,\"extra_body\":{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
        --eval-batch-size 16
        --seed 42
        --timeout 1200
        --work-dir "$work_dir"
        --no-timestamp
    )
    if [[ -d "$work_dir/predictions" ]]; then
        command+=(--use-cache "$work_dir")
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    "${command[@]}"
}

if [[ ",$DATASETS," == *,arc,* ]]; then
    run_dataset arc 2048 \
        '{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge","ARC-Easy"]}}'
fi
if [[ ",$DATASETS," == *,hellaswag,* ]]; then
    run_dataset hellaswag 512 \
        '{"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"}}'
fi
if [[ ",$DATASETS," == *,winogrande,* ]]; then
    run_dataset winogrande 1024 \
        '{"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"}}'
fi
if [[ ",$DATASETS," == *,gsm8k,* ]]; then
    run_dataset gsm8k 2048 \
        '{"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0}}'
fi
if [[ ",$DATASETS," == *,math_500,* ]]; then
    run_dataset math_500 4096 \
        '{"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"}}'
fi
if [[ ",$DATASETS," == *,mmlu,* ]]; then
    run_dataset mmlu 2048 \
        '{"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"}}'
fi