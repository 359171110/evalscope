#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 MODEL_ID API_URL VARIANT RESULTS_ROOT" >&2
  exit 2
fi

MODEL_ID=$1
API_URL=$2
VARIANT=$3
RESULTS_ROOT=$4
PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
REPO_ROOT=/data01/home/xinpei.gao/evalscope
DATASETS=${DATASETS:-arc,hellaswag,gsm8k,math_500}

export PYTHONPATH="$REPO_ROOT"
export NUMEXPR_MAX_THREADS=16

run_dataset() {
  local dataset=$1
  local limit=$2
  local max_tokens=$3
  local dataset_args=$4
  local work_dir="$RESULTS_ROOT/$VARIANT/$dataset"

  "$PYTHON_BIN" -m evalscope.cli.cli eval \
    --model "$MODEL_ID" \
    --model-id "$MODEL_ID-$dataset" \
    --eval-type openai_api \
    --api-url "$API_URL/v1/chat/completions" \
    --api-key EMPTY \
    --datasets "$dataset" \
    --dataset-args "$dataset_args" \
    --limit "$limit" \
    --generation-config "{\"max_tokens\":$max_tokens,\"temperature\":0.0,\"do_sample\":false}" \
    --eval-batch-size 16 \
    --seed 42 \
    --timeout 1200 \
    --work-dir "$work_dir" \
    --no-timestamp
}

if [[ ",$DATASETS," == *,arc,* ]]; then
  run_dataset arc 100 64 \
    '{"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge"]}}'
fi
if [[ ",$DATASETS," == *,hellaswag,* ]]; then
  run_dataset hellaswag 100 32 \
    '{"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"}}'
fi
if [[ ",$DATASETS," == *,gsm8k,* ]]; then
  run_dataset gsm8k 32 1024 \
    '{"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0,"prompt_template":"{question}\nPlease reason step by step. End with a non-empty LaTeX \\boxed expression containing the computed final number inside its braces. Do not use an empty box or a placeholder."}}'
fi
if [[ ",$DATASETS," == *,math_500,* ]]; then
  run_dataset math_500 5 4096 \
    '{"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500","prompt_template":"{question}\nPlease reason step by step. End with a non-empty LaTeX \\boxed expression containing the computed final number inside its braces. Do not use an empty box or a placeholder."}}'
fi