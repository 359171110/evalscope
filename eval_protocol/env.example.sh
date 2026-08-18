#!/usr/bin/env bash
# Copy to eval_protocol/env.sh and edit for the current server. Do not commit env.sh.
# Usage: source eval_protocol/env.sh

EVAL_PROTOCOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$(cd "$EVAL_PROTOCOL_DIR/.." && pwd)"
export CODE_ROOT="${CODE_ROOT:-$ROOT/static_moe_prunning/code}"
export RESULT_ROOT="${RESULT_ROOT:-$ROOT/result}"
export PYTHONPATH="${ROOT}:${CODE_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Python used by EvalScope launchers and protocol tools.
export PYTHON_BIN="${PYTHON_BIN:-python3}"

# Unified vLLM env for Gemma4, Qwen3-30B-A3B, and Qwen3.6-35B-A3B.
# Recreate it on a new server with eval_protocol/envs/gemma4-vllm-cu128/setup_gemma4_vllm_cu128.sh
export VLLM_ENV="${VLLM_ENV:-$HOME/.conda/envs/gemma4-vllm-cu128}"
export VLLM_PYTHON="${VLLM_PYTHON:-$VLLM_ENV/bin/python}"

# Base model checkpoints live outside this repo.
export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-30B-A3B-Instruct-2507}"
export MODEL_NAME="${MODEL_NAME:-Qwen330BA3BInstruct}"

# Frozen local benchmark copies. Put the six Quick9/full6 datasets here.
export DATASET_ROOT="${DATASET_ROOT:-/path/to/evalscope_benchmarks}"
export ARC_PATH="${ARC_PATH:-$DATASET_ROOT/arc}"
export HELLASWAG_PATH="${HELLASWAG_PATH:-$DATASET_ROOT/hellaswag}"
export WINOGRANDE_PATH="${WINOGRANDE_PATH:-$DATASET_ROOT/winogrande/winogrande_1.1.zip}"
export GSM8K_PATH="${GSM8K_PATH:-$DATASET_ROOT/gsm8k}"
export MATH_500_PATH="${MATH_500_PATH:-$DATASET_ROOT/math_500}"
export MMLU_PATH="${MMLU_PATH:-$DATASET_ROOT/mmlu}"

export PROTOCOL="${PROTOCOL:-quick9}"
export SEED="${SEED:-42}"
export VLLM_PORT="${VLLM_PORT:-18080}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

mkdir -p "$RESULT_ROOT"

echo "ROOT=$ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "VLLM_PYTHON=$VLLM_PYTHON"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "PROTOCOL=$PROTOCOL"
