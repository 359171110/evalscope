#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <arm> <physical-gpu-id> [preflight]" >&2
    exit 2
fi

ARM="$1"
GPU_ID="$2"
MODE="${3:-run}"

ROOT="/data01/home/xinpei.gao/evalscope"
EXPERIMENT_ROOT="$ROOT/static_moe_prunning"
PROFILE_ROOT="$EXPERIMENT_ROOT/experiments/profiles/reap_50pct_screening"
CALIBRATION_ROOT="$EXPERIMENT_ROOT/experiments/calibration/reap_50pct_screening"
RESULT_ROOT="$EXPERIMENT_ROOT/experiments/results/qwen3_50pct_downstream_full_20260801"
PYTHON="/data01/home/xuzk/anaconda3/envs/xhquant/bin/python"
MODEL_PATH="/data01/datasets/Qwen3-30B-A3B-Instruct-2507"
CHANNEL_CACHE="$CALIBRATION_ROOT/tail_channels/qwen3_channels_b64_tail_0p50.pt"
CHANNEL_SHA256="ac777898ba9d26b8772692ca4163fee12b183f7901fea595c24e702edecf16ed"

case "$ARM" in
    dense)
        PROFILE="$PROFILE_ROOT/dense_full_width.pt"
        PROFILE_SHA256="6201d3dc15ed5222507b02d47a65a3f486b9985c59609565b37e802aad90c006"
        ;;
    official_reap)
        PROFILE="$PROFILE_ROOT/reap_official_50pct_per_layer.pt"
        PROFILE_SHA256="b99f909f06fd466f278cc9b6a860d720f8ba929a1debc554711ee9d7345b5d80"
        ;;
    route_tail_global)
        PROFILE="$PROFILE_ROOT/route_tail_50pct_global.pt"
        PROFILE_SHA256="c39ec13b816134645a1e0a527045486e191bb226c55f57429f1c7ccce400cbb9"
        ;;
    tail_risk_global)
        PROFILE="$PROFILE_ROOT/tail_risk_50pct_global.pt"
        PROFILE_SHA256="c6d628cef5f04516f85db85a96e4da40f7f4ba0f1bf063f42cae77233e855675"
        ;;
    route_tail_per_layer)
        PROFILE="$PROFILE_ROOT/route_tail_50pct_per_layer.pt"
        PROFILE_SHA256="fadc96dd262c3ff04c5ee3a01772d95a074f88ec8dbbfdf2ca50944d74a25416"
        ;;
    tail_risk_per_layer)
        PROFILE="$PROFILE_ROOT/tail_risk_50pct_per_layer.pt"
        PROFILE_SHA256="326140a86b5c3790fba150847d7f1facbd88df4e2ed0e2f351d2e30c807f6c2e"
        ;;
    *)
        echo "Unknown arm: $ARM" >&2
        exit 2
        ;;
esac

EXTRA_ARGS=()
WORK_DIR="$RESULT_ROOT/$ARM"
if [[ "$MODE" == "preflight" ]]; then
    WORK_DIR="$RESULT_ROOT/preflight/$ARM"
    EXTRA_ARGS+=(--preflight-only)
elif [[ "$MODE" != "run" ]]; then
    echo "Mode must be 'run' or 'preflight'." >&2
    exit 2
fi

DATASET_ARGS='{"mmlu_pro":{"local_path":"/data01/datasets/evalscope_benchmarks/mmlu_pro"},"mmlu":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/mmlu"},"arc":{"local_path":"/data01/datasets/evalscope_benchmarks/arc","subset_list":["ARC-Challenge"]},"hellaswag":{"local_path":"/data01/datasets/evalscope_benchmarks/hellaswag"},"winogrande":{"local_path":"/data01/home/xuzk/workspace/OSTQuant/utils/my_utils/llm_pipeline/datasets/winogrande/data/winogrande_1.1.zip"},"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0},"math_500":{"local_path":"/data01/datasets/evalscope_benchmarks/math_500"},"ifeval":{"local_path":"/data01/datasets/evalscope_benchmarks/ifeval"},"boolq":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/super_glue/boolq"},"openbookqa":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/openbookqa","subset_list":["main"]},"rte":{"local_path":"/data01/home/xuzk/workspace/moe_prune/code/ablation/DiEP/hails/glue/RTE","subset_list":["rte"]}}'

SANDBOX_CONFIG="${SANDBOX_JSON:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LD_LIBRARY_PATH="/data01/home/xuzk/anaconda3/envs/xhquant/lib:${LD_LIBRARY_PATH:-}"
export NUMEXPR_MAX_THREADS=16
export PYTHONPATH="$ROOT:$EXPERIMENT_ROOT/code"

exec "$PYTHON" "$EXPERIMENT_ROOT/code/scripts/run_evalscope_static_profile.py" \
    --model-path "$MODEL_PATH" \
    --model-id "qwen3-50pct-$ARM" \
    --model-family qwen3 \
    --profile "$PROFILE" \
    --channel-cache "$CHANNEL_CACHE" \
    --expected-profile-file-sha256 "$PROFILE_SHA256" \
    --expected-channel-file-sha256 "$CHANNEL_SHA256" \
    --work-dir "$WORK_DIR" \
    --datasets arc hellaswag gsm8k math_500 ifeval mmlu_pro mmlu winogrande boolq openbookqa rte \
    --dataset-args "$DATASET_ARGS" \
    --generation-config '{"max_tokens":1024}' \
    --eval-batch-size 1 \
    --seed 42 \
    --correction-mode none \
    --moe-backend torch_index_add \
    --no-enable-thinking \
    --no-timestamp \
    ${SANDBOX_CONFIG:+--sandbox "$SANDBOX_CONFIG"} \
    "${EXTRA_ARGS[@]}"