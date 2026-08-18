#!/usr/bin/env bash
set -euo pipefail

ROOT=/data01/home/xinpei.gao/evalscope
CODE_ROOT="$ROOT/static_moe_prunning/code"
PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
MODEL_PATH=/data01/datasets/Qwen3-30B-A3B-Instruct-2507
CALIBRATION_CACHE="$ROOT/static_moe_prunning/experiments/calibration/qwen3_mixed_train_wikitext256_mbpp128_gsm8k64_math64_20260802/mixed_train_512x1024_code_augmented.pt"
CALIBRATION_ROOT="$ROOT/static_moe_prunning/experiments/calibration/qwen3_mixed_512x1024_code_augmented_20260802"
PROFILE_ROOT="$ROOT/static_moe_prunning/experiments/profiles/qwen3_mixed_512x1024_code_augmented_20260802"
LOG_ROOT="$ROOT/static_moe_prunning/experiments/logs/qwen3_mixed_512x1024_code_augmented_20260802"
CHANNEL_PID_FILE="$LOG_ROOT/channel_calibration.pid"
CHANNEL_CACHE="$CALIBRATION_ROOT/channels_rms_512x1024.pt"
TAIL_CACHE="$CALIBRATION_ROOT/tail_channels/qwen3_channels_b64_tail_0p50.pt"
TEACHER_CACHE="$CALIBRATION_ROOT/conditional_dual_teacher_512x1024_50pct.pt"
AMP_CACHE="$ROOT/static_moe_prunning/experiments/calibration/static_expert_priors_20260728/amp_scores.pt"
AIMER_CACHE="$ROOT/static_moe_prunning/experiments/calibration/static_expert_priors_20260728/aimer_scores.pt"
REAP_ROOT="$ROOT/reap"
REAP_COMMIT=1970473c51ca3caeb98c10392f15b3a08a672974

export LD_LIBRARY_PATH=/data01/home/xuzk/anaconda3/envs/xhquant/lib
export PYTHONPATH="$CODE_ROOT"

mkdir -p "$CALIBRATION_ROOT" "$PROFILE_ROOT" "$LOG_ROOT"

if [[ -f "$CHANNEL_PID_FILE" ]]; then
    channel_pid=$(cat "$CHANNEL_PID_FILE")
    if kill -0 "$channel_pid" 2>/dev/null; then
        printf 'waiting_for_channel_calibration pid=%s\n' "$channel_pid"
        tail --pid="$channel_pid" -f /dev/null
    fi
fi

[[ -s "$CHANNEL_CACHE" ]] || { echo "missing channel cache: $CHANNEL_CACHE"; exit 1; }
[[ -s "$TAIL_CACHE" ]] || { echo "missing tail cache: $TAIL_CACHE"; exit 1; }

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$CODE_ROOT/scripts/build_official_reap_profile.py" \
    --official-reap-root "$REAP_ROOT" \
    --official-reap-commit "$REAP_COMMIT" \
    --model-path "$MODEL_PATH" \
    --model-family qwen3 \
    --calibration-cache "$CALIBRATION_CACHE" \
    --channel-cache "$CHANNEL_CACHE" \
    --output-observer "$CALIBRATION_ROOT/reap_official_observer_512x1024.pt" \
    --output-profile "$PROFILE_ROOT/reap_official_50pct_per_layer.pt" \
    --experts-to-prune-per-layer 64 \
    --sequence-length 1024 \
    --batch-group-size 8 \
    --device-map cuda:0 \
    >"$LOG_ROOT/official_reap.log" 2>&1 &
reap_pid=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "$CODE_ROOT/scripts/collect_dynamic_regret_teacher.py" \
    --model-path "$MODEL_PATH" \
    --amp-score-cache "$AMP_CACHE" \
    --aimer-score-cache "$AIMER_CACHE" \
    --channel-cache "$CHANNEL_CACHE" \
    --output-cache "$TEACHER_CACHE" \
    --target-pruning-ratio 0.50 \
    --sequence-length 1024 \
    --calibration-sequences 512 \
    --calibration-token-cache "$CALIBRATION_CACHE" \
    --parent-mode dual \
    >"$LOG_ROOT/conditional_dual_teacher.log" 2>&1 &
teacher_pid=$!

"$PYTHON_BIN" "$CODE_ROOT/scripts/build_static_expert_profiles.py" \
    --channel-cache "$TAIL_CACHE" \
    --output-profile "$PROFILE_ROOT/route_tail_50pct_global.pt" \
    --mode route_rms \
    --target-pruning-ratio 0.50 \
    --allocation-scope global \
    >"$LOG_ROOT/route_tail_global.log" 2>&1

wait "$reap_pid"
wait "$teacher_pid"

if [[ ! -s "$PROFILE_ROOT/tail_risk_50pct_global.pt" ]]; then
    "$PYTHON_BIN" "$CODE_ROOT/scripts/build_tail_risk_profile.py" \
        --teacher-cache "$TEACHER_CACHE" \
        --reference-channel-cache "$CHANNEL_CACHE" \
        --tail-channel-cache "$TAIL_CACHE" \
        --output-profile "$PROFILE_ROOT/tail_risk_50pct_global.pt" \
        --target-pruning-ratio 0.50 \
        --allocation-scope global \
        --risk-floor-min-width 2 \
        --risk-floor-early-layers 48 \
        --risk-floor-quantile 0.995 \
        --risk-floor-relative-max 0.10 \
        >"$LOG_ROOT/tail_risk_global.log" 2>&1
fi

"$PYTHON_BIN" - <<'PY'
from hashlib import sha256
from pathlib import Path
import json
import torch

root = Path('/data01/home/xinpei.gao/evalscope')
calibration = root / 'static_moe_prunning/experiments/calibration/qwen3_mixed_train_wikitext256_mbpp128_gsm8k64_math64_20260802/mixed_train_512x1024_code_augmented.pt'
profile_root = root / 'static_moe_prunning/experiments/profiles/qwen3_mixed_512x1024_code_augmented_20260802'
paths = {
    'official_reap': profile_root / 'reap_official_50pct_per_layer.pt',
    'route_tail_global': profile_root / 'route_tail_50pct_global.pt',
    'tail_risk_global': profile_root / 'tail_risk_50pct_global.pt',
}
calibration_payload = torch.load(calibration, map_location='cpu', weights_only=True)
expected_token_sha = calibration_payload['input_ids_sha256']
audit = {'calibration_input_ids_sha256': expected_token_sha, 'profiles': {}}
for name, path in paths.items():
    payload = torch.load(path, map_location='cpu', weights_only=True)
    provenance = payload['cache_provenance']['calibration']
    token_sha = provenance['input_ids_sha256']
    total_blocks = int(payload['total_blocks'])
    if token_sha != expected_token_sha:
        raise ValueError(f'{name} calibration token SHA mismatch')
    if total_blocks != 36864:
        raise ValueError(f'{name} retained {total_blocks} blocks, expected 36864')
    audit['profiles'][name] = {
        'path': str(path),
        'file_sha256': sha256(path.read_bytes()).hexdigest(),
        'profile_sha256': payload['profile_sha256'],
        'total_blocks': total_blocks,
        'maximum_blocks': int(payload['maximum_blocks']),
        'actual_structural_pruning_ratio': float(payload['actual_structural_pruning_ratio']),
        'calibration_input_ids_sha256': token_sha,
        'sequence_length': provenance['sequence_length'],
        'calibration_sequences': provenance['calibration_sequences'],
    }
output = profile_root / 'three_profile_audit.json'
output.write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(output)
PY

echo three_profiles_complete