#!/usr/bin/env bash

set -euo pipefail

WANDA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PID="${1:-}"
MODEL="${2:-}"
GPU="${3:-}"
PORT="${4:-}"
TIMESTAMP="${5:-}"
RATIO="${RATIO:-25}"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ -n "$WAIT_PID" && -n "$MODEL" && -n "$GPU" && -n "$PORT" && -n "$TIMESTAMP" ]] ||
    die "Usage: $0 WAIT_PID MODEL GPU PORT TIMESTAMP"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

gpu_env_name() {
    case "$MODEL" in
        qwen3) printf 'QWEN3_GPU\n' ;;
        gemma4) printf 'GEMMA4_GPU\n' ;;
        qwen36) printf 'QWEN36_GPU\n' ;;
        *) die "Unknown model '$MODEL'." ;;
    esac
}

port_env_name() {
    case "$MODEL" in
        qwen3) printf 'QWEN3_PORT\n' ;;
        gemma4) printf 'GEMMA4_PORT\n' ;;
        qwen36) printf 'QWEN36_PORT\n' ;;
        *) die "Unknown model '$MODEL'." ;;
    esac
}

log "waiting for $MODEL 50% eval pid=$WAIT_PID before RATIO=$RATIO eval on GPU $GPU port $PORT"
wait_status=0
if kill -0 "$WAIT_PID" 2>/dev/null; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 20
    done
    wait "$WAIT_PID" 2>/dev/null || wait_status=$?
else
    log "pid $WAIT_PID already gone"
fi
log "50% eval pid $WAIT_PID exited with status $wait_status"

for _ in $(seq 1 90); do
    if pgrep -af "vllm.entrypoints.openai.api_server" | grep -E -- "--port ${PORT}( |$)" >/dev/null; then
        sleep 5
        continue
    fi
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | awk '{print $1}')"
    if [[ "${mem:-999999}" -lt 2048 ]]; then
        break
    fi
    sleep 5
done

# Wait until the 25% checkpoint exists even if export is still running.
artifact_ready="false"
for _ in $(seq 1 360); do
    manifest="$(
        RATIO="$RATIO" TIMESTAMP="$TIMESTAMP" \
            bash "$WANDA_ROOT/run_wikitext128x2048_full6.sh" "$MODEL" dry-run |
            awk -F= '/^experiment=/{print $2}'
    )/checkpoints/Wanda/pruning_export_manifest.json"
    if [[ -f "$manifest" ]]; then
        artifact_ready="true"
        break
    fi
    log "waiting for 25% checkpoint: $manifest"
    sleep 20
done
[[ "$artifact_ready" == "true" ]] || die "Timed out waiting for $MODEL 25% checkpoint."

gpu_key="$(gpu_env_name)"
port_key="$(port_env_name)"
log "starting $MODEL RATIO=$RATIO eval on GPU $GPU port $PORT timestamp=$TIMESTAMP"
env \
    RATIO="$RATIO" \
    TIMESTAMP="$TIMESTAMP" \
    "$gpu_key=$GPU" \
    "$port_key=$PORT" \
    bash "$WANDA_ROOT/run_wikitext128x2048_full6.sh" "$MODEL" eval
log "$MODEL RATIO=$RATIO eval finished"
