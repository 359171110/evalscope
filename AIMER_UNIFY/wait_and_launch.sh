#!/usr/bin/env bash
# Wait until GPUs 0/1/2 are idle, then launch Unify full8 on qwen3 / gemma4 / qwen36.
# Does not touch GPUs 3-7.
set -euo pipefail

ROOT="/home/xinpeigao/evalscope"
LAUNCHER="$ROOT/AIMER_UNIFY/run_one_model_full8.sh"
LOG_DIR="${LOG_DIR:-/data/xinpeigao/evalscope_results/_launchers}"
export RESULT_ROOT="${RESULT_ROOT:-/home/xinpeigao/evalscope/results}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
export METHOD_TOKEN="${METHOD_TOKEN:-AIMERUnify}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-19890}"
export PYTHONUNBUFFERED=1
IDLE_MIB="${IDLE_MIB:-3072}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$LOG_DIR"

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

port_for() {
    printf '%s\n' "$(( MASTER_PORT_BASE + $1 ))"
}

model_for() {
    case "$1" in
        0) printf 'qwen3\n' ;;
        1) printf 'gemma4\n' ;;
        2) printf 'qwen36\n' ;;
        *) echo "ERROR: Unify wait script only claims GPUs 0-2 (got $1)" >&2; return 1 ;;
    esac
}

already_running() {
    local model="$1"
    pgrep -f "/AIMER_UNIFY/run_one_model_full8.sh ${model} " >/dev/null
}

claim_and_launch() {
    local gpu="$1"
    local model port mem log
    model="$(model_for "$gpu")"
    port="$(port_for "$gpu")"
    log="$LOG_DIR/unify.${model}.out"
    while true; do
        if already_running "$model"; then
            echo "[$(date -Is)] SKIP GPU ${gpu} ${model} already running"
            return 0
        fi
        mem="$(gpu_used_mib "$gpu")"
        if [[ "$mem" -lt "$IDLE_MIB" ]]; then
            if ss -ltn | grep -q ":${port} "; then
                echo "[$(date -Is)] ERROR port ${port} busy; not launching ${model}" >&2
                return 1
            fi
            echo "[$(date -Is)] CLAIM GPU ${gpu} mem=${mem}MiB port=${port} ${model} -> ${log}"
            RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" METHOD_TOKEN="$METHOD_TOKEN" \
                MASTER_PORT_BASE="$MASTER_PORT_BASE" PYTHONUNBUFFERED=1 \
                nohup stdbuf -oL -eL bash "$LAUNCHER" "$model" "$gpu" "$port" \
                >"$log" 2>&1 &
            echo "[$(date -Is)] STARTED ${model} pid=$! log=${log}"
            return 0
        fi
        echo "[$(date -Is)] WAIT GPU ${gpu} ${model} mem=${mem}MiB"
        sleep "$POLL_SECONDS"
    done
}

echo "[$(date -Is)] WAIT Unify on GPUs 0/1/2 idle_mib=$IDLE_MIB timestamp=$TIMESTAMP method=$METHOD_TOKEN"
status=0
pids=()
for gpu in 0 1 2; do
    claim_and_launch "$gpu" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
echo "[$(date -Is)] WAIT dispatcher done status=${status} timestamp=$TIMESTAMP"
exit "$status"
