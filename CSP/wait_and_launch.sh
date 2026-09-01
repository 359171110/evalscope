#!/usr/bin/env bash
# Wait for idle GPUs, then launch CSP full8 (50% then 25%) on four models.
# Prefer GPUs 3-7; do not steal GPUs 0-2 unless they become idle later.
# Scoring/export is serialized by CSP/run_one_model_full8.sh (csp.score.lock);
# vLLM evals may still overlap on different GPUs.
set -euo pipefail

ROOT="/home/xinpeigao/evalscope"
LAUNCHER="$ROOT/CSP/run_one_model_full8.sh"
LOG_DIR="${LOG_DIR:-/data/xinpeigao/evalscope_results/_launchers}"
export RESULT_ROOT="${RESULT_ROOT:-/data/xinpeigao/evalscope_results}"
export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M)}"
export METHOD_TOKEN="${METHOD_TOKEN:-CSP}"
export PYTHONUNBUFFERED=1

POLL_SECONDS="${POLL_SECONDS:-20}"
IDLE_STREAK_NEED="${IDLE_STREAK_NEED:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-4096}"
MAX_UTIL="${MAX_UTIL:-10}"
GPU_ORDER="${GPU_ORDER:-3 4 5 6 7 0 1 2}"
MODELS="${MODELS:-qwen3 gemma4 qwen36 deepseek}"
PORT_BASE="${PORT_BASE:-19720}"

mkdir -p "$LOG_DIR"
LOCK="$LOG_DIR/csp.dispatch.lock"
DISPATCH_LOG="$LOG_DIR/csp.wait.out"

log() {
    echo "[$(date -Is)] $*" | tee -a "$DISPATCH_LOG"
}

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

gpu_compute_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$1" 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' || true
}

port_for_gpu() {
    printf '%s\n' "$(( PORT_BASE + $1 * 4 ))"
}

port_free() {
    local port="$1"
    if ss -ltn | grep -q ":${port} "; then
        return 1
    fi
    return 0
}

model_running() {
    pgrep -f "/CSP/run_one_model_full8.sh ${1} " >/dev/null
}

gpu_assigned() {
    local gpu="$1"
    [[ -n "${CLAIMED_GPUS[$gpu]:-}" ]] && return 0
    pgrep -f "/CSP/run_one_model_full8.sh [^ ]+ ${gpu} " >/dev/null
}

gpu_looks_idle() {
    local gpu="$1"
    local pids mem util
    if gpu_assigned "$gpu"; then
        return 1
    fi
    pids="$(gpu_compute_pids "$gpu")"
    if [[ -n "$pids" ]]; then
        return 1
    fi
    mem="$(gpu_used_mib "$gpu")"
    util="$(gpu_util "$gpu")"
    [[ "$mem" -lt "$MAX_USED_MIB" && "$util" -le "$MAX_UTIL" ]]
}

next_model() {
    local model
    for model in $MODELS; do
        if model_running "$model"; then
            continue
        fi
        if grep -q " STARTED ${model} " "$DISPATCH_LOG" 2>/dev/null; then
            continue
        fi
        printf '%s\n' "$model"
        return 0
    done
    return 1
}

launch_on_gpu() {
    local gpu="$1"
    local model port logf
    model="$(next_model)" || return 1
    port="$(port_for_gpu "$gpu")"
    if ! port_free "$port"; then
        log "SKIP GPU ${gpu} port ${port} busy for ${model}"
        return 1
    fi
    if ! gpu_looks_idle "$gpu"; then
        log "SKIP GPU ${gpu} raced busy before launch"
        return 1
    fi
    logf="$LOG_DIR/csp.${model}.out"
    log "CLAIM GPU ${gpu} mem=$(gpu_used_mib "$gpu")MiB util=$(gpu_util "$gpu")% port=${port} ${model} ts=${TIMESTAMP} -> ${logf}"
    RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" METHOD_TOKEN="$METHOD_TOKEN" \
        PYTHONUNBUFFERED=1 \
        nohup stdbuf -oL -eL bash "$LAUNCHER" "$model" "$gpu" "$port" \
        >"$logf" 2>&1 9>&- &
    CLAIMED_GPUS[$gpu]="$model"
    log "STARTED ${model} pid=$! gpu=${gpu} port=${port} log=${logf}"
}

declare -A STREAK=()
declare -A CLAIMED_GPUS=()
for gpu in $GPU_ORDER; do
    STREAK[$gpu]=0
done

log "WAIT idle GPUs order=[$GPU_ORDER] models=[$MODELS] timestamp=$TIMESTAMP poll=${POLL_SECONDS}s streak=$IDLE_STREAK_NEED method=$METHOD_TOKEN"
log "idle = no compute PIDs AND mem<${MAX_USED_MIB}MiB AND util<=${MAX_UTIL}%"

while true; do
    pending=0
    for model in $MODELS; do
        if model_running "$model"; then
            continue
        fi
        if grep -q " STARTED ${model} " "$DISPATCH_LOG" 2>/dev/null; then
            continue
        fi
        pending=$((pending + 1))
    done
    if [[ "$pending" -eq 0 ]]; then
        log "ALL models launched or already running; dispatcher exiting"
        exit 0
    fi

    for gpu in $GPU_ORDER; do
        if gpu_looks_idle "$gpu"; then
            STREAK[$gpu]=$(( STREAK[$gpu] + 1 ))
        else
            STREAK[$gpu]=0
            continue
        fi
        if [[ "${STREAK[$gpu]}" -lt "$IDLE_STREAK_NEED" ]]; then
            log "IDLE-CANDIDATE GPU ${gpu} streak=${STREAK[$gpu]}/${IDLE_STREAK_NEED} mem=$(gpu_used_mib "$gpu")MiB"
            continue
        fi
        exec 9>"$LOCK"
        if flock -n 9; then
            if launch_on_gpu "$gpu"; then
                STREAK[$gpu]=0
            else
                STREAK[$gpu]=0
            fi
            flock -u 9
        fi
        exec 9>&-
        pending=0
        for model in $MODELS; do
            if model_running "$model"; then
                continue
            fi
            if grep -q " STARTED ${model} " "$DISPATCH_LOG" 2>/dev/null; then
                continue
            fi
            pending=$((pending + 1))
        done
        [[ "$pending" -eq 0 ]] && break
    done
    sleep "$POLL_SECONDS"
done

log "ALL models launched or already running; dispatcher exiting"
exit 0
