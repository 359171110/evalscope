#!/usr/bin/env bash
# Launch HSP 25% evals on idle GPUs so they are not stuck behind 50% on the same card.
set -euo pipefail

ROOT="/home/xinpeigao/evalscope"
LAUNCHER="$ROOT/CSP/run_one_model_hsp_full8.sh"
LOG_DIR="${LOG_DIR:-/data/xinpeigao/evalscope_results/_launchers}"
RESULT_ROOT="${RESULT_ROOT:-/data/xinpeigao/evalscope_results}"
TIMESTAMP="${TIMESTAMP:-202608290144}"
ART="${ART:-/data/xinpeigao/evalscope_results/_artifacts/hsp}"
FILL_LOG="$LOG_DIR/hsp.fill25.out"
GPU_ORDER="${GPU_ORDER:-7 6 5 4 3}"
PORT_BASE="${PORT_BASE:-19820}"

declare -A NAME=(
    [qwen3]=Qwen330BA3BInstruct
    [gemma4]=Gemma4-26B-A4B
    [qwen36]=Qwen3.6-35B-A3B
    [deepseek]=DeepSeek-V2-Lite-Chat
)

log() {
    echo "[$(date -Is)] $*" | tee -a "$FILL_LOG"
}

eval25_done() {
    local dir="$RESULT_ROOT/${NAME[$1]}_25_vllm_CalibrationFree_full8_v1_HSP_${TIMESTAMP}_42" ds
    [[ -d "$dir" ]] || return 1
    for ds in arc hellaswag winogrande gsm8k math_500 mmlu humaneval mbpp; do
        [[ -d "$dir/HSP/$ds/reports" ]] || return 1
    done
}

overflow_alive() {
    local pidfile="$LOG_DIR/hsp.$1.25.pid" pid
    [[ -f "$pidfile" ]] || return 1
    pid="$(cat "$pidfile")"
    kill -0 "$pid" 2>/dev/null
}

gpu_idle() {
    local gpu="$1" pids mem
    pgrep -f "/CSP/run_one_model_hsp_full8.sh [^ ]+ ${gpu} " >/dev/null && return 1
    pgrep -f "/CSP/run_one_model_full8.sh [^ ]+ ${gpu} " >/dev/null && return 1
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)"
    [[ -n "$pids" ]] && return 1
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    [[ "$mem" -lt 4096 ]]
}

launch_25() {
    local model="$1" gpu="$2" port pid
    port=$((PORT_BASE + gpu * 4))
    log "CLAIM GPU ${gpu} ${model} 25% port=${port}"
    RESULT_ROOT="$RESULT_ROOT" TIMESTAMP="$TIMESTAMP" RATIOS=25 \
        PYTHONUNBUFFERED=1 \
        nohup stdbuf -oL -eL bash "$LAUNCHER" "$model" "$gpu" "$port" \
        >"$LOG_DIR/hsp.${model}.25.out" 2>&1 9>&- &
    pid=$!
    echo "$pid" >"$LOG_DIR/hsp.${model}.25.pid"
    log "STARTED ${model} 25% pid=${pid} gpu=${gpu} log=$LOG_DIR/hsp.${model}.25.out"
}

mkdir -p "$LOG_DIR"
log "FILL HSP 25% onto idle GPUs timestamp=$TIMESTAMP order=[$GPU_ORDER]"

while true; do
    all_done=1
    for model in qwen3 gemma4 qwen36 deepseek; do
        if eval25_done "$model"; then
            continue
        fi
        all_done=0
        overflow_alive "$model" && continue
        [[ -f "$ART/$model/csp_rankings.pt" ]] || continue
        for gpu in $GPU_ORDER; do
            gpu_idle "$gpu" || continue
            launch_25 "$model" "$gpu"
            break
        done
    done
    if [[ "$all_done" -eq 1 ]]; then
        log "ALL HSP 25% evals complete; filler exiting"
        exit 0
    fi
    sleep 20
done
