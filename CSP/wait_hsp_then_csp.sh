#!/usr/bin/env bash
# After all HSP full8 model jobs finish, launch CSP full8 on GPUs 3-7.
set -euo pipefail

ROOT="/home/xinpeigao/evalscope"
LOG_DIR="${LOG_DIR:-/data/xinpeigao/evalscope_results/_launchers}"
RESULT_ROOT="${RESULT_ROOT:-/data/xinpeigao/evalscope_results}"
CHAIN_LOG="$LOG_DIR/hsp_then_csp.out"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
    echo "[$(date -Is)] $*" | tee -a "$CHAIN_LOG"
}

hsp_running() {
    pgrep -f "/CSP/run_one_model_hsp_full8.sh (qwen3|gemma4|qwen36|deepseek) " >/dev/null
}

mkdir -p "$LOG_DIR"
log "WAIT for HSP model jobs to finish before CSP"

while hsp_running; do
    log "HSP still running pids=$(pgrep -f '/CSP/run_one_model_hsp_full8.sh (qwen3|gemma4|qwen36|deepseek) ' | tr '\n' ' ')"
    sleep "$POLL_SECONDS"
done

status=0
for model in qwen3 gemma4 qwen36 deepseek; do
    logf="$LOG_DIR/hsp.${model}.out"
    if ! grep -q " ALL DONE ${model} status=0" "$logf" 2>/dev/null; then
        log "WARN HSP ${model} did not report ALL DONE status=0"
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    log "CSP not started because some HSP model jobs failed"
    exit 1
fi

export TIMESTAMP="${CSP_TIMESTAMP:-$(date +%Y%m%d%H%M)}"
export RESULT_ROOT
export METHOD_TOKEN=CSP
export GPU_ORDER="${GPU_ORDER:-3 4 5 6 7}"
rm -f "$LOG_DIR/csp.wait.out"
log "START CSP dispatcher timestamp=$TIMESTAMP gpus=[$GPU_ORDER]"
nohup bash "$ROOT/CSP/wait_and_launch.sh" >>"$LOG_DIR/csp.wait.out" 2>&1 &
log "CSP dispatcher pid=$! log=$LOG_DIR/csp.wait.out"
exit 0
