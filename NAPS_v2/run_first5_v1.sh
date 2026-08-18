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
export PYTHONPATH="$REPO_ROOT"
export NUMEXPR_MAX_THREADS=16

env -u LD_LIBRARY_PATH "$PYTHON_BIN" - "$MODEL_ID" "$API_URL" "$METHOD" "$RESULTS_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from NAPS_v2.full6_v1 import DATASETS, EVAL_BATCH_SIZE, SEED, TIMEOUT, generation_config

model_id, api_url, method, results_root = sys.argv[1:]
root = Path(results_root)
for item in DATASETS:
    if item.name == "mmlu":
        continue
    work_dir = root / method / item.name
    if (work_dir / "predictions").exists():
        raise RuntimeError(f"Refusing cache reuse in first5 work dir: {work_dir}")
    command = [
        sys.executable,
        "-m",
        "evalscope.cli.cli",
        "eval",
        "--model",
        model_id,
        "--model-id",
        f"{model_id}-{item.name}",
        "--eval-type",
        "openai_api",
        "--api-url",
        f"{api_url}/v1/chat/completions",
        "--api-key",
        "EMPTY",
        "--datasets",
        item.name,
        "--dataset-args",
        item.dataset_args,
        "--limit",
        str(item.limit),
        "--generation-config",
        json.dumps(generation_config(item.max_tokens)),
        "--eval-batch-size",
        str(EVAL_BATCH_SIZE),
        "--seed",
        str(SEED),
        "--timeout",
        str(TIMEOUT),
        "--work-dir",
        str(work_dir),
        "--no-timestamp",
    ]
    subprocess.run(command, check=True)
PY