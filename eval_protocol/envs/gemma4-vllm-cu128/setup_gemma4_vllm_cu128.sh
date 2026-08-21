#!/usr/bin/env bash
# Recreate the frozen Gemma4 / Qwen3 / Qwen3.6 vLLM environment on a new server.
# This env was captured from ~/.conda/envs/gemma4-vllm-cu128 on CUDA driver 12.8.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/gemma4-vllm-cu128}"
VLLM_SRC="${VLLM_SRC:-$HOME/vllm-cu128-src}"
VLLM_REMOTE="${VLLM_REMOTE:-https://github.com/vllm-project/vllm.git}"
VLLM_COMMIT="$(tr -d '[:space:]' < "$SCRIPT_DIR/vllm.commit")"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

need_cmd conda
need_cmd git
need_cmd nvidia-smi

echo "Creating conda env at $ENV_PREFIX"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create -p "$ENV_PREFIX" \
        python=3.10 pip \
        gcc_linux-64=11.4 gxx_linux-64=11.4 \
        sysroot_linux-64=2.17 kernel-headers_linux-64=3.10 \
        -y
else
    echo "Conda env already exists, reusing it."
fi

PYTHON="$ENV_PREFIX/bin/python"
PIP=("$PYTHON" -m pip)

export CC="${CC:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export PATH="$ENV_PREFIX/bin:$PATH"

[[ -x "$PYTHON" ]] || die "python not found: $PYTHON"
[[ -x "$CC" ]] || die "conda gcc not found: $CC"

"${PIP[@]}" install -U pip wheel
"${PIP[@]}" install 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0'
"${PIP[@]}" install -r "$SCRIPT_DIR/requirements.txt"

if [[ ! -d "$VLLM_SRC/.git" ]]; then
    git clone "$VLLM_REMOTE" "$VLLM_SRC"
fi
git -C "$VLLM_SRC" fetch --tags origin
git -C "$VLLM_SRC" checkout --detach "$VLLM_COMMIT"
git -C "$VLLM_SRC" reset --hard "$VLLM_COMMIT"
git -C "$VLLM_SRC" apply --whitespace=nowarn "$SCRIPT_DIR/patches/vllm-cu128-py310.patch"
git -C "$VLLM_SRC" apply --whitespace=nowarn "$SCRIPT_DIR/patches/deepseek_shared_width.patch"
cp "$SCRIPT_DIR/patches/glibc_compat.cpp" "$VLLM_SRC/csrc/glibc_compat.cpp"

"${PIP[@]}" install -e "$VLLM_SRC" --no-build-isolation

"$PYTHON" - <<'PY'
import torch
import vllm
import transformers
print(f"python ok")
print(f"torch {torch.__version__} cuda {torch.version.cuda} available={torch.cuda.is_available()}")
print(f"vllm {vllm.__version__}")
print(f"transformers {transformers.__version__}")
from vllm.model_executor.models.registry import _VLLM_MODELS
for name in ("Qwen3MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration", "Gemma4ForCausalLM"):
    print(f"  {name}: {'yes' if name in _VLLM_MODELS else 'NO'}")
PY

echo
echo "Done. Point eval_protocol/env.sh at:"
echo "  export VLLM_ENV=$ENV_PREFIX"
echo "  export VLLM_PYTHON=$ENV_PREFIX/bin/python"
