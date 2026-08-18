# gemma4-vllm-cu128

Frozen vLLM environment used for Gemma4, Qwen3-30B-A3B, and Qwen3.6-35B-A3B serving.
It matches `~/.conda/envs/gemma4-vllm-cu128` on the original server.

Do not upload the 11 GiB conda env or the 12 GiB vLLM build tree. Recreate it with
`setup_gemma4_vllm_cu128.sh`.

## Captured versions

| Item | Value |
| --- | --- |
| Python | 3.10.20 |
| torch | 2.11.0+cu128 |
| vLLM | 0.23.1.dev0 at `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` plus local CUDA 12.8 / Python 3.10 patches |
| transformers | 5.6.2 |
| flashinfer | 0.6.12 |
| Driver | CUDA 12.8 (`nvidia-smi` reports 12.8) |
| Compiler | conda `gcc_linux-64=11.4` |

Supported architectures after install:

- `Qwen3MoeForCausalLM` (Qwen3-30B-A3B)
- `Qwen3_5MoeForConditionalGeneration` (Qwen3.6-35B-A3B)
- `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration`

## New server

Needs a NVIDIA driver that reports CUDA 12.8, `conda`, `git`, and enough disk for the env plus a from-source vLLM build (plan for ~40 GiB while compiling).

```bash
cd eval_protocol/envs/gemma4-vllm-cu128
bash setup_gemma4_vllm_cu128.sh
```

Optional overrides:

```bash
ENV_PREFIX=$HOME/.conda/envs/gemma4-vllm-cu128 \
VLLM_SRC=$HOME/vllm-cu128-src \
TORCH_CUDA_ARCH_LIST='8.0;9.0' \
MAX_JOBS=16 \
bash setup_gemma4_vllm_cu128.sh
```

A100 is `8.0`. Include `9.0` if the new server has H100.

Then:

```bash
cp eval_protocol/env.example.sh eval_protocol/env.sh
# env.example.sh already defaults VLLM_PYTHON to this conda env
source eval_protocol/env.sh
```

Start the server with:

```bash
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --model "$CHECKPOINT_DIR" ...
```

Keep EvalScope / profile export on a separate Python if needed. This env is for vLLM serving.

## Local patches applied on top of upstream vLLM

- Skip the Python 3.11-only `spinloop` extension so the tree builds on 3.10.
- Add `csrc/glibc_compat.cpp` so the CUDA extensions link on older glibc.
- Leave `torch==2.11.0+cu128` to pip/PyTorch index instead of vLLM's un-suffixed `torch==2.11.0` pin.
