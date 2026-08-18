# ENP / TENP Reproduction

This directory reproduces the two algorithms in
*TENP: Trapezoidal Expert Neuron Pruning for Mixture-of-Experts*:

- **ENP** ranks each routed expert's SwiGLU channels by signed output
  projection and keeps one common width in every routed expert. Uniform ENP
  exports a standard Hugging Face checkpoint and is evaluated with vLLM.
- **TENP** additionally ranks routed experts and keeps more full-width experts
  in deeper layers. The remaining experts stay routable but use narrower
  ENP-ranked channel prefixes. Heterogeneous TENP cannot be expressed as a
  uniform HF checkpoint, so it stays on the profile runtime.

Only routed expert channels are pruned. Routers, shared experts, dense MLPs,
and multimodal modules are left unchanged.

Framework integration:

```text
static_moe_prunning/code/src/enp_tenp.py
static_moe_prunning/code/scripts/build_enp_tenp_profiles.py
static_moe_prunning/code/scripts/export_uniform_enp_qwen3_moe.py
```

The exporter handles both Qwen3 separate expert tensors and packed
Gemma4 / Qwen3.6 `gate_up_proj` / `down_proj` tensors.

## WikiText128x2048 + full6_v1 (ENP)

This is the current comparison protocol. Calibration is the frozen train-only
WikiText cache `128 x 2048`. Results are written under `result/` with the
framework naming scheme.

| Model | Launcher | 25% retained | 50% retained | Collect Python |
| --- | --- | ---: | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | `TENP/run_enp_wikitext128x2048.sh` | 576 / 768 | 384 / 768 | `xhquant` |
| Gemma4-26B-A4B | `TENP/run_gemma4_enp_wikitext128x2048.sh` | 512 / 704 | 384 / 704 | `gemma4-vllm-cu128` |

Gemma4 expert width 704 is not divisible into exact 25%/50% at 64-channel
blocks, so the launcher keeps 8 and 6 blocks (512 and 384 channels).

### Qwen3 ENP

```bash
cd /path/to/evalscope
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
export VLLM_PYTHON=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python

bash TENP/run_enp_wikitext128x2048.sh dry-run
BUILD_GPU=1 bash TENP/run_enp_wikitext128x2048.sh build
bash TENP/run_enp_wikitext128x2048.sh export
EVAL_GPU_25=1 bash TENP/run_enp_wikitext128x2048.sh eval-25
EVAL_GPU_50=1 bash TENP/run_enp_wikitext128x2048.sh eval-50
```

### Gemma4 ENP

Use one idle GPU. Sequential eval on GPU 6:

```bash
export VLLM_PYTHON=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
export TIMESTAMP=$(date +%Y%m%d%H%M)

BUILD_GPU=6 bash TENP/run_gemma4_enp_wikitext128x2048.sh dry-run
BUILD_GPU=6 bash TENP/run_gemma4_enp_wikitext128x2048.sh build
TIMESTAMP="$TIMESTAMP" bash TENP/run_gemma4_enp_wikitext128x2048.sh export
EVAL_GPU_25=6 TIMESTAMP="$TIMESTAMP" bash TENP/run_gemma4_enp_wikitext128x2048.sh eval-25
EVAL_GPU_50=6 TIMESTAMP="$TIMESTAMP" bash TENP/run_gemma4_enp_wikitext128x2048.sh eval-50
```

`build` loads Gemma4 with `--device-map none --device cuda` in the
`gemma4-vllm-cu128` environment. Formal scores must come from the exported
checkpoint and vLLM `full6_v1`, not the profile runtime.

Override `MODEL_PATH`, `CALIBRATION_CACHE`, `BUILD_GPU`, `EVAL_GPU_25`,
`EVAL_GPU_50`, `PORT_25`, `PORT_50`, and `RESULT_ROOT` as needed.

## Paper mixed-cache TENP reproduction

The paper-level defaults are routed-expert retention `0.60`, TENP full-expert
ratio `0.30`, trapezoid weights `1.0 / 2.0`, and block size `64`. Calibration
is the frozen mixed train cache `512 x 1024`.

```bash
bash TENP/run_qwen3_enp_tenp_reproduction.sh dry-run
BUILD_GPU=1 bash TENP/run_qwen3_enp_tenp_reproduction.sh build
GPUS_CSV=1,3 bash TENP/run_qwen3_enp_tenp_reproduction.sh preflight
GPUS_CSV=1,3 bash TENP/run_qwen3_enp_tenp_reproduction.sh eval
```

That launcher still uses the profile runtime because TENP widths are
heterogeneous.

## Tests

```bash
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python

"$PYTHON_BIN" -m pytest \
  static_moe_prunning/code/test/test_enp_tenp.py \
  static_moe_prunning/code/test/test_export_uniform_enp_qwen3_moe.py \
  static_moe_prunning/code/test/test_run_gemma4_enp_wikitext128x2048.py \
  static_moe_prunning/code/test/test_run_enp_wikitext128x2048.py -q
```

## Paper ambiguities

The paper does not publish its per-layer trapezoid counts. TENP uses linear
depth weights and largest-remainder integer allocation. At 64-channel runtime
granularity, an indivisible remaining budget is distributed by stable
expert-score order, so narrow experts can differ by at most one block while
the total routed-expert budget remains exact.
