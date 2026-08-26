# Routing-aware self-calibrated heterogeneous channel pruning

This folder contains a runnable implementation of the method described in the accompanying design note. It keeps the two statistics separate:

- `natural_mass` and `natural_visitation` are collected only from natural self-generated sequences and then frozen;
- guided sequences are used only to complete expert-conditioned sample pools.

The scoring, conditional damage, exact discrete allocation, and ridge folding operations stay in PyTorch tensors on the adapter device. The default width levels are `100%`, `75%`, `50%`, and `25%` of the original expert width. The output is a logical heterogeneous plan; checkpoint export/runtime dispatch can be added separately for a target inference backend.

## Smoke test

```bash
PYTHONPATH="$PWD" /data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python \
  -m ROUTING_AWARE_HETEROGENEOUS.run_toy --device cuda
```

Use `--device cpu` when CUDA is unavailable. The toy runner does not download a model and does not run evaluation.

## Real model artifact build

This optional entry point loads a local Hugging Face Qwen3/Qwen3.6/Gemma4 model, generates self-calibration sequences, builds pruning artifacts, and saves them. It does not run benchmark evaluation or inference after artifact creation.

```bash
PYTHONPATH="$PWD" /data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python \
  -m ROUTING_AWARE_HETEROGENEOUS.run_model \
  --model-path /path/to/model \
  --output /path/to/pruning_result.pt \
  --device cuda --dtype float16
```

The installed `transformers` version must expose the native model implementation used by the repository's `ROUTER_LAYERPROP` adapters. Keep sequence count and length modest when first validating a new checkpoint.

The real-model path is intentionally memory conservative: calibration uses micro-batches (default `1`), hooks retain at most `max_samples_per_expert` expert-conditioned rows, backbone-only forward is used instead of allocating language-model logits, and generated sequences are moved back to CPU between batches. On a shared GPU, keep `--calibration-batch-size 1 --generation-batch-size 1` until the checkpoint has been validated.

## Checks

```bash
PYTHONPATH="$PWD" /data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python -m py_compile ROUTING_AWARE_HETEROGENEOUS/*.py
PYTHONPATH="$PWD" /data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python -m pytest ROUTING_AWARE_HETEROGENEOUS/tests -q
```