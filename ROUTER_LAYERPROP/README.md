# Router-conditioned Multi-origin LayerProp

This folder contains a data-free implementation of:

- router-conditioned probes in raw residual coordinates;
- source-0 native long-bank propagation;
- target-local router-region coverage;
- route-weighted output-energy channel initialization;
- recoverability-aware local swaps;
- held-out ridge residual folding with a trust region;
- uniform physical export for Qwen3, Qwen3.6, and Gemma4 routed experts.

The implementation does not modify the original model while constructing a plan. Shared experts, dense branches, attention modules, and auxiliary branches are preserved. Gemma4 local banks are used for target-layer scoring only; stateless intermediate refresh propagation is disabled for that family.

## Build a plan

Use a PyTorch/Transformers environment on the experiment server:

```bash
PYTHONPATH="$PWD" python -m ROUTER_LAYERPROP.build_plan \
  --model-path /path/to/original/checkpoint \
  --output-plan /path/to/artifacts/router-layerprop-plan.pt \
  --device cuda \
  --pruning-ratio 0.5 \
  --overwrite
```

The default synthetic budget is 2048 pseudo tokens packed as sequences of length 32. `--retained-channels` overrides the ratio and is rounded down to the requested channel multiple. The default multiple is 64; use 128 when the target inference kernel requires it.

## Export a checkpoint

```bash
PYTHONPATH="$PWD" python -m ROUTER_LAYERPROP.export_checkpoint \
  --model-path /path/to/original/checkpoint \
  --plan /path/to/artifacts/router-layerprop-plan.pt \
  --output-dir /path/to/exported/checkpoint
```

The exporter narrows routed experts only. It writes `router_layerprop_export_manifest.json` and updates `moe_intermediate_size`. Run the native model parity and downstream inference checks on the experiment server after export.

## Scope and limitations

The selection stage is a recoverability-aware local search, not a proof of global combinatorial optimality. The fixed keep set's ridge folding is the closed-form least-squares step. Full Gemma4 block replay validation is intentionally left to the target server because it depends on the installed Transformers implementation and model-specific PLE/shared-KV settings.

## Local checks

```bash
python -m py_compile ROUTER_LAYERPROP/*.py
PYTHONPATH="$PWD" pytest ROUTER_LAYERPROP/tests -q
```

The tests use small fake modules and tensors only; they do not perform real experiments or download models.
