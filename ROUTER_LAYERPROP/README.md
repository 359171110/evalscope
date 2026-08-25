# Router-conditioned Multi-origin LayerProp

This folder contains a data-free implementation of:

- router-conditioned probes in raw residual coordinates;
- source-0 native long-bank propagation;
- stride-based short refresh banks with a bounded propagation horizon;
- target-local router-region coverage;
- route-weighted output-energy channel initialization;
- recoverability-aware local swaps;
- held-out ridge residual folding with a trust region;
- uniform physical export for Qwen3, Qwen3.6, and Gemma4 routed experts.

The implementation does not modify the original model while constructing a plan. Shared experts, dense branches, attention modules, and auxiliary branches are preserved. The default `long_short` mode uses a router-conditioned source-0 long bank, Qwen3/Qwen3.6 refresh origins at the configured stride, and target-local fallback coverage. Each refresh origin is propagated through native decoder layers for at most `refresh_horizon` layers, so Qwen3.6 linear/full attention mixing remains native. Gemma4 automatically falls back to the stable `source0_long + target_local` mode because stateless intermediate refresh propagation would require copying PLE/shared-KV state.

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

The default synthetic budget is 2048 pseudo tokens packed as sequences of length 32. `--retained-channels` overrides the ratio and is rounded down to the requested channel multiple. The default multiple is 64; use 128 when the target inference kernel requires it. Use `--propagation-mode stable` to reproduce the original vocabulary-lattice native-forward plus local implementation. `--refresh-stride 4` and `--refresh-horizon 8` are the default long-short settings.

## Export a checkpoint

```bash
PYTHONPATH="$PWD" python -m ROUTER_LAYERPROP.export_checkpoint \
  --model-path /path/to/original/checkpoint \
  --plan /path/to/artifacts/router-layerprop-plan.pt \
  --output-dir /path/to/exported/checkpoint
```

The exporter narrows routed experts only. It writes `router_layerprop_export_manifest.json` and updates `moe_intermediate_size`. Run the native model parity and downstream inference checks on the experiment server after export.

## Scope and limitations

The selection stage is a recoverability-aware local search, not a proof of global combinatorial optimality. The fixed keep set's ridge folding is the closed-form least-squares step. Full Gemma4 block replay validation is intentionally left to the target server because it depends on the installed Transformers implementation and model-specific PLE/shared-KV settings. The long-short Qwen path is implemented by calling the installed native decoder layers directly from a synthetic post-attention bank; it does not hand-approximate attention or linear attention.

## Local checks

```bash
python -m py_compile ROUTER_LAYERPROP/*.py
PYTHONPATH="$PWD" pytest ROUTER_LAYERPROP/tests -q
```

The tests use small fake modules and tensors only; they do not perform real experiments or download models.
