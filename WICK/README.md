# WICK

This directory implements the data-free method specified in `DESIGN.md`:

1. Rank aligned SwiGLU channels by the geometric mean of gate, up, and down L2 norms.
2. Build expert-specific pseudo probes from each router row and its cosine Top-K neighbors.
3. Apply the layer RMSNorm to those probes and score channel responses with Top-Q mean output contribution.
4. Protect the highest pseudo-response channels, then retain the remaining channels by weight-only rank.

The current runtime uses 64-channel blocks. At 50% pruning on Qwen3-30B-A3B-Instruct-2507, every expert retains
6 of 12 blocks. The channel cache stores the WICK channel order, while the standard static profile stores the fixed
expert widths.

## Build

```bash
cd /data01/home/xinpei.gao/evalscope

export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
export MODEL_PATH=/data01/datasets/Qwen3-30B-A3B-Instruct-2507
export PROFILE_ROOT=$PWD/WICK/experiments/profiles/qwen3_wick_gram_protect_20260806
export PYTHONPATH=$PWD:$PWD/static_moe_prunning/code

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" WICK/build_wick_profile.py \
  --model-path "$MODEL_PATH" \
  --output-profile "$PROFILE_ROOT/wick_gram_protect_50pct_per_expert.pt" \
  --output-channel-cache "$PROFILE_ROOT/wick_gram_protect_rankings.pt" \
  --target-pruning-ratio 0.50 \
  --protection-ratio 0.10 \
  --router-neighbors 8 \
  --top-q 4 \
  --channel-block-size 64 \
  --device cuda:0
```

The generated 2026-08-06 artifacts are:

| Artifact | SHA256 |
| --- | --- |
| `wick_gram_protect_50pct_per_expert.pt` | `5eaff87e5fc0c35e6675bfd6f75cdfd1dc990a05c6081c4d224c13ee675ea674` |
| `wick_gram_protect_rankings.pt` | `7f85bd613c1af0e1f9fff7bd89c04b7ab64ee8836856542b35d9ca8a03d8cbf0` |

Audit values:

- Profile shape: `48 x 128` experts.
- Width: every expert retains `6 / 12` blocks.
- Total retained blocks: `36,864 / 73,728`.
- Structural pruning ratio: `0.5`.
- Protected channels: `77 / 768` per expert for `protection_ratio=0.10`.

## Validate

```bash
PYTHONPATH=$PWD:$PWD/static_moe_prunning/code "$PYTHON_BIN" -m pytest \
  WICK/tests/test_build_wick_profile.py \
  static_moe_prunning/code/test/test_run_downstream_matrix.py \
  static_moe_prunning/code/test/test_evalscope_model_api.py -q
```

Run the shared artifact preflight on physical GPU2:

```bash
bash static_moe_prunning/code/scripts/run_downstream_matrix.sh \
  --model-path "$MODEL_PATH" \
  --model-id qwen3-wick-gram-protect-smoke \
  --model-family qwen3 \
  --pruning-ratio 50pct \
  --gpus 2 \
  --datasets gsm8k \
  --methods wick_gram_protect \
  --profile-root "$PROFILE_ROOT" \
  --channel-cache "$PROFILE_ROOT/wick_gram_protect_rankings.pt" \
  --results-root WICK/experiments/results/qwen3_wick_gram_protect_smoke_20260806 \
  --dataset-args '{"gsm8k":{"local_path":"/data01/datasets/evalscope_benchmarks/gsm8k","few_shot_num":0}}' \
  --dataset-limits '{"gsm8k":1}' \
  --generation-config '{"max_tokens":1024}' \
  --eval-batch-size 1 \
  --seed 42 \
  --correction-mode none \
  --max-correction-ratio 0.20 \
  --moe-backend torch_index_add \
  --preflight-only
```

Remove `--preflight-only` for the one-sample downstream smoke. The completed smoke produced structural and routed
pruning ratios of `0.5` in every layer. Its single GSM8K output reached the 1024-token cap through repetition, so the
score is only an execution check and is not evidence about method quality.

## Quick9

The single-GPU launcher follows the frozen Quick9 order, limits, and dataset-specific generation lengths. It runs the
six independent dataset shards sequentially on physical GPU2 and reuses compatible prediction caches when restarted.

```bash
GPU_ID=2 bash WICK/run_wick_quick9.sh
```

Results are written under `WICK/experiments/results/qwen3_wick_gram_protect_quick9_20260806`. Set `DRY_RUN=true` to
print all six commands or `PREFLIGHT_ONLY=true` to validate all shards without evaluating them.