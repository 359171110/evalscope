# Pure-Pseudo

Pure-Pseudo is a calibration-free channel pruning method for Qwen3 MoE. It reuses WICK's router-derived pseudo
probes, but removes both weight-only ranking and pseudo protection. Every expert channel is ranked directly by its
pseudo-probe output contribution score.

For expert `e`, the method selects its router row and the cosine-nearest `K` router rows, applies the layer RMSNorm,
and sends all `K + 1` probes directly through the target expert's SwiGLU projections. For channel `c`, the score is:

```text
TopQMean_j(abs(SiLU(gate_e(x_j)) * up_e(x_j))[c] * L2(down_e[:, c]))
```

The highest-scoring channels are retained directly. There is no protection ratio and no Random, AIMER, or
weight-path score in the ranking.

## Build The Profile

```bash
cd /data01/home/xinpei.gao/evalscope

export ROOT=$PWD
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
export MODEL_PATH=/data01/datasets/Qwen3-30B-A3B-Instruct-2507
export CODE_ROOT=$ROOT/static_moe_prunning/code
export PYTHONPATH=$ROOT:$CODE_ROOT
export PROFILE_ROOT=$ROOT/PP/experiments/profiles/PurePseudo-K8-Q4

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" PP/build_pure_pseudo_profile.py \
  --model-path "$MODEL_PATH" \
  --output-profile "$PROFILE_ROOT/pure_pseudo_50pct_per_layer.pt" \
  --output-channel-cache "$PROFILE_ROOT/pure_pseudo_rankings.pt" \
  --target-pruning-ratio 0.50 \
  --router-neighbors 8 \
  --top-q 4 \
  --channel-block-size 64 \
  --device cuda:0
```

The profile has fixed per-expert width and is compatible with the shared static-profile validator. The channel cache
contains the full Pure-Pseudo channel ordering used by both the Transformers reference runtime and checkpoint export.

## Profile Preflight

The shared launcher registers the method as `pure_pseudo`. This path is for preflight and semantic checks, not the
formal downstream result when a uniform checkpoint can be exported.

```bash
bash static_moe_prunning/code/scripts/run_downstream_matrix.sh \
  --model-path "$MODEL_PATH" \
  --model-id Qwen330BA3BInstruct-50-PurePseudo-K8-Q4 \
  --model-family qwen3 \
  --pruning-ratio 50pct \
  --gpus 0 \
  --datasets gsm8k \
  --methods pure_pseudo \
  --profile-root "$PROFILE_ROOT" \
  --channel-cache "$PROFILE_ROOT/pure_pseudo_rankings.pt" \
  --results-root "$ROOT/result/pure-pseudo-preflight" \
  --dataset-limits '{"gsm8k":1}' \
  --preflight-only
```

## Export A Uniform Checkpoint

Formal evaluation follows `checkpoint -> vLLM -> EvalScope openai_api`. Create the result directory with the shared
helper and export the 384-channel checkpoint into its `checkpoints/` directory:

```bash
export METHOD=PurePseudo-K8-Q4
export RESULT_ROOT=$ROOT/result
export EXPERIMENT_DIR="$($CODE_ROOT/scripts/create_result_dir.sh \
  --inference vllm \
  --calibration CalibrationFree \
  --method "$METHOD")"
export CHECKPOINT_DIR=$EXPERIMENT_DIR/checkpoints/$METHOD

"$PYTHON_BIN" PP/export_uniform_qwen3_moe.py \
  --model-path "$MODEL_PATH" \
  --profile "$PROFILE_ROOT/pure_pseudo_50pct_per_layer.pt" \
  --channel-cache "$PROFILE_ROOT/pure_pseudo_rankings.pt" \
  --output-dir "$CHECKPOINT_DIR" \
  --retained-channels 384
```

The export manifest records the source model, profile, channel cache, export script, output shard hashes, and every
pruned tensor's source and exported shapes.

Before formal Quick9, complete the framework acceptance gates and update the manifest validation status:

1. Transformers greedy smoke loads the exported checkpoint.
2. vLLM passes `/health`, `/v1/models`, and one chat completion.
3. Transformers and vLLM agree without systematic answer shifts on 5 to 20 frozen prompts.

## Run vLLM Quick9

After the acceptance gates pass:

```bash
export VLLM_PYTHON=/data01/home/xuzk/anaconda3/envs/vllm/bin/python
export GPU_ID=0
export PORT=18080

bash PP/run_vllm_quick9.sh
```

Results are written to the experiment directory created by `create_result_dir.sh`; no new formal results are written
under `PP/experiments/`.

## Tests

```bash
PYTHONPATH=$ROOT:$CODE_ROOT "$PYTHON_BIN" -m pytest \
  PP/tests/test_build_pure_pseudo_profile.py \
  static_moe_prunning/code/test/test_run_downstream_matrix.py -q
```