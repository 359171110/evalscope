# Structured MoE Wanda

Train-calibrated, structured Wanda pruning for routed MoE experts. Only those
channels are pruned; routers, shared experts, dense MLPs, multimodal modules,
and auxiliary/MTP tensors are preserved.

## Quick start (WikiText128x2048 + full6_v1)

```bash
cd /path/to/evalscope
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python
export VLLM_PYTHON=/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

# Inspect frozen paths, GPUs, and result names.
RATIO=50 bash Wanda/run_wikitext128x2048_full6.sh all dry-run

# Collect statistics, build the profile, and export a Hugging Face checkpoint.
RATIO=50 QWEN3_GPU=1 bash Wanda/run_wikitext128x2048_full6.sh qwen3 prepare
RATIO=50 GEMMA4_GPU=2 bash Wanda/run_wikitext128x2048_full6.sh gemma4 prepare
RATIO=50 QWEN36_GPU=4 bash Wanda/run_wikitext128x2048_full6.sh qwen36 prepare

# Evaluate the exported checkpoint with vLLM + full6_v1.
RATIO=50 QWEN3_GPU=1 bash Wanda/run_wikitext128x2048_full6.sh qwen3 eval
RATIO=50 GEMMA4_GPU=2 bash Wanda/run_wikitext128x2048_full6.sh gemma4 eval
RATIO=50 QWEN36_GPU=4 bash Wanda/run_wikitext128x2048_full6.sh qwen36 eval
```

Set `RATIO=25` for the 25% width. Keep the same `TIMESTAMP` across `prepare`
and `eval` so results land in one experiment directory. Gemma4 needs a
tokenizer-compatible WikiText cache first:

```bash
bash Wanda/run_wikitext128x2048_full6.sh gemma4 cache
```

Formal scores must come from `eval` (exported checkpoint + vLLM), not the
profile runtime.

Supported models:

| Model | Routed tensor layout | Width / alignment | 50% retained width |
| --- | --- | ---: | ---: |
| Qwen3-30B-A3B-Instruct-2507 | Separate gate/up/down per expert | 768 / 64 | 384 |
| Gemma4-26B-A4B | Packed gate-up and down | 704 / 32 | 352 |
| Qwen3.6-35B-A3B | Packed gate-up and down | 512 / 64 | 256 |

Only routed expert channels are pruned. Routers, shared experts, dense MLPs, multimodal modules, and auxiliary/MTP tensors are preserved.

## Method

A removable MoE channel is the coupled structural group consisting of gate row $c$, up row $c$, and down column $c$. For routed expert input $x$, intermediate response $z$, and route observation weight $p$, the default score is:

$$
S_c = \sqrt{
\lVert W_g[c,:] \odot \operatorname{RMS}_p(x) \rVert_2^2 +
\lVert W_u[c,:] \odot \operatorname{RMS}_p(x) \rVert_2^2 +
\operatorname{RMS}_p(z_c)^2 \lVert W_d[:,c] \rVert_2^2
}.
$$

The default `mass` mode uses native router probabilities as observation weights. `none` and `square` are explicit alternatives. An expert with no train calibration observations uses a deterministic coupled weight-L2 fallback, and the artifact records the number of such experts.

## Build artifacts

Use a frozen framework token cache built from a train split. Each model requires its own tokenizer-compatible cache.

```bash
export ROOT=/data01/home/xinpei.gao/evalscope
export PYTHONPATH="$ROOT:$ROOT/static_moe_prunning/code"
export PYTHON_BIN=/data01/home/xuzk/anaconda3/envs/xhquant/bin/python

"$PYTHON_BIN" -m Wanda.collect_wanda_statistics \
  --model-path "$MODEL_PATH" \
  --calibration-cache "$CALIBRATION_CACHE" \
  --output "$ARTIFACT_DIR/statistics.pt" \
  --sequence-length 2048 \
  --calibration-sequences 128 \
  --route-weighting mass

"$PYTHON_BIN" -m Wanda.build_wanda_artifacts \
  --model-path "$MODEL_PATH" \
  --statistics "$ARTIFACT_DIR/statistics.pt" \
  --output-channel-cache "$ARTIFACT_DIR/channels.pt" \
  --output-profile "$ARTIFACT_DIR/wanda_50pct_per_layer.pt" \
  --target-pruning-ratio 0.5
```

For Gemma4, 25% nominal pruning requests 528 retained channels, which is not divisible by the frozen 32-channel alignment. The builder reports the aligned physical width and actual pruning ratio; use `--retained-channels` to make the chosen physical budget explicit.

## WikiText 50% + full6_v1

The frozen three-model comparison uses `WikiText128x2048`, 50% routed-expert width, exported Hugging Face checkpoints, and `full6_v1`:

```bash
bash Wanda/run_wikitext128x2048_full6.sh all dry-run
bash Wanda/run_wikitext128x2048_full6.sh gemma4 cache
bash Wanda/run_wikitext128x2048_full6.sh qwen3 collect   # GPU 1, xhquant
bash Wanda/run_wikitext128x2048_full6.sh gemma4 collect  # GPU 2, gemma4-vllm-cu128
bash Wanda/run_wikitext128x2048_full6.sh qwen36 collect  # GPU 4, gemma4-vllm-cu128
bash Wanda/run_wikitext128x2048_full6.sh qwen3 prepare   # build + export after collect
```

Each model keeps its own tokenizer-compatible WikiText cache. Formal scores must come from `eval` (vLLM + `full6_v1`), not the profile runtime.

## Export and evaluate

Uniform profiles are exported to a standard Hugging Face checkpoint and must use the vLLM evaluation path required by the framework manual.

```bash
"$PYTHON_BIN" -m Wanda.export_wanda_checkpoint \
  --model-path "$MODEL_PATH" \
  --profile "$ARTIFACT_DIR/wanda_50pct_per_layer.pt" \
  --channel-cache "$ARTIFACT_DIR/channels.pt" \
  --output-dir "$CHECKPOINT_DIR"
```

Before formal evaluation, run a Transformers greedy smoke, vLLM health/model/chat checks, and a frozen-prompt Transformers-vLLM consistency check. Gemma4 vLLM must use `/data01/home/xinpei.gao/.conda/envs/gemma4-vllm-cu128/bin/python`. Run Quick9 through the generic protocol launcher and store results under the repository `result/` directory using the mandated experiment naming scheme.

The profile launcher accepts method `wanda` for profile preflight/reference runs. Formal downstream scores must come from the exported checkpoint and vLLM, not the profile runtime.

## Tests

```bash
"$PYTHON_BIN" -m pytest Wanda/tests -q
"$PYTHON_BIN" -m pytest \
  static_moe_prunning/code/test/test_static_expert_pruning.py \
  static_moe_prunning/code/test/test_evalscope_model_api.py \
  static_moe_prunning/code/test/test_run_downstream_matrix.py -q
```