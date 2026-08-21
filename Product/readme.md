# Product Uniform Channel Pruning

## Method

Product 是 data-free、weight-only structured channel pruning baseline。

对于一个 SwiGLU expert 的 intermediate channel `c`，只使用 gate / up 的
channel L2 乘积作为 importance：

```text
S_prod(c) = ||W_gate[c,:]||_2 * ||W_up[c,:]||_2
```

`W_down[:,c]` 不进入 ranking。较大的 score 优先保留。

这接近 GateUp50 的 gate×up 能量，但这里按 **per-expert 等宽 Top-K** 做 structured
prune，和 Magnitude / AIMER-Channel 同一套导出流程。

---

## Selection

对每个 expert 独立计算全部 `d_ff` 个 channel score，按从大到小排序，保留 Top-K：

`K = round((1-rho) * d_ff)`，再按模型 alignment 对齐。

---

## Structured Pruning

删除 channel `c` 时仍同步删除：

- `W_gate[c,:]`
- `W_up[c,:]`
- `W_down[:,c]`

不剪 router、shared expert、dense MLP、多模态或 MTP 张量。

---

## Uniform-width

Product 只负责 expert 内 ranking。所有 routed experts 等宽。25% 与 50% 共用同一
`product_rankings.pt`。

---

## Data Requirement

严格 data-free：只用 pretrained weights。禁止 calibration / WikiText /
activations / gradients / router statistics。校准身份 `CalibrationFree`。

---

## Pruning Ratios

必须测试 25% 和 50%。

---

## Model-specific Implementation

实现位于 `Product/`。方法 token 为 `Product`。评测走
`checkpoint -> vLLM -> EvalScope openai_api`，协议 `full8_v1`。

DeepSeek-V2 的 fused shared MLP 宽度拆分与 Magnitude 相同：routed 按保留通道切片，
`moe_intermediate_size=K`，`shared_expert_intermediate_size` 写成源 fused 宽度
（Lite-Chat 为 2816）。

| 模型 | `MODEL_NAME` | 布局 | Width / alignment | 25% K | 50% K | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen330BA3BInstruct` | 独立 gate/up/down | 768 / 64 | 576 | 384 | 无 shared expert |
| Gemma4-26B-A4B-it | `Gemma4-26B-A4B` | packed gate-up + down | 704 / 32 | 512 | 352 | 25% 的 `round(0.75*704)=528` 不是 32 对齐，nearest 落到 512 |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B` | packed gate-up + down | 512 / 64 | 384 | 256 | 不剪 shared expert |
| DeepSeek-V2-Lite-Chat | `DeepSeek-V2-Lite-Chat` | 独立 gate/up/down | 1408 / 32 | 1056 | 704 | 跳过 dense layer 0；不剪 shared experts |

默认本地路径：

- `/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507`
- `/data/xinpeigao/models/gemma-4-26B-A4B-it`
- `/data/xinpeigao/models/Qwen3.6-35B-A3B`
- `/data/xinpeigao/models/DeepSeek-V2-Lite-Chat`

Artifacts 写到 `/data/xinpeigao/evalscope_results/_artifacts/product/<model>/`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash Product/run_calibration_free_full8.sh all dry-run
bash Product/run_one_model_full8.sh qwen3 2 18880
```

测试：

```bash
"$PYTHON_BIN" -m pytest Product/tests -q
```
