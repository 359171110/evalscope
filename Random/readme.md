# Random Uniform Channel Pruning

## Method

Random 是本实验最基本的 data-free structured pruning baseline。

对于每个 routed expert，随机生成 intermediate channel 的排列：

`pi = RandomPermutation({0, ..., d_ff - 1})`

在 pruning ratio 为 `rho` 时，每个 expert 保留：

`K = round((1-rho) * d_ff)`

个 channel。

保留：

`pi[:K]`

删除：

`pi[K:]`

---

## Structured Pruning

若 channel `c` 被删除，则同步删除：

- `W_gate[c, :]`
- `W_up[c, :]`
- `W_down[:, c]`

必须真实重建缩窄后的 expert 权重矩阵。

---

## Uniform-width

每个 expert 必须保留完全相同数量的 channels。

Random 只决定：

> 哪些 channels 被删除。

不能决定：

> 每个 expert 删除多少 channels。

---

## Random Seed

默认：

`seed = 42`

所有实验必须保证 deterministic。

必须保存最终 retained channel indices，以确保 checkpoint 可以完全复现。

如果并行实现可能导致随机数调用顺序变化，应直接保存每个
`layer/expert` 的 selection mask，而不能只依赖全局随机状态。

---

## Data Requirement

Random 为严格 data-free 方法。

禁止读取或使用：

- WikiText calibration set
- downstream dataset
- activation
- router statistics
- gradient

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `Random/`，不再依赖 `WICK/build_random_profiles.py` 的 pseudo-ranking cache。
每个 `(layer, expert)` 使用独立 RNG：

`seed_le = SHA256(f"{seed}:{layer_id}:{expert_id}")[:8]`

因此并行或 25%/50% 复用同一 ranking cache 时，permutation 不随调用顺序变化。
完整 permutation 写入 `random_rankings.pt`；每个预算再额外保存
`random_retained_<K>ch.json`。

`K = round((1-rho) * d_ff)`，然后按模型 alignment 对齐。只剪 routed expert
channels；router、shared expert、dense MLP、多模态和 MTP 张量保持不变。

评测走 `checkpoint -> vLLM -> EvalScope openai_api`，协议为 `full8_v1`，
校准身份为 `CalibrationFree`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash Random/run_calibration_free_full8.sh all dry-run
RATIO=50 bash Random/run_calibration_free_full8.sh qwen3 prepare
RATIO=50 bash Random/run_calibration_free_full8.sh qwen3 eval
```

`prepare` 生成 ranking / profile 并导出 Hugging Face checkpoint；`eval` 启动
vLLM 后跑 `full8_v1`（ARC, HellaSwag, WinoGrande, GSM8K, MATH-500, MMLU,
HumanEval, MBPP）。25% 与 50% 共用同一 `random_rankings.pt`，只改变保留前缀长度。

DeepSeek-V2 的 fused shared MLP 宽度原来绑在 `moe_intermediate_size * n_shared_experts`
上。Random 导出把这两个宽度拆开：routed 张量按保留通道真实切片，
`moe_intermediate_size=K`，`shared_expert_intermediate_size` 写成源 fused 宽度
（Lite-Chat 为 2816），`n_shared_experts` 不变。加载端必须使用会读该字段的
vLLM（见 `eval_protocol/envs/gemma4-vllm-cu128/patches/deepseek_shared_width.patch`）。

| 模型 | `MODEL_NAME` | 布局 | Width / alignment | 25% K | 50% K | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen330BA3BInstruct` | 独立 gate/up/down | 768 / 64 | 576 | 384 | 无 shared expert |
| Gemma4-26B-A4B-it | `Gemma4-26B-A4B` | packed gate-up + down | 704 / 32 | 512 | 352 | 25% 的 `round(0.75*704)=528` 不是 32 对齐，nearest 落到 512 |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B` | packed gate-up + down | 512 / 64 | 384 | 256 | 不剪 shared expert |
| DeepSeek-V2-Lite-Chat | `DeepSeek-V2-Lite-Chat` | 独立 gate/up/down | 1408 / 32 | 1056 | 704 | 跳过 dense layer 0；不剪 shared experts。routed 真实切片；`moe_intermediate_size` 写成保留宽度，`shared_expert_intermediate_size` 写成 fused shared 宽度 `1408*2=2816`。需要 `--trust-remote-code`，并使用会读这两个字段的 vLLM |

默认本地路径：

- `/data01/datasets/Qwen3-30B-A3B-Instruct-2507`
- `/data01/datasets/gemma-4-26B-A4B-it`
- `/data01/datasets/Qwen3.6-35B-A3B`
- `/data01/datasets/DeepSeek-V2-Lite-Chat`

可用 `QWEN3_MODEL_PATH` / `GEMMA4_MODEL_PATH` / `QWEN36_MODEL_PATH` /
`DEEPSEEK_MODEL_PATH` 覆盖。Qwen3.6 若要用多卡，设置 `QWEN36_GPU=4,5`，launcher
会按 GPU 个数传 `--tensor-parallel-size`。

测试：

```bash
"$PYTHON_BIN" -m pytest Random/tests -q
```