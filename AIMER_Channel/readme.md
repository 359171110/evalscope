# AIMER-Channel Uniform Pruning

## Method

AIMER-Channel 是 AIMER weight-only criterion 的 channel-level structured adaptation。

该方法严格 data-free，不使用 calibration samples、activations 或 gradients。

对于 expert 中第 `c` 个 intermediate channel，将三个 projection 中与该
channel 对应的参数拼接：

w_c =
Concat(
    W_gate[c,:],
    W_up[c,:],
    W_down[:,c]
)

其中：

`w_c ∈ R^N`

---

## Channel Importance

使用 channel-level inverse-AIMER score：

S_AIMER(c) =
RMS(w_c) / (MeanAbs(w_c) + epsilon)

其中：

RMS(w_c) =
sqrt(mean(w_c^2))

MeanAbs(w_c) =
mean(|w_c|)

默认：

`epsilon = 1e-8`

较大的 `S_AIMER` 被认为对应更重要、应优先保留的 channel。

该形式与原始 AIMER 的
`MeanAbs / RMS`
排序关系互为倒数，因此可以方便地统一成：

> importance 越大，越优先保留。

---

## Near-zero Protection

计算 score 时必须避免 near-zero channel 引起数值异常。

推荐：

1. FP32 计算；
2. 如果整个 channel 的能量低于实现规定的 effective-zero threshold，
   将其判为 near-zero；
3. near-zero channel 不允许由于 denominator 数值异常而获得异常高排名。

具体 threshold 必须在 Model-specific Implementation 中记录。

不得根据 downstream 结果调整 threshold。

---

## Selection

对于每个 expert：

1. 构造全部 channel vectors；
2. 计算 `S_AIMER(c)`；
3. 从大到小排序；
4. 保留 Top-K。

其中：

`K = round((1-rho) * d_ff)`

---

## Structured Pruning

对未保留 channel，同步删除：

- `W_gate[c,:]`
- `W_up[c,:]`
- `W_down[:,c]`

---

## Uniform-width

所有 routed experts 必须等宽。

AIMER score 只用于 expert 内 channel ranking。

不得：

- 根据 score 改变 expert width；
- 跨 expert 做 global Top-K；
- 进行 heterogeneous allocation。

---

## Data Requirement

严格禁止使用：

- calibration data
- WikiText
- downstream data
- activations
- gradients
- router statistics

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `AIMER_Channel/`。

通道 importance 来自原先
`static_moe_prunning/code/scripts/build_aimer_channel_profile.py` 的
`channel_aimer_importance`（`score_variant=original`）：对每个 routed expert 的
`concat(W_gate[c,:], W_up[c,:], W_down[:,c])` 在 FP32 中计算

`S(c) = RMS(w_c) / (MeanAbs(w_c) + 1e-8)`。

near-zero 保护：若该 concat 向量的能量（平方和）低于 `1e-12`，分数设为 `-inf`，
因此不会因为分母过小而排到 keep 列表前面。该阈值取自旧实现 `stable_concat` 的
`--effective-zero-threshold` 默认值，不随下游结果调整。

相对旧 `build_aimer_channel_profile.py` 的分配方式：本目录改为与 Random / Magnitude
相同的 **per-expert 等宽 Top-K**，不再做 per-layer heterogeneous block allocation。
旧脚本里的 `gauge_balanced` / `shape` / `stable_concat` 变体不进入本实验。

完整 ranking 写入 `aimer_channel_rankings.pt`；每个预算再保存
`aimer_channel_retained_<K>ch.json`。25% 与 50% 共用同一 ranking cache。

`K = round((1-rho) * d_ff)`，然后按模型 alignment 对齐。只剪 routed expert
channels；router、shared expert、dense MLP、多模态和 MTP 张量保持不变。

评测走 `checkpoint -> vLLM -> EvalScope openai_api`，协议为 `full8_v1`，
校准身份为 `CalibrationFree`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash AIMER_Channel/run_calibration_free_full8.sh all dry-run
RATIO=50 bash AIMER_Channel/run_calibration_free_full8.sh qwen3 prepare
RATIO=50 bash AIMER_Channel/run_calibration_free_full8.sh qwen3 eval
```

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

- `/data01/datasets/Qwen3-30B-A3B-Instruct-2507`
- `/data01/datasets/gemma-4-26B-A4B-it`
- `/data01/datasets/Qwen3.6-35B-A3B`
- `/data01/datasets/DeepSeek-V2-Lite-Chat`

本机评测：

```bash
bash AIMER_Channel/run_one_model_full8.sh qwen3 0 18280
```

测试：

```bash
"$PYTHON_BIN" -m pytest AIMER_Channel/tests -q
```
