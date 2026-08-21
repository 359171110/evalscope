    Q                      # Magnitude Uniform Channel Pruning

## Method

Magnitude 是 data-free、weight-only structured channel pruning baseline。

对于一个 SwiGLU expert 的 intermediate channel `c`，定义联合 channel：

`C_c = {W_gate[c,:], W_up[c,:], W_down[:,c]}`

使用三个 projection 对应权重的联合 L2 magnitude 作为 importance：

S_mag(c) =
sqrt(
    ||W_gate[c,:]||_2^2
  + ||W_up[c,:]||_2^2
  + ||W_down[:,c]||_2^2
)

较大的 score 表示该 channel 的参数整体幅值更大，因此认为更重要。

---

## Selection

对于每个 expert 独立计算全部 `d_ff` 个 channel score：

`S_mag(0), ..., S_mag(d_ff-1)`

然后按 score 从大到小排序。

在 pruning ratio `rho` 下保留 Top-K：

`K = round((1-rho) * d_ff)`

`S_keep = TopK(S_mag, K)`

其余 channels 被删除。

---

## Structured Pruning

删除 channel `c` 时同步删除：

- gate projection 第 `c` 行
- up projection 第 `c` 行
- down projection 第 `c` 列

---

## Uniform-width

每个 routed expert 独立排序，但是必须保留相同的 K。

不允许：

- layer-global Top-K；
- expert-global Top-K；
- 根据 magnitude 给不同 experts 分配不同 width。

---

## Data Requirement

Magnitude 为严格 data-free 方法。

importance 只允许访问 pretrained weights。

禁止使用：

- calibration samples
- activation
- router outputs
- gradients

---

## Numerical Implementation

importance 建议使用 FP32 accumulator：

1. 将需要计算 norm 的权重转换为 FP32；
2. 计算三部分 squared L2 norm；
3. 求和后开平方。

排序完成后再对原始 dtype 权重执行结构化 slicing。

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `Magnitude/`。每个 routed expert 独立计算 FP32 coupled L2 magnitude，再按
score 从大到小排序。完整 ranking 写入 `magnitude_rankings.pt`；每个预算再额外保存
`magnitude_retained_<K>ch.json`。25% 与 50% 共用同一 ranking cache，只改变 Top-K 前缀长度。

`K = round((1-rho) * d_ff)`，然后按模型 alignment 对齐。只剪 routed expert
channels；router、shared expert、dense MLP、多模态和 MTP 张量保持不变。

评测走 `checkpoint -> vLLM -> EvalScope openai_api`，协议为 `full8_v1`，
校准身份为 `CalibrationFree`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash Magnitude/run_calibration_free_full8.sh all dry-run
RATIO=50 bash Magnitude/run_calibration_free_full8.sh qwen3 prepare
RATIO=50 bash Magnitude/run_calibration_free_full8.sh qwen3 eval
```

`prepare` 从预训练权重生成 ranking / profile 并导出 Hugging Face checkpoint；`eval`
启动 vLLM 后跑 `full8_v1`（ARC, HellaSwag, WinoGrande, GSM8K, MATH-500, MMLU,
HumanEval, MBPP）。

DeepSeek-V2 的 fused shared MLP 宽度原来绑在 `moe_intermediate_size * n_shared_experts`
上。Magnitude 导出把这两个宽度拆开：routed 张量按保留通道真实切片，
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

本机评测：

```bash
bash Magnitude/run_one_model_full8.sh qwen3 0 18280
```

测试：

```bash
"$PYTHON_BIN" -m pytest Magnitude/tests -q
```