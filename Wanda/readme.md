# Structured MoE Wanda Uniform Channel Pruning

## Method

Wanda 是 train-calibrated structured channel pruning baseline。

对于一个 SwiGLU expert 的 intermediate channel `c`，定义联合 channel：

`C_c = {W_gate[c,:], W_up[c,:], W_down[:,c]}`

校准阶段在 WikiText-2 train 上跑冻结的 128×2048 token cache，记录 routed expert 的输入
二阶矩 `RMS_p(x)` 与 gated 中间响应二阶矩 `RMS_p(z)`。默认用 native router 概率作为
observation weight（`mass`）。

importance：

S_Wanda(c) =
sqrt(
    ||W_gate[c,:] ⊙ RMS_p(x)||_2^2
  + ||W_up[c,:] ⊙ RMS_p(x)||_2^2
  + RMS_p(z_c)^2 ||W_down[:,c]||_2^2
)

较大的 score 表示该 channel 在校准观测下更重要。校准中从未被路由到的 expert 使用
确定性的 coupled weight-L2 fallback，artifact 会记录 unseen expert 数量。

---

## Selection

对于每个 expert 独立计算全部 `d_ff` 个 channel score，按 score 从大到小排序。

在 pruning ratio `rho` 下保留 Top-K：

`K = round((1-rho) * d_ff)`

然后按模型 alignment 对齐。

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
- 根据 Wanda score 给不同 experts 分配不同 width。

25% 从 50% profile 克隆 ranking，只改变 Top-K 前缀长度。

---

## Data Requirement

Wanda 必须使用冻结的 train-only WikiText 校准 cache。

禁止：

- 用 downstream / test split 做校准
- 用评测分数回头改 ranking
- 剪 router、shared expert、dense MLP、多模态或 MTP 张量

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `Wanda/`。每个 routed expert 用 grouped Wanda score 得到完整 permutation，
写入 `channels.pt`；每个预算再保存 `wanda_<ratio>pct_per_layer.pt`。

校准身份为 `WikiText128x2048`（128 sequences × 2048 tokens，train split）。每个模型
必须用自己的 tokenizer 建 cache。评测走 `checkpoint -> vLLM -> EvalScope openai_api`，
协议为 `full8_v1`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash Wanda/run_wikitext128x2048_full8.sh all dry-run
RATIO=50 bash Wanda/run_wikitext128x2048_full8.sh qwen3 prepare
RATIO=50 bash Wanda/run_wikitext128x2048_full8.sh qwen3 eval
```

`prepare` 建 WikiText cache、收集 routed statistics、生成 ranking / profile 并导出
Hugging Face checkpoint；`eval` 启动 vLLM 后跑 `full8_v1`（ARC, HellaSwag, WinoGrande,
GSM8K, MATH-500, MMLU, HumanEval, MBPP）。

DeepSeek-V2 的 fused shared MLP 宽度原来绑在 `moe_intermediate_size * n_shared_experts`
上。Wanda 导出把这两个宽度拆开：routed 张量按保留通道真实切片，
`moe_intermediate_size=K`，`shared_expert_intermediate_size` 写成源 fused 宽度
（Lite-Chat 为 2816），`n_shared_experts` 不变。加载端必须使用会读该字段的
vLLM（见 `eval_protocol/envs/gemma4-vllm-cu128/patches/deepseek_shared_width.patch`）。

| 模型 | `MODEL_NAME` | 布局 | Width / alignment | 25% K | 50% K | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen330BA3BInstruct` | 独立 gate/up/down | 768 / 64 | 576 | 384 | 无 shared expert |
| Gemma4-26B-A4B-it | `Gemma4-26B-A4B` | packed gate-up + down | 704 / 32 | 512 | 352 | 25% 的 `round(0.75*704)=528` 不是 32 对齐，nearest 落到 512 |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B` | packed gate-up + down | 512 / 64 | 384 | 256 | 不剪 shared expert。校准加载 `ForConditionalGeneration` |
| DeepSeek-V2-Lite-Chat | `DeepSeek-V2-Lite-Chat` | 独立 gate/up/down | 1408 / 32 | 1056 | 704 | 跳过 dense layer 0；不剪 shared experts。routed 真实切片；`moe_intermediate_size` 写成保留宽度，`shared_expert_intermediate_size` 写成 fused shared 宽度 `1408*2=2816`。需要 `--trust-remote-code`，并使用会读这两个字段的 vLLM |

默认本地路径：

- `/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507`
- `/data/xinpeigao/models/gemma-4-26B-A4B-it`
- `/data/xinpeigao/models/Qwen3.6-35B-A3B`
- `/data/xinpeigao/models/DeepSeek-V2-Lite-Chat`

可用 `QWEN3_MODEL_PATH` / `GEMMA4_MODEL_PATH` / `QWEN36_MODEL_PATH` /
`DEEPSEEK_MODEL_PATH` 覆盖。Qwen3.6 若要用多卡，设置 `QWEN36_GPU=4,5`，launcher
会按 GPU 个数传 `--tensor-parallel-size`。校准 cache 默认写到
`/data/xinpeigao/evalscope_results/_artifacts/wanda/<model>/calibration.pt`，
数据源为本地 `Salesforce/wikitext` / `wikitext-2-raw-v1` train split。

本机评测（同一张卡先 collect 再 50% / 25% eval）：

```bash
bash Wanda/run_one_model_full8.sh qwen3 2 18580
```

测试：

```bash
"$PYTHON_BIN" -m pytest Wanda/tests -q
```
