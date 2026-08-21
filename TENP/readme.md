# Uniform ENP-COS Channel Pruning

## Method

ENP 指 TENP 论文中的 Expert Neuron Pruning。

本实验不使用 TENP 的：

- important-expert selection
- trapezoidal layer-wise budget allocation

只保留其核心 neuron importance criterion（ENP-COS），并对所有 routed experts
执行相同 pruning ratio。因此本 baseline 是 **uniform-width ENP**，不是完整 TENP。

---

## Calibration Dataset

固定使用冻结的 train-only WikiText cache：`wiki128x2048`

- 128 calibration sequences
- sequence length = 2048
- split = train

每个模型必须用自己的 tokenizer 建 cache。禁止用 downstream / test split 做校准，
也禁止用评测分数回头改 ranking。

---

## Expert Forward

对于一个 SwiGLU expert：

```text
m = act(W_gate x) ⊙ (W_up x)
y = W_down m
```

其中 `x ∈ R^(T × d_model)`，`m ∈ R^(T × d_ff)`，`y ∈ R^(T × d_model)`。
`act` 取该模型 native expert activation（Qwen3 / Qwen3.6 / DeepSeek 为 SiLU；
Gemma4 为 `gelu_pytorch_tanh`）。

---

## Per-neuron Output Contribution

对于 intermediate neuron/channel `c`，其独立产生的 output contribution 为：

```text
C_c,t = m[t,c] * W_down[:,c]
y_t   = Σ_c C_c,t
```

---

## ENP-COS Importance

ENP 使用单 neuron output 在完整 expert output 方向上的投影长度作为 importance。

```text
P_c,t   = <C_c,t, y_t> / (||y_t||_2 + epsilon)
S_ENP(c) = Mean_t P_c,t
```

`epsilon = 1e-8`。较大的 score 优先保留。默认且本实验唯一使用的变体是投影
**ENP-COS**，不用 ENP-L2 替代。

Mean 只对实际 routed 到该 expert 的 **unique calibration tokens** 计算。
不得把全部 WikiText tokens 强制送入每个 expert，不得改 routing。

若某个 expert 的 unique routed token 数为 0，使用确定性的 coupled weight-L2
fallback（`sqrt(||W_gate[c,:]||_2^2 + ||W_up[c,:]||_2^2 + ||W_down[:,c]||_2^2)`），
并在 artifact 中记录 `unseen_experts` / `unseen_expert_ids`。禁止静默改 scoring rule。

---

## Selection

对于每个 expert：

1. 收集 routed calibration hidden states；
2. 计算 native gated intermediate `m`；
3. 计算完整 expert output `y`；
4. 计算每个 channel 的 projection contribution；
5. 对 unique routed tokens 求平均；
6. 按 score 从大到小排序；
7. 保留 Top-K。

`K = round((1-rho) * d_ff)`，然后按模型 alignment 对齐。

---

## Structured Pruning

删除 channel `c` 时同步删除：

- `W_gate[c,:]`
- `W_up[c,:]`
- `W_down[:,c]`

不剪 router、shared expert、dense MLP、多模态或 MTP 张量。

---

## Uniform-width

ENP 只负责 channel ranking，不负责 width allocation。

所有 routed experts：

- 使用相同 pruning ratio；
- 在同一层原始 width 相同时保留相同 K；
- 不保留所谓 important experts 的完整宽度。

25% 从 50% profile 克隆 ranking，只改变 Top-K 前缀长度。

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `TENP/`。校准身份为 `WikiText128x2048`。评测走
`checkpoint -> vLLM -> EvalScope openai_api`，协议为 `full8_v1`，方法名 `ENP`。

完整 ranking 写入 `channels.pt`；每个预算再保存 `enp_<ratio>pct_per_layer.pt`。

DeepSeek-V2 的 fused shared MLP 宽度原来绑在 `moe_intermediate_size * n_shared_experts`
上。ENP 导出把这两个宽度拆开：routed 张量按保留通道真实切片，
`moe_intermediate_size=K`，`shared_expert_intermediate_size` 写成源 fused 宽度
（Lite-Chat 为 2816），`n_shared_experts` 不变。加载端必须使用会读该字段的
vLLM（见 `eval_protocol/envs/gemma4-vllm-cu128/patches/deepseek_shared_width.patch`）。
不要复用 8 月 19 日把 shared MLP 压成 K 的旧 DeepSeek ENP checkpoint。

| 模型 | expert input | native routing | activation | projection 矩阵 | token 累积 | zero-token fallback | tensor slicing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B-Instruct-2507 | gate `forward_pre_hook` 截获 hidden | gate 输出 logits 或 `(logits, weights, indices)`，按 native `top_k` / `norm_topk_prob` | SiLU SwiGLU | 独立 `down_proj.weight`，`[hidden, d_ff]` | unique routed rows；score **sum** 后除以 count 得 mean | coupled weight-L2 | 切 `gate_proj`/`up_proj` 行与 `down_proj` 列 |
| Gemma4-26B-A4B-it | `experts` `forward_pre_hook` 的第 0 个参数 | 第 1/2 个参数即 native `top_k_index` / `top_k_weights`，不改 router | `gelu_pytorch_tanh` | packed `down_proj[expert]` | 同上 | 同上 | packed `gate_up_proj` 按 channel 同步切 gate/up 两半，再切 `down_proj` 列 |
| Qwen3.6-35B-A3B | Qwen gate hook，同 Qwen3 | `Qwen3_5MoeSparseMoeBlock.gate` native top-k | SiLU SwiGLU | packed `experts.down_proj` | 同上 | 同上 | packed 切片同 Gemma4；不剪 `shared_expert`。校准加载 `ForConditionalGeneration` |
| DeepSeek-V2-Lite-Chat | `DeepseekV2MoE` `forward_pre_hook` 的 hidden | `gate.weight` linear + `route_tokens_to_experts`，或 native `gate()` 返回的 indices/weights | SiLU SwiGLU | 独立 routed `down_proj.weight` | 同上；跳过 dense layer 0 | 同上 | 只切 routed experts；shared fused 宽度保持 `1408*2=2816` |

| 模型 | `MODEL_NAME` | 布局 | Width / alignment | 25% K | 50% K | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen330BA3BInstruct` | 独立 gate/up/down | 768 / 64 | 576 | 384 | 无 shared expert |
| Gemma4-26B-A4B-it | `Gemma4-26B-A4B` | packed gate-up + down | 704 / 32 | 512 | 352 | 25% 的 `round(0.75*704)=528` 不是 32 对齐，nearest 落到 512 |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B` | packed gate-up + down | 512 / 64 | 384 | 256 | 不剪 shared expert |
| DeepSeek-V2-Lite-Chat | `DeepSeek-V2-Lite-Chat` | 独立 gate/up/down | 1408 / 32 | 1056 | 704 | 跳过 dense layer 0；不剪 shared experts。需要 `--trust-remote-code` |

默认本地路径：

- `/data/xinpeigao/models/Qwen3-30B-A3B-Instruct-2507`
- `/data/xinpeigao/models/gemma-4-26B-A4B-it`
- `/data/xinpeigao/models/Qwen3.6-35B-A3B`
- `/data/xinpeigao/models/DeepSeek-V2-Lite-Chat`

可用 `QWEN3_MODEL_PATH` / `GEMMA4_MODEL_PATH` / `QWEN36_MODEL_PATH` /
`DEEPSEEK_MODEL_PATH` 覆盖。WikiText cache 优先复用 Wanda 已建好的
`/data/xinpeigao/evalscope_results/_artifacts/wanda/<model>/calibration.pt`；
DeepSeek 若没有 Wanda cache，则在 ENP artifact 目录现建一份当前 tokenizer 的 128×2048 cache。
ENP statistics / ranking / checkpoint 写到
`/data/xinpeigao/evalscope_results/_artifacts/enp/<model>/`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash TENP/run_wikitext128x2048_full8.sh all dry-run
RATIO=50 bash TENP/run_wikitext128x2048_full8.sh qwen3 prepare
RATIO=50 bash TENP/run_wikitext128x2048_full8.sh qwen3 eval
```

本机评测（同一张卡先 collect 再 50% / 25% eval）：

```bash
bash TENP/run_one_model_full8.sh qwen3 0 18680
```

测试：

```bash
"$PYTHON_BIN" -m pytest TENP/tests -q
```
