# AIMER-Mix Uniform Pruning

## Method

AIMER-Mix 是为缓解 AIMER-Channel 在 Gemma 上 MATH/code 崩塌而设计的
**calibration-free** channel ranking。它不按模型名切换，也不写死 `α=0.5`。

对每个 routed expert，先算三类投影的 channel L2，再取 expert 均值：

```text
Ē_g = mean_c ||W_gate[c,:]||_2
Ē_u = mean_c ||W_up[c,:]||_2
Ē_d = mean_c ||W_down[:,c]||_2
α   = min(Ē_g, Ē_u, Ē_d) / max(Ē_g, Ē_u, Ē_d)
```

若 `max(Ē) < 1e-8`，则 `α = 1.0`。

AIMER 项与 `AIMER_Channel` 相同：concat
`[W_gate[c,:], W_up[c,:], W_down[:,c]]` 后

```text
S_AIMER(c) = RMS(w_c) / (MeanAbs(w_c) + 1e-8)
```

near-zero concat 能量 `< 1e-12` 记为 `-inf`。

默认能量项是 geom

```text
S_geom(c) = (||g_c|| ||u_c|| ||d_c||)^{1/3}
```

可选 `--energy-mode l2` 对齐 Magnitude 的 coupled L2。不要用 Product /
GateUp50 / Hoyer / CV / 熵 / AM-GM / whitened-only AIMER，也不要按 CHANNEL
去调 `α`。

在 rank 空间混合（1 = 最重要，ties 取平均秩；AIMER `-inf` 的秩为 0）：

```text
s = α · rank(AIMER) + (1-α) · rank(geom 或 L2)
```

然后对每个 expert 按 `s` 从大到小做 **uniform-width Top-K**。本实验不加
NAPS-v2 PP。

落地时 `α` 由权重自己决定，预期（不是代码开关）：

- Gemma4：`α ≈ 0.43`，更偏 Magnitude，避免纯 AIMER 把 MATH/code 打崩
- Qwen3 / Qwen3.6：`α ≈ 0.87–0.90`，keep 集约 98% 仍是 AIMER

build 时每层打印并写入 `mean_alpha`，导出后可核对。

---

## Structured Pruning

删除 channel `c` 时同步删除：

- `W_gate[c,:]`
- `W_up[c,:]`
- `W_down[:,c]`

不剪 router、shared expert、dense MLP、多模态或 MTP 张量。

---

## Uniform-width

Mix score 只用于 expert 内 ranking。所有 routed experts 等宽。不得：

- 根据 score 改变 expert width
- 跨 expert 做 global Top-K
- 进行 heterogeneous allocation

25% 与 50% 共用同一 `aimer_mix_rankings.pt`，只改 Top-K 前缀长度。

---

## Data Requirement

严格禁止使用 calibration data、WikiText、downstream data、activations、
gradients 或 router statistics。校准身份为 `CalibrationFree`。

---

## Pruning Ratios

必须测试：

- 25%
- 50%

---

## Model-specific Implementation

实现位于 `AIMER_Mix/`。方法 token 为 `AIMERMix`。评测走
`checkpoint -> vLLM -> EvalScope openai_api`，协议 `full8_v1`。

DeepSeek-V2 的 fused shared MLP 宽度拆分与 Magnitude / AIMER-Channel 相同：
routed 按保留通道切片，`moe_intermediate_size=K`，
`shared_expert_intermediate_size` 写成源 fused 宽度（Lite-Chat 为 2816）。

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

Artifacts 写到 `/data/xinpeigao/evalscope_results/_artifacts/aimer_mix/<model>/`。

```bash
cd /path/to/evalscope
export PYTHONPATH="$PWD:$PWD/static_moe_prunning/code"

RATIO=50 bash AIMER_Mix/run_calibration_free_full8.sh all dry-run
RATIO=25 bash AIMER_Mix/run_one_model_full8.sh qwen3 2 18780
RATIO=25 bash AIMER_Mix/run_one_model_full8.sh gemma4 3 18781
```

默认 `ENERGY_MODE=geom`。对齐 Magnitude keep 集时用 `ENERGY_MODE=l2`。

测试：

```bash
"$PYTHON_BIN" -m pytest AIMER_Mix/tests -q
```
