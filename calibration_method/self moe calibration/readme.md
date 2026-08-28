# Checkpoint-Native MoE Self-Calibration

## 1. 方法定位

本目录实现的是 **Checkpoint-Native MoE Self-Calibration（CN-MoE-SC）** 的生成与检查部分。

它的目标不是生成主题均衡、语言均衡或“看起来高质量”的文本，而是从待分析 checkpoint 自身产生 token，并测量这些 token 在模型中的 MoE 行为：

1. 每一层每个 expert 的命中频率；
2. router 对各 expert 的 routing mass；
3. expert 被命中之后，各 channel 的条件激活强度；
4. channel activation 的稳定性、相关性和潜在冗余。

目标统计可以写成：

$$
P(e_i\mid x),
\qquad
E[a_{i,c}^{2}\mid e_i\text{ is hit}],
\qquad
E[a_i a_i^{\mathsf T}\mid e_i\text{ is hit}].
$$

这些统计服务于后续的 expert 重要性判断、channel selection、width allocation 和冗余分析。文本是否像一篇完整文章不是本方法的主要评价标准。

本方法仍然是 **external-data-free**：不引入 WikiText、benchmark 样本或人工主题语料；只使用 checkpoint 自带的 tokenizer、chat template 和 control tokens。它不是 calibration-free，而是 calibration data 由模型自身生成。

---

## 2. 当前实现状态

| 能力 | 当前状态 |
|---|---|
| 从 checkpoint 的 native chat template 构造 scaffold | 已实现 |
| v1 `user_role_continuation` | 已实现 |
| v2 `assistant_bootstrap` | 已实现 |
| `temperature=1`、`top_p=1` 的最小采样 | 已实现 |
| `ignore_eos=False`、`min_tokens=0` | 已实现 |
| 独立 episode 生成 | 已实现 |
| 机械退化 gate | 已实现 |
| user 在 assistant 之前的独立 gate | 已实现 |
| clarification/refusal/list/fixed-format 诊断 | 已实现为启发式统计 |
| 固定长度 cache packing | 已实现 |
| episode/block boundary provenance | 已实现 |
| 现有 Wanda/ENP loader 兼容的 `[1, N]` cache | 已实现 |
| episode-aware MoE collector | **本目录未实现** |
| Natural Discovery / Coverage Reserve 的实际统计隔离 | **目前只有 metadata** |
| split-half expert/channel stability | **未实现** |
| High/Medium/Low confidence mask | **未实现** |
| router-guided conditional fallback | **未实现** |

最重要的限制是：当前 cache 虽然记录了 episode boundary，但现有 Wanda/ENP collector 仍按固定长度 block forward。只有 collector 读取 boundary 并逐 episode forward，才能完全满足 independent-episode 的统计语义。

---

## 3. 之前遇到的问题

### 3.1 BOS-only 固定长度生成退化

旧方法采用：

$$
\text{BOS}\rightarrow2048\text{ tokens},
$$

并设置 `ignore_eos=True`、`min_tokens=2047`。这会让模型无法自然结束。模型完成一个短回答或进入文本边界后，只能继续生成，容易掉入重复吸引子。

典型现象包括：

* 单字符或单词重复；
* 短片段循环；
* 标点和模板符号循环；
* 多语言碎片循环；
* 生成内容与正常 instruction operating format 不一致。

Qwen3 在 BOS-only continuation 下相对稳定，能生成代码、技术文档和教程；Qwen3.6、Gemma4 更容易退化。因此不能把“128 条 token 行互不相同”当成 calibration quality 的证据。

### 3.2 user 内容退化被 assistant 掩盖

早期实现将 user 和 assistant 合并后再做质量判断：

$$
\text{bad user}+\text{normal assistant}
\longrightarrow
\text{apparently normal episode}.
$$

Gemma4 曾出现 `ትን-ትን-ትን`、`implementation-implementation`、`de-spa-el` 等 user 碎片，而 assistant 将其解释为键盘故障、音译或歌词。assistant 100% 自然结束并不能证明 user 正常。

### 3.3 2048 packing 掩盖坏 episode

固定长度 block 是 cache 和旧 loader 的存储兼容格式，不是语义上的上下文边界。短 episode 被大量打包后，坏 episode 的局部指标会被其他内容冲淡：

$$
\text{episode-level failure}
\not\Rightarrow
\text{block-level failure}.
$$

所以必须同时检查 block、episode、user 和 assistant 四个粒度。

### 3.4 v2 的语义模式集中

assistant bootstrap 能改善 Gemma4 的机械碎片问题，但 Qwen3 在空 assistant channel 后容易生成：

> It seems like your message might be incomplete. Could you please clarify…

这类文本语法正确、重复率正常、能够自然结束，因此会通过机械 gate；但如果绝大多数 episode 都是澄清套话，测量分布就会发生 mode concentration。它不是机械生成失败，也不应被静默删除，必须作为 semantic-mode concentration 报告。

### 3.5 过度语义过滤会引入 selection bias

澄清、拒答、列表、固定格式、多语言和短回答都是真实模型行为。删除它们会改变：

$$
\hat P(e_i)=P(e_i\mid x\text{ passes a semantic filter}),
$$

而不是想要测量的模型原生协议分布。因此语义模式只做诊断，不能作为默认 hard rejection 条件。

---

## 4. v1/v2：最低限度工程适配

v1/v2 不是两种预设的语义数据集，也不是“哪个文本更好”的比较。它们是根据 checkpoint 的训练格式、指令遵循特征和 native control structure，为了让模型稳定地产生可观测 token 流而选择的两个工程入口。

统一的是：

* MoE 测量目标；
* token 统计口径；
* mechanical health 检查；
* provenance 和稳定性报告。

不强行统一的是每个模型的 generation entry point。

### 4.1 v1：`user_role_continuation`

从 checkpoint 的 native user-role prefix 继续生成：

$$
P_\theta(x\mid\text{native user-role prefix}).
$$

不注入人工主题、关键词或外部 prompt。对于 Qwen3/Qwen3.6，这条路径可以生成用户问题、代码、文档、数学和说明性文本，当前实验中相对稳定。

### 4.2 v2：`assistant_bootstrap`

如果 user-role continuation 对某个 checkpoint 容易进入碎片循环，则先使用 native assistant generation channel 产生候选内容，再包装进 user turn：

$$
P_\theta(x\mid\text{native assistant-generation scaffold}).
$$

v2 的作用是降低生成系统自身失效的概率，而不是把 assistant 内容伪装成“更好的 user 文本”。它会改变 measured input distribution，必须在 provenance 中记录。Gemma4 当前适合 v2；Qwen3 的 v2 容易集中到澄清套话。

### 4.3 当前推荐

| 模型 | generation mode | 原因 |
|---|---|---|
| Qwen3 | v1 `user_role_continuation` | user-role continuation 稳定，内容和 routing 覆盖较丰富 |
| Qwen3.6 | v1 `user_role_continuation` | 当前实测比 assistant bootstrap 更稳定，避免澄清模式集中 |
| Gemma4 | v2 `assistant_bootstrap` | assistant channel 比 user-role continuation 更稳定，显著减少碎片循环 |

这不是 benchmark tuning，而是 checkpoint-native stability adaptation。不同模型采用不同入口时，应分别解释统计结果，不能无说明地声称输入分布完全相同。

---

## 5. 最新生成协议

### 5.1 Native scaffold

`build_native_calibration.py` 使用 tokenizer 的 `apply_chat_template()`，通过 sentinel 分割出：

* user prefix；
* user-to-assistant bridge；
* assistant suffix。

不手写 Qwen 或 Gemma4 的 token ID。当前实际解析结果为：

| 模型 | user stop | assistant stop |
|---|---|---|
| Qwen3 | native `<|im_end|>` | native `<|im_end|>` |
| Qwen3.6 | native `<|im_end|>` | native `<|im_end|>` |
| Gemma4 | native `<turn|>` | native `<turn|>` |

Gemma4 的 channel 前缀和 Qwen3.6 的 thinking-control 前缀由 native template 保留。

### 5.2 Sampling

默认采样尽量接近 checkpoint 自身分布：

$$
T=1,
\qquad top_p=1,
\qquad top_k=0,
\qquad repetition\_penalty=1.
$$

同时使用：

* `ignore_eos=False`；
* `min_tokens=0`；
* 每个 episode 独立 seed；
* user 和 assistant 各自的最大 token 上限；
* 不使用人工 lexical prompt；
* 不使用 English stopword allowlist；
* 不使用 frequency penalty、presence penalty 或 no-repeat n-gram。

达到上限但没有 EOS 的 episode 会被记录为 capped，而不是被错误标记为 natural termination。

### 5.3 User/assistant 生成顺序

user 先生成。user 通过最低限度 mechanical gate 后，才请求 assistant：

$$
\text{generate user}
\rightarrow
\text{mechanical check}
\rightarrow
\text{generate assistant}
\rightarrow
\text{mechanical check}.
$$

user 不通过时不会调用 assistant，因此 assistant 的正常回答不能稀释坏 user 的统计。

### 5.4 Health pilot 和 warm-up

正式生成前运行 pilot。若 pilot 的 mechanical rejection rate 超过 `--max-pilot-rejection-rate`，才启用短前缀 temperature warm-up，默认候选为：

$$
T_{\rm prefix}=1.5,
\qquad m=8.
$$

这只是 generation-health fallback，不是语义质量优化。warm-up 后仍不健康时直接失败，不无限重采样。

---

## 6. Hard gate 与 semantic diagnosis

### 6.1 Hard gate：只拒绝机械故障

当前 user/assistant 使用相同的最低限度 gate：

* token 输出非空；
* 若有 tokenizer，解码文本非空；
* `distinct_token_ratio >= 0.02`；
* `dominant_token_ratio <= 0.85`；
* `max_run_ratio <= 0.60`；
* `repeated_4gram_ratio <= 0.90`；
* `periodic_loop_ratio <= 0.60`。

这些阈值只针对明显机械退化，不能解释为语义质量阈值。极端案例仍应保留 rejected count 和 rejection rate，最好同时保留对应 diagnostics。

### 6.2 Diagnostic-only semantic modes

以下模式默认保留，不参与 hard rejection：

* clarification；
* refusal；
* list/tutorial；
* fixed-format/Markdown/template；
* code/software；
* math/science；
* dialogue/QA；
* story/narrative；
* business/policy/society；
* history/culture；
* health/lifestyle；
* multilingual text。

当前 inspector 使用正则和关键词做启发式诊断，不是语义模型。类别允许重叠，不能相加为总数。应重点看：

* mode rate；
* 平均/中位 episode 长度；
* distinct mode/template 数；
* top clarification/refusal/fixed-format forms；
* user 与 assistant 的分别统计；
* mode concentration，而不是“模式是否足够丰富”。

---

## 7. Token budget、packing 与统计语义

### 7.1 固定 budget 的真实含义

默认目标是：

$$
128\times2048=262144
$$

个 **serialized cache tokens**，不是 128 个必须连续生成 2048 token 的回答，也不是 262144 个纯 lexical assistant tokens。

预算包含：

* native user prefix；
* user content；
* turn bridge；
* assistant content；
* native assistant suffix。

因此 payload 中的 `calibration_tokens` 应理解为 cache token budget。若实验需要报告 lexical token budget，必须另行统计，不能与该字段混用。

### 7.2 Packing

独立 episode 会按顺序打包成 `[1, blocks × block_length]`，并记录：

* `token_stream.episode_boundaries`；
* `token_stream.block_boundaries`；
* episode 是否跨 block；
* source offset；
* user/assistant token 数量和终止状态。

packing 只是为了兼容现有 Wanda/ENP loader。它不能改变以下统计要求：

$$
\text{attention state}(e_j)
\not\leftarrow
\text{attention state}(e_{j-1}).
$$

当前 collector 尚未读取 boundary，因此现有 profile 结果不能自动宣称已经完成 episode-isolated forward。后续 collector 必须重建 episode，逐 episode forward 后再聚合 activation 和 routing statistics；不能把 block boundary 当成 episode boundary。

---

## 8. Calibration pool 与下游统计

设计上可以将 serialized budget 分为：

$$
96\times2048\rightarrow\mathcal D_N,
\qquad
32\times2048\rightarrow\mathcal D_R.
$$

其中：

* `D_N` 用于估计 natural expert prevalence 和条件 channel statistics；
* `D_R` 只增加低样本 expert 的条件观测，不重新估计 natural prevalence；
* guided samples 如未来启用，只能作为有偏的 conditional fallback，不能改变 prevalence。

当前 builder 只把这些范围写入 `calibration_pools` metadata，还没有实现对应 collector 的统计隔离。因此现阶段不能把 metadata 当成已完成的 discovery/reserve 实验。

后续应计算：

### Expert hit frequency

$$
n_{l,i}=\sum_t\mathbf 1[e_i\in TopK_l(x_t)],
\qquad
m_{l,i}=\sum_t p_{l,i}(x_t).
$$

`n` 是 hard hit count，`m` 是 soft routing mass，两者必须分别报告。

### Conditional channel activation

$$
A_{l,i,c}
=
\frac{
\sum_t\mathbf 1[e_i\in TopK_l(x_t)]a_{l,i,c,t}^{2}
}{
\sum_t\mathbf 1[e_i\in TopK_l(x_t)]
}.
$$

对于 Gemma4，需要按 native forward hook 区分 router input 和 routed-expert input；不能假设二者相同。

### Channel functional contribution

仅有 activation magnitude 不足以判断冗余。还应结合 down projection：

$$
F_{i,c}
\approx
E[(W_{i,\mathrm{down}}[:,c]a_{i,c})^{2}\mid e_i\text{ is hit}],
$$

以及 channel covariance：

$$
G_i=E[a_i a_i^{\mathsf T}].
$$

低 activation 不必然等于冗余，高 activation 也不必然等于不可剪；需要结合 activation、功能方向和 channel correlation。

---

## 9. 运行方式

### 9.1 单元测试

```bash
pytest "calibration_method/self moe calibration/test_native_calibration.py" -q
```

测试覆盖：

* token/短周期机械退化检测；
* semantic mode 只诊断不硬过滤；
* v1/v2 scaffold 处理；
* warm-up mock；
* 固定 block packing；
* episode boundary 重构。

### 9.2 单模型生成

Qwen3/Qwen3.6 推荐显式使用 v1：

```bash
python build_native_calibration.py \
  --model-path /data01/datasets/Qwen3.6-35B-A3B \
  --output /path/to/qwen36_cn_moe_sc.pt \
  --blocks 128 \
  --block-length 2048 \
  --user-generation-mode user_role_continuation \
  --seed 42
```

Gemma4 推荐 v2：

```bash
python build_native_calibration.py \
  --model-path /data01/datasets/gemma-4-26B-A4B-it \
  --output /path/to/gemma4_cn_moe_sc.pt \
  --blocks 128 \
  --block-length 2048 \
  --user-generation-mode assistant_bootstrap \
  --seed 42
```

当前默认 CLI mode 是 `assistant_bootstrap`，因此 Qwen 单模型运行时不要省略 `--user-generation-mode user_role_continuation`。

### 9.3 三模型 pilot

```bash
GPUS="0 1 2" bash run_three_model_pilots.sh
```

脚本会在启动前检查三张 GPU 是否存在 compute PID；不应抢占他人任务。它当前使用：

* Qwen3：v1；
* Qwen3.6：v1；
* Gemma4：v2；
* 每个模型 2 个 256-token block；
* 生成日志、cache 和 inspection JSON 分开保存。

### 9.4 Inspection

```bash
python inspect_native_calibration.py \
  --cache /path/to/qwen36_cn_moe_sc.pt \
  --model-path /data01/datasets/Qwen3.6-35B-A3B \
  --sample-count 8
```

检查时至少查看：

* `generation_health`；
* `quality.block`、`quality.episode`、`quality.user`、`quality.assistant`；
* `semantic_modes` 的 block/episode/user/assistant 结果；
* `user_samples` 和 `assistant_samples`；
* episode 数量、平均长度和 capped/terminated 比例。

---

## 10. 结果解释原则

### 应该回答的问题

1. expert hit ranking 在不同 seed 或 split-half 下是否稳定？
2. router mass ranking 是否稳定？
3. 条件 channel activation ranking 是否稳定？
4. v1/v2 的入口适配是否明显改变 expert prevalence？
5. activation ranking 与 down-projection functional contribution 是否一致？
6. pruning 后的性能变化能否由这些统计解释？

### 不应该回答的问题

* 校准文本是否主题均衡；
* 是否每个 block 都包含代码、数学和故事；
* clarification 是否“足够少”；
* 文本是否像人工编写的 benchmark；
* `unique_sequences=128` 是否足以证明多样性。

最终应遵循：

$$
\boxed{
\text{model-native generation}
\rightarrow
\text{minimum mechanical validation}
\rightarrow
\text{preserve semantic modes}
\rightarrow
\text{episode-isolated MoE statistics}
}
$$

v1/v2 的选择是为了让不同 checkpoint 能稳定地产生可测量状态；校准集最终是否有价值，应由 expert 命中频率、conditional activation 和统计稳定性判断，而不是由文本表面质量判断。
