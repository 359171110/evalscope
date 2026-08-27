可以。解决你这次复现暴露出的生成退化问题以后，我建议把原来的 Self-Calibration 改造成一套更适合现代 MoE instruction checkpoints 的：

$$
\boxed{\textbf{Checkpoint-Native MoE Self-Calibration (CN-MoE-SC)}}
$$

核心思想不再是机械地：

$$
\text{BOS}\rightarrow2048\text{ tokens},
$$

而是：

$$
\boxed{
\text{从 checkpoint 自己的原生交互分布生成有效状态，
自然终止，以固定 token budget 收集 MoE expert-conditioned statistics，
并把 coverage 不足转化成“不确定性”而不是强行补齐。}
}
$$

整个 Self-Calibration 阶段只负责为后面的：

$$
\text{Selection}
\rightarrow
\text{Width Allocation}
\rightarrow
\text{Compensation}
$$

提供可靠统计。

---

# 1. 总体定义

我们不使用任何外部 calibration corpus：

$$
\boxed{\mathcal D_{\rm external}=\varnothing}.
$$

所有 lexical content 都由待剪枝模型 \(M_\theta\) 自己生成。

但我们允许使用 checkpoint 自带的：

* tokenizer；
* BOS/EOS；
* chat template；
* role token；
* turn delimiter；
* channel/control token。

这些属于模型自身 metadata，而不是外部语义 calibration data。

因此方法仍然属于：

$$
\boxed{\text{external-data-free}}
$$

但不是 calibration-free。

---

# 2. 第一项修改：Checkpoint-Native Generation Mode

不再规定所有模型统一 BOS-only。

首先根据 checkpoint 类型选择原生生成模式。

---

## 2.1 Base / continuation checkpoint

如果模型本身是 base LM，则保持原 Self-Calibration：

$$
\boxed{
BOS
\rightarrow
x_1,\ldots,x_T
}
$$

其中：

$$
x_t\sim p_\theta(x_t|x_{<t}).
$$

允许模型正常 EOS。

---

## 2.2 Instruction / Chat checkpoint

对于目前的：

* Qwen3-Instruct；
* Qwen3.6；
* Gemma4-it；

统一采用：

$$
\boxed{\textbf{Native-Template Self-Dialogue}}
$$

而不是裸 BOS。

一段 calibration episode：

$$
e=
[\mathcal S_U,\;u,\;\mathcal E_U,\;
 \mathcal S_A,\;a,\;\mathcal E_A].
$$

其中：

* \(\mathcal S_U\)：checkpoint 原生 user-role prefix；
* \(\mathcal E_U\)：原生 user-turn terminator；
* \(\mathcal S_A\)：原生 assistant-role prefix；
* \(\mathcal E_A\)：原生 assistant terminator。

这些全部从 tokenizer/chat template 获取，**不手写 token ID**。

---

# 3. Self-Generated User Turn

这里不能人工写：

> Solve a math problem.

也不能写：

> Tell me something.

否则又引入：

$$
P(x|\text{human prompt}).
$$

我们直接在 native user scaffold 后让模型自己生成 user content：

$$
u_t
\sim
p_\theta
\left(
u_t
\mid
\mathcal S_U,u_{<t}
\right).
$$

直到：

$$
\mathcal E_U
$$

或 episode 上限。

于是：

$$
\boxed{
u\text{ 也是模型自己产生的}
}
$$

没有任何外部 task semantics。

这利用的是模型 instruction tuning 时已经学到的：

$$
P(\text{user content}\mid\text{user role prefix}).
$$

---

# 4. Self-Generated Assistant Turn

得到自生成 user turn 后，加入 checkpoint 原生：

$$
\mathcal S_A.
$$

再采样：

$$
a_t
\sim
p_\theta
\left(
a_t
\mid
\mathcal S_U,u,\mathcal E_U,\mathcal S_A,a_{<t}
\right).
$$

直到：

$$
\mathcal E_A
$$

或 episode 上限。

因此整个 episode：

$$
\boxed{
\text{structural scaffold来自checkpoint，
所有语义内容来自模型自己}
}
$$

这比 BOS-only 更适合 instruction checkpoints，又没有引入人工 prompt。

---

# 5. 三个模型的实现原则

## Qwen3 / Qwen3.6

由 tokenizer 自己产生：

$$
\texttt{<|im\_start|>user}
$$

等 native role/turn boundaries。

不要再手工：

$$
\texttt{<|endoftext|>}
$$

作为唯一启动或终止依据。

尤其 Qwen3.6 中不同 EOS/turn terminator 口径必须由 native template 处理。

---

## Gemma4

同样直接使用 tokenizer/checkpoint 定义的：

* BOS；
* user turn；
* assistant turn；
* channel；
* role；
* end-of-turn

等控制结构。

不允许：

$$
\boxed{\text{裸 <bos> 后直接 lexical sampling}}
$$

作为默认 calibration protocol。

这样可以避免你现在观测到的：

$$
45/128
$$

首 token 落入同一个韩语片段这种异常初始 mode concentration。

---

# 6. 第二项修改：Natural Termination

这个必须严格修改。

所有生成：

$$
\boxed{
\texttt{ignore\_eos=False}
}
$$

并且：

$$
\boxed{
\texttt{min\_tokens=0}.
}
$$

绝不能再：

$$
\texttt{min\_tokens}=2047
$$

强迫模型生成。

如果模型在：

$$
T=173
$$

时认为一个 turn/episode 已自然结束，那么：

$$
\boxed{\text{就在 173 结束}}
$$

而不是继续采到 2048。

---

# 7. 不再要求“一条 sequence 必须连续 2048 tokens”

这是一个非常重要的修改。

我们的目标仍然保持标准 calibration budget：

$$
\boxed{
128\times2048
=
262144
}
$$

个 model tokens。

但是这里的 \(128\times2048\) 定义为：

$$
\boxed{\textbf{token-equivalent calibration budget}}
$$

而不是：

> 128 次必须连续生成满 2048 token。

---

## Calibration Block

定义 calibration block：

$$
B_j
$$

目标 token 数：

$$
|B_j|=2048.
$$

它可以包含多个独立 episode：

$$
B_j=
\{e_1,e_2,\ldots,e_m\}
$$

满足：

$$
\sum_{k=1}^{m}|e_k|
\approx2048.
$$

但是每个：

$$
e_k
$$

**独立 forward**，attention state 不跨 episode 延续。

这样如果模型自然只生成：

$$
300
$$

tokens：

> 结束 → 重启一个新的 native episode。

而不是：

> 强迫原 episode 再吐 1700 个 token。

---

# 8. 为什么 independent episodes 更适合我们的 MoE pruning

后面我们需要的是：

$$
E[a_ca_q],
$$

$$
P(h_l^E|E_i),
$$

$$
D_i(d),
$$

这些 activation statistics。

它们不要求所有 262k token 位于连续的 2048-context 中。

所以：

$$
\boxed{
\text{多个真实有效 episode}
>
\text{一段被强制拖到2048的退化文本}
}
$$

对于 calibration statistics 更合理。

同时仍然维持完全相同的：

$$
262144
$$

token budget。

---

# 9. 第三项修改：Sampling Protocol

默认严格使用最少干预的 sampling：

$$
\boxed{
T=1
}
$$

$$
\boxed{
top_p=1
}
$$

$$
\boxed{
top_k=\varnothing
}
$$

$$
\boxed{
repetition\_penalty=1
}
$$

不默认使用：

* repetition penalty；
* frequency penalty；
* presence penalty；
* no-repeat ngram；
* beam search。

因为这些都会主动改变：

$$
p_\theta(x).
$$

所以默认目标仍然尽量接近：

$$
\boxed{x\sim p_\theta}.
$$

---

# 10. Prefix-collapse fallback

但是你已经证明 Qwen3.6/Gemma4 可能出现：

$$
\text{从第一批 lexical tokens 就进入极低熵 mode}.
$$

因此在正式生成前先运行：

$$
\boxed{16\text{-episode pilot}}
$$

检查 prefix collapse。

如果 native-template 以后仍然出现明显：

* first-token concentration；
* prefix diversity collapse；
* catastrophic loop；

则允许使用原 Self-Cal 已经讨论过的 **temperature warm-up**：

$$
T_t
=
T_0-
\frac{t}{m}(T_0-1),
\qquad t\le m,
$$

之后：

$$
T_t=1.
$$

例如默认候选：

$$
T_0=1.5,\qquad m=8.
$$

即：

$$
\boxed{
\text{前8个 lexical tokens 稍微提高探索，
随后恢复 }T=1.
}
$$

注意这只是：

$$
\boxed{\text{generation-health fallback}}
$$

而不是所有模型强制使用。

Qwen3 如果：

$$
T=1
$$

已经健康，就完全不启用。

---

# 11. 不使用 English-stopword 强约束作为默认方案

原 Self-Calibration 对一些模型做过 first-token English constraint。

我们这里不作为默认方法。

因为最终 benchmark 包含：

* code；
* math；
* reasoning；
* natural language。

如果把第一 token 限定为英语 stopword，会人为削弱其他模式。

所以优先级是：

$$
\boxed{
\text{Native Template}
>
\text{Natural EOS}
>
\text{temperature warm-up}
}
$$

而不是 lexical allowlist。

---

# 12. 第四项修改：Generation Validity Gate

我们不能再接受：

$$
\text{token序列不同}
\Rightarrow
\text{有效 calibration}.
$$

每个 episode 生成后计算 mechanical-degeneration diagnostics。

---

## 12.1 Distinct-token ratio

$$
r_{\rm distinct}
=
\frac{
|\operatorname{UniqueTokens}(e)|
}{
|e|
}.
$$

---

## 12.2 Dominant-token ratio

$$
r_{\rm dom}
=
\max_t
\frac{
\operatorname{count}(t)
}{
|e|
}.
$$

---

## 12.3 Repeated n-gram ratio

例如：

$$
r_{4g}
$$

统计重复 4-gram / 8-gram 所覆盖的 token 比例。

---

## 12.4 Periodic-loop detection

检测类似：

```text
dddddddd...
```

```text
不可使用 不可使用 不可使用...
```

以及固定长度：

$$
k
$$

的周期循环。

---

# 13. 只过滤 catastrophic degeneration

Validity Gate **不能做 semantic quality filtering**。

不能因为：

* 两段都在讲代码；
* 两段语言类似；
* 两段主题相似；

就删除。

否则：

$$
P_{\rm calibration}
$$

又被人为改掉。

只拒绝明显：

$$
\boxed{\text{mechanical generation failure}}
$$

例如默认可以把：

$$
r_{\rm dom}>0.5
$$

或者：

$$
r_{\rm distinct}<0.02
$$

或者极端 n-gram loop

作为 invalid。

这些阈值需要做小规模 sensitivity check，而不是作为理论常数。

---

# 14. Invalid episode 怎么处理

如果某个 episode 触发 degeneration gate：

$$
e_j\rightarrow\text{invalid}.
$$

立即结束当前 episode 并重新开启一个 independent native episode。

invalid token：

$$
\boxed{\text{不计入262144 token calibration budget}}
$$

但必须记录：

$$
\boxed{\text{rejection rate}}
$$

作为 generation health metric。

---

# 15. 不能无限重采样

如果某 checkpoint 在 pilot/正式生成中：

$$
R_{\rm invalid}>\tau_{\rm health}
$$

例如：

$$
10\%-20\%
$$

量级，

就不能靠不断 rejection 来“洗”出漂亮数据。

应该判定：

$$
\boxed{
\text{当前 generation protocol 对该 checkpoint 不健康}
}
$$

然后：

1. BOS-only → Native Template；
2. Native Template → Prefix Temperature Warm-up；
3. 仍失败 → 该 checkpoint 的 Self-Cal confidence 降级。

而不是无限 retry。

---

# 16. 第五项修改：Calibration Pool 不再用“unique sequence count”判断质量

生成结束后，我们记录：

$$
\boxed{
\mathcal D_{\rm SC}
}
$$

的整体 health profile：

* valid token count；
* episode count；
* natural termination ratio；
* mean/median episode length；
* distinct-token ratio；
* dominant-token ratio；
* n-gram diversity；
* invalid/retry rate；
* first-token entropy。

因此不再用：

$$
\texttt{unique\_sequences}=128
$$

这种没有意义的指标代表 diversity。

---

# 17. 第六项修改：MoE-Conditioned Statistics

完成有效生成以后，才进入 MoE-specific calibration。

对于每个 token，在每个 MoE layer 捕获：

$$
x_l^R
$$

和：

$$
x_l^E.
$$

其中：

* \(x_l^R\)：真实 router input；
* \(x_l^E\)：真实 routed-expert input。

---

## Qwen3/Qwen3.6

通常：

$$
x_l^R=x_l^E.
$$

---

## Gemma4

必须：

$$
\boxed{
x_l^R\neq x_l^E
}
$$

分别按模型 native forward hook。

所有 expert channel statistics 只使用：

$$
x_l^E.
$$

---

# 18. 128×2048 内部分成 Natural Discovery + Coverage Reserve

为了避免你之前指出的 circular bias，我建议总 budget：

$$
128\times2048
$$

内部这样分：

$$
\boxed{
96\times2048
\quad
\mathcal D_N
}
$$

和：

$$
\boxed{
32\times2048
\quad
\mathcal D_R
}
$$

但两者**全部采用相同的 checkpoint-native natural generation**。

这里不再默认做 router-guided generation。

---

# 19. Natural Discovery Set

前：

$$
96\times2048
$$

只负责估计模型自己的自然 routing statistics。

例如：

$$
p_{l,i}^{N}
=
P_{\mathcal D_N}
(i\in TopK)
$$

以及 expert-conditioned hit pool：

$$
\mathcal H_{l,i}^{N}.
$$

一旦计算：

$$
p_{l,i}^{N},
$$

之后永久冻结。

---

# 20. Coverage Reserve

剩下：

$$
32\times2048
$$

仍然只是：

$$
x\sim p_\theta
$$

自然生成。

它的作用不是重新估计：

$$
P(E_i).
$$

而只是给低样本 expert 增加：

$$
P(h_l^E|E_i)
$$

的条件样本。

因此：

$$
\boxed{
\mathcal D_R
\text{ 不参与 expert natural prevalence estimation}
}
$$

避免：

> 为了补 coverage → 又把某 expert 判得更重要。

---

# 21. Gemma4 不要求强制覆盖全部 expert

这是这版最重要的修改之一。

以前：

$$
n_{l,i}<32
\rightarrow
\text{guided补到32}.
$$

现在取消。

对于：

$$
n_{l,i}
=
|\mathcal H^N_{l,i}\cup\mathcal H^R_{l,i}|,
$$

coverage 只用于估计：

$$
\boxed{\text{confidence}}
$$

而不是硬性要求。

---

# 22. Split-Half Stability

而且不能再简单：

$$
n_i\ge32
\Rightarrow\text{safe}.
$$

把 Natural pool 分成：

$$
\mathcal D_N^A,\qquad
\mathcal D_N^B.
$$

对 expert \(i\) 分别估计 channel statistics，比如：

$$
m_{i,c}^{A}
=
E_A[a_c^2],
$$

$$
m_{i,c}^{B}
=
E_B[a_c^2].
$$

计算：

$$
\rho_i
=
Spearman(m_i^A,m_i^B)
$$

以及 provisional retained overlap：

$$
O_i^{50}.
$$

所以 expert 的可靠性由：

$$
\boxed{
\text{coverage}
+
\text{statistical stability}
}
$$

共同决定。

---

# 23. Confidence 分级

例如：

### High-confidence

样本充分，而且：

$$
\rho_i
$$

和 retained-set overlap 足够稳定。

允许后续：

* full covariance；
* set-aware selection；
* heterogeneous width；
* joint compensation。

---

### Medium-confidence

有一定样本，但统计不够稳定。

后续：

* covariance shrinkage；
* width 保持接近 uniform；
* bounded compensation。

---

### Low-confidence

几乎没有自然 samples，或者 A/B 极不稳定。

后续：

$$
\boxed{
d_i=d_0
}
$$

保持 uniform baseline；

不参与 heterogeneity；

不做 aggressive compensation。

---

# 24. 这就是解决 Gemma4 coverage 问题的核心

Gemma4 如果仍然有：

$$
n_i\approx0
$$

我们**不再强迫模型人为生成这些 expert 的数据**。

而是：

$$
\boxed{
\text{无法可靠 self-calibrate}
\Rightarrow
\text{不允许对该 expert 做高风险决策}.
}
$$

所以方法不会因为：

$$
\text{Gemma4 natural routing skew}
$$

而失效。

它只会自动变得更保守。

---

# 25. Router-Guided generation 降级为可选 fallback

我们之前：

$$
W_R
\rightarrow
\text{vocabulary token}
\rightarrow
\text{guided generation}
$$

的想法不完全删除。

但是从核心 Self-Cal protocol 中拿掉。

只有当：

$$
n_i=0
$$

导致连最低 activation statistic 都无法获得时，才允许作为 optional fallback。

而且 router-guided samples：

$$
\boxed{
\text{只能用于 conditional statistics}
}
$$

不能用于：

$$
P(E_i)
$$

不能直接让 expert 获得 high-confidence 状态。

并且默认只能支持：

* diagonal activation estimate；
* conservative baseline selection。

不能仅因为 guided samples 多，就允许 aggressive heterogeneity。

---

# 26. 为什么这版方法理论上更干净

我们的最终数据角色非常明确：

$$
\boxed{
\mathcal D_N
\rightarrow
P_\theta(E_i)
+
P_\theta(h|E_i)
}
$$

$$
\boxed{
\mathcal D_R
\rightarrow
\text{additional natural samples of }P_\theta(h|E_i)
}
$$

而 optional guided：

$$
\boxed{
\mathcal D_G
\rightarrow
\text{biased conditional fallback only}
}
$$

它们不会混在一起重新统计 prevalence。

---

# 27. 最终生成算法

可以概括为：

$$
\boxed{
\begin{array}{l}
\textbf{Input: } M_\theta,\ tokenizer,\ B=128\times2048 \\[2mm]

1.\ \text{Detect checkpoint type} \\
\quad\text{base}\rightarrow\text{BOS continuation}\\
\quad\text{instruct}\rightarrow\text{native-template self-dialog}\\[2mm]

2.\ \text{Run 16-episode health pilot} \\[1mm]

3.\ \text{Use }T=1,\ top_p=1,\ ignore\_eos=False \\[1mm]

4.\ \text{Generate independent naturally terminated episodes} \\[1mm]

5.\ \text{Reject only catastrophic mechanical degeneration} \\[1mm]

6.\ \text{If prefix collapse persists, enable short temperature warm-up} \\[1mm]

7.\ \text{Accumulate }262144\text{ valid model tokens} \\[1mm]

8.\ \text{First }96\times2048\rightarrow\mathcal D_N \\[1mm]

9.\ \text{Remaining }32\times2048\rightarrow\mathcal D_R \\[1mm]

10.\ \text{Collect native router/expert states} \\[1mm]

11.\ \text{Estimate per-expert coverage + split-half stability} \\[1mm]

12.\ \text{Assign High/Medium/Low confidence} \\[1mm]

13.\ \text{Return calibration statistics + confidence masks.}
\end{array}
}
$$

---

# 28. Qwen3 / Qwen3.6 / Gemma4 的最终统一策略

| 项目                    | Qwen3                | Qwen3.6              | Gemma4               |
| --------------------- | -------------------- | -------------------- | -------------------- |
| checkpoint 类型         | instruct             | instruct             | instruct             |
| 默认启动                  | native chat scaffold | native chat scaffold | native chat scaffold |
| 手写 BOS-only           | 否                    | 否                    | 否                    |
| lexical prompt        | **无**                | **无**                | **无**                |
| user 内容               | self-generated       | self-generated       | self-generated       |
| assistant 内容          | self-generated       | self-generated       | self-generated       |
| EOS                   | native               | native               | native               |
| `ignore_eos`          | False                | False                | False                |
| `min_tokens`          | 0                    | 0                    | 0                    |
| default T             | 1                    | 1                    | 1                    |
| default top-p         | 1                    | 1                    | 1                    |
| repetition penalty    | 1                    | 1                    | 1                    |
| 2048 强制续写             | **否**                | **否**                | **否**                |
| episode reset         | ✓                    | ✓                    | ✓                    |
| degeneration gate     | ✓                    | ✓                    | ✓                    |
| expert必须全覆盖           | **否**                | **否**                | **否**                |
| low-confidence expert | 保守                   | 保守                   | 保守                   |

---

# 29. 这版 Self-Calibration 输出什么

CN-MoE-SC 最终不是简单输出一个：

$$
\texttt{calibration.pt}.
$$

它应该输出四类统计。

### Generation statistics

$$
\mathcal S_{\rm gen}
$$

包括：

* valid token count；
* episode length；
* EOS rate；
* degeneration rate；
* diversity statistics。

### Natural routing statistics

$$
\mathcal S_{\rm route}
$$

包括：

$$
p_{l,i}^{N}
$$

和实际 router coefficients。

### Expert-conditioned state statistics

$$
\mathcal H_{l,i}.
$$

### Confidence metadata

$$
\boxed{
c_{l,i}\in
\{\text{High},\text{Medium},\text{Low}\}
}
$$

后续所有 pruning 模块必须尊重这个 confidence mask。

---

# 30. 然后和我们最新整套 pruning 方法对接

最终：

$$
\boxed{
\text{CN-MoE-SC}
}
$$

提供：

$$
A_i,\quad
G_i=E[a_ia_i^\top],\quad
\text{confidence}_i.
$$

再进入：

$$
\boxed{
\text{Set-aware Functional Selection}
}
$$

其中：

$$
Q_i
=
E[a_ia_i^\top]
\odot
W_{d,i}^{\top}W_{d,i}.
$$

然后：

$$
\boxed{
\text{Conservative Width Redistribution}
}
$$

只有 high-confidence expert 才允许明显偏离 uniform baseline。

最后：

$$
\boxed{
\text{Bounded Joint Output Compensation}.
}
$$

---

## 这版方法最核心的变化可以浓缩成一句话

原 Self-Cal 是：

$$
\boxed{
\text{BOS}\rightarrow
\text{fixed-length synthetic text}
\rightarrow
\text{calibration}.
}
$$

我们的 MoE Self-Cal 应该是：

$$
\boxed{
\textbf{Checkpoint-native interaction}
\rightarrow
\textbf{natural termination}
\rightarrow
\textbf{valid independent episodes}
\rightarrow
\textbf{MoE-conditioned statistics}
\rightarrow
\textbf{confidence-aware calibration}.
}
$$

也就是说，我们不再认为：

> “模型自己生成出来的 token 都天然是好的 calibration data。”

而是更严格地要求：

$$
\boxed{
\text{生成过程必须符合 checkpoint 自身的 operating format，
生成状态必须非退化，
而 MoE pruning 只在 self-cal statistics 足够稳定时才做激进决策。}
}
$$

这版可以直接作为后续 Qwen3 / Qwen3.6 / Gemma4 统一重新生成 calibration 的协议。

---

# 31. Reference implementation and validation

本目录现在包含可执行的参考实现：

* `build_native_calibration.py`：使用 checkpoint 自带 chat template 构造 user/assistant native scaffold，分别独立生成 user turn 和 assistant turn，允许自然终止，仅拒绝机械重复，并将有效 episode 打包为固定长度 calibration blocks。
* `inspect_native_calibration.py`：解码校准块，统计语言、主题、重复率和生成健康度，并输出代表性样本。
* `test_native_calibration.py`：验证重复检测、固定 block 打包和 episode boundary 记录。

## 31.1 Gemma4/Qwen3.6 的 user-turn 修复

旧版虽然使用了 native chat template，但仍然错误地假设：

$$
P(\text{user content}\mid\text{user role prefix})
$$

是 checkpoint 的可靠生成分布。对 instruction checkpoint，这个条件通常没有得到充分训练；因此 user turn 可能从一开始就进入跨语言片段循环。旧版还在 `user + assistant + packed block` 上做质量检查，正常结束的 assistant 会把坏 user 的统计冲淡。

当前 `cn_moe_sc_native_dialogue_v2` 的默认行为是：

1. 使用 checkpoint native generation scaffold 做 bootstrap；
2. 默认从训练得更充分的 assistant generation channel 生成候选语义内容，再把它作为下一轮 user content；
3. 在 assistant 请求之前单独检查 user turn；
4. 对 user 使用更严格的 distinct、dominant-token、max-run、重复 4-gram、重复词组和 script-switch gate；
5. 只有 user 通过后才生成 assistant；
6. inspector 同时输出 user、assistant、episode 和 block 四种粒度，不能再用 block 指标掩盖坏 episode。

注意：旧版 `cn_moe_sc_native_dialogue_v1` cache 没有 user-turn 级 provenance 和质量保证，不能通过重新运行 inspector 将其升级为 v2 健康 cache；必须重新生成。

默认协议参数为：

$$
T=1,
\quad top\_p=1,
\quad top\_k=0,
\quad repetition\_penalty=1,
\quad ignore\_eos=False,
\quad min\_tokens=0.
$$

运行时不再使用人工 lexical prompt，也不再使用裸 BOS-only continuation。native scaffold 由 tokenizer 的 `apply_chat_template()` 动态解析，校准缓存会记录：

* native prefix、turn bridge 和 assistant suffix 的 token 长度；
* 每个 episode 的 block boundary；
* 自然终止率、拒绝率和拒绝原因对应的机械退化指标；
* 固定 block 的 SHA256 和生成参数。

示例：

```bash
python build_native_calibration.py \
	--model-path /data01/datasets/Qwen3.6-35B-A3B \
	--output /path/to/qwen36_cn_moe_sc.pt \
	--blocks 128 --block-length 2048 --seed 42
```

生成后检查：

```bash
python inspect_native_calibration.py \
	--cache /path/to/qwen36_cn_moe_sc.pt \
	--model-path /data01/datasets/Qwen3.6-35B-A3B
```

该实现将独立 episode 打包为固定长度 block，并在 `token_stream.episode_boundaries` 中保留边界；后续 MoE statistics collector 应按这些边界分别 forward，而不是把跨 episode 的 attention state 当作连续上下文。

三个目标模型的小规模真实生成检查可以运行：

```bash
GPUS="0 1 2" bash run_three_model_pilots.sh
```

脚本启动前会要求三张卡都没有 compute PID，避免抢占其他任务。每个模型先生成 2×256-token pilot cache，并输出：

* `<model>_generation.log`：生成过程和 validity gate；
* `<model>_cn_moe_sc_pilot.pt`：共享 cache；
* `<model>_inspection.json`：解码样本、语言/主题分布和重复退化指标。

## Compatibility note

固定 block cache 保持了现有 Wanda/ENP loader 所需的 `[1, blocks×block_length]` 结构，因此可以直接加载。但当前 Wanda/ENP collector 仍把每个 2048-token block 作为连续 attention context。严格遵循 CN-MoE-SC 定义时，collector 必须读取 `token_stream.episode_boundaries`，对每个 episode 分别 forward，再聚合 activation statistics。仅仅记录 boundary 而继续跨 episode attention，不等价于设计中要求的 independent forward。
