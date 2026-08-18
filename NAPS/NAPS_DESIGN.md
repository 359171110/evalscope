# NAPS：Native-Route AIMER Protection and Selection

## 0. 方法定位与当前版本

NAPS 是一套 **data-free、training-free、固定 expert width、结构化 channel pruning** 方法。它不尝试重新发明一个全局 channel score，而是把现有实验中已经验证过的能力拆成三个层次：

$$
\boxed{
\text{Effective-zero filtering}
\rightarrow
\text{Stable-AIMER global ranking}
\rightarrow
\text{native-route PP rescue + local subset selection}
}
$$

NAPS 同时构建并评测 mask-only 与低自由度补偿两个版本：

$$
\boxed{
\mathrm{NAPS\text{-}Mask}
\quad\text{and}\quad
\mathrm{NAPS\text{-}Bounded\text{-}Merge}
}
$$

各模块的职责严格分离：

| 模块 | 职责 |
| --- | --- |
| Effective-zero mask | 强制删除数值上已经失效的 channel |
| Stable-AIMER | 提供全局稳定 ranking backbone |
| Native-route PP | 在 AIMER prune set 中发现少量疑似误剪 channel |
| Evidence gate | 根据 routed probes 的有效样本量和有效秩限制 PP 的挑战权限 |
| Subset selection | 只在 AIMER cutoff 附近决定 rescue/displace |
| Bounded merge | 只补偿 NAPS 新换出的 AIMER tail channel，不补偿全部 prune set |

NAPS 的核心主张是：

> Stable-AIMER 负责全局判断；PP 只提供局部反证；任何 channel replacement 都必须通过集合级输出保持目标，并且允许完全退回 Stable-AIMER。

---

## 1. 设计所依据的实验事实

### 1.1 Stable Concat 是当前最可靠的 AIMER backbone

Qwen3.6 的 Original Concat-AIMER 会把接近零的 sentinel channel 异常高排。显式屏蔽 effective-zero channel 后，Stable Concat 在 B9/B6 都优于 Original-Old 和 Shape-AIMER：

| 稀疏度 | Original-Old Macro | Shape Macro | Stable Concat Macro |
| --- | ---: | ---: | ---: |
| B9 / 25% | 0.8861 | 0.8871 | **0.8885** |
| B6 / 50% | 0.7343 | 0.7458 | **0.7501** |

因此 NAPS 不修改 active channel 的 Concat-AIMER score，只在排序前加入明确的 effective-zero mask。

### 1.2 Gauge balancing 对现有 checkpoint 是弱干预

Original 与 Gauge-Balanced AIMER 的 retained-set overlap 接近 99%。这说明已有结果不足以支持“任意 up/down scale imbalance 是 AIMER 的主要收益来源”，也不支持在 NAPS 中继续增加新的 gauge prior。

### 1.3 Shape 捕获大部分信号，但 concat joint information 仍有价值

Shape-AIMER 与 Original Concat-AIMER 高度相关，但 Stable Concat 在 Qwen3.6 B9/B6 仍优于 Shape。NAPS 因此保留 Stable Concat 作为 backbone，不把 gate/up/down shape average 替换为默认全局分数。

### 1.4 Router-row pseudo reconstruction 不能证明真实可补偿性

此前固定 mask 的 pseudo-space output reconstruction 达到：

- B6 pseudo recovery：99.7332%；
- B9 pseudo recovery：99.8523%。

但下游 Macro 相对 AIMER+PP 分别下降 22.05 和 2.25 个百分点。该结果说明：

$$
\boxed{
\text{pseudo-space reconstruction quality}
\not\Rightarrow
\text{real-distribution recovery}
}
$$

因此 NAPS-v1 将 mask-only 与 bounded merge 作为两个并列实验版本。补偿不能覆盖全部被剪 channel，其结果必须相对同一 NAPS-Mask 独立报告，不能用 pseudo-space loss 代替下游结论。

### 1.5 Native routing 后的 probe 数量不等于有效证据量

对 Qwen3 与 Qwen3.6 的全部 router-row pseudo tokens 做静态路由诊断，得到：

| 模型 | self 进入 native Top-k | $M_e$ 中位数 | $N_{\mathrm{eff}}$ 中位数 | probe 有效秩中位数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 | 100.0% | 4 | 1.00 | 2.12 |
| Qwen3.6 | 88.0% | 5 | 1.94 | 2.10 |

两种模型的 $M_e$ 均值都是 native Top-k 的 8，但分布高度不均衡：

- Qwen3 的 $M_e$ 范围为 1--124；
- Qwen3.6 的 $M_e$ 范围为 0--200，约 5.10% 的 expert 没有 routed router-row probe；
- Qwen3 的 router 权重尤其集中，虽然 $M_e$ 中位数为 4，但 $N_{\mathrm{eff}}$ 中位数接近 1。

因此 NAPS 不再使用单独的 $M_e/M_0$ 规则判断 PP 可信度，而是同时使用有效样本量和有效秩。

---

## 2. 记号与固定宽度约束

对第 $l$ 层 expert $e$ 的 structured channel $c$，定义：

$$
g_{e,c}=W_g^{(e)}[c,:],
\qquad
u_{e,c}=W_u^{(e)}[c,:],
\qquad
d_{e,c}=W_d^{(e)}[:,c].
$$

一个 channel 的完整参数三元组为：

$$
w_{e,c}=[g_{e,c};u_{e,c};d_{e,c}].
$$

原始 expert intermediate width 为 $C$，目标稀疏率为 $\rho$，每个 expert 最终保留：

$$
K=(1-\rho)C.
$$

当前实验预算：

| 模型 | $C$ | B9 / 25% 保留 | B6 / 50% 保留 |
| --- | ---: | ---: | ---: |
| Qwen3 | 768 | 576 | 384 |
| Qwen3.6 | 512 | 384 | 256 |

所有 scoring、norm、router logits、response 和 selection objective 均在 FP32 中计算。最终导出时再转换回 checkpoint dtype。

---

## 3. 第一阶段：Effective-zero mandatory pruning

### 3.1 判定规则

不使用需要平方的 concat RMS 作为 zero 判定，避免 tiny sentinel 在 FP32 中平方下溢。直接定义：

$$
z_{e,c}
=
\max\left(
\|g_{e,c}\|_\infty,
\|u_{e,c}\|_\infty,
\|d_{e,c}\|_\infty
\right).
$$

给定：

$$
\tau_0=10^{-12},
$$

effective-zero set 为：

$$
\boxed{
\mathcal Z_e=\{c:z_{e,c}<\tau_0\}.
}
$$

### 3.2 强制约束

对任意 $c\in\mathcal Z_e$：

- Stable-AIMER score 强制设为 $-\infty$；
- 不参与 PP scoring；
- 不进入 rescue candidates；
- 不允许作为 merge representative；
- 不执行 compensation；
- 最终必须位于 prune set。

构建前先验证：

$$
|\mathcal Z_e|\le C-K.
$$

若该条件成立，所有 effective-zero channel 必须位于 prune set。若该条件不成立，则固定宽度与
“全部 zero 必剪”在数学上不可同时满足。此时允许且仅允许保留：

$$
F_e^{0}=|\mathcal Z_e|-(C-K)
$$

个 capacity-forced zero filler。它们按原始 channel index 升序确定性选择，始终排在所有 active
channel 之后，不参与 PP scoring、rescue、drop/merge representative 或 compensation。构建器
必须逐 expert 记录 `forced_zero_retained`，不能把该情况静默描述为 mandatory-prune 已完全满足。

实现中应保存独立的 `effective_zero_mask`，不能只依赖排序中的 $-\infty$。

---

## 4. 第二阶段：Stable-AIMER global ranking

对 active channel：

$$
\mathcal V_e=\{1,\ldots,C\}\setminus\mathcal Z_e,
$$

计算现有 Stable Concat-AIMER：

$$
s_{e,c}^{A}
=
\frac{
\sqrt{\operatorname{mean}(w_{e,c}^2)}
}{
\operatorname{mean}(|w_{e,c}|)+\epsilon_A
},
\qquad c\in\mathcal V_e.
$$

完整 score 定义为：

$$
\widetilde s_{e,c}^{A}
=
\begin{cases}
-\infty,&c\in\mathcal Z_e,\\
s_{e,c}^{A},&c\in\mathcal V_e.
\end{cases}
$$

按 score 从高到低稳定排序。score 并列时使用原始 channel index 升序打破平局。

取前 $K$ 个 channel 得到 Stable-AIMER baseline keep set：

$$
\boxed{
\mathcal K_e^A=\operatorname{TopK}_c\widetilde s_{e,c}^{A}
}
$$

以及 baseline prune set：

$$
\mathcal P_e^A=\{1,\ldots,C\}\setminus\mathcal K_e^A.
$$

当 $|\mathcal Z_e|\le C-K$ 时必须满足：

$$
\mathcal Z_e\subseteq\mathcal P_e^A.
$$

否则必须满足 retained zero 数量精确为 $F_e^0$，且所有 active channel 排在 zero filler 之前。

NAPS 后续只能在局部边界修改该 baseline，不能重新排序整个 expert。

---

## 5. 第三阶段：Router-derived pseudo-token bank

对第 $l$ 层 router：

$$
W_r^{(l)}\in\mathbb R^{E\times d}.
$$

第 $j$ 个 router row 为 $w_j^r$。使用该层真实的 pre-MoE RMSNorm 权重 $\gamma^{(l)}$ 和 `rms_norm_eps` 构造：

$$
x_j^P
=
\operatorname{RMSNorm}
\left(w_j^r;\gamma^{(l)},\epsilon_{\mathrm{norm}}\right).
$$

该层 pseudo-token bank 为：

$$
X^P=[x_1^P,\ldots,x_E^P]^\top\in\mathbb R^{E\times d}.
$$

第一版不增加 spectral probe、previous-write probe、随机 probe 或 calibration token，以保持方法归因清晰。

---

## 6. 第四阶段：严格复现 native Router

### 6.1 路由集合

计算 FP32 router logits：

$$
Z=X^P W_r^\top.
$$

对每个 pseudo token $j$，按照模型原生 Top-$k$ 得到：

$$
\mathcal T_j=\operatorname{TopK}(Z_j,k).
$$

Top-k 选择必须使用与模型运行时一致的稳定规则。构建器不能只读取 `num_experts_per_tok`，还必须通过 model adapter 显式记录：

- router score 类型；
- native Top-$k$；
- 是否先对全专家做 softmax；
- Top-k 后是否重新归一化；
- 是否存在 router bias；
- 是否存在 group-limited routing；
- 是否存在 routing scaling factor；
- 是否存在 shared expert。

### 6.2 路由权重

对于 Qwen3 和当前 Qwen3.6 vLLM 实现，选中 Top-k 后使用归一化权重。可等价写成：

$$
\pi_{j,e}
=
\frac{\exp Z_{j,e}}
{\sum_{e'\in\mathcal T_j}\exp Z_{j,e'}},
\qquad e\in\mathcal T_j.
$$

若未来模型的 `norm_topk_prob=false`，则不能使用上式，必须保留全专家 softmax 后的原始 selected probability。NAPS 的 routing helper 必须调用或逐项复现目标模型的 native router 语义，不能假定所有 MoE 模型都和 Qwen3 相同。

### 6.3 每个 expert 的 routed probes

对 expert $e$，定义：

$$
\mathcal I_e=\{j:e\in\mathcal T_j\},
\qquad
M_e=|\mathcal I_e|.
$$

仅使用这些真正被 native Router 分配给 expert $e$ 的 pseudo tokens：

$$
X_e^P=X^P[\mathcal I_e].
$$

Qwen3.6 的 shared expert 在本阶段保持不变。它不进入 routed expert 的 channel ranking，也不加入 per-expert output-loss denominator，避免用固定 shared path 稀释 routed expert 的剪枝误差。

---

## 7. 第五阶段：Route-consistent PP response

对 expert $e$ 的 routed probes 计算真实 SwiGLU intermediate response：

$$
A_e^P
=
\operatorname{SiLU}(X_e^P W_g^{(e)\top})
\odot
(X_e^P W_u^{(e)\top}),
$$

其中：

$$
A_e^P\in\mathbb R^{M_e\times C}.
$$

第 $c$ 个 channel 的 response signature 为：

$$
a_{e,c}=A_e^P[:,c].
$$

如果 $M_e=0$，该 expert 立即 fallback 到 Stable-AIMER，不执行 PP rescue、subset replacement 或 compensation。

---

## 8. 第六阶段：PP evidence quality gate

### 8.1 有效样本量

对 expert $e$ 的 routed weights $\pi_{m,e}$，定义：

$$
\boxed{
N_{\mathrm{eff},e}
=
\frac{
(\sum_{m\in\mathcal I_e}\pi_{m,e})^2
}{
\sum_{m\in\mathcal I_e}\pi_{m,e}^2+\epsilon
}
}
$$

$N_{\mathrm{eff}}$ 用于检测“有多个 probes，但权重几乎全部集中在一个 probe”这一情况。

### 8.2 加权 probe 有效秩

先对 routed pseudo tokens 做行 L2 归一化，得到 $\bar X_e^P$。定义小型加权 Gram matrix：

$$
G_e
=
\Omega_e^{1/2}
\bar X_e^P\bar X_e^{P\top}
\Omega_e^{1/2},
$$

其中：

$$
\Omega_e=\operatorname{diag}(\pi_{m,e}).
$$

有效秩为：

$$
\boxed{
r_{\mathrm{eff},e}
=
\frac{(\operatorname{tr}G_e)^2}
{\operatorname{tr}(G_e^2)+\epsilon}
}
$$

该计算只需要一个 $M_e\times M_e$ matrix，不需要构造 hidden-size covariance。

### 8.3 第一版预注册门槛

NAPS-v1 使用以下保守门槛。这些值是根据静态诊断选择的首版预注册值，不表示已经通过下游实验验证为最优：

$$
N_{\min}=2,
\qquad
r_{\min}=2,
\qquad
N_{\mathrm{sat}}=4,
\qquad
r_{\mathrm{sat}}=4.
$$

若：

$$
N_{\mathrm{eff},e}<N_{\min}
\quad\text{或}\quad
r_{\mathrm{eff},e}<r_{\min},
$$

则：

$$
B_e=0,
$$

该 expert 完全使用 Stable-AIMER。

否则定义 evidence confidence：

$$
q_e
=
\min\left(1,\frac{N_{\mathrm{eff},e}}{N_{\mathrm{sat}}}\right)
\min\left(1,\frac{r_{\mathrm{eff},e}}{r_{\mathrm{sat}}}\right).
$$

首版最大 replacement budget 预注册为：

$$
B_{\max}=\operatorname{round}(0.025C).
$$

每个 expert 的最大实际 rescue 数量为：

$$
\boxed{
B_e=\left\lfloor B_{\max}q_e\right\rfloor.
}
$$

若 $B_e=0$，同样 fallback 到 Stable-AIMER。

Native Top-k 的第 $k$ 与第 $k+1$ 个 logit margin 需要记录为稳定性诊断，但 NAPS-v1 不用 raw margin 直接调整 $B_e$，避免引入模型间不可比较的新阈值。

---

## 9. 第七阶段：从 AIMER prune set 生成 rescue candidates

只考虑 active AIMER-pruned channels：

$$
\mathcal P_e^{A,+}=\mathcal P_e^A\setminus\mathcal Z_e.
$$

### 9.1 Input-side activity score

定义 router-weighted mean absolute response：

$$
\boxed{
s_{e,c}^{\mathrm{act}}
=
\frac{
\sum_{m\in\mathcal I_e}\pi_{m,e}|a_{m,e,c}|
}{
\sum_{m\in\mathcal I_e}\pi_{m,e}+\epsilon
}
}
$$

该 score 用于发现 gate/up response 活跃的 channel。

### 9.2 Output-side proxy score

为避免候选生成阶段完全忽略 down projection，额外定义低成本 output proxy：

$$
\boxed{
s_{e,c}^{\mathrm{out}}
=
\sqrt{
\frac{
\sum_{m\in\mathcal I_e}\pi_{m,e}^2a_{m,e,c}^2
}{
\sum_{m\in\mathcal I_e}\pi_{m,e}^2+\epsilon
}
}
\|d_{e,c}\|_2
}
$$

不把 $s^{\mathrm{act}}$ 与 $s^{\mathrm{out}}$ 直接线性相加，因为二者量纲和分布不同。

### 9.3 候选集合

分别从 $\mathcal P_e^{A,+}$ 中取 Top-$B_e$：

$$
\mathcal R_e^{\mathrm{act}}
=
\operatorname{TopB_e}_{c\in\mathcal P_e^{A,+}}
s_{e,c}^{\mathrm{act}},
$$

$$
\mathcal R_e^{\mathrm{out}}
=
\operatorname{TopB_e}_{c\in\mathcal P_e^{A,+}}
s_{e,c}^{\mathrm{out}}.
$$

最终 rescue candidate pool 为：

$$
\boxed{
\mathcal R_e
=
\mathcal R_e^{\mathrm{act}}
\cup
\mathcal R_e^{\mathrm{out}}.
}
$$

因此：

$$
|\mathcal R_e|\le 2B_e.
$$

这里 $B_e$ 是最大允许的实际 replacement 数量，而不是 candidate pool 的硬大小。

---

## 10. 第八阶段：构造 AIMER drop candidates

从 Stable-AIMER keep set 的尾部取：

$$
D_e=\min(2B_e,K)
$$

个 channel，构成：

$$
\boxed{
\mathcal D_e
=
\text{AIMER-kept bottom-}D_e.
}
$$

Stable-AIMER 高置信冻结集合为：

$$
\mathcal F_e=\mathcal K_e^A\setminus\mathcal D_e.
$$

局部候选池为：

$$
\mathcal C_e=\mathcal D_e\cup\mathcal R_e.
$$

最终保留集合必须写成：

$$
\mathcal S_e=\mathcal F_e\cup\mathcal Q_e,
$$

其中：

$$
\mathcal Q_e\subseteq\mathcal C_e,
\qquad
|\mathcal Q_e|=D_e,
\qquad
|\mathcal Q_e\cap\mathcal R_e|\le B_e.
$$

该定义保证：

- $|\mathcal S_e|=K$；
- 每个 expert 最多 rescue $B_e$ 个 channel；
- Stable-AIMER baseline 始终是可行解：$\mathcal Q_e=\mathcal D_e$；
- 方法可以选择一个 rescue 都不做。

---

## 11. 第九阶段：Mask-only subset objective

### 11.1 完整 expert 输出

在 routed pseudo probes 上，完整 routed expert 输出为：

$$
Y_{e,\mathrm{full}}^P=A_e^P W_d^{(e)\top}.
$$

给定保留集合 $\mathcal S$：

$$
Y_{e,\mathcal S}^P
=
A_e^P[:,\mathcal S]
W_d^{(e)}[:,\mathcal S]^\top.
$$

### 11.2 Native-weighted output loss

由于 expert 输出最终乘 router gate，主目标使用 $\pi^2$：

$$
\boxed{
L_{\mathrm{native}}(\mathcal S)
=
\frac{
\sum_{m\in\mathcal I_e}
\pi_{m,e}^2
\|Y_{e,\mathrm{full}}^P[m]-Y_{e,\mathcal S}^P[m]\|_2^2
}{
\sum_{m\in\mathcal I_e}
\pi_{m,e}^2
\|Y_{e,\mathrm{full}}^P[m]\|_2^2
+\epsilon
}
}
$$

### 11.3 Uniform-weight robustness guard

为避免 Qwen3 中接近单 probe 的 router weight 支配所有 replacement，再计算：

$$
\boxed{
L_{\mathrm{uniform}}(\mathcal S)
=
\frac{
\sum_{m\in\mathcal I_e}
\|Y_{e,\mathrm{full}}^P[m]-Y_{e,\mathcal S}^P[m]\|_2^2
}{
\sum_{m\in\mathcal I_e}
\|Y_{e,\mathrm{full}}^P[m]\|_2^2
+\epsilon
}
}
$$

Native-weighted loss 是主目标，uniform loss 只作为 robustness guard，不与 AIMER score 或 PP score 相加。

### 11.4 Greedy swap selection

从：

$$
\mathcal S_e^{(0)}=\mathcal K_e^A
$$

开始。每一步考虑：

$$
p\in\mathcal R_e\setminus\mathcal S_e^{(t)},
\qquad
d\in\mathcal D_e\cap\mathcal S_e^{(t)},
$$

并构造：

$$
\mathcal S'=(\mathcal S_e^{(t)}\setminus\{d\})\cup\{p\}.
$$

一次 swap 只有同时满足以下条件才可接受：

$$
L_{\mathrm{native}}(\mathcal S')
<
L_{\mathrm{native}}(\mathcal S_e^{(t)})-\eta,
$$

$$
L_{\mathrm{uniform}}(\mathcal S')
\le
L_{\mathrm{uniform}}(\mathcal S_e^{(t)}),
$$

其中 $\eta$ 只用于过滤 FP32 数值噪声，第一版固定为相对当前 native loss 的 $10^{-4}$。

每轮选择 native-loss 改善最大的合法 swap。改善并列时依次使用：

1. uniform-loss 改善更大者；
2. rescue candidate 的 Stable-AIMER rank 更高者；
3. drop candidate 的 Stable-AIMER rank 更低者；
4. channel index 升序。

最多接受 $B_e$ 次 swap。若不存在合法改善，则立即停止。

最终 mask-only 集合记为：

$$
\boxed{
\mathcal S_e^{\mathrm{NAPS}}
}
$$

并必须满足 baseline guard：

$$
L_{\mathrm{native}}(\mathcal S_e^{\mathrm{NAPS}})
\le
L_{\mathrm{native}}(\mathcal K_e^A),
$$

$$
L_{\mathrm{uniform}}(\mathcal S_e^{\mathrm{NAPS}})
\le
L_{\mathrm{uniform}}(\mathcal K_e^A).
$$

如果任一条件失败，该 expert 完全回退到 $\mathcal K_e^A$。

---

## 12. NAPS-Mask 实验版本

NAPS-v1 首先只导出：

$$
W_g[\mathcal S_e^{\mathrm{NAPS}},:],
\qquad
W_u[\mathcal S_e^{\mathrm{NAPS}},:],
\qquad
W_d[:,\mathcal S_e^{\mathrm{NAPS}}].
$$

该版本不修改 retained `down_proj`，不执行 regression 或 merge，用于隔离 native-route rescue 与 subset selection 本身的效果。

已有实验已经证明 pseudo-space 上的高重构率可能严重误导下游判断。因此 NAPS-Mask 与下一节的 NAPS-Bounded-Merge 都必须完成相同的下游评测，并分别与 Stable-AIMER 以及彼此比较。

---

## 13. NAPS-Bounded-Merge 实验版本

### 13.1 只补偿新换出的 AIMER tail channel

不对全部：

$$
\{1,\ldots,C\}\setminus\mathcal S_e^{\mathrm{NAPS}}
$$

执行补偿。

只考虑原本由 Stable-AIMER 保留、但被 NAPS rescue 替换出去的 channel：

$$
\boxed{
\mathcal M_e
=
\mathcal D_e\setminus\mathcal S_e^{\mathrm{NAPS}}.
}
$$

因此：

$$
|\mathcal M_e|
=
|\mathcal S_e^{\mathrm{NAPS}}\cap\mathcal R_e|
\le B_e.
$$

Stable-AIMER 原本就剪除的深尾部 channel、effective-zero channel 以及未被 rescue 的 PP candidate 均保持 hard prune，不参与 merge。

### 13.2 Pairwise scalar coefficient

定义：

$$
\Lambda_e=\operatorname{diag}(\pi_{m,e}^2).
$$

对被新剪 channel $p\in\mathcal M_e$ 与 retained representative $r\in\mathcal S_e^{\mathrm{NAPS}}$：

$$
\beta_{p\to r}
=
\frac{
a_{e,r}^\top\Lambda_e a_{e,p}
}{
a_{e,r}^\top\Lambda_e a_{e,r}+\lambda_\beta
}.
$$

pairwise output residual 为：

$$
C_{p\to r}
=
\left\|
\Lambda_e^{1/2}
(a_{e,p}-\beta_{p\to r}a_{e,r})
\right\|_2^2
\|d_{e,p}\|_2^2.
$$

### 13.3 真正的一对一 matching

NAPS 使用 bipartite matching：

- 左侧节点为 $p\in\mathcal M_e$；
- 右侧节点为 $r\in\mathcal S_e^{\mathrm{NAPS}}$；
- 每个 $p$ 必须最多选择一个 representative；
- 每个 $r$ 最多吸收一个 $p$；
- 边代价为 $C_{p\to r}$；
- 不满足数值约束的边直接删除。

这与 many-to-one merge 明确区分。NAPS-v1 不允许多个 pruned channel 同时累加到同一个 retained `down_proj` column。

### 13.4 保守数值约束

第一版冻结以下约束，不根据下游结果调参：

$$
|\beta_{p\to r}|\le\beta_{\max}=1,
$$

$$
\frac{\|d'_{e,r}\|_2}
{\|d_{e,r}\|_2+\epsilon}
\le g_{\mathrm{column}}=1.25,
$$

$$
\frac{\|\Delta W_{d,e}\|_F}
{\|W_{d,e}[:,\mathcal S_e^{\mathrm{NAPS}}]\|_F+\epsilon}
\le g_{\mathrm{expert}}=0.05.
$$

若某条 matching edge 违反 $\beta$ 或 column growth 约束，则删除该 edge。若最终 expert-level delta 超过上限，则该 expert 整体回退到 mask-only weights。

### 13.5 最终 down projection 更新

对每个合法 matching pair $(p,r)$：

$$
\boxed{
d'_{e,r}=d_{e,r}+\beta_{p\to r}d_{e,p}.
}
$$

没有匹配到 pruned channel 的 retained column 保持不变。

### 13.6 Final-model objective guard

必须直接用更新后的 $W_d'$ 计算最终模型输出：

$$
\widehat Y_{e,\mathrm{merge}}^P
=
A_e^P[:,\mathcal S_e^{\mathrm{NAPS}}]
W_d'[:,\mathcal S_e^{\mathrm{NAPS}}]^\top.
$$

只有当 native-weighted 与 uniform-weighted output loss 都不高于 mask-only 版本时，才接受该 expert 的 merge。否则该 expert 回退到 NAPS-Mask。

这保证 subset objective 与最终实际导出的 `down_proj` 修改保持一致，而不是只优化独立 pairwise proxy。

---

## 14. 完整算法

对每一层、每个 routed expert：

1. 在 FP32 中计算 `effective_zero_mask`；
2. 对 active channels 计算 Stable Concat-AIMER；
3. 构造 $\mathcal K_e^A$ 与 $\mathcal P_e^A$；
4. 从 router rows 构造 RMSNorm pseudo-token bank；
5. 严格复现 native Router Top-k 与 routing weights；
6. 构造 $X_e^P$ 与 $A_e^P$；
7. 计算 $M_e$、$N_{\mathrm{eff},e}$、$r_{\mathrm{eff},e}$ 和 route margins；
8. evidence 不足时直接输出 $\mathcal K_e^A$；
9. evidence 充分时计算 $B_e$；
10. 从 AIMER prune set 生成 activity/output 双来源 rescue candidates；
11. 从 AIMER keep tail 构造 drop candidates；
12. 从 Stable-AIMER mask 开始做最多 $B_e$ 次合法 greedy swap；
13. 输出 NAPS-Mask retained set；
14. 基于同一 NAPS-Mask，对新换出的 AIMER tail channel 构建 bounded one-to-one scalar merge 版本；
15. 任一 expert 的 merge objective 或数值约束失败时，局部回退到 mask-only weights。

---

## 15. 模型与张量适配要求

### 15.1 Qwen3

每个 expert 的独立权重通常为：

```text
gate_proj.weight: [C, hidden_size]
up_proj.weight:   [C, hidden_size]
down_proj.weight: [hidden_size, C]
```

配置中：

```text
num_experts_per_tok = 8
norm_topk_prob = true
```

### 15.2 Qwen3.6

packed routed-expert 权重为：

```text
gate_up_proj: [num_experts, 2 * C, hidden_size]
down_proj:    [num_experts, hidden_size, C]
```

必须按照运行时 `.chunk(2, dim=-1)` 对应的顺序验证前半段 gate、后半段 up。不能只根据名字猜测 axis。

Qwen3.6 还存在 shared expert。NAPS 只裁剪 routed experts，shared expert 权重、gate 和运行路径保持完全不变。

### 15.3 Block/runtime 约束

当前 ranking/profile runtime 使用 `channel_block_size=64`。实现前必须明确最终导出路径是否允许任意 channel index selection：

- 若 exporter 根据 ranking 前 $K$ 个 index 物理裁剪，则 NAPS 可按单 channel 选择；
- 若运行时要求 retained set 本身按 block 对齐，则候选、swap 和 merge 必须全部提升为 block 单位；
- 不允许构建 channel-level NAPS mask 后，再由 block runtime 静默改变实际 retained set。

构建测试必须比较 ranking 前 $K$ 个 index 与导出 checkpoint 的真实 gate/up/down channel，保证完全一致。

---

## 16. Artifact metadata 与构建不变量

### 16.1 必须记录的 metadata

- source model/config/index SHA256；
- Stable-AIMER cache SHA256；
- effective-zero definition 与 $\tau_0$；
- AIMER epsilon、dtype、排序方向和 tie-break；
- RMSNorm tensor name 与 epsilon；
- router score、Top-k、renormalization 和 scaling 语义；
- shared-expert policy；
- $N_{\min}$、$r_{\min}$、$N_{\mathrm{sat}}$、$r_{\mathrm{sat}}$；
- $B_{\max}$ 与每个 expert 的 $B_e$；
- activity/output candidate 数量；
- greedy swap acceptance threshold；
- merge 是否启用；
- $\lambda_\beta$、$\beta_{\max}$ 和 norm-growth caps；
- channel block size、目标宽度和实际宽度；
- ranking/profile/export diagnostics SHA256。

### 16.2 每个 expert 的构建不变量

必须验证：

1. retained channel 恰好为 $K$ 个；
2. retained index 唯一且位于 $[0,C)$；
3. $\mathcal Z_e\cap\mathcal S_e=\varnothing$；
4. 所有 replacement 都来自 $\mathcal R_e$；
5. 所有 displaced channel 都来自 $\mathcal D_e$；
6. replacement 数量不超过 $B_e$；
7. evidence gate 失败的 expert 与 Stable-AIMER 完全相同；
8. NAPS-Mask 不修改任何 retained weight value，只修改 channel index；
9. NAPS-Merge 只修改 matched retained `down_proj` columns；
10. gate/up retained weights与 source checkpoint bitwise 相等；
11. Qwen3.6 shared expert 与 source checkpoint bitwise 相等；
12. ranking 前 $K$ 个 index 与 checkpoint 物理裁剪 index 完全一致。

---

## 17. 必须输出的诊断

### 17.1 Effective-zero

- 每层、每 expert 的 zero count；
- 全局 zero rate；
- zero 在 Stable-AIMER prune set 中的比例；
- zero 是否全部被最终剪除。

### 17.2 Router/probe evidence

- $M_e$；
- $N_{\mathrm{eff},e}$；
- $r_{\mathrm{eff},e}$；
- self-route Top-k/Top-1 比例；
- Top-k boundary margin；
- fallback expert 比例；
- $B_e$ 分布。

### 17.3 Candidate 与 selection

- activity/output candidate overlap；
- candidate 的 Stable-AIMER rank 分布；
- 每个 expert 的 attempted/accepted swap 数；
- 与 Stable-AIMER retained-set overlap；
- native/uniform loss before/after；
- 被 rescue 与 displaced channel 的 AIMER/PP/output score 分布。

### 17.4 Bounded merge

- merge-eligible channel 数；
- 成功 matching 数与 rejection 原因；
- $\beta$ 的均值、分位数和极值；
- per-column norm growth；
- expert-level $\|\Delta W_d\|_F/\|W_d\|_F$；
- mask-only 与 merge 后的 native/uniform pseudo loss；
- 回退到 mask-only 的 expert 比例。

所有统计至少汇总 mean、p10、median、p90、min、max，并保留 per-expert JSON 以支持后续 O/H exchange analysis。

---

## 18. 实验协议与结果判定

### 18.1 对照组

主实验只保留三个方法：

| 组别 | 方法 | 目的 |
| --- | --- | --- |
| A | Stable-AIMER | 当前最强且数值稳定的 backbone |
| B | NAPS-Mask | 验证 native-route rescue 与 subset selection |
| C | NAPS-Bounded-Merge | 与 B 并列验证低自由度补偿的真实收益与风险 |

B、C 两组都必须完成全部模型与预算的评测。C 必须复用 B 的 retained mask，只允许按第 13 节修改匹配到的 retained `down_proj` columns。不同时修改 AIMER 公式、pseudo-token source、candidate budget 和 merge 约束，避免无法归因。

### 18.2 模型与预算

必须同时验证：

- Qwen3 B9 / 25%；
- Qwen3 B6 / 50%；
- Qwen3.6 B9 / 25%；
- Qwen3.6 B6 / 50%。

不能只依据单模型或单稀疏度决定方法有效，因为此前 Qwen3 与 Qwen3.6 对 norm-asymmetric/joint concat channel 的响应不同，高稀疏度行为也明显不同。

### 18.3 Full6 固定协议

所有下游评测继续使用现有 full6 协议：

| Dataset | 样本数 | `max_tokens` |
| --- | ---: | ---: |
| ARC | 3548 | 2048 |
| HellaSwag | 10042 | 512 |
| WinoGrande | 1267 | 1024 |
| GSM8K | 1319 | 2048 |
| MATH-500 | 500 | 4096 |
| MMLU | 14042 | 2048 |

固定：

```text
eval_batch_size = 16
temperature = 0
do_sample = false
seed = 42
enable_thinking = false
timeout = 1200
```

### 18.4 执行与报告规则

#### Stage 0：静态构建诊断

任一条件失败则不启动下游评测：

- fixed-width、zero-mask、candidate source 或 export index 不变量失败；
- native router helper 与目标模型运行时不一致；
- 输出出现 NaN/Inf；
- merge 修改到 gate/up、shared expert 或未匹配 down column。

#### NAPS-Mask

对四个模型/预算设置全部完成构建与 full6，不设置进入 Bounded merge 的性能门槛。分别报告相对 Stable-AIMER 的逐任务差值、Macro 差值、retained-set overlap 和 replacement 统计。

如果只有部分模型或预算提升，则记录为模型/预算依赖结果，不在同一轮调整 $B_{\max}$、evidence threshold 或 candidate source。

#### NAPS-Bounded-Merge

不论 NAPS-Mask 的下游结果如何，都基于完全相同的 retained mask 构建并评测 Bounded merge。分别报告相对 Stable-AIMER 和 NAPS-Mask 的逐任务差值、Macro 差值、pseudo loss、$\beta$ 分布、down-norm growth 和 per-expert fallback 比例。

若 pseudo loss 改善但 full6 下降，结论应明确记为“pseudo-space merge objective 未转化为下游收益”，不在同一轮调整 $\lambda_\beta$、$\beta_{\max}$ 或 norm-growth cap。

---

## 19. 当前版本的最小消融顺序

完成 Stable-AIMER、NAPS-Mask 和 NAPS-Bounded-Merge 的主实验后，再按以下顺序做消融：

1. activity-only candidates vs activity/output union；
2. 仅 $M_e$ gate vs $N_{\mathrm{eff}}+r_{\mathrm{eff}}$ gate；
3. native-weighted objective only vs native + uniform guard；
4. mask-only vs bounded one-to-one merge；
5. one-to-one vs bounded many-to-one merge。

主实验完成前，不做 threshold sweep、candidate-width sweep、probe-source expansion 或 merge hyperparameter sweep。

---

## 20. 一句话总结

NAPS-v1 不是：

$$
\text{AIMER score}+\text{PP score}+\text{pseudo reconstruction}.
$$

而是：

$$
\boxed{
\begin{aligned}
&\text{显式删除 effective-zero channels；}\\
&\text{保留 Stable Concat-AIMER 的全局 active-channel 排序；}\\
&\text{让 router-row pseudo tokens 严格经过 native Router；}\\
&\text{用有效样本量和有效秩限制 PP 的挑战权限；}\\
&\text{从 AIMER prune set 中生成少量 activity/output rescue candidates；}\\
&\text{只在 AIMER keep tail 与 rescue pool 间做可回退的集合级 swap；}\\
&\text{并列评测 mask-only 与受约束的一对一 scalar merge；}\\
&\text{任何不稳定 expert 都局部回退到 Stable-AIMER。}
\end{aligned}
}
$$

该版本把主要科学问题收敛为：

> 在不破坏 Stable-AIMER 全局判断的前提下，route-consistent PP evidence 能否稳定识别 cutoff 附近的 AIMER 误剪 channel？

NAPS-Mask 回答 route-consistent PP evidence 是否能改善 cutoff selection；NAPS-Bounded-Merge 在完全相同的 retained mask 上独立回答低自由度 down-projection merge 是否带来额外下游收益。
