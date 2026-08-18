# 从动态 APA-BME 迁移到静态专家缩宽

> 状态：研究方案，尚未实现、尚无 Expert-Static 结构剪枝结果。
>
> 核心判断：当前 APA-BME 可以迁移到静态专家剪枝，将 768-channel 大专家结构化缩为 64、128、192、……、768-channel 的异构小专家。纯静态版本适合作为必要强基线和部署分支；更值得优先研究的版本是“APA 轨迹指导的静态容量上限 + 缩小后专家内部的动态执行”。

## 1. 研究问题

当前 Qwen3-30B-A3B 的 routed expert 具有 768 个 intermediate channels。APA-BME 先按 RMS-fidelity 对每个 physical expert 的 channels 排序，再拆成 12 个 64-channel prefix blocks，并对每个 token 动态决定：

\[
w_{l,t,e}\in\{0,\ldots,12\}.
\]

这里，`l` 表示 MoE layer，`t` 表示 token，`e` 表示 physical expert，`w` 表示该 token 实际执行的 prefix block 数。

静态专家缩宽把逐 token 宽度改为校准后固定的结构宽度：

\[
\bar w_{l,e}\in\{0,\ldots,12\}.
\]

随后永久删除 expert `e` 中位于 prefix 之外的 channels，使原 768-channel expert 变成宽度为

\[
d_{l,e}=64\bar w_{l,e}
\]

的异构小专家。

必须区分两类结构变化：

1. **专家内部缩宽**：保留原 physical expert identity 和 router 输出空间，只裁剪 expert FFN channels。这是本文档的主方向。
2. **拆成多个可独立路由的小专家**：改变 router 输出空间，让原大专家的不同子块成为新的 router targets。这会改变 MoE 架构，通常需要重新训练或蒸馏 router，不是当前方法的直接无训练迁移。

## 2. 为什么迁移是自然的

当前 APA-BME 已包含四层决策：

```text
跨层预算
→ token 内 parent/expert admission
→ parent/expert 执行宽度
→ expert 内部 channel 顺序
```

静态化后变为：

```text
全局存储或期望计算预算
→ 跨层预算
→ physical expert 固定宽度
→ expert 内部 RMS-prefix channels
→ 物理删除尾部权重
```

当前方案中的 RMS-fidelity 排序、64-channel 对齐、prefix 约束、边际效用建模和离散凹资源分配都可迁移。需要移除的是逐 token 在线 parent competition；需要新增的是结构成本定义、physical-expert 固定宽度求解、权重重建和真实部署评测。

## 3. 可直接复用的 RMS-fidelity 排序

当前每个 expert 的 channel score 为：

\[
F_{l,e,c}
=
\sqrt{\mathbb E[z_{l,e,c}^{2}]}
\left\|W_{\mathrm{down},l,e}[:,c]\right\|_2.
\]

对每个 expert 独立降序排列 channels，并连续组成 12 个 64-channel blocks。记归一化 block coverage 为：

\[
f_{l,e,1}\ge f_{l,e,2}\ge\cdots\ge f_{l,e,12}\ge0.
\]

当静态宽度为 `bar_w[l,e]` 时，只保留排序后的前 `64*bar_w[l,e]` 个 channels。

对 SwiGLU expert，必须使用相同 channel indices 同步裁剪：

- `gate_proj` 的对应输出行；
- `up_proj` 的对应输出行；
- `down_proj` 的对应输入列。

这样得到的仍是合法的 dense SwiGLU 小专家，而不是运行时稀疏 mask。

## 4. 静态 block value

### 4.1 直接聚合当前 parent utility

当前动态 block utility 可写为：

\[
u_{l,t,e,j}=p_{l,t,e}^{q}f_{l,e,j},
\]

其中 `p` 融合 router gate、AMP 与 AIMER 等 parent signal。

静态版本可在固定 train calibration set 上定义：

\[
\bar u_{l,e,j}
=
\mathbb E_t\left[
\mathbf 1(e\in\mathcal R_{l,t})p_{l,t,e}^{q}
\right]f_{l,e,j},
\]

其中 `R[l,t]` 是 token `t` 的原始 top-k routed expert 集合。

该定义同时考虑：

- expert 被路由到的频率；
- 被路由时的 parent/router 重要性；
- expert 内部第 `j` 个 block 的 RMS fidelity。

因为 `f[l,e,j]` 对 `j` 非增，同一个 expert 的静态 marginals 仍满足：

\[
\bar u_{l,e,1}\ge\cdots\ge\bar u_{l,e,12}.
\]

### 4.2 用 APA 动态轨迹作为 teacher

更推荐先在固定 train calibration set 上运行冻结的 APA，记录：

\[
w^{\mathrm{APA}}_{l,t,e}.
\]

定义第 `j` 个 block 的动态需求概率：

\[
d_{l,e,j}
=
\Pr_t\left(
w^{\mathrm{APA}}_{l,t,e}\ge j
\mid e\text{ routed}
\right).
\]

进一步定义需求加权价值：

\[
v_{l,e,j}
=
\mathbb E_t\left[
\mathbf 1(e\in\mathcal R_{l,t})
\mathbf 1(w^{\mathrm{APA}}_{l,t,e}\ge j)
p_{l,t,e}^{q}
\right]f_{l,e,j}.
\]

`v[l,e,j]` 衡量某个 physical expert 的第 `j` 个 block 在真实动态决策中被需要的频率和强度。以它分配静态宽度，相当于把 APA 的 token-conditioned support-width 决策蒸馏成可物理部署的异构结构。

这比直接使用

\[
\operatorname{round}\mathbb E_t[w^{\mathrm{APA}}_{l,t,e}]
\]

更合理，因为平均宽度无法区分“经常需要某个 block”和“少数关键 token 强烈依赖某个 block”。

## 5. 静态优化问题

### 5.1 存储预算

若目标是降低模型参数和权重显存，每个被保留的 physical block 都持续占用存储，成本近似为：

\[
C_{\mathrm{storage}}
\propto
\sum_{l,e}\bar w_{l,e}.
\]

可求解：

\[
\max_{\{\bar w_{l,e}\}}
\sum_{l,e}\sum_{j=1}^{\bar w_{l,e}}v_{l,e,j}
\]

满足：

\[
\sum_{l,e}\bar w_{l,e}\le B_{\mathrm{storage}},
\qquad
\bar w_{l,e}\in\{L,\ldots,12\}.
\]

第一版建议设置 `L=1`，即每个原始 physical expert 至少保留一个 64-channel block，从而不改变 router topology。

### 5.2 期望计算预算

若目标是降低期望 active FLOPs，则 block 成本依赖 expert route frequency：

\[
r_{l,e}=\Pr_t(e\in\mathcal R_{l,t}),
\]

\[
C_{\mathrm{compute}}
\propto
\sum_{l,e}r_{l,e}\bar w_{l,e}.
\]

此时优化为带不同 block cost 的 prefix allocation。可以使用：

- value-per-cost greedy 作为近似；
- 整数动态规划或 multiple-choice knapsack 得到精确解；
- 先分配逐层预算，再在层内求 expert widths，控制求解规模。

### 5.3 联合存储—计算预算

如需同时约束参数和期望 FLOPs，应显式写成双约束：

\[
\sum_{l,e}\bar w_{l,e}\le B_s,
\]

\[
\sum_{l,e}r_{l,e}\bar w_{l,e}\le B_c.
\]

不能用单一“剪枝率”同时代替参数、FLOPs、显存和延迟。这些量必须分别报告。

## 6. Prefix greedy 的理论迁移边界

当每个 64-channel block 成本相同，且每条 physical-expert chain 的静态 marginals 非增时，全局逐次选择最大可行 next marginal，可以精确求解定义的 prefix surrogate：

```text
bar_w[l,e] = L
repeat until the global block budget is exhausted:
    candidate[l,e] = value[l,e,bar_w[l,e]+1]
    choose the largest eligible candidate
    bar_w[l,e] += 1
```

该证明继承当前 APA 的离散凹 prefix 结构，但仍只证明 surrogate 最优，不证明最终 PPL、生成质量或吞吐最优。

如果计算成本按 route frequency 加权，不同 expert block 不再等成本，简单最大 marginal greedy 不再自动具有相同的全局最优性声明；需要改用精确 DP、knapsack 或明确标注近似算法。

## 7. 三条候选方法

### 7.1 RF Expert-Static

使用 RMS-fidelity、route frequency 和静态 parent importance 计算 `bar_u[l,e,j]`，直接求每个 physical expert 的固定宽度并物理裁剪。

目的：判断当前 RMS-prefix 几何本身能否产生强静态结构。

特点：

- training-free；
- 无在线 allocator；
- router identity 不变；
- 可直接减少参数和权重显存；
- 是最基础的结构剪枝基线。

### 7.2 APA-Distilled Expert-Static

先运行 APA calibration trajectory，再用 `v[l,e,j]` 分配静态宽度。

目的：把动态 allocator 作为 teacher，将其 token-conditioned block demand 蒸馏为静态异构结构。

特点：

- 部署时无在线动态决策；
- calibration 成本高于 RF Expert-Static；
- 可能保留更多 APA 的经验决策；
- 仍会丢失同一 expert 对不同 token 使用不同宽度的能力。

### 7.3 Static-Cap Dynamic-APA

为每个 physical expert 学习一个永久最大宽度：

\[
m_{l,e}\in\{1,\ldots,12\}.
\]

删除第 `m[l,e]+1` 到第 12 个 blocks；推理时 APA 只能在保留的物理容量内分配：

\[
w_{l,t,e}\in\{0,\ldots,m_{l,e}\}.
\]

完整链条为：

```text
冻结 APA 动态轨迹
→ 估计 physical-expert block demand
→ 学习静态容量上限 m[l,e]
→ 永久删除低需求尾部 blocks
→ 在缩小后的 experts 内继续执行动态 APA
```

这是最推荐的方向，因为它同时提供：

- 参数和权重显存下降；
- 保留 token-conditioned support-width 能力；
- 不改变原 router 的 expert identity；
- 64-channel 对齐和可分桶的异构 expert shapes；
- 从动态质量上界向实际部署模型迁移的清晰路径。

## 8. 为什么不能把 Rank-Static 直接当作结构剪枝

已有 Rank-Static 对每层的 router rank 使用固定宽度：

\[
q_l=(q_{l,0},\ldots,q_{l,7}).
\]

同一个 physical expert 在不同 token 上可能位于不同 router rank，因此仍会获得不同执行宽度。它不能直接编译成一个固定形状的 expert 权重张量。

真正可结构化部署的 Expert-Static 必须按 `(layer, physical expert)` 固定：

\[
\bar w_{l,e}.
\]

当前 80% full 结果为：

| 方法 | WikiText-2 full PPL |
|---|---:|
| LABC floor=1 | 13.221244 |
| Rank-Static | 12.054723 |
| 动态 APA | 11.922485 |
| Cached Token-Shuffle | 12.144752 |

Rank-Static 与动态 APA 只差 `0.132238` PPL，说明静态非均匀 width profile 能解释 APA 相对 LABC 的大部分收益；但 cached Token-Shuffle 在严格保持 width/support/compute 后仍退化 `0.222268` PPL，说明 token 与 width assignment 的匹配具有独立价值。

因此合理但尚待实验验证的预期是：

```text
动态 APA
≈ Rank-Static
≥ APA-Distilled Expert-Static
≥ RF Expert-Static
> Uniform Static
```

Rank-Static 结果不能替代真正的 physical-expert static experiment。

## 9. Whole-expert removal 的额外风险

静态设置 `bar_w[l,e]=0` 意味着永久删除整个 physical expert，而不是像 APA 那样只对当前 token 跳过一次。原 router 仍可能把后续 token 路由到这个已删除 expert。

可选处理包括：

1. 删除对应 router logit 并重新 top-k；
2. 把已删除 expert 的流量映射到相似 expert；
3. 微调或蒸馏 router；
4. 保留一个 null/no-op expert 并重新校准 gate mass。

这些都会引入新的结构和训练变量。因此第一阶段应固定：

\[
\bar w_{l,e}\ge1,
\]

只研究“大专家缩成至少64-channel的小专家”，保持 router topology 和 expert identity 不变。Whole-expert removal 应作为后续独立实验，不应混入首轮结果。

## 10. Completion 与恢复训练

静态结构剪枝后可比较三种恢复路径：

1. **No recovery**：直接使用缩小后的 expert，最能隔离结构分配本身的质量。
2. **Risk-limited completion**：复用当前 coverage 和 completion 机制，在推理时估计已删除尾部的 missing mass。
3. **LoRA/DoRA/distillation recovery**：剪枝后进行少量恢复训练，让保留 channels 吸收部分被删除函数。

建议至少报告：

```text
Static + no recovery
Static + current completion
Static + recovery tuning
Static-Cap Dynamic-APA + completion
```

如果目标是强调 training-free，主结果必须来自前两项，恢复训练只能作为额外上界；如果目标是部署效率，则可将恢复训练作为标准工程配置，但需与无训练结果分开归因。

## 11. 异构小专家的实现注意事项

结构剪枝后，不同 `(layer, expert)` 可能具有不同 intermediate width。实际 kernel 需要考虑：

- 按 64-channel 宽度将 experts 分桶；
- 同宽 experts 使用 grouped GEMM；
- 避免每个 expert 单独发起小 GEMM；
- 报告 weight packing 和重排开销；
- 区分理论 active channels 与真实 latency；
- 测量首 token latency、decode latency、吞吐、峰值显存和模型文件大小；
- 确认 shared expert 不被错误纳入 routed-expert 剪枝预算。

静态结构版本比当前 Python 动态 slicing 原型更容易获得真实速度收益，但异构 shapes 仍可能导致 kernel 碎片，不能只凭参数或 FLOPs 下降声称等比例加速。

## 12. 最小实验计划

### 阶段 A：只验证静态化是否成立

1. 从 train calibration 收集 `(layer, physical expert)` route frequency、parent score 和 RMS block coverage。
2. 实现 RF Expert-Static，约束每个 expert 至少保留一个 block。
3. 实现 APA-Distilled Expert-Static，使用冻结 APA trajectory，不读取 test PPL 选择 profile。
4. 在 50%、60%、80% 三个 matched budget 上直接运行 WikiText-2 full。
5. 同时报告参数、期望 active channels 和每 token FLOPs 分布。

### 阶段 B：验证结构收益与动态收益能否兼得

1. 根据 APA demand 学习 `m[l,e]`。
2. 构建 Static-Cap Dynamic-APA。
3. 在相同物理参数预算下比较纯静态与 hybrid。
4. 在相同 active-channel 预算下比较原始 APA 与 hybrid。
5. 记录动态 allocator 遇到 cap 的频率以及被 cap 截断的 utility mass。

### 阶段 C：部署和质量门禁

1. 至少一个 EvalScope 生成或推理任务；不能只报告 PPL。
2. 实际模型文件大小、加载显存、峰值显存。
3. prefill/decode latency 和吞吐。
4. grouped-GEMM 或实际后端限制。
5. 如使用恢复训练，报告训练 tokens、时间、显存和新增参数。

## 13. 必要对照

| 方法 | 物理参数减少 | Token-conditioned | Router 改变 | 用途 |
|---|---:|---:|---:|---|
| Dense MoE | 否 | 原始 top-k | 否 | 未剪枝基线 |
| Uniform Static | 是 | 否 | 否 | 等宽结构基线 |
| RF Expert-Static | 是 | 否 | 否 | RMS-prefix 静态基线 |
| APA-Distilled Static | 是 | 否 | 否 | 动态到静态蒸馏 |
| Rank-Static | 否 | 否/按 rank 固定 | 否 | 已有动态性对照，不是物理结构模型 |
| APA-BME | 否 | 是 | 否 | 动态质量上界 |
| Static-Cap Dynamic-APA | 是 | 是 | 否 | 推荐部署方法 |
| Whole-Expert Static | 是 | 否 | 是 | 后续高风险扩展 |

## 14. 结果判定规则

### 14.1 纯静态版本接近 APA

如果 APA-Distilled Expert-Static 与 APA 的差距很小，则说明当前方法的大部分收益可以固化为异构专家结构。论文应降低“逐 token 动态性”的中心地位，并强化 training-free structural compression 与部署证据。

### 14.2 纯静态明显退化，hybrid 接近 APA

这是最支持 Static-Cap Dynamic-APA 的结果：静态 tail removal 提供存储收益，保留的动态 width matching 提供质量收益。

### 14.3 静态和 hybrid 都明显退化

说明 APA 经常依赖每个 expert 的长尾 blocks，或者静态 calibration 不能稳定预测部署分布。此时不应继续调 test profile，应保留原动态方法，并把结构剪枝降为负结果或部署受限分支。

### 14.4 静态版本优于 APA

说明当前动态 utility 可能含有噪声，静态聚合产生了更稳定的正则化。应重新审视在线 allocator，而不是选择性忽略静态结果。

## 15. 创新边界

“把大 MoE expert 静态剪成不同宽度的小 expert”本身已有强相关工作覆盖。MoE-Slimming 已研究跨层、跨 expert 的静态 channel budget、64/128/256 对齐、异构固定宽度、0-channel expert、Qwen3-30B-A3B 和剪枝后恢复训练。因此以下说法不能作为主创新：

- 首次做 MoE expert 内部 channel pruning；
- 首次让不同 experts 使用不同静态宽度；
- 首次使用 64-channel 对齐；
- 仅把 768 channels 拆成 12 个 blocks 就形成新方法。

当前可能保留的差异应定位为：

> 使用无需更新主模型的 token-conditioned APA support-width trajectory，估计每个 physical expert 的真实 prefix-block demand；将动态决策蒸馏为64-channel对齐的结构容量上限，并在物理缩小后的异构 experts 内继续进行严格预算的动态 micro-expert execution。

纯 Expert-Static 更适合作为必要强 baseline、部署分支和动态到静态蒸馏实验；Static-Cap Dynamic-APA 才是更可能同时保留当前论文差异和真实部署价值的扩展方向。

## 16. 推荐优先级

建议按以下顺序推进：

1. **APA-Distilled Expert-Static**：最先回答“当前方案能否直接固化为小专家结构”。
2. **RF Expert-Static**：作为不依赖动态 teacher 的干净基线。
3. **Static-Cap Dynamic-APA**：若静态版与 APA 存在可见差距，验证混合方案能否兼顾参数和质量。
4. **真实模型重建与 grouped-GEMM**：只有质量门禁通过后再投入 kernel 工程。
5. **Whole-expert removal/router surgery**：最后单独研究，不混入首轮专家缩宽实验。

第一阶段的停止条件应预先固定：完成 train-calibrated physical-expert width profile，在 50%、60%、80% 三个 full 协议上得到无恢复训练结果，并与 Uniform Static、Rank-Static 和 APA 严格区分预算与模型结构。少量 window 只能用于排错，不能承担方法选择。

