# 从平均效用到剪枝边界后悔度：面向混合专家模型的尾部风险约束静态异构缩宽

> 中文论文草稿：Motivation 与 Method
>
> 本文档只总结当前仓库中已经实现并获得相应验证证据的方法。为避免过度陈述，本文将
> Frontier Committee Regret 作为当前正式 PPL 最优的核心方法，将 Reference-Centered
> Route Envelope 作为已经通过跨 checkpoint、train-only holdout 门禁的计算稳健扩展，
> 但不把后者描述为已经完成 validation/test 验证的最终结论。

## 1. Motivation

### 1.1 为什么需要把“大专家”静态缩成“小专家”

混合专家模型（Mixture-of-Experts, MoE）通过路由器仅激活少量专家，使模型能够在不同比例地
增加单 token 计算量的情况下扩展参数容量。然而，稀疏激活并不等价于低部署成本。即使每个
token 只访问少数专家，完整 checkpoint 仍需存储所有专家参数；被路由到的专家也仍执行完整
的中间维度。因此，MoE 的参数容量、显存占用和实际 routed compute 之间存在明显错位。

一种自然思路是对每个物理专家赋予固定但允许不同的中间宽度，将原始专家静态缩成一组
异构“小专家”。与逐 token 动态宽度相比，静态 profile 具有三个部署优势：第一，结构在推理
前冻结，可以被哈希、审计和物化；第二，执行路径规则，适合固定 kernel、编译和批处理；第三，
无需在线控制器，不会在生成过程中产生额外的 token-level 决策开销。

但是，已有研究已经覆盖异构 expert width、全局 channel budget、prefix slicing、router 或
activation importance 等基本思想。因此，本文的研究问题不是“能否给不同专家分配不同
宽度”，而是：

> 在不读取 validation/test 指标、不改变路由器、不给模型恢复训练的条件下，如何识别真正
> 不可删除的物理专家前缀，并在精确结构预算下构造既低失真、又对 routed-compute 漂移可审计
> 的静态异构宽度 profile？

### 1.2 现有静态重要性指标存在四类层级错配

#### 1.2.1 路由频率不等于条件效用

令某层路由集合为 \(\mathcal R_{l,t}\)，路由权重为 \(g_{l,t,e}\)。单独使用 route count
隐含地假设“被调用得越多就越重要”。这一假设忽略了两个事实：不同专家被调用后对输出的
边际影响不同；同一个专家的价值还取决于它与哪些专家共同组成 routed committee。因而，
全局频率或独立 expert prior 不能完整描述专家在当前 routed set 中的条件作用。

本项目的消融表明，将 AMP/AIMER 先与 route count 做静态相乘，不如在每个 token 的实际
routed set 内归一化后再聚合。这说明静态容量应从 token 条件效用中蒸馏，而不是从彼此独立的
专家标量中直接拼接。

#### 1.2.2 平均效用会掩盖稀有但灾难性的路径

期望效用目标倾向于把容量分配给高频、平均贡献稳定的专家，却可能把极少激活、但一旦删除
就会造成巨大误差的路径分配为零宽度。强剪枝实验中，Conditional Dual-Utility 曾将一个关键
物理专家完全删除，使 PPL 急剧恶化；只恢复该专家最前面的一个或两个 64-channel block，便能
恢复大部分质量。这一现象揭示了均值目标与尾部风险的本质差异：

\[
\text{small expected loss}\;\not\Rightarrow\;\text{small rare-event loss}.
\]

因此，常规容量分配应由条件期望效用负责，而低频灾难路径应由单独的尾部安全约束负责。

#### 1.2.3 整专家标量与实际剪枝单元不在同一层级

整专家 output norm、路由唯一性或 expert-level contribution 都把一个专家压缩为单个标量。
但本文实际删除的是专家内部的 64-channel prefix block。某专家整体输出很大，不代表它的
“第一个将被删除的 block”不可替代；反之，整体不显著的专家也可能包含一个与同层其他
专家输出方向高度正交的关键 frontier block。

仓库中的多组负结果验证了这一层级错配：output-contribution 融合、output safety floor 和
co-routing uniqueness floor 均未稳定超过 Tail-Risk fallback。真正产生稳定增益的是将
committee non-redundancy 直接计算在冻结参考 profile 的 first-pruned block 上。由此得到本文
的核心原则：

> 重要性估计的粒度应与最终结构决策的粒度一致；若决策变量是 expert-prefix block，则
> 不可替代性也应在 block frontier 上估计。

#### 1.2.4 结构预算相同不代表实际 routed compute 相同

静态结构预算是所有专家保留 block 数之和，而推理计算取决于路由分布加权后的宽度。设
\(p_{l,e}\) 为某数据分布下专家被路由到的概率，\(w_{l,e}\) 为保留宽度，则 routed cost
近似为

\[
C_p(\mathbf w)=\sum_{l,e}p_{l,e}w_{l,e}.
\]

两个 profile 即使具有完全相同的结构 block 数，也可能因为容量是否落在热门专家上而产生
不同的执行计算。更困难的是，训练语料上的 \(p_{l,e}\) 与评估或部署分布上的路由概率并不
相同。有限 train folds 上满足计算约束，并不能自动保证新分布逐场景不劣。当前实验中，
Frontier Committee Regret 在多个 checkpoint 上稳定降低 PPL，却反复伴随很小的 evaluation
routed-compute concession，正是这一 train-to-evaluation route shift 的表现。

### 1.3 从单一 profile 搜索转向“参考方案 + 条件增强 + 可回退选择”

上述失效模式说明，不存在一个已被当前证据支持、可在所有 MoE 架构上无条件启用的统一
打分公式。例如，完整 Tail-Risk 方法在 Qwen3/Qwen3.5 上有效，但在 Qwen1.5-MoE 上没有
超过更简单的 Route×RMS；相反，Route×Tail 在跨架构实验中表现得更稳健。

因此，本文采用分层方法而不是无条件替换：

1. 用 Route×Tail 构造低假设、跨架构较稳健的静态 fallback；
2. 只有当不重叠 train-only folds 支持时，才启用 Conditional Dual 与 Tail-Risk refinement；
3. 只有当 block-level Frontier Committee Regret 在独立选择折上稳定获胜时，才恢复高后悔度
   frontier blocks；
4. 以冻结参考 profile 为中心约束 routed cost；若多场景约束不可行，则拒绝候选而不是静默
   放宽预算；
5. profile、校准 cache、选择决策和评估 token cache 全部记录 SHA256，且在读取 evaluation
   split 前冻结。

这一设计把“寻找一个永远正确的剪枝指标”转化为“在可审计的 fallback 之上，只接受被
train-only 证据支持的结构增量”。它既回应了跨架构差异，也避免使用 test PPL 反复搜索 profile。

### 1.4 本文的核心观点

本文方法建立在以下三个互补观点之上：

1. **条件容量观点**：常规容量应由 routed-set 内的条件效用决定，而不是由独立专家频率决定；
2. **风险—后悔度分解观点**：activation tail 负责阻止灾难性路径被置零，frontier committee
   regret 负责识别在当前剪枝边界上无法被同层专家委员会替代的 block；
3. **参考中心稳健性观点**：计算约束应围绕冻结参考 profile 描述候选的增量成本，而不是只
   拟合单一平均路由分布。

这三个观点分别对应“分配多少常规容量”“哪些最小前缀绝不能删除”和“容量重新分配后是否
仍满足 routed-compute 约束”，从而形成一个统一的静态 MoE 缩宽框架。

## 2. Method

### 2.1 问题定义

考虑一个具有 \(L\) 个 MoE 层的模型。第 \(l\) 层包含 \(E_l\) 个物理专家，每个专家的
中间通道按照一个由 train-only calibration 得到的固定排列划分为 \(K_l\) 个连续 block。
在当前 Qwen3 实现中，每个 block 包含 64 个通道，原始专家包含 12 个 block。

对物理专家 \((l,e)\)，定义整数宽度

\[
w_{l,e}\in\{0,1,\ldots,K_l\}.
\]

宽度 \(w_{l,e}\) 表示仅保留排序后的前 \(w_{l,e}\) 个 block。对应的二值选择变量为

\[
x_{l,e,j}(\mathbf w)=\mathbb I[j<w_{l,e}],
\]

因此天然满足 prefix 单调约束

\[
x_{l,e,j+1}\le x_{l,e,j}.
\]

在目标结构预算 \(B\) 下，所有候选必须满足

\[
\sum_{l=1}^{L}\sum_{e=1}^{E_l}w_{l,e}=B.
\]

本文所有方法比较均在相同 block 大小和相同精确结构预算下进行。结构剪枝率定义为

\[
\rho_{\mathrm{struct}}
=1-\frac{B}{\sum_l E_lK_l}.
\]

本文不改变原始 router，也不以 router rank 代替物理专家索引。所有宽度、风险 floor 和
计算成本均以 \((l,\text{physical expert})\) 为索引。

### 2.2 物理专家通道排序与前缀覆盖

对于专家中间通道 \(c\)，先根据训练语料上的激活统计和 down-projection 列权重建立固定排序。
令 \(a^{\mathrm{typ}}_{l,e,c}\) 表示典型激活覆盖，\(a^{\mathrm{tail}}_{l,e,c}\) 表示尾部激活
覆盖。本文使用几何插值构造尾部感知通道分数：

\[
s_{l,e,c}
=\left(a^{\mathrm{typ}}_{l,e,c}+\epsilon\right)^{1-\lambda}
 \left(a^{\mathrm{tail}}_{l,e,c}+\epsilon\right)^{\lambda},
\]

其中当前已验证配置使用 \(\lambda=0.5\)。每个专家内部按 \(s_{l,e,c}\) 降序排列通道，再
按 64-channel 划分 block。令 \(c_{l,e,j}\ge0\) 表示第 \(j\) 个 block 的边际覆盖分数，
并保证

\[
c_{l,e,0}\ge c_{l,e,1}\ge\cdots\ge c_{l,e,K_l-1}.
\]

固定排序把任意宽度映射为连续前缀，避免非结构化掩码；非增边际还使后续精确预算分配可以
由全局 next-block greedy 等价求解。

### 2.3 Routed-Set Conditional Dual Utility

#### 2.3.1 token 条件双效用

对于第 \(l\) 层 token \(t\) 的实际 routed set \(\mathcal R_{l,t}\)，令 \(g_{l,t,e}\)
为路由权重，\(A^{\mathrm{AMP}}_{l,e}\) 与 \(A^{\mathrm{AIMER}}_{l,e}\) 为两类冻结的专家
先验。先在当前 routed set 内分别归一化：

\[
q^{\mathrm{AMP}}_{l,t,e}
=\frac{g_{l,t,e}A^{\mathrm{AMP}}_{l,e}}
{\sum_{e'\in\mathcal R_{l,t}}g_{l,t,e'}A^{\mathrm{AMP}}_{l,e'}+\epsilon},
\]

\[
q^{\mathrm{AIMER}}_{l,t,e}
=\frac{g_{l,t,e}A^{\mathrm{AIMER}}_{l,e}}
{\sum_{e'\in\mathcal R_{l,t}}g_{l,t,e'}A^{\mathrm{AIMER}}_{l,e'}+\epsilon}.
\]

定义条件双效用

\[
u_{l,t,e}
=\sqrt{q^{\mathrm{AMP}}_{l,t,e}q^{\mathrm{AIMER}}_{l,t,e}+\epsilon}.
\]

实现中将当前 token 的 top-1 routed expert 赋予单位 parent score，以防主路由路径在局部
归一化后被意外压低。所有正式 Tail-Risk profile 的 teacher 均显式要求
`parent_mode=dual`，不允许 collector 隐式选择其他模式。

#### 2.3.2 从动态条件效用蒸馏静态容量

将 token 条件效用与 block 覆盖结合，并按物理专家聚合：

\[
\widetilde v_{l,e,j}
=\sum_t\mathbb I[e\in\mathcal R_{l,t}]\,
u_{l,t,e}\left(c_{l,e,j}+\epsilon\right).
\]

为将旧 teacher 中的 expert-level utility 与新的 tail-aware 通道排序解耦，先恢复专家效用

\[
U_{l,e}
=\frac{\sum_j\widetilde v^{\mathrm{old}}_{l,e,j}}
{\sum_j c^{\mathrm{old}}_{l,e,j}+\epsilon},
\]

再绑定到新的 block coverage：

\[
v^{\mathrm{base}}_{l,e,j}
=U_{l,e}\left(c^{\mathrm{tail}}_{l,e,j}+\epsilon\right).
\]

这一 factorization 将“专家在 routed committee 中应获得多少总容量”与“该容量应落在哪些
专家内部通道上”分开：前者由 conditional dual utility 决定，后者由 typical/tail coverage
决定。

### 2.4 Tail-Risk Safety Floor

平均条件效用仍可能把低频灾难专家分配为零。为此，对专家 \((l,e)\) 定义 train-only 尾部
风险代理：

\[
r_{l,e}
=\max_c\left(A^{\max}_{l,e,c}\,\lVert W^{\mathrm{down}}_{l,e,:,c}\rVert_2\right),
\]

其中 \(A^{\max}_{l,e,c}\) 是校准语料上通道的最大绝对中间激活，
\(W^{\mathrm{down}}_{l,e,:,c}\) 是相应 down-projection 列。该指标近似一个稀有激活经输出
投影放大后可能造成的最坏单通道影响。

使用全层统一阈值

\[
\tau_{\mathrm{risk}}
=\max\left(Q_{0.995}(\{r_{l,e}\}),\;0.1\max_{l,e}r_{l,e}\right),
\]

并定义最小宽度约束

\[
m^{\mathrm{risk}}_{l,e}
=\begin{cases}
2,&r_{l,e}\ge\tau_{\mathrm{risk}},\\
0,&\text{otherwise}.
\end{cases}
\]

当前正式方法使用全层 selector，不依赖人工 Super-Expert ID 或手工 layer whitelist。被 floor
保护的 block 仍计入同一个结构预算；求解器必须从其他低边际效用 block 回收完全相同的数量。
因此，Tail-Risk 不是额外增加模型容量，而是在等预算下把少量容量从低风险区域转移到灾难
尾部路径。

### 2.5 Frontier Committee Regret

#### 2.5.1 为什么只评价 first-pruned block

设 \(\mathbf w^0\) 为冻结的 Tail-Risk reference profile。对专家 \((l,e)\)，其实际剪枝
边界为 \(j=w^0_{l,e}\)：\(j<w^0_{l,e}\) 已保留，而 \(j=w^0_{l,e}\) 是第一个被删除的
block。本文不对所有 block 重新做无约束全局排序，而只问一个与当前结构决策直接对应的
反事实问题：

> 如果在 reference profile 上再恢复这个 first-pruned block，它能否提供同层其他 routed
> experts 无法替代的输出方向？

这种 reference-centered 评价避免让整专家显著性掩盖真正处于剪枝临界点的局部结构。

#### 2.5.2 委员会残差估计

对 token \(t\) 上被路由的专家 \(e\)，令 \(\mathbf y_{t,e}\) 为其加权输出，其他 routed
专家构成的委员会输出为

\[
\mathbf y_{t,-e}=\sum_{e'\in\mathcal R_{l,t}\setminus\{e\}}\mathbf y_{t,e'}.
\]

将其归一化为 \(\widehat{\mathbf y}_{t,-e}\)。对专家中间通道 \(c\)，令
\(h_{t,e,c}\) 为非线性中间激活，\(\mathbf d_{l,e,c}\) 为 down-projection 的第 \(c\) 列。
通道在委员会输出正交补空间中的近似能量为

\[
\delta_{t,l,e,c}
=|g_{l,t,e}|^2h_{t,e,c}^2
\left(
\lVert\mathbf d_{l,e,c}\rVert_2^2
-\langle\widehat{\mathbf y}_{t,-e},\mathbf d_{l,e,c}\rangle^2
\right)_+.
\]

对排序后 block \(\mathcal B_{l,e,j}\) 聚合：

\[
R_{t,l,e,j}
=\sqrt{\sum_{c\in\mathcal B_{l,e,j}}\delta_{t,l,e,c}}.
\]

该估计保留下投影 Gram 矩阵的对角项，避免为每个 token、expert 和 block 显式物化完整
hidden-size 输出。它不是精确的全协方差正交分解，而是一个计算可控、非负的 diagonal-Gram
近似。

#### 2.5.3 跨折稳健聚合与 frontier floor

在 \(F\) 个互不重叠的 train-only estimator folds 上分别计算 block residual。为消除层间
尺度差异，每个 fold、每一层先除以该层 eligible frontier score 的均值。令归一化结果为
\(\bar R^{(f)}_{l,e,j}\)。本文采用 minimum-fold 聚合：

\[
S^{\mathrm{FCR}}_{l,e}
=\min_{f=1}^{F}\bar R^{(f)}_{l,e,w^0_{l,e}}.
\]

仅对 \(w^0_{l,e}<K_l\) 的专家定义 frontier score。以全局 99.5% 分位数为阈值，构造

\[
m^{\mathrm{FCR}}_{l,e}
=\begin{cases}
\min(w^0_{l,e}+1,K_l),
&S^{\mathrm{FCR}}_{l,e}\ge Q_{0.995}(S^{\mathrm{FCR}}),\\
0,&\text{otherwise}.
\end{cases}
\]

最终 floor 为

\[
m_{l,e}=\max\left(m^{\mathrm{risk}}_{l,e},m^{\mathrm{FCR}}_{l,e}\right).
\]

minimum-fold 使某个 block 只有在所有 estimator folds 上都保持较高相对残差时才被保护，
从而避免单个区间的瞬时异常值直接形成结构约束。

### 2.6 精确结构预算下的静态前缀分配

基础分配问题写为

\[
\max_{\mathbf w}\quad
\sum_{l,e}\sum_{j=0}^{w_{l,e}-1}v^{\mathrm{base}}_{l,e,j}
\]

满足

\[
\sum_{l,e}w_{l,e}=B,
\qquad
m_{l,e}\le w_{l,e}\le K_l,
\qquad
w_{l,e}\in\mathbb Z.
\]

由于每个 block 具有相同结构成本，且 \(v^{\mathrm{base}}_{l,e,j}\) 在每个 expert 内非增，
可以从所有 expert 的“下一个可选 block”中反复选择边际价值最大的一个，直到达到 \(B\)。
实现使用稳定全局排序与逐 expert 计数得到等价解，平局时优先较早 block，从而保持 prefix
可行性。

**命题 1（等成本前缀分配的最优性）**  若每个 expert 内的边际价值非增，所有 block 的
结构成本相同，且 floor/cap 均为合法前缀，则全局 next-block greedy 给出上述离散问题的
最优解。

**证明思路。** 任意可行解均可表示为从若干非增序列中各取一个前缀。若某解包含边际价值
较小的已选 block，却遗漏了某个可行且价值更大的 next block，则交换二者不会破坏前缀约束，
并能提高目标值。不断进行交换即可得到 greedy 解。因此不存在比 greedy 更优的可行解。

该命题只对应“等结构 block 成本 + 非增边际”的内部优化问题，不代表这些 proxy values
等于真实语言建模损失。

### 2.7 Train-Only Routed-Compute Calibration

#### 2.7.1 单一训练分布下的计算锚点

对训练路由计数 \(n_{l,e}\)，定义保留 routed blocks

\[
C_n(\mathbf w)=\sum_{l,e}n_{l,e}w_{l,e}.
\]

为了在保持 \(B\) 不变的同时接近目标 routed-pruning ratio，可构造拉格朗日调整后的边际：

\[
\widetilde v_{l,e,j}(\mu)
=\frac{v_{l,e,j}}{s_v}-\mu\frac{n_{l,e}}{s_n}.
\]

对给定 \(\mu\)，仍使用精确前缀 allocator；再通过一维搜索选择最接近目标 routed cost 的
离散 profile。由于同一 expert 的所有 block 具有相同计算代价，减去该代价不会破坏 expert
内部的边际非增性。

#### 2.7.2 多折 non-inferiority 约束

给定冻结参考 profile \(\mathbf w^0\) 和 \(F\) 个 train route folds，候选满足

\[
C_f(\mathbf w)\le C_f(\mathbf w^0),\qquad f=1,\ldots,F,
\]

其中

\[
C_f(\mathbf w)=\sum_{l,e}p^{(f)}_{l,e}w_{l,e}.
\]

本文用 projected multi-dual search 求解：

\[
\widetilde v_{l,e,j}(\boldsymbol\mu)
=\frac{v_{l,e,j}}{s_v}
-\sum_f\mu_f\frac{p^{(f)}_{l,e}}{s_c},
\qquad \mu_f\ge0.
\]

每次 dual 更新的内层问题仍由精确结构 allocator 解决。若在预设迭代和容差内找不到满足
全部约束的离散 profile，系统输出不可行证书并拒绝候选，不会静默降低 floor、放宽结构预算
或删除某个不利 fold。

### 2.8 Reference-Centered Route Envelope

有限 folds 上的硬约束只能保证观测场景，不能直接覆盖 unseen route shift。本文进一步围绕
冻结参考 profile 构造坐标级路由不确定包络。

首先将每个 fold 的 route counts 归一化为分布 \(p^{(f)}_{l,e}\)，并计算

\[
p^-_{l,e}=\min_f p^{(f)}_{l,e},\qquad
p^+_{l,e}=\max_f p^{(f)}_{l,e},
\]

\[
\Delta_{l,e}=p^+_{l,e}-p^-_{l,e}.
\]

给定预注册扩张系数 \(\alpha\)，定义

\[
\underline p_{l,e}=\max(0,p^-_{l,e}-\alpha\Delta_{l,e}),
\qquad
\overline p_{l,e}=p^+_{l,e}+\alpha\Delta_{l,e}.
\]

当前已验证的 train-only 配置使用 \(F=8\) 和
\(\alpha=1/\sqrt{8}\)，但本文不将该半径宣称为普适统计保证。

关键设计是以 reference width \(w^0_{l,e}\) 为中心定义 block cost：

\[
q_{l,e,j}=\begin{cases}
\underline p_{l,e},&j<w^0_{l,e},\\
\overline p_{l,e},&j\ge w^0_{l,e}.
\end{cases}
\]

候选相对 reference 增加的 block 使用 upper route cost，删除的 reference block 使用 lower
route cost。于是 envelope 约束为

\[
\sum_{l,e,j}q_{l,e,j}
\left(x_{l,e,j}(\mathbf w)-x_{l,e,j}(\mathbf w^0)\right)\le0.
\]

**命题 2（参考中心增量成本上界）**  对任意满足
\(p_{l,e}\in[\underline p_{l,e},\overline p_{l,e}]\) 的路由分布，候选相对参考 profile 的
retained-cost 增量满足

\[
C_p(\mathbf w)-C_p(\mathbf w^0)
\le
\sum_{l,e}
\left[
\overline p_{l,e}(w_{l,e}-w^0_{l,e})_+
-\underline p_{l,e}(w^0_{l,e}-w_{l,e})_+
\right].
\]

**证明。** 当 \(w_{l,e}>w^0_{l,e}\) 时，增量系数为正，最坏情况取
\(p_{l,e}=\overline p_{l,e}\)；当 \(w_{l,e}<w^0_{l,e}\) 时，增量系数为负，其最大值在
\(p_{l,e}=\underline p_{l,e}\) 处取得。对所有坐标求和即可。

因此，只要 envelope 右侧不大于零，候选对该坐标盒中的任一路由分布都不会比 reference
保留更多 routed blocks。该命题是确定性的 box-uncertainty 结论，但前提是未来路由分布确实
落在所构造的区间内；当前实验并未证明这一覆盖事件具有 distribution-free 概率保证。

### 2.9 Frontier Committee Regret 的理论解释

设某个 frontier block 的加权输出为 \(\mathbf z_{t,l,e,j}\)，其他 routed experts 的输出张成
子空间 \(\mathcal S_{t,l,-e}\)。理想的不可替代性可写为

\[
\left\lVert
(I-\Pi_{\mathcal S_{t,l,-e}})\mathbf z_{t,l,e,j}
\right\rVert_2,
\]

其中 \(\Pi\) 是正交投影。直接计算该量需要显式形成 block 输出和委员会子空间，代价较高。
本文的 diagonal-Gram estimator 将委员会压缩为当前 `other_output` 方向，并忽略不同
down-projection 列之间的交叉项，得到第 2.5 节的非负近似。

**命题 3（方向冗余的零响应）**  在单方向委员会近似下，若某通道的 down-projection 列
\(\mathbf d_{l,e,c}\) 完全平行于 \(\mathbf y_{t,-e}\)，则该通道的 committee residual 为零；
若完全正交，则 residual 保留其完整的加权通道输出能量。

这一性质解释了 FCR 与 output norm 的差异：output norm 衡量“输出有多大”，FCR 近似衡量
“这部分输出中有多少不能由当前 routed committee 的主方向解释”。因此，FCR 更适合决定
一个 frontier block 是否值得从零宽度或较窄宽度恢复。

需要强调的是，diagonal-Gram 省略通道间协方差，minimum-fold 也只是经验稳健聚合。命题 3
描述的是估计器的代数性质，而不是对真实 PPL 改善的理论保证。

### 2.10 Cross-Fitted Applicability Selector

由于 Tail-Risk/Conditional Dual 在不同架构上的相对排序可能翻转，本文不无条件启用复杂
refinement。设 fallback profile 为 \(\mathbf w^{\mathrm{fb}}\)，候选为
\(\mathbf w^{\mathrm{cand}}\)。在与 estimator/calibration 区间不重叠的 \(H\) 个 train-only
selection folds 上计算完整 PPL。只有同时满足

\[
\#\{h:\mathrm{PPL}^{(h)}_{\mathrm{cand}}<
\mathrm{PPL}^{(h)}_{\mathrm{fb}}\}\ge \lceil(H+1)/2\rceil
\]

以及

\[
\frac1H\sum_h\mathrm{PPL}^{(h)}_{\mathrm{cand}}
<
\frac1H\sum_h\mathrm{PPL}^{(h)}_{\mathrm{fb}},
\]

才冻结候选；否则自动回退。对于有 compute 要求的候选，还需同时满足预注册的 mean 和
per-fold routed-compute non-inferiority 门禁。

selector 决策后重新计算 profile 文件 SHA256，并在读取 validation/test 前冻结 profile、
超参数、folds、token cache 和成功标准。该协议保证 evaluation 指标不参与 profile 搜索，
但它并不意味着有限 train folds 必然泛化到任意测试分布。

### 2.11 完整算法

给定模型、train corpus、目标结构预算 \(B\) 和多个不重叠 train folds，完整流程如下：

1. 对每个物理专家采集 typical RMS/Hessian、activation-tail、route count、AMP 和 AIMER；
2. 在每个专家内部生成固定通道排序，并划分为 64-channel prefix blocks；
3. 在 routed set 内构造 Conditional Dual teacher，蒸馏专家条件效用 \(U_{l,e}\)；
4. 将 \(U_{l,e}\) 重新绑定到 typical/tail 几何融合后的 block coverage；
5. 根据全局 activation-tail 阈值生成稀疏 Tail-Risk minimum-width floors；
6. 在精确结构预算下生成 Tail-Risk fallback，并根据需要校准到 train routed-compute anchor；
7. 在多个独立 estimator folds 上计算 block-level committee residual；
8. 只读取 fallback 的 first-pruned block，采用逐层归一化、minimum-fold 和 global q99.5
   生成 Frontier Committee Regret floors；
9. 在相同精确结构预算下重新分配其他 block，得到 nominal Frontier candidate；
10. 若要求计算稳健性，则加入 observed-fold non-inferiority 与 reference-centered route
    envelope 约束，由 multi-dual exact-prefix allocator 求解；不可行时拒绝候选；
11. 在不重叠 train-only selection/holdout folds 上比较 fallback 与 candidate；只有通过
    预注册 PPL/compute 门禁才冻结候选；
12. 记录 profile、所有 cache、选择结果和协议的 SHA256，然后只运行一次正式 evaluation。

### 2.12 静态执行与审计

最终 profile 是一个整数矩阵

\[
\mathbf W=[w_{l,e}]\in\mathbb Z^{L\times E}.
\]

推理时，router 仍输出原始 physical expert ID；系统根据 \((l,e)\) 查询固定宽度，只执行该
专家排序后的前缀通道。结构剪枝率与实际 routed-compute 剪枝率分开报告：

\[
\rho_{\mathrm{route}}
=1-\frac{\sum_{t,l,e\in\mathcal R_{l,t}}w_{l,e}}
{\sum_{t,l,e\in\mathcal R_{l,t}}K_l}.
\]

profile payload 至少记录：模型路径、训练语料与 split、token offset、sequence length、
channel block size、精确预算、width histogram、所有 floor、compute constraint margin、
reference profile、cache provenance、`test_metrics_used_for_profile=false` 以及 profile SHA256。
这种审计使“模型结构缩减”和“路由加权执行计算”不会被混为同一指标。

## 3. 当前方法应如何在论文中定位

### 3.1 可以作为主方法写入的部分

当前证据最充分的论文主干是：

1. physical-expert 64-channel prefix geometry 与 exact global block budget；
2. Routed-Set Conditional Dual Utility；
3. train-only Tail-Risk minimum-width safety constraints；
4. frozen fallback 上的 first-pruned Frontier Committee Regret；
5. cross-fitted train-only applicability selector 与 profile/hash audit；
6. 将结构预算和 routed compute 分开报告。

其中 Frontier Committee Regret 是当前正式 WikiText-2 PPL 最优的核心增量：在 Qwen3
Instruct 上由 \(8.688653\) 降至 \(8.658925\)，在独立 Qwen3 Base 上由
\(10.341598\) 降至 \(10.322812\)；在 Qwen3.5 的 tokenizer-specific secondary full-corpus
协议上也由 \(7.834395\) 降至 \(7.820017\)。三组结果均保持相同模型内的精确结构预算，
但均伴随很小的 routed-compute concession，因此应表述为跨 checkpoint/topology 的
lower-PPL replication，而不是 strict PPL-compute Pareto dominance。

### 3.2 应作为计算稳健扩展写入的部分

Reference-Centered Route Envelope 已经在 Qwen3 Instruct 和独立 Qwen3 Base 的全新
train holdout 上，相对 Tail-Risk fallback 同时获得 4/4 PPL wins 与 4/4 compute
non-inferiority。它适合写为 method extension、robust allocation 或 prospective analysis。

但是，目前没有对应的正式 validation/test 结果。尤其在 Base holdout 上，route-envelope
虽然比 Tail-Risk 更好，却比 nominal Frontier 的 mean PPL 高 \(0.007668\)。因此不能把它
描述为全面支配 nominal Frontier，也不能把 \(1/\sqrt F\) 扩张半径写成已证明的概率界。

### 3.3 当前不能作出的主张

基于现有证据，论文不应声称：

- 首次提出静态异构 expert width 或 64-channel block slimming；
- 方法在所有 MoE 架构上都优于简单 route/activation baseline；
- PPL 改善自动意味着 factual reliability 或 instruction-following 保持；
- Frontier Committee Regret 已实现零额外 routed compute 的严格 Pareto 提升；
- Route Envelope 提供 distribution-free 或 architecture-universal 保证；
- 任何 validation/test 指标参与了 profile、quantile、floor 或 envelope radius 的选择。

TruthfulQA 的混合负结果已经表明，语言建模失真降低不能外推为事实可靠性保持；Qwen1.5-MoE
的负结果也表明，完整 Conditional Dual + Tail-Risk recipe 需要 applicability selector，而不是
无条件迁移。

## 4. 建议用于论文正文的简短方法概述

本文研究无需恢复训练的 MoE 静态异构缩宽。我们将每个物理专家的中间通道排序并划分为
固定大小的连续 prefix blocks，以每个 \((\text{layer},\text{physical expert})\) 的整数宽度为
结构变量，在全局精确 block 预算下分配容量。常规容量由 routed-set 内归一化的 AMP/AIMER
条件双效用蒸馏得到，并重新绑定到融合典型激活和尾部激活的通道覆盖；稀有但灾难性的
专家路径则由全局 activation-tail 风险阈值施加最小宽度约束。进一步地，我们不使用整专家
输出幅值评价不可替代性，而只考察冻结参考 profile 的 first-pruned block：对该 block 的
加权输出能量，估计其相对于同 token 其他 routed experts 输出方向的正交残差，并通过多折
minimum 聚合形成 Frontier Committee Regret。高后悔度 frontier block 被恢复一个结构块，
同时从最低边际效用位置回收完全相同的预算。为控制结构相同但路由加权计算不同的问题，
我们还围绕冻结参考 profile 构造 route-probability lower/upper envelope，对新增 block 使用
上界成本、对删除 block 使用下界成本，并在多个 train folds 上施加 compute non-inferiority。
所有候选均通过不重叠 train-only folds 选择，在读取 evaluation split 前冻结 profile 与哈希。

## 5. 证据与推论边界

- **直接证据**：Tail-Risk 在 Qwen3 Instruct/Base 及 WikiText/C4 matched-domain 多预算实验中
  稳定改善相应均值型基线；Frontier Committee Regret 在 Qwen3 Instruct、Qwen3 Base 和
  Qwen3.5 三个 checkpoint/topology 上复现 lower-PPL 方向；Route Envelope 在 Instruct/Base
  的新 train holdout 上通过 PPL 与逐折 compute 门禁。
- **机制推论**：平均效用和稀有灾难风险是两个不同目标；整专家标量无法稳定预测
  within-expert frontier block demand；reference-centered 增量成本比单一平均 route anchor
  更适合表达静态 profile 的 compute robustness。
- **未知项**：更远的非 Qwen 架构、生成式 hallucination、instruction following、真实硬件
  latency、恢复训练后的排序，以及 Route Envelope 在正式 evaluation split 上能否保持优势。

## 6. 大白话解释：本文的方法到底解决了什么问题

这一章不再使用论文里的复杂符号，直接用一个“公司里分配员工和工作量”的例子解释本文。

### 6.1 先把 MoE 想成一家公司

一个 MoE 模型可以想成一家公司：

- 模型里有很多个专家，每个专家负责某一类事情；
- 每次来了一个 token，就像来了一个客户；
- 路由器决定这个客户应该交给哪几个专家处理；
- 一个专家内部又有很多个通道，可以理解成这个专家手下的很多员工；
- 专家的完整中间层很大，相当于一个专家手下有 768 个员工；
- 本文把员工按 64 人分成一个小组，也就是一个 64-channel block。

原始模型的问题是：每个专家虽然不一定每次都被调用，但只要被调用，通常就会把自己的
全部 768 个通道都算一遍。这样模型参数很大，实际计算也不够省。

本文想做的事情非常直观：

> 不把整个专家粗暴删除，而是调查每个专家内部哪些小组最值得保留，最后让不同专家
> 保留不同数量的小组，把“大专家”变成大小不同的“小专家”。

例如：

```text
原始状态：每个专家都有 12 个 block

专家 A：保留 12 个 block
专家 B：保留  8 个 block
专家 C：保留  4 个 block
专家 D：保留  1 个 block
专家 E：保留  0 个 block
```

但本文真正难的地方不在“切小”，而在于回答下面几个问题：

1. 哪个专家应该保留得更多？
2. 一个专家内部到底应该保留前面的哪些 block？
3. 有没有某些平时很少出现、但删掉就会造成严重错误的 block？
4. 同样保留这么多 block，是否因为路由变化而实际算得更多？
5. 如果某个新方法在当前数据上不稳定，能不能自动退回一个可靠的旧方案？

### 6.2 为什么不能只看“这个专家被调用了多少次”

最容易想到的办法是：统计每个专家被调用的次数，调用次数多的专家保留宽一点，调用次数
少的专家保留窄一点。

这个办法有一定道理，但不够好。

可以类比成公司里的员工：一个员工经常接电话，不代表他每次解决的问题都重要；另一个员工
可能平时很少接电话，但他专门处理重大事故，一旦没有他，整个系统就会出问题。

在 MoE 中也一样：

- 高频专家不一定是最有价值的专家；
- 低频专家不一定可以全部删掉；
- 一个专家的价值还取决于它和同一 token 上其他专家一起工作时，是否提供了别人没有的内容。

因此，本文不只统计“专家出现了多少次”，而是进一步问：

> 对于当前这个 token，已经被选中的几个专家放在一起时，这个专家到底贡献了多少独特信息？

这就是 Routed-Set Conditional Dual Utility 的直观含义。

### 6.3 Conditional Dual Utility：先看专家在当前小组里的真实作用

对于每个 token，路由器会选出几个专家。这几个专家组成一个临时小组。

本文对每个临时小组做两件事：

1. 看路由器给每个专家分了多少权重；
2. 看 AMP 和 AIMER 两种独立的专家先验是否都认为这个专家重要。

然后，不在全模型范围内直接比较专家，而是在当前临时小组内部重新归一化。这样得到的不是：

> 这个专家在整个数据集上有多大。

而是：

> 在这一次具体的专家小组里，这个专家相对于同组其他专家有多重要。

再把大量 token 上的这种局部判断累积起来，就得到每个物理专家大概应该获得多少容量。

大白话说，普通 route count 是“这个员工来上班多少天”，Conditional Dual Utility 更像是：

> 每次项目组开会时，这个员工在当前项目里到底承担了多少不可替代的工作，再把很多次项目
> 会议的结果汇总起来。

这一步主要解决的是“不能只按调用次数分配宽度”的问题。

### 6.4 为什么还要加 Tail-Risk：防止把“平时不常用但关键”的专家删光

只看平均贡献仍然有一个问题：平均数会掩盖极端情况。

举个极端例子：

- 专家 A 处理 99% 的普通请求，每次贡献一点点；
- 专家 B 只处理 1% 的特殊请求，但每次都承担关键任务。

如果只看平均贡献，专家 B 很可能排名靠后，甚至被分配零宽度。

这就像医院里急诊科：平时普通门诊人数可能更多，但不能因为急诊科平时人数少，就把急诊
科整个关掉。因为真正的严重事故一旦发生，急诊科是不可替代的。

本文因此增加了一条独立的“尾部风险保护”：

- 观察每个专家通道在训练数据上出现过的极端激活；
- 再结合这个通道经过 down projection 后可能放大的程度；
- 找出极少数风险特别高的 physical expert；
- 至少给这些专家保留一个很小的前缀宽度。

当前实现中，风险阈值大致采用全局 99.5% 分位数和全局最大值比例保护。直观上，这相当于：

> 不因为某个专家平时不常用，就允许它完全没有应急能力。

需要强调，Tail-Risk 不是偷偷增加参数。它给某些高风险专家加宽后，必须从其他低风险位置
回收同样数量的 block，所以总结构预算没有变化。

### 6.5 为什么普通“专家重要性”还不够：真正被删的是专家里面的一小段

假设一个专家有 12 个 block。我们最终可能决定只保留前 5 个 block，那么真正被删除的第一块
是第 6 个 block。

很多已有方法会给整个专家算一个总分，例如：

- 这个专家的总输出有多大；
- 这个专家被调用多少次；
- 这个专家和别人有多不一样。

这些分数可以告诉我们“这个专家整体上是否重要”，但不能精确回答：

> 这个专家当前最先被删除的那一个 block，到底是不是别人无法替代的？

这就像评估一个部门：部门整体很重要，不代表部门里每个岗位都同样重要；某个部门整体看起来
不大，也不代表其中没有一个关键岗位。

所以本文把判断粒度下沉到真正的剪枝边界：

1. 先用 Tail-Risk profile 得到一个参考宽度；
2. 对每个专家找到“参考宽度之后的第一个 block”；
3. 假设这个 block 被删掉；
4. 看同一个 token 上其他专家的输出能不能解释它；
5. 如果其他专家解释不了，就说明这个 block 具有较高的 committee regret；
6. 对少数最高 regret 的 frontier block 恢复一个 block；
7. 从别处回收同样数量的 block。

因此，Frontier Committee Regret 的核心不是“再给专家算一个总分”，而是：

> 只检查当前真正要删掉的那一小块，判断它是不是当前专家委员会中不可替代的部分。

这也是为什么它比整专家 output norm 更贴近实际剪枝决策。

### 6.6 “委员会后悔度”到底是什么意思

可以把一个 token 的多个 routed experts 想成一个委员会。

- 某个专家的某个 block 被保留下来时，委员会能看到它提供的内容；
- 如果这个 block 被删掉，而其他专家完全可以提供同样的方向，那么删除它的后悔度很低；
- 如果其他专家都无法补上它提供的方向，那么删除它就会让委员会损失独特信息，后悔度很高。

本文用一个近似的正交残差来衡量这种独特性：

- 如果一个 block 的输出方向和其他专家输出方向很接近，说明它比较冗余；
- 如果一个 block 的输出方向和其他专家明显不同，说明它可能包含独特功能；
- 输出越大、方向越独特，committee regret 越高。

这里的“后悔”不是模型真的在线试错，而是一个离线反事实问题：

> 如果现在把这个 block 删除，和它一起工作的其他专家能不能把损失补回来？

本文不直接为每个 block 生成完整输出矩阵，而是使用 down projection 的对角 Gram 近似，降低
校准开销。这是一个工程上可计算的近似指标，不是对真实损失的精确解析计算，所以论文中
应称其为 committee residual/regret estimator，而不能称为真实 PPL 的闭式等价物。

### 6.7 为什么必须把“模型结构变小”和“实际计算变少”分开

假设两个方案都保留了 40% 的结构 block，看起来结构剪枝率完全一样。

但是：

- 方案 A 可能把较多 block 留给经常被路由到的热门专家；
- 方案 B 可能把较多 block 留给很少被路由到的冷门专家。

这两个方案的参数量一样，但方案 A 每次推理可能执行更多 block。

所以本文始终分开报告两个指标：

- **结构剪枝率**：模型总共删掉了多少 block；
- **routed-compute 剪枝率**：真实路由到某个专家时，平均少执行了多少 block。

这一区分很重要，因为当前 Frontier Committee Regret 的正式 PPL 确实更低，但在某些测试
分布上会多执行极少量 routed block。正确的说法是：

> 它在相同结构预算下取得更低 PPL，但还不能说它在 PPL 和计算两个维度上都严格支配旧方案。

这也是本文继续研究 Route Envelope 的原因。

### 6.8 Route Envelope：提前防止“训练时省计算，换个数据分布又变费”

训练数据上观察到的路由分布，不一定等于部署数据上的路由分布。

例如，训练期间某个专家平均只被调用 5%，但新领域数据中它可能突然被调用 8%。如果 profile
刚好给这个专家增加了很多 block，那么结构预算虽然没变，实际 routed compute 就会增加。

Route Envelope 的直观做法是：

1. 收集多个互不重叠的 train route folds；
2. 对每个 physical expert 观察它在这些 folds 中的最低和最高路由概率；
3. 给这个范围增加一个预先固定的安全余量；
4. 对参考 profile 原本已经保留的 block，按较低路由成本估计删除收益；
5. 对参考 profile 之外新增加的 block，按较高路由成本估计增加代价；
6. 只有在这个保守估计下仍不比参考方案更费计算，才接受候选 profile。

可以把它理解成给每个专家设置一个“路由天气预报范围”：

- 原来就保留的 block 被删掉时，按较保守的低频情况计算；
- 新增加的 block 则按较保守的高频情况计算；
- 只要在这两个方向都没有突破计算预算，profile 才能通过。

目前 Route Envelope 在 Qwen3 Instruct 和 Qwen3 Base 的全新 train holdout 上都通过了 PPL
和逐折 routed-compute 门禁，说明它是一个有实际证据支持的计算稳健扩展。但它还没有正式
validation/test 揭榜结果，因此现阶段更准确的定位是：

> 它是用来增强 Frontier profile 计算稳健性的候选分配器，而不是已经证明全面优于 Frontier
> 的最终方案。

### 6.9 整个方法用一句大白话概括

如果把 MoE 专家想成很多个部门，本文的方法就是：

> 先根据每次项目中各部门的真实分工，决定每个部门大概需要多少人；再防止那些平时不忙但
> 出事时关键的部门被裁光；接着只检查每个部门当前最先准备裁掉的那个小组，判断它是不是
> 其他部门无法替代；最后考虑换一批客户后各部门忙闲变化，保证总裁员数不变、预计工作量
> 也不要突然增加；如果复杂方案在训练数据上不稳定，就自动退回一个更简单可靠的方案。

对应到模型就是：

```text
Conditional Dual Utility
    = 判断常规情况下每个专家应该分到多少容量

Tail-Risk Floor
    = 防止罕见但关键的专家路径被完全删掉

Frontier Committee Regret
    = 判断当前第一个要删的 block 是否真的不可替代

Exact Prefix Budget
    = 总共只能保留固定数量的 block，新增多少就必须从别处回收多少

Route Envelope
    = 防止换数据分布后，结构虽然一样但实际 routed compute 变高

Cross-Fitted Selector
    = 复杂方案没有在独立训练折上稳定赢，就自动回退，不拿 test 结果调参
```

### 6.10 这套方法最终解决的不是“剪得最多”，而是“剪得有依据”

本文不是单纯追求把专家缩得越小越好，而是试图同时回答三个部署问题：

1. 哪些容量是平均任务真正需要的？
2. 哪些低频容量虽然不常用，却不能被置零？
3. 在路由分布发生变化时，怎样避免静态 profile 带来不可控的额外计算？

因此，本文的核心价值可以用一句话概括：

> 把“大专家变小专家”从一个粗粒度的参数删减问题，变成一个以物理专家、专家内部
> prefix block、routed committee、稀有风险和路由分布不确定性为对象的可审计容量分配问题。

当前最有证据支持的结论是：Frontier Committee Regret 能在相同精确结构预算下稳定改善
Qwen3 Instruct、Qwen3 Base 以及 Qwen3.5 secondary evaluation 的 PPL；Route Envelope
则进一步展示了如何在尚未读取正式 evaluation split 的情况下，对新 train route distribution
进行更保守的 routed-compute 约束。前者是当前正式 PPL 主方法，后者是计算稳健扩展；两者
都不能被夸大为对所有架构、所有任务和所有部署分布都成立的普适定理。
