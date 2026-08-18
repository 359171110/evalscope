# 当前 Method 详细说明：面向 MoE 的尾部风险约束与剪枝边界委员会后悔度

> 本文档独立解释当前仓库已经收敛的方法，不按实验发生时间复述研究过程。
> 当前正式 PPL 主方法是 **Frontier Committee Regret**；
> **Reference-Centered Route Envelope** 是已经通过跨 checkpoint、全新 train-only holdout
> 验证的计算稳健扩展，但尚未完成正式 validation/test 评价。

## 1. 方法要解决什么问题

本文研究的是无需恢复训练的静态 MoE 专家缩宽。给定一个已经训练完成的 MoE 模型，我们不
改变 router，不删除整个 MoE 层，也不在推理时动态决定宽度，而是在部署前为每个物理专家
冻结一个固定的中间维度。

设第 \(l\) 个 MoE 层的第 \(e\) 个物理专家原本有 \(d_{\mathrm{ff}}\) 个中间通道。我们先根据
train-only calibration 得到一个固定通道排序，再把排序后的通道划分为相同大小的连续 block。
当前 Qwen3-30B 实现使用 64-channel block，一个 768-channel 专家因此包含 12 个 block。

最终，每个物理专家只需要一个整数宽度：

\[
w_{l,e}\in\{0,1,\ldots,K_l\},
\]

其中 \(K_l\) 是该层每个专家的最大 block 数。\(w_{l,e}=5\) 表示只执行该专家排序后的前
5 个 block；\(w_{l,e}=0\) 表示这个物理专家仍可被原 router 选中，但其 routed expert 分支
不保留中间通道。

方法需要同时回答四个问题：

1. 哪些专家应当获得更多常规容量？
2. 哪些专家虽然低频，却绝不能被压缩到零宽度？
3. 当前最先准备删除的 block 是否能被同一 routed committee 中的其他专家替代？
4. 在结构预算相同的情况下，候选 profile 是否会因为路由分布变化而执行更多通道？

本文分别使用 Conditional Dual Utility、Tail-Risk、Frontier Committee Regret 和
Reference-Centered Route Envelope 回答这四个问题。

## 2. 整体方法概览

当前方法不是单一 importance score，而是一个 reference-centered 的分层流程：

```text
train-only 通道统计与专家先验
        |
        v
Routed-Set Conditional Dual Utility
        |
        v
Typical/Tail-Aware Prefix Coverage
        |
        v
Tail-Risk Minimum-Width Floors
        |
        v
Exact-Budget Tail-Risk Reference Profile
        |
        v
First-Pruned Frontier Committee Regret
        |
        v
Exact-Budget Nominal Frontier Candidate
        |
        +--> 可选：Reference-Centered Route Envelope
        |             |
        |             v
        |      Compute-Robust Frontier Candidate
        |
        v
Disjoint Train-Only Cross-Fitted Selection
        |
        v
冻结 profile、cache、选择决策与 SHA256
        |
        v
一次正式 evaluation
```

其中，Tail-Risk profile 是冻结参考方案。Frontier Committee Regret 不重新推翻整个容量
分配，而只在参考方案的实际剪枝边界上增加少量结构约束。Route Envelope 进一步改变容量
回收位置，使候选相对参考方案的 routed-compute 增量在预定义路由不确定集合中受到控制。

## 3. 静态物理专家前缀结构

### 3.1 为什么使用物理专家索引

所有 profile 都以

\[
(l,e)=(\text{MoE layer},\text{physical expert ID})
\]

为索引。这里的 \(e\) 不是某个 token 上 top-k router 输出中的局部 rank，而是 checkpoint
中的真实专家编号。

这个约束非常重要。router rank 会随 token 改变，不能用于定义静态结构；只有 physical
expert ID 才能对应到固定的 up/gate/down projection 参数和最终可物化的专家宽度。

### 3.2 Prefix block 决策变量

对专家 \((l,e)\) 的第 \(j\) 个 block，定义选择变量

\[
x_{l,e,j}(\mathbf w)=\mathbb I[j<w_{l,e}].
\]

于是天然满足 prefix 单调约束：

\[
x_{l,e,j+1}\le x_{l,e,j}.
\]

这表示不允许保留第 7 个 block 却删除第 6 个 block。每个专家最终只执行一个连续前缀，
便于静态切片、编译、kernel 对齐和结构审计。

### 3.3 精确结构预算

给定总 block 预算 \(B\)，所有候选必须满足

\[
\sum_{l,e}w_{l,e}=B.
\]

结构剪枝率定义为

\[
\rho_{\mathrm{struct}}
=1-\frac{B}{\sum_l E_lK_l}.
\]

任何安全 floor 或 frontier block 恢复都不能增加 \(B\)。某处增加一个 block，allocator 必须
从其他边际价值较低的位置回收一个 block。因此，方法比较首先是 exact matched structural
budget，而不是通过偷偷扩大模型换取更低 PPL。

## 4. 专家内部的 Typical/Tail-Aware 通道排序

### 4.1 典型激活覆盖

对专家中间通道 \(c\)，典型覆盖统计使用 train-only 激活尺度与 down projection 列权重。
可写成

\[
a^{\mathrm{typ}}_{l,e,c}
\approx
\sqrt{\mathbb E_t[z_{l,e,t,c}^2]}
\left\lVert W^{\mathrm{down}}_{l,e,:,c}\right\rVert_2,
\]

其中 \(z_{l,e,t,c}\) 是 SwiGLU 等专家非线性之后的中间激活。

该统计描述通道在常规 token 上的平均输出覆盖，但均方量可能忽略极少出现的大幅激活。

### 4.2 尾部激活覆盖

尾部覆盖使用最大绝对激活：

\[
a^{\mathrm{tail}}_{l,e,c}
=
\max_t |z_{l,e,t,c}|
\left\lVert W^{\mathrm{down}}_{l,e,:,c}\right\rVert_2.
\]

它近似衡量某个稀有通道激活经过 down projection 放大后，可能对专家输出造成的极端影响。

### 4.3 Typical 与 tail 的几何融合

最终通道排序分数为

\[
s_{l,e,c}
=\left(a^{\mathrm{typ}}_{l,e,c}+\epsilon\right)^{1-\lambda}
 \left(a^{\mathrm{tail}}_{l,e,c}+\epsilon\right)^\lambda.
\]

当前冻结配置使用 \(\lambda=0.5\)。这个选择保留典型激活和极端激活之间的平衡：

- \(\lambda=0\) 只看平均覆盖，容易漏掉 rare channels；
- \(\lambda=1\) 只看最大值，容易过度追逐偶发极值；
- \(\lambda=0.5\) 使用几何平均，使两类证据都必须具有一定强度。

每个物理专家内部按 \(s_{l,e,c}\) 降序排列通道，再划分为固定大小的 block。记第 \(j\) 个
block 的覆盖分数为 \(c_{l,e,j}\)，并整理为非增边际：

\[
c_{l,e,0}\ge c_{l,e,1}\ge\cdots\ge c_{l,e,K_l-1}.
\]

这一步只决定“一个专家内部先保留哪些通道”，尚未决定不同专家之间各自获得多少容量。

## 5. Routed-Set Conditional Dual Utility

### 5.1 为什么 route frequency 不够

route frequency 只回答一个专家出现了多少次，却没有回答它在当时的 routed committee 中
相对其他专家有多重要。静态地把 route count 与专家先验相乘，也会丢失 token 条件竞争关系。

本文因此在每个 token 的实际 routed set 内计算相对效用，再将这种动态相对效用聚合为静态
physical-expert capacity。

### 5.2 Routed-set 内归一化

对第 \(l\) 层 token \(t\)，设 router 选择的专家集合为 \(\mathcal R_{l,t}\)，专家 \(e\) 的
路由权重为 \(g_{l,t,e}\)。给定两个冻结的 expert priors：

- \(A^{\mathrm{AMP}}_{l,e}\)
- \(A^{\mathrm{AIMER}}_{l,e}\)

分别在当前 routed set 内计算

\[
q^{\mathrm{AMP}}_{l,t,e}
=\frac{g_{l,t,e}A^{\mathrm{AMP}}_{l,e}}
{\sum_{e'\in\mathcal R_{l,t}}
g_{l,t,e'}A^{\mathrm{AMP}}_{l,e'}+\epsilon},
\]

\[
q^{\mathrm{AIMER}}_{l,t,e}
=\frac{g_{l,t,e}A^{\mathrm{AIMER}}_{l,e}}
{\sum_{e'\in\mathcal R_{l,t}}
g_{l,t,e'}A^{\mathrm{AIMER}}_{l,e'}+\epsilon}.
\]

然后取几何融合：

\[
u_{l,t,e}
=\sqrt{q^{\mathrm{AMP}}_{l,t,e}
q^{\mathrm{AIMER}}_{l,t,e}+\epsilon}.
\]

这个量表达的不是“专家 \(e\) 在全局有多重要”，而是“在当前 token 已经选中的专家中，
两个先验是否共同支持专家 \(e\) 获得较高相对容量”。

### 5.3 从动态效用蒸馏为静态专家容量

将 token 条件效用按物理专家聚合，得到 expert-level utility \(U_{l,e}\)。再把它与新的
typical/tail block coverage 重新绑定：

\[
v^{\mathrm{base}}_{l,e,j}
=U_{l,e}\left(c^{\mathrm{tail}}_{l,e,j}+\epsilon\right).
\]

这样完成两个层级的分工：

- \(U_{l,e}\) 决定不同专家之间的常规容量份额；
- \(c_{l,e,j}\) 决定某个专家内部的 block 保留顺序。

这种 factorization 避免把 expert-level routing semantics 与 channel-level tail ranking 混成
一个不可解释的全局标量。

## 6. Tail-Risk Safety Floor

### 6.1 平均效用的 zero-width 失效

即使 Conditional Dual Utility 在平均意义上有效，也可能把低频专家分配为零宽度。如果该
专家只在极少数 token 上承担关键功能，平均效用会系统性低估它，但一旦删除便可能出现 PPL
capacity cliff。

因此，Tail-Risk 不再尝试微调所有专家的连续分数，而是只对极少数高风险专家设置最低宽度。

### 6.2 Expert-level 风险代理

对专家 \((l,e)\)，定义

\[
r_{l,e}
=\max_c\left(
A^{\max}_{l,e,c}
\left\lVert W^{\mathrm{down}}_{l,e,:,c}\right\rVert_2
\right),
\]

其中 \(A^{\max}_{l,e,c}\) 是 train calibration 上该通道的最大绝对激活。

全局风险阈值为

\[
\tau_{\mathrm{risk}}
=\max\left(
Q_{0.995}(\{r_{l,e}\}),
0.1\max_{l,e}r_{l,e}
\right).
\]

最终 safety floor 为

\[
m^{\mathrm{risk}}_{l,e}
=
\begin{cases}
2,&r_{l,e}\ge\tau_{\mathrm{risk}},\\
0,&\text{otherwise}.
\end{cases}
\]

当前正式方法对所有 MoE 层使用统一 selector，不编码已知 Super Expert ID，也不使用人工
layer whitelist。

### 6.3 Tail-Risk 的作用边界

Tail-Risk 解决的是“不能让哪些专家变成 zero/tiny width”，不是重新定义全部常规容量。
被保护的专家仍然计入总预算，且 allocator 从其他低边际 block 回收完全相同的数量。

因此它更接近一个稀疏 safety constraint，而不是另一套全局 importance ranking。

## 7. Tail-Risk Reference Profile

在基础 block values \(v^{\mathrm{base}}_{l,e,j}\) 和风险 floors
\(m^{\mathrm{risk}}_{l,e}\) 下，求解

\[
\max_{\mathbf w}
\sum_{l,e}\sum_{j=0}^{w_{l,e}-1}v^{\mathrm{base}}_{l,e,j}
\]

满足

\[
\sum_{l,e}w_{l,e}=B,
\qquad
m^{\mathrm{risk}}_{l,e}\le w_{l,e}\le K_l.
\]

由于所有 block 的结构成本相同，并且每个专家内部的边际价值非增，可以使用全局 next-block
greedy：每次从所有专家的下一个合法 block 中选择价值最大的一个，直到达到精确预算。

实现上使用稳定全局排序得到等价解。稳定 tie-breaking 保证同一专家中更早的等值 block
优先被选择，从而保持 prefix 可行性。

得到的宽度记为

\[
\mathbf w^0.
\]

它是后续 Frontier Committee Regret 和 Route Envelope 的冻结 reference profile。

## 8. Frontier Committee Regret

### 8.1 为什么只评价 first-pruned block

对参考 profile \(\mathbf w^0\)，专家 \((l,e)\) 已保留的 blocks 是

\[
j=0,\ldots,w^0_{l,e}-1.
\]

其第一个被删除的 block 是

\[
j^*_{l,e}=w^0_{l,e}.
\]

如果 \(w^0_{l,e}=K_l\)，该专家已经 full width，不存在 frontier block。

FCR 不给整个专家重新计算一个总分，也不对所有被删除 block 做无限制全局重排。它只问：

> 在当前参考 profile 下，真正处于保留/删除边界上的第一个 block，是否提供了同一 token 上
> 其他 routed experts 无法替代的输出方向？

这一设计使 importance estimation 的粒度与最终结构决策的粒度一致。

### 8.2 Routed committee

对 token \(t\) 上被路由的专家 \(e\)，令其 gate-weighted 输出为

\[
\mathbf y_{t,e}.
\]

同一 token 上其他 routed experts 的委员会输出为

\[
\mathbf y_{t,-e}
=\sum_{e'\in\mathcal R_{l,t}\setminus\{e\}}
\mathbf y_{t,e'}.
\]

归一化后记为

\[
\widehat{\mathbf y}_{t,-e}
=\frac{\mathbf y_{t,-e}}
{\lVert\mathbf y_{t,-e}\rVert_2+\epsilon}.
\]

如果某个 frontier block 的输出方向已经与委员会输出高度一致，那么恢复它的边际价值可能较
低；如果它主要位于委员会输出方向的正交补空间，则删除它更可能损失独特功能。

### 8.3 Diagonal-Gram 通道残差

令 \(h_{t,e,c}\) 为专家非线性后的中间通道激活，\(\mathbf d_{l,e,c}\) 为 down projection
的第 \(c\) 列。定义通道残差能量

\[
\delta_{t,l,e,c}
=|g_{l,t,e}|^2h_{t,e,c}^2
\left(
\lVert\mathbf d_{l,e,c}\rVert_2^2
-\langle
\widehat{\mathbf y}_{t,-e},
\mathbf d_{l,e,c}
\rangle^2
\right)_+.
\]

括号中的部分表示 down-projection 列在委员会主方向正交补中的能量：

- 如果 \(\mathbf d_{l,e,c}\) 与委员会方向平行，残差接近 0；
- 如果两者正交，则保留完整列范数能量；
- gate 和中间激活越大，该通道在当前 token 上的残差越大。

对 frontier block \(\mathcal B_{l,e,j^*}\) 聚合：

\[
R_{t,l,e,j^*}
=\sqrt{
\sum_{c\in\mathcal B_{l,e,j^*}}
\delta_{t,l,e,c}
}.
\]

该估计只保留 down-projection Gram 矩阵的对角项，并把其他专家委员会压缩成一个输出方向。
它显著降低校准代价，但不是完整 block 输出相对委员会子空间的精确正交投影。

### 8.4 多折稳健聚合

在多个互不重叠的 train-only estimator folds 上独立收集 frontier residual。为了避免不同层的
输出尺度直接支配全局排序，每个 fold 内先进行逐层 mean-one 归一化。

记第 \(f\) 折的归一化 frontier score 为

\[
\bar R^{(f)}_{l,e,j^*}.
\]

跨折使用 minimum 聚合：

\[
S^{\mathrm{FCR}}_{l,e}
=\min_f\bar R^{(f)}_{l,e,w^0_{l,e}}.
\]

minimum-fold 的含义是：只有某个 frontier block 在所有 estimator folds 上都保持较高的相对
不可替代性，才允许它形成硬结构约束。单折瞬时异常不能直接恢复 block。

### 8.5 Frontier floor

对所有 eligible frontier scores 取全局 99.5% 分位数。高分专家的最低宽度设置为 reference
width 加一个 block：

\[
m^{\mathrm{FCR}}_{l,e}
=
\begin{cases}
\min(w^0_{l,e}+1,K_l),
&S^{\mathrm{FCR}}_{l,e}\ge Q_{0.995}(S^{\mathrm{FCR}}),\\
0,&\text{otherwise}.
\end{cases}
\]

最终 floor 为

\[
m_{l,e}
=\max\left(
m^{\mathrm{risk}}_{l,e},
m^{\mathrm{FCR}}_{l,e}
\right).
\]

重新运行 exact-budget allocator 后，高 FCR frontier blocks 被恢复，而相同数量的低边际 blocks
从其他位置回收。所得 profile 是 nominal Frontier Committee Regret candidate。

### 8.6 FCR 与其他 expert score 的区别

FCR 不等同于以下指标：

- **route frequency**：只衡量出现次数；
- **output norm**：只衡量输出幅值；
- **co-routing uniqueness**：衡量整专家路由上下文差异；
- **activation tail**：衡量稀有极端激活风险；
- **完整 block ablation**：直接重跑删除反事实，但计算成本更高。

FCR 近似衡量的是：

> 当前第一个被剪 block 的输出能量中，有多少不能被同 token 的其他 routed experts 主方向
> 解释。

它只负责 reference frontier 上的局部结构修正，不替代 Conditional Dual 和 Tail-Risk 的
基础容量分配。

## 9. Routed Compute 与结构预算的区别

即使两个 profile 保留完全相同的 block 总数，它们的实际执行计算也可能不同。令
\(p_{l,e}\) 为某个数据分布下专家 \((l,e)\) 被路由到的概率，则 routed cost 近似为

\[
C_p(\mathbf w)
=\sum_{l,e}p_{l,e}w_{l,e}.
\]

把更多 block 分给热门专家，会提高 \(C_p\)；把容量分给冷门专家，则可能在结构参数量不变
时降低平均执行计算。

因此本文分开报告：

\[
\rho_{\mathrm{struct}}
=1-\frac{\sum_{l,e}w_{l,e}}{\sum_lE_lK_l},
\]

以及

\[
\rho_{\mathrm{route}}
=1-\frac{\sum_{t,l,e\in\mathcal R_{l,t}}w_{l,e}}
{\sum_{t,l,e\in\mathcal R_{l,t}}K_l}.
\]

前者描述静态结构缩减，后者描述给定语料路由下的执行 block 缩减。两者不能互相替代。

## 10. Train-Only Compute Calibration

### 10.1 单一 expected-compute anchor

给定 train route counts \(n_{l,e}\)，可以在 block utility 中加入专家级计算惩罚：

\[
\widetilde v_{l,e,j}(\mu)
=\frac{v_{l,e,j}}{s_v}
-\mu\frac{n_{l,e}}{s_n}.
\]

对固定 \(\mu\)，内层仍是 exact prefix allocation。通过一维搜索 \(\mu\)，可以在保持精确
结构预算的同时，使候选接近预冻结的 train expected routed-compute anchor。

### 10.2 多场景 non-inferiority constraints

若有多个 route folds，可要求候选在每个观测场景都不比 reference 执行更多 blocks：

\[
C_f(\mathbf w)\le C_f(\mathbf w^0),
\qquad f=1,\ldots,F.
\]

实现使用 projected multi-dual search。对于给定乘子 \(\boldsymbol\mu\)，调整后的 block
边际为

\[
\widetilde v_{l,e,j}(\boldsymbol\mu)
=\frac{v_{l,e,j}}{s_v}
-\sum_f\mu_f\frac{p^{(f)}_{l,e}}{s_c},
\qquad \mu_f\ge0.
\]

每次 dual 更新的内层仍调用 exact prefix allocator。如果无法同时满足结构预算、floors 和
全部 compute constraints，builder 必须输出不可行证书并拒绝候选，而不是静默放宽条件。

有限场景约束只能保证已观察 route folds。仓库实验已经证明，即使 8/8 train scenarios 可行，
也不能自动推出新 holdout 上逐折 compute non-inferiority。因此进一步引入 reference-centered
route envelope。

## 11. Reference-Centered Route Envelope

### 11.1 为什么以 reference 为中心

普通 route uncertainty 若直接给每个专家一个最坏成本，会对保留和删除两个方向一视同仁，
容易过度保守。本文真正关心的是候选相对 reference profile 的增量成本：

- 候选新增的 block 可能在新分布上变得更热门，应使用较高 route cost；
- 候选删除的 reference block 可能在新分布上变得更冷，应使用较低 route cost估计删除收益。

这形成一个围绕 \(\mathbf w^0\) 的 sign-split cost。

### 11.2 路由坐标包络

由 \(F\) 个 train route folds 得到归一化路由概率 \(p^{(f)}_{l,e}\)。定义

\[
p^-_{l,e}=\min_f p^{(f)}_{l,e},
\qquad
p^+_{l,e}=\max_f p^{(f)}_{l,e},
\]

\[
\Delta_{l,e}=p^+_{l,e}-p^-_{l,e}.
\]

给定预注册扩张系数 \(\alpha\)，构造

\[
\underline p_{l,e}
=\max(0,p^-_{l,e}-\alpha\Delta_{l,e}),
\]

\[
\overline p_{l,e}
=p^+_{l,e}+\alpha\Delta_{l,e}.
\]

当前已验证 train-only 配置使用

\[
\alpha=\frac{1}{\sqrt 8}.
\]

该半径是预冻结的经验扩张，不应解释为 distribution-free 置信界。

### 11.3 Reference-centered block cost

对专家 \((l,e)\) 的 block \(j\)，定义

\[
q_{l,e,j}
=
\begin{cases}
\underline p_{l,e},&j<w^0_{l,e},\\
\overline p_{l,e},&j\ge w^0_{l,e}.
\end{cases}
\]

也就是说：

- reference 已经保留的部分使用 lower route cost；
- candidate 在 reference 之外恢复的 frontier 部分使用 upper route cost。

要求候选满足

\[
\sum_{l,e,j}q_{l,e,j}
\left(
x_{l,e,j}(\mathbf w)-x_{l,e,j}(\mathbf w^0)
\right)
\le0.
\]

对于任意满足

\[
p_{l,e}\in[\underline p_{l,e},\overline p_{l,e}]
\]

的路由分布，候选相对 reference 的真实 retained-cost 增量都有上界

\[
C_p(\mathbf w)-C_p(\mathbf w^0)
\le
\sum_{l,e}
\left[
\overline p_{l,e}(w_{l,e}-w^0_{l,e})_+
-\underline p_{l,e}(w^0_{l,e}-w_{l,e})_+
\right].
\]

因此，只要 envelope 约束右侧不大于零，并且未来路由分布确实位于该坐标盒中，候选就不会
比 reference 执行更多 routed blocks。

### 11.4 Envelope 的正确定位

Route Envelope 是 FCR 的 compute-robust allocation extension，而不是替代 FCR score。

- FCR 决定哪些 frontier blocks 值得恢复；
- Route Envelope 决定为了恢复它们，应从哪些位置回收容量，才能对 route shift 更保守。

当前它已经在 Qwen3 Instruct 和 Qwen3 Base 的全新 train-only holdout 上，相对 Tail-Risk
fallback 同时取得 4/4 PPL wins 和 4/4 compute non-inferiority。但尚无正式 evaluation split
结果，因此当前论文中应将其描述为已验证的计算稳健扩展，而不是最终正式最优 profile。

## 12. Cross-Fitted Applicability Selector

复杂 refinement 不一定适合所有 MoE 架构。仓库中的 Qwen1.5-MoE 结果已经表明，完整
Conditional Dual + Tail-Risk 在某些 routing semantics 下可能不如更简单的 Route×Tail。

因此，当前方法不按模型名硬编码分支，而是在与 calibration/estimator 不重叠的 train-only
selection folds 上比较 fallback 和 candidate。

设共有 \(H\) 个 selection folds。候选只有同时满足以下条件才被接受：

\[
\#\left\{
h:\mathrm{PPL}^{(h)}_{\mathrm{cand}}
<\mathrm{PPL}^{(h)}_{\mathrm{fb}}
\right\}
\ge\left\lceil\frac{H+1}{2}\right\rceil,
\]

以及

\[
\frac1H\sum_h\mathrm{PPL}^{(h)}_{\mathrm{cand}}
<
\frac1H\sum_h\mathrm{PPL}^{(h)}_{\mathrm{fb}}.
\]

如果候选涉及 compute claim，还必须通过预注册的 mean 或 per-fold routed-compute gate。

如果候选不稳定、平票、均值不优或 compute 条件不满足，系统自动回退到 frozen fallback，
而不是打开 test 后再调 quantile、floor、fold 或 multiplier。

selector 输出至少记录：

- fallback/candidate profile path；
- 实际重新计算的 profile SHA256；
- 每个 selection token cache 的 SHA256；
- 每折 PPL 与 routed-compute；
- calibration、estimator 和 selection 区间；
- 最终接受或回退原因；
- `test_metrics_used_for_profile=false`。

## 13. 完整算法步骤

给定模型、train calibration corpus、结构预算和多个互不重叠的 train folds，当前完整流程为：

1. 读取模型的物理 MoE 拓扑、专家数、top-k、专家中间维度和 shared-expert 语义。
2. 在 train calibration 上收集 RMS/Hessian、最大激活、route count 和专家先验。
3. 对每个物理专家生成 typical/tail 几何融合后的固定通道排序。
4. 将排序后的通道划分为固定大小的连续 prefix blocks。
5. 在每个 token 的实际 routed set 内计算 Conditional Dual Utility。
6. 将动态条件效用聚合为 physical-expert utility，并重新绑定到 tail-aware block coverage。
7. 根据全局 activation-tail 规则生成 sparse minimum-width risk floors。
8. 在 exact block budget 下生成 Tail-Risk reference profile \(\mathbf w^0\)。
9. 在独立 estimator folds 上只计算 \(\mathbf w^0\) 的 first-pruned block committee residual。
10. 逐层归一化、跨折 minimum 聚合，并以 global q99.5 生成 FCR floors。
11. 在相同结构预算下重新分配容量，得到 nominal Frontier candidate。
12. 如果要求 compute robustness，则加入 observed-fold constraints 和 route-envelope constraint，
    生成 compute-robust Frontier candidate；不可行时拒绝该候选。
13. 在不重叠 train-only selection/holdout folds 上比较 candidate 与 fallback。
14. 只有通过预注册 PPL/compute gate 才冻结 candidate，否则自动回退。
15. 重新计算并记录所有 profile、cache、选择决策和语料文件的 SHA256。
16. 冻结成功标准后，只运行一次正式 validation/test evaluation。

## 14. 推理时如何执行

最终 profile 是整数矩阵

\[
\mathbf W=[w_{l,e}]\in\mathbb Z^{L\times E}.
\]

推理过程中：

1. 原始 router 正常计算 top-k physical expert IDs 和 gate weights；
2. 对每个 routed physical expert 查询固定宽度 \(w_{l,e}\)；
3. 只读取该专家排序后前 \(w_{l,e}\) 个 blocks 对应的 gate/up/down 参数；
4. 计算专家前缀输出并按原 gate weights 合并；
5. 不运行在线 width controller，也不根据当前 token 改写 profile。

当某一层所有专家均为 full width 时，当前实现直接走冻结的 native fast path，以避免通用静态
路径和 fused native kernel 因 BF16 累加顺序不同产生跨层数值漂移。

## 15. 方法中各组件的职责

| 组件 | 解决的问题 | 不负责什么 |
|---|---|---|
| Route×Tail fallback | 提供跨架构较稳健的低假设基础分配 | 不建模 routed committee 条件作用 |
| Conditional Dual Utility | 分配常规 token mass 下的 expert-level 容量 | 不保护极低频灾难路径 |
| Typical/Tail Coverage | 决定专家内部通道和 block 顺序 | 不决定专家间总容量 |
| Tail-Risk Floor | 防止 rare critical experts 被压成 zero/tiny width | 不重新排序所有 expert utility |
| FCR | 恢复当前 first-pruned、委员会不可替代的 block | 不对所有被剪 block 做全局重搜索 |
| Exact Prefix Allocator | 保证前缀可行和精确结构预算 | 不保证 proxy 等于真实 PPL 损失 |
| Compute Calibration | 匹配 train expected routed cost | 不自动覆盖 unseen route shift |
| Route Envelope | 控制候选相对 reference 的保守增量成本 | 不提供 distribution-free 覆盖保证 |
| Cross-Fitted Selector | test-free 决定是否启用 refinement | 不保证有限 train folds 代表所有部署域 |

## 16. 当前正式方法与扩展的关系

为了避免论文表述混乱，当前应区分三层 profile：

### 16.1 稳健 fallback

Route×Tail 是相对跨架构稳健的基础分配。在复杂 refinement 没有通过 train-only selector 时，
系统回退到该 profile。

### 16.2 Tail-Risk reference

在适用 Conditional Dual 的模型上，以 Conditional Dual + typical/tail coverage + sparse
risk floors 生成 Tail-Risk reference。它是 FCR 的冻结起点，也定义了“第一个被剪 block”。

### 16.3 Frontier candidates

- **Nominal FCR**：只根据 frontier floors 和 expected-compute calibration 重分配容量；当前
  是正式 WikiText PPL 最优方法。
- **Route-Envelope FCR**：保留同一 frontier signal，但加入 reference-centered route
  uncertainty constraint；当前是 train-only holdout 上已验证的计算稳健扩展。

## 17. 当前理论性质与局限

### 17.1 可以明确说明的性质

1. 在 equal block cost、每专家边际非增、合法 prefix floors 下，next-block greedy 对内部
   exact structural allocation 问题是最优的。
2. FCR 的通道残差在 down-projection 列与委员会方向平行时为零，在正交时保留完整加权列
   能量，具有清楚的方向冗余响应。
3. Route Envelope 给出了候选相对 reference 的 coordinate-box 增量成本上界。
4. 所有 floors 都在同一结构预算内重新分配，不增加总 block 数。

### 17.2 不能过度声称的性质

1. Conditional Dual、Tail-Risk 或 FCR score 不等于真实语言建模损失。
2. Diagonal-Gram FCR 不是完整委员会子空间的精确反事实投影。
3. Minimum-fold 和 q99.5 是经验稳健选择，不是统计最优性定理。
4. \(1/\sqrt F\) envelope expansion 没有 distribution-free 覆盖概率保证。
5. 相同结构预算不等于相同实际 latency、FLOPs 或 routed compute。
6. 更低 PPL 不意味着 factual reliability、instruction following 或 hallucination safety 保持。

## 18. 一句话解释当前方法

当前方法可以概括为：

> 先用 token 条件效用决定每个物理专家通常需要多少容量，用激活尾部风险防止稀有关键专家
> 被删空，再只检查参考方案中第一个准备删除的 block 是否包含同一 routed committee 无法
> 替代的输出方向；恢复高后悔度 block 时保持精确结构预算，并通过参考中心路由包络约束其
> 在路由漂移下的增量计算，最后由不重叠 train-only folds 决定接受候选还是自动回退。

## 19. 当前最稳妥的论文主张

基于仓库现有证据，当前最稳妥的主张是：

1. 静态 MoE 缩宽需要区分常规条件效用、稀有灾难风险和剪枝边界不可替代性。
2. 整专家 scalar contribution 难以稳定预测实际 first-pruned block 的结构需求。
3. FCR 将 committee non-redundancy 下沉到 frozen reference 的真实 block frontier，在相同
   精确结构预算下跨 Qwen3 Instruct、Qwen3 Base 和 Qwen3.5 topology 复现 lower-PPL 方向。
4. Reference-Centered Route Envelope 能在两个独立 Qwen3 checkpoints 的全新 train-only
   holdout 上保留相对 Tail-Risk 的 PPL 收益，并实现逐折 routed-compute non-inferiority。
5. 方法不是 architecture-universal recipe；复杂 refinement 必须经过 test-free cross-fitted
   applicability selection，不稳定时回退到较简单的 Route×Tail。

当前不能声称 FCR 已经在正式 evaluation 上实现严格 PPL-compute Pareto，也不能声称该方法
保护事实可靠性或对所有 MoE 架构普适。