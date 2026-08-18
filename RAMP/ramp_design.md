# Reconstructability-Aware MoE Channel Pruning：方法流程与可行性分析

## 一、方法的核心目标

给定一个预训练 MoE 模型和全局通道剪枝率 $\rho$，我们希望同时解决四个问题：

1. 每个专家最终保留多少个中间通道；
2. 每个专家具体保留哪些通道；
3. 哪些通道可以被安全删除；
4. 删除通道后，如何利用保留通道补偿专家输出。

与传统 channel pruning 不同，我们不单纯删除“贡献最小”的通道，而是优先删除：

$$
\boxed{
\text{输出冗余、能够由保留通道稳定重构的通道}
}
$$

最终模型中的专家数量、Router 和 Top-$k$ 路由机制保持不变，但不同专家可以具有不同的固定中间宽度。

---

## 二、专家输出分解

对于第 $l$ 层第 $e$ 个 SwiGLU 专家，定义其中间激活：

$$
a_{l,e}(x)
=
\operatorname{SiLU}
\left(
W_{g,l,e}x
\right)
\odot
W_{u,l,e}x,
$$

其中：

$$
a_{l,e}(x)\in\mathbb R^{d_{\mathrm{ffn}}}.
$$

专家输出为：

$$
E_{l,e}(x)
=
W_{d,l,e}a_{l,e}(x).
$$

将中间通道划分为保留集合 $\mathcal K_{l,e}$ 和剪枝集合 $\mathcal P_{l,e}$：

$$
\mathcal K_{l,e}\cup\mathcal P_{l,e}
=
\{1,\ldots,d_{\mathrm{ffn}}\},
$$

$$
\mathcal K_{l,e}\cap\mathcal P_{l,e}
=
\varnothing.
$$

则专家输出可以写为：

$$
E_{l,e}(x)
=
W_{d,l,e}^{\mathcal K}
a_{l,e}^{\mathcal K}(x)
+
W_{d,l,e}^{\mathcal P}
a_{l,e}^{\mathcal P}(x).
$$

第一项是保留通道的输出，第二项是被剪通道原本产生的输出：

$$
R_{l,e}(x)
=
W_{d,l,e}^{\mathcal P}
a_{l,e}^{\mathcal P}(x).
$$

如果直接剪枝，压缩专家只能保留：

$$
E_{l,e}^{\mathrm{pruned}}(x)
=
W_{d,l,e}^{\mathcal K}
a_{l,e}^{\mathcal K}(x),
$$

损失正是 $R_{l,e}(x)$。

---

## 三、利用保留通道补偿被删通道

我们的假设是：被删通道的输出残差虽然不能直接保留，但可以由保留通道的激活近似预测：

$$
R_{l,e}(x)
\approx
\Delta W_{l,e}
a_{l,e}^{\mathcal K}(x).
$$

因此，压缩专家的输出写成：

$$
\widehat E_{l,e}(x)
=
W_{d,l,e}^{\mathcal K}
a_{l,e}^{\mathcal K}(x)
+
\Delta W_{l,e}
a_{l,e}^{\mathcal K}(x).
$$

合并后：

$$
\widehat E_{l,e}(x)
=
\left(
W_{d,l,e}^{\mathcal K}
+
\Delta W_{l,e}
\right)
a_{l,e}^{\mathcal K}(x).
$$

定义补偿后的输出投影：

$$
\widetilde W_{d,l,e}
=
W_{d,l,e}^{\mathcal K}
+
\Delta W_{l,e},
$$

则最终专家仍然是普通的缩窄 SwiGLU：

$$
\widehat E_{l,e}(x)
=
\widetilde W_{d,l,e}
\left[
\operatorname{SiLU}
\left(
W_{g,l,e}^{\mathcal K}x
\right)
\odot
W_{u,l,e}^{\mathcal K}x
\right].
$$

这意味着补偿不会引入新的运行时模块。它最终被直接融合进保留通道对应的 `down_proj`。

所以部署后的模型只包含：

$$
W_{g,l,e}^{\mathcal K},
\qquad
W_{u,l,e}^{\mathcal K},
\qquad
\widetilde W_{d,l,e}.
$$

被剪通道的参数被物理删除。

---

## 四、校准统计的收集

使用校准集运行原始 MoE 模型。

对于每个专家，只收集真正路由到该专家的 token。对每个 token $x$，记录：

$$
a_{l,e}(x),
\qquad
E_{l,e}(x),
\qquad
g_{l,e}(x),
$$

其中 $g_{l,e}(x)$ 是 Router 分配给专家 $e$ 的 gate 权重。

这样，每个专家拥有独立的数据集合：

$$
\mathcal X_{l,e}
=
\left\{
x:
e\in\operatorname{TopK}_l(x)
\right\}.
$$

校准数据最好划分为两部分：

- calibration split：选择通道并求补偿矩阵；
- validation split：选择正则系数、补偿秩，并判断是否过拟合。

如果只在同一批 token 上选择通道并计算补偿，重构误差很容易被低估。

---

## 五、怎样评价一组通道是否适合被剪

假设某个专家最终保留集合为 $\mathcal K$。

在校准 token 上构造保留通道激活矩阵：

$$
A_{\mathcal K}
=
\begin{bmatrix}
a^{\mathcal K}(x_1)&
a^{\mathcal K}(x_2)&
\cdots&
a^{\mathcal K}(x_N)
\end{bmatrix}.
$$

原专家输出矩阵为：

$$
Y
=
\begin{bmatrix}
E(x_1)&
E(x_2)&
\cdots&
E(x_N)
\end{bmatrix}.
$$

保留通道未经补偿的输出为：

$$
Y_{\mathcal K}
=
W_d^{\mathcal K}A_{\mathcal K}.
$$

需要补偿的目标残差为：

$$
R
=
Y-Y_{\mathcal K}.
$$

给定保留集合 $\mathcal K$，补偿矩阵通过下面的回归求解：

$$
\Delta W^*
=
\arg\min_{\Delta W}
\left\|
\left(
R-\Delta WA_{\mathcal K}
\right)D_g
\right\|_F^2
+
\lambda\|\Delta W\|_F^2,
$$

其中：

$$
D_g
=
\operatorname{diag}
\left(
g(x_1),\ldots,g(x_N)
\right).
$$

Router gate 被加入目标，是因为专家输出最终进入 MoE 层时还会乘：

$$
g_e(x).
$$

对应的验证重构误差可以定义为：

$$
\mathcal E(\mathcal K)
=
\frac{
\sum_x
g_e(x)^2
\left\|
E_e(x)-\widehat E_e(x)
\right\|_2^2
}{
\sum_x
g_e(x)^2
\left\|
E_e(x)
\right\|_2^2
}.
$$

$\mathcal E(\mathcal K)$ 越低，说明该保留集合越好；等价地，被删除的通道越容易由剩余通道补偿。

---

## 六、决定保留哪些 channel

这是整个方法最难的部分，因为 channel 不能只独立排序。

假设两个通道功能高度相似：

$$
C_i(x)\approx C_j(x).
$$

它们单独看可能都很重要，但保留一个就可能重构另一个。因此，传统的独立 importance score 无法正确判断这种冗余关系。

建议采用“重要性保护 + 可重构性选择”的两阶段方案。

### 6.1 先保护不可轻易删除的重要通道

定义单通道输出贡献：

$$
C_j(x)
=
w_{d,j}a_j(x).
$$

基础重要性可以写成：

$$
s_j
=
\mathbb E_x
\left[
g_e(x)^2
\|C_j(x)\|_2^2
\right].
$$

分数特别高的通道先作为 anchor channels 保留。

这一阶段不是最终剪枝依据，而是避免搜索算法为了降低平均重构误差，错误删除少量具有强独立功能的通道。

### 6.2 根据通道输出轨迹识别冗余

对每个通道构造其在校准 token 上的输出轨迹：

$$
T_j
=
\begin{bmatrix}
C_j(x_1)&
C_j(x_2)&
\cdots&
C_j(x_N)
\end{bmatrix}.
$$

两个通道的功能相似度可以定义为：

$$
\operatorname{sim}(i,j)
=
\frac{
\langle T_i,T_j\rangle_F
}{
\|T_i\|_F\|T_j\|_F
}.
$$

如果两个通道在不同 token 上总是产生方向和幅度相似的输出，那么它们具有较强冗余，可以只保留其中更重要或更具代表性的一个。

实际中不能完整保存所有 $T_j$，可以使用：

- token 子采样；
- 随机投影；
- 分块统计；
- 低维 sketch；
- 对激活和 `down_proj` 分别建立近似相似度。

### 6.3 贪心选择保留集合

初始化保留集合 $\mathcal K$ 为 anchor channels，再逐步添加能够最大程度降低重构误差的通道：

$$
j^*
=
\arg\max_{j\notin\mathcal K}
\left[
\mathcal E(\mathcal K)
-
\mathcal E(\mathcal K\cup\{j\})
\right].
$$

直到达到候选宽度 $k$。

也可以反过来，从完整集合开始，每次删除一个经补偿后误差增加最小的通道：

$$
j^*
=
\arg\min_{j\in\mathcal K}
\mathcal E(\mathcal K\setminus\{j\}).
$$

对于上万个通道，逐通道重新求解回归会非常昂贵，因此实践中更适合：

1. 先按冗余聚成 channel groups；
2. 每次删除一个通道组；
3. 使用增量矩阵更新近似评估误差；
4. 最后做少量 channel swap 微调集合。

---

## 七、为每个专家建立误差—宽度曲线

对于每个专家，不直接决定一个最终宽度，而是先测试若干候选宽度。

例如：

$$
k_{l,e}
\in
\{
0.25d_{\mathrm{ffn}},
0.50d_{\mathrm{ffn}},
0.75d_{\mathrm{ffn}},
d_{\mathrm{ffn}}
\}.
$$

或者使用更硬件友好的档位：

$$
k_{l,e}
\in
\{
1024,2048,3072,4096
\}.
$$

对于每个候选宽度 $k$：

1. 选择大小为 $k$ 的保留集合；
2. 求对应的补偿矩阵；
3. 在独立 validation tokens 上计算重构误差。

最终得到每个专家的误差—宽度曲线：

$$
k
\longrightarrow
\mathcal E_{l,e}(k).
$$

例如：

| 专家 | 25% 宽度 | 50% 宽度 | 75% 宽度 | 100% 宽度 |
|---|---:|---:|---:|---:|
| Expert 1 | 0.35 | 0.08 | 0.02 | 0 |
| Expert 2 | 0.12 | 0.03 | 0.01 | 0 |
| Expert 3 | 0.62 | 0.25 | 0.06 | 0 |

这说明：

- Expert 2 冗余较强，可以保留较小宽度；
- Expert 3 难以压缩，需要保留较多通道；
- Expert 1 处于中间状态。

---

## 八、在全局剪枝率下决定每个专家最终保留多宽

设所有 routed experts 的原始总通道数为：

$$
C_{\mathrm{total}}
=
\sum_{l,e}d_{l,e}.
$$

目标剪枝率为 $\rho$，则总保留预算为：

$$
B
=
(1-\rho)C_{\mathrm{total}}.
$$

全局宽度分配问题为：

$$
\min_{\{k_{l,e}\}}
\sum_{l,e}
\omega_{l,e}
\mathcal E_{l,e}(k_{l,e})
$$

满足：

$$
\sum_{l,e}k_{l,e}
\leq B.
$$

其中 $\omega_{l,e}$ 可以考虑：

- 专家访问频率；
- 平均 Router gate；
- 专家所在层的敏感度；
- 数学、代码和通用领域中的最坏误差；
- 专家对最终任务损失的影响。

### 边际收益分配

可以从所有专家的最小宽度开始，每次增加一个宽度档位。

对专家 $e$，从宽度 $k$ 增加到 $k+\Delta k$ 的边际收益为：

$$
G_{l,e}(k)
=
\frac{
\mathcal E_{l,e}(k)
-
\mathcal E_{l,e}(k+\Delta k)
}{
\Delta k
}.
$$

每次把额外通道分配给 $G_{l,e}(k)$ 最大的专家，直到耗尽全局预算。

最终自然得到不同专家的异构宽度：

```text
Layer 0:
Expert 0: 2048
Expert 1: 3072
Expert 2: 1024
Expert 3: 4096

Layer 1:
Expert 0: 3072
Expert 1: 2048
Expert 2: 2048
Expert 3: 4096
```

这些宽度在模型导出后完全固定。

---

## 九、最终补偿和模型导出

在最终宽度和 channel 集合确定后，使用完整 calibration split 为每个专家重新拟合一次：

$$
\Delta W_{l,e}^*.
$$

构造：

$$
\widetilde W_{d,l,e}
=
W_{d,l,e}^{\mathcal K}
+
\Delta W_{l,e}^*.
$$

然后物理删除：

$$
W_{g,l,e}^{\mathcal P},
\qquad
W_{u,l,e}^{\mathcal P},
\qquad
W_{d,l,e}^{\mathcal P}.
$$

最终保存：

$$
W_{g,l,e}^{\mathcal K},
\qquad
W_{u,l,e}^{\mathcal K},
\qquad
\widetilde W_{d,l,e}.
$$

Router 不变，专家数量不变，Top-$k$ 不变。

最终压缩的是：

$$
\boxed{
\text{每个激活专家内部真正执行的 FFN 宽度}
}
$$

因此可以同时降低：

- checkpoint 大小；
- 专家权重显存；
- 每 token 的专家 FFN FLOPs；
- 理论推理延迟。

---

## 十、完整流程总结

给定剪枝率 $\rho$，整个方法可以概括为：

### 1. 收集校准统计

记录每个专家的中间激活、专家输出和 Router gate。

### 2. 识别重要和冗余通道

保护高贡献 anchor channels，分析其他通道的功能冗余。

### 3. 在每个候选宽度下选择保留集合

选择使“补偿后专家输出误差”最小的通道集合。

### 4. 拟合补偿矩阵

利用保留通道激活重构被删通道的输出残差。

### 5. 建立专家误差—宽度曲线

评估每个专家缩到不同宽度后的可重构性。

### 6. 全局分配宽度预算

在全模型目标剪枝率下，为不同 layer–expert 分配不同宽度。

### 7. 重新拟合最终补偿

在最终 channel 集合上计算补偿后的 `down_proj`。

### 8. 物理导出异构小专家

删除被剪参数，保存固定宽度的压缩 checkpoint。

---

## 十一、可行性判断

### 11.1 数学上是可行的

本质上，我们在做：

$$
\text{非线性特征子集选择}
+
\text{条件线性输出重构}.
$$

虽然 SwiGLU 前半部分是非线性的，但在中间激活 $a(x)$ 已经计算出来之后，`down_proj` 是线性的。

因此，只要保留通道的激活包含足够的信息，就可以通过重新拟合 `down_proj` 近似完整专家输出。

这类似于从一组过完备非线性特征中选择一个子集，然后重新训练线性读出层。

从模型结构角度，这个假设是合理的，因为大规模 FFN 中间维度通常明显过参数化，通道之间可能存在：

- 激活相关性；
- 输出方向相关性；
- 功能重复；
- 低维输出子空间；
- 可被其他通道线性组合近似的冗余。

### 11.2 中等剪枝率下成功概率较高

我认为在以下范围内成功概率较高：

$$
25\%\text{–}50\%
$$

的 routed-expert channel pruning。

原因是：

- 仍然保留大量原始非线性特征；
- `down_proj` 有足够自由度重新组合这些特征；
- 被删通道更有可能是冗余部分；
- 补偿不需要恢复所有中间激活，只需恢复最终专家输出。

如果直接达到 70%–80% 通道剪枝，保留特征空间可能不足，补偿很可能失效。

### 11.3 相比普通 channel pruning 有合理优势

普通 channel pruning 的目标一般是：

$$
\text{删除独立贡献最小的通道}.
$$

我们的目标是：

$$
\text{删除经剩余通道重构后误差最小的通道集合}.
$$

两者区别类似于：

- importance：一个特征本身重要不重要；
- reconstructability：即使重要，它是否可以被其他特征替代。

因此，对于高度冗余但单独贡献较大的 channel，本方法应该更合理。

---

## 十二、主要风险

### 12.1 补偿可能只在校准数据上有效

最大风险是：

$$
\mathcal E_{\mathrm{calib}}
\ll
\mathcal E_{\mathrm{validation}}.
$$

尤其当保留通道很多、$\Delta W$ 参数自由度很高时，补偿可以直接拟合 calibration tokens。

需要至少使用：

$$
\lambda\|\Delta W\|_F^2
$$

并限制：

$$
\frac{
\|\Delta W\|_F
}{
\|W_d^{\mathcal K}\|_F
}
\leq\eta.
$$

还可以尝试：

- rank-$r$ 补偿；
- truncated SVD；
- ridge regression；
- calibration/validation 分离；
- 多领域验证；
- 提前停止增加补偿复杂度。

### 12.2 满秩补偿可能掩盖通道选择质量

如果允许任意满秩 $\Delta W$，那么即使保留集合不好，也可能在校准集上被强行拟合。

这样论文贡献可能被质疑为：

> 只是剪完 channel 后重新训练了一个新的 `down_proj`。

因此建议主方法限制补偿复杂度，例如：

$$
\operatorname{rank}(\Delta W)\leq r,
\qquad
r\in\{1,4,8,16\}.
$$

或者限制修改能量：

$$
\|\Delta W\|_F^2
\leq \tau.
$$

需要通过消融证明：

- 好的 channel 选择本身有效；
- 补偿只是进一步恢复，而不是完全掩盖错误选择。

### 12.3 保留通道不一定能预测被删通道

被删通道可能产生独立的非线性特征：

$$
a_j(x)
\not\approx
f(a_{\mathcal K}(x)).
$$

由于补偿只允许重新组合保留激活，如果某种非线性响应被彻底删除，`down_proj` 无法重新创造它。

需要测量：

$$
R^2_{\mathrm{pred}}
=
1-
\frac{
\|R-\Delta W^*A_{\mathcal K}\|_F^2
}{
\|R-\overline R\|_F^2
}.
$$

如果很多专家的 $R^2_{\mathrm{pred}}$ 很低，说明核心假设不成立。

### 12.4 Channel 集合搜索成本可能很高

每个专家有上万通道。对每个候选删除操作重新计算一次补偿基本不可行。

需要设计高效近似：

- 先聚类再选代表通道；
- 使用协方差矩阵而不是保存全部激活；
- 使用 Woodbury 更新回归逆矩阵；
- 只测试有限宽度档位；
- 先用 cheap score 缩小候选集合；
- 分块处理 channel；
- 使用随机投影估计重构误差。

否则压缩成本可能远高于 TENP、REAP 等基线。

### 12.5 低频专家统计不稳定

有些专家在校准集中只接收很少 token。

如果：

$$
N_{l,e}\ll k_{l,e},
$$

则回归问题严重欠定，补偿矩阵不可信。

可以采用：

- 低频专家保持较大最小宽度；
- 对低频专家使用统一宽度；
- 提高正则系数；
- 增加校准数据；
- 主动采样能路由到低频专家的 token；
- 跨相似专家共享协方差先验；
- 对专家按有效样本量设置补偿 rank。

### 12.6 局部输出重构不等于最终任务性能

即使：

$$
E_{l,e}(x)
\approx
\widehat E_{l,e}(x),
$$

误差也可能逐层累积，并改变后续 Router 选择。

因此需要同时测量：

$$
\text{Top-k Routing Overlap},
$$

$$
D_{\mathrm{KL}}
\left(
p_{\mathrm{router}}^{\mathrm{original}}
\parallel
p_{\mathrm{router}}^{\mathrm{compressed}}
\right),
$$

以及最终：

- PPL；
- 数学能力；
- 代码能力；
- 通用任务；
- 长文本生成；
- 不同校准领域外的泛化。

专家级 reconstruction error 只能作为优化代理，不能作为最终结论。

---

## 十三、RAMP-E0 结果与方法修正

RAMP-E0 在 24 个代表性专家、固定保留 $384/768$ 通道的条件下验证了原始方案。正式 audit 中，RAMP 明显优于随机选择，但相对 RMS/Tail 的 median error improvement 只有约 $1.0\%$；扩展 ridge 网格后，median residual $R^2$ 约为 $0.104$，RAMP 相对最佳 RMS/Tail 的提升仍不足 $0.5\%$。

这些结果只说明原始方法没有通过专家级局部机制门槛，不能直接推出压缩模型的 PPL 或下游性能一定较差，因为 RAMP-E0 没有构造完整压缩模型，也没有测量 logits、PPL 和任务准确率。

更重要的是，RAMP-E0 暴露了原始方法中的目标错位：

1. 低 importance 只表示通道的边际输出能量较小，不表示它可以由保留通道预测；
2. 被删通道可能是独立的非线性基函数，删除后无法由固定线性读出重新创造；
3. pairwise correlation 不能描述通道在整个保留集合条件下的剩余信息；
4. 多个被删通道可能存在联合抵消，只独立评价单通道会破坏这种结构；
5. 一个固定补偿矩阵可能无法同时覆盖不同领域、路由强度和激活模式；
6. 满秩补偿在低频专家上容易过拟合，选择集合、ridge 强度和补偿 rank 必须联合验证。

因此，修正版 RAMP 不再把“重要性排序”和“补偿拟合”视为两个相对独立的步骤，而是让通道选择直接优化最终补偿所面对的条件输出残差。

## 十四、RAMP v2：补偿对齐的通道选择

### 14.1 Ridge 一致的条件残差

对某个专家省略下标，令 gate 加权激活二阶矩为：

$$
C
=
\sum_n g_n^2 a_n a_n^{\top}.
$$

给定保留集合 $\mathcal K$ 和删除集合 $\mathcal P$，固定集合下的 ridge 补偿为：

$$
\Delta W_{\mathcal K}^*
=
W_{\mathcal P}
C_{\mathcal P\mathcal K}
\left(
C_{\mathcal K\mathcal K}+\lambda I
\right)^{-1}.
$$

删除激活在保留激活条件下的 ridge 条件协方差为：

$$
C_{\mathcal P\mid\mathcal K}^{(\lambda)}
=
C_{\mathcal P\mathcal P}
-
C_{\mathcal P\mathcal K}
\left(
C_{\mathcal K\mathcal K}+\lambda I
\right)^{-1}
C_{\mathcal K\mathcal P}.
$$

真正需要最小化的不是原始激活残差，而是经过 `down_proj` 输出几何加权后的条件残差：

$$
J_{\mathrm{recon}}(\mathcal K;\lambda)
=
\operatorname{tr}
\left(
W_{\mathcal P}
C_{\mathcal P\mid\mathcal K}^{(\lambda)}
W_{\mathcal P}^{\top}
\right).
$$

该目标和最终 ridge 补偿严格对齐。它同时考虑：

- 被删通道能否由整个保留集合条件预测；
- 被删激活误差经过 `down_proj` 后是否真正影响专家输出；
- 被删通道之间的联合协方差和抵消关系；
- Router gate 对实际 MoE 层输出的权重。

### 14.2 补偿复杂度与稳定性正则

只最小化 fit residual 仍可能选择一个需要高秩、大范数或病态补偿的集合。RAMP v2 的集合分数定义为：

$$
J(\mathcal K)
=
J_{\mathrm{recon}}^{\mathrm{fit}}(\mathcal K;\lambda)
+
\beta_{\mathrm{norm}}
\frac{\|\Delta W_{\mathcal K}^*\|_F^2}
{\|W_{\mathcal K}\|_F^2+\epsilon}
+
\beta_{\mathrm{rank}}
r_{\mathrm{eff}}(\Delta W_{\mathcal K}^*)
+
\beta_{\mathrm{gap}}
G_{\mathrm{val}}(\mathcal K),
$$

其中 $G_{\mathrm{val}}$ 描述 fit 与 validation 的重构误差差距。第一版实现不需要同时启用全部正则项，应通过消融逐步回答：输出加权、条件化和稳定性约束分别带来多少收益。

补偿 rank、ridge 系数和集合选择正则只能使用 fit/validation 数据确定；独立 audit 只允许评估一次。

### 14.3 集合搜索

精确求解固定大小的最优 $\mathcal K$ 是组合优化问题。实际采用以下近似流程：

1. 从空集合或很小的强保护集合开始；
2. 用 Schur complement 或 Cholesky 增量更新估计加入候选通道后的 $\Delta J$；
3. 按组加入或删除高度耦合的通道，避免破坏联合抵消结构；
4. 达到目标宽度后执行少量 swap refinement；
5. 在 validation 上比较候选集合，而不是只比较同一集合的补偿超参数。

anchor 不再作为默认正确的先验。`anchor=0`、小比例 anchor 和原 10% anchor 必须作为显式消融；只有 validation 稳定支持时才保留 anchor 机制。

### 14.4 低频专家和多激活模式

当 routed token 数与保留宽度接近时，专家级 covariance 和满秩回归不稳定。RAMP v2 根据有效样本量采用：

- 更强 covariance shrinkage；
- 更低补偿 rank；
- 更大的最小保留宽度；
- 主动补充会路由到低频专家的 calibration token；
- 必要时在相似专家之间共享 covariance 先验，但不共享未经验证的最终补偿权重。

若单个固定 $\Delta W$ 在分域或分激活模式上表现明显分裂，可将轻量 mode-aware compensation 作为后续消融，例如使用 Router 已有信号选择少量低秩补偿分支。该扩展会增加运行时复杂度，不属于 RAMP v2 的首个主实验，只有固定补偿版本证明条件选择有效后才进入验证。

### 14.5 分层验证与结论边界

RAMP v2 使用三层证据链：

1. 专家级机制：条件残差、归一化输出误差、residual $R^2$ 和补偿稳定性；
2. 模型级代理：原模型与压缩模型的 hidden-state drift、Router overlap 和 logits KL；
3. 最终质量与部署：held-out PPL、下游任务、checkpoint 大小、显存、吞吐和延迟。

专家级指标用于筛选方法和定位失败原因，但不能替代模型级结论。只有局部收益稳定地传递到 logits/PPL，并且最终任务质量和部署收益达到预注册门槛，才能支持 RAMP 的整体可行性主张。

新版方法的首轮实验设计见 [compensation_aligned_experiment.md](compensation_aligned_experiment.md)。

### 12.7 平均目标可能删除 specialist channels

低频数学或代码 channel 对平均 token 的贡献可能较小，但对特定任务关键。

因此不建议只使用：

$$
\mathbb E_{x\sim\mathcal D_{\mathrm{mixed}}}
\mathcal L(x).
$$

可以分别计算：

$$
\mathcal E_{\mathrm{general}},
\qquad
\mathcal E_{\mathrm{math}},
\qquad
\mathcal E_{\mathrm{code}},
$$

然后使用：

$$
\mathcal E
=
\sum_d\alpha_d\mathcal E_d
+
\gamma\max_d\mathcal E_d.
$$

这样避免某个领域被平均目标牺牲。

### 12.8 异构宽度的真实部署加速不确定

理论上，专家宽度下降可以降低 FLOPs。

但如果每个专家宽度都不同，会导致：

- grouped GEMM 形状不规则；
- padding 浪费；
- kernel 数量增加；
- expert batching 效率下降；
- 负载不平衡；
- tensor parallel 切分困难。

所以最终宽度不能任意变化，最好限制为少量 bucket：

$$
k_{l,e}
\in
\{1024,2048,3072,4096\}.
$$

或者至少对齐到：

$$
128,\ 256,\ 512
$$

的倍数。

还应该同时提供：

- 等宽版本；
- 少量宽度 bucket 版本；
- 完全异构版本。

这样才能区分算法收益和工程不规则性。

### 12.9 与 FlexMoE 的方法重合风险

FlexMoE 已经研究了逐专家异构宽度分配，因此仅仅提出“每个专家保留不同宽度”不够新。

必须强调的差异是：

$$
\boxed{
\text{宽度分配依据是补偿后的可重构性，而不是学习离散 action}
}
$$

以及：

$$
\boxed{
\text{通道选择、宽度分配和补偿由同一重构目标统一决定}
}
$$

理想情况下还应具备：

- training-free；
- 不更新原模型权重；
- 只使用前向校准统计；
- 无 LoRA recovery；
- 补偿直接融合；
- 更低压缩成本；
- 更好的部署兼容性。

---

## 十三、建议的最小可行版本

第一版不要同时实现所有复杂内容。

建议先固定所有专家等宽，只验证“可重构性通道选择 + 融合补偿”是否成立。

### 第一阶段：固定等宽

所有专家统一保留：

$$
50\%
$$

通道。

比较：

1. 随机 channel pruning；
2. magnitude/activation importance；
3. TENP-style channel score；
4. 可重构性 channel selection，不补偿；
5. 可重构性 channel selection，rank-1 补偿；
6. rank-4 补偿；
7. ridge full-rank 补偿。

先验证最核心的两个假设：

- 可重构性选择是否优于普通 importance；
- 补偿是否能在 validation 和任务指标上持续有效。

### 第二阶段：异构宽度分配

如果第一阶段有效，再为每个专家建立：

$$
\mathcal E_e(k)
$$

并进行全局预算分配。

否则一开始加入异构宽度，会很难判断最终收益来自：

- channel selection；
- width allocation；
- compensation；
- 更多搜索成本。

---

## 十四、最终判断

我认为这套方法有较高研究价值，原因是它形成了一个统一问题：

$$
\boxed{
\text{在固定全局预算下，联合优化通道子集、专家宽度和输出补偿}
}
$$

并且最终补偿可以融合进 `down_proj`，不增加运行时分支。

但其成功与否主要取决于三个实证条件：

$$
\boxed{
\text{专家内部是否存在足够强的通道冗余}
}
$$

$$
\boxed{
\text{保留通道是否能跨数据分布预测被删通道输出}
}
$$

$$
\boxed{
\text{异构宽度能否在真实 MoE kernel 中转化为加速}
}
$$

最可能的结果是：在 25%–50% 剪枝率下，相比无补偿 channel pruning 获得稳定收益；在极高剪枝率下，由于独立非线性特征被删除，补偿能力会迅速饱和。

---

# 十五、RAMP-v2：与补偿目标对齐的通道选择

RAMP-E0 暴露出一个需要修正的核心问题：原始 channel selection 虽然使用了输出感知的条件残差 gain，但最终选择目标、补偿 rank、正则强度和多领域泛化并没有被统一优化。特别是，低重要性不等于可预测；一个被删通道可能是独立的非线性基函数，也可能只有与其他通道联合时才表现出重要的抵消关系。

因此 RAMP-v2 不再把“channel importance”和“补偿”视为前后独立的两个步骤，而是直接选择一个保留集合，使被删部分的 gate-weighted 输出在给定保留激活后具有最小、稳定且复杂度可控的条件残差。

## 15.1 目标定义

对一个 expert，令 $\mathcal K$ 为保留集合，$\mathcal P$ 为被删集合，$W$ 为完整 `down_proj`，$C^{(d)}$ 为数据模式 $d$ 上的 gate-weighted activation covariance。补偿后的保留投影为：

$$
\widetilde W_{\mathcal K}=W_{:\!,\mathcal K}+\Delta W_{\mathcal K}.
$$

给定 $\mathcal K$ 和正则 $\lambda$，补偿通过 fit split 求解：

$$
\Delta W_{\mathcal K}^{*}
=
W_{:\!,\mathcal P}C_{\mathcal P\mathcal K}^{(\mathrm{fit})}
\left(C_{\mathcal K\mathcal K}^{(\mathrm{fit})}+\lambda I\right)^{-1}.
$$

选择目标使用补偿后的 gate-weighted 输出残差，而不是单独的 activation importance：

$$
\mathcal L_{\mathrm{res}}(\mathcal K)
=
\sum_{d\in\mathcal D}\alpha_d
\operatorname{Tr}\left(
Q_{\mathcal K}^{*}C^{(d)}Q_{\mathcal K}^{*\top}
\right),
$$

其中 $Q_{\mathcal K}^{*}$ 是将保留列误差和被删列原始权重合并后的输出误差矩阵。为避免某个领域被平均目标掩盖，主目标使用平均误差和最坏领域误差的组合：

$$
\mathcal L_{\mathrm{select}}(\mathcal K)
=
(1-\gamma)\mathcal L_{\mathrm{mean}}(\mathcal K)
+\gamma\max_{d\in\mathcal D}\mathcal L_d(\mathcal K)
+\beta\Omega(\Delta W_{\mathcal K}^{*}).
$$

推荐的补偿复杂度项为：

$$
\Omega(\Delta W)
=
\frac{\|\Delta W\|_F^2}{\|W_{:\!,\mathcal K}\|_F^2+\epsilon}
+\xi\frac{\operatorname{rank}_{\mathrm{eff}}(\Delta W)}{|\mathcal K|}.
$$

其中 $\operatorname{rank}_{\mathrm{eff}}$ 由补偿矩阵在保留 covariance metric 下的有效奇异值确定。这样选择算法不会通过一个巨大、病态的 full-rank $\Delta W$ 掩盖错误的保留集合。

## 15.2 条件残差而非简单相关性

RAMP-v2 需要区分三个概念：

1. **激活相关性**：$a_j$ 是否能由 $a_{\mathcal K}$ 预测；
2. **输出重要性**：被删 channel 的 $W_{:,j}a_j$ 是否真的影响 expert 输出；
3. **补偿稳定性**：同一 $\Delta W$ 是否能跨 fit、validation、audit 和数据领域工作。

对被删集合 $\mathcal P$，条件 covariance 为：

$$
C_{\mathcal P\mid\mathcal K}
=
C_{\mathcal P\mathcal P}
-
C_{\mathcal P\mathcal K}
\left(C_{\mathcal K\mathcal K}+\lambda I\right)^{-1}
C_{\mathcal K\mathcal P}.
$$

RAMP-v2 的 channel gain 必须近似衡量加入候选 channel 后，**加权输出条件残差下降了多少**：

$$
G_j(\mathcal K)
=
\frac{
\mathcal L(\mathcal K)-\mathcal L(\mathcal K\cup\{j\})
}{\operatorname{cost}(j)}.
$$

只按 RMS、Tail 或 $C_{ij}$ 排序的方案不能作为 RAMP-v2 的主方法，因为它们没有直接包含 $W$、被删输出目标和补偿复杂度。

## 15.3 多模式统计和稳定选择

单一 mixed covariance 可能把不同激活模式平均掉。RAMP-v2 至少按 `wikitext`、`code`、`gsm8k`、`math` 保存：

$$
C_e^{(d,s)}=
\sum_{n\in(d,s)}g_{n,e}^2a_{n,e}a_{n,e}^{\top}.
$$

通道集合只能由 fit split 决定；validation 用于选择 $\lambda$、补偿 rank 和 $\beta/\gamma$，audit 只做一次最终评估。候选集合必须满足：

- mixed validation 残差下降；
- 每个重要领域的 validation 残差不出现明显恶化；
- fit 到 validation 的 gap 在预设范围内；
- 补偿范数和有效 rank 在预算内。

对于模式依赖明显的 expert，允许使用少量补偿基：

$$
\widehat R(x)=
\sum_{q=1}^{Q}m_q(x)\Delta W_q a_{\mathcal K}(x),
$$

但 `Q` 必须很小，并优先测试不增加 runtime 分支的方案。第一版仍以单一 $\Delta W$ 为主，multi-mode compensator 只作为诊断或消融。

## 15.4 新版算法流程

1. 对每个 expert 按数据领域和 split 收集 covariance、输出能量、routed-token 数及必要的 activation tail 统计。
2. 先保护少量 anchor，但 anchor 只能作为约束，不能替代条件输出残差目标。
3. 从 anchor 集合开始，用增量 Schur complement 计算候选 channel 对多领域条件输出残差的边际降低。
4. 在候选集合达到目标宽度前，同时计算补偿 norm、有效 rank 和 validation 稳定性；拒绝收益来自病态补偿的候选集合。
5. 对 `rank ∈ {0, 16, 32, 64, full}` 和扩展 ridge 网格进行 validation 选择，冻结后才读取 audit。
6. 将最终 $\Delta W$ 融合到保留 `down_proj`，导出物理压缩 expert。
7. 依次验证 expert 输出、完整模型 logits KL、PPL、路由重叠和下游任务，不得用局部 reconstruction error 代替端到端结果。

## 15.5 RAMP-v2 的边界

RAMP-v2 的局部目标是必要条件，不是最终模型质量指标。只有同时满足以下条件，才能声称方法具有模型级价值：

$$
\mathrm{local\ residual\ stable}
\land
\mathrm{logits\ KL\ controlled}
\land
\mathrm{PPL\ degradation\ controlled}
\land
\mathrm{downstream\ degradation\ controlled}.
$$

如果局部条件残差改善但 PPL 或下游任务恶化，结论应是“补偿目标与模型损失不一致”，而不是继续调局部阈值。
