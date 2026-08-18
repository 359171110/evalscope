# ENP / TENP 主体算法复现规范

> 适用目标：在用户自定义的上游校准集上统计重要性，生成 ENP 或 TENP 剪枝模型，再接入用户自定义的下游测试集。  
> 本文只规定主体算法、参数预算和模型改造，不绑定论文原始数据集、prompt、评测框架或生成参数。  
> 依据：He, Zhu, and Xiong, **TENP: Trapezoidal Expert Neuron Pruning for Mixture-of-Experts**, Findings of ACL 2026。  
> 论文未明确说明的部分均在文中标注为“补全设计”。

---

## 1. 方法目标与边界

### 1.1 ENP

ENP（Expert Neuron Pruning）对所有 routed experts 做相同保留率的结构化中间通道剪枝：

- routed expert 数量不变；
- Router 参数与 Top-k 设置不变；
- shared experts 默认不剪；
- dense FFN 默认不剪；
- 每个 routed expert 的 `gate_proj`、`up_proj`、`down_proj` 同步删除同一组中间通道；
- 所有 routed experts 的最终宽度相同，便于重新堆叠为规则张量。

### 1.2 TENP

TENP 在 ENP 之前增加“重要专家完整保留”：

- 每层先按专家重要性排序；
- 按随深度增加的梯形预算，选择一部分重要专家保持原始宽度；
- 其余专家仍保留并可被 Router 选择，但内部缩窄；
- routed expert 总数和 Router 拓扑不变；
- 最终形成“完整专家 + 窄专家”的静态异构结构。

TENP 不是整专家剪枝，也不是动态宽度推理。

---

## 2. 统一符号与张量方向

对第 \(l\) 个 MoE 层、第 \(e\) 个 routed expert：

- 隐藏维度：\(d\)
- 原始中间宽度：\(K_{l,e}\)
- 专家输入：\(X_{l,e}\in\mathbb{R}^{T_{l,e}\times d}\)
- 最终 Router 权重：\(g_{l,e}\in\mathbb{R}^{T_{l,e}}\)

按 PyTorch `nn.Linear(in_features, out_features)` 的权重布局定义：

\[
W^{l,e}_{\mathrm{gate}}\in\mathbb{R}^{K_{l,e}\times d},
\]

\[
W^{l,e}_{\mathrm{up}}\in\mathbb{R}^{K_{l,e}\times d},
\]

\[
W^{l,e}_{\mathrm{down}}\in\mathbb{R}^{d\times K_{l,e}}.
\]

标准 SwiGLU routed expert：

\[
M_{l,e}
=
\operatorname{SiLU}
\left(
X_{l,e} W_{\mathrm{gate}}^{l,e\top}
\right)
\odot
\left(
X_{l,e} W_{\mathrm{up}}^{l,e\top}
\right),
\]

\[
Y_{l,e}
=
M_{l,e} W_{\mathrm{down}}^{l,e\top}.
\]

其中：

- \(M_{l,e}\in\mathbb{R}^{T_{l,e}\times K_{l,e}}\)
- \(Y_{l,e}\in\mathbb{R}^{T_{l,e}\times d}\)

如模型采用其他 GLU/FFN 变体，只替换中间激活函数，通道的三矩阵绑定关系不变。

### 2.1 一个结构化通道包含什么

第 \(k\) 个中间通道对应：

\[
\Theta_{l,e,k}
=
\left\{
W_{\mathrm{gate}}^{l,e}[k,:],\;
W_{\mathrm{up}}^{l,e}[k,:],\;
W_{\mathrm{down}}^{l,e}[:,k]
\right\}.
\]

删除通道 \(k\) 时，必须同步删除以上三处。

### 2.2 默认不处理的参数

默认保持不变：

- Router；
- Attention；
- shared experts；
- dense FFN；
- embedding、LM head、norm；
- expert 的输出维度 \(d\)。

---

## 3. 实现前必须完成的模型适配

不要将算法直接写死到某个 Hugging Face 类。先实现统一适配层。

```python
class MoEModelAdapter:
    def iter_moe_layers(self):
        """返回所有包含 routed experts 的层，按深度排序。"""

    def num_routed_experts(self, layer):
        """返回该层 routed expert 数量。"""

    def get_expert(self, layer, expert_id):
        """返回单个 routed expert。"""

    def logical_weights(self, expert):
        """
        返回逻辑视图：
        W_gate: [K, d]
        W_up:   [K, d]
        W_down: [d, K]
        以及可选 bias。
        对 fused gate_up 或转置存储，在适配器内转换。
        """

    def intermediate(self, expert, x):
        """返回专家中间激活 M: [T, K]，严格复用原模型激活定义。"""

    def expert_output(self, expert, x):
        """返回未乘 Router 权重的专家输出 Y: [T, d]。"""

    def replace_expert(self, layer, expert_id, new_expert):
        """替换物理专家。"""

    def register_router_recorder(self):
        """
        记录每层：
        X_router: [T, d]
        topk_idx: [T, top_k]
        topk_weight: [T, top_k]
        valid_token_mask: [T]
        """
```

### 3.1 Router 权重的明确选择

必须使用实际 dispatch 前的最终权重：

- 已经过 Top-k；
- 已经过 softmax 或模型原有归一化；
- 与专家输出相乘的权重完全一致。

不要使用：

- 原始 router logits；
- Top-k 之前的全专家概率；
- 未归一化的 score；
- load-balancing auxiliary loss 中的概率。

### 3.2 记录的专家输入

记录 Router 和 routed expert 实际接收的张量，即：

- 通常是 MoE block 内 norm 之后的 hidden states；
- 不是上一层 residual add 前的张量；
- 不是 MoE 输出；
- 不是 token embedding。

---

## 4. 校准数据接口

算法只要求校准批次能被原模型正常前向。

推荐批次格式：

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "domain_id": optional,
}
```

规则：

1. 只统计 `attention_mask == 1` 的 token。
2. 不使用下游测试集做校准。
3. 校准阶段只前向，不更新模型参数。
4. 使用与部署模型一致的 tokenizer 和 chat template。
5. 多领域校准时保留 `domain_id`；单领域或不区分领域时全部视为同一域。
6. 不强制生成 response。teacher forcing 文本、prompt+reference、普通语料均可。
7. 不建议将不相关样本直接拼成一条普通因果序列，除非使用 block-diagonal attention 或接受 packing 引入的上下文变化。

### 4.1 覆盖度检查

ENP 需要每个 routed expert 至少接收到一定数量的校准 token。

默认要求：

```yaml
min_tokens_per_expert: 32
recommended_tokens_per_expert: 128
```

如果某个专家低于阈值：

1. 首选增加校准数据；
2. 其次增加校准域覆盖；
3. 不要静默地用零分裁剪该专家。

若经过合理扩充后仍为 0，可选补救：

- 将该专家保持完整；
- 或从 Router 排名靠前但未进入 Top-k 的 token 中补采输入，再离线运行该专家。

默认实现应在覆盖不足时报警，并输出每层每专家 token count。

---

## 5. 一次校准前向需要统计的内容

建议一次校准同时统计：

1. 专家重要性，供 TENP 使用；
2. 每个专家的通道重要性，供 ENP/TENP 使用；
3. 每层每专家路由次数；
4. 每层每专家最终 Router 权重均值；
5. 每层每专家有效 token 数。

这样 ENP 和 TENP 可共享同一份统计结果。

---

# 6. 专家重要性：TENP 第一阶段

## 6.1 单 token 专家分数

对被路由到专家 \(e\) 的 token 输入 \(x\)、专家未加权输出 \(y_e\)、最终 Router 权重 \(g_e>0\)：

\[
c_{e}(x)=g_e\|y_e\|_2,
\]

\[
s_e(x)
=
1-
\frac{\langle x,y_e\rangle}
{\|x\|_2\|y_e\|_2+\epsilon},
\]

\[
q_e(x)=c_e(x)s_e(x).
\]

专家分数：

\[
I_{l,e}
=
\sum_{x\in\mathcal{X}_{l,e}}
q_e(x).
\]

其中 \(\mathcal{X}_{l,e}\) 是校准期间实际被路由到该专家的 token 集合。

### 6.2 关于论文“求和/平均”歧义的处理

论文正文称对 token 求平均，但 Eq. 5 写成求和。本文采用：

\[
I_{l,e}=\sum q_e(x)
\]

作为主复现规则，原因是：

- 与论文公式一致；
- 保留专家使用频率的信息；
- 若统一除以该域总有效 token 数，只是层内公共常数，不改变排序。

实现中可除以该域的总有效 token 数 \(T_{\tau,l}\) 防止数值随数据量增长，但不能除以该专家自己的激活次数，否则会改变高频/低频专家之间的排序。

### 6.3 多领域聚合

对每个领域 \(\tau\) 独立累计一组层内专家分数：

\[
I^\tau_l
=
[I^\tau_{l,1},\ldots,I^\tau_{l,N_l}].
\]

按论文 Eq. 6 做层内 \(L_2\) 归一化：

\[
\widehat I^\tau_{l,e}
=
\frac{I^\tau_{l,e}}
{\sqrt{\sum_j (I^\tau_{l,j})^2}+\epsilon}.
\]

最终：

\[
I^{\mathrm{mix}}_{l,e}
=
\sum_{\tau}
w_\tau \widehat I^\tau_{l,e}.
\]

默认 \(w_\tau=1\)。如希望各领域严格等权，可令 \(\sum_\tau w_\tau=1\)。

单领域时直接使用 \(I_{l,e}\)。

### 6.4 专家分数代码

```python
@torch.no_grad()
def update_expert_score(
    x: torch.Tensor,          # [T, d]
    y: torch.Tensor,          # [T, d], expert output before router weight
    gate: torch.Tensor,       # [T], final dispatch weight
    score_sum: torch.Tensor,  # scalar accumulator
    eps: float = 1e-8,
):
    xf = x.float()
    yf = y.float()
    gf = gate.float()

    y_norm = yf.norm(dim=-1)
    cosine = (xf * yf).sum(dim=-1) / (
        xf.norm(dim=-1) * y_norm + eps
    )
    direction_change = 1.0 - cosine
    value = gf * y_norm * direction_change
    score_sum += value.sum()
```

数值累计使用 FP32；无需梯度。

---

# 7. 通道重要性：ENP 核心

## 7.1 论文定义

第 \(k\) 个中间通道对 token \(t\) 的独立输出：

\[
y_{t,k}
=
m_{t,k}W_{\mathrm{down}}[:,k].
\]

完整专家输出：

\[
y_t=\sum_{k=1}^{K}y_{t,k}.
\]

论文使用有符号投影：

\[
p_{t,k}
=
\frac{
\langle y_{t,k},y_t\rangle
}{
\|y_t\|_2+\epsilon
}.
\]

通道分数：

\[
P_k
=
\frac{1}{T}
\sum_{t=1}^{T}p_{t,k}.
\]

保留 \(P_k\) 最大的 Top-\(K_{\mathrm{keep}}\) 通道。

## 7.2 “projection magnitude”歧义

文字使用 magnitude，但公式和 Algorithm 1 都没有绝对值。因此主复现必须使用：

```yaml
neuron_score: signed_projection
```

即保留有符号投影。

可作为消融额外实现：

```yaml
neuron_score: abs_projection
neuron_score: l2_contribution
```

但这些不能代替论文主版本。

## 7.3 完全等价的低内存公式

论文 Algorithm 1 显式构造：

\[
C\in\mathbb{R}^{K\times T\times d},
\]

实际容易爆显存。可将投影严格等价地化简。

因为：

\[
\langle y_{t,k},y_t\rangle
=
m_{t,k}
\langle W_{\mathrm{down}}[:,k],y_t\rangle,
\]

先计算：

\[
Q=YW_{\mathrm{down}}\in\mathbb{R}^{T\times K},
\]

则：

\[
p_{t,k}
=
\frac{M_{t,k}Q_{t,k}}
{\|Y_t\|_2+\epsilon}.
\]

因此无需构造 \(K\times T\times d\) 张量。

## 7.4 推荐实现

```python
@torch.no_grad()
def signed_projection_scores(
    expert,
    x: torch.Tensor,     # [T, d], tokens actually routed to this expert
    adapter,
    eps: float = 1e-8,
):
    # M: [T, K]
    m = adapter.intermediate(expert, x)

    W_gate, W_up, W_down, biases = adapter.logical_weights(expert)
    # W_down: [d, K]

    # 为保证通道可加分解，忽略 down bias 计算 y_no_bias。
    y = m.float() @ W_down.float().T        # [T, d]

    # Q[t,k] = <y_t, W_down[:,k]>
    q = y @ W_down.float()                  # [T, K]

    denom = y.norm(dim=-1, keepdim=True) + eps
    p = m.float() * q / denom               # [T, K]

    return p.sum(dim=0), x.shape[0]
```

最终：

```python
score = score_sum / max(token_count, 1)
```

### 7.5 Bias 处理

论文模型默认无 bias。为兼容有 bias 的专家：

- `gate_proj`、`up_proj` bias 与对应行一起切片；
- `down_proj` bias 保留不变；
- 通道评分中的完整输出使用 `y_no_bias = M @ W_down.T`；
- 不将 `down_bias` 强行分摊到各通道。

这样保持“完整输出是各通道贡献之和”的定义。

### 7.6 多领域通道聚合：补全设计

论文只明确给出了专家分数的跨域归一化，没有明确说明通道分数如何跨域合并。

本文默认：

1. 每个领域独立计算 \(P^\tau_{l,e}\in\mathbb{R}^{K}\)；
2. 对每个“层—专家—领域”的通道分数向量做 \(L_2\) 归一化；
3. 跨域求和：

\[
P^{\mathrm{mix}}_{l,e}
=
\sum_\tau
w_\tau
\frac{P^\tau_{l,e}}
{\|P^\tau_{l,e}\|_2+\epsilon}.
\]

若某专家在某领域没有 routed token，则跳过该领域，不将零向量加入平均。

这个补全与论文的专家跨域处理原则一致，并避免 token 数较大的领域支配通道排序。

如果校准数据不区分领域，直接对全部 routed token 求平均。

---

# 8. ENP 完整流程

输入：

```yaml
method: enp
routed_param_retention: rho
width_multiple: 1
```

其中 \(\rho\in(0,1]\) 表示 routed-expert 参数保留率，不包含 shared experts、Router、Attention 等固定模块。

## 8.1 决定每个专家宽度

对原始宽度 \(K_{l,e}\)：

\[
K^{\mathrm{keep}}_{l,e}
=
\operatorname{round}(\rho K_{l,e}).
\]

默认纯算法复现：

```yaml
width_multiple: 1
rounding: nearest
```

面向 kernel 的部署版本可设：

```yaml
width_multiple: 64 / 128 / 256
rounding: floor
```

使用 `floor` 时实际保留率可能略低，必须报告实际值。

## 8.2 选择通道

```python
ranked = torch.argsort(score, descending=True, stable=True)
keep = ranked[:k_keep]
keep = torch.sort(keep).values
```

最后将 `keep` 升序排列，再同步切片三组矩阵。升序不是数学要求，但能保证：

- checkpoint 稳定；
- 原通道相对顺序保留；
- 易于调试和比较。

## 8.3 物理切片

\[
\widetilde W_{\mathrm{gate}}
=
W_{\mathrm{gate}}[\mathcal S,:],
\]

\[
\widetilde W_{\mathrm{up}}
=
W_{\mathrm{up}}[\mathcal S,:],
\]

\[
\widetilde W_{\mathrm{down}}
=
W_{\mathrm{down}}[:,\mathcal S].
\]

若存在 gate/up bias：

\[
\widetilde b_{\mathrm{gate}}=b_{\mathrm{gate}}[\mathcal S],
\quad
\widetilde b_{\mathrm{up}}=b_{\mathrm{up}}[\mathcal S].
\]

`down_bias` 不变。

## 8.4 ENP 伪代码

```python
def run_enp(model, adapter, stats, rho, width_multiple=1):
    plan = {}

    for layer in adapter.iter_moe_layers():
        for e in range(adapter.num_routed_experts(layer)):
            expert = adapter.get_expert(layer, e)
            K = adapter.logical_weights(expert)[0].shape[0]

            k_keep = aligned_width(
                target=rho * K,
                multiple=width_multiple,
                mode="nearest" if width_multiple == 1 else "floor",
            )

            scores = stats.neuron_scores[layer.id][e]
            keep = stable_topk_indices(scores, k_keep)
            plan[(layer.id, e)] = keep

    pruned_model = rebuild_model(model, adapter, plan)
    return pruned_model, plan
```

---

# 9. TENP 完整流程

TENP 输入：

```yaml
method: tenp
routed_param_retention: rho
important_expert_ratio: alpha
trapezoid:
  shallow_weight: 1.0
  deep_weight: 2.0
width_multiple: 1
```

其中：

- \(\rho\)：最终 routed-expert 总参数保留率；
- \(\alpha\)：完整保留的重要专家占全部 routed experts 的比例；
- 约束 \(0\leq\alpha\leq\rho\)。

论文在 routed-expert 保留率 60% 时：

- Qwen1.5-MoE 使用 \(\alpha=30\%\)；
- DeepSeek-V2-Lite 使用 \(\alpha=20\%\)。

若没有专门搜索，推荐：

\[
\alpha=\rho/2.
\]

这对应论文所说的“完整专家预算与窄专家所保留预算大致相等”。  
更稳妥的做法是在上游校准集内部额外划分一个 selection split，对：

\[
\alpha\in
\{0,\;0.25\rho,\;0.5\rho,\;0.75\rho\}
\]

使用 NLL、PPL、logit KL 或隐藏状态重构误差进行选择，不能使用下游测试集选择。

---

## 9.1 梯形层预算：论文缺失与补全设计

论文只说明：

- 深层完整保留的专家更多；
- 浅层更少；
- 总体形成梯形；
- 没有公布逐层数量、闭式公式或列表。

本文规定以下确定性线性梯形分配。

设共有 \(L\) 个 MoE 层，第 \(l\) 个 MoE 层深度归一化为：

\[
z_l=
\begin{cases}
0, & L=1,\\
\frac{l}{L-1}, & L>1.
\end{cases}
\]

定义层权重：

\[
w_l
=
w_{\mathrm{shallow}}
+
\left(
w_{\mathrm{deep}}-w_{\mathrm{shallow}}
\right)z_l.
\]

默认：

\[
w_{\mathrm{shallow}}=1,\qquad
w_{\mathrm{deep}}=2.
\]

设第 \(l\) 层有 \(N_l\) 个 routed experts，总完整专家数：

\[
M=
\operatorname{round}
\left(
\alpha\sum_l N_l
\right).
\]

实数目标：

\[
\widetilde m_l
=
M
\frac{N_lw_l}
{\sum_jN_jw_j}.
\]

整数分配采用 largest-remainder：

1. \(m_l=\lfloor\widetilde m_l\rfloor\)；
2. 按 \(\widetilde m_l-m_l\) 从大到小补足剩余名额；
3. 每层满足 \(0\leq m_l\leq N_l\)；
4. 若某层饱和，将溢出名额继续分配给未饱和层。

这样：

- 深层目标数量约为浅层的 2 倍；
- 总数严格等于 \(M\)；
- 不依赖模型层数；
- 对每层专家数不同的模型仍然有效。

```python
def trapezoid_counts(num_experts, alpha, shallow=1.0, deep=2.0):
    L = len(num_experts)
    total_slots = sum(num_experts)
    M = round(alpha * total_slots)

    if L == 1:
        z = [0.0]
    else:
        z = [l / (L - 1) for l in range(L)]

    weights = [
        num_experts[l] * (shallow + (deep - shallow) * z[l])
        for l in range(L)
    ]
    targets = [M * w / sum(weights) for w in weights]

    counts = [min(num_experts[l], int(targets[l] // 1)) for l in range(L)]
    remain = M - sum(counts)

    while remain > 0:
        candidates = [
            l for l in range(L) if counts[l] < num_experts[l]
        ]
        l = max(
            candidates,
            key=lambda j: (targets[j] - counts[j], j)
        )
        counts[l] += 1
        remain -= 1

    return counts
```

这是本文的补全设计，不是作者公开的精确 schedule。实验报告中应注明：

> TENP is reproduced with a deterministic linear trapezoidal layer allocation because the original paper does not specify the per-layer schedule.

---

## 9.2 每层选择完整专家

第 \(l\) 层保留 \(m_l\) 个完整专家：

```python
important_ids_l = stable_topk_indices(
    expert_scores_l,
    m_l,
)
```

同分时按 expert id 较小者优先，保证确定性。

---

## 9.3 计算次要专家的宽度比例

若所有 routed experts 原始大小相同，论文公式为：

\[
\beta
=
\frac{\rho-\alpha}{1-\alpha},
\]

其中 \(\beta\) 是次要专家通道保留率。

例如：

\[
\rho=0.6,\quad\alpha=0.3
\Rightarrow
\beta\approx0.4286.
\]

### 参数加权通用版本

对于原始宽度或隐藏维度不完全一致的模型，不能直接用专家数量计算。

设：

- routed expert 原始总参数 \(P_{\mathrm{all}}\)；
- 完整专家集合参数 \(P_{\mathrm{full}}\)；
- 次要专家原始参数 \(P_{\mathrm{narrow}}\)。

则：

\[
\beta
=
\frac{
\rho P_{\mathrm{all}}-P_{\mathrm{full}}
}{
P_{\mathrm{narrow}}
}.
\]

若 \(\beta<0\)，说明完整专家已经超过总预算，应减小 \(\alpha\)。  
若 \(\beta>1\)，说明完整专家比例过低或预算定义错误。

对每个次要专家：

\[
K^{\mathrm{keep}}_{l,e}
=
\operatorname{round}
(\beta K_{l,e}).
\]

完整专家：

\[
K^{\mathrm{keep}}_{l,e}=K_{l,e}.
\]

---

## 9.4 TENP 伪代码

```python
def run_tenp(
    model,
    adapter,
    stats,
    rho,
    alpha,
    shallow_weight=1.0,
    deep_weight=2.0,
    width_multiple=1,
):
    layers = list(adapter.iter_moe_layers())
    num_experts = [
        adapter.num_routed_experts(layer)
        for layer in layers
    ]

    full_counts = trapezoid_counts(
        num_experts,
        alpha,
        shallow=shallow_weight,
        deep=deep_weight,
    )

    important = {}
    for layer, m_l in zip(layers, full_counts):
        scores = stats.expert_scores[layer.id]
        important[layer.id] = set(
            stable_topk_indices(scores, m_l).tolist()
        )

    P_all = routed_expert_parameter_count(model, adapter)
    P_full = sum(
        expert_parameter_count(adapter.get_expert(layer, e))
        for layer in layers
        for e in important[layer.id]
    )
    P_narrow = P_all - P_full

    beta = (rho * P_all - P_full) / P_narrow
    if not (0.0 <= beta <= 1.0):
        raise ValueError(
            f"Infeasible TENP budget: beta={beta:.4f}. "
            "Reduce alpha or check budget definition."
        )

    plan = {}
    for layer in layers:
        for e in range(adapter.num_routed_experts(layer)):
            expert = adapter.get_expert(layer, e)
            K = adapter.logical_weights(expert)[0].shape[0]

            if e in important[layer.id]:
                keep = torch.arange(K)
            else:
                k_keep = aligned_width(
                    target=beta * K,
                    multiple=width_multiple,
                    mode="nearest" if width_multiple == 1 else "floor",
                )
                scores = stats.neuron_scores[layer.id][e]
                keep = stable_topk_indices(scores, k_keep)
                keep = torch.sort(keep).values

            plan[(layer.id, e)] = keep

    pruned_model = rebuild_model(model, adapter, plan)
    return pruned_model, plan, important, beta
```

---

# 10. 统计实现的两种方式

## 10.1 推荐：在线集成统计

在模型真实 MoE forward 中获得：

- dispatch 后每个专家的 \(X_{l,e}\)；
- 对应 gate weight；
- 已计算的中间激活 \(M\)；
- 专家输出 \(Y\)。

额外计算：

\[
Q=YW_{\mathrm{down}}
\]

即可更新全部通道分数。

优势：

- 不存大量 hidden states；
- 与实际路由完全一致；
- 一次校准前向即可完成。

代价：

- 需要修改模型的 MoE forward 或加深度 hook；
- 额外增加一个与 down projection 同数量级的矩阵乘。

## 10.2 更易适配：记录后离线回放

第一遍模型前向只保存每层每专家：

```python
{
    "x": routed_inputs,
    "gate": router_weights,
    "domain_id": ...
}
```

之后逐专家离线计算 expert score 和 neuron score。

为控制内存，可对每个专家做 reservoir sampling：

```yaml
max_tokens_per_expert: 2048
```

注意：

- reservoir 必须固定随机种子；
- expert score 若希望保留真实频率信息，应在线累计全部 token；
- neuron score 可使用每专家等上限的采样 token。

推荐组合：

- 专家重要性：在线全量累计；
- 通道重要性：每专家 reservoir 后离线计算。

---

# 11. 物理模型重建

## 11.1 ENP

所有 routed experts 宽度相同，通常可：

- 新建统一宽度 expert；
- 重新堆叠权重；
- 更新 `moe_intermediate_size`；
- 保存为规则 checkpoint。

这是首先应完成的版本。

## 11.2 TENP

TENP 会产生异构宽度：

- 完整专家宽度 \(K\)；
- 次要专家宽度 \(\beta K\)。

若原实现将专家权重堆叠为：

```python
[E, K, d]
```

则不能直接存不同 \(K\)。算法验证阶段推荐：

- 将每个专家改为独立 `ModuleList`；
- Router 和 dispatch 保持不变；
- 每个专家单独 forward；
- 先验证精度，不声称 kernel 加速。

物理 checkpoint 可增加：

```json
{
  "expert_intermediate_sizes": {
    "layer_0": [K_0, K_1, ...],
    "layer_1": [...]
  }
}
```

并使用自定义模型类加载。

只用 full-size tensor + channel mask 可以验证数值，但不算物理缩小 checkpoint。

---

# 12. 参数预算与报告指标

## 12.1 routed-expert 静态参数保留率

\[
R_{\mathrm{static}}
=
\frac{
\sum_{l,e}P^{\mathrm{pruned}}_{l,e}
}{
\sum_{l,e}P^{\mathrm{original}}_{l,e}
}.
\]

只统计 routed experts。

## 12.2 整模型参数保留率

\[
R_{\mathrm{model}}
=
\frac{P^{\mathrm{pruned}}_{\mathrm{whole}}}
{P^{\mathrm{original}}_{\mathrm{whole}}}.
\]

必须与 routed-expert 保留率分开报告。

## 12.3 平均激活专家参数保留率

对校准或测试 token 的实际路由：

\[
R_{\mathrm{active}}
=
\frac{
\sum_{l,t}\sum_{e\in\operatorname{TopK}_{l,t}}
P^{\mathrm{pruned}}_{l,e}
}{
\sum_{l,t}\sum_{e\in\operatorname{TopK}_{l,t}}
P^{\mathrm{original}}_{l,e}
}.
\]

这里不乘 Router 权重，因为它衡量实际执行的参数规模。可额外报告 gate-weighted 版本，但不能替代该指标。

ENP 中通常：

\[
R_{\mathrm{active}}\approx R_{\mathrm{static}}.
\]

TENP 中两者可能不同，因为高频专家可能恰好是完整专家。

---

# 13. 默认配置

## 13.1 ENP 默认配置

```yaml
method: enp
routed_param_retention: 0.60

scoring:
  neuron_score: signed_projection
  expert_score: gate_norm_direction
  domain_balance: l2
  eps: 1.0e-8
  accumulation_dtype: float32

calibration:
  min_tokens_per_expert: 32
  max_tokens_per_expert: 2048
  reservoir_seed: 42

pruning:
  width_multiple: 1
  rounding: nearest
  stable_tie_break: expert_or_channel_id
  prune_shared_experts: false
  prune_dense_ffn: false
```

## 13.2 TENP 默认配置

```yaml
method: tenp
routed_param_retention: 0.60
important_expert_ratio: 0.30

trapezoid:
  schedule: linear_largest_remainder
  shallow_weight: 1.0
  deep_weight: 2.0

scoring:
  neuron_score: signed_projection
  expert_score: gate_norm_direction
  domain_balance: l2
  eps: 1.0e-8
  accumulation_dtype: float32

calibration:
  min_tokens_per_expert: 32
  max_tokens_per_expert: 2048
  reservoir_seed: 42

pruning:
  width_multiple: 1
  rounding: nearest
  prune_shared_experts: false
  prune_dense_ffn: false
```

---

# 14. 必须完成的单元测试

## 14.1 高效投影公式等价性

在小张量上比较：

1. 显式构造每个通道输出 \(C[K,T,d]\)；
2. 使用 \(P=M\odot(YW_{\mathrm{down}})/\|Y\|\)。

要求：

```python
torch.testing.assert_close(
    score_explicit,
    score_fast,
    rtol=1e-5,
    atol=1e-6,
)
```

## 14.2 全宽重建恒等测试

令所有通道都保留。原专家与重建专家输出应一致：

```python
max_abs_error < 1e-5  # FP32
```

BF16/FP16 可适当放宽。

## 14.3 通道同步切片测试

随机选择通道集合 \(\mathcal S\)，检查：

```python
new_gate.weight == old_gate.weight[S, :]
new_up.weight   == old_up.weight[S, :]
new_down.weight == old_down.weight[:, S]
```

## 14.4 Router 结构保持测试

剪枝后：

- 每层 routed expert 数量不变；
- Router weight 逐元素不变；
- Top-k 超参数不变；
- 在同一个固定输入 hidden state 上，Router 输出与剪枝前一致。

注意：端到端模型后续层的路由可能因前层剪枝误差发生变化，这不表示 Router 结构被改动。

## 14.5 预算测试

输出：

- 目标 \(\rho\)；
- 实际 routed-expert retention；
- 整模型 retention；
- 每层 retention；
- 平均激活参数 retention。

若 `width_multiple=1`，实际 routed-expert retention 应与目标仅有整数舍入误差。

## 14.6 保存与重载测试

保存 checkpoint 后重新加载，同一输入输出必须与保存前模型一致。

---

# 15. 推荐的最小复现顺序

## 阶段 A：只实现单专家通道评分

1. 对一个 expert 收集 routed inputs；
2. 实现显式 Algorithm 1；
3. 实现低内存等价公式；
4. 验证两者一致；
5. 测试 Top-K 切片后 expert forward。

## 阶段 B：实现 ENP

1. 所有 routed experts 等宽裁剪；
2. 不改 Router；
3. 先做 in-memory 模型；
4. 再导出统一宽度 checkpoint；
5. 接入用户测试集。

ENP 是验证通道评分是否正确的最干净基线。

## 阶段 C：实现 TENP

1. 累计专家分数；
2. 实现确定性线性梯形分配；
3. 选择完整专家；
4. 按剩余预算计算 \(\beta\)；
5. 构建异构专家模型；
6. 接入同一测试集。

## 阶段 D：再做工程优化

只有在算法结果正确后再考虑：

- width buckets；
- grouped GEMM；
- vLLM/SGLang；
- Tensor Parallel checkpoint；
- 真实吞吐与显存。

---

# 16. 最小对照实验

在同一校准集、同一参数预算和同一测试集上至少比较：

1. Full model；
2. Random channel；
3. Weight norm channel；
4. ENP-L2；
5. ENP signed projection；
6. TENP；
7. 整专家剪枝基线。

推荐保留率：

```text
90%, 75%, 60%, 45%, 30%
```

快速筛选可先做：

```text
75%, 60%, 45%
```

TENP 的必要消融：

- `alpha = 0`：退化为 ENP；
- rectangle：各层完整专家比例相同；
- trapezoid：本文线性梯形；
- random important experts；
- random neurons；
- signed projection vs L2；
- `alpha` 小网格。

---

# 17. 复现时必须保存的中间文件

```text
stats/
  expert_scores.pt
  neuron_scores.pt
  route_counts.pt
  gate_weight_stats.pt
  token_counts.pt

plans/
  enp_plan.json
  tenp_plan.json
  per_layer_widths.json
  important_experts.json

reports/
  budget_audit.json
  calibration_coverage.json
  sanity_checks.json
  evaluation_results.json
```

`plan` 至少记录：

```json
{
  "layer_id": 0,
  "expert_id": 3,
  "original_width": 1408,
  "kept_width": 844,
  "kept_indices": [0, 2, 5],
  "is_full_expert": false
}
```

---

# 18. 论文歧义与本文最终选择汇总

| 问题 | 论文状态 | 本文主复现规则 |
|---|---|---|
| 专家分数求和还是平均 | 文字平均，公式求和 | 对 routed token 求和；可除以域总 token 数 |
| 通道 projection 是否取绝对值 | 文字写 magnitude，公式不取绝对值 | 不取绝对值，使用 signed projection |
| 多领域通道分数如何融合 | 未明确 | 每专家每域 L2 归一化后求和 |
| 梯形逐层预算 | 未给公式或列表 | 深度线性权重 1→2，largest remainder |
| 重要专家比例 | 论文按模型调节 | 默认 \(\alpha=\rho/2\)，推荐仅用校准 selection split 搜索 |
| 稀有专家无 token | 未说明 | 覆盖不足报警并扩充校准，默认不静默剪枝 |
| Algorithm 1 显存过高 | 显式构造 \(K\times T\times d\) | 使用严格等价的 \(YW_{\mathrm{down}}\) 公式 |
| shared experts | 实验中存在但目标是 routed experts | 默认完整保留 |
| bias | 论文模型通常无 bias | gate/up bias 同步切；down bias 不参与通道分解 |
| 异构专家 checkpoint | 未提供工程格式 | 自定义 per-expert width 配置和模型类 |

---

# 19. 完成标准

主体算法复现完成应满足：

- [ ] 能从任意用户校准集统计 expert/neuron scores；
- [ ] ENP 能物理导出统一窄宽度 checkpoint；
- [ ] TENP 能生成完整专家集合和窄专家宽度；
- [ ] Router、expert 数量和 Top-k 不变；
- [ ] shared experts 与 dense FFN 默认不变；
- [ ] 所有预算可审计；
- [ ] full-width 重建通过一致性测试；
- [ ] 快速投影公式通过等价性测试；
- [ ] checkpoint 可保存、重载和评测；
- [ ] 所有论文未说明的补全项写入配置与实验日志。
