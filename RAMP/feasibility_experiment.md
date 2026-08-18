# RAMP-E0：等宽专家输出重构可行性实验

## 1. 实验定位

本实验只验证 RAMP 最核心、也最容易被证伪的假设：

> 在固定保留 50% 通道时，依据“补偿后输出可重构性”选择的通道集合，能否在未参与拟合的 routed token 上，比传统 importance 排序获得更低的专家输出误差。

实验暂不验证：

- 异构专家宽度分配；
- 全模型 checkpoint 物理导出；
- 真实 kernel 加速；
- 大规模下游任务收益。

原因是这四项都依赖“保留通道确实能预测被删通道输出”这一前提。若该前提在独立 holdout token 上不成立，则没有必要先投入全模型 profile、runtime 和部署改造。

实验代号：`RAMP-E0`

实验模型：`Qwen3-30B-A3B-Instruct-2507`

模型结构：

| 项目 | 数值 |
| --- | ---: |
| MoE 层数 | 48 |
| routed experts / layer | 128 |
| routed experts / token | 8 |
| expert intermediate size | 768 |
| hidden size | 2048 |
| 首轮保留宽度 | 384 channels |
| 等价结构剪枝率 | 50% |

## 2. 预注册假设

### H1：补偿假设

给定任意合理的 384 通道保留集合，ridge 补偿应在 audit split 上显著降低直接剪枝产生的输出残差：

$$
\mathcal E_{\mathrm{audit}}(\mathcal K,\mathrm{ridge})
<
\mathcal E_{\mathrm{audit}}(\mathcal K,\mathrm{no\ compensation}).
$$

H1 验证“保留激活是否包含恢复被删输出的信息”。

### H2：选择假设

在使用完全相同的 ridge 拟合协议时，RAMP 选择的通道集合应优于 RMS、Tail 和随机选择：

$$
\mathcal E_{\mathrm{audit}}(\mathcal K_{\mathrm{RAMP}},\mathrm{ridge})
<
\min_b
\mathcal E_{\mathrm{audit}}(\mathcal K_b,\mathrm{ridge}),
$$

其中 $b$ 为传统 baseline。

H2 验证收益是否来自 reconstructability-aware selection，而不只是“剪完以后重新拟合 `down_proj`”。

### H3：泛化假设

RAMP 的收益不能只出现在拟合 split。其 audit 误差相对 validation 误差不应明显失控：

$$
\frac{\mathcal E_{\mathrm{audit}}}
{\mathcal E_{\mathrm{validation}}+\epsilon}
\leq 1.5.
$$

该阈值只作为过拟合报警线，不代替 H1 和 H2 的主检验。

## 3. 冻结实验输入

复用当前静态剪枝框架已经冻结的 train-only mixed token cache：

```text
static_moe_prunning/experiments/calibration/
qwen3_mixed_train_wikitext256_mbpp128_gsm8k64_math64_20260802/
mixed_train_512x1024_code_augmented.pt
```

冻结身份：

```text
input_ids SHA256: 588f0e45bc49601c3fb951828c0b1bb78bf15809e193ff9c5a854ef10483c03a
cache file SHA256: 8052987634bab450559c18e5ebfb55ccd82b8240cd93434268e911bcf91db1a7
```

该 cache 包含 512 条、每条 1024 token 的训练序列，来源为 WikiText、代码、GSM8K 和 MATH。不得使用任何 evaluation split 选择通道、正则系数或专家样本。

以下已有 channel cache 只用于在读取任何 RAMP reconstruction metric 前冻结代表专家，并作为结果复核用的外部参考：

```text
RMS:
static_moe_prunning/experiments/calibration/
qwen3_mixed_512x1024_code_augmented_20260802/channels_rms_512x1024.pt

Tail:
static_moe_prunning/experiments/calibration/
qwen3_mixed_512x1024_code_augmented_20260802/tail_channels/
qwen3_channels_b64_tail_0p50.pt
```

冻结 SHA：

```text
RMS channel file SHA256: 001ae6846df0512121181683b1a03a57fe6cba9f9400a3b53d81175de2348f95
Tail channel file SHA256: 98f99772df910883ebc7dd8e73f62c76645c5e2d322aaff4e42e1b8ad98eac9d
```

主比较中的 RMS 和 Tail 排名必须从 fit split 重新计算，不能直接使用上述覆盖全部 512 条序列的排名，否则 baseline 会看到 validation 和 audit token。

## 4. 校准、验证和审计划分

从 512 条冻结训练序列中只使用 192 条。cache 的 `source.sequence_order` 已保存每条序列的来源，因此采用确定的分层划分：

| split | WikiText | GSM8K | code | MATH | 总序列数 | token 数 | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fit | 48 | 12 | 24 | 12 | 96 | 98,304 | 选择通道、拟合补偿 |
| validation | 24 | 6 | 12 | 6 | 48 | 49,152 | 选择 ridge 系数和补偿秩 |
| audit | 24 | 6 | 12 | 6 | 48 | 49,152 | 只做一次最终报告 |

划分规则：

1. 使用固定随机种子 `42`。
2. 按 cache 中的 `source_order = [wikitext, gsm8k, mbpp, math]`，使用同一个 `torch.Generator` 依次对各来源的 sequence index 调用 `torch.randperm()`。
3. 每个来源先取 fit 配额，再取 validation 配额，最后取 audit 配额；每个 split 内的全局 sequence index 按升序保存。
4. 保存三个 index tensor、原始 cache SHA 和 index tensor SHA。
5. audit split 在方法、anchor 比例、ridge 系数和 rank 全部冻结前不可读取或累计统计量。

冻结 index tensor SHA256：

```text
fit:       473b0a464fce3306843b4fa77a6aa0f239972e4fbe151fe7b30ba00dd2323f48
validation: 30f479bc6a84bc268c91ef6a4fc6edf91123f1d98740366850dbe76c5924acf8
audit:     d9688f32228b6373de05a6937c36d7f1e6722028b61917456ed2cc54981ca8b8
```

完整 index 数组写入 manifest，不在本文重复展开。

按均匀路由估计，每个专家在 fit / validation / audit 中分别约有 6,144 / 3,072 / 3,072 个 routed token。实际记录每个专家的有效样本数和 $\sum g^2$。

## 5. 代表专家的预注册选择

首轮不直接处理全部 $48\times128=6,144$ 个专家，而是选择 24 个具有代表性的专家。

固定层：

```text
layer = [0, 15, 31, 47]
```

在每个固定层中，根据现有 RMS cache 的 `route_counts` 对 128 个物理专家排序，并从以下三个访问频率层级各取两个专家。该统计只用于预注册被观察对象，不参与通道选择、补偿拟合或超参数选择：

```text
low:    最接近 route-count 10% 分位数的两个专家
medium: 最接近 route-count 50% 分位数的两个专家
high:   最接近 route-count 90% 分位数的两个专家
```

总数为：

$$
4\ \text{layers}
\times 3\ \text{frequency strata}
\times 2
=24\ \text{experts}.
$$

按上述规则冻结得到：

| layer | frequency | expert ID（完整 512 序列 route count） |
| ---: | --- | --- |
| 0 | low | 42（15,076），56（14,323） |
| 0 | medium | 0（31,557），103（31,299） |
| 0 | high | 27（49,739），7（50,199） |
| 15 | low | 62（8,832），15（8,852） |
| 15 | medium | 51（27,705），102（27,453） |
| 15 | high | 98（64,545），32（65,498） |
| 31 | low | 70（6,949），90（6,836） |
| 31 | medium | 37（24,575），116（27,000） |
| 31 | high | 1（68,090），61（69,390） |
| 47 | low | 64（3,490），49（3,655） |
| 47 | medium | 2（28,971），93（29,446） |
| 47 | high | 82（62,236），90（61,672） |

约束：

- 使用物理 expert ID，不使用 router rank。
- 专家 ID 必须在计算任何 reconstruction metric 前写入 manifest。
- 若某个专家在 fit split 中 routed token 少于 768，则保留该专家用于暴露低频失败，但单独标记为 `underdetermined`，不允许静默替换。
- 主结果同时报告全部 24 个专家和排除 `underdetermined` 专家后的结果。

## 6. 只保存充分统计量

不保存每个 token 的完整激活轨迹。对每个目标专家和每个 split，只累计 gate 加权的激活二阶矩：

$$
C_e^{(s)}
=
\sum_{n\in s,\ e\in\operatorname{TopK}(n)}
g_{n,e}^2 a_{n,e}a_{n,e}^{\top},
$$

其中：

$$
C_e^{(s)}\in\mathbb R^{768\times768}.
$$

同时保存：

$$
N_e^{(s)},
\qquad
G_e^{(s)}=\sum_n g_{n,e}^2,
\qquad
\sum_n g_{n,e}^2\|E_e(x_n)\|_2^2.
$$

为在 fit split 上公平重算 RMS 和 Tail baseline，还需累计未乘 gate 的对角统计：

$$
U_{e,j}=\sum_n a_{n,e,j}^2,
\qquad
V_{e,j}=\max_n|a_{n,e,j}|.
$$

建议用 float64 累计、float32 落盘，并保存对称性误差和最小特征值诊断。24 个专家、三个 split 的 covariance 约占 170 MiB float32，远小于保存 routed activation 和 expert output。

### 6.1 为什么二阶矩足够

记完整 `down_proj` 为：

$$
W\in\mathbb R^{2048\times768}.
$$

给定保留集合 $\mathcal K$ 和其补偿后权重 $\widetilde W_{\mathcal K}$，构造系数误差矩阵：

$$
Q_{:,\mathcal K}=W_{:,\mathcal K}-\widetilde W_{\mathcal K},
$$

$$
Q_{:,\mathcal P}=W_{:,\mathcal P}.
$$

则任意 split 上的 gate 加权残差平方和可以精确写为：

$$
\operatorname{SSE}^{(s)}
=
\operatorname{tr}
\left(
Q C_e^{(s)}Q^{\top}
\right).
$$

完整专家输出能量为：

$$
\operatorname{Energy}^{(s)}
=
\operatorname{tr}
\left(
W C_e^{(s)}W^{\top}
\right).
$$

因此无需重新回放激活，即可精确计算归一化输出误差。

## 7. RAMP 通道选择

### 7.1 Anchor channels

在 fit split 上定义单通道直接输出能量：

$$
s_j=C_{jj}^{(\mathrm{fit})}\|W_{:,j}\|_2^2.
$$

先固定保留最终宽度的 10% 作为 anchor：

```text
anchor channels = 38
```

若出现并列，按原始 channel ID 升序打破平局。

另做 `anchor=0` 消融，但不能根据 audit 结果选择主版本。

### 7.2 Output-aware conditional residual selection

其余通道使用 fit covariance 做输出感知的条件残差贪心。目标不是恢复被删激活，而是选择最能解释当前“尚未被已选通道表示的专家输出”的通道。

定义输出方向 Gram 矩阵：

$$
M=W^{\top}W.
$$

初始化条件激活 covariance 和对应输出残差核：

$$
S=C^{(\mathrm{fit})},
\qquad
H=SMS.
$$

对于尚未选择的候选通道 $j$，定义 gain：

$$
G_j
=
\frac{H_{jj}}
{S_{jj}+\lambda_{\mathrm{sel}}}.
$$

其中 $H_{jj}=\|WS_{:,j}\|_2^2$ 衡量该通道在条件化掉当前集合后仍能解释的输出残差能量。每次选择 gain 最大的通道 $j^*$，然后用 Schur-complement rank-one update 更新 $S$；$H$ 也使用对应的 rank-one 公式更新。初始化 anchor 集合时按同一更新规则依次条件化。

该实现每一步同时更新所有候选的条件统计，不为每个候选重新求解一次回归。若 gain 并列，按原始 channel ID 升序选择。

选择阶段固定：

$$
\lambda_{\mathrm{sel}}
=
10^{-4}
\cdot
\frac{\operatorname{tr}(C)}{768}.
$$

每次加入 gain 最大的一个通道，直到：

$$
|\mathcal K|=384.
$$

这一步只使用 fit split。validation 只能选择最终补偿正则和 rank，不能回头修改通道集合。

## 8. 补偿拟合

对任意方法产生的保留集合 $\mathcal K$，使用完全相同的补偿求解器。

令 $\mathcal P$ 为被删集合，ridge 补偿闭式解为：

$$
\Delta W
=
W_{:,\mathcal P}
C_{\mathcal P,\mathcal K}^{(\mathrm{fit})}
\left(
C_{\mathcal K,\mathcal K}^{(\mathrm{fit})}
+\lambda I
\right)^{-1}.
$$

最终权重为：

$$
\widetilde W_{\mathcal K}
=
W_{:,\mathcal K}+\Delta W.
$$

使用尺度无关的正则网格：

$$
\lambda
=
\alpha
\cdot
\frac{
\operatorname{tr}
\left(C_{\mathcal K,\mathcal K}^{(\mathrm{fit})}\right)
}{|\mathcal K|},
$$

$$
\alpha\in
\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}.
$$

每个专家、每个通道选择方法独立使用 validation split 选择 $\alpha$，然后冻结并在 audit split 上评估一次。最终报告同时给出“逐专家选择 $\alpha$”和“24 个专家共享一个 $\alpha$”两种结果，主结果使用共享 $\alpha$，降低小样本调参自由度。

同时评估：

- `no_compensation`；
- `ridge_full_rank`；
- `ridge_rank_16`。

`ridge_rank_16` 从 full-rank $\Delta W$ 得到，但截断必须在保留激活的 covariance metric 下进行，不能直接对未白化的 $\Delta W$ 做普通 SVD。若 $C_{\mathcal K,\mathcal K}=LL^{\top}$，先对 $\Delta WL$ 做 rank-16 truncated SVD，再右乘 $L^{-1}$ 还原。

## 9. 对照组

所有方法均严格保留 384 个通道。

| ID | 通道选择 | 补偿 | 目的 |
| --- | --- | --- | --- |
| `random_none` | 随机，5 seeds | 无 | 随机下界 |
| `random_ridge` | 随机，5 seeds | full-rank ridge | 补偿是否对任意集合都有效 |
| `rms_none` | fit-only RMS top-384 | 无 | 标准 importance baseline |
| `rms_ridge` | fit-only RMS top-384 | full-rank ridge | 公平分离 selection 与 compensation |
| `tail_none` | fit-only Tail-$\lambda=0.5$ top-384 | 无 | 当前 tail-aware baseline |
| `tail_ridge` | fit-only Tail-$\lambda=0.5$ top-384 | full-rank ridge | 强 baseline |
| `ramp_none` | anchor + supervised OMP | 无 | RAMP 集合自身的直接剪枝表现 |
| `ramp_rank16` | anchor + supervised OMP | rank-16 ridge | 受限补偿 |
| `ramp_ridge` | anchor + supervised OMP | full-rank ridge | 可重构性的性能上界 |
| `dense` | 768 channels | 不适用 | 零误差参照 |

随机 baseline 的主结果使用五个 seed 的均值，并报告标准差；与 RAMP 做配对比较时使用每个专家上最优随机 seed 和随机均值两种口径。

## 10. 主指标

### 10.1 Gate 加权归一化专家输出误差

主指标为：

$$
\mathcal E_e^{(s)}
=
\frac{
\operatorname{tr}
\left(Q C_e^{(s)}Q^{\top}\right)
}{
\operatorname{tr}
\left(W C_e^{(s)}W^{\top}\right)+\epsilon
}.
$$

主报告使用 audit split。

### 10.2 被删残差的可预测比例

定义直接剪枝残差为分母：

$$
R^2_{\mathrm{pruned},e}
=
1-
\frac{
\operatorname{SSE}_{\mathrm{compensated},e}
}{
\operatorname{SSE}_{\mathrm{no\ compensation},e}+\epsilon
}.
$$

该指标直接回答“被删通道输出有多少可以由保留通道恢复”。

### 10.3 补偿规模

$$
\eta_e
=
\frac{\|\Delta W_e\|_F}
{\|W_{e,:,\mathcal K}\|_F+\epsilon}.
$$

过大的 $\eta$ 表示补偿可能在强行重写专家功能。

### 10.4 次要诊断

每个专家还报告：

- fit、validation、audit 三个 split 的误差；
- routed token 数、$\sum g^2$ 和有效样本/保留宽度比；
- 输出 cosine similarity；
- 每 token 相对误差的 median、P95 和 P99；
- covariance condition number；
- 低、中、高访问频率分组结果；
- 早、中、晚层分组结果；
- 若能恢复来源标签，则报告 WikiText、code、GSM8K、MATH 分域结果。

其中 per-token 分位数需要在 audit forward 中流式计算，不能仅由 covariance 恢复。

## 11. 成功和失败标准

### 11.1 Go：进入全层等宽实验

必须同时满足：

1. `ramp_ridge` 的 audit 主误差在 24 个专家上的 median relative reduction 比最佳 `rms_ridge` / `tail_ridge` 至少高 15%，其中 relative reduction 定义为 $(\mathcal E_b-\mathcal E_{\mathrm{RAMP}})/(\mathcal E_b+\epsilon)$。
2. 至少 18 / 24 个专家上，`ramp_ridge` 优于最佳 importance + ridge baseline。
3. `ramp_ridge` 相比 `ramp_none` 的 median $R^2_{\mathrm{pruned}}\geq0.50$。
4. `ramp_rank16` 保留 `ramp_ridge` 至少 70% 的误差下降，证明收益不完全依赖满秩补偿。
5. 至少 20 / 24 个专家满足 audit / validation error ratio $\leq1.5$。
6. median $\eta\leq0.5$，且 P90 $\eta\leq1.0$。

统计报告使用 expert-level paired bootstrap，固定 seed `42`、10,000 次重采样，给出 RAMP 相对最佳 baseline 的 median improvement 95% confidence interval。置信区间下界应大于 0。

### 11.2 Revise：补偿成立但选择方法不成立

出现以下情况时，不进入异构宽度阶段，而是只修改 channel selection：

- H1 通过，但 H2 不通过；
- `rms_ridge` 或 `tail_ridge` 与 `ramp_ridge` 持平；
- anchor=0 明显优于 anchor=10%；
- RAMP 只对高频专家有效。

这说明“输出可补偿”有价值，但 supervised OMP、anchor 或选择正则需要调整。

### 11.3 No-Go：核心可重构性不成立

满足任一情况应暂停全模型实现：

- `ramp_ridge` 的 median audit $R^2_{\mathrm{pruned}}<0.20$；
- fit 明显改善，但 audit / validation ratio 在多数专家上大于 2；
- 只有 full-rank 补偿有效，rank-16 基本无收益且 $\eta$ 很大；
- RAMP 不优于随机 + ridge；
- 低频和 specialist-domain 专家出现系统性负收益。

## 12. 实施步骤

### Step 1：冻结 manifest

创建：

```text
RAMP/experiments/ramp_e0/manifest.json
```

记录：

- 模型路径和 checkpoint 身份；
- calibration cache 路径、文件 SHA、input token SHA；
- RMS/Tail cache 路径和 SHA；
- split sequence indices 和 SHA；
- 24 个 `(layer, physical_expert)`；
- 随机 seed；
- 方法、正则网格、rank 和成功门槛；
- `test_metrics_used_for_selection=false`。

### Step 2：采集充分统计量

建议新增：

```text
RAMP/code/collect_ramp_covariances.py
```

复用静态框架中的：

- `iter_moe_layer_bindings()`；
- `route_qwen3_topk()`；
- `split_gate_up_proj()`；
- `compute_moe_weighted_hidden_states()`；
- frozen calibration token loader。

第一遍只对 manifest 中的 24 个专家累计 fit 和 validation covariance，模型输出路径保持 dense-equivalent。audit sequence indices 虽然已经冻结，但此时不得送入模型，也不得生成 audit covariance。

### Step 3：离线选择和拟合

建议新增：

```text
RAMP/code/run_ramp_reconstruction_probe.py
```

该脚本只读取：

- manifest；
- covariance cache；
- 模型中的 `down_proj`；
- fit-only covariance、未加权 RMS 对角统计和 Tail 统计。

输出每个专家、每个方法、每个 $\alpha$ 的 fit/validation 指标。选择完 $\alpha$ 后写入冻结 decision 文件，不读取 audit covariance。

### Step 4：一次性 audit

建议新增：

```text
RAMP/code/audit_ramp_reconstruction.py
```

该脚本要求 decision 文件已存在，并拒绝重新选择 channel、$\alpha$、rank 或 anchor 比例。此时才运行 48 条 audit 序列，累计 audit covariance 并流式计算 per-token 分位数。

### Step 5：生成报告

保存：

```text
RAMP/experiments/ramp_e0/results.json
RAMP/experiments/ramp_e0/per_expert.csv
RAMP/experiments/ramp_e0/summary.md
RAMP/experiments/ramp_e0/logs/
```

报告至少包含：

1. 各方法 audit error 的 paired expert plot；
2. reconstruction error 对 route frequency；
3. fit、validation、audit generalization gap；
4. $R^2_{\mathrm{pruned}}$ 与 $\eta$；
5. rank-16 对 full-rank 的收益保留比例；
6. 24 个专家逐项结果，不只报告平均值。

## 13. 资源估计

### 采集阶段

- 只运行 192 / 512 条序列，dense forward 成本约为现有完整 Hessian calibration 的 37.5%。
- 只在 4 个层、24 个专家上累计完整 covariance。
- covariance 落盘约 170 MiB float32，加上诊断和元数据应低于 300 MiB。
- GPU 只能使用物理 GPU 4–7，并显式设置 `CUDA_VISIBLE_DEVICES`。

### 离线阶段

- 每个专家维度只有 768，RAMP OMP 使用 Cholesky 增量更新。
- 24 个专家可以串行运行，优先保证可复现和数值稳定，不需要增加依赖。
- 所有矩阵运算使用 float64 求解，最终权重和指标可转 float32 保存。

## 14. 最小结论边界

RAMP-E0 通过时，只能得出：

> 在代表性 Qwen3 MoE 专家和独立 train-only holdout routed token 上，保留通道能够重构被删通道输出，并且 reconstructability-aware selection 优于相同预算下的 importance selection。

它不能直接证明：

- 全模型 PPL 或下游任务一定提升；
- 不同专家应采用异构宽度；
- 补偿跨更远分布仍然有效；
- 物理导出后一定获得真实延迟收益。

只有 RAMP-E0 达到 Go 标准，下一实验才扩展到“所有专家统一 50% 宽度 + 融合补偿”，并在冻结 WikiText PPL、路由重叠和小规模下游任务上验证端到端效果。异构宽度分配应放在等宽端到端实验之后。