# RAMP-E1：补偿对齐通道选择与模型级传递实验

## 1. 实验目标

RAMP-E1 验证修正版方法的核心主张：

> 当通道选择直接最小化 ridge 补偿后的输出加权条件残差时，能否稳定优于边际 importance、pairwise correlation 和 RAMP-E0 selector，并且该局部收益能否传递到完整模型的 logits 与 PPL。

实验分为三个阶段。阶段 A 只做专家级离线选择与独立 audit；阶段 B 只对通过阶段 A 的候选构造有限层干预模型；阶段 C 才构造全层等宽压缩模型并测试 PPL、下游质量和部署收益。任何阶段失败都停止扩展，不能用后续 test 数据返回调整前一阶段的方法。

实验代号：`RAMP-E1`

目标模型：`Qwen3-30B-A3B-Instruct-2507`

首轮固定设置：

| 项目 | 数值 |
| --- | ---: |
| expert intermediate size | 768 |
| 保留宽度 | 384 |
| 结构剪枝率 | 50% |
| 首轮层 | 0, 15, 31, 47 |
| 每层专家 | low / medium / high frequency 各 2 个 |
| 总代表专家 | 24 |

固定 50% 宽度是为了与 RAMP-E0 直接比较，不在本实验中同时引入异构宽度分配。

## 2. 预注册假设

### H1：条件选择优于边际选择

使用相同 ridge 求解器和 validation 协议时，输出加权条件残差 selector 的 audit error 应显著低于 RMS、Tail、pairwise correlation 和 RAMP-E0 selector：

$$
\mathcal E_{\mathrm{audit}}(\mathcal K_{\mathrm{conditional}})
<
\min_b
\mathcal E_{\mathrm{audit}}(\mathcal K_b).
$$

### H2：收益来自目标对齐

逐级消融应表现出以下趋势：

$$
\text{pairwise correlation}
\rightarrow
\text{set-conditional residual}
\rightarrow
\text{output-weighted conditional residual}
\rightarrow
\text{stability-regularized objective}.
$$

若只有最后一次复杂化有效，需要通过消融确认收益来自哪一项，而不能把所有机制作为不可分割的整体报告。

### H3：补偿复杂度可控

局部收益不能仅依赖满秩、大范数补偿。validation 选择的低秩补偿应保留 full-rank 大部分收益，并且 audit 上不出现明显反转。

### H4：局部收益能够传递

通过阶段 A 的 selector 在有限层干预和全层等宽模型中，应同时降低原模型与压缩模型之间的 logits KL，并改善 held-out PPL。专家级 reconstruction error 的改善若不能传递到这两个指标，则不能进入下游任务结论。

## 3. 数据和冻结协议

### 3.1 数据划分

继续使用冻结的 mixed train-only token cache，但扩大统计支持。建议从 512 条序列中使用：

| split | 序列数 | token 数 | 用途 |
| --- | ---: | ---: | --- |
| fit | 256 | 262,144 | 选择集合、拟合 covariance 和补偿 |
| validation | 128 | 131,072 | 选择 selector 超参数、ridge、rank 和 anchor |
| audit | 128 | 131,072 | 冻结后一次性专家级评估 |

三个 split 按 WikiText、code、GSM8K 和 MATH 来源分层，固定 seed `42`。必须保存 sequence indices、来源配额、cache SHA 和 index SHA。

audit covariance 在全部 selector、anchor、group size、ridge、rank 和正则权重冻结前不得生成或读取。

### 3.2 低频专家支持

每个专家报告 routed-token 数和有效样本量：

$$
N_{\mathrm{eff}}
=
\frac{(\sum_n g_n^2)^2}{\sum_n g_n^4}.
$$

按 $N_{\mathrm{eff}}/|\mathcal K|$ 分层报告结果。若 fit routed-token 数低于 1,536 或 $N_{\mathrm{eff}}/|\mathcal K|<2$，标记为 `low_support`，不得静默替换。另建主动采样扩展作为诊断，但不能与自然路由主结果混合。

## 4. 充分统计量

每个专家、每个 split 收集 gate 加权 covariance：

$$
C^{(s)}
=
\sum_{n\in s}g_n^2a_na_n^{\top}.
$$

同时保存：

- routed-token 数、$\sum g^2$、$\sum g^4$；
- 原专家输出能量 $\operatorname{tr}(WCW^{\top})$；
- fit-only RMS 和 Tail 所需统计；
- covariance spectrum、condition number 和 shrinkage 诊断；
- 按数据来源分开的 validation/audit covariance，用于域稳定性报告。

使用 float64 累计和求解，落盘格式保持与 RAMP-E0 兼容。所有 selection 方法必须读取同一份统计量。

## 5. 通道选择对照

所有方法严格保留 384 个通道，并使用相同的补偿、validation 和 audit 协议。

| ID | 选择规则 | 作用 |
| --- | --- | --- |
| `random` | 5 个固定 seeds | 随机下界 |
| `rms` | fit-only RMS top-384 | 边际 importance baseline |
| `tail` | fit-only Tail top-384 | tail-aware importance baseline |
| `ramp_e0` | E0 anchor + supervised OMP | 原方法基线 |
| `pair_corr` | 输出轨迹 pairwise correlation 聚类/选代表 | 检验简单相关性是否足够 |
| `conditional_activation` | 最小化未做 $W$ 加权的条件激活残差 | 分离 set conditioning 收益 |
| `conditional_output` | 最小化输出加权条件残差 | RAMP v2 核心版本 |
| `conditional_stable` | 条件输出残差 + 复杂度/validation 稳定性 | RAMP v2 完整版本 |

核心目标为：

$$
J_{\mathrm{recon}}(\mathcal K;\lambda)
=
\operatorname{tr}
\left(
W_{\mathcal P}
\left[
C_{\mathcal P\mathcal P}
-
C_{\mathcal P\mathcal K}
(C_{\mathcal K\mathcal K}+\lambda I)^{-1}
C_{\mathcal K\mathcal P}
\right]
W_{\mathcal P}^{\top}
\right).
$$

完整版本使用：

$$
J(\mathcal K)
=
J_{\mathrm{recon}}^{\mathrm{fit}}
+
\beta_{\mathrm{norm}}\eta^2
+
\beta_{\mathrm{gap}}G_{\mathrm{val}},
$$

其中 $\eta=\|\Delta W\|_F/(\|W_{\mathcal K}\|_F+\epsilon)$。首轮不把过多正则项同时放入主方法；有效 rank 和 condition number 先作为诊断，只有预注册消融证明必要时才加入正式 score。

### 5.1 搜索策略

比较以下搜索实现：

1. 单通道 forward selection；
2. 以 4 或 8 个通道为组的 group selection；
3. 达到 384 通道后的固定次数 swap refinement。

搜索只使用 fit statistic。validation 可以从预注册候选集合中选择方案，但不能根据 audit 重新搜索。

anchor 比例作为消融：

```text
0%, 2.5%, 5%, 10%
```

主版本由 validation 聚合误差选择一个全局比例，避免逐专家自由调参。

## 6. 补偿和正则

所有 selector 使用相同的 ridge 补偿：

$$
\Delta W
=
W_{\mathcal P}C_{\mathcal P\mathcal K}
(C_{\mathcal K\mathcal K}+\lambda I)^{-1}.
$$

预注册尺度无关 ridge 网格：

```text
alpha = 1e-3, 1e-2, 1e-1, 3e-1, 1, 3, 10, 30
```

其中：

$$
\lambda
=
\alpha
\frac{\operatorname{tr}(C_{\mathcal K\mathcal K})}{|\mathcal K|}.
$$

补偿 rank：

```text
none, 16, 32, 64, 128, full
```

低秩截断必须在保留激活 covariance metric 下完成。validation 选择一个跨专家共享的主 rank 和 alpha；逐专家最优值只作为诊断上界。

## 7. 阶段 A：专家级独立 audit

### 7.1 主指标

Gate 加权归一化专家输出误差：

$$
\mathcal E_e^{(s)}
=
\frac{\operatorname{tr}(Q C_e^{(s)}Q^{\top})}
{\operatorname{tr}(W C_e^{(s)}W^{\top})+\epsilon}.
$$

直接剪枝残差被补偿消除的比例：

$$
R^2_{\mathrm{pruned},e}
=
1-
\frac{\operatorname{SSE}_{\mathrm{compensated},e}}
{\operatorname{SSE}_{\mathrm{none},e}+\epsilon}.
$$

### 7.2 稳定性指标

同时报告：

- fit/validation/audit error 和 generalization gap；
- $\|\Delta W\|_F/\|W_{\mathcal K}\|_F$；
- compensation effective rank 和奇异值谱；
- $C_{\mathcal K\mathcal K}$ condition number；
- low/medium/high frequency 和各层分组结果；
- WikiText、code、GSM8K、MATH 分域误差；
- rank-16/32/64 相对 full-rank 的收益保留率。

### 7.3 阶段 A 通过条件

`conditional_output` 或 `conditional_stable` 必须同时满足：

1. 相对最佳 RMS/Tail/RAMP-E0 的 median audit error 至少降低 8%；
2. 至少 18/24 个专家优于最佳基线；
3. expert-level paired bootstrap 95% CI 下界大于 0；
4. median audit residual $R^2\geq0.20$；
5. rank-64 保留 full-rank 至少 80% 的误差下降；
6. 至少 20/24 个专家的 audit/validation error ratio 不超过 1.5；
7. low-support 专家没有系统性负收益。

8% 是进入模型级验证的工程门槛，不是方法正确性的数学证明。若 improvement 为 3%–8% 且置信区间为正，归类为 `REVISE_LOCAL_SELECTION`，只允许继续改进局部方法，不进入全层模型。

## 8. 阶段 B：有限层模型级传递

只对阶段 A 最优 selector 构造以下模型变体：

| 变体 | 修改范围 | 目的 |
| --- | --- | --- |
| `dense` | 不修改 | 原模型参照 |
| `best_importance` | 层 0/15/31/47 的目标专家 | 公平模型级 baseline |
| `ramp_e1` | 相同层和专家 | 检验局部收益能否传递 |

需要真实替换 `gate_proj`、`up_proj` 和融合补偿后的 `down_proj`，不能只用离线 covariance 模拟。

在与 calibration/audit 不重叠的 held-out 文本上测量：

- 每层 hidden-state normalized MSE 和 cosine drift；
- 后续层 Router top-$k$ overlap；
- Router probability KL；
- 最终 token logits KL；
- next-token negative log-likelihood 差值。

阶段 B 通过条件：

1. RAMP-E1 的 median token logits KL 比最佳 importance baseline 至少低 5%；
2. held-out NLL 增量不高于最佳 importance baseline；
3. Router top-$k$ overlap 不低于最佳 importance baseline；
4. 不出现随层深持续放大的异常 hidden-state drift。

阶段 B 不报告下游能力结论，只验证局部指标是否具有模型级预测力。

## 9. 阶段 C：全层等宽压缩模型

阶段 B 通过后，才将同一冻结协议扩展到全部 routed experts，并构造 50% 等宽压缩 checkpoint。低支持专家可使用预注册的保守策略，例如更低补偿 rank 或回退到 importance selection，但必须单独报告比例和影响。

### 9.1 质量指标

至少测量：

- WikiText held-out PPL；
- GSM8K；
- Winogrande；
- 一个代码任务；
- 一个未出现在 calibration 来源中的泛化任务；
- 原模型与压缩模型的 logits KL。

所有 generation 配置、few-shot 数、seed 和样本上限在运行前冻结。先运行小规模 smoke，再运行正式评测。

### 9.2 部署指标

至少报告：

- checkpoint 大小；
- routed-expert 权重显存；
- 单 token 和固定 batch 的峰值显存；
- prefill/decode throughput；
- 相同硬件和 batch/sequence 配置下的 latency；
- 是否使用真正支持缩窄专家的 kernel。

理论 FLOPs 下降不能替代实测延迟。

### 9.3 最终判定

`GO_RAMP_V2` 必须同时满足：

1. 相对最佳 importance + compensation baseline，held-out PPL 增量更小；
2. 预注册下游任务的平均相对退化不超过 2%，且没有单项灾难性退化；
3. logits KL 与阶段 B 的方向一致；
4. checkpoint 或 routed-expert 显存接近理论 50% 宽度收益；
5. 在可用缩窄 kernel 下获得可重复的吞吐或延迟收益。

若局部指标通过但 PPL/logits 不通过，结论为 `NO_TRANSFER_FROM_LOCAL_METRIC`。若质量通过但没有实测部署收益，结论为 `QUALITY_GO_RUNTIME_NO_GO`，不能宣称端到端加速。

## 10. 产物布局

建议新增：

```text
RAMP/experiments/ramp_e1/
├── manifest.json
├── covariances_fit_validation.pt
├── decisions_fit_validation.json
├── covariances_audit.pt
├── expert_audit.json
├── per_expert.csv
├── stage_a_report.md
├── stage_b_model_metrics.json
├── stage_b_report.md
├── stage_c_evalscope_results/
├── stage_c_runtime.json
├── final_report.md
└── logs/
```

每个阶段的 decision 文件保存输入 SHA、代码版本、超参数、候选集合和 `test_metrics_used_for_selection=false`。进入下一阶段前冻结上一阶段产物的 SHA256。

## 11. 推荐执行顺序

1. 扩展 covariance collector，加入 $\sum g^4$、分域 covariance 和 condition diagnostics；
2. 实现统一 selector 接口，先复现 RAMP-E0/RMS/Tail；
3. 实现 `conditional_activation` 和 `conditional_output`，立即在旧 E0 covariance 上做兼容性 smoke；
4. 收集扩大后的 fit/validation statistic，冻结 selector 与补偿超参数；
5. 一次性生成 audit statistic 并完成阶段 A；
6. 只有阶段 A 通过时，构造有限层真实干预模型完成阶段 B；
7. 只有阶段 B 通过时，扩展到全层等宽 checkpoint 和 EvalScope 阶段 C。

该顺序把“选择目标是否更好”“局部指标是否能传递”“最终质量与部署是否成立”拆成三个可独立证伪的问题，避免再次用单一专家重构指标替代完整模型结论。