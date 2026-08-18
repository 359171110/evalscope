# RAMP-E1：补偿对齐通道选择与端到端验证实验

## 1. 实验目的

RAMP-E0 只验证了局部 expert 输出重构，结果显示 RAMP 明显优于随机选择，但与 RMS/Tail 基本持平。RAMP-E1 验证新版方法是否真正解决以下问题：

1. 被删 channel 是否是相对保留激活可预测的输出基函数；
2. 通道选择是否与最终补偿目标一致，而不是只依赖低重要性或高相关性；
3. 条件残差是否跨数据领域、层和专家频率稳定；
4. 局部 expert 重构是否能转化为完整模型 logits、PPL 和下游任务收益。

实验不允许把 audit、PPL test 或下游任务结果用于选择 channel、rank、正则或报告主版本。

实验代号：`RAMP-E1`。

## 2. 新版方法定义

### 2.1 统一选择目标

对每个 expert 和候选保留集合 $\mathcal K$，在 fit split 上求解 ridge compensation：

$$
\Delta W_{\mathcal K}^{*}
=
W_{:\!,\mathcal P}C_{\mathcal P\mathcal K}^{(\mathrm{fit})}
\left(C_{\mathcal K\mathcal K}^{(\mathrm{fit})}+\lambda I\right)^{-1}.
$$

选择 gain 直接定义为 gate-weighted output residual 的下降：

$$
G_j(\mathcal K)
=
\mathcal L_{\mathrm{fit}}(\mathcal K)-
\mathcal L_{\mathrm{fit}}(\mathcal K\cup\{j\}).
$$

在本轮等宽、逐通道选择中，每个 channel 的部署成本相同，因此不额外除以 cost；异构宽度或 group selection 阶段再引入单位部署成本收益。

新版目标使用四个 calibration domain：`wikitext`、`code`、`gsm8k`、`math`：

$$
\mathcal L_{\mathrm{fit}}(\mathcal K)
=
(1-\gamma)\operatorname{mean}_d\mathcal E_d(\mathcal K)
+\gamma\max_d\mathcal E_d(\mathcal K)
+\beta\Omega(\Delta W_{\mathcal K}).
$$

其中：

- $\mathcal E_d$ 是领域 $d$ 的 gate-weighted normalized output error；
- $\Omega$ 同时惩罚 compensation Frobenius ratio 和有效 rank；
- $\gamma$ 防止 specialist domain 被 mixed average 掩盖；
- $\beta$ 防止选择依赖巨大或病态补偿。

首轮固定：

```text
keep width: 384 / 768
anchor_count: 0, 38（两个预注册版本）
gamma: 0.0, 0.25（领域最坏项消融）
beta: 0.0, 0.01, 0.1（补偿复杂度惩罚）
compensation: none, rank-16, rank-32, rank-64, full-rank
alpha: 1e-3, 1e-2, 1e-1, 3e-1, 1, 3, 10
```

`none` 表示不使用补偿，不把它解释为 rank-0 回归。主版本不在 audit 上挑选这些超参数；先在 fit 上分别生成预注册的候选集合，再由 validation 在候选集合、补偿 rank 和正则之间冻结唯一主配置，最后一次性运行 audit。

### 2.2 选择算法

采用 forward selection：

1. 计算 anchor 集合，或从空集合开始；
2. 对每个候选 channel 计算加入后的多领域条件输出残差 gain；
3. 用 Schur-complement / Cholesky 增量更新，避免对每个候选重新求完整逆矩阵；
4. 加入 gain 最大且补偿复杂度约束可接受的 channel；
5. 直到保留 `384` 个 channel；
6. 在 validation 上选择 $\alpha$、$\gamma$ 和 rank，冻结 decision file。

此外保留以下 ablation：

- `importance_rms`：fit-only RMS；
- `importance_tail`：fit-only Tail；
- `corr_only`：只按 activation/output correlation；
- `conditional_output`：新版主方法；
- `conditional_output_no_complexity`：去掉补偿复杂度惩罚；
- `conditional_output_anchor38`：保留 38 个 anchor；
- `random_42/43/44`：随机对照。

所有方法保持相同的最终宽度和补偿求解器。

## 3. 数据划分

### 3.1 局部 calibration

使用已有 mixed train-only token cache，但扩展正式使用量，避免低频 expert 的 fit 样本过少：

| split | 序列数 | 每条长度 | 用途 |
| --- | ---: | ---: | --- |
| fit | 256 | 1024 | 选择 channel、拟合候选补偿 |
| validation | 128 | 1024 | 选择 rank、$\alpha$、$\gamma$ 和复杂度系数 |
| audit | 128 | 1024 | 冻结后的局部重构审计 |

现有 512 条 mixed cache 按原始来源比例冻结为：

| split | WikiText | code | GSM8K | MATH | 总数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fit | 128 | 64 | 32 | 32 | 256 |
| validation | 64 | 32 | 16 | 16 | 128 |
| audit | 64 | 32 | 16 | 16 | 128 |

三部分合计恰好覆盖已有 512 条序列。不得把 evaluation test 混入 fit，也不得在冻结主配置前累计 audit covariance。

每个 split 维持四个 domain 的分层配额，并保存 sequence indices、input IDs SHA256、来源标签和 cache SHA256。

### 3.2 完整模型 PPL

PPL 使用独立冻结的 WikiText-2 raw test token cache，不能与 channel selection 使用同一序列。复用静态剪枝框架的 formal WikiText protocol：

- sequence length `2048`；
- 原模型和所有压缩模型使用完全相同的 token stream；
- 报告 token 数、NLL、PPL 和窗口数；
- PPL test 只用于最终比较，不用于选择。

### 3.3 下游任务

第一轮使用固定的小规模任务：

- `winogrande`：400 samples；
- `gsm8k`：128 samples；
- `humaneval_plus`：完整可用 parquet，若成本过高则预注册 164 samples；
- `mbpp_plus`：完整可用 parquet，若成本过高则预注册 200 samples。

固定 `seed=42`、`max_tokens=1024`、`eval_batch_size=1`，原模型、RAMP-v2 和所有 baseline 使用同一配置。

## 4. 代表模型与导出范围

### 4.1 第一阶段代表专家

沿用 E0 冻结的 24 个 `(layer, physical_expert)`，先验证方法差异，不直接处理全部 `48×128` experts。

### 4.2 端到端小模型范围

为了测 PPL 和下游任务，必须导出一个完整可运行的 compressed checkpoint。E1 端到端主实验使用：

- 全部 48 层、128 experts/layer；
- 所有 routed expert 等宽保留 `384/768`；
- shared expert、router、attention 和 lm_head 保持原样；
- 只替换 routed experts 的 `gate_proj`、`up_proj` 和融合补偿后的 `down_proj`；
- router 不重新训练、不改 Top-k。

24 个代表专家用于先冻结 selection 规则和超参数；之后必须用同一规则应用到所有 experts，再导出完整模型。若完整导出成本过高，先做一个只替换 24 个专家的 debug checkpoint，但它不能作为最终 PPL/下游结论。

对全部 `6,144` 个 routed experts 直接保存四领域完整 `768 x 768` covariance 会产生过大的存储和求解成本。因此全专家阶段采用以下两级策略：

1. 24 个代表专家保存完整分域 covariance，用于验证新版选择目标；
2. 机制层通过后，全专家按 layer 流式处理，每次只保存或求解当前层的统计量，写出 decision 和补偿权重后释放 covariance；
3. 若单层完整统计仍超出资源预算，使用预注册的 block covariance 或低秩 sketch，但必须先在 24 个代表专家上证明其排序与完整 covariance 足够一致。

不得因为资源限制只压缩 24 个专家后就把 PPL 或下游结果解释为 50% 全模型 channel pruning。

## 5. 必须报告的指标

### 5.1 局部 expert 指标

对 24 个代表专家和所有专家分别报告：

$$
\mathcal E_e^{(s)}
=
\frac{\operatorname{tr}(Q C_e^{(s)} Q^\top)}
{\operatorname{tr}(W C_e^{(s)} W^\top)+\epsilon}.
$$

同时报告：

- `R2_pruned`：相对 no-compensation 的误差降低；
- compensation Frobenius ratio；
- effective compensation rank；
- fit/validation/audit gap；
- per-domain error；
- low/medium/high route frequency 分组；
- covariance condition number；
- audit per-token error 的 median/P95/P99。

### 5.2 完整模型 logits 指标

在同一输入上运行原模型和压缩模型，记录：

$$
\mathrm{KL}(p_{\mathrm{dense}}\|p_{\mathrm{compressed}})
$$

以及：

- next-token top-1 agreement；
- logits cosine similarity；
- router top-k overlap；
- 每层 hidden-state cosine similarity；
- shared expert 输出是否保持一致。

这些指标用于连接“局部 expert 误差”和“完整模型行为变化”。

### 5.3 PPL 和下游指标

主表必须同时包含：

| Model | WikiText PPL | Winogrande | GSM8K | HumanEval+ | MBPP+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | baseline | baseline | baseline | baseline | baseline |
| random | - | - | - | - | - |
| RMS | - | - | - | - | - |
| Tail | - | - | - | - | - |
| RAMP-v2 | - | - | - | - | - |

至少报告相对 dense 的变化：

$$
\Delta\mathrm{PPL}=\frac{\mathrm{PPL}_{\mathrm{compressed}}-\mathrm{PPL}_{\mathrm{dense}}}
{\mathrm{PPL}_{\mathrm{dense}}},
$$

以及每个任务的 absolute / relative score drop。

## 6. 预注册判定标准

### 6.1 机制层通过条件

RAMP-v2 只有同时满足以下条件，才进入完整 checkpoint 评测：

1. audit 上 median `R2_pruned >= 0.20`；
2. 相对最佳 RMS/Tail 的 median output-error improvement `>= 10%`；
3. 至少 `18/24` 个代表专家优于最佳 RMS/Tail；
4. 至少 `20/24` 个专家 audit/validation ratio `<= 1.5`；
5. median compensation ratio `<= 0.5`，P90 `<= 1.0`；
6. rank-16 或 rank-32 保留 full-rank 误差下降的至少 `70%`。

如果机制层未通过，仍可导出一个 exploratory checkpoint，但必须标记为诊断结果，不能宣称 RAMP-v2 成功。

### 6.2 模型层通过条件

在机制层通过后，RAMP-v2 还必须满足：

1. WikiText PPL relative degradation `<= 2%`；
2. Winogrande absolute drop `<= 2` percentage points；
3. GSM8K absolute drop `<= 3` percentage points；
4. HumanEval+ 和 MBPP+ relative pass@1 drop `<= 5%`；
5. logits KL 和 router overlap 不出现明显异常，具体阈值在 manifest 中冻结；
6. RAMP-v2 不得在主要任务上系统性劣于 RMS/Tail baseline。

最终只有同时通过机制层和模型层，才能称为 `GO_RAMP_V2`。

## 7. 实验产物

```text
RAMP/experiments/ramp_e1/
  manifest.json
  local/
    covariances_fit.pt
    covariances_validation.pt
    covariances_audit.pt
    decisions.json
    local_results.json
    local_summary.md
  checkpoints/
    dense_reference/
    random_50pct/
    rms_50pct/
    tail_50pct/
    ramp_v2_50pct/
  model_eval/
    dense_logits.jsonl
    compressed_logits.jsonl
    ppl_results.json
    downstream_results.json
    comparison.md
  logs/
```

每个产物保存：模型/数据 SHA256、命令行、随机种子、方法超参数、split indices 和 evaluator 版本。

## 8. 执行顺序

1. 冻结 `manifest.json`，包括 calibration、PPL、下游任务、rank、alpha、gamma、anchor 和判定阈值。
2. 用 256/128/128 序列采集分域 covariance；先不读取 audit。
3. 在 fit/validation 上运行 `conditional_output`、RMS、Tail、correlation-only 和 random 对照。
4. 生成冻结 `decisions.json`，锁定所有 experts 的 channel、rank、alpha 和补偿系数。
5. 一次性采集 audit covariance 和 per-token 输出误差。
6. 若机制层通过，使用相同决策规则对全部 experts 导出完整 compressed checkpoint。
7. 在固定 WikiText test 上测 PPL，再测 logits KL、router overlap 和下游任务。
8. 生成最终比较表，按“局部机制结果”和“模型级结果”分别给出结论。

## 9. 结果解释边界

- 局部 reconstruction error 低，只说明 expert 输出近似较好，不等于 PPL 或任务保持。
- PPL 保持但某个 specialist task 下降，说明 mixed calibration 目标不够，应提高 worst-domain 权重或增加领域约束。
- RAMP-v2 局部优于 RMS/Tail，但模型级劣于 baseline，说明 selection objective 与模型损失仍未对齐。
- 只有完整模型 PPL、下游任务、logits 和部署指标都通过，才进入异构宽度和真实 latency 实验。