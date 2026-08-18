# 实验结果：AIMER + ESP10（已完成）

## 当前决策

ESP10 已完成 B6/B9 Quick9。两个预算的 Macro 都低于冻结的 AIMER + PP10 baseline，因此按 ESP 文档中的停止条件停止 ESP10，不运行 PP5+ESP5、不加入 eigenvalue weighting、不加入 $W_{down}$，也不调整 spectral rank。

正式结果如下：

| 预算 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESP10 B6 | 0.8116 | 0.4430 | 0.5225 | 0.8047 | 0.5100 | 0.5807 | 0.6121 |
| ESP10 B9 | 0.9584 | 0.7040 | 0.7225 | 0.9375 | 0.8200 | 0.8298 | 0.8287 |
与冻结 AIMER + PP10 baseline 对比：
| 预算 | AIMER + PP10 | ESP10 | 差值 |
|---|---:|---:|---:|
| B6 | 0.6163 | 0.6121 | -0.0042 |
| B9 | 0.8403 | 0.8287 | -0.0116 |
| 两预算平均 | 0.7283 | 0.7204 | -0.0078 |

逐任务差值显示，ESP B6 仅 GSM8K 高于 baseline，ARC、HellaSwag、WinoGrande、MATH-500 和 MMLU 均下降；ESP B9 的 ARC 略高于 baseline，但 HellaSwag、WinoGrande、GSM8K、MATH-500 和 MMLU 均下降。ESP10 没有形成稳定的下游收益。
构建诊断覆盖全部 $48\times128=6144$ 个 experts：
| 诊断 | Mean | P10 | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| PP10 probe overlap | 0.2290 | 0.1039 | 0.2078 | 0.3766 | 0.0130 | 0.7922 |
| spectral concentration | 0.0486 | 0.0307 | 0.0479 | 0.0654 | 0.0197 | 0.1788 |
| smallest selected eigenvalue | 6.5995 | 4.3661 | 6.4526 | 8.7565 | 2.7646 | 20.3224 |

profile 与 ranking 审计确认：四个 ESP/PWRP 变体共用同一原始模型来源，ESP B6/B9 最终 profile 分别覆盖 6144 个宽度 6/9 的 expert；ESP B6/B9 ranking SHA256 分别为 `40c37de769297d129a17e80f351fef0579b44dbf5204b7347d1e5b847154df43` 和 `6d7f84164e7b9589a2fd960e8e2c2fe9ee1cb79a29b8c1498be73d380e6aa218`。每个 expert 的 ranking 都是 0..767 的完整置换，导出 manifest 分别声明 384/576 retained channels。

评测审计确认 B6/B9 各存在且仅存在 6 份任务 JSON，样本数依次为 `600/1000/400/128/100/570`；Macro 独立复算为 `0.6120833/0.8287000`。结果目录为：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERESP-PPFv1-G10-B6of12_202608080130_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERESP-PPFv1-G10-B9of12_202608080130_42/`

## ESP 协议结论

ESP 的输入侧 spectral probes、router sign orientation 和 SwiGLU Q4 聚合实现已通过 focused tests 与构建审计，但本轮下游结果不支持其替代 PP10。实验到此关闭，不启动 Hybrid 或额外 sweep。

# 实验结果：AIMER + PWRP10（已完成）

## 当前决策

PWRP10 已完成 B6/B9 Quick9。两个预算的 Macro 都低于冻结的 AIMER + PP10 baseline，因此按 PWRP 文档中的停止条件停止 PWRP10，不运行 PP5+PWRP5、不运行 Hybrid，也不做 previous-source filtering、down-column weighting 或 probe/budget sweep。

正式结果如下：
| 预算 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| PWRP10 B6 | 0.7850 | 0.4550 | 0.5600 | 0.7969 | 0.4600 | 0.5175 | 0.5957 |
| PWRP10 B9 | 0.9600 | 0.7040 | 0.7175 | 0.9688 | 0.8500 | 0.8228 | 0.8372 |
与冻结 AIMER + PP10 baseline 对比：
| 预算 | AIMER + PP10 | PWRP10 | 差值 |
|---|---:|---:|---:|
| B6 | 0.6163 | 0.5957 | -0.0205 |
| B9 | 0.8403 | 0.8372 | -0.0031 |
| 两预算平均 | 0.7283 | 0.7165 | -0.0118 |
逐任务差值显示，PWRP B6 仅 WinoGrande 高于 baseline，PWRP B9 的 ARC、GSM8K 和 MATH-500 高于 baseline，但 HellaSwag、WinoGrande 和 MMLU 下降；B6 的 ARC、HellaSwag、GSM8K、MATH-500 和 MMLU 均下降。PWRP 的 B9 正向任务信号不足以抵消两个预算的 Macro 下降。
构建诊断覆盖全部 $48\times128=6144$ 个 experts：
| 诊断 | Mean | P10 | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| PP10 probe overlap | 0.3431 | 0.1948 | 0.3247 | 0.4805 | 0.0390 | 1.0000 |
| selected affinity mean | 9.9653 | 7.5431 | 9.7783 | 12.9586 | 0.0000 | 25.6042 |
| fallback to PP10 | 0.0208 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

`fallback_pp` 的均值为 0.0208，符合仅第一个 MoE layer 使用 PP10 fallback 的设计；没有发现 fallback 逻辑异常。PWRP B6/B9 最终 profile 分别覆盖 6144 个宽度 6/9 的 expert；ranking SHA256 分别为 `74d8be59b29e5f50bc8681e4aca19d12e00ecf637af671fb19b986b80f99603b` 和 `be161dc06d65419c9edb73466ab6df0c67c76b583dffe9f73f7632665e3d8f75`。每个 expert 的 ranking 都是 0..767 的完整置换，导出 manifest 分别声明 384/576 retained channels。

评测审计确认 B6/B9 各存在且仅存在 6 份任务 JSON，样本数依次为 `600/1000/400/128/100/570`；Macro 独立复算为 `0.5957333/0.8371833`。结果目录为：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERPWRP-PPFv1-G10-B6of12_202608080130_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERPWRP-PPFv1-G10-B9of12_202608080130_42/`

## PWRP 协议结论

PWRP 的原始 previous-layer down-column candidate、绝对 router affinity、target-oriented sign 和首层 PP fallback 均已通过 focused tests 与构建审计，但本轮没有产生足够的正向信号，不进入 Hybrid 或其他扩展。

# 实验结果：AIMER + PP + Local Triad Removal Energy（已完成）

## 当前决策

Local TRE 已完成 B6/B9 Quick9。B6 Macro 小幅高于冻结 AIMER + PPFv1，B9 Macro 小幅下降，属于明确的预算依赖结果。按预先冻结的停止条件，本轮记录 B6 正增益，但不把 TRE 提升为通用 AIMER boundary tie-breaker；实验到此关闭，不调整 $T$，不做 energy normalization、layer scaling、boundary sweep 或全局 TRE ranking。

正式结果如下：

| 预算 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local TRE B6 | 0.8433 | 0.5470 | 0.5525 | 0.7812 | 0.4600 | 0.5439 | 0.6213 |
| Local TRE B9 | 0.9583 | 0.7150 | 0.7325 | 0.9453 | 0.8500 | 0.8298 | 0.8385 |

与冻结 AIMER + PPFv1 baseline 对比：

| 预算 | AIMER + PPFv1 | Local TRE | 差值 |
|---|---:|---:|---:|
| B6 | 0.6163 | 0.6213 | +0.0051 |
| B9 | 0.8403 | 0.8385 | -0.0018 |
| 两预算平均 | 0.7283 | 0.7299 | +0.0016 |

逐任务差值显示，TRE 在两个预算中都提高了 HellaSwag、WinoGrande 和 MMLU，但同时降低 GSM8K 和 MATH-500；B6 的 MATH-500 下降 0.0600，B9 下降 0.0300。因此当前结果不是稳定的整体质量提升，而是 boundary replacement 对任务类型敏感。B6 的正 Macro 证据可以保留，但不足以覆盖 B9 退化并支持默认采用。

构建与结果审计已覆盖全部 $48\times128=6144$ 个 experts：

| 预算 | AIMER overlap Mean | P10 | Median | P90 | Replacement Mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| B6 | 0.950683 | 0.942708 | 0.950521 | 0.958333 | 18.9378 | 11 | 29 |
| B9 | 0.967070 | 0.961806 | 0.967014 | 0.972222 | 18.9678 | 11 | 27 |

审计确认：

- 精确边界宽度为 $T=38$，每个 expert 的 replacement 均不超过 38。
- B6/B9 每个 expert 分别恰好保留 384/576 个唯一 channel。
- 全部 77 个 PPFv1 channel 在所有 experts 中保持保留。
- AIMER 高置信冻结区全部保持不变，所有 retained-set 变化只发生在 76-channel cutoff pool 内。
- AIMER cache SHA256 为 `266646957cddf9b91645d50f247c6ff79ef11773b464e04d2e4432698fd3158c`。
- PP cache SHA256 为 `0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- B6 ranking SHA256 为 `b97bcb2e8a55969525aa909faa3f3df9e982e3358489fc546e1d0b15bb6c1500`；diagnostics SHA256 为 `bb2accd9d90715ba48afb73a0b69341332744070478061814a47a30965fac66d`。
- B9 ranking SHA256 为 `65e094b9c401a5927b22b27911ef3dc5d9cf098f29068a089f47f1d1070ca5d9`；diagnostics SHA256 为 `27d49f183ec2912af1ec46f2a31d64cd8705f5525d80c3ae24f28c207066be35`。
- 每个预算均存在且仅存在 6 份任务 JSON 报告；Macro 独立复算为 0.6213167/0.8384833。

结果目录：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERLocalTRE-PPFv1-G10-T5-B6of12_202608072208_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERLocalTRE-PPFv1-G10-T5-B9of12_202608072208_42/`

## 当前问题

WeightMoment 已证明 parameter norm 不能接管全局 channel ranking；Local-BFC 也证明，即使只在 cutoff 附近修正，bilinear diversity 仍不足以改善 AIMER。本轮不恢复 WeightMoment，也不改变 AIMER 的全局判断，只验证一个更弱的问题：

$$
\boxed{\text{在 AIMER 已认为接近的 cutoff channel 中，TRE 能否优先删除参数函数变化更小的 channel？}}
$$

Triad Removal Energy 只决定 pruning boundary 附近的 tie-breaking。PP、AIMER 高置信区域和边界外排序完全冻结。

## Triad Removal Energy

一个 SwiGLU channel 由 $(g_j,u_j,d_j)$ 构成。在 SiLU 局部近似下：

$$
h_j(x)\approx\frac{1}{2}(g_j^\top x)(u_j^\top x),
$$

$$
f_j(x)\approx\frac{1}{2}d_j(g_j^\top x)(u_j^\top x).
$$

考虑 $g/u$ 对称性，对应的 parameterized functional operator 可写为：

$$
T_j=d_j\otimes\frac{g_j\otimes u_j+u_j\otimes g_j}{2}.
$$

删除 channel $j$ 时，忽略不影响排序的公共常数，使用：

$$
\boxed{
E_j^{\mathrm{TRE}}
=\|d_j\|_2^2
\left[
\|g_j\|_2^2\|u_j\|_2^2+(g_j^\top u_j)^2
\right]
}.
$$

该公式与失败的 WeightMoment 可能形式相近，但用途严格不同：WeightMoment 曾尝试用近似 functional energy 对整个 expert 全局排序；TRE 只在 AIMER cutoff 两侧的窄边界池中选择，不允许接管 global ranking。

## 固定协议

- 模型、Quick9、seed 42 和生成设置保持不变。
- PP 使用 `PP-Frozen-v1`：positive-only、K8、Q4、NoDownNorm。
- PP cache SHA256 固定为 `0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- AIMER 使用冻结 channel-level cache，不修改 AIMER 公式或全局顺序。
- $D=768$，$P=\operatorname{round}(0.1D)=77$。
- 边界宽度固定为 $T=\operatorname{round}(0.05D)=38$，不做 width sweep。
- 只运行 B6 和 B9，不运行 G0。
- 不使用 calibration data，不修改权重，不做 output reconstruction。

## Local TRE Boundary

设目标保留数为 $M$，非 PP 保留预算为：

$$
R=M-P.
$$

先从 AIMER 顺序中移除 PP channel，得到：

$$
j_1,j_2,\ldots,j_{D-P}.
$$

冻结集合为：

$$
\mathcal F=\mathcal P\cup\{j_1,\ldots,j_{R-T}\}.
$$

只构造 cutoff 两侧的边界池：

$$
\mathcal B_{\mathrm{in}}=\{j_{R-T+1},\ldots,j_R\},
$$

$$
\mathcal B_{\mathrm{out}}=\{j_{R+1},\ldots,j_{R+T}\}.
$$

从 $|\mathcal B_{\mathrm{in}}\cup\mathcal B_{\mathrm{out}}|=2T$ 个 channel 中，按 $E_j^{\mathrm{TRE}}$ 从高到低选择 $T$ 个保留。能量并列时按原始 AIMER 顺序稳定打破平局。最终集合为：

$$
S_{\mathrm{TRE}}=\mathcal F\cup\operatorname{TopT}_{j\in\mathcal B_{\mathrm{in}}\cup\mathcal B_{\mathrm{out}}}E_j^{\mathrm{TRE}}.
$$

因此每个 expert 最多替换 $T=38$ 个 channel，且与 AIMER+PP baseline 至少重合 $M-T$ 个 channel。

## 预算

| 预算 | $M$ | $P$ | $R$ | $T$ | 冻结的非 PP AIMER | 边界池 | TRE 选择 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B6 | 384 | 77 | 307 | 38 | 269 | 76 | 38 |
| B9 | 576 | 77 | 499 | 38 | 461 | 76 | 38 |

实验命名：

- `AIMERLocalTRE-PPFv1-G10-T5-B6of12`
- `AIMERLocalTRE-PPFv1-G10-T5-B9of12`

## 构建诊断

对全部 $48\times128=6144$ 个 experts 记录并汇总均值与分位数：

1. 与 AIMER+PP baseline 的 retained-set overlap。
2. 边界内侧 channel 的保留比例。
3. 实际 replacement 数量，范围必须为 $[0,T]$。
4. 原 AIMER 边界内侧、TRE 选择边界和完整边界池的平均 TRE energy。
5. ranking cache SHA256、PP SHA256、宽度 histogram 和每个 expert 的 channel 唯一性。

## 停止条件

- B6/B9 均不低于 AIMER+PP，且至少一个预算提升：TRE 可保留为 AIMER boundary tie-breaker。
- 仅一个预算提升：记录预算依赖，不调 $T$，不自动扩展方法。
- 两个预算均下降：停止 TRE，不做 energy normalization、layer scaling、boundary sweep 或全局 TRE ranking。
- 任一构建不变量失败：不评测，先修复构建。

---

# 实验结果：AIMER + PP + Data-Free Output Reconstruction（已完成）

## 当前决策

Fixed-mask Data-Free Output Reconstruction 已完成 B6/B9 Quick9。Router-derived pseudo space 中的删除输出几乎被完全重建，但两个预算的下游质量都低于冻结 AIMER+PP baseline，B6 出现严重退化。按预先冻结的停止条件，停止当前 pseudo reconstruction objective，不进入 reconstruction-aware boundary selection，也不调整 ridge、probe bank 或 selection。

本轮固定 AIMER+PP mask，不替换任何 channel，只验证：

$$
\boxed{\text{删除通道造成的 expert output，能否在 data-free probe space 中折叠进保留通道？}}
$$

正式结果如下：

| 预算 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reconstruction B6 | 0.6267 | 0.3900 | 0.4825 | 0.3594 | 0.1300 | 0.3860 | 0.3958 |
| Reconstruction B9 | 0.9550 | 0.6790 | 0.6675 | 0.9531 | 0.8400 | 0.8123 | 0.8178 |

与冻结 AIMER+PP baseline 对比：

| 预算 | AIMER + PPFv1 | Reconstruction | 差值 |
|---|---:|---:|---:|
| B6 | 0.6163 | 0.3958 | -0.2205 |
| B9 | 0.8403 | 0.8178 | -0.0225 |
| 两预算平均 | 0.7283 | 0.6068 | -0.1215 |

构建诊断覆盖全部 $48\times128=6144$ 个 experts。均值与分位数如下：

| 预算 | 诊断 | Mean | P10 | Median | P90 |
|---|---|---:|---:|---:|---:|
| B6 | $E_{\mathrm{before}}$ | 0.428095 | 0.262270 | 0.450739 | 0.555877 |
| B6 | $E_{\mathrm{after}}$ | 0.000774 | 0.000298 | 0.000471 | 0.001435 |
| B6 | $R_{\mathrm{rec}}$ | 0.997332 | 0.994883 | 0.998951 | 0.999436 |
| B6 | $\|\Delta\|_F/\|D_S\|_F$ | 0.695935 | 0.612995 | 0.679379 | 0.784319 |
| B9 | $E_{\mathrm{before}}$ | 0.294330 | 0.180605 | 0.306869 | 0.386194 |
| B9 | $E_{\mathrm{after}}$ | 0.000278 | 0.000138 | 0.000197 | 0.000526 |
| B9 | $R_{\mathrm{rec}}$ | 0.998523 | 0.997284 | 0.999361 | 0.999617 |
| B9 | $\|\Delta\|_F/\|D_S\|_F$ | 0.298584 | 0.266219 | 0.295171 | 0.330759 |

该结果直接触发停止条件：pseudo-space 恢复率高达 99.73%/99.85%，却未转化为下游恢复。尤其 B6 的 retained down 补偿幅度很大，且 Macro 下降 22.05 个百分点，说明 Router-row probe space 上的可重建性不能作为真实输入分布上的可靠补偿目标。当前证据不支持继续调 ridge 或把该目标纳入 channel selection；由于 reconstruction 未超过 AIMER+PP，compensation-aware selection 的前置条件未满足，不启动该后续实验。

产物审计已确认：每个预算均有 6 份任务报告；B6/B9 Macro 独立复算为 0.3957667/0.8178167；diagnostics 均包含不重不漏的 6144 个 `(layer_id, expert_id)`。导出 manifest 的 fixed-mask 标记、唯一修改项 `down_proj.weight`、ranking SHA256 和 diagnostics SHA256 均与磁盘产物一致。进一步逐张量比较 reconstruction 与对应 AIMER baseline checkpoint：B6/B9 各自 6144 个 `gate_proj` 和 6144 个 `up_proj` 全部完全相等，各自 6144 个 `down_proj` 全部发生补偿变化，确认零 channel replacement 和 down-only mutation：

- B6 ranking SHA256：`4e3469b56e4b87dc8f2504515e881296adcdb5aceaf3c6d67dd6c2ce2e46c6d6`；diagnostics SHA256：`2f64757b7e68d4fe82519dc2ed025e801c4692807cd8d9febff77318bff1675f`。
- B9 ranking SHA256：`e6dee7cfb3fa6b61b4f3e83f8c50b9a3c556119e7e9b645ccce32f30ccb78d06`；diagnostics SHA256：`d8fd53b5df1be50c9a9a58433f1e363cd05788442861826f55c0110f169a1bee`。

结果目录：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERRecon-PPFv1-G10-B6of12_202608072114_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERRecon-PPFv1-G10-B9of12_202608072114_42/`

## 冻结协议

- 模型、Quick9、seed 42 和生成设置保持不变。
- PP 使用 `PP-Frozen-v1`：positive-only、K8、Q4、NoDownNorm。
- PP cache SHA256 固定为 `0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- AIMER+PP 的 B6/B9 rankings 和最终 mask 完全不变。
- 只运行 B6 和 B9，不运行 G0。
- 不修改 `gate_proj`、`up_proj`，只修改 retained `down_proj`。
- 无训练、无 calibration data，不做 channel replacement。
- reconstruction probe bank 使用该层全部 $N=128$ 个 RMSNorm Router rows，不使用 PP 的局部 K+1 probe 子集。

实验命名：

- `AIMERRecon-PPFv1-G10-B6of12`
- `AIMERRecon-PPFv1-G10-B9of12`

## Data-Free Output Reconstruction

对每层 Router row $r_i$ 构造：

$$
p_i=\operatorname{RMSNorm}(r_i),\qquad X^R=\{p_1,\ldots,p_N\}.
$$

对目标 expert 计算完整 SwiGLU response：

$$
H[i,j]=\operatorname{SiLU}(g_j^\top p_i)(u_j^\top p_i),\qquad H\in\mathbb R^{N\times D}.
$$

按冻结 mask 分为 retained 集合 $S$ 和 pruned 集合 $P$：

$$
H_S\in\mathbb R^{N\times|S|},\qquad H_P\in\mathbb R^{N\times|P|},
$$

$$
D_S=W_{\mathrm{down}}[:,S],\qquad D_P=W_{\mathrm{down}}[:,P].
$$

删除通道造成的输出损失为：

$$
Y_{\mathrm{lost}}=H_PD_P^\top.
$$

求解：

$$
\min_{\Delta}\|H_S\Delta-Y_{\mathrm{lost}}\|_F^2+\lambda\|\Delta\|_F^2,
$$

其中 $\Delta\in\mathbb R^{|S|\times d_{\mathrm{model}}}$。使用 dual form：

$$
\Delta=H_S^\top(H_SH_S^\top+\lambda I_N)^{-1}Y_{\mathrm{lost}}.
$$

最终仅替换：

$$
D'_S=D_S+\Delta^\top.
$$

第一版 ridge 固定为：

$$
\boxed{\lambda=10^{-4}\frac{\operatorname{tr}(H_SH_S^\top)}{N}}.
$$

## 构建时诊断

每个 expert 记录：

$$
E_{\mathrm{before}}=\frac{\|H_PD_P^\top\|_F}{\|HD^\top\|_F},
$$

$$
E_{\mathrm{after}}=\frac{\|H_PD_P^\top-H_S\Delta\|_F}{\|HD^\top\|_F},
$$

$$
R_{\mathrm{rec}}=1-\frac{E_{\mathrm{after}}}{E_{\mathrm{before}}}.
$$

同时记录 $\|\Delta\|_F/\|D_S\|_F$ 和实际 $\lambda$，再对 $48\times128=6144$ 个 experts 汇总均值与分位数。诊断不需要额外下游 evaluation。

## 停止条件

- Quick9 优于 AIMER+PP：保留 fixed-mask reconstruction，并考虑 reconstruction-aware boundary selection。
- pseudo-space 恢复率高但 Quick9 下降：停止 Router-derived pseudo reconstruction objective。
- pseudo-space 恢复率低：当前 probe bank 不足以表达删除输出，不调 selection，先停止本方案。

---

# 实验结果：AIMER + PP + Local-BFC（已完成）

## 结论与下一问题

Local-BFC 已完成 B6/B9 Quick9。正式结果如下：

| 方法 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local-BFC B6 | 0.8334 | 0.4940 | 0.5125 | 0.7266 | 0.4200 | 0.5544 | 0.5902 |
| Local-BFC B9 | 0.9600 | 0.6960 | 0.7200 | 0.9219 | 0.8100 | 0.8246 | 0.8221 |

与冻结 AIMER + PPFv1 baseline 对比：

| 预算 | AIMER + PPFv1 | Local-BFC | 差值 |
|---|---:|---:|---:|
| B6 | 0.6163 | 0.5902 | -0.0262 |
| B9 | 0.8403 | 0.8221 | -0.0182 |
| 两预算平均 | 0.7283 | 0.7061 | -0.0222 |

Local-BFC 比 Global-BFC 明显恢复，但仍未超过 AIMER。按预先冻结的停止条件，停止 bilinear coverage、down-direction BFC 和 conditional BFC，不再调 boundary width 或 BFC similarity。下一阶段应转向 parameter-removal error、weight reconstruction 或 compensation-aware pruning。

构建诊断覆盖全部 $48\times128=6144$ 个 experts：

| 预算 | 方法 | AIMER overlap 均值 | AIMER rank 均值 | 正相关冗余均值 |
|---|---|---:|---:|---:|
| B6 | AIMER | 1.0000 | 208.5991 | 0.000539 |
| B6 | Global-BFC | 0.6684 | 304.4723 | 0.000437 |
| B6 | Local-BFC | 0.8977 | 217.2943 | 0.000514 |
| B9 | AIMER | 1.0000 | 291.1094 | 0.000502 |
| B9 | Global-BFC | 0.8528 | 345.6859 | 0.000455 |
| B9 | Local-BFC | 0.9321 | 296.8219 | 0.000489 |

Local-BFC 确实把扰动限制在 cutoff 附近，也只小幅降低正相关冗余；但这仍带来 B6/B9 一致退化。因此当前证据不支持把该 bilinear diversity 作为 AIMER boundary tie-breaker。

结果目录：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERLocalBFC-PPFv1-G10-B6of12_202608072045_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERLocalBFC-PPFv1-G10-B9of12_202608072045_42/`

---

## 实验动机

Global-BFC 已完成并明确失败：

| 方法 | B6 Macro | B9 Macro |
|---|---:|---:|
| AIMER + PPFv1 | 0.6163 | 0.8403 |
| AIMER + PPFv1 + Global-BFC | 0.3757 | 0.7464 |
| 差值 | -0.2406 | -0.0939 |

Global-BFC 不是轻量 redundancy correction。B6 中它决定 307/384 个最终位置，B9 中决定 499/576 个最终位置，实际上大范围接管了 AIMER 排序。当前结果证明：归一化 bilinear diversity 不能替代 AIMER importance，但尚未证明它不能在 AIMER pruning cutoff 附近作为局部修正。

本轮只验证：

$$
\boxed{\text{BFC 能否作为 AIMER pruning boundary 附近的 tie-breaker？}}
$$

禁止加入 `down_proj`、RMSNorm covariance、补偿、conditional residual，禁止调整 PP、boundary width 或 BFC similarity。

## 固定协议

- 模型、Quick9、seed 42 和生成设置保持不变。
- PP 使用 `PP-Frozen-v1`：positive-only、K8、Q4、NoDownNorm。
- PP cache SHA256 固定为 `0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- $D=768$，$P=T=\operatorname{round}(0.1D)=77$。
- 只运行 B6 和 B9，不运行 G0。
- AIMER 使用现有 channel-level cache，不修改公式。
- BFC 使用 $\Sigma=I$、正相关截断 `max(C, 0)`，不使用 `down_proj`。

## Local-BFC 定义

对每个 expert 独立操作。设目标保留数为 $M$，非 PP 保留预算为：

$$
R=M-P.
$$

去除 PP 通道后，按 AIMER 从高到低得到：

$$
j_1,j_2,\ldots,j_{D-P}.
$$

AIMER + PP baseline 为：

$$
S_{\mathrm{AIMER}}=\mathcal P\cup\{j_1,\ldots,j_R\}.
$$

冻结集合为：

$$
\boxed{\mathcal F=\mathcal P\cup\{j_1,\ldots,j_{R-T}\}}.
$$

边界内侧和外侧分别为：

$$
\mathcal B_{\mathrm{in}}=\{j_{R-T+1},\ldots,j_R\},
$$

$$
\mathcal B_{\mathrm{out}}=\{j_{R+1},\ldots,j_{R+T}\}.
$$

只允许从大小为 $2T=154$ 的边界池：

$$
\mathcal B=\mathcal B_{\mathrm{in}}\cup\mathcal B_{\mathrm{out}}
$$

中重新选择 $T=77$ 个通道。PP 和 AIMER 前 $R-T$ 个非 PP 通道不可移动，因此 Local-BFC 与 baseline 至少重合 $M-T$ 个通道。

初始化 $S=\mathcal F$。对 $j\in\mathcal B$，使用现有 bilinear similarity：

$$
C_{ij}^{+}=\max\left(
\frac{(g_i^\top g_j)(u_i^\top u_j)+(g_i^\top u_j)(u_i^\top g_j)}
{\sqrt{K_{ii}K_{jj}}+\epsilon},0\right).
$$

每步选择：

$$
j^*=\arg\min_{j\in\mathcal B\setminus S}\max_{i\in S}C_{ij}^{+},
$$

直到从边界池选择了 $T$ 个通道，最终 $|S|=M$。novelty 相同时按 AIMER 顺序稳定打破平局。

## 预算

| 预算 | $M$ | $P=T$ | $R$ | 冻结的非 PP AIMER | 边界池 | BFC 选择 |
|---|---:|---:|---:|---:|---:|---:|
| B6 | 384 | 77 | 307 | 230 | 154 | 77 |
| B9 | 576 | 77 | 499 | 422 | 154 | 77 |

实验命名：

- `AIMERLocalBFC-PPFv1-G10-B6of12`
- `AIMERLocalBFC-PPFv1-G10-B9of12`

## 构建时诊断

无需额外 evaluation，在每个 expert 上比较 AIMER baseline、Global-BFC 和 Local-BFC：

1. 与 AIMER baseline 的 overlap：$|S\cap S_{\mathrm{AIMER}}|/M$。
2. 最终集合在原始全通道 AIMER 排名中的平均 1-based rank。
3. 集合内正相关 BFC similarity 的非对角平均值：

$$
\bar C(S)=\frac{2}{M(M-1)}\sum_{i<j}\max(C_{ij},0).
$$

诊断先按 expert 计算，再报告 48 层 × 128 experts 的均值和分位数。如果 Global-BFC 明显降低 $\bar C$、同时降低 overlap 并选择更差的 AIMER rank，而准确率大幅下降，则说明优化出的数学 diversity 与模型质量不一致。

## 停止条件

- Local-BFC 提升：后续方法仍必须保持 AIMER-anchored local correction，再考虑其他信息。
- Local-BFC 仍下降：停止 bilinear coverage、down-direction BFC 和 conditional BFC，转向 parameter-removal error、weight reconstruction 或 compensation-aware pruning。

---

# 历史方案：Global-BFC（已淘汰）

可以。下一轮实验我建议只验证一个问题：

[
\boxed{\text{在 AIMER+PP 基础上，显式去除功能冗余是否有收益？}}
]

因此这一轮**不要加入 `down_proj`、RMSNorm covariance、补偿、conditional residual，也不要重新调 PP**。只新增 Bilinear Functional Coverage，保证实验结论干净。

## 一、固定设置

对每个 expert 独立操作。设原始中间宽度为 (D)，最终保留通道数为

[
M=B\cdot D.
]

例如：

[
B6=\frac{6}{12}=50%,\qquad
B9=\frac{9}{12}=75%.
]

PP 完全使用你当前已经确定的版本：

[
s_j^{PP}
========

\operatorname{MeanTopQ}_{p}
\left|
\operatorname{SiLU}(g_j^\top p)
(u_j^\top p)
\right|,
]

**不乘 `down_proj` 列范数**。

固定当前最优 (K,Q)，并先固定

[
G=10%.
]

所以每个 expert 的 PP 保护集合大小为：

[
P=|\mathcal P|=0.1D.
]

PP 通道永远不能被 Coverage 删除。

AIMER 也**完全使用你当前已有的 channel-level AIMER 实现**，不要修改公式。

---

# 二、实验 Baseline

首先继续保留现有：

[
\boxed{\text{AIMER+PP}}
]

具体就是：

1. PP 保护 Top-(G)：
   [
   \mathcal P.
   ]

2. 在剩余 (D-P) 个通道中按 AIMER 排序。

3. 选择最高的：

[
M-P
]

个。

最终：

[
\mathcal S_{\text{baseline}}
============================

\mathcal P
\cup
\operatorname{Top}_{M-P}(\mathrm{AIMER}).
]

这个就是你现在的 0.6064 / 0.8364 baseline。

---

# 三、新方法的核心：AIMER 只负责筛候选

Coverage 不能直接面对全部 channel，否则容易选择“非常独特但实际上没什么价值”的通道。

所以 AIMER 的职责改成：

[
\boxed{\text{先保证候选通道基本重要}}
]

而不是直接决定最终全部保留集合。

定义：

[
R=M-P
]

为 PP 之外还需要保留的通道数。

我们给 Coverage 多提供一部分候选通道。

第一轮固定：

[
\boxed{
C=R+\frac12(D-M)
}
]

其中 (C) 是**非 PP candidate 数量**。

解释非常直观：

* (R)：最终一定需要从非 PP 中留下的数量；
* (D-M)：整个 expert 最终会被删掉的数量；
* 再额外给 Coverage 一半的“待删通道”作为候选，让它有机会纠正 AIMER。

于是：

[
\mathcal C
==========

\operatorname{Top}_{C}
\left(
s_j^{AIMER},
j\notin\mathcal P
\right).
]

第一轮**不要调这个 (1/2)**。

---

## 对你的两个预算具体是多少？

### B6

[
M=0.5D,\qquad P=0.1D
]

所以：

[
R=0.4D.
]

候选：

[
C
=

# 0.4D+\frac12(0.5D)

0.65D.
]

因此：

* PP 固定保护 (0.10D)；
* AIMER 从剩余通道中筛出 (0.65D) candidate；
* Coverage 最终从这 (0.65D) 中选 (0.40D)。

最终总保留：

[
0.10D+0.40D=0.50D.
]

---

### B9

[
M=0.75D,\qquad P=0.1D
]

所以：

[
R=0.65D.
]

候选：

[
C
=

# 0.65D+\frac12(0.25D)

0.775D.
]

因此：

* PP：(0.10D)；
* AIMER candidate：(0.775D)；
* Coverage 从中选 (0.65D)。

最终：

[
0.10D+0.65D=0.75D.
]

这个 candidate 定义比简单的 `Top-2M` 更适合你的 B6/B9，因为 B9 本来就保留 75%，`2M` 会直接覆盖整个 expert，AIMER screening 就失去意义。

---

# 四、如何构造 Bilinear Functional Similarity

现在只使用：

[
W_g,\qquad W_{up}.
]

**完全不使用 `down_proj`。**

对候选 channel (j)：

[
g_j=W_g[j,:],\qquad
u_j=W_{up}[j,:].
]

SwiGLU channel：

[
h_j(x)
======

\operatorname{SiLU}(g_j^\top x)
(u_j^\top x).
]

利用局部近似：

[
\operatorname{SiLU}(z)\approx \frac12z,
]

得到：

[
h_j(x)
\approx
\frac12(g_j^\top x)(u_j^\top x).
]

第一轮直接假设各输入方向等权：

[
\boxed{\Sigma=I}
]

不要再加入 RMSNorm-aware (\Sigma)，因为 WeightMoment 已经失败，这一轮只想验证 redundancy。

---

## 两个 channel 的 bilinear covariance

对于 channel (i,j)：

[
\boxed{
K_{ij}
======

(g_i^\top g_j)(u_i^\top u_j)
+
(g_i^\top u_j)(u_i^\top g_j)
}
]

对角：

[
K_{ii}
======

|g_i|^2|u_i|^2
+
(g_i^\top u_i)^2.
]

然后归一化：

[
\boxed{
C_{ij}
======

\frac{K_{ij}}
{\sqrt{K_{ii}K_{jj}}+\epsilon}
}
]

它可以理解为两个 channel 的近似 SwiGLU 功能相关性。

---

# 五、这里我建议不要取绝对值

这一点我修正前面的建议。

第一版使用：

[
\boxed{
C_{ij}^{+}
==========

\max(C_{ij},0)
}
]

而不是：

[
|C_{ij}|.
]

原因是：

如果：

[
C_{ij}\approx1,
]

说明两个 channel 在 bilinear function 上同向高度相似，可以视为冗余。

但如果：

[
C_{ij}\approx-1,
]

说明它们的 hidden response 近乎反向。

在我们**还没有使用 `down_proj` 方向**的情况下，不能认为：

[
h_i\approx-h_j
]

也是冗余。

只有未来加入 `down_proj` 后，如果同时：

[
d_i\approx-d_j,
]

它们的最终输出才可能真正冗余。

所以这一轮：

[
\boxed{
\operatorname{sim}(i,j)=\max(C_{ij},0)
}
]

最稳妥。

---

# 六、Coverage 怎么选择？

初始化：

[
\boxed{S=\mathcal P}
]

即 PP protected channels 直接作为已经存在的 basis。

对每个 AIMER candidate：

[
j\in\mathcal C\setminus S
]

计算它与当前保留集合最相似的程度：

[
r_j(S)
======

\max_{i\in S}\operatorname{sim}(i,j).
]

定义 novelty：

[
\boxed{
n_j(S)
======

1-r_j(S)
}
]

选择 novelty 最大的：

[
\boxed{
j^*
===

\arg\max_{j\in\mathcal C\setminus S}
n_j(S)
}
]

然后：

[
S\leftarrow S\cup{j^*}.
]

重复直到：

[
|S|=M.
]

最终删除所有不在 (S) 中的 channel。

---

# 七、算法完整流程

对每个 expert：

**Step 1 — PP**

计算当前 NoDownNorm PP：

[
s_j^{PP}.
]

得到：

[
\mathcal P=\operatorname{Top}_{0.1D}(s_j^{PP}).
]

**Step 2 — AIMER candidate screening**

令：

[
R=M-P.
]

候选大小：

[
C=R+\frac12(D-M).
]

从非 PP 通道中选择：

[
\mathcal C
==========

\operatorname{Top}_{C}
(s_j^{AIMER}).
]

**Step 3 — 构造 bilinear similarity**

只对：

[
\mathcal V=\mathcal P\cup\mathcal C
]

构造 (K_{ij}) 和 normalized similarity：

[
\operatorname{sim}(i,j)
=======================

\max
\left[
\frac{
(g_i^\top g_j)(u_i^\top u_j)
+
(g_i^\top u_j)(u_i^\top g_j)
}{
\sqrt{K_{ii}K_{jj}}+\epsilon
},
0
\right].
]

**Step 4 — PP-seeded greedy coverage**

[
S=\mathcal P.
]

不断选择：

[
j^*
===

\arg\min_{j\in\mathcal C\setminus S}
\max_{i\in S}\operatorname{sim}(i,j).
]

直到：

[
|S|=M.
]

**Step 5 — structured pruning**

保留相同索引：

[
W_g[S,:],
]

[
W_{up}[S,:],
]

[
W_{down}[:,S].
]

---

# 八、实现时不用显式做很大的高阶张量

只对：

[
V=|\mathcal P\cup\mathcal C|
]

个候选构造矩阵。

令：

[
G\in\mathbb R^{V\times d}
]

是 candidate 的 gate rows，

[
U\in\mathbb R^{V\times d}
]

是 up rows。

计算四个矩阵中的三个：

[
A=GG^\top,
]

[
B=UU^\top,
]

[
E=GU^\top.
]

那么：

[
\boxed{
K=A\odot B+E\odot E^\top
}
]

这样就是几个 GEMM，不需要构造 (d\times d) 的 quadratic matrix。

对角：

[
q_i=K_{ii}.
]

归一化：

[
C_{ij}
======

\frac{K_{ij}}
{\sqrt{q_iq_j}+\epsilon}.
]

再：

[
C\leftarrow\max(C,0).
]

实现成本相对可控。

---

# 九、这一轮只跑哪几个设置

我建议只跑这 4 个：

| 方法                   | B6 | B9 |
| -------------------- | -: | -: |
| AIMER + PP           | 已有 | 已有 |
| **AIMER + PP + BFC** | 新跑 | 新跑 |

其他全部固定：

* 同一个模型；
* 同一个 checkpoint；
* 相同 (K,Q)；
* (G=10%)；
* NoDownNorm PP；
* 相同 expert-wise pruning；
* 相同 evaluation subsets；
* 不补偿；
* 不微调；
* 不改变各 expert 宽度分配；
* BFC 使用 (\Sigma=I)；
* 不使用 `down_proj`；
* candidate extra ratio 固定 0.5。

也就是说整个实验只改变：

[
\boxed{
\text{AIMER直接取Top-R}
\rightarrow
\text{AIMER筛candidate + BFC选Top-R}
}
]

这样结果非常容易解释。

---

# 十、结果怎么判断

目前 baseline：

[
B6=0.6064,
\qquad
B9=0.8364.
]

如果：

### B6、B9 都提升

这是最理想情况：

[
\boxed{\text{AIMER存在明显的channel redundancy问题}}
]

那么下一步立即加入 `down_proj` direction。

---

### 只有 B6 提升

其实也非常合理，甚至很有价值。

因为 B6 压缩更激进：

[
50%\text{ retain}
]

有限预算下避免重复 channel 更重要。

B9：

[
75%\text{ retain}
]

冗余造成的损失可能没那么明显。

如果观察到：

[
\Delta B6\gg\Delta B9,
]

这会非常符合 coverage 的理论 motivation。

---

### 两个都下降

那么不要立即去调：

* candidate ratio；
* (\Sigma)；
* absolute value；
* down direction；
* conditional residual。

先做一个诊断：

比较 AIMER baseline 保留集合

[
S_A
]

和 BFC 集合

[
S_B
]

的平均 pairwise redundancy：

[
\bar C(S)
=========

\frac{2}{|S|(|S|-1)}
\sum_{i<j}
C_{ij}.
]

如果：

[
\bar C(S_B)\ll\bar C(S_A)
]

但性能反而下降，那么会得到一个很重要的结论：

> **这个 weight-derived bilinear similarity 确实能产生更“多样”的集合，但这种数学 diversity 并不对应模型需要的 functional diversity。**

那时就应该停止这条 pure weight coverage 路线，而不是继续调参。

---

## 我建议这次实验命名暂时叫

[
\boxed{\text{AIMER + PP + BFC}}
]

其中 BFC = **Bilinear Functional Coverage**。

整个实验现在只有一个核心假设：

> AIMER 已经能够提供较可靠的 individual importance；PP 保护少量 expert-specialized channels；BFC 进一步避免剩余有限预算被功能高度相似的 SwiGLU channels 重复占用。

先跑这一个，不加其他组件。它会非常清楚地告诉我们 **redundancy/coverage 这条路线到底值不值得继续。**

---

# 实验结果：Gate Accessibility 与 Hybrid PP5 + GA5（已完成）

## 当前决策

Gate 与 Hybrid 均已完成 B6/B9 Quick9，使用 seed 42 和既有任务限制。四组 Macro 都低于冻结的 AIMER + PPFv1 baseline，因此按预先冻结的停止条件，本轮停止 Gate 与 Hybrid，不晋升任何一个方法，不运行 G0，也不做额外比例、probe、Top-Q 或 budget sweep。

正式结果如下：

| 方法 | 预算 | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gate GA10 | B6 | 0.7984 | 0.4430 | 0.5275 | 0.7344 | 0.4800 | 0.5351 | 0.5864 |
| Gate GA10 | B9 | 0.9550 | 0.7030 | 0.7150 | 0.9453 | 0.8300 | 0.8316 | 0.8300 |
| Hybrid PP5 + GA5 | B6 | 0.8233 | 0.4700 | 0.5425 | 0.7344 | 0.4700 | 0.5596 | 0.6000 |
| Hybrid PP5 + GA5 | B9 | 0.9584 | 0.7090 | 0.7250 | 0.9297 | 0.8300 | 0.8176 | 0.8283 |

与冻结 AIMER + PPFv1 baseline 对比：

| 预算 | AIMER + PPFv1 | Gate GA10 | 差值 | Hybrid PP5 + GA5 | 差值 |
|---|---:|---:|---:|---:|---:|
| B6 | 0.6163 | 0.5864 | -0.0299 | 0.6000 | -0.0163 |
| B9 | 0.8403 | 0.8300 | -0.0103 | 0.8283 | -0.0120 |

Macro 是六个任务分数的非加权平均；报告样本数固定为 ARC 600、HellaSwag 1000、WinoGrande 400、GSM8K 128、MATH-500 100、MMLU 570。

## Gate Accessibility

Gate 只使用 gate weight 与冻结 PP probe 的正余弦可达性，不使用 `up_proj`、`down_proj` 或 PP response score。对 expert $e$、probe $k$ 和 channel $j$：

$$
c_{e,k,j}=
\frac{g_{e,j}^{\top}p_{e,k}}
{\|g_{e,j}\|_2\|p_{e,k}\|_2+\epsilon},
\qquad
a_{e,k,j}=\max(c_{e,k,j},0),
$$

$$
s_{e,j}^{GA}=\operatorname{MeanTopQ}_{Q=4}
\{a_{e,k,j}\}_{k=0}^{K},
\qquad K=8.
$$

Gate GA10 保护 Top-77 GA channels，剩余位置全部由 AIMER 顺序填充。Hybrid 先保护 PP Top-38，再从不属于 PP Top-38 的 GA 顺序中选择 39 个 channel，形成精确的 77-channel protection budget：

$$
\operatorname{Hybrid}=\operatorname{Top}_{38}(PP)
\cup \operatorname{Top}_{39}(GA\setminus\operatorname{Top}_{38}(PP)).
$$

这里的整数拆分是 `PP=round(0.05D)=38`、`GA=round(0.10D)-38=39`，其中 $D=768$。PP 使用不可变的 `PP-Frozen-v1` cache，AIMER 仍是唯一的 global fill ranking。PP 与 GA 各自 Top-77 的 expert-level overlap 均值为 `0.5424361`，全局报告中记为 `0.54244`。

## 构建与结果审计

- 两种方法各有 B6/B9 两个 profile；每个预算均有且仅有 6 份 Quick9 report JSON。
- 四个 profile 的 `profile_widths` 分别在全部 $48\times128=6144$ 个 experts 上固定为 6 blocks（B6）或 9 blocks（B9），即每 expert 保留 384 或 576 channels。
- 四组 ranking 均包含 6144 行、每行 768 个 channel index；每行都是完整且唯一的 `0..767` permutation。
- Gate 的 protection metadata 为 77 GA channels；Hybrid 为 38 PP + 39 GA，且两部分按构建逻辑 disjoint。
- AIMER cache SHA256：`266646957cddf9b91645d50f247c6ff79ef11773b464e04d2e4432698fd3158c`。
- PP-Frozen-v1 cache SHA256：`0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- Gate B6/B9 ranking SHA256：`7c19f4af1fd460affeabb6d09f1b9387caebba1727c1932fc2e52e0fca43f98b`。
- Hybrid B6/B9 ranking SHA256：`c9b96e8b82854b19cc0402213fb9bbb614b76b4dec2942b61f1caca3e08b3e7c`。
- Gate B6 profile SHA256：`050fbf11b83f8bffdae83bcb3833d5803cb7843927d78a4a311af12a5938005f`；Gate B9：`a8409e2211711ee0fd57fc35c579b8a65613aa63893c9e44d0f7ddf433fa7f00`。
- Hybrid B6 profile SHA256：`d5412c1b2c44c9a093c27654096e6dc3f51dac757a5c03c911356f3b8ef2c8d3`；Hybrid B9：`ffa41ca71420a98627b77d5bd11e39a6b5eef0740f6bf3f690ee6e0145cc7f23`。
- Gate diagnostics SHA256：`e08dd82682541f293053c56a6a50d07574d9e03d9621632f0b764673fdf17bc4`；Hybrid diagnostics SHA256：`48d41ef964e1f06e131f07e1b65f2e18091576ece69e5136a45c8124179ea96c`。

## 停止结论

Gate 在 B6 和 B9 都下降，说明 gate-only accessibility 不能在当前固定 77-channel protection budget 下替代冻结 PP。Hybrid 在 B6 相比 Gate 有所改善，但仍低于 baseline；在 B9 也没有恢复 baseline，且略低于 Gate。因而 GA 与 PP 的 overlap 虽然只有约 54.24%，但这部分互补性没有转化为稳定的下游收益。

本轮结论是停止两条方法路线，保留实现、测试、profile、ranking、diagnostic 和 Quick9 结果作为负结果记录；不修改 `PP/analysis_pp.md`，不改变冻结 cache，不继续扩展方法或预算。

---

# Baseline 机制分析：PP 从 AIMER rescue 的 channels

## 分析目的

Gate 与 Hybrid 在 B6/B9 都没有超过冻结的 AIMER + PPFv1 baseline，因此不再继续扩展新的 protection method，转而直接分析 baseline 的 retained-set substitution。这里测试的具体 motivation 是：PP 是否会从 AIMER 认为较弱、因而原本落在 pruning 区域的 channels 中，救回一批对 PP probes 有明显响应的 channels。

对每个 layer/expert 分别定义：

$$
S_A=	ext{AIMER-only ranking prefix},
\qquad
S_{AP}=	ext{AIMER+PP ranking prefix}.
$$

在相同 retained width 下，定义：

$$
R=S_{AP}\setminus S_A,
\qquad
D=S_A\setminus S_{AP}.
$$

$R$ 是 PP-rescued channels，$D$ 是被 PP displaced 的 AIMER-retained channels。由于两边使用相同预算，逐 expert 必须满足 $|R|=|D|$。AIMER rank 为完整 768-channel ranking 中的 rank，rank 1 表示 AIMER 认为最重要。PP score 使用冻结 PP-Frozen-v1 的 positive-only、router self + K=8 cosine neighbors、Top-Q mean（Q=4）、NoDownNorm 定义；为了跨 expert 比较，另报告 `PP score / expert median PP score`。

分析脚本为 [`PP/analyze_aimer_pp_rescue.py`](analyze_aimer_pp_rescue.py)，完整 channel-level records 为 [`rescue_displaced_channels.csv`](experiments/analysis/aimer_pp_rescue_20260807/rescue_displaced_channels.csv)，汇总为 [`summary.json`](experiments/analysis/aimer_pp_rescue_20260807/summary.json)。

## 结果

`PP-high` 在本分析中定义为 normalized PP score 不低于本 expert 中位数的 2 倍，即：

$$
	ext{PP-high}(j)
\Longleftrightarrow
\frac{s_j^{PP}}{\operatorname{median}_{j'\in e}s_{j'}^{PP}}\ge 2.
$$

| 预算 | population | 数量 | AIMER rank Median | AIMER rank P10--P90 | PP score / expert median Median | PP-high 比例 |
|---|---|---:|---:|---:|---:|---:|
| B6 | rescued $R$ | 185,648 | 569 | 420--727 | 3.592x | 99.42% |
| B6 | displaced $D$ | 185,648 | 368 | 351--381 | 0.912x | 9.87% |
| B9 | rescued $R$ | 88,983 | 671 | 595--748 | 3.563x | 99.42% |
| B9 | displaced $D$ | 88,983 | 568 | 559--575 | 0.919x | 9.89% |

预算特定的 AIMER retention threshold 为 B6=384、B9=576。所有 $R$ 都位于对应 threshold 之后，这是集合定义直接保证的；但它们并不只是 cutoff 附近的微小扰动：

- B6 rescued channels 的 AIMER rank 中位数为 569，距离 cutoff 向 pruning 方向 185 个 rank；82.41% 至少低于 cutoff 64 个 rank，65.18% 至少低于 cutoff 128 个 rank。
- B9 rescued channels 的 AIMER rank 中位数为 671，距离 cutoff 向 pruning 方向 95 个 rank；66.19% 至少低于 cutoff 64 个 rank，33.04% 至少低于 cutoff 128 个 rank。
- 对照之下，B6 displaced channels 的 AIMER rank 中位数为 368，B9 为 568；分别有 99.98% 和 100.00% 位于 cutoff 前 64 个 rank 内，符合 PP 主要在 AIMER boundary 附近替换 channels 的结构。
- Rescue 与 displaced 的 normalized PP-score median ratio 为 B6 `3.94x`、B9 `3.88x`。PP-high 在 rescued 中约为 99.42%，而在 displaced 中约为 9.9%。

因此，用户预期的 `AIMER-low / PP-high` population 在 B6/B9 都非常明显：大量 PP-rescued channels 的 AIMER rank 远低于 retention threshold，同时 frozen PP score 显著高于各自 expert 的典型水平。主图同时展示 rescued/displaced 的 AIMER rank 分布，并在下方 scatter 中标出 AIMER cutoff 与 `2x expert median` 的 PP-high 水平线：

![AIMER rescue population](experiments/analysis/aimer_pp_rescue_20260807/aimer_pp_rescue_population.png)

对应的矢量图版本为 [`aimer_pp_rescue_population.svg`](experiments/analysis/aimer_pp_rescue_20260807/aimer_pp_rescue_population.svg)。

## 对 baseline 机制的判断

这组结果支持以下描述性机制：AIMER 与 PP 确实在识别不同类型的 channel。AIMER+PP 并不是简单地把 AIMER cutoff 附近的 channel 做随机交换，而是用 PP 高响应 channel 替换一批 AIMER rank 较低的 channel；被替换掉的 AIMER channels 的 PP score 通常接近或低于 expert 中位数。这为 “AIMER 提供全局重要性，PP 补充 AIMER 漏掉的 expert-specialized / probe-responsive channels” 提供了直接的集合层证据，也解释了为什么冻结 AIMER+PP baseline 可能比 AIMER-only 更稳。

但这仍然是 rescue-set 的描述性证据，不是因果证明。`AIMER-low / PP-high` 只说明 PP 选择的 channels 具有该统计属性，不能单凭分布证明这些 channels 造成了下游恢复，或证明每个 rescued channel 都比 displaced channel 对真实任务更有用。更强的验证需要已有 Quick9 样本上的 per-task/per-layer/channel ablation，或另一种与真实输入激活直接关联的 performance attribution signal。本轮不启动额外评测，结论保持为：**baseline motivation 在 channel-set substitution 层面得到强支持，但 causal attribution 尚未建立。**

## 制品与冻结协议审计

- AIMER cache SHA256：`266646646cddf9b91645d50f247c6ff79ef11773b464e04d2e4432698fd3158c`。
- PP-Frozen-v1 cache SHA256：`0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da`。
- 重新计算 PP score 与冻结 PP ranking 的 Top-77 set 在全部 experts 上一致；少量 tail permutation 差异仅来自数值 ties，分析中的 PP rank/set 以冻结 permutation 为准。
- B6/B9 各覆盖全部 6144 个 experts；每个 expert 的 $|R|=|D|$，aggregate 数量也分别相等。
- 本分析只读取既有 AIMER、PP 和 AIMER+PP artifacts，不运行 G0，不改变 frozen cache，不进行新的模型评测。

---

# 实验结果：AIMER + PP + ESP + PWRP 保护预算扩展

## 实验目的与冻结协议

ESP 与 PWRP 的 standalone B6/B9 Quick9 结果均未超过冻结的 AIMER + PP10 baseline，
因此本轮不重新调 probe 数、$Q$ 或 spectral rank。这里仅验证在保持三种 source scoring
完全不变的条件下，增加 multi-source protection budget 是否能把 source complementarity
转化为稳定的下游收益。

模型、Quick9 数据集、seed `42`、batch size `16`、greedy generation、PP-Frozen-v1、
ESP 和 PWRP 的全部配置均保持冻结。multi-source selection 使用
`ordered_unique_quota_fill`：source 按命令行声明顺序依次选择，后续 source 跳过已被前
序 source 选择的 channel，最终 protection set 大小严格等于 quota 总和。AIMER 只用于
填充 protection set 之后的 ranking remainder，以及计算 overlap/rescue 诊断。

对每个 expert，$D=768$，G10、G12.5 和 G15 的 protected channel 数分别为：

| 预算 | PP | ESP | PWRP | ratio | protected channels/expert |
|---|---:|---:|---:|---:|---:|
| G10 | 27 | 25 | 25 | 0.10 | 77 |
| G12.5 | 34 | 31 | 31 | 0.125 | 96 |
| G15 | 40 | 38 | 37 | 0.15 | 115 |

## G12.5 与 G15 正式结果

两组预算均完成 B6/B9 Quick9，且每个预算均有且仅有 6 份官方 task JSON report；
Macro 为六个任务分数的非加权平均。

| 预算 | block | ARC | HellaSwag | WinoGrande | GSM8K | MATH-500 | MMLU | Macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G12.5 | B6 | 0.8250 | 0.5050 | 0.5350 | 0.7734 | 0.5500 | 0.5491 | 0.6229167 |
| G12.5 | B9 | 0.9550 | 0.7080 | 0.7125 | 0.9531 | 0.8500 | 0.8316 | 0.8350333 |
| G15 | B6 | 0.8234 | 0.4890 | 0.5375 | 0.7891 | 0.4600 | 0.5509 | 0.6083167 |
| G15 | B9 | 0.9516 | 0.7150 | 0.7325 | 0.9609 | 0.8400 | 0.8193 | 0.8365500 |

结果目录为：

- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERMix-PP34-ESP31-PWRP31-B6of12_202608071705_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERMix-PP34-ESP31-PWRP31-B9of12_202608071705_42/`
- `result/Qwen330BA3BInstruct_Prune6of12_vllm_CalibrationFree_quick9_AIMERMix-PP40-ESP38-PWRP37-B6of12_202608071705_42/`
- `result/Qwen330BA3BInstruct_Prune3of12_vllm_CalibrationFree_quick9_AIMERMix-PP40-ESP38-PWRP37-B9of12_202608071705_42/`

## AIMER overlap 与 source-specific rescue

对最终 protection set $P$ 和 AIMER Top-$M$ set $S_{AIMER}$，定义：

$$
\mathrm{already}=|P\cap S_{AIMER}|,
\qquad
\mathrm{rescue}=|P\setminus S_{AIMER}|.
$$

下表为全部 $48\times128=6144$ 个 experts 的均值。B6 使用 Top-384 AIMER cutoff，
B9 使用 Top-576 cutoff。source rescue 是实际 ordered、去重后的 source selection 相对
对应 AIMER cutoff 的 rescue 数；它们不是三个 source rescue 的简单相加，因为后续 source
会跳过前序 source 已选 channel。

| 预算 | block | protected | AIMER already | already ratio | AIMER rescue | rescue ratio | PP rescue | ESP rescue | PWRP rescue |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G12.5 | B6 | 96 | 60.0933 | 0.6260 | 35.9067 | 0.3740 | 11.1343 | 11.4110 | 13.3615 |
| G12.5 | B9 | 96 | 79.0939 | 0.8239 | 16.9061 | 0.1761 | 5.2274 | 5.1942 | 6.4845 |
| G15 | B6 | 115 | 70.5641 | 0.6136 | 44.4359 | 0.3864 | 13.6198 | 14.5234 | 16.2926 |
| G15 | B9 | 115 | 93.9603 | 0.8170 | 21.0397 | 0.1830 | 6.4320 | 6.6696 | 7.9382 |

四个 profile 均覆盖 6144 个 experts，最终 protected size 严格满足 96 或 115，且诊断
中的 AIMER already/rescue 对每个 expert 满足 `already + rescue = protected`。这些统计
确认 ESP/PWRP 与 PP 对 AIMER 存在实际集合层互补性，但 rescue 数量本身不能证明被救回
的 channel 对下游任务有用。

## 与 G10 和冻结 baseline 对比

冻结的 AIMER + PP baseline Macro 为 B6 `0.6162667`、B9 `0.8402833`。已完成的 G10
multi-source 结果为：

| 方法 | B6 Macro | B9 Macro |
|---|---:|---:|
| AIMER + PP27 + ESP25 + PWRP25 | 0.6265833 | 0.8409667 |
| AIMER + PP39 + ESP38 | 0.6177167 | 0.8321667 |
| AIMER + PP39 + PWRP38 | 0.5961667 | 0.8366500 |
| 冻结 AIMER + PP baseline | 0.6162667 | 0.8402833 |
| AIMER + PP34 + ESP31 + PWRP31 (G12.5) | 0.6229167 | 0.8350333 |
| AIMER + PP40 + ESP38 + PWRP37 (G15) | 0.6083167 | 0.8365500 |

相对 G10 三源组合 PP27+ESP25+PWRP25，G12.5 的 B6 Macro 下降 `0.0036666`，B9
下降 `0.0059334`；G15 的 B6 下降 `0.0182666`，B9 下降 `0.0044167`。相对冻结
AIMER + PP baseline，G12.5 在 B6 上升 `0.0066500`，但 B9 下降 `0.0052500`；G15
在 B6 下降 `0.0079500`、B9 下降 `0.0037333`。

## 最终停止结论

增加 protection budget 没有带来单调收益。G12.5 的 rescue 数量确实高于 G10，但
Quick9 Macro 没有随之提高：B6 从 G10 的 `0.6265833` 降至 `0.6229167`，B9 从
`0.8409667` 降至 `0.8350333`。G15 进一步增加预算后，B6 退化到 `0.6083167`，
B9 也只有 `0.8365500`，两个预算都低于冻结 AIMER + PP baseline。

因此本轮支持的结论是：三种 source 在 protection-set 层面具有互补性，但当前 frozen
scoring 与 ordered quota fill 没有把更多 rescued channels 稳定地转化为任务收益；有
用收益在 G10 附近已经达到峰值，继续扩展到 G12.5/G15 反而引入有害替换。按 B6-centered
stopping rule，本实验停止在 G10，不启动更大的 protection budget，也不重新调 ESP/PWRP
的 probe 数、$Q$ 或 spectral rank。
