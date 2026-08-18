# Static Expert Pruning — Related Work and Novelty Audit

> 状态：持续更新。当前结论是“静态异构专家宽度”不构成创新；Dynamic-Regret 的
> admission-event 解释已被消融否决。Conditional Dual-Utility 在 50%–80% 均优于
> route×RMS，但强压缩下会漏掉 rare catastrophic paths。当前研究方向升级为
> Tail-Risk Constrained Conditional Utility；自动检测、WikiText/C4 matched-domain PPL
> 已完成，Qwen3 Base 跨 checkpoint 复现也通过；TruthfulQA 可靠性预注册标准失败，
> Qwen1.5-MoE 跨架构预注册标准也失败。train-only cross-fitted applicability selector 已
> 在 Qwen1.5/Qwen3 retrospective diagnostic 中分别 4/4 选择正确分支；Qwen3.5
> prospective validation 正在运行，故仍不能声称最终创新成立。

## 直接相关工作

| 工作 | 主要机制 | 与本项目的重叠 | 当前边界 |
|---|---|---|---|
| TENP / Trapezoidal Expert Neuron Pruning, arXiv:2606.09885 | 结构化 expert-neuron pruning；保留重要专家；以 neuron output 对完整 expert output 的投影评分；随深度变化的 trapezoidal sparsity；保持 routing topology | 直接覆盖“routed expert 内结构化 neuron pruning、重要专家保护、输出方向贡献与 depth-aware widths” | 不能声称首次静态 expert-neuron pruning、首次保护重要专家或首次深度异构宽度；贡献必须落在 test-free applicability selection 与明确的 fallback/refinement 失败处理 |
| MoE-Slimming / Attribution-Guided and Coverage-Maximized Pruning, arXiv:2606.18304 | attribution expert/channel scoring，全局 coverage-maximized 异构 channel allocation，物理结构裁剪，64/128/256 对齐，Qwen3-30B-A3B，恢复微调 | 几乎覆盖“静态异构专家宽度 + block alignment + 全局预算” | 不能把异构宽度或 64-channel 静态 slimming 当创新；必须比较其 attribution/coverage 思路 |
| TransAct | 用 transitional activation statistics 做结构化剪枝，并显式结合 normal 与 maximum/outlier activation | 与 typical/tail channel coverage 的“均值/常态 + 极值/尾部”动机高度相邻 | 不能把 average+max activation 或 outlier protection 单独当新意；需要证明 MoE physical-expert parent/refinement applicability 与 exact-budget fallback 机制 |
| FlexMoE, arXiv:2606.27866 | Taylor channel ranking、专家内通道重排、prefix slicing、每专家离散 retention action、单次 action training 导出多预算 nested subnetworks，可选单点恢复微调 | 直接覆盖“重要性排序后的 expert prefix + heterogeneous width + 多预算静态子网” | 本项目不能声称首次 prefix 静态缩宽、nested expert 或离散宽度分配；区别必须落在 retraining-free token-conditional utility 与 rare-event safety constraints，而不是结构形式 |
| REAP, arXiv:2510.13999 | router-weighted expert activation pruning；静态 whole-expert 剪枝；Qwen3 | 路由加权的静态结构重要性 | 本项目是 expert 内 prefix width，不是 whole-expert-only；仍须把 REAP 作为 scoring 先例 |
| MoE-Pruner, arXiv:2410.12013 | router-weighted Wanda structured pruning；one-shot；可选 expert-wise KD | router×activation/weight score 与蒸馏 | Conditional Dual-Utility 必须区别于一般 router-weighted Wanda/KD；“蒸馏”命名本身不构成新机制 |
| MoSE, arXiv:2602.06154 | nested/slimmable experts，runtime width selection，多宽预训练/持续预训练 | router-conditioned width allocation | 本项目 post-training 冻结静态结构；不能声称首次按 token/expert 选宽度 |
| POP, arXiv:2602.06822 | retained/candidate/pruned partition，context-conditioned online pruning | static cap + dynamic tail | 若未来保留动态尾部，必须明确与 POP 的 online pruning 区别 |
| MAESTRO, arXiv:2607.08601 | 跨层 expert routing trajectory 的 Markov 建模 | routing trajectory | “使用 routing trajectory”本身不新；本项目只能主张 prefix truncation regret 的特定蒸馏目标 |
| DTop-p MoE, arXiv:2512.13996 | 以可学习 top-p threshold 和 PI controller 做全局稀疏度可控的动态 expert routing | top-p / cumulative routing mass | 若 top-p residual component 成为最终信号，必须明确本项目不是首次 top-p MoE；区别只能是 post-training 将 token-level cumulative mass 蒸馏为静态 physical-expert channel capacity |
| Mixture Compressor, arXiv:2410.06270, ICLR 2025；MC# arXiv:2510.10962 | activation frequency、activated weight、quantization loss 组成 expert significance；逐层 mixed-bit ILP；另有 online dynamic pruning | route frequency + 局部压缩损失 + 离散静态预算优化 | route×RMS 已是基线；当前区别只剩 token routed-set 内的 dual-prior 条件归一化与 expert 内 prefix capacity，不能泛化声称 importance allocation 新颖 |
| D²-MoE, arXiv:2502.17298 | shared base + expert delta compression，semi-dynamic structured pruning | 静态/半动态结构压缩 | 参数化方式不同；需在 related work 说明 shared-delta 路线 |
| SlimMoE, arXiv:2506.18349 | structural expert slimming + staged distillation | slimming 与 distillation | “slimming + distillation”组合本身不新 |
| MoNE, arXiv:2507.00390, ICLR 2026 | 用轻量 novice 替换冗余 experts | 大专家变小专家 | 本项目是通道截断，不是 novice replacement；标题/表述需避免“首次大专家变小专家” |
| STUN, arXiv:2409.06211 | structured-then-unstructured expert pruning | 专家内部结构化剪枝 | 本项目保持 prefix/hardware alignment，不做后续 unstructured 阶段 |
| Unveiling Super Experts in MoE LLMs, arXiv:2507.23279, ICLR 2026 | 用 rare extreme `down_proj` output activations 识别少量 Super Experts；Qwen3-30B-A3B 报告 L1-E68/L2-E92/L3-E82 | MoE rare activation tail 与关键专家机制 | 不能声称首次发现 Super Experts；论文将其用于 post-training compression 留作未来工作，本项目只能研究如何把该风险自动纳入静态 prefix allocation |
| Supernodes and Halos / SCAR, arXiv:2604.23475 | Fisher-style activation-gradient loss proxy；显式保护 top-1% loss-critical dense-FFN channel core | 小规模 critical-channel protection | “保护关键通道”本身不新；本项目必须证明 MoE routed conditional utility 与 rare-event constraints 的联合必要性，并避免依赖人工 expert ID |
| BitsMoE, arXiv:2606.00079 | MoE mixed-precision/importance-aware compression | expert 异构资源分配 | 进一步限制“按专家重要性分配容量”的宽泛 claim |
| ConMoE, arXiv:2605.29350 | conditional/context-aware MoE compression | 条件信号用于压缩 | 条件性本身不新；贡献必须落在固定结构中对 expected utility 和 catastrophic tail risk 的分工 |
| EASY-EP, arXiv:2504.06792 | few-shot domain expert localization；gate、expert-output norm 与 token representation contribution；whole-expert domain pruning | output-aware/token-aware expert importance | 进一步限制“条件输出贡献”claim；本项目区别仍是 physical-expert 内 prefix width、train-only rare-event channel coverage 和 exact global risk constraints |
| MC-Suite / Finding Fantastic Experts, arXiv:2504.05586 | 多类 expert importance、迭代 expert dropping、instruction-following 能力损伤与恢复 | whole-expert importance 与能力退化 | 单一 PPL 不能覆盖 capability preservation；后续必须补 instruction following/可靠性评估，但其 dropping recipe 不覆盖本项目的 tail-constrained prefix allocation |
| On the Utility and Factual Reliability of Pruned MoE Models, arXiv:2607.01444 | 四模型、六种 whole-expert pruning、跨域 utility 与 factual reliability；极端剪枝 hallucination 增加 | 证明平均 benchmark utility 不足以判断剪枝安全性 | 不得把低 PPL 等同于可靠性保持；它没有提出 activation-tail channel risk constraint，但要求本项目后续增加 reliability/downstream evidence |
| Half the Experts, All the Code, arXiv:2607.16721 | Qwen3.6/Gemma-4 domain pruning；五种 selector；代码生成、能力 trade-off 与 matched-memory quantization；报告最佳策略跨模型翻转且 PPL 可偏好 broken model | 强化 per-model validation 与 PPL–behavior decoupling | 单模型 PPL 不能支撑通用 recipe 或能力保持；必须补跨 checkpoint/架构和直接 factual/instruction probes，本项目已冻结 Qwen3 base 与 TruthfulQA follow-up |
| Qwen MoE Hybrid Pruning engineering reproduction, `tylevnovik/qwen-moe-prune`, 2026-07 | Qwen3.6 浅层全保留、深层按 hit-count 留 top experts；按 layer Gini 决定 whole-expert pruning | layer-dependent routing concentration 和静态部署 | 非同行评审的新工程证据；说明 layer scale/concentration 可能混淆 global risk，需用无白名单 selector 与跨校准段复现，而不能把浅层保护当新颖性 |

## 通用结构化 LLM 剪枝先例

| 工作 | 相关机制 | 对本项目的约束 |
|---|---|---|
| FLAP, arXiv:2312.11983 | fluctuation metric、全局标准化、bias compensation | 需要考虑 activation fluctuation 与 compensation baseline；不能把 retraining-free global structured allocation 当新 |
| Wanda, arXiv:2306.11695 | weight magnitude × activation norm | RMS/Hessian baseline 属于该类激活加权思想的结构化变体 |
| LLM-Pruner, arXiv:2305.11627 | dependency-aware channel pruning、Taylor criterion | channel dependency/Taylor importance 是必要 related-work 边界 |
| Unified MoE Compression, arXiv:2406.02500 | expert trimming、slimming、layer/block drop 的统一研究 | 必须说明本项目只聚焦静态 physical-expert prefix width，不声称统一压缩框架 |

## 当前候选贡献的最小可辩护形式

新增文献审计后，Route×Tail 本身不能作为 ICLR-level novelty：router-weighted activation
已有 REAP/MoE-Pruner，coverage-maximized heterogeneous widths 已有 MoE-Slimming，
normal+maximum/outlier activation 已有 TransAct，结构化 expert-neuron pruning 与重要专家
保护又有 TENP。当前贡献应移到更窄的算法层：用多个 disjoint train-only folds 判断
conditional-utility/risk-floor refinement 是否稳定优于 Route×Tail fallback；若不稳定则在
不读取 test 的情况下自动回退，并保持 exact block budget、profile/cache hash audit 与
prospective validation 顺序。

对 train calibration token 的每个 routed physical expert，分别在当时的 routed set 内
归一化 `gate×AMP` 与 `gate×AIMER`，几何融合后与 RMS/Hessian prefix coverage 相乘：

\[
v_{l,e,j}=\mathbb E_t[\mathbf 1(e\text{ routed})\,
\sqrt{\bar u^{AMP}_{l,t,e}\bar u^{AIMER}_{l,t,e}}\,c_{l,e,j}].
\]

然后在同一个 64-channel block 总参数预算下求解静态 prefix profile。候选区别点不是
“使用频次”“使用损失”“使用动态路由”中的任一个，也不再是动态 allocator 的 admission
事件，而是将 token-specific routed-set conditional dual utility 聚合成 static physical-
expert channel capacity。静态 `route×dual-prior×RMS` 结果 8.217790，而条件归一化版本
为 8.079852；这是当前支持该区别的主要消融证据。

## 新发现的强压缩失效模式

Conditional Dual-Utility 在 80% 结构剪枝下把 published Super Expert L2-E92 分配为
width 0。手工等预算保护三个 published Super Experts 将 PPL 从 `30.0623` 降到
`14.9950`，随机 early-layer 三专家保护为 `30.3728`。进一步消融显示 L2-E92 单独完整
保护为 `14.9092`，而 L1-E68/L3-E82 分别为 `30.5585/29.9945`；L2-E92 仅保留
1/2 个 64-channel blocks 已达到 `15.0749/14.9356`。route-count 几乎相同的 L2-E18
完整保护仍为 `30.2631`。

这些结果支持一个比平均 expected utility 更具体的缺口：稀有但灾难性的 expert-channel
路径可能在均值目标中得到零容量，而失效由极小 prefix 主导。人工 published-ID 保护只用于
发现机制，不能成为最终算法。下一步必须用 train-only tail statistic 自动检测，保持相同结构
block 总预算，并用随机及 route-count-matched controls 证明选择特异性。

## 修订后的最小可辩护候选

最终候选若验证成功，应表述为 **Tail-Risk Constrained Conditional Utility**：

1. routed-set conditional dual utility 分配常规概率质量下的静态容量；
2. train-only rare-event proxy 检测均值目标遗漏的高风险 physical expert-channel prefix；
3. 只给自动选中的极少数 prefix 设置最小 floor，并从最低边际 utility blocks 回收相同预算；
4. 保持 exact global structural budget、固定 profile、test-free calibration。

该组合仍有显著 novelty 风险：Super Experts 覆盖 outlier mechanism，SCAR 覆盖 critical
channel protection，MoE-Slimming 覆盖异构 channel widths。只有自动选择确实识别 L2-E92
或同类路径、PPL 明显优于纯 expected utility、随机/频次控制失败且跨预算/模型复现时，才有
资格讨论 ICLR-level contribution。

## 80% 初步验证结果

train-only proxy 将真正控制崩溃的 L2-E92 排为全局第 2，并在固定 99.5% quantile、
0.1×global-max、前 4 层 gate 下自动选出三个 published Super Experts。自动 width=2
floors 与 λ=0.50 tail coverage 联合后，80% exact structural budget 的 PPL 为 `13.9065`，
相比 Conditional Dual `30.0623`、tail-only `16.2754`、floor-only `14.9464`、手工
L2-E92 full `14.9092` 均更低。随机三专家 width=2 为 `29.9870`，route-count-matched
L2-E18 width=2 为 `30.0614`。

这组证据支持“期望效用和灾难尾部风险是两个不同目标”的 formulation：tail coverage
改变 expert 内 prefix 排序，risk floor 防止跨 expert 分配把 rare critical path 置零；联合
优于二者单独。它仍不足以完成 novelty claim，因为 early-layer gate 受 Super-Expert 文献
启发，λ 在同一 WikiText test 上比较；现已补 C4 matched-domain 与 Qwen3 Base 跨
checkpoint 复现，但仍缺跨架构和
独立任务复现。后续论文表述
必须把当前结果称为机制验证，不得写成已完成的 ICLR-level 结论。

后续全层 selector 去掉 early-layer gate 后选择 31 个 global-tail experts，并在 50%/60%/80%
得到 `8.0634/8.6887/13.7617`，均优于 Conditional Dual；60%/80% 也优于 early-only
selector。这显著降低了“编码 published Super-Expert 层位”的风险，但 quantile=99.5%、
λ=0.50、minimum width=2 仍是在单模型/单数据研究循环中确定，尚需完全独立的模型和校准域
冻结复现。Qwen3 Base 的 60% PPL `10.341598` 也优于 Conditional Dual `10.371195`
和 Route×RMS `10.623834`，且 routed-compute 剪枝率更高；当前可以主张同架构跨
checkpoint 的强机制证据，仍不能主张完成 ICLR-level novelty validation。

2026-07 最新 reliability 工作进一步收紧结论：即使 PPL 和常规任务 utility 保持，极端
whole-expert pruning 仍可能增加 hallucination，且跨域退化更快。因此本项目当前 PPL 结果
只能证明语言建模失真下降，不能外推为 factual reliability 或 instruction-following 保持。
这不否定 Tail-Risk formulation，但把跨域/可靠性评估提升为最终论文的必要门禁。

本项目的 TruthfulQA 直接实验复现了这一解耦：Tail-Risk 的 MC1 比 Conditional Dual 高
`1/817`，但 MC2 低 `0.002856`，未通过预注册标准；所有 60% profiles 都明显低于 Dense。
因此“rare activation tail constraint 同时保护 factual reliability”不能作为当前贡献，除非
后续加入直接 reliability-aware 目标并在未见任务上验证。

Qwen1.5-MoE 的跨架构结果进一步否定通用 recipe claim：Tail-Risk `10.239141` 虽优于
Conditional Dual `10.626735`，但差于 Route×RMS `10.114996`，且执行更多 routed
channels。由此不能把 routed-set conditional dual utility 宣称为架构无关贡献；论文若保留
该组件，必须将适用条件或 architecture-aware normalization 明确建模并重新预注册验证。

## 跨语料机制边界审计

C4→WikiText 的正反向分解进一步收紧 novelty 表述。C4-calibrated risk floors 与
WikiText target-domain coverage 组合时，四折 60% PPL 为 `8.6907±0.0015`，几乎复现
WikiText selector `8.6887`；C4 channel coverage 无论配 C4 还是 WikiText risk floors，
PPL 都约为 `9.02`。旧“完整同源 C4”`11.2977±0.1949` 后续查明使用了
`parent_mode=combined`，协议无效，不能作为 Conditional Dual 的跨域证据。

因此不能声称 tail-aware channel ranking 或整个 static profile 是 corpus-invariant。
当前可防守的新增机制只剩更窄的 factorization：expert-level rare catastrophic-risk set
表现为 corpus-stable safety constraint，而 conditional expected utility 与 prefix coverage
是 domain-adaptive capacity allocator。这一分工与 ConMoE/EASY-EP 的 domain-aware
importance 有相邻性，论文必须把“跨域稳定 risk floor”作为机制发现而非宽泛的条件压缩
首创。

修正后的显式 dual matched-domain C4 证据为：60% 四折 Route×RMS、Conditional Dual、
Tail-Risk 均值分别 `17.2958/17.0769/16.6090`，Tail-Risk 4/4 胜出；fold-0 的
coverage-only、floor-only、combined 分别 `16.8961/16.8030/16.6282`，相对 dual
`17.0505` 再次呈互补；50%/60%/80% 三个预算也全部胜出。这使当前 claim 从“单一
WikiText 机制迹象”升级为“两个 matched domains 的一致 PPL 和机制证据”，但仍需另一
模型和 factual reliability 才能判断是否足够 ICLR 级别。

任何 C4 profile 到 WikiText 的 cross-domain 结果都不等于 C4 本域质量；论文表格必须把
`calibration domain`、`evaluation domain`、teacher/coverage/risk source 分栏列出，禁止
把 cross-domain transfer 与 matched-domain compression 混为一谈。

## 否决条件

- 若 full WikiText-2 PPL 未严格低于 route×RMS 和 dual-route×RMS，同预算下直接否决。
- 若收益完全由 routed compute 增加解释，不能声称更优的同计算压缩，只能报告参数容量结果。
- selection indicator 已被实验证明有害，因此不得再把 admission/truncation-event distillation 作为贡献。
- 若 60%/80% scaling 不稳定，或更简单的同计算 route-weighted importance 可复现收益，则将贡献降级为负结果/工程经验。
- 若 test PPL 参与 profile 搜索或 profile 修改，结果协议作废。

## Prospective selector 后的 novelty 边界

Qwen3.5 上已经完成真正的先选择后 test：四个 disjoint train-only folds 在 test 前 4/4
选择 Tail-Risk，full WikiText PPL 为 `7.834395`，严格低于同一 60% exact structural
budget 的 Route×Tail `8.062748`。这使最可辩护的贡献从单独的 route-weighted tail channel
score，升级为 **robust fallback + cross-fitted applicability gating + frozen exact-budget
profile audit**。现有相关工作虽分别覆盖 route-weighted importance、heterogeneous expert
widths、outlier protection 与 global budget redistribution，但当前审计尚未发现把候选 refinement
是否启用本身作为 disjoint train folds 上的可审计、test-free model-selection 问题，并在未见
第三 checkpoint 上先冻结后验证的直接等价方案。

该表述仍必须克制。Route×Tail 分数本身不是 novelty；Tail-Risk 的 channel/floor 因子也与
outlier-aware pruning 相邻。真正可能新增的是 applicability protocol 与 failure-driven
factorization，而不是某个 importance 乘法式。此外 Qwen3.5 虽有 hybrid attention、更多
experts 和不同 tokenizer，仍与 Qwen3 同属相邻谱系；Qwen1.5 的 joint recipe 负结果要求
论文明确 fallback 的必要性。最后，Tail-Risk 比 Route×Tail 多执行 `0.1692` 个百分点 routed
channels，prospective 胜利不能写成 matched-compute superiority。

协议审计也必须进入论文 limitations：Qwen3 的 `233,368` full-corpus token 数并非 tokenizer
不变量；Qwen3.5 对相同文本为 `231,940`。项目已冻结 model-tokenizer-specific cache、记录
Arrow/cache SHA256，并将 formal 门禁修为 exact windows+tokens。首次误标结果没有删除，
修正复跑数值完全一致。该透明修正增强可复现性，但不能被省略为“从一开始协议完全无误”。

## 双预算 compute-calibrated pruning 查重

针对新增的 exact-structure + expected-routed-compute allocator，使用 Agent Reach 的 GitHub
`gh CLI` 与 Web/Jina Reader 做了定向审计。MoE-Slimming（arXiv:2606.18304，ICML 2026
Spotlight）已经覆盖 attribution-guided channel scoring、coverage-maximized expert widths、
global prune-ratio planning 和 alignment adjustment；其公开 planner 以保留 channel 数/coverage
为预算，仓库中对 latency、FLOPs、throughput 的 compute-constrained planner 检索为空。
因此不能把“global structural coverage allocation”当作本项目创新，但当前公开实现没有同时
匹配 train route-weighted executed blocks。

TENP（arXiv:2606.09885）已做 expert-neuron pruning，并报告 average activated expert
parameters；其 trapezoidal shallow-to-deep pattern 和 expert importance 设计说明 activated
parameter cost 必须作为强对照，但摘要与可见方法未给出在 exact global structural budget 下
再显式匹配 route-distribution expected compute 的双约束 prefix allocator。EvoESAP 则在固定
whole-expert removal budget 下，用 ESAP fitness 与 evolutionary search 寻找非均匀 layer
allocation；它证明“固定全局预算下搜索 allocation”不是创新，但预算单位、拓扑和 objective
均不同于 expert-neuron prefix + route-weighted compute 的双预算问题。

arXiv 对精确短语 `"dual budget" mixture experts pruning` 未返回结果；`"routed compute"`
相关搜索只返回系统 benchmark，而 GitHub 对 `routed compute pruning ratio`、MoE-Slimming
中的 latency/FLOP/throughput planner 均未找到直接实现。这不是穷尽性证明，只支持一个谨慎
结论：**Lagrangian 本身不是创新；可能可辩护的是把 exact static expert-prefix structure、
train expected routed compute、tail-risk capacity utility 与冻结审计统一为一个双预算问题。**

Qwen3 两个预冻结 anchors 上，Tail-Risk 相对 Route×Tail 分别降低 PPL `3.2849%` 和
`2.1205%`，同时 test routed-compute pruning 分别高 `0.0646/0.0225` 个百分点，形成严格
Pareto dominance。这关闭了“Tail-Risk 只是多算”的内部替代解释，但实验由已见 Qwen3 test
后的公平性问题驱动；在新的未见模型/语料上预注册 anchors 前，双预算只能作为机制增强，
不能单独宣称 ICLR-level novelty 已完成。

## 检索证据

- Web：通过 Agent Reach 的 Jina Reader 读取 arXiv 搜索和论文页面。
- GitHub：通过 `gh` 检索并审阅 `Aaronhuang-778/Mixture-Compressor-MoE` 的
  `precision_solver.py`、`expert_weight.py`、wrapper reconstruction-loss 逻辑；其静态
  ILP 明确使用 expert activation frequency、activated weight 与 quantization loss。
- 2026-07 增量审计：读取 arXiv:2607.01444、EASY-EP、MC-Suite、IFP 页面；通过 `gh`
  审阅 `gucci-j/moe-pruning-reliability` 与 `tylevnovik/qwen-moe-prune` 的方法和协议边界。
- Exa 当前未配置，因此本审计不声称使用 Exa；后续仍需持续追踪 2026 年新投稿。

## Output-Contribution 与 Routing-Entropy 增量审计

- **REAP**（`sroecker/reap`，arXiv:2510.13999）使用
  `mean_active_token(gate × ||expert_output||₂)` 作为 expert saliency，并将其用于 whole-expert
  deletion。本文不能声称首次提出 router-weighted expert output contribution。
- **Qwen hybrid pruning / IFP reproduction**（`tylevnovik/qwen-moe-prune`）强调浅层 routing
  更分散、global top-N 容易误伤，但其公开实现不能使“按路由熵保护层”成为本文创新点。
- 本轮可区分于 REAP 的技术组合是：output contribution 仅作为 Conditional-Dual expert
  utility 的校准因子，最终执行 physical expert 内 64-channel prefix slimming，并继续保留
  tail-aware coverage、sparse rare-risk floors、exact structural block budget 与 expected
  routed-compute anchor；这不是 whole-expert removal。
- 但 train-only selector 已否决 β=`0.25/0.50/1.00` 的乘法融合，routing-entropy prior 也
  4/4 folds 失败。因此该组合目前只能作为经过验证的负结果，不能进入 novelty claim。
- 当前最谨慎的研究结论仍是：可辩护贡献来自 robust Route×Tail fallback、Conditional-Dual
  refinement、rare-risk floors、双预算审计与 train-only applicability selection 的组合；
  output contribution 若再次使用，应限于 sparse anomaly gate 或 selector feature，并重新
  预注册，不能用本轮 test-free negative selection 反推新超参数。

## 2026-07-28 Protected-Core、统一评分与 Functional Redundancy 增量审计

- **How to Score Experts for One-Shot MoE Expert Pruning: A Unified Formulation and
  Selection Principle**（arXiv:2606.15716）已把 expert pruning criteria 统一为 routing
  frequency、gate weighting、activation strength 三因素，并提出 MAN/MSAN。不能把
  frequency/gate/output activation 的任意普通乘积作为创新。
- **Half the Experts, All the Code**（arXiv:2607.16721，`anik-jha/moep`）公开实现不仅包含
  REAP，还包含 `guard_union`：用 reference-domain gate mass 保护每层 top-N 核心 experts，
  其余预算再按 target REAP 填充；同时实现 antithetic half-mask causal ablation。因而
  “给高分 experts 设置不可剪 guard”本身已有直接 whole-expert 先例。
- **Is MoE Routing a Huffman Code?**（arXiv:2607.20427）在 Qwen3.5-35B-A3B 上报告
  functional-profile pairwise cosine similarity >0.90，并用 Subset Difference Pruning：保护
  top-10% 高频 core，再从相似 expert pairs 中删除低频成员。不能把 protected core 或
  similarity-based redundant-expert deletion 当作本文首次提出。
- 本文 Output Safety Floor 的可区分点仅在组合：四折 router-weighted output contribution
  触发 physical-expert 内 64-channel prefix minimum widths，并与 activation-tail floors、
  exact structure 和 expected routed-compute budgets 联合。但正式 WikiText PPL
  `8.689485 > 8.688653`，该组合已被否决，不能进入主 novelty claim。
- 新的可检验缺口是 **unique contribution**：连续 output norm 与 protected-core methods
  都没有在 expert 内 prefix budget 中联合衡量“实际输出幅值”和“相对 co-routed committee
  的功能非冗余度”。若实现，应明确区别于 Subset Difference 的 whole-expert deletion，并
  先用 train-only fingerprint stability/selector 验证。

## Frontier Regret 与逐场景 Compute Non-Inferiority 的最新边界

Frontier Committee Regret 已把 score 从 expert scalar 下沉到 frozen fallback 的 first-pruned
64-channel block，并在 Qwen3 Instruct、Qwen3 Base、Qwen3.5 三个 checkpoint 上同方向降低
PPL。这一区分点强于普通 heterogeneous width 或 protected-core floor，但 diagonal residual
approximation、committee projection 和 Lagrangian/multi-dual optimization 各自都不是可单独
宣称的新算法原语；潜在贡献仍是它们在 exact physical-expert prefix、train-only freezing、
frontier hard floors 与可审计 compute constraints 下的联合 formulation。

最新 8-fold 实验进一步限制 claim：对 8 个 disjoint train route scenarios 逐个施加
non-inferiority 后，profile 在四个全新 holdout 上 PPL 4/4 胜出且 mean compute 略优，但逐折
compute 只有 2/4 不劣。因此不能把 finite-scenario feasibility certificate 表述为 distributionally
robust compute guarantee，也不能声称 prospective Pareto 已完成。更可防守的论文叙事是：
block-level regret signal 具有稳定 PPL 泛化，而 route-cost guarantee 暴露了一个可测量的
train-to-holdout distribution-shift failure mode。

后续若引入 Wasserstein ball、convex-hull expansion、bootstrap UCB、CVaR 或其他 distributionally
robust optimization，必须明确这些都是成熟优化工具；novelty 只能来自针对 MoE physical-expert
prefix cost geometry 的专门 uncertainty construction、exact budget feasibility 与预注册实证，
并需在新的不相邻 holdout 上证明逐折 compute gate，而不能用本轮两个负折反调半径。

Qwen3-30B 的新 train-only 结果提供了首个 prospective 支持：reference-centered coordinate
envelope（8 folds + `range/sqrt(F)` padding）在四个全新 holdout 上同时取得 PPL 4/4 wins、
mean PPL `9.315571 < 9.337112`，以及 routed-compute 4/4 non-inferiority、mean
`0.16308199 > 0.16282288`。这使“finite route constraints 不能覆盖 shift”从失败诊断推进为
可验证的 robust capacity-allocation hypothesis，但不改变相关工作边界：coordinate envelopes、
CVaR/Wasserstein/UCB 和 multi-dual optimization 都是成熟工具；可辩护的技术增量只能是把
reference-centered sign-split envelope 映射到 physical-expert prefix block geometry，并提供
exact-budget feasibility 与冻结 provenance。由于本轮仍是同一 Qwen3 checkpoint 的 train split，
不能声称 architecture-universal 或 distribution-free guarantee。
