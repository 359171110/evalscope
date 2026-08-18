# Static Expert Pruning Ideas

## 2026-07-28 — Dynamic-Regret Distillation

- 状态：机制假设已否决；full PPL 虽优于 route×RMS，但不如去掉 selection indicator 的消融。
- Idea：冻结已验证的动态 APA allocator 作为 teacher。对每个
  `(layer, physical expert, prefix block)`，估计当静态 cap 永久删掉该 block、
  而 teacher 在 train calibration token 上会选中它时造成的期望边际 regret。
- 形式：`v[l,e,j] = E_t[route(l,t,e) * I(width_APA>=j) * utility(l,t,e,j)]`，
  然后在相同 64-channel block 总预算下求解全局 prefix allocation。
- 动机：把 token-conditioned 动态决策压缩为真正可部署的固定结构；不同于
  MoE-Slimming 的静态 attribution coverage，也不同于 MoSE 的多宽预训练。
- 风险：teacher utility 可能只是自身启发式的自蒸馏；必须以 full PPL 严格超过
  RMS、route×RMS、dual-prior×route×RMS 等静态基线才保留。
- 新增查重边界：ICLR 2025 Mixture Compressor 已用 activation frequency、activated
  weight 与 quantization loss 做 expert-wise mixed-bit ILP，再叠加独立的 online dynamic
  pruning。因此“路由频次 + 局部损失 + 全局预算”也不能单独声称创新；本候选必须强调并
  实验证明的是把 token-conditioned prefix allocation 的实际截断事件蒸馏为静态
  channel-width regret，而不是一般的 expert importance 加权。
- 验证结果：Dynamic-Regret full PPL 8.103253，优于 route×RMS 8.175469；但
  uncensored Dynamic Expected-Utility 为 8.093239，反而再优 0.010014。说明
  `I(teacher selects block)` 会删失有用容量信号，不能作为贡献机制。

## 2026-07-28 — Dynamic Expected-Utility Distillation

- 状态：已被更简单且更好的 dual-only 分量取代；保留为必要消融，不再作为主候选。
- Idea：对每个 train token 的 routed physical expert，计算 APA 的 token-conditioned
  parent utility，但不按同预算 teacher selection 截断；对所有 prefix blocks 聚合
  `E[parent_utility × RMS_coverage]`，再做静态全局 prefix allocation。
- 结果：50% 参数结构剪枝下 PPL 8.093239，优于 route×RMS 0.082230、优于
  Dynamic-Regret 0.010014；routed compute 剪枝 7.3489%，略高于 route×RMS 7.0948%。
- 当前创新风险：它可能退化为更复杂的 route-weighted importance。下一步只做
  top-p residual、dual prior、max-combination 三项分解，确认非线性 token utility 的来源；
  若任何简单分量等价或更好，则继续收缩 claim，不包装成虚假 teacher distillation。

## 2026-07-28 — Conditional Dual-Utility Distillation

- 状态：当前主候选；50% 分量消融及 60%/80% scaling full PPL 均已验证。
- Idea：对每个 train token 的 top-8 routed expert set，将 `gate×AMP` 和 `gate×AIMER`
  分别在 set 内归一化，再做几何融合；将 uncensored dual utility 乘 RMS/Hessian prefix
  coverage，按 `(layer, physical expert, block)` 聚合并做等成本全局静态 prefix allocation。
- 50% 结果：PPL `8.079852`，比 route×RMS `8.175469` 低 `0.095616`，也优于 raw gate
  `8.134232`、top-p residual `8.103036`、combined max `8.093239` 和 selection-censored
  Dynamic-Regret `8.103253`；routed compute 剪枝 `7.3696%`，没有靠增加计算量取胜。
- 关键机制证据：静态 dual-route×RMS 为 `8.217790`，说明 AMP/AIMER 本身并不自动有效；
  token-level routed-set conditional normalization 是当前唯一保留下来的解释。
- Scaling：60% 时 `9.008105` vs route×RMS `9.255834`；80% 时 `30.062264` vs
  `31.186814`。候选在三个预算下均同时获得更低 PPL 和略高 routed compute 剪枝率；
  但 80% 绝对质量不可用，只作为机制压力测试。
- 创新风险：该方法仍可能被审稿人视为 route-weighted importance 的细粒度归一化变体。
  在 scaling、同计算对照和跨模型/任务结果完成前，只使用保守命名和增量式 novelty 判断。

## 2026-07-28 — Static Physical-Expert Baseline Suite

- 状态：实现完成，等待完整 PPL。
- 基线：uniform、RMS coverage、route-frequency×RMS、
  geometric AMP/AIMER×route-frequency×RMS。
- 不作为创新点：异构静态专家宽度、64-channel alignment、物理结构裁剪本身已被
  MoE-Slimming 等工作覆盖。

## 2026-07-28 — Tail-Risk Constrained Conditional Utility

- 状态：进行中；手工机制探针已验证，train-only 自动风险统计正在校准。
- Idea：Conditional Dual-Utility 负责常规 token mass 下的期望容量分配；另从 train
  activation tails 自动估计 rare catastrophic-path risk，仅对极少数高风险 physical
  expert-channel prefixes 设置最小宽度 floor，再从低边际 utility blocks 中回收相同预算。
- 机制证据：80% 剪枝时原候选 PPL `30.0623`；保护文献 Super Experts 后为
  `14.9950`，随机等预算保护为 `30.3728`。单专家消融表明恢复几乎完全来自 L2-E92；
  保护一个 64-channel block 已达 `15.0749`，两个 blocks 达 `14.9356`，完整 12 blocks
  为 `14.9092`。route-count 匹配的 L2-E18 为 `30.2631`，排除普通频次解释。
- 自动统计：按 physical expert/channel 收集 `max_t|z_tc|×||W_down[:,c]||_2`，与
  RMS/Hessian score 几何融合；先固定 Conditional Dual 的 expert-level utility，再重绑定
  expert 内 coverage，以隔离“跨专家期望 utility”和“专家内 rare-event risk”。
- 最终目标：以 train-only、预注册阈值自动识别高风险最小 prefix；保持 exact global
  structural budget，不使用已知 expert ID，不以 test PPL 选择阈值。
- 创新边界：Super Experts 已发现 MoE 极端激活专家；SCAR 已保护 dense FFN 的关键
  channels；MoE-Slimming 已覆盖静态异构宽度。因此可辩护的新意只能是 MoE 场景下
  `token-conditional expected utility + automatic rare-event prefix constraints + exact matched
  budget` 的联合 formulation 与机制验证，而不是“保护关键通道”本身。
- 80% full 验证：tail-only λ=0.50 为 `16.2754`，自动 floor-only 为 `14.9464`，联合
  λ=0.50 为 `13.9065`，均优于原候选 `30.0623`；随机 width=2 和 route-count-matched
  width=2 controls 分别为 `29.9870/30.0614`。联合优于手工 L2-E92 full `14.9092`。
- 新风险：early-layer gate 具有文献启发的先验，可能被视为针对 Qwen3 Super Experts 的
  post-hoc rule；需要全层 selector、层尺度归一化和跨模型验证。λ=1.00 的 `15.3899`
  也显示最大值风险不应独占目标，λ=0.50 的典型/尾部平衡才是当前最强机制。
- 全层修订：移除 early-layer gate，统一选择 global risk top 0.5% 的 31 个 experts；
  50%/60%/80% PPL 为 `8.0634/8.6887/13.7617`，均优于 Conditional Dual，且
  60%/80% 优于 early-only。当前默认算法改为全层 selector；early-only 降为消融。
- 当前状态：机制和 scaling 已验证；跨模型/跨校准域尚未验证，仍是“强候选”而非已完成
  ICLR-level claim。

## 2026-07-28 — Calibration-Interval Stability

- 状态：进行中；四个不重叠 WikiText-train 区间正在 GPU 0–3 并行校准。
- Idea：冻结 λ=0.50、global 99.5% quantile、0.1×global-max、minimum width=2 和 60%
  structural budget，只替换 tail-risk calibration interval。区间 offsets 为
  `262,144 / 524,288 / 786,432 / 1,048,576`，每段仍为 262,144 tokens，与原 offset 0
  完全不重叠。
- 目的：量化 risk ranks、selected-set Jaccard 和 full PPL 对校准样本的稳定性；若 L2-E92
  或整体收益只在首段出现，则当前 automatic-tail claim 降级为数据特化。
- 协议边界：Conditional Dual expert utility teacher 暂固定为 offset 0，因此本实验只隔离
  tail coverage/risk selector 的跨区间稳定性，不等同于整个 calibration pipeline 的完全独立复现。
- 文献动机：2026-07 MoE reliability 研究显示平均 utility 不能代表事实可靠性；Qwen3.6
  hybrid pruning 工程也显示 layer routing concentration 高度不均。稳定性必须先于下游扩展。
- 结果：四个 interval PPL 为 `8.7366/8.7897/8.7660/8.6739`，全部优于 Conditional
  Dual `9.0081`；均值 `8.7416±0.0434`。L2-E92 始终 global rank 2，三个关键 experts
  5/5 入选。次级 selected-set Jaccard 0.55–0.68，但 profile width 只有约 4% experts
  改变，说明核心稳定、边缘 max-tail 有噪声。

## 2026-07-28 — Cross-Interval Consensus Tail Risk

- 状态：已验证但不进入主算法；作为 robustness/negative ablation 保留。
- Idea：每个不重叠 train interval 独立应用相同 risk threshold，只给至少 `m` 个 interval
  重复识别的 physical experts 设置 width-2 floor。3/5 选择 27 experts，5/5 选择 21；
  三个 published Super Experts 在两者中都保留。
- 动机：单段 max statistic 对 late-layer 次级 experts 有 0.55–0.68 Jaccard 波动；cross-fold
  voting 保留稳定 catastrophic core，同时删除偶发极值，不改变 offset-0 tail channel coverage。
- 创新潜力：从单次 rare-event detection 升级为 cross-calibrated chance constraint，更接近
  distributionally robust static allocation；但若 PPL 不优于单段 global selector，则只作为
  稳定性/负结果，不包装为主算法。
- 结果：60% 的 3/5、5/5 为 `8.6923/8.6994`，80% 为 `13.7906/13.7805`；均保持
  强收益但略差于单段 global selector `8.6887/13.7617`。因此 consensus claim 正式否决，
  其价值仅是证明 21–27 个跨段稳定 core 已足以维持方法效果。

## 2026-07-28 — Cross-Corpus Tail-Risk Calibration 与 Matched-Domain C4

- 状态：已完成跨语料机制分解与 C4 matched-domain PPL；跨模型、factual reliability
  和 instruction following 仍待验证。
- Idea：冻结主方法的全部超参数与 WikiText-train Conditional Dual teacher，只将
  tail-aware channel coverage 和 automatic risk-floor 的校准语料替换为 C4 train。
  在不接触 WikiText test 的情况下冻结四个 60% exact-budget profiles，再统一运行完整
  WikiText-2 test PPL。
- 动机：若关键 risk proxy 只在目标语料 WikiText 上成立，审稿人可合理质疑它是
  calibration/evaluation domain coupling；跨语料仍识别 L2-E92 并保持 PPL 增益，才支持
  rare-risk structure 更接近模型内生属性而非语料偶然性。
- 预注册协议：C4 本地 train Arrow 源固定并逐文件 SHA256；offsets 为
  `0/262144/524288/786432`，每段 `262,144` tokens；λ=0.50、global quantile=0.995、
  relative-max=0.10、minimum width=2、48 layers eligible、60% structural pruning。
- 解释边界：Conditional Dual expert utility teacher 仍固定来自 WikiText offset 0，故本实验
  只验证 tail-risk 分支的跨语料稳定性，不构成 full-pipeline cross-domain independence；
  不论结果正负都必须报告，且不能据此声称跨模型或 factual reliability 已完成。
- 初步结果与新分解：完整替换 C4 tail branch 的四折 WikiText PPL 均值为 `9.02136`，
  0/4 优于 Conditional Dual，属于负结果。固定 WikiText coverage、只换 C4 risk floors 后
  均值恢复到 `8.69072`；反向固定 WikiText floors、只换 C4 coverage 为 `9.02008`。
  因此 expert-level catastrophic-risk set 跨语料稳定，而 prefix channel coverage 明显依赖域。
- Source-coherent follow-up：为排除“WikiText teacher × C4 coverage”的混源失配，新增
  多数据源 Conditional Dual teacher collector；四个 C4 folds 分别用同源 C4 teacher、
  RMS reference、tail coverage 和 risk floor 冻结 60% profiles，再统一评估 WikiText。
  该 follow-up 为观察负结果后的诊断实验，必须明确标注 post-hoc；即使跨域失败，也不能
  推导它在 C4 本域 PPL 上失败。
- 协议纠错：旧 source-coherent EXP-030–032 实际遗漏 `--parent-mode dual`，teacher 为
  `combined`，故 `11.2977±0.1949` 不能作为 Conditional Dual 或目标 Tail-Risk 证据。
  这些产物保留为协议失败；collector 已强制显式 parent mode，builder 已拒绝非 dual teacher。
- Corrected matched-domain 结果：显式 dual teacher 的四折 C4 60% PPL 均值为 Route×RMS
  `17.29584`、Conditional Dual `17.07693`、Tail-Risk `16.60900`；Tail-Risk 4/4 胜出，
  相对 dual 改善 `2.7401%`，且 routed-compute 剪枝略高。50%/60%/80% fold-0 也分别
  改善 `0.0723/0.4223/16.8073` PPL。
- C4 机制复现：coverage-only `16.89605`、floor-only `16.80301`、combined `16.62821`，
  均优于 dual `17.05048`，且 combined 最佳。说明两组件在第二语料上仍互补。
- 新的保留方向：把算法解释为双角色 factorization——跨语料稳定的 expert-level
  rare-risk floor 是 safety prior；conditional utility 与 prefix coverage 是 domain-adaptive
  allocator。后续优先做 mixed-domain calibration 或 C4 matched-domain PPL，而不是把单域
  channel ranking 包装成 corpus-invariant。状态：双角色 factorization 已获 WikiText+C4
  matched-domain 支持，但仍需第二模型和可靠性任务验证。

## 2026-07-28 — Qwen3 Base Cross-Checkpoint Freeze

- 状态：已完成，预注册成功标准通过。
- Idea：不改 Tail-Risk 的 λ、global risk quantile、relative-max、minimum width、block size
  或 selector 范围，在 Qwen3-30B-A3B base checkpoint 上独立重建全部 train-only caches，
  再比较 60% exact-budget Route×RMS、Conditional Dual 与 Tail-Risk。
- 动机：当前证据来自 Instruct-2507；base checkpoint 能快速检验 post-training 是否改变
  conditional utility、rare-risk core 或方法相对优势，同时保持架构和评估协议不变。
- 成功标准：Tail-Risk 必须严格低于两条基线，且 routed-compute pruning 不低于
  Conditional Dual。失败同样记录，用于判断方法是否被 instruction tuning 特化。
- 边界：即使成功也只称 cross-checkpoint evidence，不能替代 Qwen1.5-MoE/Qwen3.5-MoE
  这类跨架构复现。
- 正式结果：Route×RMS、Conditional Dual、Tail-Risk 的 60% full WikiText PPL 分别为
  `10.623834/10.371195/10.341598`；Tail-Risk 严格优于两基线，同时 routed-compute
  剪枝率最高（`13.9044%`）。因此“同一固定超参数 recipe 能跨 Qwen3 checkpoint
  复现”的 idea 已验证；跨架构状态仍为待验证。

## 2026-07-28 — PPL–Reliability Decoupling Audit

- 状态：已完成，预注册成功标准失败；保留为混合负结果。
- Idea：对同一批已经冻结的 60% profiles 同时报告 PPL、MC1 和 MC2，检查 Tail-Risk 的
  rare-event safety constraint 是否不仅避免语言建模崩溃，也保住 factual answer ranking。
- 文献动机：arXiv:2607.16721 报告 PPL 可把 broken pruned model 排在 intact model 之前；
  arXiv:2607.01444 显示极端 MoE expert pruning 会增加 hallucination。因此 reliability
  不能由 PPL 间接推出。
- 成功标准：Tail-Risk 的 MC1/MC2 均不低于 Conditional Dual；若只改善 PPL 而损伤事实性，
  则现有 tail proxy 需要升级为 task/reliability-aware constraint。
- 边界：TruthfulQA MC 是首个可靠性探针，后续仍需生成式 hallucination 和 instruction
  following 评价。
- 正式结果：Dense、Route×RMS、Conditional Dual、Tail-Risk 的 MC1 分别为
  `0.343941/0.266830/0.270502/0.271726`，MC2 分别为
  `0.518451/0.429845/0.440415/0.437559`。Tail-Risk 比 Dual 多答对 1 题，但 MC2 下降
  `0.002856`，故“双指标均不劣”被否决。
- 新方向：把语言建模 tail-risk 与事实答案概率视为不同风险维度。后续可研究
  reliability-aware calibration、混合任务 calibration 或 practical-point 较低剪枝率；在取得
  直接证据前，禁止把 PPL 改善表述为 hallucination safety。

## 2026-07-28 — Qwen1.5-MoE Cross-Architecture Replication

- 状态：已完成，预注册成功标准失败；保留为跨架构负结果。
- Idea：将完全相同的 Tail-Risk hyperparameters 迁移到 `Qwen2MoeForCausalLM`，保持
  exact 60% structural block budget，并重新生成所有 architecture-specific priors、
  channel caches、dual teacher 和 profiles。
- 架构差异：24×60 routed experts、top-4、1,408-channel experts、22×64-channel blocks，
  `norm_topk_prob=false`，且 sparse block 返回 `(hidden, router_logits)` 并叠加 shared
  expert。该实验能检验方法是否依赖 Qwen3 的 48×128/top-8/768-channel 实现。
- 成功标准：Tail-Risk PPL 严格低于 Route×RMS 与 Conditional Dual，且 routed-compute
  剪枝不低于 Conditional Dual；test PPL 不得修改 profile。
- 正式结果：Dense `7.459207`；Route×RMS `10.114996`；Conditional Dual `10.626735`；
  Tail-Risk `10.239141`。Tail-Risk 明显修复 Dual，却未超过 Route×RMS，因此原“固定
  joint recipe 跨架构成立”的 idea 被否决。
- 新诊断假设：在 Qwen2MoE top-4、`norm_topk_prob=false`、shared-expert 架构中，
  routed-set dual normalization 可能扭曲专家间绝对 gate mass；而 route frequency/RMS
  已更接近最优静态信号。下一步只做预先标注的机制分解，不允许用 test 重新选 λ/quantile。
- 机制分解：tail coverage only `10.243799`，risk floor only `10.621897`，combined
  `10.239141`，对照 Dual `10.626735`。coverage 几乎承担全部修复，floor 在该架构上只
  提供 `0.004658` 的联合边际收益。
- 新 idea（待独立验证）：将典型/尾部 channel coverage 视为相对可迁移组件，expert parent
  则按 train-only routing semantics 选择 route-mass 或 conditional dual。由于该想法来自
  已见 WikiText test，不能再用同一结果验证；必须在 C4 matched-domain 或第三模型上预注册。
- WikiText 探索结果：Route×Tail `10.013247`，优于 Route×RMS `10.114996`，且多剪
  `0.2671` 个百分点 routed compute。状态仍为“探索性支持”，因为方法来自同一 test 后诊断。
- 独立门禁：已在读取 Qwen1.5 C4 validation 前预注册 C4 train matched-domain replication；
  若不能同时改善 PPL 与 routed compute，则 route-parent×tail-coverage 假设否决。
- C4 独立结果：Route×RMS `17.328660`，Route×Tail `16.203895`，改善 `6.4908%`，且
  routed-compute 剪枝增加 `0.1396` 个百分点。预注册门禁通过，状态升级为“已在一个
  Qwen2MoE 模型、两个语料上支持”。
- 当前创新风险：route-weighted activation/channel importance 已有 REAP、MoE-Pruner、
  MoE-Slimming 等先例；可辩护点若存在，只能是 retraining-free typical/tail coverage 的
  factorization、跨架构 failure-driven parent selection，以及 exact-budget/compute evidence。
- Qwen3 C4 breadth：Route×Tail 四折均值 `17.100295±0.093404`，4/4 优于 Route×RMS
  `17.295838`，但略差于 Dual `17.076927`、明显差于 Tail-Risk `16.609000`。
- 修订方向：把 Route×Tail 定位为 architecture-robust base allocation；dual parent 与 sparse
  floors 是 architecture-conditional refinements。要形成单一论文方法，需要 train-only 的
  parent/floor applicability criterion，而不是按 test 结果人工选择分支。

## 2026-07-28 — Train-only Cross-Fitted Refinement Applicability

- 状态：进行中；选择规则与候选 profiles 已在读取 selection-fold PPL 前冻结。
- Idea：不按模型名或已见 test PPL 手工选择 parent。以跨架构更稳健的 Route×Tail 为
  fallback，在多个互不重叠的 train-only selection folds 上比较完整 Tail-Risk refinement；
  只有 refinement 严格多数折获胜且 mean selection PPL 更低时才启用，否则自动回退。
- 动机：Qwen3 上 dual parent + sparse floors 明显有益，Qwen1.5-MoE 上却不如 Route×Tail；
  一个可审计的 applicability criterion 比硬编码 architecture family 更接近统一方法。
- 防泄漏约束：profile calibration 区间、各 selection fold 两两不重叠；profile 与 token
  caches 先冻结并记录 SHA256；selector 只接受 `split=train`、profile frozen、
  `test_metrics_used_for_profile=false` 的输入。
- 第一阶段只做 retrospective diagnostic，检查 train folds 是否复现两个已知 test 排序；
  即使成功也不能称独立验证。第二阶段必须在未读取 test PPL 的第三模型上先选择、再评估。
- 研究风险：小型 train folds 可能有高方差，且 WikiText selection ranking 未必代表跨域
  reliability；若多数投票不稳定，应诚实回退并报告 applicability failure，而不是调阈值。
- Retrospective 结果：Qwen1.5 的四折全部偏好 Route×Tail，mean PPL
  `9.893688 < 10.323414`；Qwen3 的四折全部偏好 Tail-Risk，`8.992431 < 9.289503`。
  两个选择均与已存在 formal test 排序一致，说明 criterion 在当前架构对上具有稳定诊断力。
- 当前状态：第一阶段已验证；第二阶段“第三模型先选择后 test”的 prospective validation
  待执行。此前 test 排序已知，因此禁止把第一阶段包装成 independent generalization。

## 2026-07-28 — Qwen3.5 Prospective Applicability Validation

- 状态：已验证；train-only selector 在 test 前选择 Tail-Risk，formal full-corpus test 上
  Tail-Risk `7.834395` 严格优于 Route×Tail `8.062748`，相对改善 `2.8322%`。
- 模型：`Qwen3.5-35B-A3B`，40×256 routed experts、top-8、512-channel experts、shared
  expert，且 decoder 混合 linear attention 与 full attention；比已有 Qwen3/Qwen2MoE
  验证提供新的 expert scale 与 backbone dynamics。
- Idea：完全沿用已冻结的 Route×Tail fallback、Tail-Risk refinement、λ=0.50、60% exact
  block budget 与四折 selector，不根据该模型 test 调参。先在 train folds 选择并冻结，再
  同时评估两个候选，验证选择是否真的命中较低 test PPL 分支。
- 运行时门禁：native text forward、全宽 exact equivalence 与 7/8-block sparse path 已通过；
  正式 128×2048 calibration、两个 exact-budget profiles 与四折 selector 均已冻结。
- 失败解释：若 selector 选错，说明 train-fold applicability criterion 不足以泛化到 hybrid
  architecture；必须作为核心负结果报告，不能再增加 test-derived threshold。
- 正式结果：selector 4/4 选择 Tail-Risk；两候选均为 `32,768/81,920` blocks。Tail-Risk
  相对 Route×Tail 降低 `0.228353` PPL，但 routed-compute 剪枝低 `0.1692` 个百分点，
  因此成功限于 matched structural budget，仍需 compute-matched follow-up。
- 协议审计：Qwen3.5 tokenizer 对同一 filtered full WikiText test 得到 `231,940` tokens，
  不是 Qwen3 tokenizer 的 `233,368`。首次误标产物保留，修正为冻结 model-tokenizer cache
  后三路指标完全复现；这属于 post-reveal metadata/audit correction，不改变 selector 成功结论。
- 下一 idea：把“fallback/refinement applicability”与 routed-compute 预算联合起来，研究
  compute-constrained cross-fitted selector；Qwen3 retrospective 双预算机制实验已率先通过，
  但独立 prospective 版本仍待验证，严禁用已见 test 调 threshold。

## 2026-07-28 — Exact-Structure + Expected-Compute Dual-Budget Allocation

- 状态：Qwen3 retrospective mechanism audit 已验证；两个预冻结 compute anchors 均取得
  Tail-Risk 对 Route×Tail 的严格 PPL–compute Pareto dominance。
- Idea：结构参数预算与实际 routed compute 是不同约束。保持 exact 64-channel structural
  block budget，同时用 train route counts 定义 expected compute，通过 Lagrangian multiplier
  调整 block marginals；同一 expert 的 block cost 恒定，因此保留 prefix 单调性与硬 risk floors。
- Route anchor：Tail-Risk `8.701386/14.9328%` 对 Route×Tail
  `8.996928/14.8682%`（PPL/compute pruning），PPL 改善 `3.2849%` 且少算更多。
- Tail anchor：Tail-Risk `8.688653/15.2451%` 对 Route×Tail
  `8.876885/15.2226%`，PPL 改善 `2.1205%` 且仍少算更多。
- 机制结论：Tail-Risk 的收益不是把结构 blocks 集中给热门 experts 后多执行计算，而是更好的
  risk-aware capacity placement。Route×Tail 在更强 compute pruning anchor 下自身 PPL 也从
  `8.996928` 改善到 `8.876885`，说明 compute calibration 本身能纠正部分频率偏置；但两个
  anchors 上 Tail-Risk 仍明显占优。
- Novelty 风险：Lagrangian 双约束是标准工具；MoE-Slimming 已有结构 coverage 最大化，
  EvoESAP 已有 fixed expert-count budget 下的非均匀搜索，TENP 报告 activated expert
  parameters。当前定向审计未发现同时约束 exact expert-neuron structure 与 train routed
  compute、并保持 static expert-prefix topology 的直接等价实现。状态需继续查重和独立复现。
- C4 replication：两个 anchors 上 Tail-Risk PPL 仍改善 `2.0290%/2.1639%`，但 validation
  compute pruning 分别少 `0.0898/0.1125` 个百分点，strong Pareto 失败。新诊断是单 fold
  route counts 存在 train→validation distribution shift；下一轮改为 multi-fold route consensus，
  不能用 validation mismatch 直接反推 multiplier。
## 2026-07-28 — Multi-fold Route-Consensus 未能关闭 C4 Compute Shift

- 状态：已验证（primary PPL 通过，strong Pareto 否决）。
- 方法：四个不重叠 C4 train folds 的 normalized route-count 均值，构造 exact structural
  budget + expected routed-compute 双预算 profiles；两个 anchors 为 `15.947146%` 与
  `16.463321%`，均在 validation 前冻结。
- 结果：Tail-Risk 在两个 anchors 上 PPL 分别优于 Route×Tail `2.1063%/2.1001%`，但
  validation routed-compute pruning 少 `0.1012/0.1158` 个百分点。
- 结论：平均 route consensus 能保留 PPL 优势，却不足以提供 compute-risk 保证；不能宣称
  C4 Pareto replication 成功。
- 下一步：只在 train folds 上预注册 worst-case/CVaR compute calibration，或给 cross-fitted
  selector 增加 compute-risk applicability gate；禁止用 validation mismatch 调整 multiplier。

## 2026-07-28 — Robust-Consensus Follow-ups 被 WikiText PPL 否决

- CVaR/worst-case route cost：CVaR(α=0.5) 在两个 anchor 上退化到 `9.390907/8.991009`，
  worst-case 退化到 `9.714727/9.714727`；说明 compute-risk 上尾惩罚过度抑制了目标域有用容量。
- Mean coverage consensus：`8.821189/8.781863`；mean teacher+coverage consensus：
  `8.810717/8.759046`。简单跨折平均不能替代 offset-0 domain-adaptive signal。
- Teacher LCB（mean−κ·std）在 train-only selection 上 κ=0.25 仅 2/4 胜出、均值更差，selector
  正确回退到 offset-0 Tail-Risk；没有打开 LCB full test。
- 当前 formulation 收敛：**domain-adaptive Tail-Risk profile + train-only applicability gate**。
  Robustness 应作为是否启用 refinement 的选择问题，而不是强行平均进静态 utility/cost。
- 状态：以上三条内部改造均已验证并否决，负结果保留，避免重复搜索。

## 2026-07-28 — Routing-Entropy Layer Prior

- Idea：用 train route distribution 的层级归一化熵构造 mean-one placement prior；正 gamma
  保护路由分散层，负 gamma 检验反向假设，不改变 exact structural block 数。
- 动机：公开 Qwen hybrid pruning 实现显示 global top-N 容易误伤高路由熵浅层。
- 结果：`gamma=+0.5/+1.0/-0.5` 的四折 mean train PPL 分别为
  `9.051775/9.037242/9.059306`，均差于 fallback `8.992431`，且 fallback 4/4 胜出。
- 状态：已否决；不得为这些候选打开 full test。层路由熵可作为诊断统计，但不能直接作为
  Tail-Risk capacity utility prior。

## 2026-07-28 — Contribution-Calibrated Tail-Risk

- Idea：把 REAP 风格的 `mean_active_token(gate × ||expert_output||₂)` 从 whole-expert
  removal 改造为 Conditional-Dual expert utility 的校准因子，再使用本文的 tail-aware
  within-expert prefix coverage、rare-risk floors 和 exact structure/compute budgets。
- 实现：四个 disjoint train folds 逐折、逐层归一化后求均值；候选 β=`0.25/0.50/1.00`。
- 结果：四折 selection mean PPL 为 `9.032436/9.039200/9.115010`，均高于 fallback
  `8.992431`；fallback 3/4 胜出，selector 回退，未读取 candidate formal test。
- 机制解释：输出贡献与 Conditional-Dual utility 存在部分重复或 scale mismatch；连续乘法
  会过度重排 expert capacity，β 越大退化越明显。
- 状态：当前乘法 formulation 已否决。可保留的新方向是把 output contribution 仅用于
  sparse anomaly floor、置信度估计或 applicability selector 特征，而不是全局重权 utility。

## 2026-07-28 — Output-Contribution Safety Floor

- Idea：不再连续乘入 utility，只把四折 output contribution 全局 99.5% 尾部变成 2-block
  safety floors，并与原 activation-tail floors 取并集。
- Train selection：3/4 folds 获胜，mean `8.988168 < 8.992431`，因此按预注册门禁打开一次
  formal test。
- Formal result：PPL `8.689485`，略差于 fallback `8.688653`；compute pruning 略高
  `0.00201` 个百分点。primary 与 strong Pareto 均失败。
- 状态：已否决。不得根据该 test 继续搜索 quantile/min-width；该实验说明小幅 train PPL
  majority 仍可能是假阳性。

## 2026-07-28 — Redundancy-Aware Unique Contribution（下一候选）

- 动机：REAP/output norm 无法区分独特贡献与功能克隆；Frequency-Diversity Law / Subset
  Difference 在 Qwen3.5 中发现 co-activation functional profiles 的 pairwise cosine similarity
  可超过 0.90，说明高幅值 expert 也可能只是冗余副本。
- Idea：在 train-only collector 中记录每个 physical expert 的 co-routing context fingerprint，
  以 `1 - max cosine similarity` 或 residual-to-nearest-representative 衡量 committee uniqueness；
  只保护“高 output contribution × 高 uniqueness”的 expert-prefix safety floors。
- 预期：避免 q99.5 output norm 把高幅值 clone 误判为必须保护，同时保留 rare/unique experts。
- 实现：累计 `sum_token(dense_topk_gate_outer_product)`，去对角后计算最近邻 cosine；每折
  output contribution×uniqueness，再取四折 minimum 作为 robust score。固定 q99.5、2-block
  floor、relative-max=0。
- 结果：候选四折 PPL 全部退化，mean `8.997872 > 8.992431`，0/4 wins；selector 回退，
  未打开 full test。
- 状态：已否决。expert-level redundancy correction 仍未解决“整 expert 标量无法映射到内部
  channel-prefix demand”的层级错配；后续不得继续调此类 floor quantile。

## 2026-07-31 — Official REAP vs. Route×Tail / Tail-Risk Matched-Budget Matrix

- Idea：改用 `CerebrasResearch/reap` 官方实现，在同 checkpoint、同 train-only calibration
  cache、同 routed-expert FFN 参数预算下比较 REAP whole-expert pruning 与 64-channel
  Route×Tail/Tail-Risk prefix shrinking；同时增加 per-layer budget 受控组，分离 global
  allocation 与 prefix geometry 的贡献。
- 关键审计：当前 `third_party/reap` 是非官方 `sroecker/reap`，旧 59.375% Qwen3 结果只保留
  为工程参考；官方实现需要启用 top-k gate renormalization。
- 模型资格：Qwen3-30B 可直接进入；Qwen3.6 是 40×256 routed、top-8+1 shared、512-channel
  experts，需要官方 REAP adapter；第三个目标模型固定改为 Gemma 4 26B A4B MoE，本地路径
  待提供，收到后从 checkpoint 自身审计拓扑、shared expert、block divisibility 和 adapter。
- Benchmark 决策：当前没有原生覆盖 MoE 剪枝预算公平性的公用平台。采用固定 commit 的
  `lm-evaluation-harness` 统一 text-only 能力任务，结构预算、routed compute、checkpoint
  bytes、shared expert 和 SHA256 继续由 V4 validator 审计；不采用面向多模态的 `lmms-eval`
  作为主平台，也不在主表混用多个 harness。
- 计划：先跑 Qwen3 50% C1 single-seed 最小反证，再扩展 Qwen 双模型 25%/50%/约60%、C2
  论文规模校准、C4 与生成任务；Gemma 26B A4B 在路径到位并通过独立门禁后作为第三分支。
  完整预注册见
  `docs/REAP_ROUTE_TAIL_TAIL_RISK_EXPERIMENT_PLAN.md`。
- 状态：计划已建立；尚未 checkout 官方实现、生成新 profiles 或读取新正式评价结果。

## 2026-07-28 — Channel-Conditioned Committee Regret（后续候选）

- Idea：不再把 committee/output evidence 压缩为 expert scalar；直接估计 routed token 上每个
  64-channel block 的 counterfactual residual contribution，例如移除该 block 后相对同委员会
  其他 experts 无法补偿的输出残差，再与现有 Conditional-Dual/tail coverage 联合。
- 动机：Contribution β、output safety floor、unique-contribution floor 三次负结果共同表明，
  expert-level contribution 与 within-expert structural demand 存在稳定层级错配。
- 预期：把功能冗余判断落到实际被剪的 block 层级，避免保护高贡献 expert 中并不重要的
  channel prefixes。
- 实现：使用 diagonal down-Gram 近似，估计每个 64-channel block 相对同 token 其他 routed
  experts 委员会输出方向的正交残差；只保护 frozen fallback 的 first-pruned frontier block。
- Train selector：固定 worst-fold minimum + global q99.5，候选 4/4 folds 获胜，mean
  `8.974187 < 8.992431`。
- Formal：PPL `8.658925 < 8.688653`，相对改善 `0.34214%`；test compute pruning 少
  `0.01003` 个百分点，primary 成功、strong Pareto 未通过。
- 状态：已验证为当前新最佳 PPL。下一步是独立模型/matched-domain 的预注册双预算复现；
  禁止根据本次 test compute 差值反调 multiplier 并包装成 prospective 结果。

## 2026-07-28 — Frontier Committee Regret 的 Base 独立复现与双预算后续

- 独立性：在 Qwen3-30B-A3B Base checkpoint 上重新采集 estimator folds、selector caches，
  并使用 Base 自有 teacher/channel/profile；不复用 Instruct frontier cache 或 widths。
- Train selector：candidate 4/4 folds 获胜，mean `10.678721 < 10.708844`。
- Formal：PPL `10.322812 < 10.341598`，相对改善 `0.18166%`；test routed-compute pruning
  少 `0.02514` 个百分点，primary 通过、strict Pareto 未通过。
- 机制结论：block-level committee residual 的 lower-PPL 方向跨 Instruct/Base checkpoint
  复现，明显强于 expert-level output/uniqueness floors 的负结果；但小 compute concession 在
  两个 checkpoint 都存在。
- 新 idea：下一轮只允许在未见 evaluation 的独立 domain/checkpoint 上预注册两组 train-only
  compute anchors，检验同一 frontier floors 在不降低 routed-pruning 时是否仍保留 PPL 收益；
  不允许用本次 `0.02514` 个百分点 test mismatch 反调 multiplier。
- 状态：Base lower-PPL 复现已验证；prospective strict Pareto 仍待验证。

## 2026-07-28 — Qwen3.5 跨 256-Expert Topology Frontier 迁移

- Idea：保持完全相同的 minimum-fold/q99.5/one-block frontier rule，不针对 Qwen3.5 调整
  quantile，在 40×256 topology 上检验 block-level committee residual 的结构可迁移性。
- Selector：3/4 folds 获胜，mean `8.001427 < 8.008323`；唯一负折只退化 `0.000262`。
- Secondary full-corpus：PPL `7.820017 < 7.834395`，相对改善 `0.18352%`；routed-pruning
  少 `0.03644` 个百分点。
- 结论：lower-PPL 机制已跨三 checkpoint、两种 expert-count topology 同方向复现；但三次
  evaluation 都存在小 compute concession，说明下一创新点应聚焦 train→evaluation route-cost
  shift，而不是继续搜索 frontier quantile。
- 状态：跨拓扑 lower-PPL 已验证；Qwen3.5 结果因 tokenizer 为 231,940 tokens，仅作为
  secondary evidence，不作为标准 233,368-token formal claim。

## 2026-07-28 — Per-Fold Compute Non-Inferiority Frontier Allocation

- 诊断：四折 mean route consensus 保留 3/4 PPL wins，并把 mean compute deficit 缩小
  `46.23%`，但四折 compute delta 仍全部为负。
- 新 idea：在下一独立 checkpoint/domain 的 profile allocator 中，把每个 train route fold 的
  candidate retained-cost 约束为不高于 fallback，而不是只约束平均 route distribution；目标是
  exact structural budget + per-fold compute non-inferiority + frontier floors 的多约束分配。
- 预期：避免少数 route distribution shift 让 q99.5 frontier floors 系统性落到更热门 experts，
  同时保留 block-level committee residual 的 PPL 收益。
- 实现：已加入 exact multi-scenario prefix allocator、逐折 feasibility audit 与不可行拒绝写出；
  focused tests 31 passed。四个原 estimator constraints 全部满足。
- 原四折结果：candidate mean PPL 与 mean compute 同时改善，PPL 3/4 wins、compute 3/4
  non-inferior，但 strong 4/4 compute gate 失败。
- 状态：部分验证；不声称 Pareto，不运行 test。

## 2026-07-28 — Route-Distribution Uncertainty Set（由 8-Fold 负结果触发）

- 诊断：把 8 个已见 train route folds 全部设为 hard non-inferiority constraints 后，profile
  在约束上 8/8 可行；但四个全新 holdout 虽 PPL 4/4 胜出、mean compute 略优，逐折 compute
  仅 2/4 不劣。
- Idea：不要继续简单增加相邻 folds；用 train route vectors 构造 uncertainty set，例如
  layer-aware convex hull 扩张、bootstrap upper confidence bound、Wasserstein/CVaR stress
  scenarios 或显式 per-layer route-mass perturbation，并对 worst plausible retained cost 约束。
- 动机：当前约束是有限样本场景保证，不是分布外保证；route shift 的量级约 `1e-4`，需要把
  estimation uncertainty 与真正的 compute regression 区分。
- 预期：保持已在 unseen folds 4/4 复现的 PPL 收益，同时把 unseen per-fold compute
  non-inferiority 从 2/4 提升到至少 3/4；必须预注册 tolerance 与新不相邻 holdout。
- 状态：待验证。禁止根据本轮 holdout 的两个负折直接拟合 perturbation 或运行 test。

## 2026-07-28 — Reference-Centered Route Envelope

- Idea：以 frozen fallback width 为中心，对 reference 内 blocks 使用 observed train route 的
  coordinate-wise lower envelope，对 reference frontier 及以上 blocks 使用 upper envelope；
  用 `range/sqrt(F)` 做预注册 padding，并将该 block-dependent cost 接入 exact prefix allocator。
- 动机：8 个 finite route scenarios 的 hard non-inferiority 在 Qwen3.5 新 holdout 上仅 2/4
  compute noninferior；问题不是 frontier PPL signal，而是跨折坐标极值组合未被约束。
- 实现：成本在 expert prefix 内单调不减，候选-minus-reference cost 是可审计的 conservative
  separable upper bound；Qwen3-30B 使用 8 observed folds + 1 envelope constraint，全部满足。
- 新 holdout：route-envelope PPL 4/4 胜 Tail-Risk fallback，mean `9.315571 < 9.337112`；
  routed-compute 4/4 不劣，mean `0.16308199 > 0.16282288`；相对 nominal Frontier 也取得
  3/4 PPL wins 与 4/4 compute noninferiority。
- 状态：已验证（Qwen3-30B train-only prospective gate）；下一步必须在独立 checkpoint/domain
  预注册半径，不得声称 `1/sqrt(F)` 是普适保证。

## 2026-07-28 — Qwen3 Base Route-Envelope Independent Replication

- Idea：在独立 Qwen3 Base checkpoint 上冻结同一 reference-centered route envelope，检验
  strict train-only PPL–compute gate 是否可跨 checkpoint 迁移。
- 结果：四个新 train holdout 上相对 Tail-Risk 4/4 PPL wins，mean PPL 改善 `0.035987`，
  mean routed-compute pruning 提升 `0.000109`，compute 4/4 non-inferior；primary/strong
  gate 均通过。
- 边界：相对 nominal Frontier，mean PPL 退化 `0.007668`，因此 envelope 是 compute-robust
  candidate，不是全面 Pareto dominance；未运行 validation/test。
- 状态：已验证（跨 Qwen3 Instruct/Base 的 train-only replication）；下一步仍需独立
  matched-domain 或非 Qwen 架构预注册复现。

## 2026-07-28 — 静态 MoE 路由包络与委员会后悔度专利扩展构思

- Idea：围绕 physical expert 的连续前缀结构，把冻结参考 profile、首个被剪 block 的
  Frontier Committee Regret、reference-centered sign-split route envelope、Tail-Risk 最小宽度
  与精确全局 block 预算组合为可审计的静态异构缩宽流程；profile 在读取 evaluation split
  指标前冻结，不依赖线上逐 token 动态改变专家结构。
- 区分化：创新重点不是一般性的“给不同专家分配不同宽度”，而是以 reference profile 为
  中心区分缩窄侧与扩宽侧的路由成本边界，并用委员会反事实后悔度评价 frontier block；当
  多场景 routed-compute 约束不可行时，系统拒绝候选 profile 并回退到带哈希的参考 profile。
- 新扩展 1：为不同硬件、时延上限和显存上限建立可哈希的多硬件 profile bank；部署时只在
  批次边界从冻结 profile 集合中切换，不在单个 token 内动态剪枝。
- 新扩展 2：构造共享通道排序的嵌套静态 profile，使较小 profile 是较大 profile 的结构子集，
  从而支持版本化增量恢复、差分传输及低成本回退。
- 新扩展 3：保留若干不计入默认激活宽度的 shadow frontier blocks；当仅由 train-only 路由
  漂移统计触发预注册阈值时，在批次边界切换到包含对应影子 block 的已冻结 profile。
- 新扩展 4：将单步 frontier regret 推广为多步反事实委员会后悔度，评估连续删除多个 block
  后 teacher-student 差异的非线性累积，同时保持 physical expert、64-channel block 和精确结构
  预算索引不变。
- 预期：提升跨域路由漂移下的静态 profile 稳健性、回退可解释性及多设备部署复用能力。
- 状态：专利构思已形成；除既有 reference-centered route envelope 相关结果外，上述 profile
  bank、嵌套结构、shadow frontier、版本化恢复及多步反事实机制均待独立实验验证，不作为
  已验证效果陈述。
