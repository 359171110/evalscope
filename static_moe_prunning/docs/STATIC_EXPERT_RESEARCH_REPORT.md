# Static Expert Pruning for Qwen3-30B-A3B

> 状态：原 Dynamic-Regret 机制已被消融否决；Conditional Dual-Utility 奠定 expected
> utility 基线。当前主候选 **Tail-Risk Constrained Conditional Utility** 已完成
> 50%/60%/80% full-PPL、机制分解、随机/频次控制、全层 selector 和四个不重叠 train
> 区间复现，并完成 C4 跨语料机制解耦、C4 matched-domain 正式复现及 Qwen3 Base
> 跨 checkpoint 正式复现。Tail-Risk 在 WikiText 与 C4 两个匹配域、Instruct-2507 与
> Base 两个 Qwen3 checkpoints 均优于 Conditional Dual。TruthfulQA 首个直接可靠性探针
> 已完成但未通过预注册标准；Qwen1.5-MoE 跨架构复现也未通过预注册标准，因此仍不能
> 声称 ICLR-level 创新已最终成立。新增 train-only cross-fitted selector 已在 Qwen1.5 与
> Qwen3 上分别 4/4 选择 Route×Tail fallback 与 Tail-Risk refinement，并复现已知 test
> 排序；真正 prospective 的 Qwen3.5 验证已在读取 test PPL 前预注册并开始正式校准。

## 1. 目标与协议

本项目研究 post-training 静态物理专家缩宽：对 Qwen3-30B-A3B-Instruct-2507
的每个 routed expert，把 768 intermediate channels 按 train-only RMS/Hessian 排名后
切为 12 个 64-channel prefix blocks，并冻结：

\[
w_{l,e}\in\{0,1,\ldots,12\}.
\]

`e` 始终是 physical expert ID，不是 router rank。首轮统一保留 36,864/73,728
blocks，即 routed-expert 参数结构剪枝率 50%。正式评价是 WikiText-2 raw-v1 test
完整 114 windows、233,368 tokens、context 2048。profile 只使用 train 校准并在
读取 test PPL 前冻结。

## 2. 当前实现

- 精确 equal-cost 全局 prefix allocation。
- uniform、RMS、route×RMS、dual-prior×route×RMS 静态 profiles。
- 固定 physical-expert width 的 runtime structural emulation。
- 结构参数剪枝率与 routed compute 剪枝率分离报告。
- profile/cache SHA256、train-only provenance 和 mission validator。
- floor-free APA teacher 的 Dynamic-Regret 收集与静态 distillation profile 构建。
- per-expert 最小 prefix floor、exact-budget 重新分配与 protected-profile provenance。
- train-only activation-tail channel calibration、tail/RMS 几何融合和 utility rebinding。

当前完整测试套件为 **72 passed**，覆盖 exact budget、prefix、0-width、CUDA device boundary、dense
equivalence、profile leakage/budget validator、risk floors、calibration offsets、
cross-interval consensus、跨语料 Arrow/provenance、coverage/risk source 解耦、任意语料
冻结 token-cache PPL、显式 parent-mode 约束、cross-fit selection/hash audit、Qwen3.5 模型族
推断与 full-width exact-native boundary。

## 3. 50% 静态基线 Full PPL

| 方法 | Full PPL | 结构参数剪枝率 | Routed compute 剪枝率 |
|---|---:|---:|---:|
| Uniform width=6 | 12.398542 | 50.0% | 50.0% |
| RMS heterogeneous | 12.700110 | 50.0% | 47.14% |
| Route frequency × RMS | **8.175469** | 50.0% | 7.09% |
| Geometric AMP/AIMER × route frequency × RMS | 8.217790 | 50.0% | 7.06% |
| Dynamic-Regret Distillation | **8.103253** | 50.0% | 8.64% |
| Dynamic Expected-Utility Distillation | **8.093239** | 50.0% | 7.35% |
| Raw-gate Expected Utility | 8.134232 | 50.0% | 7.09% |
| Top-p Expected Utility | 8.103036 | 50.0% | 7.48% |
| **Conditional Dual-Utility Distillation** | **8.079852** | 50.0% | **7.37%** |

当前结论：

1. RMS coverage 单独决定跨 expert 容量不成立；它比 uniform 退化 0.30157 PPL。
2. 路由频次是 50% 静态参数预算下的主导信号。
3. AMP/AIMER geometric prior 在 route×RMS 上未带来收益，退化 0.04232 PPL。
4. route-aware profiles 通过将大量冷专家设为 width=0、热门专家保留 width=12，
   把 routed compute 保持在近 dense 水平。因此 8.175469 是强参数压缩基线，不能解释为
   50% FLOPs 剪枝结果。
5. Dynamic-Regret 比最强 route×RMS 低 0.072215 PPL（相对 0.8833%），且 routed
   compute 剪枝率高 1.546 个百分点，因此改善不是靠增加执行宽度获得。
6. 去掉 teacher-selection indicator 后，PPL 进一步降到 8.093239。原本的
   “动态 admission/truncation 事件蒸馏”解释被实验否决；indicator 会删失静态容量所需的
   uncensored utility signal。
7. Parent utility 分解显示 raw gate、top-p residual、`max(top-p, dual)` 均不如纯
   geometric dual utility。最终候选为 8.079852，比 route×RMS 低 0.095616 PPL，且
   routed compute 剪枝率略高 0.275 个百分点。

## 4. 修订主候选：Conditional Dual-Utility Distillation

在 WikiText-2 train 的每个 token/layer 上，对 top-8 routed expert set 分别计算并归一化
`gate×AMP` 与 `gate×AIMER`，再做几何融合。该 utility 不经过 top-p residual、动态
block allocator 或同预算 teacher selection mask。对每个 routed physical expert/block 聚合：

\[
v_{l,e,j}=\mathbb E_t[\mathbf 1(e\text{ routed})
\sqrt{\bar u^{AMP}_{l,t,e}\bar u^{AIMER}_{l,t,e}}\,c_{l,e,j}],
\]

其中 `c` 是 RMS/Hessian prefix-block coverage。随后在相同 36,864-block 静态参数预算下
做一次精确全局 prefix allocation。与 route×RMS 的关键差异是 dual prior 在每个 token
实际 routed set 内分别归一化并融合，而不是先按 expert 全局平均后再乘 route count。

修订候选实测为 8.079852282581943，低于 route×RMS 8.175468584805818、被否决的
Dynamic-Regret 8.10325319409488，以及 combined expected utility 8.093238821101226。
静态 dual-route×RMS 为 8.2177904137，说明收益不能由“加入 AMP/AIMER 静态先验”解释；
token-level routed-set conditional normalization 是当前最有证据的机制。所有 profile 均在
full 前冻结，未用 smoke/full PPL 修改。

## 5. 创新性边界

- MoE-Slimming 已覆盖 attribution-guided channel scoring、全局异构 width allocation、
  physical structural pruning 和硬件对齐。因此静态异构宽度不是创新。
- FlexMoE 已覆盖 Taylor channel ranking、通道重排、expert prefix slicing、每专家离散宽度
  action 和多预算 nested subnetworks。因此 prefix 静态缩宽、nested expert 或离散宽度本身
  也不是创新；本项目必须依靠 retraining-free conditional utility 与 rare-risk constraints。
- MoSE 已覆盖 nested/slimmable experts 和 router-conditioned runtime width；本项目不能
  声称首次动态选宽度。
- POP 已覆盖 retained/candidate/pruned partition 和 online context-conditioned pruning；
  static-cap + dynamic-tail 不是天然新颖。
- REAP 已覆盖 router-weighted expert activation pruning；route awareness 本身不新。
- MAESTRO 已使用跨层 routing trajectories；trajectory 本身不新。
- DTop-p MoE 已研究可学习 top-p threshold、全局稀疏控制和动态 expert count；若
  top-p residual 分量最终胜出，本项目只能主张其 post-training static channel-capacity
  distillation，而不能主张首次 top-p/cumulative-mass routing。
- FLAP 已覆盖 retraining-free fluctuation-based global structured pruning 和 compensation；
  one-shot global allocation 本身不新。
- MoE-Pruner 已覆盖 router-weighted Wanda-style pruning 与可选 expert-wise KD；
  router×importance 或“蒸馏”标签本身不新。
- Mixture Compressor（ICLR 2025）已将 activation frequency、activated weight、
  quantization loss 组合为 expert significance，并通过 ILP 分配静态 mixed bit-width，
  另叠加 online dynamic pruning。因此“频次 + 压缩损失 + 离散预算”也不是创新。
- 最新跨模型 domain-pruning 研究还显示最佳 selector 会在 Qwen3.6 与 Gemma-4 间翻转，
  且 PPL 可能把行为损坏的模型排在完整模型之前。因此单 checkpoint PPL 不足以支持通用
  recipe 或能力保持；Qwen3 base 和 TruthfulQA follow-up 已分别预注册。

当前最小可辩护候选是：把每 token routed-set 内条件归一化的 uncensored dual utility
聚合并蒸馏成固定 physical-expert channel capacity。原 admission/truncation-event 与
top-p claim 均已被消融否决。它是否足够达到 ICLR 级别，仍取决于同计算对照、更多
模型/任务外推和最终查重；现阶段更准确的判断是“有稳定改进迹象，但
相对 REAP、Mixture Compressor、MoE-Slimming 仍可能只是增量式 scoring 改进”。

## 6. Scaling 结果

| 结构剪枝率 | Route×RMS PPL | Conditional Dual-Utility PPL | 绝对改善 | 相对改善 | 候选 routed compute 剪枝率 |
|---:|---:|---:|---:|---:|---:|
| 50% | 8.175469 | **8.079852** | 0.095616 | 1.1695% | 7.37% |
| 60% | 9.255834 | **9.008105** | 0.247729 | 2.6765% | 14.89% |
| 80% | 31.186814 | **30.062264** | 1.124551 | 3.6059% | 45.24% |

三个预算均为 matched structural blocks、correction=none，且候选 routed compute
剪枝率分别比 route×RMS 高约 0.275、0.458、1.241 个百分点。因而改进不是用更多执行
通道换来的。相对改善随剪枝率上升，但 80% 时两方法绝对 PPL 都严重恶化；该点只能说明
排序在强压缩压力下仍有优势，不能作为实用部署点。

## 7. 强压缩失效定位：L2-E92 的最小关键 Prefix

已发表 Qwen3 Super Experts 在原 Conditional Dual profiles 中的宽度如下：

| 结构剪枝率 | L1-E68 | L2-E92 | L3-E82 |
|---:|---:|---:|---:|
| 50% | 1 | 1 | 6 |
| 60% | 1 | 0 | 1 |
| 80% | 1 | 0 | 1 |

将三者完整保护且从其他 blocks 回收相同预算，在 50%/60%/80% 的 PPL 分别为
`8.0850/8.9253/14.9950`；对应未保护为 `8.0799/9.0081/30.0623`。因此固定硬保护在
50% 有害，但在 80% 避免灾难崩溃。随机 early-layer 三专家等预算保护在 60%/80% 为
`9.0145/30.3728`，不能复现收益。

80% 单专家消融进一步定位到 L2-E92：L1-E68、L2-E92、L3-E82 完整保护分别为
`30.5585/14.9092/29.9945`。L2-E92 width 1/2/4/8/12 的 PPL 为
`15.0749/14.9356/14.9147/14.9076/14.9092`，表明第一个 64-channel block 恢复绝大
多数质量，第二个 block 后基本饱和。在 60% 下，width 1/2 也达到 `8.9021/8.8944`。
train route count 接近的 L2-E18（939 vs E92 的 948）完整保护后仍为 `30.2631`，排除
普通 route frequency 解释。

这揭示了 expected utility 的结构性盲点：均值排序可把低频但高损失风险路径删成 width 0，
而只需极小 critical prefix 即可避免崩溃。80% 保护同时降低 routed-compute 剪枝率约 2 个
百分点，故现阶段它是机制证据，不是同计算胜利。

## 8. 当前新候选：Tail-Risk Constrained Conditional Utility

候选把两个目标明确分工：

\[
U_{l,e,j}=\mathbb E_t[\mathbf 1(e\text{ routed})\,
\sqrt{\bar u^{AMP}_{l,t,e}\bar u^{AIMER}_{l,t,e}}\,c^{RMS}_{l,e,j}]
\]

负责常规 token mass 的期望效用；另以 train-only

\[
R_{l,e,c}=\max_t |z_{l,e,t,c}|\,\|W^{down}_{l,e,:,c}\|_2
\]

估计 rare-event channel risk。第一阶段用 tail-aware ranking 重绑定 expert 内 coverage；
第二阶段使用预注册、仅依赖 train cache 的 sparse risk floors。所有 floor 都从最低边际
utility blocks 回收等量预算，因此结构剪枝率严格不变。

手工 known-ID protection 不能作为算法；最终方法必须自动识别风险，profile 在 test 前冻结
并哈希，且接受随机和 route-count-matched controls。Super Experts、SCAR、MoE-Slimming
分别已覆盖 outlier experts、critical-channel protection 与异构 slimming，因此潜在贡献只能是
MoE-specific conditional expected utility 与 automatic rare-event prefix constraints 的联合。

80% full PPL 的机制分解如下：

| 方案 | PPL | Routed compute 剪枝率 |
|---|---:|---:|
| Conditional Dual（原候选） | 30.062264 | 45.24% |
| Tail coverage only，λ=0.50 | 16.275436 | 43.75% |
| Automatic risk floor only，λ=0.00 | 14.946378 | 43.21% |
| Combined，λ=0.10 | 14.587363 | 43.17% |
| Combined，λ=0.25 | 14.563407 | 43.25% |
| **Combined，λ=0.50** | **13.906462** | **43.58%** |
| Combined，λ=1.00 | 15.389882 | 45.20% |
| Random early experts width=2 | 29.986957 | 45.25% |
| Route-count-matched L2-E18 width=2 | 30.061409 | 45.24% |

tail-only profile 仍给 L2-E92 width 0，却把 PPL 从 30.06 降到 16.28，说明 expert 内
rare-channel ranking 本身修复了另一部分崩溃；floor-only 降到 14.95，说明跨 expert 的
zero-capacity failure 同样独立存在。两者联合 λ=0.50 达 13.91，显著优于任一单项以及
手工 L2-E92 full 的 14.91。λ=1.00 退化表明纯最大激活会过度追逐 tail，平均/尾部信号
必须联合。随机和 route-count-matched width=2 controls 均接近原始 30.06，支持风险选择
的特异性。

该结果是同结构参数预算胜利，不是同 FLOPs 胜利：联合方法比原 Conditional Dual 执行
更多 routed channels。但它比手工 L2-E92 full 具有更低 PPL和更高 routed-compute
剪枝率，说明收益不只是粗暴扩大已知 expert。

去掉 early-layer gate 后，统一全层 99.5% risk quantile + 0.1×global-max selector 选择
31 个 experts，80% PPL 进一步降到 `13.761718`。同一冻结规则在 50%/60% 的 PPL 为
`8.063362/8.688653`，均优于 Conditional Dual 的 `8.079852/9.008105`。60% tail-only
为 `8.792107`、early-selector combined 为 `8.710267`，全层 selector 仍最佳。这表明
方法不依赖文献给出的 early-layer/Super-Expert 白名单；风险约束收益随压缩接近 capacity
cliff 而快速增强，50% 仅小幅改善，60% 已有稳定提升，80% 主要用于压力机制验证。

| 结构剪枝率 | Route×RMS | Conditional Dual | Global Tail-Risk | 相对 Conditional Dual 改善 |
|---:|---:|---:|---:|---:|
| 50% | 8.175469 | 8.079852 | **8.063362** | 0.2041% |
| 60% | 9.255834 | 9.008105 | **8.688653** | 3.5463% |
| 80% | 31.186814 | 30.062264 | **13.761718** | 54.2225% |

## 9. 不重叠校准区间稳定性

为避免把 offset-0 WikiText train 首段当作隐式超参数，固定 λ=0.50、global 99.5%
quantile、0.1×global-max、minimum width=2 和 60% structural budget，在四个完全不重叠的
262,144-token train intervals 上重新收集 tail cache。Conditional Dual expert-level utility
仍固定为 offset 0，因此该实验隔离的是 tail coverage 与 risk selector 稳定性。

| Train token offset | Risk Spearman vs offset 0 | Selected-set Jaccard | Full PPL |
|---:|---:|---:|---:|
| 262,144 | 0.9110 | 0.5500 | 8.736616 |
| 524,288 | 0.9351 | 0.6757 | 8.789737 |
| 786,432 | 0.9275 | 0.5500 | 8.766030 |
| 1,048,576 | 0.9231 | 0.6316 | **8.673861** |

四段 PPL 均低于 Conditional Dual `9.008105`，均值 `8.741561`、population std
`0.043380`。L2-E92 在每段都为 global rank 2/layer rank 1，三个 published Super
Experts 均 5/5 入选；说明关键 catastrophic-risk core 稳定。31-member set 的 Jaccard
仅 0.55–0.68，波动主要来自 late-layer secondary max tails，但最终 profile 只有
3.56%–4.43% experts 的宽度变化，PPL 仍稳定改善。

该结果排除了“只在首段 calibration 出现”的简单反例，但仍不是跨语料/跨模型验证。
针对次级 max-tail 抖动，下一步使用 5 个区间的 majority/strict consensus floor；通道
coverage 保持 offset-0，以单独检验 risk-set 去噪。

Consensus 结果为负消融：3/5 在 60%/80% 为 `8.692268/13.790637`，5/5 为
`8.699382/13.780490`，都略差于单段 global selector `8.688653/13.761718`。因此
cross-fold voting 不进入主算法。它证明稳定出现于 5 个区间的 21–27 expert core 已足以
保持大部分收益，但删除单段边缘成员没有带来进一步 PPL 改善。

## 10. C4 跨语料校准：可迁移 Risk Floor 与域相关 Capacity Allocation

为检验 WikiText calibration/evaluation domain coupling，固定 λ=0.50、global 99.5%
quantile、0.1×global-max、minimum width=2、全 48 层和 60% exact structural budget，
从本地 C4 train 子集取四个不重叠的 262,144-token intervals：offsets
`0/262144/524288/786432`。两个离线 Arrow 文件按固定顺序加载并逐文件 SHA256；所有
profiles 均在读取 WikiText test 前冻结。该 C4 cache 对 WikiText offset-0 risk 的
Spearman 为 `0.8430–0.8615`，selected-set Jaccard 为 `0.4762–0.5500`。L2-E92 在四折
始终 layer rank 1、global rank 4/5/7/9，L1-E68/L2-E92/L3-E82 全部 4/4 自动入选。

但“risk rank 稳定”不等于整个 static profile 可跨域迁移：

| Calibration / decomposition | WikiText full PPL（四折均值±population std） | 相对 Conditional Dual |
|---|---:|---:|
| WikiText teacher + C4 coverage + C4 risk | `9.021363±0.009827` | `+0.013258`，0/4 胜出 |
| **WikiText coverage + C4 risk** | **`8.690719±0.001505`** | **`-0.317386`，4/4 胜出** |
| C4 coverage + WikiText risk | `9.020079±0.008175` | `+0.011974`，0/4 胜出 |
| 旧 C4 `combined` teacher + C4 coverage + C4 risk | `11.297659±0.194902` | **协议无效，不得标为 Conditional Dual** |

正反向解耦给出清晰机制边界：只将 sparse expert-level risk floors 换成 C4，PPL 几乎
贴住 WikiText 主方法 `8.688653`，四折范围仅 `8.689402–8.693223`；只替换 C4
within-expert channel coverage 则回到约 `9.02`。因此跨语料稳定部分是“哪些 physical
experts 绝不能被压成 zero/tiny capacity”的 rare-risk safety prior；prefix channel ranking
和 Conditional Dual expected utility 则显著依赖目标域或代表性 calibration mixture。

旧 source-coherent follow-up 后续审计发现 teacher collector 命令遗漏了显式
`--parent-mode dual`，并使用了当时的默认 `combined` objective。因此 EXP-030–032 及
`11.297659±0.194902` 只保留为协议失败/combined-objective 诊断，不再支持任何
Conditional Dual 或目标 Tail-Risk claim。修复后 collector 强制显式 parent mode，Tail-Risk
builder 也拒绝 `parent_mode != dual` 的 teacher；正确结果统一使用
`conditional_dual_teacher_c4_dual_offset_*`，见 EXP-034–041。

新的最小机制表述应是双时间尺度/双角色 factorization：corpus-stable rare-risk floors 作为
safety constraint，domain-adaptive conditional utility 与 channel coverage 负责主容量分配。
这比笼统声称“tail score 跨域鲁棒”更准确，也提示后续应做 representative mixed-domain
calibration 或 matched-domain evaluation，而不是强迫一个单域 profile 零样本迁移。

## 11. C4 匹配域正式复现

C4 evaluation 在任何 PPL 运行前冻结为 `c4_validation_114x2048_v1`：本地 C4
validation Arrow 的前 233,472 tokens，114×2048 windows，token cache SHA256 为
`46b1ccaa71fc61aa798273362d597d5ab4e3e4d87fa0f6e9f5c181ea8e96751c`。Dense PPL 为
`12.529323`。四个 C4-train 校准 folds 均重新采集显式 `parent_mode=dual` teacher，profile
在读取 validation PPL 前冻结；60% 时每个方法精确保留 29,491/73,728 个 64-channel
blocks。

| C4 train offset | Route×RMS | Conditional Dual | Tail-Risk |
|---:|---:|---:|---:|
| 0 | 17.331996 | 17.050479 | **16.628205** |
| 262,144 | 17.119347 | 16.790504 | **16.466717** |
| 524,288 | 17.268787 | 17.233710 | **16.621989** |
| 786,432 | 17.463222 | 17.233013 | **16.719089** |
| **均值±population std** | `17.295838±0.123699` | `17.076927±0.181440` | **`16.609000±0.090694`** |

Tail-Risk 4/4 folds 同时优于 Conditional Dual 和 Route×RMS；相对 Conditional Dual
平均降低 `0.467927` PPL（`2.7401%`），相对 Route×RMS 降低 `0.686838`
（`3.9711%`）。其 routed-compute 剪枝率平均还比 Conditional Dual 高 `0.6202` 个百分点，
因此该收益不是靠执行更多 routed channels 获得。

fold-0 机制分解再次复现两个组件的互补性：Conditional Dual、tail coverage only、risk
floor only、combined Tail-Risk 分别为 `17.050479/16.896053/16.803006/16.628205`。
coverage 和 floor 单独分别改善 `0.154426/0.247473`，联合改善 `0.422274`，且联合优于
任一单项。这与 WikiText 上“expert 内 rare-channel ranking + expert 间 sparse safety
floor”的机制一致。

| C4 结构剪枝率 | Conditional Dual | Tail-Risk | 绝对改善 |
|---:|---:|---:|---:|
| 50% | 14.579660 | **14.507353** | 0.072307 |
| 60% | 17.050479 | **16.628205** | 0.422274 |
| 80% | 49.518980 | **32.711715** | 16.807265 |

三个预算均为 exact matched structural budget，且 Tail-Risk routed-compute 剪枝率分别从
`9.0374%/16.7682%/46.6897%` 提高到 `9.3203%/17.3828%/47.4539%`。80% 仍是 capacity
cliff 压力测试，不是实用部署点。

综合 WikiText 与 C4，当前最小可辩护经验结论是：**Tail-Risk 在两个 matched domains
都改善静态专家缩宽 PPL；expert-level risk floors 可跨语料迁移，而 Conditional Dual
expected utility 与 within-expert channel coverage 需要目标域或代表性混合域校准。**
这不等价于整个 profile 跨域不变，也不证明 factual reliability 或 instruction following。

## 12. Qwen3 Base 跨 Checkpoint 复现

为避免主结果只依赖 Instruct-2507 checkpoint，在 `/data01/datasets/Qwen3-30B-A3B`
base checkpoint 上从头生成 WikiText-train RMS/tail caches、AMP/AIMER priors、显式
`parent_mode=dual` teacher 和三个冻结 profile。没有复用 Instruct profile，也没有用
WikiText test 指标调整 λ、risk quantile、relative-max guard、minimum width 或 block size。

| 60% exact structural pruning | Full PPL | Routed compute 剪枝率 |
|---|---:|---:|
| Route×RMS | 10.623834 | 13.0469% |
| Conditional Dual | 10.371195 | 13.5545% |
| **Tail-Risk** | **10.341598** | **13.9044%** |

Tail-Risk 相对 Conditional Dual 改善 `0.029597` PPL（`0.2854%`），相对 Route×RMS
改善 `0.282236`（`2.6567%`）；同时比 Conditional Dual 多剪 `0.3499` 个百分点 routed
compute，完整通过 EXP-042 预注册成功标准。Base 与 Instruct 的 automatic risk set 交集
为 21/31、Jaccard `0.5122`，三个 published Super Experts 均被保留；最终 widths 有
`19.60%` physical experts 不同。这支持“catastrophic-risk core 部分稳定，而常规容量分配
随 checkpoint 自适应”的解释。

该结果只构成同一 Qwen3 架构内的跨 checkpoint 证据。它尚不能排除方法对 Qwen3 MoE
模块结构的依赖，也不能证明事实可靠性或指令遵循保持。

## 13. TruthfulQA：PPL 与事实可靠性解耦

对同一批冻结的 Instruct-2507 profiles 运行 TruthfulQA multiple-choice validation 全 817
题。MC1 检查最高条件 likelihood 答案是否为真；MC2 统计 softmax-normalized likelihood
分配给全部真实答案的概率质量。

| 方法 | MC1 accuracy | MC2 true probability |
|---|---:|---:|
| Dense | **0.343941** | **0.518451** |
| 60% Route×RMS | 0.266830 | 0.429845 |
| 60% Conditional Dual | 0.270502 | **0.440415** |
| 60% Tail-Risk | **0.271726** | 0.437559 |

Tail-Risk 相对 Conditional Dual 的 MC1 只多 `1/817` 题，但 MC2 低 `0.002856`，因此
没有通过“MC1/MC2 均不低于 Conditional Dual”的预注册成功标准。它相对 Route×RMS
两指标都更好，却仍明显低于 Dense。这一结果直接否定“较低 PPL 即保持 factual
reliability”的外推；当前 tail proxy 主要覆盖激活幅度主导的语言建模 capacity risk，尚未
覆盖事实答案 ranking/probability risk。

结果后没有修改 profile。论文必须把该实验作为 mixed negative evidence，并将 direct
reliability-aware constraint、较低 practical pruning ratio 和生成式 hallucination/
instruction-following 评价列为后续必要门禁。

## 14. Qwen1.5-MoE 跨架构负结果

在 `Qwen2MoeForCausalLM` 的 Qwen1.5-MoE-A2.7B 上，从头生成 24×60 expert priors、
WikiText train 262,144-token RMS/tail caches、显式 dual teacher 与冻结 profiles。该架构为
top-4、`norm_topk_prob=false`、1,408-channel routed experts，并包含 shared expert；每个
expert 划分为 22 个 64-channel blocks。

| 60% exact structural pruning | Full PPL | Routed compute 剪枝率 |
|---|---:|---:|
| Dense | 7.459207 | 0% |
| **Route×RMS** | **10.114996** | **55.6537%** |
| Conditional Dual | 10.626735 | 54.8476% |
| Tail-Risk | 10.239141 | 54.9541% |

Tail-Risk 相对 Conditional Dual 改善 `0.387593`（`3.6473%`），说明 rare-channel coverage
与 sparse floors 仍能修复均值 utility 的一部分错误；但它比 Route×RMS 退化 `0.124145`
（`1.2273%`），同时还执行更多 routed channels。因此 EXP-047 的跨架构成功标准失败，
当前方法不能声称 architecture-universal。

该负结果把问题定位到联合 formulation 的 expected-utility parent，而不是证明 Tail-Risk
分支毫无价值。后续机制分解应固定既有 λ/floor，只比较 tail coverage only、risk floor only
和 combined；不得依据该 test 结果重新选择超参数。

固定 profile 后的分解为：Conditional Dual `10.626735`、tail coverage only `10.243799`、
risk floor only `10.621897`、combined `10.239141`。也就是说，Qwen2MoE 上
`0.387593` 的 combined repair 中，`0.382935` 已由 tail coverage 单独解释；floor 相对
tail-only 只增加 `0.004658`。这与 Qwen3/C4 上两个组件都显著有效不同，进一步说明
expert parent、channel coverage 与 safety floor 的权重不能假定跨架构恒定。

由此提出的探索性 Route×Tail（route-frequency expert parent + λ=0.50 typical/tail
within-expert coverage）在同一 WikiText test 上从 Route×RMS `10.114996` 降到
`10.013247`。由于该想法来自已见 test，这一数值不算独立证据。随后在读取 Qwen1.5 C4
validation 前冻结 C4 train matched-domain profile，取得 Route×RMS `17.328660`、
Route×Tail `16.203895`；改善 `1.124765`（`6.4908%`），且 routed-compute 剪枝从
`59.2175%` 增至 `59.3571%`。

这组预注册独立语料结果支持更窄但更稳健的 factorization：**expert-level parent 可依赖
架构的 routing semantics，而 typical/tail within-expert coverage 是当前更可迁移的组件。**
它尚不足以成为 ICLR-level 新意，因为 route-weighted activation/channel importance 已有
大量先例，且目前只在一个 Qwen2MoE 模型上完成独立验证。

Qwen3 C4 四折 exploratory breadth check 得到 Route×Tail
`17.100295±0.093404`，4/4 优于 Route×RMS `17.295838±0.123699`，但略差于 Conditional
Dual `17.076927±0.181440`，并明显差于完整 Tail-Risk `16.609000±0.090694`。因此当前
最符合证据的层次结构是：

1. Route×Tail 是跨 Qwen2MoE/Qwen3MoE 都有效的稳健 base allocation；
2. routed-set Conditional Dual 和 sparse risk floors 是 Qwen3 上有效的 conditional
   refinements；
3. 是否启用这些 refinements 必须由 train-only applicability criterion 决定，不能按 test
   PPL 人工切换。

## 15. Train-only Cross-Fitted Applicability Selector

为把上述架构依赖变成单一、可执行且不看 test 的算法，冻结如下 selector：

1. Route×Tail 始终作为 fallback；完整 Tail-Risk 作为候选 refinement；
2. profile calibration 固定使用 train `[0,262144)`；四个选择折固定使用 train
   `[1572864,1605632)`、`[1605632,1638400)`、`[1638400,1671168)`、
   `[1671168,1703936)`，每折 16×2048 tokens；所有区间严格不重叠；
3. refinement 只有在四折中严格多数获胜且 mean selection PPL 更低时才启用；平票或不稳定
   结果回退 fallback，不设置由结果调出的 improvement threshold；
4. selector 强制校验 `split=train`、profile frozen、test-independent、profile 实际 SHA256、
   每折引用的 profile path/SHA256 以及 token-cache SHA256。

Retrospective diagnostic 结果如下：

| 模型 | Route×Tail train-fold mean PPL | Tail-Risk train-fold mean PPL | Fold wins | 选择 |
|---|---:|---:|---:|---|
| Qwen1.5-MoE-A2.7B | **9.893688** | 10.323414 | Route×Tail 4/4 | Route×Tail fallback |
| Qwen3-30B-A3B-Instruct-2507 | 9.289503 | **8.992431** | Tail-Risk 4/4 | Tail-Risk refinement |

Qwen1.5 上 selector 避免了 Tail-Risk `4.3434%` 的 mean selection PPL 退化；Qwen3 上启用
Tail-Risk 后改善 `3.1979%`。两个 train-only 决策都复现此前正式 WikiText test 的相反排序：
Qwen1.5 test 偏好 Route×Tail，Qwen3 test 偏好 Tail-Risk。这说明 applicability signal 至少
在当前两个架构上强且稳定，而不是模型名硬编码或单折偶然性。

但该诊断不算独立验证：两个 formal test 排序在 selector 实验前已经存在。当前可辩护的
新贡献方向因此收窄为：**以 Route×Tail 为 robust fallback，通过 disjoint train-only
cross-fitting 决定是否启用 conditional-utility/risk-floor refinement，并在 exact structural
budget 下冻结最终 profile。** 下一门禁是在第三个未读取 test PPL 的模型上先运行 selector、
冻结选择，再打开正式 test。

## 16. Qwen3.5 Prospective Applicability Validation

第三模型使用 `Qwen3.5-35B-A3B`：40 个 MoE layers、每层 256 routed experts、top-8、
每 expert 512 channels，并同时包含 shared expert、linear attention 与 full attention。
在读取该模型 test PPL 前，已完成如下不可逆顺序：

1. 以 WikiText train `[0,262144)` 独立生成 RMS/tail caches、weight priors 与 Conditional
   Dual teacher；
2. 冻结同为 `32,768/81,920` blocks 的 60% Route×Tail 与 Tail-Risk profiles；
3. 在四个互不重叠的 16×2048 train folds 上运行 selector；Tail-Risk 4/4 获胜，mean PPL
   `8.009124 < 8.227062`，因此在 test 前冻结选择 Tail-Risk；
4. selection decision SHA256 固定为
   `6262c05cd3cd57ab2c7717a44552508f120dfa602276d69d6d7d16f54f8fd4e7`。

正式 full WikiText test 结果如下：

| 方法 | PPL | Structural pruning | Routed-compute pruning |
|---|---:|---:|---:|
| Dense | **6.839253** | 0% | 0% |
| Route×Tail（未选候选） | 8.062748 | 60% | **25.7194%** |
| **Tail-Risk（test 前选中）** | **7.834395** | **60%** | 25.5502% |

预先选中的 Tail-Risk 严格优于 Route×Tail，绝对降低 `0.228353`、相对降低 `2.8322%`，
并回收 Route×Tail 相对 Dense PPL gap 的 `18.6640%`。因此 cross-fitted applicability
selector 的 prospective success criterion 通过。这是当前证据链中最关键的新结果：它不再是
用已知 test 排序做 retrospective plausibility check，而是在未读取 Qwen3.5 test 指标时先冻结
分支，再由 test 验证分支选择正确。

该结论有两个必须保留的限制。第一，Tail-Risk 的 routed-compute 剪枝少 `0.1692` 个百分点，
即实际执行的 routed channels 略多；因此表格只支持 exact structural-budget 下的方法质量，
不能冒充 matched compute 结论。第二，协议审计发现 `233,368` 是 Qwen3 tokenizer 对既有
filtered WikiText full corpus 的 token 数，而非跨 tokenizer 常数。Qwen3.5 对相同 test 文本
得到 114 windows、`231,940` tokens。首次运行的 Dense 已正确标记非 formal，但 static
evaluator 旧逻辑只核对 windows，误标为 formal。

为避免隐去该问题，首次三路 JSON/日志完整保留在 `protocol_error_initial/`。随后冻结同一
Qwen3.5-tokenized 全语料 cache（SHA256 `fe45a0ea...fb65e`；test Arrow SHA256
`2b8a3efa...328ea`），把 formal 门禁改为 windows 与 tokens 均精确匹配，并用完全未变的
profiles 重跑。三路 PPL 和 routed-compute 数值与首次运行逐项相同。因此修正改变的是协议
元数据和审计强度，不是输入语料、方法选择或结果排序；但论文应明确称其为 post-reveal
protocol metadata correction，不能把过程写成从未发生过偏差。

综合三个模型，当前统一方法的证据层次变为：Route×Tail 是 Qwen2MoE/Qwen3/Qwen3.5 上
较稳健的 fallback；cross-fitted train-only selector 能在 Qwen1.5 回退 fallback、在 Qwen3
和 Qwen3.5 启用 Tail-Risk refinement；其中 Qwen3.5 已完成真正 prospective 验证。这个
贡献比单纯“静态异构 expert widths”更可辩护，但 Qwen3.5 与 Qwen3 仍是相邻模型谱系，
且 novelty 与 matched-compute/reliability 门禁尚未完全关闭。

## 17. 双预算 PPL–Routed-Compute Pareto 审计

仅匹配结构参数量仍可能受到一个审稿质疑：静态 profile 可以把更多 blocks 分给高频 routed
experts，从而在相同参数量下执行更多实际计算。为直接检验 Tail-Risk 的收益是否来自这种
compute advantage，在 Qwen3-30B-A3B-Instruct-2507 上增加双预算 allocator：

\[
\max_{w}\; \sum_{l,e,j < w_{l,e}} v_{l,e,j}
\quad \text{s.t.}\quad
\sum_{l,e} w_{l,e}=B,\qquad
\sum_{l,e} r_{l,e}w_{l,e}\approx C,
\]

其中 `B=29,491` 是精确全局结构 block 数，`r[l,e]` 是 WikiText train calibration 上的
physical-expert route count，`C` 是预冻结 expected routed-compute anchor。实现使用标量
Lagrangian multiplier，把每个 expert 的 train routed cost 从其 block marginal utility 中
扣除；同一 expert 的所有 blocks 使用相同 cost，因此不会破坏 expert 内 prefix marginal 的
单调性。每次候选仍由 exact global block allocator 产生，最终选择离散 compute error 最小、
原始 utility 最大的 profile。risk floors 继续作为硬约束保留。

冻结两个 anchors：原 Route×Tail train routed pruning `14.910874%`，以及原 Tail-Risk
`15.232707%`。两个方法在每个 anchor 各生成一个 profile，共四个 profiles；全部仍为
`29,491/73,728` blocks，最大 train compute-anchor 误差仅 `0.000677` 个百分点。随后四卡
同时运行 full WikiText test，结果为：

| Train compute anchor | 方法 | Full PPL | Test routed-compute pruning |
|---|---|---:|---:|
| Route anchor | Route×Tail | 8.996928 | 14.8682% |
| Route anchor | **Tail-Risk** | **8.701386** | **14.9328%** |
| Tail anchor | Route×Tail | 8.876885 | 15.2226% |
| Tail anchor | **Tail-Risk** | **8.688653** | **15.2451%** |

在 Route anchor，Tail-Risk 相对降低 PPL `3.2849%`，同时多剪 `0.0646` 个百分点 routed
compute；在 Tail anchor，相对降低 `2.1205%`，同时多剪 `0.0225` 个百分点 compute。
因此两个 anchor 都形成严格 Pareto dominance：Tail-Risk 不是通过执行更多热门专家通道
获得收益，而是在同一 exact structural budget 与近乎相同的 train expected compute 下，
选择了更有用的 expert-prefix capacity placement。

这个结果显著加强机制解释，但不能被包装成完全独立验证。双预算实验由已有 Qwen3 结果后的
compute fairness 问题驱动，属于 retrospective audit；Lagrangian 本身也是标准优化工具。
可能具备新增价值的组合是：**physical-expert prefix structural budget + train routed-compute
budget + tail-risk utility/floors + frozen auditable profiles**。要把它提升为论文主贡献，仍需在
未见模型/语料上预先冻结 compute anchors 并复现 Pareto dominance。

## 18. C4 四折 Route-Consensus 双预算复现：PPL 保持、Pareto 仍失败

为修复单一 C4 train fold 对 validation routed compute 的迁移偏差，本轮在读取 validation
指标前冻结四个不重叠 train folds（offset `0/262144/524288/786432`）的归一化 route-count
均值，并以该 consensus 分布构造两个 expected-compute anchors：Route anchor
`15.947146%`、Tail anchor `16.463321%`。四个 profile 均保持精确 `29,491/73,728`
blocks（60.000271% structural pruning），profile 内 `test_metrics_used=false`，validation
使用冻结 `c4_validation_114x2048_v1`（114 windows、233,472 tokens）。

| Consensus train anchor | 方法 | C4 validation PPL | validation routed-compute pruning |
|---|---|---:|---:|
| Route anchor | Route×Tail | 17.135774 | 17.0474% |
| Route anchor | **Tail-Risk** | **16.774891** | 16.9462% |
| Tail anchor | Route×Tail | 16.984889 | 17.4986% |
| Tail anchor | **Tail-Risk** | **16.628205** | 17.3828% |

primary PPL criterion 通过：Tail-Risk 在两个冻结 anchors 上分别改善 `2.1063%` 与 `2.1001%`。
但 strong Pareto criterion 仍失败：Tail-Risk 的 validation routed-compute pruning 分别低
`0.1012` 与 `0.1158` 个百分点。多折均值共识保留了跨域 PPL 优势，却没有消除
train→validation 的 route-distribution shift；不得将本轮写成 C4 Pareto 成功。

下一步只允许使用 train-only worst-case/CVaR compute calibration，或在 profile 适用性 selector
中加入预冻结的 compute-risk 门禁，不能用本轮 validation mismatch 反推 multiplier。完整记录见
`experiments/results/exp_v4_20260728_069_qwen3_c4_consensus_compute_frontier_preregistered.json`
和 `experiments/results/exp_v4_20260728_070_qwen3_c4_consensus_compute_frontier_full.json`。

## 19. WikiText Robust-Consensus Follow-ups：四条新路径均否决

为检验 C4 中 train→validation compute shift 的诊断，以及判断 Tail-Risk 是否能通过
跨折统计稳健化进一步提升，随后在 WikiText train 上冻结并测试了四类 follow-up。所有候选
仍使用 `29,491/73,728` exact structural blocks、64-channel prefix、同一 `wikitext2_raw_test_full_v1`
协议；没有使用新的 test 指标调参。

| Follow-up | Route anchor PPL | Tail anchor PPL | 判定 |
|---|---:|---:|---|
| 四折 route-cost CVaR(α=0.5) | 9.390907 | 8.991009 | 否决，过度惩罚不稳定热门 experts |
| 四折 worst-case route cost | 9.714727 | 9.714727 | 否决；离散 target 不可达并塌缩为同一 profile |
| 四折 mean tail coverage | 8.821189 | 8.781863 | 否决，低于 offset-0 domain-adaptive coverage |
| 四折 teacher + coverage mean | 8.810717 | 8.759046 | 否决，较 coverage-only 恢复但仍低于主方法 |

这组负结果具有明确的机制含义：当前收益不是来自把 route cost 做成 worst-case，也不是
来自把 within-expert coverage 或 Conditional-Dual teacher 简单平均。目标域的 domain-adaptive
utility/coverage 仍应保留；跨域稳健性更适合由 train-only applicability selector 决定是否启用
refinement，而不是强行写进每个静态 profile。

进一步尝试了 teacher utility 的 lower-confidence-bound 收缩（四折 mean − κ·std，
`κ∈{0.25,0.50,1.00}`），并在四个互不重叠的 train selection folds 上比较 offset-0
Tail-Risk fallback。κ=0.25 只赢 `2/4` folds，平均 train PPL `9.004032`，不如 fallback
`8.992431`，因此 cross-fitted selector 严格回退到 offset-0 Tail-Risk；没有为 LCB 候选打开
新的 full test。该结果支持“robustness 作为适用性选择门，而非 utility 内部平均/收缩”的当前
最小 formulation。

完整证据：
`experiments/results/exp_v4_20260728_076_qwen3_wikitext_robust_consensus_followups.json`、
`experiments/results/exp_v4_20260728_075_qwen3_wikitext_teacher_lcb_selection_decision.json`。

## 20. 路由熵层先验：train-only 四折明确否决

公开 Qwen hybrid pruning 实现提示浅层路由可能更分散，因此测试了一个只改变容量跨层放置、
不改变总结构预算的 routing-entropy prior。对每层 train route distribution 计算归一化熵，
将 mean-one 的层权重乘入 block utility；`gamma>0` 保护高熵层，`gamma<0` 反向保护低熵层。
所有候选保持 `29,491/73,728` blocks、Tail compute anchor 与相同 risk floors，并只在四个
disjoint WikiText train folds 上选择。

| Candidate | Mean train PPL | Fold wins |
|---|---:|---:|
| **Tail-Risk fallback** | **8.992431** | **4/4** |
| entropy `gamma=+0.5` | 9.051775 | 0/4 |
| entropy `gamma=+1.0` | 9.037242 | 0/4 |
| entropy `gamma=-0.5` | 9.059306 | 0/4 |

fallback 在每一折都更优，因此没有为任何 entropy candidate 打开 full test。负结果表明，
route dispersion 的层统计不能直接替代 token-conditional utility；即使 prior 不改变预算，
它仍会系统性扭曲已经由 Conditional-Dual + tail coverage 学到的容量放置。

机器可读决策：
`experiments/results/exp_v4_20260728_080_qwen3_wikitext_routing_entropy_selection_decision.json`。

## 21. Contribution-Calibrated Tail-Risk：实现成功，selector 否决

REAP 使用 `gate × ||expert_output||₂` 评估整专家删除。为检验实际输出贡献能否改善本文的
expert 内结构剪枝，本轮没有复制 whole-expert removal，而是构造：

\[
s_{l,e}^{(f)}=
\operatorname{mean}_{t:e\in R_t}
\left[g_{t,e}\lVert E_{l,e}(h_t)\rVert_2\right],
\qquad
\bar s_{l,e}=\operatorname{mean}_f
\frac{s_{l,e}^{(f)}}{\operatorname{mean}_{e'}s_{l,e'}^{(f)}},
\]

\[
u'_{l,e}=u^{\text{Conditional-Dual}}_{l,e}
(\max(\bar s_{l,e},\epsilon))^\beta.
\]

最终 block marginal 仍是 `u' × tail-aware prefix coverage`，并继续服从全局 99.5% sparse
risk floors、exact `29,491` structural blocks 和 Tail compute anchor；因此 output saliency
只校准 expert utility，不改变 physical-expert prefix 结构或预算协议。

四个正式 train folds 分别绑定物理 GPU4/5/6/7，offset 为
`0/262144/524288/786432`，每折 `128×2048` tokens。全部 cache 都是
`parent_mode=dual`、`split=train`、shape `(48,128)`、finite；正 saliency experts 分别为
`6062/6109/6087/6087`。构建 β=`0.25/0.50/1.00` 三个候选后，在四个完全不重叠的
16×2048 train selection folds 上与原 Tail-Risk fallback 比较：

| Candidate | Mean train PPL | Fold wins | Selector decision |
|---|---:|---:|---|
| **Tail-Risk fallback** | **8.992431** | **3/4** | **selected** |
| contribution β=0.25 | 9.032436 | 1/4 | rejected |
| contribution β=0.50 | 9.039200 | 0/4 | rejected |
| contribution β=1.00 | 9.115010 | 0/4 | rejected |

β=0.25 只在 offset `1572864` 折以 `11.317385 < 11.319011` 极小幅获胜，其余三折均退化；
β 越大，mean PPL 越差。selector 因此按严格多数与更低均值规则回退，不读取新的 test PPL。
这否决的是“把 output contribution 作为全局乘法 utility prior”的当前 formulation，而不是
否定 output statistics 本身。更合理的后续方向可能是把 contribution 用作稀疏异常门禁、
不确定性特征或 selector 输入，而不是连续重权整个 expert utility。

完整证据：
`experiments/results/exp_v4_20260728_081_qwen3_output_contribution_calibration_and_profiles.json`
与
`experiments/results/exp_v4_20260728_082_qwen3_contribution_tail_risk_selection_decision.json`。

## 22. Output-Contribution Safety Floor：selector 通过，formal PPL 失败

乘法融合失败后，本轮不再用 output norm 重排所有专家，而是参考 REAP、统一 expert scoring、
protected-core/Subset Difference 的共同启示，把实际输出贡献仅作为稀疏安全约束。四折 saliency
仍采用：

\[
\bar s_{l,e}=\operatorname{mean}_f
\frac{s_{l,e}^{(f)}}{\operatorname{mean}_{e'}s_{l,e'}^{(f)}}.
\]

固定全局 `99.5%` quantile，把入选 physical experts 的最小宽度设为 2 blocks，并与原
activation-tail risk floors 取并集；Conditional-Dual utility、tail coverage、29,491-block
结构预算和 offset-0 Tail compute anchor 均不变。`relative-max=0.10` 的首次冻结被两个极端
Super Experts 主导，只选择 2 个已保护 experts、新增约束为 0，因此在读取任何 PPL 前判定为
退化配置。最终预注册候选使用纯 99.5% quantile，不依据 test 调参。

候选新增 27 个显式 saliency-floor constraints，但实际只有六个原本 0/1-block 的专家发生
绑定；为维持 exact structure 与 compute anchor，共移动 15 个 blocks、改变 27 个 expert
widths。四折 train-only selection 结果为：

| Fold offset | Tail-Risk | Output Safety Floor | Delta |
|---:|---:|---:|---:|
| 1,572,864 | 11.319011 | **11.300475** | -0.018536 |
| 1,605,632 | 7.125653 | **7.122514** | -0.003139 |
| 1,638,400 | 7.974960 | **7.974871** | -0.000090 |
| 1,671,168 | **9.550100** | 9.554811 | +0.004711 |
| Mean | 8.992431 | **8.988168** | -0.004263 |

候选达到 3/4 folds 严格多数且 mean 更低，因此 selector 在 test 前冻结选择候选。随后只在
物理 GPU4 运行一次正式 WikiText-2 full test：

| 方法 | Full PPL | Test routed-compute pruning |
|---|---:|---:|
| **Tail-Risk fallback** | **8.688653** | 15.24508% |
| Output Safety Floor | 8.689485 | **15.24710%** |

候选多剪 `0.00201` 个百分点 routed compute，但 PPL 高 `0.000832`，相对退化
`0.00958%`。primary lower-PPL 与 strict Pareto criteria 均失败。该结果不能写成方法提升；
它是一次小效应 selector false positive，说明 output norm 仍混合了功能独特贡献和高幅值
冗余。后续不得根据该 test 结果继续调 quantile；若重用 output statistics，应先用 train-only
committee co-routing、functional-profile similarity 或 residual-context evidence 区分
unique contribution 与 redundant contribution。

完整证据：`experiments/results/exp_v4_20260728_083_qwen3_output_safety_floor_preregistered.json`、
`exp_v4_20260728_084_qwen3_output_safety_floor_selection_decision.json`、
`exp_v4_20260728_085_qwen3_output_safety_floor_formal_preregistered.json` 和
`exp_v4_20260728_086_qwen3_output_safety_floor_formal_result.json`。

## 23. Redundancy-Aware Unique Contribution：稳定性修正后仍被 0/4 否决

为检验 Output Safety Floor 的失败是否来自“高输出幅值功能克隆”，本轮在同一次 MoE
forward 中为每层累计 gate-weighted 共路由矩阵：

\[
C^{(f)}_{l}=\sum_t g_{l,t}g_{l,t}^{\mathsf T},\qquad
q^{(f)}_{l,e}=1-\max_{j\ne e}
\cos(C^{(f)}_{l,e,:},C^{(f)}_{l,j,:}).
\]

计算 fingerprint 时去掉矩阵对角线，避免 expert 自身 routing mass 被误当作 committee
context。每折 unique contribution 为逐层 mean-one 归一化的 output contribution 与
`q` 的乘积。四折 train-only 稳定性审计发现，直接求均值的 top-31 中存在某 expert 在一折
排名 2354、其余折极高的异常；因此在读取任何 selection PPL 前，预注册使用 worst-fold
聚合：

\[
r_{l,e}=\min_f\left(\tilde s^{(f)}_{l,e}q^{(f)}_{l,e}\right).
\]

候选固定 global 99.5% quantile、minimum width=2、relative-max=0，并与原 activation-tail
floors 合并。它仍精确保留 29,491 blocks；compute calibration 后相对 fallback 改变 23 个
experts、转移 13 blocks，31 个 unique-tail experts 中 26 个是新约束，最终 4 个 floors 实际
绑定。四折 selector 结果为：

| Fold offset | Tail-Risk | Unique Contribution Floor | Delta |
|---:|---:|---:|---:|
| 1,572,864 | **11.319011** | 11.325823 | +0.006812 |
| 1,605,632 | **7.125653** | 7.131622 | +0.005969 |
| 1,638,400 | **7.974960** | 7.976028 | +0.001068 |
| 1,671,168 | **9.550100** | 9.558014 | +0.007914 |
| Mean | **8.992431** | 8.997872 | +0.005441 |

候选 0/4 folds 获胜，因此 selector 明确回退，未打开 full test。该负结果比 output-only
失败更有诊断价值：即使去掉 co-routing functional clones，expert-level output scalar 仍不能
可靠推断 physical expert 内部 64-channel prefix 的需求。后续不应继续搜索 saliency/uniqueness
quantile；若继续利用 committee information，需要直接落到 channel/block-conditioned
contribution 或 token-level counterfactual regret，而不是再给整 expert 一个保护标量。

完整证据：`experiments/results/exp_v4_20260728_087_qwen3_unique_contribution_floor_preregistered.json`
和 `experiments/results/exp_v4_20260728_088_qwen3_unique_contribution_floor_selection_decision.json`。

## 24. Frontier Committee Regret：4/4 selector 与正式新最佳 PPL

前面三类 contribution refinement 的共同失败点，是把实际被剪的 channel/block 信息压缩成
expert-level 标量。本轮改为直接估计 frozen fallback 每个 physical expert 的第一个被剪
64-channel block。对 routed token `t`、expert `e` 和其中间通道 `i`，定义对其他 routed
experts 委员会输出的对角残差能量：

\[
R_{t,e,i}^{2}=g_{t,e}^{2}z_{t,e,i}^{2}
\left(\lVert w_{e,i}\rVert_2^2-
\langle w_{e,i},\widehat{o}_{t,-e}\rangle^2\right),
\]

其中 `z` 是 SwiGLU 中间激活，`w` 是 down-projection 列，`o_{t,-e}` 是同 token 其余
routed experts 的加权输出。每个已有 tail-ranked 64-channel block 对通道残差平方求和再开方。
该 diagonal-down-Gram 近似只需一次额外 committee projection，不需要为 12 个 blocks 分别
重跑完整 down projection。

四个 train estimator folds 分别在物理 GPU4/5/6/7 收集 16×2048 tokens。只读取 frozen
fallback width 之后的第一个 block；full-width experts 不参与，未路由证据得分为 0。每折逐层
mean-one 归一化，跨折取 minimum，随后固定 global q99.5。候选选中 23 个原宽度为 0 的
dormant experts，各恢复第一个 block；compute calibration 后仍精确保持 29,491 blocks，相对
fallback 转移 36 blocks、改变 72 个 expert widths。

四折 train-only selector 为：

| Fold offset | Tail-Risk | Frontier Committee Regret | Delta |
|---:|---:|---:|---:|
| 1,572,864 | 11.319011 | **11.282376** | -0.036635 |
| 1,605,632 | 7.125653 | **7.123696** | -0.001957 |
| 1,638,400 | 7.974960 | **7.956578** | -0.018382 |
| 1,671,168 | 9.550100 | **9.534097** | -0.016003 |
| Mean | 8.992431 | **8.974187** | -0.018244 |

候选 4/4 folds 获胜且 mean 更低，因此在 test 前冻结。一次正式 WikiText-2 full test 得到：

| 方法 | Full PPL | Test routed-compute pruning |
|---|---:|---:|
| Tail-Risk fallback | 8.688653 | **15.24508%** |
| **Frontier Committee Regret** | **8.658925** | 15.23506% |

候选 PPL 绝对改善 `0.029727`，相对改善 `0.34214%`，成为当前相同 60.000271% exact
structural pruning 下的新最佳正式 PPL。它在 test 上少剪 `0.01003` 个百分点 routed compute，
因此 lower-PPL primary 成功，但 strict PPL-compute Pareto 未通过。该结果支持核心机制：
committee redundancy 必须在实际剪枝 frontier block 上估计，而不能停留在整 expert 输出幅值。
下一步需要在未见 test 的独立模型或 matched-domain 上预注册双预算复现，不能使用当前 test
compute 差值反调 multiplier 后声称 prospective Pareto。

完整证据：`experiments/results/exp_v4_20260728_089_qwen3_frontier_committee_regret_preregistered.json`、
`exp_v4_20260728_090_qwen3_frontier_committee_regret_selection_decision.json`、
`exp_v4_20260728_091_qwen3_frontier_committee_regret_formal_preregistered.json` 和
`exp_v4_20260728_092_qwen3_frontier_committee_regret_formal_result.json`。

## 25. Qwen3 Base 独立复现：4/4 selector 与跨 checkpoint lower-PPL 成功

为避免 Instruct checkpoint 上的方法设计与评估互相污染，本轮在
`/data01/datasets/Qwen3-30B-A3B` Base checkpoint 上重新生成全部方法相关证据：独立 Base
AMP/AIMER priors、offset-0 RMS/tail channel calibration、Conditional-Dual teacher、四个
Frontier Committee Regret estimator folds，以及四个 train-only selector token caches。没有复用
Instruct frontier cache、profile 或 width。estimator 校准范围为 `0..819200`，selection offsets
固定为 `1572864/1605632/1638400/1671168`，两者严格不重叠。

Base fallback 先在 test 前冻结到 train-only expected routed-pruning anchor
`0.13992059065235984`。候选沿用预注册的 minimum-fold、global q99.5、一块 width increment 和
activation-tail q99.5/w2 floors；23 个入选 frontier experts 的 reference width 均为 0，23 个
floors 全部绑定。compute calibration 后仍精确保留 `29,491/73,728` blocks，相对 Base
fallback 改变 66 个 expert widths、转移 33 blocks。离散 train routed-pruning 为
`0.13992026448249817`，与 target 只差约 `3.26e-7`。

四折 Base train-only selector 为：

| Fold offset | Base Tail-Risk | Base Frontier Committee Regret | Delta |
|---:|---:|---:|---:|
| 1,572,864 | 13.663181 | **13.618860** | -0.044321 |
| 1,605,632 | 8.331412 | **8.308897** | -0.022515 |
| 1,638,400 | 9.498721 | **9.465001** | -0.033720 |
| 1,671,168 | 11.342063 | **11.322128** | -0.019935 |
| Mean | 10.708844 | **10.678721** | -0.030123 |

候选再次 4/4 folds 获胜，因此按预注册门禁冻结并只在物理 GPU4 运行一次完整 WikiText-2
test。正式结果为：

| 方法 | Full PPL | Test routed-compute pruning |
|---|---:|---:|
| Base Tail-Risk fallback | 10.341598 | **13.90440%** |
| **Base Frontier Committee Regret** | **10.322812** | 13.87926% |

Base candidate 的 PPL 绝对改善 `0.018786`，相对改善 `0.18166%`，在同一 exact structural
budget 上复现了 Instruct checkpoint 的 lower-PPL 方向。这是比单 checkpoint 新最佳更强的机制
证据：first-pruned block 的 committee residual 不只是 Instruct calibration 的偶然排序。候选在
test 上少剪 `0.02514` 个百分点 routed compute，因此 primary lower-PPL 成功，但 strict
PPL-compute Pareto 仍未通过。不能把该结果表述为零代价提升；正确表述是“独立 Base checkpoint
跨 checkpoint 复现，伴随很小的 routed-compute concession”。

完整证据：`experiments/results/exp_v4_20260728_093_qwen3_base_frontier_regret_preregistered.json`、
`exp_v4_20260728_094_qwen3_base_frontier_regret_selection_decision.json`、
`exp_v4_20260728_095_qwen3_base_frontier_regret_formal_preregistered.json` 和
`exp_v4_20260728_096_qwen3_base_frontier_regret_formal_result.json`。

## 26. Qwen3.5 跨拓扑迁移：3/4 selector 与 secondary lower-PPL 复现

为检验方法是否依赖 Qwen3-30B 的 48×128 expert topology，本轮进一步迁移到
`Qwen3.5-35B-A3B` 的 40×256、top-8 MoE。Qwen3.5 的 Frontier estimator folds 在本轮首次
独立采集，offset `0/262144/524288/786432` 分别绑定物理 GPU4/5/6/7，每折 16×2048 train
tokens；没有复用 Qwen3-30B 或 Base 的 frontier cache/profile/width。

冻结 fallback 的 train routed-pruning anchor 为 `0.25571807175874706`。q99.5/minimum
candidate 选中 38 个 frontier experts，reference widths 覆盖 `0/1/2/3/5`，38 个 floors
全部绑定；在精确 `32,768/81,920` blocks 下转移 52 blocks、改变 104 个 expert widths。

四折 train-only selector 结果：

| Fold offset | Qwen3.5 Tail-Risk | Frontier Committee Regret | Delta |
|---:|---:|---:|---:|
| 1,572,864 | 9.097020 | **9.094247** | -0.002772 |
| 1,605,632 | 6.540470 | **6.530508** | -0.009962 |
| 1,638,400 | **7.232415** | 7.232676 | +0.000262 |
| 1,671,168 | 9.163388 | **9.148276** | -0.015112 |
| Mean | 8.008323 | **8.001427** | -0.006896 |

候选满足预注册的 3/4 wins 与更低 mean PPL 门禁。Qwen3.5 tokenizer 对完整 WikiText-2 raw
test 产生 231,940 tokens，而不是仓库标准 Qwen3 协议的 233,368 tokens，因此后续结果严格
标记为 114-window model-tokenizer full-corpus **secondary evaluation**，不冒充标准正式协议：

| 方法 | Secondary full-corpus PPL | Routed-compute pruning |
|---|---:|---:|
| Qwen3.5 Tail-Risk | 7.834395 | **25.55021%** |
| **Qwen3.5 Frontier Committee Regret** | **7.820017** | 25.51377% |

PPL 绝对改善 `0.014378`、相对改善 `0.18352%`。这使 lower-PPL 方向在 Qwen3 Instruct、
Qwen3 Base 和 Qwen3.5 三个 checkpoint 上一致，并首次跨越 128→256 experts 的 topology
变化。candidate 少剪 `0.03644` 个百分点 routed compute，strict Pareto 再次失败。三次一致的
小 compute concession 表明下一核心问题不再是 frontier signal 是否有效，而是如何在 train-only
阶段控制 route-distribution shift，使 PPL 收益在不降低 evaluation routed-pruning 时保留。

完整证据：`experiments/results/exp_v4_20260728_097_qwen35_frontier_regret_preregistered.json`、
`exp_v4_20260728_098_qwen35_frontier_regret_selection_decision.json`、
`exp_v4_20260728_099_qwen35_frontier_regret_secondary_preregistered.json` 和
`exp_v4_20260728_100_qwen35_frontier_regret_secondary_result.json`。

## 27. Route-Consensus Compute Diagnostic：gap 缩小 46%，仍非 Pareto

三个 checkpoint 的 Frontier candidate 都出现小幅 evaluation compute concession，因此在不运行
任何新 test 的前提下，对 Qwen3.5 做四折 train-only retrospective diagnostic。fallback 与
candidate 同时使用四个 estimator folds 的 mean normalized route distribution 校准，target
固定为 fallback 在该分布下的 routed-pruning `0.27606453001499176`；两者仍精确保留 32,768
blocks。

新 candidate 在四个 selector folds 上仍有 3/4 PPL wins，mean PPL
`8.004293 < 8.008323`，但收益从原 offset-0 anchor 的 `0.006896` 收缩为 `0.004030`。平均
routed-pruning deficit 从 `0.00032918` 降到 `0.00017700`，缩小 `46.23%`；然而四折 compute
delta 仍全部为负。结论是 mean route consensus 可以缓解 route shift，但不能独立保证 strict
train Pareto。该实验不运行 test，只用于下一独立 prospective 设计：需要显式 per-fold compute
non-inferiority constraint，而不是继续改变 frontier score 或读取 test mismatch 调参。

证据：`experiments/results/exp_v4_20260728_101_qwen35_frontier_route_consensus_diagnostic_preregistered.json`
与 `exp_v4_20260728_102_qwen35_frontier_route_consensus_diagnostic_result.json`。

## 28. Per-Fold Compute Non-Inferiority：首次同时改善 held-out mean PPL 与 mean compute

为把 Route-Consensus 的平均约束升级为逐场景保证，本轮实现 exact multi-scenario prefix
allocator。它在保持 `32,768/81,920` global structural blocks、physical-expert prefix 单调性和
activation/frontier hard floors 的同时，对每个 train route fold 分别施加：

\[
C_f(w_{\mathrm{candidate}})\le C_f(w_{\mathrm{fallback}}).
\]

求解器使用 projected multi-dual search；每次内层仍调用 exact equal-structure prefix allocator。
在四个 estimator route folds 上，新 profile 8/8（当时为 4/4）约束可行，相对 fallback
改变 76 个 expert widths、转移 38 blocks。原四个 held-out train folds 的 PPL 为
`9.095029/6.531443/7.242797/9.158752`，相对 fallback 取得 3/4 wins；mean
`8.007006 < 8.008323`。mean routed-compute pruning 也从 `0.27408476` 提升到
`0.27413993`，首次同时改善 held-out mean PPL 与 mean compute。

但逐折 compute 只在 3/4 folds 不劣，offset `1572864` 少剪 `0.00012602`，因此最强的
all-fold gate 失败。该结果没有打开 test，而是触发更严格的 cross-fitted route constraint 与
全新 holdout 预注册。

证据：`experiments/results/exp_v4_20260728_103_qwen35_frontier_per_fold_compute_preregistered.json`
与 `exp_v4_20260728_104_qwen35_frontier_per_fold_compute_train_result.json`。

## 29. 8-Fold Cross-Fitted Holdout：PPL 4/4 胜出，但逐折 compute 泛化门禁失败

为避免在已见四折上判断 compute 泛化，本轮把原 selector offsets
`1572864/1605632/1638400/1671168` 降级为新增 route constraint folds，与原 estimator folds
合并成 8 个 train-only compute constraints；随后冻结四个从未读取 PPL 的 holdout offsets
`1835008/1867776/1900544/1933312`，每折 16×2048 train tokens。所有 route cache 均为
`parent_mode=dual`，holdout 起点严格晚于最大 constraint end `1703936`。

8-fold builder 在不调任何预注册参数的情况下给出可行证书：8/8 constraints 满足，maximum
relative violation 为 0。新增四折没有迫使结构改变，candidate width SHA 与 4-fold profile
相同，说明该 profile 在这些 train route distributions 上原本已可行。全新 holdout 结果为：

| Offset | Fallback PPL | Candidate PPL | PPL delta | Compute-pruning delta |
|---:|---:|---:|---:|---:|
| 1,835,008 | 8.515831 | **8.506764** | -0.009067 | +0.00007784 |
| 1,867,776 | 8.179707 | **8.173192** | -0.006515 | -0.00010283 |
| 1,900,544 | 8.810351 | **8.810132** | -0.000219 | -0.00003345 |
| 1,933,312 | 8.383443 | **8.377908** | -0.005535 | +0.00009391 |
| Mean | 8.472333 | **8.466999** | -0.005334 | +0.00000887 |

candidate 的 PPL 在 4/4 unseen folds 全胜，mean compute pruning 也略高；但 compute
non-inferiority 只有 2/4 folds，低于预注册的至少 3/4，因此 overall gate 与 strong 4/4 gate
均严格失败，不运行 validation/test。该负结果给出清晰边界：有限 train route scenarios 上的
硬约束可以产生可行证书，却不能自动外推为 unseen route distributions 的逐场景保证。下一步
不应继续堆叠相邻 contiguous folds；需要分布鲁棒 uncertainty set、统计 tolerance/置信界或
显式 route-shift stress scenarios，同时继续保留 PPL 4/4 的正向机制证据。

证据：`experiments/results/exp_v4_20260728_105_qwen35_frontier_crossfit_compute_holdout_preregistered.json`
与 `exp_v4_20260728_106_qwen35_frontier_crossfit_compute_holdout_result.json`。

## 30. Reference-Centered Route Envelope：Qwen3-30B 首次通过 strict train-only PPL–compute gate

8-fold finite-scenario constraints 暴露了 route-shift failure 后，本轮没有根据两个负 holdout
折拟合补偿，而是预注册一个 reference-centered coordinate envelope。对每个
`(layer, physical_expert)`，冻结 fallback width 以下的 blocks 使用 8 个 train route folds 的
coordinate-wise lower route probability，frontier 及其以上 blocks 使用 upper probability；上下
边界按经验 range 乘 `1/sqrt(8)=0.3535533905932738` 对称扩张。于是 candidate-minus-reference
retained cost 变成 block-separable 的 conservative upper bound，仍可用 exact prefix allocator，
且没有引入 validation/test route。

Qwen3-30B-Instruct 的 candidate 精确保留 `29,491/73,728` blocks，8 个 observed fold constraints
加 1 个 envelope constraint 全部满足，最大 relative violation 为 0；相对 Tail-Risk fallback
改变 46 个 expert widths、转移 23 blocks。四个新 train holdout（每折 16×2048）结果为：

| Offset | Fallback PPL | Nominal Frontier | Route Envelope | Envelope compute delta vs fallback |
|---:|---:|---:|---:|---:|
| 1,835,008 | 8.726038 | 8.699670 | **8.697641** | +0.00026845 |
| 1,867,776 | 9.666580 | 9.651218 | **9.641686** | +0.00028149 |
| 1,900,544 | 9.516540 | 9.496376 | 9.497382 | +0.00019454 |
| 1,933,312 | 9.439288 | 9.427808 | **9.425574** | +0.00029200 |
| Mean | 9.337112 | 9.318768 | **9.315571** | +0.00025912 |

相对 frozen Tail-Risk fallback，route-envelope candidate 在 PPL 上 4/4 胜出，mean PPL 绝对
改善 `0.021541`；routed-compute pruning 也在 4/4 folds 不劣，mean 提升 `0.00025912`。这
是当前首个在未见 Qwen3 train holdout 上同时通过 primary 与 strong per-fold compute gate 的
Frontier candidate。相对 nominal Frontier comparator，route-envelope 在 PPL 上 3/4 胜出、mean
再降 `0.003198`，并在 compute 上 4/4 不劣；该比较是 secondary refinement audit，不作为
profile 选择门禁。

该结果支持一个更窄但更强的机制结论：block-level committee-regret signal 的 PPL 收益并非
依靠更多 routed compute；参考 profile 中心化的 coordinate envelope 能把有限 train route
证据转化为对 unseen route shift 更保守的静态容量分配。仍不能把四个 holdout 概括成
distribution-free guarantee，也没有运行 validation/test。

证据：`experiments/results/exp_v4_20260728_107_qwen3_route_envelope_preregistered.json` 与
`experiments/results/exp_v4_20260728_108_qwen3_route_envelope_result.json`。

## 31. Qwen3 Base 独立 route-envelope 复现：strict train-only gate 通过，但不支配 nominal Frontier

为检验上一节不是 Instruct checkpoint 特例，使用独立冻结的
`/data01/datasets/Qwen3-30B-A3B` Base checkpoint，沿用预注册的
`1/sqrt(8)=0.3535533905932738` envelope 半径、8 个 observed train route folds、1 个
reference-centered envelope constraint 和 `29,491/73,728` exact structural budget。profile 在
读取任何 holdout PPL 前冻结，9/9 constraints 满足，最大 violation 为 0；本轮没有读取
validation/test。

四个全新 train holdout（每折 16×2048 tokens）如下：

| Offset | Tail-Risk fallback | Nominal Frontier | Route Envelope | Envelope compute pruning |
|---:|---:|---:|---:|---:|
| 1,835,008 | 10.400384 | 10.369294 | 10.378176 | 12.423645% |
| 1,867,776 | 11.534584 | 11.455058 | 11.463049 | 14.431043% |
| 1,900,544 | 11.256769 | 11.225389 | 11.225777 | 18.414989% |
| 1,933,312 | 11.186699 | 11.154074 | 11.167484 | 14.948575% |
| Mean | 11.094609 | **11.050954** | 11.058621 | 15.054563% |

相对 Tail-Risk fallback，route-envelope 在 PPL 上 4/4 胜出，mean PPL 绝对改善
`0.035987`，mean routed-compute pruning 从 `15.043662%` 提升到 `15.054563%`，且
逐折 compute 4/4 不劣；primary 与 strong train-only gate 均通过。相对 nominal Frontier，
route-envelope 的 mean PPL 退化 `0.007668`，但 compute 4/4 不劣，说明它是更稳健的
compute 约束候选，而不是对 nominal Frontier 的全面支配。

这给出当前最稳妥的结论：方法在 Qwen3 Instruct 与 Qwen3 Base 两个 checkpoint 上都保留
相对 Tail-Risk fallback 的 lower-PPL 与 compute non-inferiority；但 nominal Frontier 在
Base 上仍有更低 PPL，不能宣称 architecture-universal 或 distribution-free guarantee。

证据：`experiments/results/exp_v4_20260728_109_qwen3_base_route_envelope_preregistered.json`、
`experiments/results/exp_v4_20260728_110_qwen3_base_route_envelope_result.json`。

## 32. 尚未完成

- 在独立 checkpoint 或 matched-domain 上预注册双预算 anchors，复现 Frontier Committee
  Regret 的 PPL–compute Pareto dominance；当前 Instruct 与 Base 均复现 lower-PPL，但两者
  都有小幅 test routed-compute concession。
- 在更远的非 Qwen 架构或独立 matched-domain 上复制 prospective selector，避免把
  Qwen3.5 相邻谱系证据过度外推为 architecture-universal。
- 扩展到生成式 hallucination、instruction following 与 reliability-aware calibration；当前
  TruthfulQA 已证明 PPL 收益不自动对应可靠性保持。
- 为 per-fold compute constraint 引入可预注册的 route-distribution uncertainty set 或统计
  non-inferiority tolerance；必须使用新的不相邻 train holdout，不能读取 validation/test 调参。
- 在独立 checkpoint、matched-domain 或非 Qwen 架构上预注册 route-envelope 半径和 exact
  budget，验证本轮 Qwen3-30B 的 strict train-only gate 是否可迁移；不得把
  `1/sqrt(F)` 半径或本轮 PPL 结果当成普适定理。
