# MoE Prune V4: Static Expert Pruning

本目录专门用于 Qwen3-30B-A3B-Instruct-2507 的静态专家结构化剪枝研究。
核心目标是在固定物理专家宽度 `width[layer, physical_expert]` 下，以 64-channel
prefix block 为粒度压缩 routed experts，并使用 WikiText-2 与 C4 matched-domain full
perplexity 作为主要方法门禁。

> 发布状态：模型权重、token/channel cache、冻结 profile、日志和实验结果均不进入
> Git。当前使用的 4 组 AMP/AIMER expert-prior 表作为带 SHA256 的冻结输入进入仓库，
> 可以复现读取这些先验后的 profile 分配逻辑。AMP/AIMER 先验生成器、模型加载、物理
> expert-prefix 执行、PPL 入口和 EvalScope ModelAPI 已迁入本仓库，不再依赖历史
> `moe_prune_v2`/`moe_prune_v3` helper。模型权重和正式校准 artifact 仍须由运行方提供。

## 目录

- `code/src/`：静态 profile 分配、物理专家索引运行时和统计。
- `code/scripts/`：RMS/Hessian 校准、profile 冻结、WikiText-2/C4 frozen-corpus PPL 评估。
- `code/test/`：预算、prefix、物理专家索引和 dense-equivalence 红绿灯测试。
- `experiments/calibration/`：仅提交白名单内的冻结 AMP/AIMER 表和哈希清单；其余
  train-only 校准缓存均在本地生成且不进入 Git。
- `experiments/profiles/`：评估前冻结的静态宽度 profile，本地生成且不进入 Git。
- `experiments/results/`：机器可读实验结果，本地生成且不进入 Git。
- `experiments/logs/`：独立运行日志，本地生成且不进入 Git。
- `docs/`：研究方案、idea、实验总表和研究报告。
- `.omx/`：validator-gated autoresearch 状态。

## 当前正式协议

- 模型：通过 `MODEL_PATH` 指向本地 Qwen3-30B-A3B-Instruct-2507 checkpoint。
- 校准：WikiText-2 raw-v1 train，连续 `128 × 2048 = 262,144` tokens。
- 评估：WikiText-2 raw-v1 test，完整 `114` windows、context `2048`。token 数依赖模型
  tokenizer：Qwen3 为 `233,368`，Qwen3.5 为 `231,940`；正式结果使用冻结的模型专属
  full-corpus token cache，并记录 Arrow/cache SHA256。
- C4 评估：`c4_validation_114x2048_v1`，validation 前 `233,472` frozen tokens，
  `114 × 2048` windows；token cache 和 Arrow source 均记录 SHA256。
- GPU：仅物理 GPU 4、5、6、7；每卡一个模型进程。所有模型命令必须显式设置
  `CUDA_VISIBLE_DEVICES=4/5/6/7` 之一，禁止使用物理 GPU 0–3。
- smoke：只检查加载、形状、显存和协议，不用于选择方法。

## 环境与外部输入

建议使用 Python 3.10。已完成的实验使用 PyTorch `2.10.0+cu128`；基础依赖列在
`requirements.txt`。CUDA 版 PyTorch 应按目标服务器的 CUDA/驱动从 PyTorch 官方源安装，
再安装其余依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

运行前显式提供模型路径；模型权重是外部实验输入，不在仓库中：

```bash
export MODEL_PATH=/path/to/Qwen3-30B-A3B-Instruct-2507
export PYTHON_BIN="$(command -v python)"
```

WikiText-2/C4 可由 Hugging Face `datasets` 在线读取，也可使用 `--arrow-file` 提供离线
Arrow shards。所有正式缓存必须记录 source SHA256、token-cache SHA256、split、token 区间、
sequence length 和 window 数。缓存、profile 与结果目录会由脚本自动创建。

发布前运行：

```bash
bash scripts/audit_publication.sh
```

该审计检查机器绝对路径、超大文件、模型权重、意外 tensor 产物、冻结先验白名单、
先验 SHA256 和 shell 语法。历史兼容 helper import 会作为明确警告列出；在这些 helper
迁入 V4 前，审计通过只表示发布边界干净，不表示完整端到端运行时已经自包含。

## 方法边界

静态异构宽度本身不是论文创新点；MoE-Slimming 已覆盖 attribution-guided
heterogeneous structural slimming。Dynamic-Regret 的 selection-indicator 机制已被完整
PPL 消融否决。当前候选是 **Tail-Risk Constrained Conditional Utility**：先把每个 token
routed set 内归一化的 `gate×AMP/AIMER` 条件期望效用聚合成静态 expert-prefix 容量，
再用 train-only activation-tail proxy 自动识别均值目标遗漏的 rare catastrophic paths，
以统一全层 global-tail threshold 为极少数高风险 physical expert prefixes 设置最小 width floor，并从最低边际效用 blocks
回收完全相同的结构预算。已知 Super-Expert ID 仅用于机制验证，不进入最终 selector。

当前证据：Tail-Risk 在 WikiText 50%/60%/80% 和 C4 matched-domain 50%/60%/80% 均
优于 Conditional Dual；C4 60% 四个独立 train folds 也 4/4 胜出。在独立重建全部
train-only calibration/profile 的 Qwen3-30B-A3B base checkpoint 上，60% PPL 也从
Route×RMS `10.623834`、Conditional Dual `10.371195` 降至 Tail-Risk `10.341598`。
该结果是同架构跨 checkpoint 复现。TruthfulQA 817 题上，Tail-Risk 的 MC1 略高于
Conditional Dual，但 MC2 从 `0.440415` 降到 `0.437559`，未通过预注册双指标标准；三种
60% profiles 也都明显低于 Dense。因此 PPL 收益不能写成 factual-reliability 保持。
跨架构与 instruction-following 仍是未完成门禁。

当前主要报告：`docs/STATIC_EXPERT_RESEARCH_REPORT.md`。完整成功、失败和负实验：
`docs/ALL_EXPERIMENTS_COMPLETE.md`。

端到端复现入口见 `docs/REPRODUCTION_COMMANDS.md`。当前最佳方法可直接执行：

```bash
MODEL_PATH="$MODEL_PATH" GPU=4 PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_our_best_end_to_end.sh
```

冻结 static profile 的 EvalScope 能力评测入口为
`code/scripts/run_evalscope_static_profile.py`。入口在加载模型前校验 profile/channel 文件的
预冻结 SHA256、train-only provenance、逐层预算和 channel 拓扑，并在 manifest 中记录方法
代码与 EvalScope 代码的 commit、dirty 状态及 runtime tree SHA256。正式命令见复现文档。

外部比较方法的 checkout、补丁和运行脚本不属于本方法的 GitHub 发布范围。历史对比实验
仍保留在实验总表中，但不会作为本仓库的可运行组件提交。

最新静态剪枝结论：Reference-Centered Route Envelope 已在 Qwen3-30B-Instruct 与独立
Qwen3-30B Base 的全新 train holdout 上通过预注册的 PPL + routed-compute 门禁。Instruct
mean PPL 为 `9.315571`，Tail-Risk fallback 为 `9.337112`；Base mean PPL 为 `11.058621`，
fallback 为 `11.094609`；两者均为 4/4 PPL wins 和 4/4 compute non-inferior。Base 上相对
nominal Frontier 的 mean PPL 高 `0.007668`，因此 route-envelope 是更稳健的 compute 候选，
不是对所有方法全面支配。正式 test 的当前最佳 PPL 仍是 Frontier Committee Regret：
Instruct `8.658925`、Base `10.322812`。

与已实现静态基线的正式 WikiText-2 对比（60% structural pruning、同一 64-channel block
预算）：

| 方法 | Qwen3 Instruct PPL | Qwen3 Base PPL | 结论 |
|---|---:|---:|---|
| Route×RMS | 见历史正式记录 | 10.623834 | 基础路由/激活重要性基线 |
| Conditional Dual | 见历史正式记录 | 10.371195 | 优于 Route×RMS |
| Tail-Risk | 8.688653 | 10.341598 | 相对 Conditional Dual 稳定改善 |
| Frontier Committee Regret | **8.658925** | **10.322812** | 当前正式 test 最低 PPL；有微小 compute concession |
| Reference-Centered Route Envelope | 新 train holdout `9.315571` | 新 train holdout `11.058621` | 4/4 compute non-inferior；未运行 test |

相关工作边界已记录在 `docs/RELATED_WORK_NOVELTY_AUDIT.md`：MoE-Slimming、TENP、REAP、
MoE-Pruner、MoSE、POP、MAESTRO、FLAP 等已经覆盖异构宽度、router/activation scoring、
prefix slicing 或动态宽度等局部思想，因此不能把“静态大专家变小专家”本身作为创新。当前
最小可辩护贡献是 block-level committee residual、reference-centered sign-split route
envelope、exact physical-expert prefix geometry，以及 train-only 可审计的 compute
non-inferiority protocol。

跨架构验证另支持 `Qwen2MoeForCausalLM` 的 Qwen1.5-MoE sparse block 合约：保留
shared expert、未归一化 top-k 权重语义和 router logits 返回值。对应 profile/calibration
必须在该模型上独立生成，不能复用 Qwen3 profile。

Qwen1.5-MoE 60% 正式结果为 Route×RMS `10.114996`、Conditional Dual `10.626735`、
Tail-Risk `10.239141`。Tail-Risk 能修复 Dual，但未超过 Route×RMS，因此跨架构预注册
标准失败；当前方法不得表述为 architecture-universal recipe。

失败后提出的 Route×Tail（route-count parent + λ=0.50 typical/tail channel coverage）在
WikiText 探索中为 `10.013247`，随后在独立预注册 C4 matched-domain 验证中从 Route×RMS
`17.328660` 降到 `16.203895`，同时 routed-compute 剪枝略高。该结果支持更稳健的 coverage
组件，但仍需 Qwen3/第三模型验证和相关工作查重。

Qwen3 C4 四折 Route×Tail 均值为 `17.100295±0.093404`，4/4 优于 Route×RMS，但略差于
Conditional Dual、明显差于完整 Tail-Risk。当前应把 Route×Tail 视为跨架构底座，把
dual/floor 视为需 train-only applicability 判据控制的架构条件增强。

该判据现已实现为四折 train-only cross-fitted selector：校准区间与四个选择区间严格
不重叠，Route×Tail 为 fallback；只有 Tail-Risk 在严格多数折获胜且 mean selection PPL
更低时才启用。Retrospective diagnostic 中，Qwen1.5 四折 4/4 回退 Route×Tail
（mean `9.893688` vs `10.323414`），Qwen3 四折 4/4 启用 Tail-Risk
（`8.992431` vs `9.289503`），均复现此前正式 test 的相反偏好。该结果支持 selector
plausibility，但不是独立验证。随后在未读取 Qwen3.5 test PPL 前，四折 4/4 选择 Tail-Risk；
正式 full-corpus PPL 为 Tail-Risk `7.834395`、Route×Tail `8.062748`，相对改善 `2.8322%`，
prospective selector 门禁通过。两者均为 60% exact structural pruning；Tail-Risk routed
compute 剪枝少 `0.1692` 个百分点，因此仍需 matched-compute follow-up。

Qwen3-30B 上的 matched-compute follow-up 已进一步完成。新增 train-only Lagrangian
dual-budget allocator，在精确保留 `29,491/73,728` structural blocks 的同时，把 Route×Tail
与 Tail-Risk 校准到 `14.9109%` 和 `15.2327%` 两个 expected routed-compute anchors。
两个 anchors 上 Tail-Risk 都同时取得更低 PPL 与更高实际 compute pruning：
`8.701386 < 8.996928`、`8.688653 < 8.876885`。因此 Qwen3 上的收益不是由执行更多热门
expert channels 解释；该结果仍是已见 test 后的机制审计，独立 prospective 双预算复现待完成。

随后对四折 route-cost CVaR/worst-case、coverage consensus、teacher consensus 与
teacher lower-confidence-bound 做了 train-only follow-up。它们在 WikiText-2 上均未超过
offset-0 Tail-Risk；LCB 候选在四个独立 train selection folds 上未达到严格多数，因此
selector 回退。当前稳健 formulation 是 domain-adaptive Tail-Risk profile 配合
train-only applicability gate，而不是把跨折均值/CVaR 强行写入静态 profile。

最新实现进一步支持四折 output-contribution 校准：采集
`mean_active_token(gate × ||expert_output||₂)`，逐折、逐层归一化后平均，再以 β 控制的
乘法因子校准 Conditional-Dual expert utility。该能力已通过测试并保留完整 train-only
provenance，但 β=`0.25/0.50/1.00` 在四折 selector 上均未超过 Tail-Risk fallback
（mean PPL `9.032436/9.039200/9.115010` 对 `8.992431`），因此当前研究结论是回退，
不运行这些候选的 formal test。routing-entropy layer prior 同样被四折一致否决。

随后测试了稀疏 Output-Contribution Safety Floor：四折 train selector 以 3/4 folds 选择
候选，但正式 WikiText PPL 为 `8.689485`，略差于 fallback `8.688653`；候选只在 routed
compute pruning 上多剪 `0.00201` 个百分点。因此该 refinement 也不进入主方法。当前下一
步又实现了 redundancy-aware unique contribution：用 train-only gate-weighted co-routing
fingerprint 区分高幅值功能克隆，并以 worst-fold unique contribution 构造 q99.5 safety
floors。该候选在四折 selector 上 0/4 获胜，mean `8.997872 > 8.992431`，未打开 full test。
三组 contribution 负结果共同指向 expert-level 标量与 within-expert channel-prefix demand 的
层级错配。随后实现的 Frontier Committee Regret 将 non-redundancy 直接下沉到 frozen
fallback 的 first-pruned 64-channel block：四折 train selector 4/4 获胜，正式 WikiText-2
PPL 达到 **`8.658925`**，优于原最佳 `8.688653`，相对改善 `0.34214%`。候选保持精确
29,491/73,728 blocks，但 test routed-compute pruning 少 `0.01003` 个百分点，因此这是当前
新最佳 PPL，而不是 strict PPL-compute Pareto 结果。随后在 Qwen3-30B-A3B Base checkpoint
上独立重建全部 estimator/selector/profile：四折 selector 再次 4/4 通过，正式 PPL
`10.322812 < 10.341598`，相对改善 `0.18166%`。Base candidate 的 test routed-compute
pruning 少 `0.02514` 个百分点，因此跨 checkpoint lower-PPL 已复现，但独立 prospective
strict PPL-compute Pareto 仍待完成。进一步迁移到 40×256 expert 的 Qwen3.5-35B-A3B 后，
train selector 以 3/4 folds 通过，model-tokenizer secondary full-corpus PPL 为
`7.820017 < 7.834395`，再次复现 lower-PPL；该协议只有 231,940 tokens，不作为仓库标准
233,368-token formal claim。三个 checkpoint 的候选均有小 routed-compute concession，下一
门禁是预注册的 route-shift-aware matched-compute 复现，而不是继续调 frontier quantile。

当前已实现逐 train-fold compute non-inferiority 的 exact multi-scenario allocator。在 Qwen3.5
上，8 个 route constraint folds 全部满足且保持精确 32,768/81,920 blocks；四个全新、未见
train holdout 上 candidate PPL 4/4 胜出，mean `8.466999 < 8.472333`，mean routed-compute
pruning 也略高。但逐折 compute non-inferiority 只有 2/4，未达到预注册至少 3/4 的门禁，
因此没有运行 validation/test，也不能声称 strict Pareto。该结果把下一研究问题收窄为：对
unseen route-distribution shift 建模统计 uncertainty set，而不是继续增加相邻训练折或调整
frontier quantile。

随后在用户指定的 Qwen3-30B-Instruct 主模型上实现了 reference-centered route envelope：
reference width 以下的 blocks 使用 8-fold train route lower envelope，frontier 以上使用 upper
envelope，边界按 `1/sqrt(8)` 预注册扩张。candidate 在四个全新 train holdout 上 PPL 4/4
优于 Tail-Risk fallback，mean `9.315571 < 9.337112`；routed-compute pruning 也 4/4 不劣，
mean `0.16308199 > 0.16282288`。相对 nominal Frontier comparator，PPL 3/4 胜出且 compute
4/4 不劣。该结果只证明新的 train-only prospective gate，未运行 validation/test；下一步仍
需独立 checkpoint/domain 预注册半径复现。
