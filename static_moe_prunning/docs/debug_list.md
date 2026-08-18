# 2026-08-01: self-contained fused expert prior generation

# 2026-08-01: downstream screening matrix GPU range updated

- 现象：用户授权下游 screening 使用物理 GPU 0–5，但通用矩阵 launcher 仍拒绝 GPU 0–3。
- 修复：`run_downstream_matrix.sh` 现在接受且只接受物理 GPU 0–5；重复 GPU 和其他编号仍拒绝。
- 边界：该调整只适用于下游 screening 矩阵；正式 PPL 复现实验仍遵循各自记录的 GPU 协议。
- 验证：launcher focused suite `2 passed`，GPU 0–5 六卡 dry-run 通过。

# 2026-08-01: EvalScope runner blocked authorized idle GPUs 0-3

- 现象：用户已授权使用服务器所有真正空闲的 GPU，但 downstream runner 仍把
  `CUDA_VISIBLE_DEVICES` 硬编码限制为物理 GPU 4-7，导致六路 50% 下游矩阵无法同时绑定
  空闲 GPU 0-5。
- 根因：旧实验协议的局部调度约束被写进通用 artifact/evaluation runner，混淆了实验定义与
  当前服务器资源策略。
- 修复：runner 继续要求显式、唯一的物理 GPU ID，但接受任意非负整数 ID；具体可用性仍在
  启动前通过 `nvidia-smi` 和 compute-process 扫描决定，不自动选择或抢占 GPU。
- 验证：GPU `0` 和组合 `0,4,5,7` 被接受；重复 ID、负数和非数字 ID 仍被拒绝；focused suite
  `7 passed`。Qwen3 50% 下游矩阵由独立 launcher 固定每个 arm 的 GPU、profile SHA 和任务配置。

- 现象：迁入本仓库的 AMP/AIMER 先验生成器通过 `len(experts)` 统计专家数，Qwen3 fused expert
  容器只暴露 `gate_up_proj/down_proj` 张量时会在真实模型加载后抛出 `TypeError`。
- 根因：ModuleList 与 fused tensor container 的专家计数接口不同，历史实现按布局分别读取。
- 修复：新增统一 `count_routed_experts()`，fused 布局读取 `gate_up_proj.shape[0]`，ModuleList
  继续使用 `len(experts)`；增加无 `__len__` fused container 回归测试。
- 迁移复核：AMP 的 Frobenius norm 必须保持参考实现的 `torch.norm(..., p="fro")`；
  `torch.linalg.vector_norm` 不接受字符串 `ord="fro"`。
- 验证：`test_prior_builders_support_fused_expert_container_without_len`。

# 2026-08-01: EvalScope math prompt produced empty boxed answers

- 现象：GSM8K 单样本 smoke 中模型正确计算出 `18`，但默认提示中的 `\\boxed{{}}` 经
  Python format 后显示为空框，模型生成 `\\boxed{}18\\boxed{}`；EvalScope math parser
  对该字符串抽取为空，样本被判错。
- 根因：空框式提示没有明确要求把答案写入同一对花括号，模型沿提示表面格式复制了空框。
- 修复：静态 profile EvalScope runner 对 GSM8K/MATH-500 默认注入
  `\\boxed{ANSWER}` 提示并要求替换 `ANSWER`；用户显式 prompt 仍优先。
- 验证：runner prompt 单测及 GPU 4 单样本 GSM8K smoke。

# Debug Log

## 2026-07-28 — Qwen3 token 常数使 Qwen3.5 正式 PPL 协议误判

- 现象：Qwen3.5 三路 full WikiText 结果均为 114 windows、`231,940` tokens；Dense 正确
  标记 `formal_protocol=false`，但 static evaluator 只核对 windows，错误标记为 true。
- 根因：`233,368` 是既有 Qwen3 tokenizer 对仓库 filtered full WikiText 文本的 token 数，
  不是跨 tokenizer 常数；Qwen3.5 使用不同 tokenizer.json，对相同文本得到 `231,940`。
- 修复：新增 model-tokenizer-specific full WikiText token-cache builder，记录 test Arrow 与
  token cache SHA256；frozen-corpus formal 判定必须同时精确匹配 windows 和 tokens。
- 数据纪律：首次误标结果和日志完整保留在 `protocol_error_initial/`；修正后使用相同文本、
  相同 token IDs 重跑，三路 PPL 与 routed-compute 指标逐项不变，未调整任何候选或 selector。
- 红绿灯：新增 exact windows+tokens 判定测试与 full-corpus cache builder 测试，focused suite
  `4 passed`。

## 2026-07-28 — train-only selection cache 被统一 PPL 校验器拒绝

- 现象：`build_train_selection_token_cache.py` 生成的交叉拟合选择缓存已声明
  `frozen_before_selection=true`，但传入 `static_expert_ppl_eval.py` 时会被
  `validate_token_cache_payload()` 拒绝。
- 根因：统一冻结语料协议要求 `frozen_before_evaluation=true`；选择缓存构建器遗漏了这个
  通用字段，只写入了选择阶段专用字段。
- 修复：选择缓存同时写入 `frozen_before_selection=true` 与
  `frozen_before_evaluation=true`，继续保留 `test_metrics_used=false`；新增端到端回归测试，
  直接运行缓存构建入口、重新加载产物并通过统一 token-cache 校验器。
- 验证：`code/test/test_crossfit_parent_selection.py` 共 7 个测试通过。

## 2026-07-28 — cross-fit manifest 记录 SHA256 但 selector 未重新校验

- 现象：selector 会把 manifest 中声明的 profile SHA256 写入决策产物，但未重新计算实际
  profile 文件哈希，也未核对各 fold 结果行引用的是同一 profile。
- 风险：manifest 写错、profile 被替换或结果目录串线时，仍可能生成看似可审计的选择决策。
- 修复：选择前重新计算 candidate profile SHA256，并逐折校验 result row 的绝对 profile
  path 与 SHA256；任一不一致立即失败。
- 红绿灯：先新增 hash mismatch 失败用例，确认旧实现未抛错；实现门禁后 focused suite
  `8 passed`，两份真实 selector 决策重新生成且结果不变。

## 2026-07-28 — Qwen3.5 全宽静态路径出现跨层 BF16 漂移

- 现象：40 层 Qwen3.5 在所有 experts 保留 8/8 blocks 时，通用 static path 与 native
  forward 的最终 logits 最大误差达到 `1.078125`。
- 根因：Qwen3.5 native fused experts 按 expert 做 BF16 `index_add_`，通用路径先构造
  slot outputs 再加权求和；数学等价但累加顺序不同，逐层舍入误差被 hybrid 网络放大。
- 修复：当某层 profile 对所有 physical experts 都是 full width 时直接调用 patch 前冻结的
  native forward，同时按 `tokens×top_k` 记录全宽 routed stats；任何非全宽 profile 仍走
  静态 channel-prefix 路径。
- 验证：单测覆盖 exact native output 与零 routed pruning；真实 Qwen3.5 重试
  `max_abs_error=0`、逐元素完全相等，7/8-block 稀疏 smoke 也通过。

## 2026-07-28 — C4 teacher 静默使用错误的 combined parent objective

- 现象：首个 C4 matched-domain profile 输出 mode 为 `dynamic_expected_utility`，而实验协议
  要求 Conditional Dual；追查 teacher payload 发现 `parent_mode=combined`。
- 根因：C4 teacher 启动命令漏传 `--parent-mode dual`，collector 原默认值为 `combined`；
  Tail-Risk builder 又未校验 teacher objective，导致 profile 名称看似 dual 但输入并非 dual。
- 影响：原 EXP-030–032 的“同源 C4 Conditional Dual”解释作废，只能保留为 combined-teacher
  诊断；首波 C4 PPL 中 `17.084639` 也是 Dynamic Expected Utility，不是 Conditional Dual。
- 修复：teacher collector 改为必须显式传入 `--parent-mode`；Tail-Risk builder 强制
  `parent_mode=dual`，否则立即失败；新增失败回归测试并使用新目录重新采集四个 dual teachers。

## 2026-07-28 — 校准脚本无法导入 sibling package

- 现象：`ModuleNotFoundError: No module named 'moe_prune_v2'`。
- 根因：执行命令缺少 papers 根目录的 `PYTHONPATH`。
- 修复：实验命令显式加入 `/data01/home/xuzk/workspace/ai_agent/paper_wrighting/papers`。

## 2026-07-28 — 默认 Python 的 Transformers 依赖不一致

- 现象：`transformers` 在 import 阶段拒绝 `huggingface-hub==1.2.3`。
- 根因：共享 base 环境的包版本组合不兼容。
- 修复：不修改共享依赖，使用已安装的 `xhquant` Python；Qwen3 config 和完整模型加载均验证成功。

## 2026-07-28 — 正式规模 profile allocator 过慢

- 现象：逐 block greedy 在 48×128×12、50% budget 下需要 36,864 次张量循环，四个 profile 构建超过 40 秒仍未完成。
- 根因：实现重复计算所有 expert 的 next-block 候选；数学问题本身是 equal-cost、expert 内边际非递增的全局选择。
- 修复：改为一次全局 stable sort，再按 physical expert 统计选中 blocks；稳定 tie 保证同 expert 的早期等值 block 先于后期 block。
- 验证：21 个 v4 测试通过；uniform 正式 profile 构建耗时 1.80 秒，精确保持 36,864 blocks。

## 2026-07-28 — CPU profile 与 GPU routing mask 设备不一致

- 现象：四个 1-window smoke 均在首个 MoE layer 报错：CPU mask 用于索引 GPU routing tensor。
- 根因：`gather_static_widths` 为了读取 CPU profile，错误地把 `selected_experts` 移到 CPU，并让 gather 结果留在 CPU。
- 修复：以 router tensor 的 device 为执行边界，把当前 layer 的 width vector 移到该 device 后 gather。
- 回归测试：新增 CPU profile + CUDA selected-experts 用例；v4 全套 22 tests 通过。

## 2026-07-28 — Output-saliency profile JSON 无法序列化

- 现象：Tail-Risk profile builder 与 compute-calibrated builder 已成功写出 `.pt`，但在写
  伴随 `.json` 审计摘要时抛出 `TypeError: Object of type Tensor is not JSON serializable`。
- 根因：新增 `output_saliency_factor` 是 `[layers, experts]` Tensor；旧摘要过滤集合只排除
  `profile_widths` 与 `expert_utility`，遗漏该大张量。
- 修复：两级 builder 的 JSON summary 均显式排除 `output_saliency_factor`，Tensor 仍完整
  保存在 `.pt` profile 中；provenance、beta、fold hashes 和聚合方法继续写入 JSON。
- 红绿灯：先在 Tail-Risk builder 与 compute builder 测试中加入 saliency factor，复现两处
  序列化失败；修复后相关 focused suite `9 passed`。

## 2026-07-28 — Compute calibration 静默丢弃 output-saliency floors

- 现象：结构 profile 已记录 31 个 output-saliency safety floors，但经过
  `build_compute_calibrated_profile.py` 后，候选宽度与无 output floor 的 fallback 完全一致，
  包括原本 0/1-block 的六个目标 experts。
- 根因：`_risk_floors()` 只读取 `risk_floor`，未合并新增的
  `output_saliency_risk_floor`；compute allocator 因此在第二阶段静默丢失新硬约束。
- 修复：compute builder 逐项合并 activation-tail 与 output-saliency 两类 selected experts，
  同一 expert 取最大 min width，并增加越界校验。
- 红绿灯：先新增两类 floor 应合并为 `[[1,2]]` 的失败测试，旧实现实际返回 `[[1,0]]`；
  修复后 focused suite `11 passed`。重新冻结后候选相对 fallback 移动 15 blocks，六个目标
  experts 的 floor 均实际生效。

## 2026-07-28 — Unique-contribution Tensor 再次阻断 compute JSON 摘要

- 现象：unique-contribution structural profile 成功，compute profile `.pt` 也已写出，但伴随
  `.json` summary 抛出 `TypeError: Object of type Tensor is not JSON serializable`。
- 根因：compute builder 的摘要过滤集合已排除 `output_saliency_factor`，但新增的
  `unique_contribution_score` 与 `co_route_uniqueness_folds` 仍被直接传给 JSON encoder。
- 修复：两项大 Tensor 仅保留在完整 `.pt` profile，在 JSON summary 中排除；阈值、selected
  experts、aggregation 和 provenance 仍保留为可审计标量/列表。
- 红绿灯：在 compute builder 回归用例中加入两项 Tensor，先复现序列化失败；修复后
  focused compute suite `2 passed`，真实 compute profile 与 JSON 均重新生成成功。

## 2026-07-28 — Frontier-regret score Tensor 的摘要序列化门禁

- 现象：新增 `frontier_committee_regret_score` 后，若直接沿用 compute builder 的摘要集合，
  会再次把 `[layers,experts]` Tensor 交给 JSON encoder。
- 修复：先在回归测试的 source profile 中加入该 Tensor，确认旧过滤器抛出序列化错误；随后
  将其与其他大审计 Tensor 一样只保存在 `.pt`，JSON 保留 floor threshold、selected experts
  和 provenance。
- 结果：真实 structural/compute profiles 的 `.pt/.json` 均成功生成，避免实验运行到 profile
  冻结后才因审计文件失败。

## 2026-07-28 — 8-Fold profile builder 首次启动缺少包根路径

- 现象：第一次启动 cross-fitted builder 时在 import 阶段报
  `ModuleNotFoundError: moe_prune_v2`；模型/GPU 计算未开始，candidate profile 未创建。
- 根因：命令只设置了 v4 `code/` 的 `PYTHONPATH`，遗漏同级 packages 根目录；v4 允许使用的
  `moe_prune_v2` helper 因而不可解析。
- 修复：保留失败日志，用包含 `papers/` 与 `moe_prune_v4/code/` 的显式 `PYTHONPATH` 按完全
  相同预注册参数重跑；第二次 exit 0，8/8 constraints 可行。没有改算法或调超参数。

## 2026-07-28 — Route-envelope holdout evaluator 环境与路径启动错误

- 现象：第一次 Qwen3 holdout evaluator 使用 `xh2`，在 Qwen3 MoE residual 合并处报
  `Tensor + tuple`；第二次 retry 命令将 candidate 的 `--profile` 误写为 channel cache，
  两次都没有产生正式 PPL JSON。
- 修复：按既有 Qwen3 成功记录切换到已验证的 `xhquant` 环境，并把 candidate profile 路径改为
  frozen `candidate_expansion_inv_sqrt8.pt`；正式 retry2 四路、三 profile 全部 exit 0。
- 回归/审计：失败日志保留在 `qwen3_route_envelope_holdout_20260728` 与
  `qwen3_route_envelope_holdout_retry_20260728`；正式结果只读取
  `qwen3_route_envelope_holdout_retry2_20260728`。

## 2026-07-28 — REAP Qwen3-30B 单卡加载与保存后重载失败

- 现象一：上游 CLI 将完整 Qwen3-30B 直接放到单张 80GB GPU，在第 10/16 shard OOM。
- 修复一：为隔离的 upstream CLI 增加 `REAP_DEVICE_MAP=auto`，只暴露物理 GPU4–7，多卡
  加载后校准和 48 层 whole-expert pruning 均完成。
- 现象二：每层 router/experts 已从 128 裁到 52，但保存的 `config.json` 仍写 128，重载时报
  router weight `[52,2048]` 与配置创建的 `[128,2048]` 不匹配。
- 根因：通用 Qwen `_AttrAdapter.update_model_config()` 是空实现。
- 修复二：同步 `config.num_experts/num_local_experts/n_routed_experts`，重跑后 checkpoint
  可重载；PPL 评估再使用 `device_map=auto` 避免单卡 activation OOM。

## 2026-07-31 — 冻结先验白名单被全局 Tensor 规则覆盖

- 现象：`.gitignore` 已为 8 个 AMP/AIMER 表添加目录白名单，但后续全局 `*.pt` 规则再次
  将它们排除，发布候选集中实际没有任何先验表。
- 根因：Git ignore 规则按最后匹配项生效，目录级 negation 不能覆盖文件后方的 `*.pt`。
- 修复：在全局 Tensor 规则后逐文件重新放行 8 个冻结表；发布审计同时要求候选集中恰好
  存在这 8 个 `.pt`，拒绝其他 Tensor 文件，并校验 `FROZEN_PRIORS.sha256`。
- 验证：发布审计通过，8/8 SHA256 匹配；候选集约 2.7 MB；不依赖历史兼容层的核心测试
  `37 passed`。

## 2026-08-01 — EvalScope preflight 未交叉验证逐层预算和 channel block 数

- 现象：`allocation_scope=per_layer` 的 profile 即使声明的逐层目标与 `profile_widths` 不一致，
  或 channel cache 的 block 数与 profile `num_blocks` 不一致，仍可进入大模型加载阶段。
- 根因：preflight 只验证了 profile 总预算、文件 SHA256 和 channel tensor 形状，没有交叉验证
  逐层预算元数据与 profile/channel 拓扑。
- 修复：要求 `target_blocks_by_layer`、`actual_blocks_by_layer` 与逐层 width 总和完全一致，并
  要求每层 channel block 数等于 profile `num_blocks`。
- 验证：新增两个先失败后通过的 artifact 测试；`code/test/test_evalscope_model_api.py` 共
  `5 passed`。

## 2026-08-01 — EvalScope 入口丢失显式 model family

- 现象：runner 能解析 `--model-family`，但 `load_supported_moe()` 不接收该参数，导致没有可
  从路径或 config 自动识别的 checkpoint 无法使用显式 Qwen3.6/Gemma adapter。
- 根因：共享 loader 只调用路径推断，EvalScope API 也没有把已解析的 family 传入 loader。
- 修复：贯通 `load_supported_moe(model_family=...)` 和 EvalScope API 的显式 family 参数，并
  统一走 family alias normalization。
- 验证：新增 loader regression test；模型加载、artifact preflight 和 runner 相关测试共
  `12 passed`。

## 2026-08-01 — EvalScope manifest 无法证明使用预冻结 artifact 与精确源码

- 现象：runner 会重新计算当前 profile/channel SHA256 并立即把它们作为 expected 值，且
  manifest 不记录方法代码或 EvalScope 代码身份；文件或 dirty worktree 改动后仍可能生成
  看似完整的评测记录。
- 根因：入口缺少外部预注册 SHA 参数、只读 preflight 模式和 runtime source-tree identity。
- 修复：正式入口要求显式传入两个预冻结 SHA256；新增 `--preflight-only`，在不注册或加载
  模型的情况下完成 artifact 校验并写 manifest；manifest 记录两仓库 commit、dirty 状态、
  runtime pathspec、文件数和按相对路径/内容计算的 tree SHA256。
- 验证：SHA parser、dirty source-tree hash 和 preflight-only 不加载模型均有回归测试；真实
  smoke artifacts 的 CLI preflight 通过并生成带两套源码身份的 manifest。

## 2026-08-01 — EvalScope math prompt 的 boxed placeholder 被模型照抄

- 现象：真实 Qwen3 GSM8K 单样本 smoke 已算出正确结果 `18`，但先后使用 `\\boxed{ANSWER}`
  和 `\\boxed{}` 的提示时，模型分别输出占位词或空框，EvalScope 抽取为空，错误得到
  `mean_acc=0`。
- 根因：提示文本展示了可复制的占位/空框格式；模型遵循表面格式而不是把计算结果填入其中。
- 修复：默认 math prompt 不再展示任何 literal boxed 内容，只要求使用非空 LaTeX `\\boxed`
  表达式、把计算出的数字放入花括号，并明确禁止空框和占位符。
- 验证：先失败后通过的 prompt regression test；同一冻结 profile 在物理 GPU 4 的最终
  GSM8K 单样本 smoke 抽取 `18`，`mean_acc=1.0`，产生完整 EvalScope report/manifest。

## 2026-08-01 — REAP zero-width profile 不等价于 whole-expert deletion

- 现象：若只把 REAP 删除 expert 的 profile width 设为 0，原 router 仍可能把这些 expert
  选进 top-k，浪费 routed slots；该行为不等价于官方删除 router rows 后在保留 experts 中
  重新 top-k。
- 根因：prefix runtime 原本只控制 expert 内执行宽度，不改变 router candidate set。
- 修复：REAP profile 增加 `retained_expert_mask`，路由时先把删除 expert 的 logits mask 为
  `-inf`，再在保留集合中 top-k，并保留原始 router logits 输出契约。
- 验证：CPU 单测证明 runtime mask 与物理删除同一 experts/router rows 的 expert mixture 输出
  完全一致；Route×Tail/Tail-Risk 未提供 mask 时保持原路由行为。

## 2026-08-01 — 官方 REAP 与本方法曾各自重新 tokenize calibration

- 现象：即使数据集名称、split 和 token 数相同，官方 REAP、channel collector 和 teacher
  collector 各自 tokenize，无法证明实际 token IDs 完全一致。
- 根因：三个入口都从 raw dataset 独立构造 batch，没有共享只读 token artifact。
- 修复：新增带 tokenizer/config identity、source、token IDs、attention-mask 语义和 SHA256 的
  `shared_moe_pruning_calibration` artifact；Official REAP bridge、Hessian channel collector
  和 Tail-Risk teacher 均直接消费该 artifact，禁止静默 retokenize。
- 验证：shared reader/collector/profile/pair-audit 测试通过；真实 Qwen3 `1×2048` smoke 在
  pinned official REAP commit 上完成全部 48 层 observation，生成精确 50% profile。

## 2026-08-01 — 官方 REAP benchmark 入口与现有 xhquant vLLM 不兼容

- 现象：导入 `reap.layerwise_prune` 会因现有 vLLM 缺少 `get_open_port` 失败。
- 根因：官方 REAP benchmark/CLI 栈固定了另一套 vLLM/torch 版本；按协议不能为此修改共享环境。
- 修复：保持官方 checkout clean，不导入其 benchmark 入口；只导入并调用 pinned commit 的
  `LayerwiseMoEObserver`、observer config 和 pruning metrics 核心，能力评测统一走当前 EvalScope。
- 验证：官方 `test_pruning_metrics.py` 与 `test_layerwise_observer.py` 共 5 个测试通过；真实
  Qwen3 REAP profile 通过同一 EvalScope GSM8K smoke，正确抽取 `18`、`mean_acc=1.0`。

## 2026-08-01 — REAP 与 Route×Tail matched profile-stage 比较闭环

- 数据：两方法共享 train-only `1×2048` smoke calibration artifact，token SHA256 为
  `62451e3889253427941cab46c9ce7d2d25f80b17cd7c37826238706efb3a420e`；共享 held-out
  WikiText test `1×2048` cache，文件 SHA256 为
  `89778d42b3fdb708eae3c218433c3ee7e986133e035da54bcad40a46515b967e`。
- 预算：Qwen3 50%，两者均保留 `36,864/73,728` blocks，且每层均为 768 blocks；严格
  `per_layer_controlled` pair audit 通过。
- 官方来源：clean REAP commit `1970473c51ca3caeb98c10392f15b3a08a672974`，router
  weights renormalization 开启；48 层 layer-wise observer 全部完成。
- smoke 结果：Official REAP PPL `7.291091`，Route×Tail PPL `6.719987`；REAP routed-channel
  pruning 为 0，Route×Tail 为 `9.8308%`。两者 EvalScope GSM8K 单样本均正确抽取 `18`。
- 解释边界：该结果只有一个 calibration/evaluation window，只验证协议、预算、runtime 和
  评测链路，不能作为正式优劣结论，也不能用于新增 ratio、seed 或方法选择。

## 2026-08-04 — AIMER 恢复 shard 无法进入正式发布目录

- 现象：完整 MMLU 由中断任务的 subject 级产物和 tail jobs 恢复到
  `aimer_mmlu_consolidated/`，但发布脚本只匹配 `aimer_mmlu_gpu*`；随后各 dataset shard 的
  顶层 `reports/report.html` 又发生字节冲突。
- 修复：发布器优先接受显式 `aimer_<dataset>_consolidated`，同时仍严格要求每个 dataset
  只有一个选定来源；reports 只合并 `reports/<model-id>/` 的结构化 JSON，不合并每个 shard
  独有的顶层 HTML。
- 验证：新增 consolidated MMLU 与不同 HTML 回归场景，merger focused suite `3 passed`；
  正式发布完成并生成 merged manifest。

## 2026-08-04 — 缓存评审报告的 null perf_metrics 阻断对比生成

- 现象：统一 MMLU 使用完整 prediction cache 重跑 review，报告分数和 570 样本有效，但
  `perf_metrics=null`；下游比较器无条件调用 `.get()` 导致 `AttributeError`。
- 修复：仅在 `perf_metrics` 为对象时读取 summary，否则将 latency/throughput 显示为 `-`，
  分数与样本数照常纳入比较。
- 验证：新增 null performance regression test，comparison focused suite `3 passed`；
  `downstream_comparison.md` 成功包含 AIMER 六项结果。
