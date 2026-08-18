# Function List

- 精确全局 equal-cost prefix block 分配。
- 双预算 compute-calibrated prefix 分配：在精确全局结构 block 数不变时，以 train-only
  route counts 作为 expected routed cost，通过 Lagrangian multiplier 搜索指定计算 anchor；
  保留 physical-expert prefix 单调性、risk floors、profile/cache SHA256 与离散误差审计。
- train-fold route-cost aggregation：支持 mean、coordinate-wise worst-case 与 empirical
  upper-tail CVaR，并记录 alpha、fold 数、归一化质量和 train-only provenance；当前 PPL
  研究已验证 CVaR/worst-case 负结果并保留审计。
- per-fold compute non-inferiority prefix allocation：在 exact global structural block budget、
  physical-expert prefix 单调性和多类 hard floors 下，使用 projected multi-dual search 对每个
  train route fold 分别约束 candidate retained cost 不高于 frozen reference；输出逐折 relative
  violation、dual multipliers、feasible iteration count 与最小 violation audit，不可行时拒绝写 profile。
- reference-centered route-envelope allocation：以 frozen reference width 为中心，对 prefix 内
  blocks 使用 observed train route lower envelope、frontier 及以上 blocks 使用 upper envelope，
  支持 `range/sqrt(F)` expansion；将 block-dependent envelope cost 与逐折 hard constraints 联合
  注入 exact prefix allocator，并写出 envelope mass、expansion 与 constraint violation provenance。
- 可选 per-expert width cap 与 0-width experts。
- 可选 per-physical-expert minimum width floor，并在相同总 block 预算下自动回收容量。
- uniform、RMS、route×RMS、dual-prior×route×RMS profile 构建。
- 固定 `(layer, physical_expert)` width gather，禁止 router-rank 替代。
- Qwen3 MoE 静态 prefix runtime emulation；共享专家保持不变。
- 结构剪枝率与 routed compute 剪枝率分离统计。
- train-only profile provenance、cache SHA256 与预算校验。
- WikiText-2 one-window smoke 与 114-window full PPL 评估。
- 模型 tokenizer 专属的 full WikiText 冻结 cache：复现仓库 full-corpus row filtering，记录
  Arrow/cache SHA256，并以精确 windows+tokens 双条件判定 formal，避免把 Qwen3 token 数
  错当成跨模型常数。
- 任意文本语料的冻结 token-cache PPL：显式 protocol name、dataset/config/split、窗口数、
  token 数、sequence length、Arrow/token-cache SHA256 与 test-independent profile 元数据。
- C4 validation 正式协议 `c4_validation_114x2048_v1`：114×2048、233,472 frozen tokens。
- 原生 dense arbitrary-corpus PPL baseline，与静态 profile evaluator 共用同一 frozen token cache。
- Dense/静态 profile TruthfulQA multiple-choice evaluator：固定 prompt、条件 log-likelihood、
  MC1 accuracy、MC2 true-probability、profile/cache SHA 与 routed-compute 统计。
- 独立 AMP/AIMER expert-prior cache builder：记录 model path、layer/expert shape，并拒绝
  跨层 expert 数不一致的缓存。
- Qwen2-MoE 静态运行兼容：保留 shared expert、`norm_topk_prob=false` 和
  `(hidden, router_logits)` forward 合约；teacher top-k/block metadata 按架构推导。
- Qwen3.5/3.6 hybrid MoE 模型族推断：使用本机已有新版 Transformers 环境加载
  `Qwen3_5MoeForConditionalGeneration`，支持 40×256 routed experts、top-8、shared
  expert 与 linear/full-attention 混合 decoder；不按脚本硬编码 Qwen3 模型族。
- 全宽 exact-native fast path：profile 对某层所有 experts 保留全部 blocks 时调用原生
  forward，消除 fused BF16 累加顺序漂移，同时保持 routed pruning 统计为 0；稀疏层仍走
  physical-expert channel-prefix 静态实现。
- train-only per-channel activation-tail risk 校准：`max|z|×down-column-norm`。
- Conditional Dual expert utility 到 tail-aware prefix coverage 的重绑定。
- Contribution-Calibrated Tail-Risk：采集每个 physical expert 的
  `mean_active_token(gate × ||expert_output||₂)`，支持多个 train folds 逐折、逐层归一化后
  求均值，再以可审计的 `saliency^beta` 因子校准 Conditional-Dual expert utility；最终仍由
  exact 64-channel prefix allocator、rare-risk floors 和结构/compute 双预算约束冻结 profile。
- Output-Contribution Safety Floor：将多折归一化 output saliency 的全局尾部转为稀疏
  minimum-width constraints，与 activation-tail floors 按 expert 逐元素取最大值；compute
  calibration 同时保留两类 floors，记录 selected/newly-constrained experts、阈值与 provenance。
- Redundancy-Aware Unique Contribution：collector 在同一次 expert forward 中累计 gate-weighted
  co-routing context；支持去对角 committee fingerprint、最近邻 cosine uniqueness、四折
  mean/minimum 聚合和稀疏 unique-contribution floors。compute calibration 会与 activation-tail、
  output-saliency floors 一并合并。当前 q99.5/minimum 候选已被 0/4 train selector 否决。
- Frontier Committee Regret：直接在 64-channel block 层估计相对同 token 其他 routed experts
  委员会输出的正交残差能量；支持 diagonal down-Gram 低开销近似、frozen reference width 的
  first-pruned frontier gather、逐折逐层归一化、mean/minimum 聚合、global-quantile 一块增量
  floors，以及 compute calibration 中的硬约束保留。q99.5/minimum 候选在 Instruct 上取得当前
  正式新最佳 WikiText PPL `8.658925`，并在完全独立 Base calibration/selector 上以 4/4 folds
  通过、正式 PPL `10.322812 < 10.341598` 复现 lower-PPL 方向；在 40×256 Qwen3.5 topology
  上又以 3/4 selector 通过，并在 model-tokenizer secondary full corpus 上取得
  `7.820017 < 7.834395`。Qwen3.5 secondary 协议不等同于 233,368-token标准 formal protocol。
- 基于全局 quantile 与 relative-max 的全层自动 sparse risk-floor selector；layer/expert 白名单非必需。
- 可复现的非重叠 calibration token offset/end 选择与 provenance。
- 多 calibration intervals 的 majority/strict consensus risk-floor 构建与投票元数据。
- 可审计的跨语料 train calibration：显式 dataset/config/split/text-field、离线 Arrow
  shard 顺序、逐文件 SHA256、确定性连续 token stream 与 offset/end provenance。
- Tail channel coverage 与 expert risk-floor source 可显式解耦，分别记录路径、SHA256、
  dataset/config/split 和 token provenance，用于跨语料机制分解。
- Conditional Dual/Dynamic-Regret teacher collector 支持同一套可审计多数据源协议，且
  `--parent-mode` 必须显式提供，避免 objective 被默认值静默替换。
- Tail-Risk profile builder 支持多折 teacher block utility、RMS/tail coverage consensus
  与 lower-confidence-bound（mean−κ·std）候选；所有 consensus/LCB provenance 均写入 profile，
  并由 train-only cross-fitted selector 决定是否启用。
- Tail-Risk profile builder 强制 teacher `parent_mode=dual`，非 dual teacher 直接拒绝。
- profile 冻结、内容 SHA/file SHA 与随机/route-count-matched 机制对照。
- train-only cross-fitted parent/refinement 选择：校准区间与各选择折严格不重叠，按折比较
  Route×Tail fallback 与 refinement，只有 refinement 严格多数折获胜且均值 PPL 更低时启用；
  平票或不稳定结果自动回退 Route×Tail，选择过程不读取 validation/test 指标。
- selector 重新计算 candidate profile 文件 SHA256，并逐折核对 result row 的绝对 profile
  path/SHA256，防止 manifest 写错、profile 被替换或结果目录串线。
- train-only 选择 token cache 冻结：同时记录 selection/evaluation 冻结标记、连续 token
  offset/end、数据来源 provenance，并复用正式 frozen-corpus PPL 校验协议。
