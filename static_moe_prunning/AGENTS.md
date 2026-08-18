# V4 静态专家剪枝执行规则

本项目只承载静态专家剪枝研究。不得从 `moe_prune_v3` import、软链接或在运行时读取文件；
需要沿用的内容必须复制为本目录自有文件。模型和公开数据缓存是外部实验输入，不受此限制。

## 必须遵守

1. 正式主 profile 只使用明确标注、可哈希审计的 train corpus 校准；所有 profile 必须在
   读取对应 evaluation split PPL 前冻结，严禁使用 validation/test 指标选择 profile 或调参。
2. WikiText 正式 PPL 必须是 114 windows、233,368 tokens、sequence length 2048；C4
   正式 PPL 必须使用冻结协议 `c4_validation_114x2048_v1`：114 windows、233,472 tokens、
   sequence length 2048、validation split，并校验 token-cache SHA256。
3. smoke PPL 只作协议和显存调试，不得用于方法选择。
4. 宽度必须按 `(layer, physical_expert)` 索引，禁止用 router rank 冒充结构剪枝。
5. 对比方法必须使用相同 64-channel block 粒度与相同总结构 block 预算。
6. 每个实验（成功、失败、负结果）都追加到 `docs/ALL_EXPERIMENTS_COMPLETE.md`，
   并在 `experiments/results/` 保存 JSON；运行 stdout/stderr 写入 `experiments/logs/`。
7. 每个新 idea 写入 `docs/ideas_log.md`，每个 bug 修复写入 `docs/debug_list.md`。
8. 行为改动先写失败测试再实现；完成前运行相关 pytest、py_compile 和协议验证器。

10. 不新增依赖；优先使用已验证的 `xhquant` Python 环境。
11. Conditional Dual teacher 必须显式记录 `parent_mode=dual`；collector 不允许隐式默认，
    Tail-Risk profile builder 必须拒绝非 dual teacher。

## 完成门禁

最终候选必须在完整 WikiText-2 PPL 上、同预算下严格优于所有已实现静态基线，
并明确区分 MoE-Slimming、MoSE、POP、REAP、MAESTRO、FLAP 和 MoE-Pruner。
