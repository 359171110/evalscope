基于我们之前讨论的所有结论，下面给出**完整的端到端算法描述**。该算法将全局剪枝率 \(P\) 转化为最终的稀疏网络结构，完整覆盖了 Layer-SP 分配、Layer 0 自适应保护、逐层冗余传递，以及层内“自下而上”的专家升级。

本目录现在包含 HARP 的独立实现：`harp_core.py` 提供分配原语，`build_harp_artifacts.py`
负责构建结构 profile，`export_harp_checkpoint.py` 复用 CSP 的经过验证的 packed/separate
导出器。原有 CSP、HSP-Hetero 和 AHSP 入口不受影响。

## 当前支持的模型

HARP 复用 `CSP/model_adapter.py`，当前支持：

| 模型 | 布局 | alignment | 三档宽度（基准宽度为 `k0`） |
|---|---|---:|---:|
| Qwen3-30B-A3B | separate gate/up/down | 64 | `k0-64 / k0 / k0+64` |
| Qwen3.6-35B-A3B | packed gate_up/down | 64 | `k0-64 / k0 / k0+64` |
| Gemma4-26B-A4B | packed gate_up/down | 32 | `k0-64 / k0 / k0+64` |
| DeepSeek-V2-Lite | separate gate/up/down | 32 | `k0-64 / k0 / k0+64` |
| OLMoE-1B-7B | separate `w1/w3/w2` | 64 | `k0-64 / k0 / k0+64` |
| Mixtral-8x7B | separate `w1/w3/w2` | 64 | `k0-64 / k0 / k0+64` |

Qwen3.6 和 Gemma4 的 packed gate/up 两个物理半区会分别 padding，再重新拼接，保持
HF/vLLM 的固定边界。

## 实现入口

```text
HARP/harp_core.py                 Layer-SP anchor、layer/expert allocation
HARP/build_harp_artifacts.py      Layer-SP/Expert-SP/Channel-SP profile builder
HARP/export_harp_checkpoint.py    Shared CSP exporter wrapper
HARP/tests/test_harp_core.py      Allocation unit tests
```

构建 profile：

```bash
PYTHONPATH=/path/to/evalscope:/path/to/evalscope/static_moe_prunning/code \
python -m HARP.build_harp_artifacts \
  --model-path /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --output-channel-cache /path/to/harp/qwen3_channel.pt \
  --output-profile /path/to/harp/qwen3_profile.pt \
  --low-width 320 --budget-width 384 --high-width 448
```

导出 checkpoint：

```bash
PYTHONPATH=/path/to/evalscope:/path/to/evalscope/static_moe_prunning/code \
python -m HARP.export_harp_checkpoint \
  --model-path /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --profile /path/to/harp/qwen3_profile.pt \
  --channel-cache /path/to/harp/qwen3_channel.pt \
  --output-dir /path/to/harp/qwen3_checkpoint
```

---

### 算法全貌：Hierarchical MoE Pruning via SP-driven Allocation

---

#### 符号定义
- 全局剪枝率：\(P\)（例如 0.5，表示保留 50% 参数）
- 模型总参数量（仅 MoE Routed Experts）：\(\text{Params}_{total}\)
- MoE 层数：\(L\)，第 \(l\) 层专家数：\(E_l\)（通常为常数 \(E\)）
- 三档硬件友好宽度：\(\mathcal{K} = \{K_{low}, K_{mid}, K_{high}\}\)，且 \(K_{low} < K_{mid} < K_{high}\)
- Layer-SP：\(S^L_l\)，Expert-SP：\(S^E_{l,e}\)，Channel-SP：\(S^C_{l,e,c}\)

---

### Phase 1：全局预算与层预分配

**Step 1.1：计算全局目标容量**
\[
B_{total} = P \times \text{Params}_{total}
\]

**Step 1.2：自适应识别“关键锚定层”（Critical Anchor Layer）**
根据统计观察，首个 MoE 层（Layer 0）常为极端 Outlier，但为防止硬编码，采用自适应检测：

- 计算所有层 Layer-SP 的均值 \(\mu_L\) 与标准差 \(\sigma_L\)。
- 若 Layer 0 满足 \(S^L_0 > \mu_L + 2 \times \sigma_L\)，则标记为 **锚定层（Anchor）**。
- 否则，视作普通层（如 Gemma4 或后续模型）。

**Step 1.3：分层预分配预算（含锚定保护）**

- **Case A：存在锚定层（通常为 Layer 0）**
  - 为该层设定保护性平均宽度 \(K_{anchor}\)，通常取 \(K_{anchor} = \text{round}(0.6 \times (K_{high} - K_{low}) + K_{low})\)（即略偏向中高档），或直接设定其剪枝率 \(P_{anchor} = P - \delta\)（如 \(\delta = 0.1\)）。
  - 锚定层初始预算：\(B_{anchor}^{init} = E \times K_{anchor}\)（若换算成参数量则乘以维度，这里简化为神经元数）。
  - 剩余层可分配总预算：\(B_{remain} = B_{total} - B_{anchor}^{init}\)。

- **Case B：不存在锚定层**
  - 所有层平等对待，\(B_{remain} = B_{total}\)。

- **非锚定层的按比例分配**：
  对于所有非锚定层，按归一化的 Layer-SP 分配：
  \[
  B_l^{init} = B_{remain} \times \frac{S^L_l}{\sum_{j \notin Anchor} S^L_j}
  \]
  （注：此处 \(B_l^{init}\) 指该层可保留的神经元总数，需保证 \(B_l^{init} \ge E \times K_{low}\)，否则直接截断至 \(E \times K_{low}\) 并记录警告）。

---

### Phase 2：逐层降序处理与预算传递（Layer-level Scheduling）

这一步解决“离散化反向影响”问题。将层按 **Layer-SP 从高到低排序**（锚定层自动排最前）。

**初始化**：冗余累积器 \(R = 0\)。

**对于排序后的每一层 \(l\)（从高到低）**：

1. 该层可用预算：\(B_l^{avail} = B_l^{init} + R\)。
2. 调用 **Phase 3（层内专家分配）**，输入 \(B_l^{avail}\)，输出该层实际消耗容量 \(B_l^{actual}\)。
3. 计算该层产生的冗余：\(\Delta_l = B_l^{avail} - B_l^{actual}\)（因为 Phase 3 保证绝不超支，所以 \(\Delta_l \ge 0\)）。
4. 将冗余传递给下一层：\(R = \Delta_l\)。

**循环结束后**，若最后一层仍有冗余 \(R > 0\)，说明全局容量未能完全用满。由于误差通常小于 0.5%，可直接丢弃，或在所有层中按比例微调。

---

### Phase 3：层内专家容量分配（Low-first Upgrade Algorithm）

这是本算法的核心，输入为该层可用预算 \(B_l^{avail}\) 和该层所有专家的 Expert-SP。

**Step 3.1：初始化（全部设为最低档）**
- 所有 \(E\) 个专家初始宽度均为 \(K_{low}\)。
- 基础成本：\(C_{base} = E \times K_{low}\)。
- 若 \(B_l^{avail} < C_{base}\)（极端情况），强制所有专家为 \(K_{low}\)，\(B_l^{actual} = C_{base}\)，\(\Delta_l = B_l^{avail} - C_{base}\)（负值，但剪枝率超了，需在全局日志中警告，实际中应通过 Step 1.3 避免）。

**Step 3.2：按 Expert-SP 排序**
将 \(E\) 个专家按 Expert-SP **从高到低**降序排列，记为 \(e_1, e_2, ..., e_E\)。

**Step 3.3：离散枚举最优升级组合**
我们只需决策有多少专家升到 **Mid**（\(n_{mid}\)）和 **High**（\(n_{high}\)），其余为 Low。由于排序固定，最终分布必然是前 \(n_{high}\) 个为 High，接着 \(n_{mid}\) 个为 Mid，其余为 Low。

搜索空间：\(0 \le n_{high} \le E\)，\(0 \le n_{mid} \le E - n_{high}\)，复杂度 \(O(E^2)\)。

**搜索目标（双优先级）**：
1. **硬约束**：\(n_{high}K_{high} + n_{mid}K_{mid} + (E - n_{high} - n_{mid})K_{low} \le B_l^{avail}\)。
2. **主目标**：最大化“加权升级收益”：
   \[
   \text{Score} = n_{high} \times (K_{high} - K_{low}) + n_{mid} \times (K_{mid} - K_{low})
   \]
   这个目标确保算法优先填满 High 档，再填 Mid 档，完美契合理性。
3. **辅目标（硬件均衡）**：若多个组合 Score 相同（或差距极小），选择三档专家数量方差最小的组合，防止某个 Kernel 完全闲置（例如新增条件：\(\min \text{Var}(n_{high}, n_{mid}, n_{low})\)）。

**伪代码实现**：
```python
best_h, best_m = 0, 0
best_score = -1
best_variance = float('inf')

for n_h in range(E, -1, -1):          # 优先枚举高位数，以便早停
    for n_m in range(E - n_h, -1, -1):
        n_l = E - n_h - n_m
        cost = n_h * K_high + n_m * K_mid + n_l * K_low
        
        if cost <= B_l_avail:
            score = n_h * (K_high - K_low) + n_m * (K_mid - K_low)
            var = (n_h - E/3)**2 + (n_m - E/3)**2 + (n_l - E/3)**2
            
            if (score > best_score) or (score == best_score and var < best_variance):
                best_score = score
                best_variance = var
                best_h, best_m = n_h, n_m
```

**Step 3.4：映射与输出**
- 前 \(best\_h\) 个专家宽度设为 \(K_{high}\)。
- 接下来 \(best\_m\) 个专家宽度设为 \(K_{mid}\)。
- 剩余专家宽度设为 \(K_{low}\)。
- 计算实际消耗：\(B_l^{actual} = best\_h \times K_{high} + best\_m \times K_{mid} + (E - best\_h - best\_m) \times K_{low}\)。
- 返回 \(B_l^{actual}\) 及每个专家的目标宽度。

---

### Phase 4：Channel-SP 微观筛选（Fine-grained）

这是最后一步，对每个专家内部做精确裁剪。

对于第 \(l\) 层的第 \(e\) 个专家（其目标宽度为 \(K_e\)）：

1. 获取该专家内部的 Channel-SP 分数 \(S^C_{e,c}\)（基于权重 \(\ell_1/\ell_2\) 的变异系数推导）。
2. 将所有通道（输出神经元）按 \(S^C_{e,c}\) **从高到低**排序。
3. 保留 Top-\(K_e\) 个通道及其对应的权重，其余永久移除。

---

### Phase 5：最终验证与微调（Optional）

- 计算全局实际总容量 \(B_{total}^{actual} = \sum_l B_l^{actual}\)。
- 验证实际剪枝率 \(P_{actual} = 1 - B_{total}^{actual} / \text{Params}_{total}\) 与目标 \(P\) 的误差。
- 若误差 > 1%（通常不会），则在所有层中按 Layer-SP 倒数（低 SP 层多承担）微调 \(B_l^{init}\)，重复 Phase 2-4 一次。

---

### 总结：本算法如何处理 Layer 0 及一般情况

| 场景 | 处理逻辑 |
| :--- | :--- |
| **Qwen3.6（Layer 0 为极端 Outlier）** | Phase 1 自适应检测到锚点，赋予更大的初始预算 \(B_{anchor}^{init}\)。Phase 2 中 Layer 0 优先处理，Phase 3 中因预算充裕，会让大部分专家升级到 Mid/High（比如 60/50/18），充分保护。 |
| **Gemma4（Layer 0 非极端）** | Phase 1 不触发锚定保护，Layer 0 仅按普通高 SP 层分配略多预算。Phase 2 排序中它可能仍排第一，但因预算差距不大，Phase 3 只会让它略多于其他层升级（比如 45/45/38），不会过度倾斜。 |
| **不存在特殊层（通用情况）** | 完全退化为纯 Layer-SP 驱动的比例分配 + 逐层冗余传递，鲁棒性极强。 |

整个算法具有 **数学可解释性**（变异系数推导）、**硬件感知性**（三档 Kernel）和 **自适应鲁棒性**（自动检测锚点），可以直接作为你们论文的核心方法论。👍