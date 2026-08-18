# PP 实验分析

## 冻结配置：PP-Frozen-v1

后续探索 AIMER、Random 之外的新基础排序方法时，统一使用以下 PP 配置，不再针对候选方法或评测结果调整 PP 超参数。

| 项目 | 冻结值 |
|---|---|
| PP 版本名 | `PP-Frozen-v1` |
| 模型 | `Qwen3-30B-A3B-Instruct-2507` |
| probe 来源 | RMSNorm router self probe + cosine top-K router neighbors |
| router neighbors | `K=8` |
| 每通道聚合 | absolute SwiGLU response 的 top-`Q=4` mean |
| probe signs | `positive`，不使用 PosNeg |
| `down_proj` 列范数 | `NoNorm`，禁止相乘 |
| PP 保护比例 | `G=10%`，按原始 768 通道计算 |
| 每个 expert 的保护数 | `round(768 * 0.10) = 77` 个通道 |
| block size | 64 |
| PP 排名作用域 | 每层、每 expert 独立排序 |
| PP cache | `PP/experiments/profiles/down_proj_norm_ablation_20260807/PurePseudo-K8-Q4-NoDownNorm/rankings.pt` |
| PP cache SHA256 | `0fd53f6eada24fe531382893597a1ff2137ed35594f183b96b0d039a497d21da` |

PP 分数固定为：

$$
s_{l,e,c}^{\mathrm{PP}}
=
\operatorname{mean}\!\left(
\operatorname{TopQ}_{Q=4}
\left\{
\left|\operatorname{SwiGLUResponse}_{l,e,c}(p)\right|
: p \in \mathcal{P}_{K=8}^{+}
\right\}
\right),
$$

其中不乘任何 `down_proj` 列范数，也不加入负 probe。每个 expert 先无条件放入 PP 排名前 77 个通道；候选基础方法随后按自己的排名顺序补齐目标宽度，并跳过已被 PP 保护的重复通道。候选方法不得重排、过滤或替换这 77 个 PP 通道。

### 新方法的固定比较协议

1. 主实现预算固定为 `B6of12` 和 `B9of12`，分别覆盖激进压缩与中等压缩场景：

| 预算 | 每 expert 保留通道 | 结构化剪枝率 | PP 固定通道 | PP 占保留容量 | 候选方法补齐 |
|---|---:|---:|---:|---:|---:|
| `B6of12` | 384 | 50% | 77 | 20.1% | 307 |
| `B9of12` | 576 | 25% | 77 | 13.4% | 499 |

2. `G=10%` 始终按原始 768 通道计算，因此两个预算都固定保护同一 PP 排名的前 77 个通道，不随最终保留容量缩放。
3. 每个候选方法只完成以下两个结果，不运行 G0 对照：
	- `<Method>-PPFv1-G10-B6of12`
	- `<Method>-PPFv1-G10-B9of12`

   两个结果分别测量候选方法与冻结 PP 在激进压缩和中等压缩下的组合表现。
4. 使用同一模型、导出流程、Quick9 数据子集、seed 42 和确定性生成配置；只允许改变候选基础排序。
5. 不因某个候选方法的结果调整 `K`、`Q`、`G`、probe signs、Norm、block size 或 PP cache。任何调整都必须作为新的 PP 版本和独立消融实验，不能并入 `PP-Frozen-v1`。
6. 方法命名统一为 `<Method>-PPFv1-G10-B{6,9}of12`。

该冻结配置的目的不是声称 G10 对所有基础方法和预算都全局最优，而是固定当前证据最充分的 PP 版本，并在 B6/B9 两个代表性压缩强度下检验稳定性，使后续增益能够归因于新的基础排序方法及其与 PP 的互补性。

## `down_proj` 列范数消融

PP 分数中乘 `down_proj` 列范数不是必要项，也没有稳定、可迁移的正向收益。默认采用 NoNorm 更合理。
