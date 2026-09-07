# HARP rank 核心问题

只回答三件事：有没有稳定 head、特殊对象是不是由下层少数对象驱动、HARP 该怎么用。
2σ / z-score 只作辅助。Channel 分段只在前 25% rank 里找 head，DeepSeek 的 tail split 不计。
消融按原始公式重算上一级 SP：`S=log(N ||Θ||_2^2 / ||Θ||_1^2)`，不是剩余分数的均值。
扰动：每个序列 32 次、相对噪声 0.5%；稳定 = 分界位移 ≤ 2 的比例 ≥ 75%。

## 1. 三层 rank 的核心分段

### Qwen3

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
|---|---|---:|---:|---|
| Layer | 有弱分段但不稳定 | 3 | d=2.737 | 是（within_two=1.000） |
| Expert | 有稳定 head | 7 （29/48 层有 head） | d=3.005 | 是 （mean within_two=0.978） |
| Channel | 有弱分段但不稳定 | 39 （2926/6144 expert 有 head） | d=2.983；d@10%=2.284 | 是（pooled within_two=1.000） |

### Gemma4

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
|---|---|---:|---:|---|
| Layer | 有弱分段但不稳定 | 2 | d=2.620 | 是（within_two=1.000） |
| Expert | 有稳定 head | 7 （18/30 层有 head） | d=2.895 | 是 （mean within_two=0.954） |
| Channel | 有稳定 head | 36 （2196/3840 expert 有 head） | d=3.062；d@10%=2.342 | 是（pooled within_two=1.000） |

### Qwen3.6

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
|---|---|---:|---:|---|
| Layer | 有弱分段但不稳定 | 2 | d=3.286 | 是（within_two=1.000） |
| Expert | 有稳定 head | 13 （25/40 层有 head） | d=2.775 | 是 （mean within_two=0.913） |
| Channel | 有弱分段但不稳定 | 26 （4515/10240 expert 有 head） | d=2.752；d@10%=1.106 | 是（pooled within_two=1.000） |

### DeepSeek

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
|---|---|---:|---:|---|
| Layer | 有稳定 head | 4 | d=2.273 | 是（within_two=1.000） |
| Expert | 有弱分段但不稳定 | 4 （16/26 层有 head） | d=2.562 | 是 （mean within_two=0.820） |
| Channel | 有弱分段但不稳定 | 71 （634/1664 expert 有 head） | d=2.626；d@10%=2.143 | 是（pooled within_two=1.000） |

### OLMoE

| 层级 | 是否有自然分段 | 分界 rank | 分界前后差异 | 扰动后是否稳定 |
|---|---|---:|---:|---|
| Layer | 有弱分段但不稳定 | 4 | d=1.684 | 是（within_two=1.000） |
| Expert | 有弱分段但不稳定 | 5 （10/16 层有 head） | d=2.549 | 是 （mean within_two=0.822） |
| Channel | 有稳定 head | 52 （626/1024 expert 有 head） | d=3.331；d@10%=0.053 | 是（pooled within_two=1.000） |

## 2. 特殊对象

定义：`候选特殊对象 = rank head（top 10%）+ 明显 gap/change point`。没有 gap 时表中仍列出 top rank，但不把它当成可保护的特殊对象。

| Model | Special layers | Special experts | Special channels |
|---|---|---|---|
| Qwen3 | L0, L47, L37, L41, L25（弱分段 k=3；Layer 0 outlier；top10%=L0, L47, L37, L41, L25） | L0 top10% n=13（有额外 head gap，稳定，k=10）；L47 top10% n=13（有额外 head gap，稳定，k=7）；L37 top10% n=13（无额外 head gap，稳定，k=7）；L41 top10% n=13（有额外 head gap，稳定，k=7）；L25 top10% n=13（有额外 head gap，稳定，k=7） | 每个 expert 的 top 10% prefix；head gap 比例 2926/6144；top10% 平均 participation mass=0.110 |
| Gemma4 | L0, L17, L16（仅端点尖峰；Layer 0 outlier；top10%=L0, L17, L16） | L0 top10% n=13（有额外 head gap，稳定，k=19）；L17 top10% n=13（有额外 head gap，稳定，k=7）；L16 top10% n=13（无额外 head gap，稳定，k=7） | 每个 expert 的 top 10% prefix；head gap 比例 2196/3840；top10% 平均 participation mass=0.113 |
| Qwen3.6 | L0, L39, L33, L32（仅端点尖峰；Layer 0 outlier；top10%=L0, L39, L33, L32） | L0 top10% n=26（有额外 head gap，稳定，k=13）；L39 top10% n=26（有额外 head gap，稳定，k=13）；L33 top10% n=26（有额外 head gap，稳定，k=13）；L32 top10% n=26（有额外 head gap，稳定，k=13） | 每个 expert 的 top 10% prefix；head gap 比例 4515/10240；top10% 平均 participation mass=0.109 |
| DeepSeek | L11, L10, L8（有明显 gap；top10%=L11, L10, L8） | L11 top10% n=7（有额外 head gap，稳定，k=5）；L10 top10% n=7（有额外 head gap，不稳定，k=4）；L8 top10% n=7（无额外 head gap，稳定，k=4） | 每个 expert 的 top 10% prefix；head gap 比例 634/1664；top10% 平均 participation mass=0.105 |
| OLMoE | L8, L9（弱分段 k=4；top10%=L8, L9） | L8 top10% n=7（有额外 head gap，不稳定，k=7）；L9 top10% n=7（有额外 head gap，稳定，k=5） | 每个 expert 的 top 10% prefix；head gap 比例 626/1024；top10% 平均 participation mass=0.114 |

## 3. 层级归因（重算 SP）

### 3.1 Channel → Expert

对每个候选特殊 layer 的 top 10% expert：去掉该 expert 的 top 1%/5%/10% channel 后，用剩余 channel 的 L1/L2 重算 Expert-SP。
逐 expert 明细在 `core_questions.json`；这里只保留计数和每层最高 Expert-SP 的一行。

| Model | n | channel-driven | mixed | distributed-channel |
|---|---:|---:|---:|---:|
| Qwen3 | 65 | 36 | 23 | 6 |
| Gemma4 | 39 | 20 | 11 | 8 |
| Qwen3.6 | 104 | 33 | 39 | 32 |
| DeepSeek | 21 | 17 | 4 | 0 |
| OLMoE | 14 | 14 | 0 | 0 |

#### Qwen3

| Expert | 原 Expert-SP | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0E35（层内最高） | 1.0468 | 1.0386 (r1) | 1.0210 (r3) | 0.9996 (r4) | distributed-channel |
| L47E2（层内最高） | 0.5348 | 0.5204 (r2) | 0.4935 (r18) | 0.4834 (r38) | channel-driven |
| L37E62（层内最高） | 0.5513 | 0.5476 (r1) | 0.5417 (r1) | 0.5318 (r3) | mixed |
| L41E42（层内最高） | 0.5654 | 0.5596 (r1) | 0.5439 (r1) | 0.5310 (r3) | channel-driven |
| L25E61（层内最高） | 0.5335 | 0.5256 (r1) | 0.5051 (r5) | 0.4950 (r10) | channel-driven |

汇总：channel-driven

#### Gemma4

| Expert | 原 Expert-SP | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0E115（层内最高） | 0.6644 | 0.6615 (r1) | 0.6553 (r1) | 0.6495 (r2) | distributed-channel |
| L17E106（层内最高） | 0.6190 | 0.6124 (r1) | 0.5950 (r1) | 0.5829 (r2) | channel-driven |
| L16E93（层内最高） | 0.5840 | 0.5796 (r2) | 0.5661 (r3) | 0.5515 (r13) | channel-driven |

汇总：channel-driven

#### Qwen3.6

| Expert | 原 Expert-SP | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0E25（层内最高） | 2.2084 | 2.2518 (r1) | 2.4671 (r1) | 2.9001 (r1) | distributed-channel |
| L39E200（层内最高） | 0.7769 | 0.7729 (r1) | 0.7780 (r1) | 0.7919 (r1) | distributed-channel |
| L33E75（层内最高） | 0.6058 | 0.6027 (r1) | 0.5954 (r1) | 0.5893 (r1) | mixed |
| L32E123（层内最高） | 0.5913 | 0.5877 (r1) | 0.5838 (r1) | 0.5794 (r1) | mixed |

汇总：mixed

#### DeepSeek

| Expert | 原 Expert-SP | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
|---|---:|---:|---:|---:|---|
| L11E49（层内最高） | 0.4864 | 0.4850 (r1) | 0.4795 (r1) | 0.4767 (r1) | channel-driven |
| L10E33（层内最高） | 0.4714 | 0.4706 (r1) | 0.4694 (r4) | 0.4684 (r5) | mixed |
| L8E10（层内最高） | 0.4733 | 0.4710 (r2) | 0.4678 (r10) | 0.4652 (r18) | channel-driven |

汇总：channel-driven

#### OLMoE

| Expert | 原 Expert-SP | 去 top 1% channel | 去 top 5% | 去 top 10% | 结论 |
|---|---:|---:|---:|---:|---|
| L8E10（层内最高） | 0.5009 | 0.4925 (r7) | 0.4784 (r34) | 0.4731 (r44) | channel-driven |
| L9E21（层内最高） | 0.5113 | 0.5029 (r3) | 0.4843 (r12) | 0.4761 (r31) | channel-driven |

汇总：channel-driven

### 3.2 Expert → Layer

对每个候选特殊 layer：去掉 top-1 / top 5% / top 10% expert 后，用剩余 expert 权重的 L1/L2 重算 Layer-SP。

#### Qwen3

| Layer | 原 Layer-SP | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0 | 0.5544 | 0.5513 | 0.5339 | 0.5206 | distributed |
| L47 | 0.4794 | 0.4789 | 0.4770 | 0.4758 | not-outlier |
| L37 | 0.4780 | 0.4775 | 0.4754 | 0.4740 | not-outlier |
| L41 | 0.4775 | 0.4769 | 0.4748 | 0.4736 | not-outlier |
| L25 | 0.4754 | 0.4750 | 0.4733 | 0.4723 | not-outlier |

汇总：distributed

#### Gemma4

| Layer | 原 Layer-SP | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0 | 0.5524 | 0.5515 | 0.5471 | 0.5432 | distributed |
| L17 | 0.5381 | 0.5375 | 0.5357 | 0.5342 | not-outlier |
| L16 | 0.5300 | 0.5295 | 0.5276 | 0.5261 | not-outlier |

汇总：distributed

#### Qwen3.6

| Layer | 原 Layer-SP | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
|---|---:|---:|---:|---:|---|
| L0 | 1.0031 | 1.0004 | 0.9780 | 0.9583 | distributed |
| L39 | 0.4846 | 0.4836 | 0.4801 | 0.4778 | not-outlier |
| L33 | 0.4816 | 0.4811 | 0.4784 | 0.4766 | not-outlier |
| L32 | 0.4807 | 0.4803 | 0.4781 | 0.4765 | not-outlier |

汇总：distributed

#### DeepSeek

| Layer | 原 Layer-SP | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
|---|---:|---:|---:|---:|---|
| L11 | 0.4658 | 0.4655 | 0.4650 | 0.4647 | distributed |
| L10 | 0.4641 | 0.4640 | 0.4636 | 0.4632 | not-outlier |
| L8 | 0.4639 | 0.4637 | 0.4633 | 0.4629 | not-outlier |

汇总：distributed

#### OLMoE

| Layer | 原 Layer-SP | 去 top-1 expert | 去 top-5% | 去 top-10% | 结论 |
|---|---:|---:|---:|---:|---|
| L8 | 0.4790 | 0.4786 | 0.4777 | 0.4768 | not-outlier |
| L9 | 0.4772 | 0.4767 | 0.4753 | 0.4740 | not-outlier |

汇总：distributed

## 核心结论

| Model | Layer rank | Expert rank | Channel rank | Channel→Expert | Expert→Layer | HARP 启发 |
|---|---|---|---|---|---|---|
| Qwen3 | 有弱分段但不稳定 | 有稳定 head | 有弱分段但不稳定 | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| Gemma4 | 有弱分段但不稳定 | 有稳定 head | 有稳定 head | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| Qwen3.6 | 有弱分段但不稳定 | 有稳定 head | 有弱分段但不稳定 | mixed | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| DeepSeek | 有稳定 head | 有弱分段但不稳定 | 有弱分段但不稳定 | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |
| OLMoE | 有弱分段但不稳定 | 有弱分段但不稳定 | 有稳定 head | channel-driven | distributed | Layer quantile；Expert 按 rank 分档；Channel 用 top prefix |

### 问题一：rank 中有没有自然分段？

见上表 Layer / Expert / Channel 三列。结论只使用：有稳定 head / 有弱分段但不稳定 / 没有自然分段，只能使用 quantile。

### 问题二：特殊对象是不是由更底层对象造成？

见 Channel→Expert 与 Expert→Layer。Layer 消融只对 2σ 异常层计 majority；其余 top-10% 层标为 `not-outlier`。
真正的 Layer 异常（Qwen3/Gemma/Qwen3.6 的 L0，DeepSeek 的 L11）在去掉 top 10% expert 后仍然高，属于 distributed layer。
Channel→Expert 在后段层更容易出现 channel-driven；L0 最高 expert 往往仍是 distributed-channel（Qwen3.6 L0 去掉 top channel 后 Expert-SP 甚至上升）。

### 问题三：HARP 应该怎么利用？

没有一层异常是由少数 expert 撑起来的，因此不把 Layer rank head 做成额外保护预算。
Expert 层内有更稳定的 head，Channel 有可用的 top prefix，所以：

```text
Layer：quantile 分配预算
Expert：按 rank 分档
Channel：使用 top prefix
```

---

# 附录：第一轮完整排名分析

Generated from existing `harp_rankings.pt` caches. Scores were not recomputed. The HARP allocator was not changed.

- Design: `HARP/experiment_design.md`
- Code: `HARP/rank_analysis_core.py`, `HARP/run_rank_analysis.py`
- Tests: `HARP/tests/test_rank_analysis.py` (9 passed)
- Artifacts: `/data/xinpeigao/evalscope_results/_artifacts/harp_rank_analysis/<model>/`

`use segment` would mean a stable Layer-SP head interval is ready to become a HARP budget cut. This run does **not** reach that bar on any of the five models. Rank order is still informative, so the existing quantile-prefix policy remains the structural recommendation (design step 7: do not retier HARP from these segments).

## Practical deviations from the design

- Full channel tables are packed per layer in `channel_rank.pt` (scores, ranks, z, robust-z, participation mass, parent ranks). Expanding every `(layer, expert, channel)` to a dict row would be tens of millions of Python objects.
- Per-expert `channel_rank_curves/<layer>_<expert>.csv` is not written. Layer-mean and pooled-mean curves are in `channel_rank_curves/`; per-expert change points live in `channel_segment_summary.csv` and `channel_segments.json`.
- Participation mass is `expm1(clip(Channel-SP)).clamp_min(0)`, then normalized within the expert.
- One mean-shift change point is used. A boundary is “natural” only if the gap is at least the within-sequence p90, the split is not `k<=2`, and the mean-shift statistic is `>1`.
- Channel change points on very long vectors (DeepSeek C=1408) can land near the far end of the sequence; those are tail splits, not special-channel heads. The Layer/Expert conclusions do not depend on treating those as heads.

## Per-model conclusions

```text
Model: qwen3
Layer segments: ranks 1-3 vs 4-48; natural=False; stability_within_two=0.917; driver=distributed-shift
Expert segments: 28/48 layers have a natural head boundary
Channel segments: 2613/6144 experts have a natural head boundary; median change-rank=81/768; mean Cohen d@10%=2.10
Channel -> Expert: spearman_top10_mass=0.809; null_excess=0.806; partial_spearman_ctrl_layer=0.706
Expert -> Layer: pearson_mean=0.999; null_excess=1.010; LOO does not collapse L0
Decision: use quantile only
```

- Strong special-layer set: `{0, 37, 41, 47}`. Ordinary 2σ is only L0; robust-z also flags late layers.
- Layer prefix Jaccard at 0.5% score noise: top5=0.625, top10=0.628, top25=0.640.
- Driver counts: distributed-shift 47, mixed 1 (L0), outlier-driven 0.

```text
Model: gemma4
Layer segments: ranks 1-2 vs 3-30; natural=False (endpoint-only); stability_within_two=0.667; driver=distributed-shift
Expert segments: 18/30 layers have a natural head boundary
Channel segments: 1887/3840 experts have a natural head boundary; median change-rank=36/704; mean Cohen d@10%=2.21
Channel -> Expert: spearman_top10_mass=0.529; null_excess=0.528
Expert -> Layer: pearson_mean=0.999; null_excess=0.985
Decision: use quantile only
```

- Strong special-layer set: `{0, 17}`. Ordinary 2σ is only L0.
- Layer prefix Jaccard at 0.5% noise: top5=1.0, top10=0.688, top25=0.622.
- Driver counts: distributed-shift 28, mixed 2, outlier-driven 0.

```text
Model: qwen36
Layer segments: ranks 1-2 vs 3-40; natural=False (endpoint-only L0 spike); stability_within_two=1.0; driver=distributed-shift
Expert segments: 26/40 layers have a natural head boundary
Channel segments: 4434/10240 experts have a natural head boundary; median change-rank=26/512; mean Cohen d@10%=2.07
Channel -> Expert: spearman_top10_mass=0.863; null_excess=0.863
Expert -> Layer: pearson_mean=1.000; null_excess=0.995
Decision: use quantile only
```

- L0 score ≈ 1.003 vs bulk ≈ 0.48 (ordinary z=6.23, robust z≈112). The gap after rank 1 is an endpoint spike, not a multi-layer head segment.
- Strong special-layer set: `{0, 32, 33, 39}`. Ordinary 2σ is only L0.
- Layer prefix Jaccard at 0.5% noise: top5=0.875, top10=0.800, top25=0.749.
- Driver counts: distributed-shift 39, mixed 1 (L0), outlier-driven 0.

```text
Model: deepseek
Layer segments: ranks 1-4 vs 5-26 (ids 11,10,8,9); natural=True; stability_within_two=0.729; driver=distributed-shift
Expert segments: 16/26 layers have a natural head boundary
Channel segments: 528/1664 experts flagged natural; median change-rank=1337/1408 (mostly tail splits, not heads)
Channel -> Expert: spearman_top10_mass=0.635; null_excess=0.634
Expert -> Layer: pearson_mean=0.972; null_excess=0.973
Decision: use quantile only
```

- Mid-depth peak, not first MoE layer. Ordinary 2σ is only id 11; robust-z also flags 10, 8, 9.
- Strong special-layer set: `{8, 10, 11}`.
- Layer prefix Jaccard at 0.5% noise is low (top5=0.50, top10=0.45). The change-point gap clears p90, but perturbation moves the cut by a median of several ranks, so it is not a HARP budget boundary.
- Driver counts: distributed-shift 23, mixed 3, outlier-driven 0.

```text
Model: olmoe
Layer segments: ranks 1-9 vs 10-16 (peak ids 8,9); natural=False; stability_within_two=0.542; driver=distributed-shift
Expert segments: 10/16 layers have a natural head boundary
Channel segments: 604/1024 experts have a natural head boundary; median change-rank=52/1024; mean Cohen d@10%=2.22
Channel -> Expert: spearman_top10_mass=0.919; null_excess=0.919
Expert -> Layer: pearson_mean=0.999; null_excess=1.008
Decision: use quantile only
```

- No ordinary or robust 2σ layer. Strong set from the rank prefix only: `{8, 9}`.
- Layer prefix Jaccard at 0.5% noise: top5=0.688, top10=0.677, top25=0.729.
- Driver counts: distributed-shift 15, mixed 1, outlier-driven 0.

## Answers to the design questions

### Natural segmentation

| Level | Finding |
|---|---|
| Layer-SP | Order is real and model-specific (shallow L0 outlier vs mid-depth peak). A **stable multi-layer head/bulk/tail cut** is not supported: Qwen3/Gemma/Qwen3.6 cuts are endpoint-dominated; DeepSeek’s p90 gap is unstable under 0.5–1% noise; OLMoE’s split is neither a gap-tail nor stable. |
| Expert-SP | About half of layers show a within-layer head boundary that is not a single expert. Head length varies by layer; high Layer-SP layers are a **higher whole-layer level**, not a uniquely longer expert head. |
| Channel-SP | Top-10% Cohen’s d is ~2.1 across families, so a ranked channel prefix is a real concentration, matching existing Channel-SP keep practice. Head length is expert-dependent. DeepSeek’s one-change detector often splits near the tail because C is large; do not read that median as a special-channel width. |

Do not declare a Layer-SP segment from the Layer-0 2σ flag alone. That flag fires on Qwen3, Gemma4, Qwen3.6 L0 and DeepSeek id 11, and never on OLMoE.

### Special objects

A **strong** special layer requires rank-head membership plus outlier agreement that is not only ordinary mean/std:

- Qwen3 / Gemma4 / Qwen3.6: L0 is the only ordinary-2σ layer; robust-z also marks a few late or mid layers. L0 is special as a rank-1 spike, not as a HARP tier boundary.
- DeepSeek: id 11 plus neighbors 8–10.
- OLMoE: rank prefix `{8,9}` without z-outliers.

Special experts: within-layer top 5/10/25% prefixes plus change-point heads in `expert_segment_summary.csv`. Strong experts also need nontrivial channel concentration (`channel_top10_mass`); mean top-10 mass is ~0.11, so the prefix carries more mass than uniform 10% but is not a tiny spike.

Special channels: z>2 counts per expert are in `channel_segment_summary.csv`. A high z without participation mass is not treated as sufficient.

### Hierarchy (channel → expert → layer)

- **Expert-SP ≈ mean of Expert-SP inside the layer ≈ Layer-SP.** Pearson of Layer-SP vs expert mean is 0.972–1.000, and shuffling expert-means across layers destroys it (null excess ≈ 1). Layer-SP is a distributed expert-level shift, not an unrelated scalar.
- **Leave-one-top-expert-out** does not remove the L0 / id-11 abnormality: the rest of the experts in that layer stay high. Zero layers are classified `outlier-driven`. Majority label is `distributed-shift`.
- **Channel concentration is associated with Expert-SP**, but more weakly and not via top-1 z. Spearman of Expert-SP vs channel top-10 participation mass is 0.53–0.92, all far above the permutation null. Partial Spearman after residualizing Layer-SP remains positive on Qwen3 (0.71). Pearson vs channel top-1 z is near zero or negative: special experts are not “one huge channel”.
- Hierarchy is therefore: **broad expert-level shift → Layer-SP**; **channel-mass concentration → Expert-SP rank**, not a few channel outliers causing a special layer.

## Decision rule applied

`use segment` requires a natural Layer-SP boundary, perturbation keeping the cut within two ranks at least 75% of the time, **and** Expert-mean ↔ Layer-SP null excess > 0.2.

All five models pass the hierarchy excess. None pass a stable natural Layer-SP segment. Hence **Decision: use quantile only** for every model. HARP width allocation should keep rank operators (quantile prefixes / two-tier experts), not a change-point budget cut.

## Required artifact files

Each model directory contains `manifest.json` (cache SHA256 + provenance), `layer_rank.pt` / `layer_rank.json`, `expert_rank.pt`, `channel_rank.pt`, `layer_rank_curve.csv`, `layer_gap_curve.csv`, `layer_rank_curve.png`, `expert_rank_curves/<layer_id>.csv`, `expert_segment_summary.csv`, `channel_segment_summary.csv`, `channel_segments.json`, `segment_analysis.json`, `hierarchy_links.json`, `null_controls.json`, `conclusion.txt`.

## Cache manifests

| Model | Cache SHA256 prefix | n MoE layers | n experts | Checkpoint family |
|---|---|---|---|---|
| qwen3 | `df1a6b604676043f` | 48 | 6144 | Qwen3-30B-A3B-Instruct-2507 |
| gemma4 | `13bcd7963bf2306b` | 30 | 3840 | gemma-4-26B-A4B-it |
| qwen36 | `0046aacc1589cbc4` | 40 | 10240 | Qwen3.6-35B-A3B |
| deepseek | `9aad20fb81283566` | 26 (ids 1–26) | 1664 | DeepSeek-V2-Lite-Chat |
| olmoe | `457f5aa8a1816b2b` | 16 | 1024 | OLMoE-1B-7B-0125-Instruct |
