# HARP rank-structure analysis

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
