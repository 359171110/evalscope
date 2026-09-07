# HARP Rank Structure Experiment Design

## 1. Purpose

This experiment is designed to answer three narrow questions about the existing HARP ranking cache:

1. **Natural segmentation:** after ranking Layer-SP, Expert-SP, and Channel-SP, do the ranked values contain stable, natural groups rather than an arbitrary smooth ordering?
2. **Special objects:** can we identify genuinely special layers, experts, and channels from the rank itself?
3. **Hierarchical correlation:** are special channels concentrated in special experts, and are special experts concentrated in special layers?

This document only specifies the ranking analysis. It does not change the HARP allocator and does not use downstream test metrics to define the special objects.

## 2. Ranking inputs and fixed conventions

Use the same calibration-free, raw ranking cache already shared by HARP v1, v2, and `quantile_layer`:

```text
/data/xinpeigao/evalscope_results/_artifacts/harp/<model>/harp_rankings.pt
```

The cache contains:

- `layer_scores`: one Layer-SP score per MoE layer;
- `table[layer_id]["expert_structural_scores"]`: one Expert-SP score per expert in the layer;
- `table[layer_id]["channel_scores"]`: Channel-SP scores for every expert/channel;
- `table[layer_id]["ranked_indices"]`: descending Channel-SP permutation for every expert;
- `table[layer_id]["block_relative_scores"]` and `block_coverage_scores`: aligned channel-block summaries.

The rank definitions are fixed as follows:

- Layer rank: sort all MoE layers by Layer-SP descending;
- Expert rank: sort experts **within each layer** by Expert-SP descending;
- Channel rank: sort channels **within each expert** by Channel-SP descending;
- Ties: use the existing stable lower-index tie break;
- DeepSeek: keep its actual MoE layer ids `1..26`; do not renumber it as if layer 0 were MoE.

Every output must preserve both the original id and the rank. A rank without its parent layer/expert id is not sufficient for cross-level analysis.

## 3. Per-model rank files

For each model, create one directory under an analysis output root, for example:

```text
rank_analysis/<model>/
  layer_rank.pt
  expert_rank.pt
  channel_rank.pt
  segment_analysis.json
  hierarchy_links.json
```

The source checkpoint and ranking-cache SHA256 must be recorded in every file or in a shared manifest.

### 3.1 Layer rank file

One row per MoE layer:

```text
{
  "model": str,
  "layer_id": int,
  "rank": int,                 # 1 is highest Layer-SP
  "percentile": float,
  "score": float,
  "robust_z": float,
  "ordinary_z": float,
  "is_2sigma_candidate": bool,
  "is_robust_outlier_candidate": bool,
  "mean_expert_score": float,
  "expert_score_std": float,
  "expert_score_cv": float
}
```

### 3.2 Expert rank file

One row per `(layer_id, expert_id)`:

```text
{
  "layer_id": int,
  "layer_rank": int,
  "expert_id": int,
  "rank_in_layer": int,
  "percentile_in_layer": float,
  "score": float,
  "ordinary_z_in_layer": float,
  "robust_z_in_layer": float,
  "is_2sigma_candidate": bool,
  "is_robust_outlier_candidate": bool,
  "channel_mean": float,
  "channel_std": float,
  "channel_cv": float,
  "channel_top1_z": float,
  "channel_top5_mean_z": float,
  "channel_top10_mass": float
}
```

The `channel_top10_mass` field is a structural concentration diagnostic. It should be computed from the existing nonnegative participation mass, preferably `expm1(Channel-SP)` after handling non-finite values, and normalized within each expert. It is not a new pruning score.

### 3.3 Channel rank file

One row per `(layer_id, expert_id, channel_id)`:

```text
{
  "layer_id": int,
  "layer_rank": int,
  "expert_id": int,
  "expert_rank_in_layer": int,
  "channel_id": int,
  "rank_in_expert": int,
  "percentile_in_expert": float,
  "score": float,
  "z_in_expert": float,
  "robust_z_in_expert": float,
  "is_2sigma_candidate": bool,
  "is_3sigma_candidate": bool,
  "participation_mass": float,
  "normalized_participation_mass": float
}
```

The channel rank file is the canonical input for all channel-level scripts. Do not reconstruct channel ranks independently in downstream scripts.

## 4. Definitions of a special object

The experiment uses three complementary definitions. No single threshold is treated as the final truth.

### 4.1 Rank segment candidate

A rank segment is a contiguous rank interval whose score distribution changes sharply or whose local slope changes persistently.

For a ranked score sequence `x[1], ..., x[n]`, compute:

- first difference: `d[i] = x[i] - x[i+1]`;
- log-gap when scores are positive: `g[i] = log(x[i] + eps) - log(x[i+1] + eps)`;
- local slope over a small window;
- change-point candidates from a one-change and multi-change detector.

A candidate boundary must satisfy all of the following before being called a natural boundary:

1. the gap is in the upper tail of the within-group gap distribution;
2. the local mean/slope differs on the two sides;
3. the boundary remains close under bootstrap resampling or small perturbation;
4. it is not caused only by one extreme endpoint.

Report candidate boundaries with rank intervals, not only a single threshold.

### 4.2 Statistical outlier candidate

Use outlier flags only as candidate labels:

- ordinary z-score: `z > 2` and `z > 3`;
- robust z-score using median and MAD:

  ```text
  robust_z = (score - median) / (1.4826 * MAD + eps)
  ```

For layers, retain the existing `mean + 2 * std` Layer-0 diagnostic for comparability, but also compute robust outliers for every layer. Do not assume the first layer is special.

For experts and channels, compute outliers only within their parent group:

- Expert: within one layer;
- Channel: within one expert.

### 4.3 Functional candidate

A statistical candidate is considered functionally supported only if its rank prefix has a stable structural separation:

- top-vs-bottom effect size (`Cohen's d`) at 10%, 25%, and 50%;
- precision/recall overlap of candidate sets under bootstrap or score perturbation;
- cumulative participation mass curve has a slope change near the candidate boundary.

Downstream accuracy is not used in this first experiment. Functional here means that the candidate separates the existing ranking distribution in a stable, repeatable way.

## 5. Experiment A: natural segmentation of each rank

Run independently at three levels.

### A1. Layer rank segmentation

Input: one ranked list of all MoE layers per model.

Measure:

- ranked score curve;
- adjacent-gap curve;
- log-gap curve;
- cumulative normalized score / concentration curve;
- candidate change points;
- candidate boundary stability under bootstrap or perturbation.

Outputs:

```text
layer_rank_curve.csv
layer_gap_curve.csv
layer_segments.json
layer_rank_curve.png
```

Questions:

- Is there a clear head segment, bulk segment, and tail segment?
- Is the apparent segment caused by one Layer-SP outlier such as Qwen3.6 L0?
- Are the segments consistent across models, or model-specific?

### A2. Expert rank segmentation

Input: one Expert-SP rank list per layer.

Run both:

- pooled normalized rank analysis across layers;
- per-layer analysis preserving layer identity.

Measure:

- rank curve and adjacent gaps;
- top-1/top-5/top-10 concentration;
- change-point candidates;
- whether the first segment is a stable head or only one expert.

Outputs:

```text
expert_rank_curves/<layer_id>.csv
expert_segments.json
expert_segment_summary.csv
```

Questions:

- Is the common “one sharp head plus smooth tail” a real segment?
- Does the head length depend on the layer?
- Are high Layer-SP layers characterized by a longer head or only by a higher whole-layer level?

### A3. Channel rank segmentation

Input: one Channel-SP rank list per expert.

Measure:

- rank curve and adjacent gaps;
- normalized participation-mass cumulative curve;
- first change point in the head-to-bulk transition;
- stability of the top-10%, top-25%, and top-50% prefixes.

Outputs:

```text
channel_rank_curves/<layer_id>_<expert_id>.csv
channel_segments.json
channel_segment_summary.csv
```

Questions:

- Is there a reproducible head prefix of special channels?
- Is the head length similar across experts or strongly expert-dependent?
- Does a high Expert-SP expert have a longer or more concentrated channel head?

## 6. Experiment B: identify special layers, experts, and channels

### B1. Special layers

For each model, produce three candidate sets:

1. fixed rank prefix: top 5%, 10%, 25%;
2. ordinary 2-sigma candidates;
3. robust-z candidates and change-point head segment.

Record the overlap among these sets. A layer is a strong special-layer candidate only when:

- it is in the stable rank head or a stable segment;
- its outlier label is not caused only by the non-robust mean/std choice;
- its Layer-SP is supported by its Expert-SP distribution.

The existing Layer-0 2-sigma rule is reported, not imposed.

### B2. Special experts

Within each layer, produce:

- top 5%, 10%, 25% Expert-SP prefixes;
- Expert-SP ordinary and robust outlier sets;
- change-point head segment;
- channel-concentration diagnostics.

A strong special expert candidate should have both:

- a stable high Expert-SP rank;
- a nontrivial channel head or high channel concentration.

This explicitly adds the missing Expert-SP special-object analysis.

### B3. Special channels

Within each expert, produce:

- top 1%, 5%, 10%, 25% Channel-SP prefixes;
- ordinary z > 2 and z > 3 candidates;
- robust-z candidates;
- change-point head segment;
- normalized participation-mass concentration.

A channel is a strong special-channel candidate when its rank is stable and it contributes meaningfully to the expert's normalized participation mass. A high z-score alone is not sufficient, because a very small group can create unstable z-scores.

## 7. Experiment C: hierarchical correlation and attribution

The unit of analysis must preserve the hierarchy:

```text
channel -> expert -> layer
```

### C1. Channel-to-expert correlation

For every expert, aggregate channel-level features:

- number/fraction of special channels;
- top-1, top-5, top-10 mean z-score;
- channel head length;
- top-k normalized participation mass;
- channel CV and robust spread.

Compare these with Expert-SP using:

- Pearson correlation for approximately linear association;
- Spearman correlation for rank association;
- partial correlation controlling for layer;
- regression of Expert-SP on channel concentration features.

The key result is whether high Expert-SP experts are characterized by special channels, rather than merely having a high score by an unrelated mechanism.

### C2. Expert-to-layer correlation

For every layer, aggregate expert-level features:

- mean/median Expert-SP;
- top-1 and top-k Expert-SP;
- fraction of special experts;
- Expert-SP concentration / head length;
- mean channel concentration among experts.

Compare these with Layer-SP using:

- Pearson and Spearman correlation;
- leave-one-expert-out sensitivity;
- robust regression across layers.

The leave-one-expert-out test is important: if removing one expert makes the Layer-SP abnormality disappear, the layer is expert-outlier-driven; if it remains, the layer is broad-shift-driven.

### C3. End-to-end channel-to-layer attribution

For every layer, calculate two attribution summaries:

1. **Top-expert-driven:** contribution from the highest Expert-SP experts and their top channel prefixes;
2. **Distributed:** contribution from all experts after removing the top 1%, 5%, and 10% channels.

Classify each layer as:

- **outlier-driven:** a small number of experts/channels explain most of the layer-level separation;
- **distributed-shift:** many experts/channels contribute moderately;
- **mixed:** both effects are present.

This classification directly tests whether a special layer is caused by special channels through a small number of experts or by a broad layer-level shift.

## 8. Stability and null controls

All three-level claims need null controls.

### 8.1 Rank perturbation stability

Perturb scores with small relative noise, for example 0.1%, 0.5%, and 1%, repeat ranking, and report:

- top-k Jaccard overlap;
- Kendall's tau for full rank;
- boundary rank displacement;
- special-set precision/recall.

### 8.2 Within-parent permutation control

Randomly permute scores within each parent group while preserving the score distribution:

- channels within expert;
- experts within layer;
- layers within model.

Recompute segmentation and hierarchy correlations. This tests whether observed segments/correlations are stronger than what the marginal distributions alone would produce.

### 8.3 Aggregation control

Compare the observed Layer-SP with synthetic aggregates built from:

- real Expert-SP rank distributions randomly assigned to layers;
- real Channel-SP rank distributions randomly assigned to experts;
- the same means and variances but destroyed hierarchy.

A hierarchy claim is supported only if observed channel-to-expert-to-layer alignment exceeds these controls.

## 9. Main outputs and decision rules

For each model, the analysis must produce:

```text
rank_analysis/<model>/
  manifest.json
  layer_rank.pt or layer_rank.json
  expert_rank.pt or expert_rank.json
  channel_rank.pt or channel_rank.json
  layer_rank_curve.csv
  expert_segment_summary.csv
  channel_segment_summary.csv
  segment_analysis.json
  hierarchy_links.json
  null_controls.json
```

The cross-model summary should report:

| Question | Required evidence |
|---|---|
| Is there a natural rank segment? | stable change point, gap tail, and cumulative-mass slope change |
| Is the object truly special? | rank/outlier agreement plus perturbation stability |
| Is Expert-SP related to Channel-SP? | within-layer/within-expert Spearman and partial correlation |
| Is Layer-SP related to Expert-SP? | layer-level correlation and leave-one-expert-out attribution |
| Is the hierarchy non-random? | permutation/null-control excess over baseline |

Do not declare a segment solely because a 2-sigma or z-score threshold fired. Do not declare causality solely from Pearson correlation. The first deliverable is a reproducible structural map; only after it is stable should its segments be connected to HARP width allocation.

## 10. Minimal execution order

1. Export the three rank files from existing HARP ranking caches without recomputing scores.
2. Generate ranked curves, gap curves, cumulative participation curves, and candidate segments.
3. Generate special-object candidate sets at fixed quantiles, z-score thresholds, and change points.
4. Compute channel-to-expert and expert-to-layer correlations with layer-aware grouping.
5. Run perturbation and permutation null controls.
6. Compare the five models and classify each layer as outlier-driven, distributed-shift, or mixed.
7. Only then decide whether a rank segment should become an HARP budget tier.

## 11. Expected conclusion format

For each model, write a short conclusion in this form:

```text
Model: <name>
Layer segments: <rank intervals>; stability=<...>; driver=<outlier/distributed/mixed>
Expert segments: <per-layer or pooled intervals>; stability=<...>
Channel segments: <per-expert summary>; stability=<...>
Channel -> Expert: <effect/correlation/null excess>
Expert -> Layer: <effect/correlation/leave-one-out result>
Decision: <use segment / use quantile only / no reliable segment>
```
