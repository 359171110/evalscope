# HARP-RankAdaptive

## 1. Motivation

HARP-RankAdaptive is an independent HARP allocator derived from the rank-structure analysis in:

- `HARP/sp_rank_distribution.md`
- `HARP/rank_analysis_first_pass.md`
- `HARP/rank_analysis_report.md`

The analysis supports a separation of responsibilities:

- Layer-SP rank controls the average retained capacity of each MoE layer.
- Expert-SP rank and head structure control how heterogeneous the experts are inside that layer.
- Channel-SP rank selects the concrete retained channels inside each expert.

The method does not assume that a special layer is caused by a few special experts. The observed Layer-SP outliers are distributed shifts, so layer capacity is adjusted as a bounded whole-layer budget. Expert heterogeneity is decided separately from the within-layer Expert-SP rank.

This implementation is additive. It does not change the behavior or profile format of the existing `legacy`, `combo`, `combo_two`, or `quantile_layer` allocators.

## 2. Inputs

For one model, HARP-RankAdaptive consumes the existing calibration-free HARP ranking cache:

```text
harp_rankings.pt
  layer_scores
  table[layer_id]["expert_structural_scores"]
  table[layer_id]["ranked_indices"]
  table[layer_id]["channel_scores"]
```

The width choices remain symmetric:

\[
K_{low} < K_{mid} < K_{high},
\qquad
K_{mid}-K_{low}=K_{high}-K_{mid}.
\]

`K_mid` is the global target average width. All widths must satisfy the model channel alignment.

## 3. Layer rank budget

Let the model contain \(L\) MoE layers and \(E\) routed experts per layer. Each expert starts at \(K_{low}\). Moving from low to mid or from mid to high consumes one tier-upgrade unit.

The exact global unit budget is:

\[
U_{total}=L E.
\]

This is equivalent to global average width \(K_{mid}\).

### 3.1 Rank groups

Layers are sorted by Layer-SP in descending order and divided by rank percentile into:

- top 25%: signal \(+1\);
- middle 50%: signal \(0\);
- bottom 25%: signal \(-1\).

These are allocation quantiles, not claimed natural change points.

### 3.2 Bounded target units

Let \(a\in[0,1]\) be `layer_step_fraction`, default \(a=0.25\). The raw layer target is:

\[
\widetilde U_l
=
E + aE(s_l-\bar s),
\]

where \(s_l\in\{-1,0,+1\}\). Centering by \(\bar s\) removes imbalance caused by non-divisible layer counts.

The target is clipped to \([0,2E]\), rounded to integer units, and corrected by largest remainder so that:

\[
\sum_l U_l=LE.
\]

Consequently:

- high Layer-SP ranks receive a modest whole-layer budget increase;
- low Layer-SP ranks provide the matching budget decrease;
- no fixed Layer-0 rule is used;
- the global structural budget closes exactly.

With the default \(a=0.25\), the intended average-width shift is one quarter of a tier gap, not a full low/high jump.

## 4. Expert head detection

For every layer, Expert-SP scores are sorted descending. A layer has a stable Expert head only when all conditions hold:

1. the one-change boundary is not an endpoint-only split;
2. the boundary gap is at least the p90 adjacent gap;
3. the mean-shift statistic is greater than 1;
4. under 32 deterministic perturbations with 0.5% relative score noise, at least 75% of boundaries move by no more than two ranks.

The detector returns:

- `head_rank`;
- `natural_boundary`;
- `stability_within_two`;
- `strong_head = natural_boundary and stability_within_two >= 0.75`.

This signal controls heterogeneity amplitude. It does not change the layer budget.

## 5. Layer-internal tier allocation

For layer \(l\), let \(U_l\) be the exact number of upgrade units. Let:

- \(n_h\): number of high experts;
- \(n_m\): number of mid experts;
- \(n_l\): number of low experts.

The constraints are:

\[
n_h+n_m+n_l=E,
\]

\[
2n_h+n_m=U_l.
\]

### 5.1 Stable Expert head: strong heterogeneity

When `strong_head=True`, choose \(n_h\) as close as possible to the detected head length, subject to the exact budget constraints. The remaining units determine \(n_m\), and the rest are low.

The highest Expert-SP experts receive high width, the next experts receive mid width, and the remainder receive low width.

This permits all three tiers when the rank contains a stable head.

### 5.2 No stable Expert head: weak heterogeneity

When `strong_head=False`, minimize width spread while satisfying the same exact layer budget:

- if \(U_l\le E\), use only low and mid;
- if \(U_l>E\), use only mid and high.

Thus a smooth Expert-SP rank does not create an artificial low/high split.

## 6. Channel selection

After expert width \(K_{l,e}\) is fixed, retain the existing Channel-SP prefix:

\[
\mathcal C_{l,e}
=
\operatorname{TopK}(S^C_{l,e,:}, K_{l,e}).
\]

The shared CSP exporter reads `profile_widths` and `ranked_indices`, slices gate/up rows and down columns, and pads heterogeneous logical widths to `K_high` for standard HF/vLLM loading.

HARP-RankAdaptive does not use a Channel-SP change point as the width. Expert tier allocation decides how many channels to keep; Channel-SP decides which channels to keep.

## 7. Profile diagnostics

The profile records enough information to verify the design without evaluation:

- Layer-SP rank and rank group;
- centered layer signal;
- real and integer layer target units;
- actual average width per layer;
- Expert head rank, natural-boundary flag, and stability;
- strong/weak heterogeneity mode;
- low/mid/high expert counts per layer;
- exact global target and actual widths;
- budget error, which must be zero.

The profile mode is:

```text
harp_rank_adaptive_layer_expert_channel_sp
```

The allocation objective is:

```text
bounded_layer_rank_budget_then_adaptive_expert_head_tiers_then_channel_sp_prefix
```

## 8. Independent implementation

The implementation uses independent entry points:

```text
HARP/rank_adaptive_core.py
HARP/build_rank_adaptive_artifacts.py
HARP/tests/test_rank_adaptive.py
```

Existing HARP allocator choices and launchers are not modified.

Example:

```bash
PYTHONPATH=/path/to/evalscope:/path/to/evalscope/static_moe_prunning/code \
python -m HARP.build_rank_adaptive_artifacts \
  --model-path /path/to/model \
  --channel-cache /path/to/harp_rankings.pt \
  --output-profile /path/to/rank_adaptive_profile.pt \
  --low-width 320 \
  --budget-width 384 \
  --high-width 448 \
  --layer-step-fraction 0.25
```

This command builds only the profile. It does not export a checkpoint, start vLLM, or run evaluation.

## 9. Expected behavior checks

For each model, profile-only observation verifies:

1. the global average width equals `budget_width` exactly;
2. high Layer-SP rank groups have no smaller mean layer budget than lower groups;
3. stable Expert-head layers can use strong three-tier heterogeneity;
4. layers without stable heads use adjacent tiers only;
5. every expert width belongs to `{K_low, K_mid, K_high}`;
6. the existing ranking cache is reused without recomputing scores;
7. previous HARP profiles and allocators are unchanged.

## 10. Profile-only observations

The following subsection is filled after running HARP-RankAdaptive on Qwen3, Qwen3.6, and Gemma4. No downstream evaluation is performed.

### 10.1 Qwen3-30B-A3B

Profile-only artifacts:

```text
HARP/experiments/rank_adaptive_20260907/qwen3/rank_adaptive_50.pt
HARP/experiments/rank_adaptive_20260907/qwen3/rank_adaptive_25.pt
```

Observed for both budgets:

- global budget error: `0`;
- layer groups: 12 top / 24 middle / 12 bottom;
- mean layer widths: `budget + 16 / budget / budget - 16`;
- stable Expert-SP heads: 27/48 layers;
- heterogeneity modes: 27 `strong_three_tier`, 18 `weak_low_mid`, 3 `weak_mid_high`;
- tier counts: 510 high / 5124 mid / 510 low experts.

The absolute widths shift from `(368, 384, 400)` around the 50% target to `(560, 576, 592)` around the 25% target, while the rank-derived tier counts stay fixed. This is the intended separation between rank structure and global width target.

### 10.2 Qwen3.6-35B-A3B

Profile-only artifacts:

```text
HARP/experiments/rank_adaptive_20260907/qwen36/rank_adaptive_50.pt
HARP/experiments/rank_adaptive_20260907/qwen36/rank_adaptive_25.pt
```

Observed for both budgets:

- global budget error: `0`;
- layer groups: 10 top / 20 middle / 10 bottom;
- mean layer widths: `budget + 16 / budget / budget - 16`;
- stable Expert-SP heads: 15/40 layers;
- heterogeneity modes: 15 `strong_three_tier`, 21 `weak_low_mid`, 4 `weak_mid_high`;
- tier counts: 872 high / 8496 mid / 872 low experts.

The Layer-SP rank outlier at Layer 0 changes its layer group and budget naturally; there is no fixed Layer-0 protection. The profile therefore treats the layer as a whole-layer budget shift, while only the 15 stable Expert-head layers receive strong three-tier heterogeneity.

### 10.3 Gemma4-26B-A4B

Profile-only artifacts:

```text
HARP/experiments/rank_adaptive_20260907/gemma4/rank_adaptive_50.pt
HARP/experiments/rank_adaptive_20260907/gemma4/rank_adaptive_25.pt
```

Observed for both budgets:

- global budget error: `0`;
- layer groups: 8 top / 14 middle / 8 bottom;
- mean layer widths: `budget + 16 / budget / budget - 16`;
- stable Expert-SP heads: 16/30 layers;
- heterogeneity modes: 16 `strong_three_tier`, 10 `weak_low_mid`, 4 `weak_mid_high`;
- tier counts: 345 high / 3150 mid / 345 low experts.

The Layer-SP rank and Expert-SP head signals are used independently: Layer rank changes the layer average, while the stable Expert head changes whether all three tiers are allowed.

### 10.4 Cross-model conclusion

The profile-only run confirms the intended mechanics on Qwen3, Qwen3.6, and Gemma4:

1. The global average width is exact for all six profiles; no budget is lost during layer grouping or expert tier rounding.
2. Layer rank produces a bounded whole-layer allocation rather than a special Layer-0 bonus. The top/middle/bottom means are exactly `budget+16`, `budget`, and `budget-16` for these three models and width settings.
3. Expert-SP stability changes the heterogeneity mode as designed. Strong-head layers can use low/mid/high; other layers use only adjacent tiers.
4. The same rank-derived tier counts are reused at 50% and 25%; only the physical width choices move with the requested global target. This is expected because the ranking cache is independent of pruning ratio.
5. The existing ranking cache is reused and the old HARP allocators are untouched.

No checkpoint was exported, no vLLM server was started, and no downstream evaluation was run. These observations validate profile construction and budget allocation only; they do not establish an accuracy or latency improvement.
