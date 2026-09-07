# HARP-v2-RankAdaptive

## 1. Method status and design goal

This document records the current HARP method iteration. The goal is not to replace the empirically strong HARP-v2 allocator with a more complicated heuristic. The goal is to retain HARP-v2's effective budget allocation while incorporating the rank-structure findings from the HARP analysis into a clearer, more defensible method story.

The final candidate is named **HARP-v2-RankAdaptive**.

It is an additive refinement of HARP-v2 (`allocator="combo"`):

```text
Layer-SP rank-aware water-fill
    -> exact per-layer budget with remainder propagation
    -> Expert-SP direct low/mid/high tier search
    -> rank-ordered expert assignment
    -> Channel-SP prefix selection
```

The key design principle is hierarchical responsibility:

- **Layer-SP** controls the layer-level average capacity. It is treated as a bounded structural budget signal, not as a hard special-layer detector.
- **Expert-SP** determines the within-layer ordering and the feasible tier pattern. It is not converted directly into a high-expert count through a change-point rank.
- **Channel-SP** selects the concrete prefix of channels after the expert width is fixed.

This revision explicitly avoids the main failure mode of the first independent RankAdaptive prototype: interpreting an Expert-SP change-point rank as the number of high-width experts.

## 2. Evidence behind the design

The method is based on the existing rank analysis:

- `HARP/sp_rank_distribution.md`
- `HARP/rank_analysis_first_pass.md`
- `HARP/rank_analysis_report.md`

The observations are:

1. Layer-SP rank is informative across models, but a universal natural Layer-SP segment is not reliable. Layer outliers are usually distributed shifts across the layer's experts.
2. Expert-SP frequently has a stable head, but the head position should not be treated as an optimal high-tier count. It is evidence that Expert-SP ordering is structured and that heterogeneous tiers are meaningful.
3. Channel-SP has useful within-expert prefixes. Channel rank chooses which channels to retain; it does not determine the layer budget.
4. Removing a few top experts does not remove the main Layer-SP outlier. Layer capacity therefore needs to be allocated at the whole-layer level.
5. HARP-v2's direct tier search is a better discrete allocator than a hand-written `head_rank -> n_high` mapping because it uses the available budget and keeps tier counts feasible and balanced.

The resulting method story is:

> MoE structural ranks have different statistical roles. HARP-v2-RankAdaptive uses Layer-SP to shape the layer budget, Expert-SP to solve the discrete within-layer allocation, and Channel-SP to select channels, while preserving exact global budget closure.

## 3. Inputs and unchanged components

The method consumes the existing calibration-free HARP ranking cache:

```text
harp_rankings.pt
  layer_scores
  table[layer_id]["expert_structural_scores"]
  table[layer_id]["ranked_indices"]
  table[layer_id]["channel_scores"]
```

The following components remain unchanged:

- raw weight-only Layer-SP, Expert-SP, and Channel-SP ranking;
- packed/separate model adapters;
- Channel-SP ranked prefix selection;
- CSP checkpoint exporter and padding behavior;
- exact global structural budget contract;
- existing HARP `legacy`, `combo_two`, and `quantile_layer` allocators.

The three hardware-friendly widths satisfy:

\[
K_{low} < K_{mid} < K_{high},
\qquad
K_{mid}-K_{low}=K_{high}-K_{mid}.
\]

`K_mid` is the target global average expert width. All widths must satisfy the model's channel alignment.

## 4. Layer-SP rank-aware water-fill

Let \(S_l^L\) be the Layer-SP score for layer \(l\), \(K_{low}\) the minimum width, and \(K_{mid}\) the target global average width.

The initial layer target follows HARP-v2's water-fill:

\[
\widetilde K_l
=
K_{low}
+
(K_{mid}-K_{low})
\frac{(S_l^L / \sum_j S_j^L)^\gamma}
{\operatorname{mean}_j[(S_j^L / \sum_j S_j^L)^\gamma]}.
\]

The target is clipped to the supported interval:

\[
K_{low}\leq \widetilde K_l\leq K_{high}.
\]

Clipping can reduce the global target sum when an extreme layer reaches
`K_high`. HARP-v2-RankAdaptive therefore projects the clipped targets back onto
the exact bounded budget:

\[
\sum_l K_l^{target}=L K_{mid},
\qquad
K_{low}\leq K_l^{target}\leq K_{high}.
\]

The residual is distributed uniformly over layers that have not reached the
relevant bound. This preserves the score-induced ordering while preventing an
outlier such as Qwen3.6 Layer 0 from discarding global capacity after it
saturates at `K_high`.

The default is:

```text
gamma = 2.0
```

This preserves the performance-oriented HARP-v2 behavior while making its interpretation explicit: Layer-SP supplies a bounded layer-average budget, not a binary special-layer label.

### 4.1 Rank-aware interpretation

The Layer-SP rank is recorded for every layer and used for diagnostics and monotonicity checks. The allocator does not introduce a fixed Layer-0 bonus and does not replace the score-magnitude water-fill with a hard 25/50/25 rule.

This is intentional:

- the five-model analysis shows that Layer-SP score magnitudes contain useful model-specific information;
- Qwen3.6 Layer 0 is an extreme magnitude outlier that a pure rank bucket would flatten;
- DeepSeek's high layers are in the middle of the network, so a fixed depth prior is invalid.

Thus the final method uses:

```text
Layer-SP magnitude for the v2 budget signal
Layer-SP rank for interpretation, ordering, and validation
```

### 4.2 Exact budget closure

Layer targets are converted to per-layer budgets by multiplying by the number of experts. Layers are processed in descending Layer-SP order. If a layer's discrete tier solution uses less than its available budget, the remainder is passed to the next layer, as in HARP-v2.

After combo search, any remaining aligned tier gap is closed by the minimum
number of one-tier expert upgrades. Candidates are ordered first by Layer-SP
rank and then by Expert-SP rank. The closure never changes an expert by more
than one tier and is recorded in the profile.

The final expert widths must satisfy:

\[
\sum_{l,e} K_{l,e}=L E K_{mid}.
\]

Remainder propagation is an implementation detail of exact discrete budget closure, not an additional learnable or tuned parameter.

## 5. Expert-SP direct tier search

For each layer, let \(B_l\) be the current available width budget after water-fill and remainder propagation. Search over integer tier counts:

\[
n_h+n_m+n_{low}=E,
\]

\[
n_hK_{high}+n_mK_{mid}+n_{low}K_{low}
\leq B_l.
\]

The selected combination maximizes consumed extra width over the low baseline:

\[
\operatorname{score}(n_h,n_m,n_{low})
=
 n_h(K_{high}-K_{low})
+n_m(K_{mid}-K_{low}).
\]

Ties favor a more balanced tier-count distribution. A small fixed minimum tier coverage is retained from HARP-v2 so that the search does not collapse to one unused tier when a feasible three-tier solution exists.

The concrete experts are assigned by Expert-SP rank:

```text
highest Expert-SP experts -> high
next Expert-SP experts     -> mid
remaining experts         -> low
```

This is the critical correction over the first RankAdaptive prototype:

```text
Expert head rank != number of high experts
```

The Expert-SP head is used as a structural interpretation of why rank-ordered tiering is meaningful. The integer tier counts are selected by the budget-constrained combo search, not by the change-point position.

## 6. Optional rank-adaptive tier constraint

The default final candidate keeps the HARP-v2 combo search unchanged for the main performance path. The rank analysis is recorded in the profile as diagnostics:

- Layer-SP rank and score;
- Expert-SP rank curve;
- detected Expert-SP head rank;
- whether the head is stable under the fixed offline diagnostic;
- selected low/mid/high tier counts;
- whether the selected tier pattern is consistent with the observed head.

If an ablation is needed, a constrained variant can impose only a weak structural condition:

- stable Expert head: allow all three tiers;
- no stable head: prefer adjacent tiers when the objective is tied.

This is an ablation and not the default method. It must not override the exact combo search with `n_high = head_rank`.

## 7. Channel-SP prefix selection

After \(K_{l,e}\) is selected, retain the top-ranked Channel-SP prefix:

\[
\mathcal C_{l,e}
=
\operatorname{TopK}(S^C_{l,e,:},K_{l,e}).
\]

The shared CSP exporter slices gate/up rows and down columns using the same channel indices and pads heterogeneous logical widths to `K_high` for standard HF/vLLM loading.

Channel-SP therefore has one precise role:

```text
Expert tier allocation decides how many channels survive.
Channel-SP rank decides which channels survive.
```

No Channel-SP change point is used as a new width hyperparameter.

## 8. Minimal method hyperparameters

The final method intentionally exposes only the parameters that define the allocation problem:

| Parameter | Role | Default |
|---|---|---:|
| `low_width` | minimum hardware-aligned expert width | model/pruning target |
| `budget_width` | exact global average expert width | model/pruning target |
| `high_width` | maximum hardware-aligned expert width | model/pruning target |
| `gamma` | Layer-SP water-fill concentration | 2.0 |
| `min_fraction` | minimum tier coverage in combo search | 0.15 |

The following are fixed diagnostics rather than method-level search parameters:

- 32 perturbations;
- 0.5% relative score noise;
- two-rank head displacement tolerance;
- p90 gap and mean-shift checks.

They are used to describe rank structure and populate diagnostics. They do not replace combo search or define the global budget.

## 9. Algorithm summary

```text
Input: model, HARP ranking cache, K_low/K_mid/K_high

1. Read Layer-SP, Expert-SP, and Channel-SP rankings.
2. Compute HARP-v2 Layer-SP water-fill targets using gamma.
3. Sort layers by Layer-SP rank and propagate discretization remainder.
4. For each layer, search feasible low/mid/high expert tier counts.
5. Map high/mid/low counts to Expert-SP rank order.
6. For every expert, retain its Channel-SP top-K prefix.
7. Validate exact global width budget and alignment.
8. Export through the existing CSP exporter when a checkpoint is required.
```

The method has one coherent optimization object:

\[
\max_{K_{l,e}\in\{K_{low},K_{mid},K_{high}\}}
\sum_{l,e}
\operatorname{rank\text{-}utility}_{l,e}(K_{l,e})
\]

subject to:

\[
\sum_{l,e}K_{l,e}=LEK_{mid},
\]

with Layer-SP defining the layer budget and Expert-SP defining the within-layer rank utility order.

## 10. Why this version is the intended ICLR story

The contribution is not a new isolated score. It is a rank-aware hierarchical budget decomposition:

1. The analysis shows that Layer-SP, Expert-SP, and Channel-SP have different rank geometries.
2. Layer-level outliers are distributed shifts, so they receive bounded whole-layer budgets instead of a few-expert bonus.
3. Expert-level rank ordering is useful, but a change-point rank is not itself an allocation count.
4. Channel-level prefixes are retained because the channel rank has strong within-expert concentration.
5. HARP-v2's direct tier search preserves the empirically strong discrete allocation behavior.
6. The final method closes the exact budget and remains calibration-free.

The intended claim is:

> HARP-v2-RankAdaptive is a calibration-free, rank-aware hierarchical MoE pruning method that decomposes a global budget into Layer-SP-guided layer capacities, Expert-SP-ordered heterogeneous tiers, and Channel-SP-selected prefixes.

This is a refinement of a strong allocator guided by structural rank evidence, not a claim that every rank change point is a new pruning boundary.

## 11. Compatibility and implementation

The existing implementations remain available:

```text
legacy
combo       # HARP-v2 performance baseline
combo_two
quantile_layer
```

The final candidate is implemented as an independent allocator and profile mode. It does not change the behavior of the old allocators.

Independent entry points:

```text
HARP/build_rank_adaptive_artifacts.py
HARP/rank_adaptive_core.py
HARP/tests/test_rank_adaptive.py
```

The implementation reuses the validated HARP-v2 primitives:

```text
layer_sp_weighted_targets
allocate_combo_expert_widths
search_expert_tier_counts
assign_expert_tier_widths
```

The profile mode is:

```text
harp_v2_rank_adaptive_layer_expert_channel_sp
```

The profile additionally records the initial and projected Layer-SP targets,
rank diagnostics, selected tier counts, pre-closure remainder, and exact
closure upgrades.

## 12. Current iteration record

The earlier independent RankAdaptive prototype used:

```text
bounded 25/50/25 Layer rank groups
Expert head rank -> high expert count
weak-head layers -> adjacent tiers only
```

Profile-only construction succeeded on Qwen3, Qwen3.6, and Gemma4, but the method underperformed HARP-v2. The likely cause was not the hierarchical rank interpretation itself; it was the direct conversion of `head_rank` into `n_high` and the resulting collapse of many layers toward all-mid or weakly heterogeneous allocations.

The current iteration replaces that conversion with HARP-v2's direct tier search and retains the rank findings as the method's structural explanation and diagnostics.

## 13. Validation plan

For the final candidate, compare under the same width budget:

- HARP-v2 (`combo`);
- HARP-v2-RankAdaptive;
- HARP `quantile_layer`;
- uniform-width CSP/HARP baseline.

The core profile checks are:

- exact global budget closure;
- valid channel alignment;
- Layer-SP water-fill monotonicity by rank;
- Expert-SP rank-ordered tier assignment;
- Channel-SP prefix validity;
- no changes to existing allocator outputs.

Downstream accuracy and runtime comparisons are separate experiments. This document records the method iteration and does not claim an improvement until those comparisons are run.

## 14. Qwen profile-only observations

The implemented method was run on the existing Qwen3 and Qwen3.6 HARP ranking
caches at 50% and 25% pruning targets. No checkpoint was exported and no
downstream evaluation was run.

Artifacts:

```text
HARP/experiments/v2_rank_adaptive_20260908/qwen3/v2_rank_adaptive_50.pt
HARP/experiments/v2_rank_adaptive_20260908/qwen3/v2_rank_adaptive_25.pt
HARP/experiments/v2_rank_adaptive_20260908/qwen36/v2_rank_adaptive_50.pt
HARP/experiments/v2_rank_adaptive_20260908/qwen36/v2_rank_adaptive_25.pt
```

### 14.1 Qwen3

For both pruning targets:

- the actual global mean exactly equals `budget_width`;
- the final remainder is zero;
- Layer-SP top ranks are `0, 47, 37, 41, 25`;
- 27/48 layers have a stable Expert-SP head, recorded only as diagnostics;
- no head rank is used as a high-expert count;
- tier counts are 2054 high / 2036 mid / 2054 low;
- the exact closure requires one one-tier upgrade beyond the base combo result.

The projected layer target range is:

```text
50%: 381.621 to 408.681 around target 384
25%: 573.621 to 600.681 around target 576
```

The method therefore preserves HARP-v2's smooth layer allocation and expert
combo pattern, with only the minimum correction needed for exact budget
closure.

### 14.2 Qwen3.6

For both pruning targets:

- the actual global mean exactly equals `budget_width`;
- the final remainder is zero;
- Layer-SP top ranks are `0, 39, 33, 32, 34`;
- 15/40 layers have a stable Expert-SP head, recorded only as diagnostics;
- no head rank is used as a high-expert count;
- tier counts are 3449 high / 3342 mid / 3449 low;
- no post-search tier upgrade is needed after exact target projection.

The projected layer target range is:

```text
50%: 251.663 to 320.000 around target 256
25%: 379.663 to 448.000 around target 384
```

Layer 0 reaches `K_high`, as expected from its extreme Layer-SP. Its clipped
overflow is redistributed over the unsaturated layers instead of being lost.
This corrects the base combo behavior, whose actual average was approximately
3.5 channels below the requested Qwen3.6 target.

### 14.3 Structural conclusion

The two-model run confirms the intended algorithm behavior:

1. HARP-v2 water-fill and direct tier search remain the allocation backbone.
2. Expert head ranks are never converted into tier counts.
3. Rank diagnostics are available without changing the combo objective.
4. Bounded target projection handles extreme Layer-SP clipping.
5. Aligned discrete closure makes the global average width exact.
6. All expert widths remain in the configured low/mid/high set.

These checks establish that the method construction works as specified. They do
not establish an accuracy improvement because no model evaluation was performed.
