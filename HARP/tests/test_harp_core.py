from __future__ import annotations

import torch

from HARP.harp_core import (
    allocate_combo_expert_widths,
    allocate_expert_widths,
    allocate_layer_quantile_high_counts,
    allocate_layer_upgrade_units,
    allocate_quantile_layer_expert_widths,
    detect_anchor_layer,
    search_expert_tier_counts,
)


def test_detect_anchor_layer_uses_two_sigma_rule() -> None:
    assert detect_anchor_layer(torch.tensor([10.0] + [1.0] * 47)) == 0
    assert detect_anchor_layer(torch.tensor([1.0] * 48)) is None


def test_layer_and_expert_allocations_are_aligned_and_exact() -> None:
    layers = allocate_layer_upgrade_units(
        torch.tensor([3.0, 2.0, 1.0]), total_units=6, max_units_per_layer=4,
    )
    assert layers.tolist() == [2, 2, 2]
    experts = allocate_expert_widths(torch.tensor([1.0, 3.0, 2.0]), low_blocks=5, target_units=3)
    assert experts.tolist() == [5, 7, 6]
    assert int((experts - 5).sum()) == 3


def test_combo_search_uses_all_three_tiers_under_mid_budget() -> None:
    n_high, n_mid, n_low = search_expert_tier_counts(128 * 352, 128, 288, 352, 416)
    assert n_high >= 19 and n_mid >= 19 and n_low >= 19
    assert n_high + n_mid + n_low == 128
    cost = n_high * 416 + n_mid * 352 + n_low * 288
    assert cost <= 128 * 352
    assert n_high > 0 and n_low > 0


def test_combo_waterfill_boosts_high_layer_sp() -> None:
    layer_scores = torch.tensor([1.0, 0.5, 0.4, 0.4], dtype=torch.float64)
    expert_scores = [torch.arange(32.0, 0.0, -1.0) for _ in range(4)]
    widths, diag = allocate_combo_expert_widths(
        layer_scores,
        expert_scores,
        low_width=192,
        mid_width=256,
        high_width=320,
        global_avg_width=256.0,
        gamma=2.0,
        min_fraction=0.15,
    )
    means = widths.float().mean(dim=1)
    assert float(means[0]) > float(means[1])
    assert float(means[0]) > float(means[-1])
    assert set(widths.reshape(-1).tolist()) <= {192, 256, 320}
    assert int(diag["expert_tier_counts_by_layer"][0][0]) > int(diag["expert_tier_counts_by_layer"][-1][0])


def test_combo_two_search_drops_mid_and_keeps_128_gap() -> None:
    n_high, n_mid, n_low = search_expert_tier_counts(
        128 * 352, 128, 288, 352, 416, allow_mid=False,
    )
    assert n_mid == 0
    assert n_high + n_low == 128
    assert n_high >= 19 and n_low >= 19
    assert n_high * 416 + n_low * 288 <= 128 * 352


def test_combo_two_waterfill_never_assigns_mid() -> None:
    layer_scores = torch.tensor([1.0, 0.5, 0.4, 0.4], dtype=torch.float64)
    expert_scores = [torch.arange(32.0, 0.0, -1.0) for _ in range(4)]
    widths, diag = allocate_combo_expert_widths(
        layer_scores,
        expert_scores,
        low_width=192,
        mid_width=256,
        high_width=320,
        global_avg_width=256.0,
        gamma=2.0,
        min_fraction=0.15,
        allow_mid=False,
    )
    assert set(widths.reshape(-1).tolist()) <= {192, 320}
    assert 256 not in set(widths.reshape(-1).tolist())
    assert all(counts[1] == 0 for counts in diag["expert_tier_counts_by_layer"])
    assert float(widths.float().mean(dim=1)[0]) > float(widths.float().mean(dim=1)[-1])
    assert bool(diag["allow_mid"] is False)


def test_layer_quantile_high_counts_are_two_level_and_isotonic() -> None:
    scores = torch.tensor([1.0, 0.5, 0.4, 0.35], dtype=torch.float64)
    counts = allocate_layer_quantile_high_counts(
        scores, n_experts=32, total_high=64, min_fraction=0.15,
    )
    assert counts.tolist() == [28, 28, 4, 4]
    assert int(counts.sum()) == 64
    for left, right in zip(counts.tolist(), counts.tolist()[1:]):
        assert left >= right


def test_layer_quantile_cut_layer_absorbs_remainder() -> None:
    scores = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    counts = allocate_layer_quantile_high_counts(
        scores, n_experts=10, total_high=17, min_fraction=0.2,
    )
    assert int(counts.sum()) == 17
    assert counts.tolist()[0] >= counts.tolist()[1] >= counts.tolist()[2]


def test_quantile_layer_allocation_splits_layers_not_experts_evenly() -> None:
    layer_scores = torch.tensor([1.0, 0.5, 0.4, 0.35], dtype=torch.float64)
    expert_scores = [torch.arange(32.0, 0.0, -1.0) for _ in range(4)]
    widths, diag = allocate_quantile_layer_expert_widths(
        layer_scores,
        expert_scores,
        low_width=192,
        high_width=320,
        global_avg_width=256.0,
        min_fraction=0.15,
    )
    assert set(widths.reshape(-1).tolist()) <= {192, 320}
    assert 256 not in set(widths.reshape(-1).tolist())
    assert int(widths.sum()) == 4 * 32 * 256
    highs = [counts[0] for counts in diag["expert_tier_counts_by_layer"]]
    assert highs[:2] == [28, 28]
    assert highs[2:] == [4, 4]
    means = widths.float().mean(dim=1)
    assert float(means[0]) == float(means[1])
    assert float(means[0]) > float(means[-1])
    assert bool(diag["allow_mid"] is False)
    assert diag["gamma"] is None


