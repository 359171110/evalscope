from __future__ import annotations

import torch

from HARP.rank_adaptive_core import (
    allocate_expert_tiers,
    allocate_layer_rank_units,
    allocate_rank_adaptive_widths,
    detect_stable_expert_head,
)


def test_layer_rank_units_close_exact_budget_and_favor_rank_head() -> None:
    units, diagnostics = allocate_layer_rank_units(
        torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
        n_experts=8,
        step_fraction=0.25,
    )
    assert int(units.sum()) == 5 * 8
    assert units[0] > units[-1]
    assert diagnostics["layer_ranks"] == [1, 2, 3, 4, 5]


def test_stable_head_is_detected_and_weak_head_uses_adjacent_tiers() -> None:
    strong_scores = torch.tensor([3.0, 2.9, 2.8, 2.7] + [1.0] * 28)
    strong = detect_stable_expert_head(strong_scores, repeats=8)
    assert strong["strong_head"] is True
    tiers, diagnostics = allocate_expert_tiers(strong_scores, target_units=32, head=strong)
    assert diagnostics["heterogeneity_mode"] == "strong_three_tier"
    assert set(tiers.tolist()) == {0, 1, 2}

    smooth_scores = torch.linspace(1.0, 0.5, 32)
    weak = {"strong_head": False}
    weak_tiers, weak_diag = allocate_expert_tiers(smooth_scores, target_units=32, head=weak)
    assert weak_diag["heterogeneity_mode"] == "weak_low_mid"
    assert set(weak_tiers.tolist()) <= {0, 1}


def test_rank_adaptive_widths_preserve_global_budget() -> None:
    layer_scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    expert_scores = [torch.linspace(1.0, 0.5, 8) for _ in range(5)]
    widths, diagnostics = allocate_rank_adaptive_widths(
        layer_scores,
        expert_scores,
        low_width=64,
        mid_width=128,
        high_width=192,
        repeats=8,
    )
    assert widths.shape == (5, 8)
    assert int(widths.sum()) == 5 * 8 * 128
    assert diagnostics["budget_error"] == 0
    assert set(int(value) for value in widths.reshape(-1).tolist()) <= {64, 128, 192}