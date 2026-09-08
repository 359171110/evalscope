from __future__ import annotations

import torch

from HARP.harp_core import allocate_combo_expert_widths
from HARP.rank_adaptive_core import (
    allocate_expert_tiers,
    allocate_layer_rank_units,
    allocate_rank_adaptive_widths,
    allocate_v2_rank_adaptive_widths,
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


def test_v2_rank_adaptive_preserves_combo_base_and_adds_minimal_closure() -> None:
    layer_scores = torch.tensor([0.8, 0.6, 0.5, 0.4], dtype=torch.float64)
    expert_scores = [torch.linspace(1.0 + row, 0.5 + row, 16) for row in range(4)]
    expected, _ = allocate_combo_expert_widths(
        layer_scores,
        expert_scores,
        low_width=64,
        mid_width=128,
        high_width=192,
        global_avg_width=128.0,
        gamma=2.0,
        min_fraction=0.15,
        allow_mid=True,
    )
    actual, diagnostics = allocate_v2_rank_adaptive_widths(
        layer_scores,
        expert_scores,
        low_width=64,
        mid_width=128,
        high_width=192,
        gamma=2.0,
        min_fraction=0.15,
        repeats=8,
    )
    delta = actual - expected
    assert bool(((delta == 0) | (delta == 64)).all())
    assert int(actual.sum().item()) == 4 * 16 * 128
    assert diagnostics["budget_closure"]["missing_width_before"] >= 0
    assert diagnostics["allocator"] == "v2_rank_adaptive"
    assert diagnostics["base_allocator"] == "combo"
    assert all(layer["head_used_as_high_count"] is False for layer in diagnostics["layers"])


def test_v2_rank_adaptive_closes_budget_after_layer_target_clipping() -> None:
    layer_scores = torch.tensor([2.0] + [0.5] * 7, dtype=torch.float64)
    expert_scores = [torch.linspace(1.0, 0.5, 16) for _ in range(8)]
    widths, diagnostics = allocate_v2_rank_adaptive_widths(
        layer_scores,
        expert_scores,
        low_width=64,
        mid_width=128,
        high_width=192,
        gamma=2.0,
        min_fraction=0.15,
        repeats=8,
    )
    assert int(widths.sum().item()) == 8 * 16 * 128
    assert diagnostics["leftover_final"] == 0.0
    assert abs(sum(diagnostics["layer_target_widths"]) - 8 * 128) < 1.0e-6
    assert diagnostics["layer_target_widths"][0] >= max(diagnostics["layer_target_widths"][1:])