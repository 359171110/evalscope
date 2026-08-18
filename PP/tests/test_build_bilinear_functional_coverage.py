from __future__ import annotations

import torch

from PP.build_bilinear_functional_coverage import (
    bilinear_functional_coverage_order,
    bilinear_functional_similarity,
    candidate_channel_count,
    local_bilinear_functional_coverage_order,
    selection_diagnostics,
)


def test_candidate_count_matches_frozen_budgets() -> None:
    assert candidate_channel_count(768, 384, 77, 0.5) == 499
    assert candidate_channel_count(768, 576, 77, 0.5) == 595


def test_bilinear_similarity_does_not_treat_negative_covariance_as_redundancy() -> None:
    gate = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    up = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    similarity = bilinear_functional_similarity(gate, up)

    assert similarity[0, 0].item() == 1.0
    assert similarity[0, 1].item() == 0.0


def test_bfc_keeps_pp_prefix_and_uses_aimer_screened_candidates() -> None:
    gate = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    up = gate.clone()
    aimer_order = torch.tensor([0, 1, 2, 3])
    pseudo_order = torch.tensor([0, 3, 2, 1])

    order = bilinear_functional_coverage_order(
        gate,
        up,
        aimer_order,
        pseudo_order,
        retained_channels=2,
        protected_channels=1,
        candidate_extra_ratio=0.5,
    )

    assert order[:2].tolist() == [0, 2]
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_bfc_uses_aimer_order_to_break_novelty_ties() -> None:
    gate = torch.eye(3)
    up = torch.eye(3)
    aimer_order = torch.tensor([2, 1, 0])
    pseudo_order = torch.tensor([0, 1, 2])

    order = bilinear_functional_coverage_order(
        gate,
        up,
        aimer_order,
        pseudo_order,
        retained_channels=1,
        protected_channels=0,
        candidate_extra_ratio=0.5,
    )

    assert order[0].item() == 2


def test_local_bfc_freezes_pp_and_high_confidence_aimer_channels() -> None:
    gate = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    up = gate.clone()
    aimer_order = torch.tensor([0, 1, 2, 3, 4, 5])
    pseudo_order = torch.tensor([5, 0, 1, 2, 3, 4])

    order = local_bilinear_functional_coverage_order(
        gate,
        up,
        aimer_order,
        pseudo_order,
        retained_channels=4,
        protected_channels=1,
        boundary_channels=1,
    )

    assert order[:3].tolist() == [5, 0, 1]
    assert order[3].item() in {2, 3}
    assert sorted(order.tolist()) == list(range(6))


def test_local_bfc_selects_exactly_from_the_cutoff_boundary() -> None:
    channel_count = 12
    gate = torch.eye(channel_count)
    up = torch.eye(channel_count)
    aimer_order = torch.arange(channel_count)
    pseudo_order = torch.tensor([11, *range(11)])
    retained_channels = 8
    protected_channels = 1
    boundary_channels = 2

    order = local_bilinear_functional_coverage_order(
        gate,
        up,
        aimer_order,
        pseudo_order,
        retained_channels=retained_channels,
        protected_channels=protected_channels,
        boundary_channels=boundary_channels,
    )

    frozen = {11, *range(5)}
    boundary_pool = {5, 6, 7, 8}
    selected = set(order[:retained_channels].tolist())
    selected_boundary = selected - frozen

    assert order[:retained_channels].tolist() == [11, 0, 1, 2, 3, 4, 5, 6]
    assert frozen <= selected
    assert selected_boundary <= boundary_pool
    assert len(selected_boundary) == boundary_channels
    assert sorted(order.tolist()) == list(range(channel_count))


def test_selection_diagnostics_use_overlap_and_one_based_aimer_rank() -> None:
    gate = torch.eye(4)
    up = torch.eye(4)
    aimer_order = torch.tensor([2, 0, 3, 1])
    baseline_order = torch.tensor([2, 0, 3, 1])
    global_order = torch.tensor([1, 3, 2, 0])
    local_order = torch.tensor([2, 3, 0, 1])

    diagnostics = selection_diagnostics(
        gate,
        up,
        aimer_order,
        baseline_order,
        global_order,
        local_order,
        retained_channels=2,
    )

    assert diagnostics["aimer_overlap_with_aimer"] == 1.0
    assert diagnostics["global_bfc_overlap_with_aimer"] == 0.0
    assert diagnostics["local_bfc_overlap_with_aimer"] == 0.5
    assert diagnostics["aimer_mean_aimer_rank"] == 1.5
    assert diagnostics["global_bfc_mean_aimer_rank"] == 3.5
    assert diagnostics["local_bfc_mean_aimer_rank"] == 2.0
    assert diagnostics["aimer_mean_pairwise_redundancy"] == 0.0