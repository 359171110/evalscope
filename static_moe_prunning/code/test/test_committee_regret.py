from __future__ import annotations

import pytest
import torch

from src.committee_regret import (
    build_frontier_regret_floors,
    diagonal_block_committee_residual,
)


def test_committee_residual_suppresses_blocks_aligned_with_other_experts() -> None:
    middle = torch.ones(1, 4)
    down_weight = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    other_output = torch.tensor([[3.0, 0.0]])

    residual = diagonal_block_committee_residual(
        middle,
        down_weight,
        other_output,
        routing_weights=torch.ones(1),
        ranked_indices=torch.arange(4),
        block_sizes=torch.tensor([2, 2]),
    )

    assert torch.allclose(residual, torch.tensor([[0.0, 5.0**0.5]]))


def test_committee_residual_reduces_to_diagonal_block_output_norm_without_peer() -> None:
    middle = torch.ones(1, 4)
    down_weight = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )

    residual = diagonal_block_committee_residual(
        middle,
        down_weight,
        torch.zeros(1, 2),
        routing_weights=torch.tensor([0.5]),
        ranked_indices=torch.arange(4),
        block_sizes=torch.tensor([2, 2]),
    )

    assert torch.allclose(
        residual, torch.tensor([[5.0**0.5 / 2.0] * 2])
    )


def test_committee_residual_rejects_non_permutation_channel_order() -> None:
    with pytest.raises(ValueError, match="permutation"):
        diagonal_block_committee_residual(
            torch.ones(1, 2),
            torch.eye(2),
            torch.zeros(1, 2),
            routing_weights=torch.ones(1),
            ranked_indices=torch.tensor([0, 0]),
            block_sizes=torch.tensor([1, 1]),
        )


def test_frontier_regret_floor_uses_only_first_pruned_block_and_worst_fold() -> None:
    folds = torch.tensor(
        [
            [[[10.0, 1.0, 1.0], [1.0, 2.0, 9.0], [99.0, 99.0, 99.0]]],
            [[[8.0, 1.0, 1.0], [1.0, 4.0, 9.0], [99.0, 99.0, 99.0]]],
        ]
    )
    reference_widths = torch.tensor([[0, 1, 3]])

    min_widths, audit, score = build_frontier_regret_floors(
        folds,
        reference_widths,
        global_quantile=0.5,
        width_increment=1,
        aggregation="minimum",
    )

    assert min_widths.tolist() == [[1, 0, 0]]
    assert audit["eligible_count"] == 2
    assert audit["selected_count"] == 1
    assert audit["selected_experts"][0]["reference_width"] == 0
    assert audit["selected_experts"][0]["min_width"] == 1
    assert score[0, 0] > score[0, 1]
    assert score[0, 2] == 0.0
