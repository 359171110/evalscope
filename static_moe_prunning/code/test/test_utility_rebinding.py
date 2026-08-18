from __future__ import annotations

import pytest
import torch

from src.utility_rebinding import (
    aggregate_unique_contribution_folds,
    aggregate_output_saliency_folds,
    compute_co_route_uniqueness,
    fuse_expert_utility_with_output_saliency,
    rebind_expert_utility_to_coverage,
)


def test_rebinding_preserves_expert_utility_and_changes_only_coverage_shape() -> None:
    old_coverage = torch.tensor([[[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]])
    expert_utility = torch.tensor([[2.0, 5.0]])
    old_values = expert_utility.unsqueeze(-1) * (old_coverage + 1.0e-8)
    new_coverage = torch.tensor([[[0.8, 0.15, 0.05], [0.6, 0.25, 0.15]]])

    rebound, recovered = rebind_expert_utility_to_coverage(
        old_values, old_coverage, new_coverage
    )

    assert torch.allclose(recovered, expert_utility)
    assert torch.allclose(
        rebound, expert_utility.unsqueeze(-1) * (new_coverage + 1.0e-8)
    )


def test_rebinding_rejects_nonmonotone_new_prefix_coverage() -> None:
    old = torch.tensor([[[0.6, 0.4]]])

    with pytest.raises(ValueError, match="non-increasing"):
        rebind_expert_utility_to_coverage(
            old, old, torch.tensor([[[0.4, 0.6]]])
        )


def test_output_saliency_fusion_is_layer_normalized_and_beta_controlled() -> None:
    utility = torch.tensor([[2.0, 2.0]])
    saliency = torch.tensor([[1.0, 3.0]])
    fused, factor = fuse_expert_utility_with_output_saliency(
        utility, saliency, beta=1.0
    )
    assert torch.allclose(factor, torch.tensor([[0.5, 1.5]]))
    assert torch.allclose(fused, torch.tensor([[1.0, 3.0]]))


def test_output_saliency_fold_aggregation_normalizes_each_fold_before_mean() -> None:
    folds = torch.tensor(
        [
            [[1.0, 3.0]],
            [[30.0, 10.0]],
        ]
    )

    consensus = aggregate_output_saliency_folds(folds)

    assert torch.allclose(consensus, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(consensus.mean(dim=1), torch.ones(1))


def test_output_saliency_fold_aggregation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        aggregate_output_saliency_folds(torch.tensor([[[1.0, float("nan")]]]))


def test_co_route_uniqueness_penalizes_functional_clones() -> None:
    contexts = torch.tensor(
        [
            [
                [0.0, 0.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0, 0.0, 3.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
            ]
        ]
    )

    uniqueness = compute_co_route_uniqueness(contexts)

    assert uniqueness.shape == (1, 5)
    assert uniqueness[0, 0] == pytest.approx(0.0)
    assert uniqueness[0, 1] == pytest.approx(0.0)
    assert uniqueness[0, 3] > uniqueness[0, 0]


def test_unique_contribution_combines_fold_normalized_output_with_uniqueness() -> None:
    saliency_folds = torch.tensor(
        [
            [[1.0, 3.0, 2.0]],
            [[10.0, 30.0, 20.0]],
        ]
    )
    context_folds = torch.tensor(
        [
            [[[0.0, 2.0, 1.0], [2.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            [[[0.0, 4.0, 2.0], [4.0, 0.0, 2.0], [2.0, 2.0, 0.0]]],
        ]
    )

    score, uniqueness = aggregate_unique_contribution_folds(
        saliency_folds, context_folds
    )

    assert uniqueness.shape == (2, 1, 3)
    assert torch.allclose(uniqueness[0], uniqueness[1])
    assert torch.allclose(
        score,
        torch.tensor([[0.5, 1.5, 1.0]]) * uniqueness[0],
    )
    assert score[0, 1] > score[0, 0]

    minimum_score, _ = aggregate_unique_contribution_folds(
        saliency_folds,
        context_folds,
        aggregation="minimum",
    )
    assert torch.allclose(minimum_score, score)
