from __future__ import annotations

import torch

from evaluate_ramp_e1_stage_a import choose_primary_selection
from ramp_reconstruction import (
    conditional_activation_selection,
        pairwise_output_correlation_selection,
    fit_rank_limited_compensation,
    fit_ridge_compensation,
    normalized_output_error,
    ramp_conditional_residual_selection,
    rank_rms_channels,
    rank_tail_channels,
)
from run_ramp_reconstruction_probe import choose_shared_alpha


def test_ridge_compensation_recovers_perfectly_correlated_pruned_channel() -> None:
    down_proj = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    keep = torch.tensor([0])

    effective, delta = fit_ridge_compensation(down_proj, covariance, keep, regularization=0.0)

    assert torch.allclose(delta, torch.tensor([[1.0], [2.0]], dtype=torch.float64))
    assert torch.allclose(effective, torch.tensor([[2.0], [4.0]], dtype=torch.float64))
    assert normalized_output_error(down_proj, covariance, keep, effective) == 0.0
    assert normalized_output_error(down_proj, covariance, keep, down_proj[:, keep]) == 0.25


def test_rms_and_tail_rankings_use_fit_only_activation_statistics() -> None:
    down_proj = torch.eye(3, dtype=torch.float64)
    square_sum = torch.tensor([4.0, 9.0, 1.0], dtype=torch.float64)
    max_abs = torch.tensor([10.0, 4.0, 2.0], dtype=torch.float64)

    rms = rank_rms_channels(down_proj, square_sum, route_count=1)
    tail = rank_tail_channels(down_proj, square_sum, max_abs, route_count=1, tail_lambda=0.5)

    assert rms.tolist() == [1, 0, 2]
    assert tail.tolist() == [0, 1, 2]


def test_conditional_residual_selection_prefers_output_energy_and_breaks_ties_by_id() -> None:
    down_proj = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    covariance = torch.diag(torch.tensor([1.0, 4.0, 1.0], dtype=torch.float64))

    selected = ramp_conditional_residual_selection(
        down_proj,
        covariance,
        keep_count=2,
        anchor_count=0,
        regularization=0.0,
    )

    assert selected.tolist() == [1, 0]


def test_conditional_residual_selection_avoids_redundant_correlated_channel() -> None:
    down_proj = torch.eye(3, dtype=torch.float64)
    covariance = torch.tensor(
        [
            [4.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )

    selected = ramp_conditional_residual_selection(
        down_proj,
        covariance,
        keep_count=2,
        anchor_count=0,
        regularization=0.0,
    )

    assert selected.tolist() == [0, 2]


def test_rank_limited_compensation_preserves_rank_one_solution() -> None:
    down_proj = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    keep = torch.tensor([0])

    effective, delta = fit_rank_limited_compensation(
        down_proj,
        covariance,
        keep,
        regularization=0.0,
        rank=1,
    )

    assert torch.allclose(delta, torch.tensor([[1.0], [2.0]], dtype=torch.float64), atol=1.0e-8)
    assert normalized_output_error(down_proj, covariance, keep, effective) < 1.0e-12


def test_shared_alpha_uses_median_validation_error() -> None:
    first = {alpha: 2.0 for alpha in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)}
    second = dict(first)
    third = dict(first)
    first[1.0e-4], second[1.0e-4], third[1.0e-4] = 0.4, 0.3, 0.5
    first[1.0e-3], second[1.0e-3], third[1.0e-3] = 0.2, 0.8, 0.7

    alpha, scores = choose_shared_alpha([first, second, third])

    assert alpha == 1.0e-4
    assert scores[1.0e-4] == 0.4


def test_primary_selection_excludes_activation_ablation() -> None:
    summary = {
        "conditional_activation": {"full": {"median_validation_error": 0.1}},
        "conditional_output": {"full": {"median_validation_error": 0.3}},
        "conditional_stable": {"full": {"median_validation_error": 0.2}},
    }

    assert choose_primary_selection(summary) == "conditional_stable"


def test_conditional_activation_selection_ignores_output_projection() -> None:
    covariance = torch.diag(torch.tensor([1.0, 4.0, 2.0], dtype=torch.float64))

    selected = conditional_activation_selection(covariance, keep_count=2, regularization=0.0)

    assert selected.tolist() == [1, 2]


def test_pairwise_output_correlation_avoids_duplicate_trajectory() -> None:
    down_proj = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    covariance = torch.tensor(
        [[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
    )

    selected = pairwise_output_correlation_selection(down_proj, covariance, keep_count=2)

    assert selected.tolist() == [0, 2]