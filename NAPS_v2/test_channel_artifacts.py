from __future__ import annotations

import pytest
import torch

from NAPS_v2.build_channel_artifacts import (
    channel_utility_components,
    nested_order,
    nested_rankings_by_width,
    route_weighted_channel_utility,
    select_channel_scores,
    validate_response_energy_table,
)
from NAPS_v2.compare_channel_holdout import ExpertLossRecord, summarize_records


def test_route_weighted_channel_utility_squares_final_route_weights() -> None:
    responses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    down = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    route_weights = torch.tensor([0.5, 2.0])

    utility = route_weighted_channel_utility(responses, down, route_weights)

    assert torch.allclose(utility, torch.tensor([36.25, 260.0]))


def test_channel_utility_components_reconstruct_primary_score() -> None:
    responses = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    down = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    route_weights = torch.tensor([0.5, 2.0])

    response_energy, down_energy = channel_utility_components(responses, down, route_weights)

    assert torch.allclose(response_energy, torch.tensor([36.25, 65.0]))
    assert torch.allclose(down_energy, torch.tensor([1.0, 4.0]))
    assert torch.allclose(
        response_energy * down_energy,
        route_weighted_channel_utility(responses, down, route_weights),
    )


def test_low_coverage_scores_shrink_toward_structural_utility() -> None:
    selection = select_channel_scores(
        calibrated=torch.tensor([100.0, 1.0]),
        structural=torch.tensor([1.0, 100.0]),
        aimer=torch.tensor([5.0, 4.0]),
        fit_token_count=2,
        fit_route_mass=0.2,
        min_fit_tokens=10,
        min_fit_route_mass=1.0,
    )

    assert selection.source == "real_token_structural_shrinkage"
    assert selection.coverage_confidence == pytest.approx(0.2)
    assert selection.scores[1] > selection.scores[0]


def test_low_coverage_without_structural_keeps_real_token_score() -> None:
    calibrated = torch.tensor([2.0, 9.0])
    selection = select_channel_scores(
        calibrated=calibrated,
        structural=None,
        aimer=torch.tensor([100.0, 1.0]),
        fit_token_count=1,
        fit_route_mass=0.1,
        min_fit_tokens=10,
        min_fit_route_mass=1.0,
    )

    assert selection.source == "real_token_undercovered_no_structural"
    assert torch.equal(selection.scores, calibrated)


def test_nested_order_uses_aimer_tie_break_and_places_zero_channels_last() -> None:
    order = nested_order(
        scores=torch.tensor([5.0, 5.0, torch.nan, 100.0]),
        zero_mask=torch.tensor([False, False, False, True]),
        tie_break_scores=torch.tensor([1.0, 2.0, 9.0, 100.0]),
    )

    assert order.tolist() == [1, 0, 2, 3]


def test_rankings_repeat_one_full_permutation_for_every_width() -> None:
    orders = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])

    rankings = nested_rankings_by_width(orders, (2, 3))

    assert rankings.shape == (2, 2, 4)
    assert torch.equal(rankings[:, 0], orders)
    assert torch.equal(rankings[:, 1], orders)
    assert set(rankings[0, 0, :2].tolist()) < set(rankings[0, 1, :3].tolist())


def test_rankings_reject_non_permutation_orders() -> None:
    with pytest.raises(ValueError, match="full channel permutation"):
        nested_rankings_by_width(torch.tensor([[0, 0, 1]]), (1, 2))


def test_response_energy_table_requires_exact_finite_nonnegative_shape() -> None:
    validate_response_energy_table(torch.ones(2, 3, 4), "fit", 2, 3, 4)

    with pytest.raises(ValueError, match="expected"):
        validate_response_energy_table(torch.ones(2, 3, 5), "fit", 2, 3, 4)
    with pytest.raises(ValueError, match="non-finite"):
        validate_response_energy_table(torch.tensor([[[float("nan")]]]), "fit", 1, 1, 1)
    with pytest.raises(ValueError, match="negative"):
        validate_response_energy_table(torch.tensor([[[-1.0]]]), "holdout", 1, 1, 1)


def test_holdout_summary_uses_global_output_energy_and_coverage_strata() -> None:
    summary = summarize_records([
        ExpertLossRecord(0, 0, 4, 1.0, 10.0, 1.0, 2.0),
        ExpertLossRecord(0, 1, 40, 2.0, 30.0, 9.0, 6.0),
    ])

    assert summary["candidate_global_weighted_loss"] == pytest.approx(0.25)
    assert summary["baseline_global_weighted_loss"] == pytest.approx(0.2)
    assert summary["relative_global_loss_change"] == pytest.approx(0.25)
    assert summary["candidate_expert_win_fraction"] == pytest.approx(0.5)
    assert summary["zero_to_7_experts"] == 1
    assert summary["zero_to_7_candidate_win_fraction"] == pytest.approx(1.0)
    assert summary["32_to_127_experts"] == 1
    assert summary["32_to_127_candidate_win_fraction"] == pytest.approx(0.0)