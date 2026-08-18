from __future__ import annotations

import pytest
import torch

from NAPS_v2.build_channel_profile import (
    build_budgeted_profile,
    build_uniform_profile,
    validate_nested_rankings,
)
from NAPS_v2.compare_channel_profiles import decode_profile_widths, ranking_prefix
from NAPS_v2.export_naps_v2_heterogeneous_checkpoint import (
    export_method,
    load_width_metadata,
    selected_indices,
)


def rankings_payload() -> dict:
    orders = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
    by_width = orders.unsqueeze(1).repeat(1, 2, 1)
    response_energy = torch.tensor([[3.0, 1.0, 8.0, 2.0], [1.0, 9.0, 2.0, 4.0]])
    return {
        "schema_version": 4,
        "model_path": "/model",
        "model_family": "test",
        "source_intermediate_size": 4,
        "channel_alignment": 1,
        "width_options": (2, 3),
        "ranking_is_nested": True,
        "capture_path": "/capture.pt",
        "capture_sha256": "capture-hash",
        "calibration": {"protocol_name": "test-calibration"},
        "model_provenance": {"config_sha256": "config-hash"},
        "table": {
            0: {
                "ranked_indices": orders,
                "ranked_indices_by_width": by_width,
                "width_options": torch.tensor([2, 3]),
                "route_weighted_response_energy": response_energy,
                "down_channel_energy": torch.ones_like(response_energy),
                "coverage_confidence": torch.ones(2),
                "score_sources": ["real_token_route_weighted"] * 2,
            },
            1: {
                "ranked_indices": orders.flip(0),
                "ranked_indices_by_width": by_width.flip(0),
                "width_options": torch.tensor([2, 3]),
                "route_weighted_response_energy": response_energy.flip(0),
                "down_channel_energy": torch.ones_like(response_energy),
                "coverage_confidence": torch.ones(2),
                "score_sources": ["real_token_route_weighted"] * 2,
            },
        },
    }


def budgeted_rankings_payload() -> dict:
    orders = torch.arange(5).repeat(4, 1)
    by_width = orders.unsqueeze(1).repeat(1, 3, 1)
    utility = torch.tensor([
        [9.0, 8.0, 1.0, 0.1, 0.0],
        [9.0, 8.0, 5.0, 4.0, 0.0],
        [9.0, 8.0, 3.0, 2.0, 0.0],
        [9.0, 8.0, 0.5, 10.0, 0.0],
    ])
    return {
        "schema_version": 4,
        "model_path": "/model",
        "model_family": "test",
        "source_intermediate_size": 5,
        "channel_alignment": 1,
        "width_options": (2, 3, 4),
        "ranking_is_nested": True,
        "capture_path": "/capture.pt",
        "capture_sha256": "capture-hash",
        "calibration": {"protocol_name": "test-calibration"},
        "model_provenance": {"config_sha256": "config-hash"},
        "table": {
            0: {
                "ranked_indices": orders,
                "ranked_indices_by_width": by_width,
                "width_options": torch.tensor([2, 3, 4]),
                "route_weighted_response_energy": utility,
                "down_channel_energy": torch.ones_like(utility),
                "coverage_confidence": torch.tensor([1.0, 1.0, 0.5, 1.0]),
                "score_sources": [
                    "real_token_route_weighted",
                    "real_token_route_weighted",
                    "real_token_structural_shrinkage",
                    "real_token_route_weighted",
                ],
            },
        },
    }


def test_uniform_profile_is_exporter_compatible_and_budget_exact() -> None:
    profile = build_uniform_profile(rankings_payload(), uniform_width=3)

    assert profile["schema_version"] == 4
    assert profile["profile_widths"].shape == (2, 2)
    assert torch.equal(profile["profile_widths"], torch.full((2, 2), 3))
    assert profile["padded_intermediate_size"] == 3
    assert profile["total_blocks"] == 12
    assert profile["maximum_blocks"] == 12
    assert profile["ranking_is_nested"]
    assert profile["capture_sha256"] == "capture-hash"
    assert export_method(profile) == "channel_calibrated_nested_mask_padded"


def test_budgeted_profile_uses_fit_marginals_and_preserves_exact_layer_budget() -> None:
    profile = build_budgeted_profile(
        budgeted_rankings_payload(),
        small_width=2,
        medium_width=3,
        large_width=4,
    )

    widths = profile["profile_widths"] * profile["channel_block_size"]
    assert widths.tolist() == [[2, 3, 3, 4]]
    assert widths.sum(dim=1).tolist() == [12]
    assert profile["padded_intermediate_size"] == 4
    assert profile["holdout_used_for_profile"] is False
    assert profile["allocation_diagnostics"][0]["small_experts"] == 1
    assert profile["allocation_diagnostics"][0]["large_experts"] == 1
    assert profile["allocation_diagnostics"][0]["fit_objective_gain"] == pytest.approx(9.0)


def test_budgeted_profile_keeps_medium_when_no_balanced_transfer_is_positive() -> None:
    rankings = budgeted_rankings_payload()
    utility = rankings["table"][0]["route_weighted_response_energy"]
    utility[:, 2] = 10.0
    utility[:, 3] = 1.0

    profile = build_budgeted_profile(rankings, 2, 3, 4)

    widths = profile["profile_widths"] * profile["channel_block_size"]
    assert torch.equal(widths, torch.full((1, 4), 3))
    assert profile["allocation_diagnostics"][0]["fit_objective_gain"] == pytest.approx(0.0)


def test_budgeted_profile_rejects_asymmetric_width_steps() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        build_budgeted_profile(budgeted_rankings_payload(), 2, 3, 5)


def test_profile_comparator_decodes_widths_and_uses_nested_prefix() -> None:
    rankings = budgeted_rankings_payload()
    profile = build_budgeted_profile(rankings, 2, 3, 4)

    widths = decode_profile_widths(profile, layer_count=1, num_experts=4, source_width=5)

    assert widths.tolist() == [[2, 3, 3, 4]]
    assert torch.equal(ranking_prefix(rankings, 0, 3, 4), torch.tensor([0, 1, 2, 3]))


def test_profile_comparator_rejects_width_outside_options() -> None:
    profile = build_budgeted_profile(budgeted_rankings_payload(), 2, 3, 4)
    profile["profile_widths"][0, 0] = 5

    with pytest.raises(ValueError, match="absent"):
        decode_profile_widths(profile, layer_count=1, num_experts=4, source_width=5)


def test_profile_rejects_width_specific_non_nested_orders() -> None:
    rankings = rankings_payload()
    rankings["table"][1]["ranked_indices_by_width"][0, 1] = torch.tensor([1, 0, 3, 2])

    with pytest.raises(ValueError, match="do not share one nested order"):
        validate_nested_rankings(rankings)


def test_profile_rejects_non_permutation_order() -> None:
    rankings = rankings_payload()
    rankings["table"][0]["ranked_indices_by_width"][0, :, -1] = 2

    with pytest.raises(ValueError, match="non-permutation"):
        validate_nested_rankings(rankings)


def test_profile_rejects_unavailable_uniform_width() -> None:
    with pytest.raises(ValueError, match="not present"):
        build_uniform_profile(rankings_payload(), uniform_width=1)


def test_profile_and_rankings_round_trip_through_exporter_metadata(tmp_path) -> None:
    rankings = rankings_payload()
    profile = build_uniform_profile(rankings, uniform_width=3)
    torch.save(rankings, tmp_path / "rankings.pt")
    torch.save(profile, tmp_path / "profile.pt")

    loaded_rankings, loaded_profile, widths, width_options, padded_width = load_width_metadata(tmp_path)

    assert width_options == (2, 3)
    assert padded_width == 3
    assert torch.equal(widths, torch.full((2, 2), 3))
    assert torch.equal(selected_indices(loaded_rankings, 0, 0, 3), torch.tensor([2, 0, 3]))
    assert loaded_profile["calibration"]["protocol_name"] == "test-calibration"