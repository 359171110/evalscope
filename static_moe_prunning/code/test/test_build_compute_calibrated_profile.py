from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from scripts import build_compute_calibrated_profile
from src.static_expert_pruning import validate_static_profile_payload


def test_compute_builder_merges_all_train_only_safety_floors() -> None:
    source = {
        "risk_floor": {
            "selected_experts": [
                {"layer": 0, "expert": 0, "min_width": 1}
            ]
        },
        "output_saliency_risk_floor": {
            "selected_experts": [
                {"layer": 0, "expert": 1, "min_width": 2}
            ]
        },
        "unique_contribution_risk_floor": {
            "selected_experts": [
                {"layer": 0, "expert": 2, "min_width": 3}
            ]
        },
        "frontier_committee_regret_floor": {
            "selected_experts": [
                {"layer": 0, "expert": 3, "min_width": 4}
            ]
        },
    }

    floors = build_compute_calibrated_profile._risk_floors(source, (1, 4))

    assert floors is not None
    assert floors.tolist() == [[1, 2, 3, 4]]


def test_builder_preserves_exact_structure_and_train_only_provenance(
    monkeypatch, tmp_path
) -> None:
    channel_path = tmp_path / "channel.pt"
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    channel = {
        "split": "train",
        "sequence_length": 2048,
        "table": {
            0: {
                "block_coverage_scores": torch.tensor(
                    [[10.0, 9.0], [2.0, 1.0]]
                )
            }
        },
        "route_counts": {0: torch.tensor([10.0, 1.0])},
    }
    torch.save(channel, channel_path)
    source = {
        "schema_version": 1,
        "method": "static_expert_route_rms",
        "mode": "route_rms",
        "model_path": "/models/test",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": 2,
        "channel_block_size": 64,
        "target_pruning_ratio": 0.5,
        "total_blocks": 2,
        "maximum_blocks": 4,
        "min_blocks_per_expert": 0,
        "profile_widths": torch.tensor([[2, 0]]),
        "output_saliency_factor": torch.tensor([[0.5, 1.5]]),
        "unique_contribution_score": torch.tensor([[0.25, 0.75]]),
        "co_route_uniqueness_folds": torch.tensor([[[0.5, 0.5]]]),
        "frontier_committee_regret_score": torch.tensor([[1.0, 0.5]]),
        "cache_provenance": {
            "channel": {
                "path": str(channel_path),
                "sha256": build_compute_calibrated_profile.file_sha256(channel_path),
            }
        },
    }
    torch.save(source, source_path)
    args = Namespace(
        source_profile=source_path,
        output_profile=output_path,
        target_routed_pruning_ratio=1.0 - 2.0 / 22.0,
        compute_route_cache=[],
        search_iterations=32,
    )
    monkeypatch.setattr(build_compute_calibrated_profile, "parse_args", lambda: args)

    assert build_compute_calibrated_profile.main() == 0
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    widths = validate_static_profile_payload(payload)

    assert widths.tolist() == [[0, 2]]
    assert int(widths.sum()) == 2
    assert payload["compute_calibration"]["split"] == "train"
    assert payload["compute_calibration"]["test_metrics_used"] is False
    assert payload["compute_calibration"]["route_distribution_fold_count"] == 1
    assert output_path.with_suffix(".json").is_file()


def test_builder_enforces_per_fold_compute_noninferiority(
    monkeypatch, tmp_path
) -> None:
    channel_path = tmp_path / "channel.pt"
    source_path = tmp_path / "source.pt"
    reference_path = tmp_path / "reference.pt"
    output_path = tmp_path / "output.pt"
    fold_paths = [tmp_path / "fold0.pt", tmp_path / "fold1.pt"]
    channel = {
        "split": "train",
        "sequence_length": 2048,
        "table": {
            0: {
                "block_coverage_scores": torch.tensor(
                    [[10.0, 9.0], [8.0, 7.0]]
                )
            }
        },
        "route_counts": {0: torch.tensor([10.0, 1.0])},
    }
    torch.save(channel, channel_path)
    for path, counts in zip(
        fold_paths,
        (torch.tensor([10.0, 1.0]), torch.tensor([1.0, 10.0])),
    ):
        torch.save(
            {
                "split": "train",
                "sequence_length": 2048,
                "route_counts": {0: counts},
            },
            path,
        )
    base = {
        "schema_version": 1,
        "method": "static_expert_route_rms",
        "mode": "route_rms",
        "model_path": "/models/test",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": 2,
        "channel_block_size": 64,
        "target_pruning_ratio": 0.5,
        "total_blocks": 2,
        "maximum_blocks": 4,
        "min_blocks_per_expert": 0,
        "cache_provenance": {
            "channel": {
                "path": str(channel_path),
                "sha256": build_compute_calibrated_profile.file_sha256(channel_path),
            }
        },
    }
    source = {**base, "profile_widths": torch.tensor([[2, 0]])}
    reference = {**base, "profile_widths": torch.tensor([[1, 1]])}
    torch.save(source, source_path)
    torch.save(reference, reference_path)
    args = Namespace(
        source_profile=source_path,
        output_profile=output_path,
        target_routed_pruning_ratio=0.5,
        compute_route_cache=fold_paths,
        search_iterations=32,
        route_aggregation="mean",
        cvar_alpha=0.75,
        layer_entropy_gamma=0.0,
        compute_noninferiority_reference_profile=reference_path,
        fold_dual_iterations=256,
        fold_dual_step_size=2.0,
        fold_relative_tolerance=1.0e-10,
    )
    monkeypatch.setattr(build_compute_calibrated_profile, "parse_args", lambda: args)

    assert build_compute_calibrated_profile.main() == 0
    payload = torch.load(output_path, map_location="cpu", weights_only=True)

    assert payload["profile_widths"].tolist() == [[1, 1]]
    audit = payload["compute_calibration"]["per_fold_noninferiority"]
    assert audit["all_fold_constraints_satisfied"] is True
    assert max(audit["candidate_minus_reference_retained_cost_by_fold"]) <= 1.0e-12


def test_builder_records_reference_centered_route_envelope(
    monkeypatch, tmp_path
) -> None:
    channel_path = tmp_path / "channel.pt"
    source_path = tmp_path / "source.pt"
    reference_path = tmp_path / "reference.pt"
    output_path = tmp_path / "output.pt"
    fold_paths = [tmp_path / "fold0.pt", tmp_path / "fold1.pt"]
    channel = {
        "split": "train",
        "sequence_length": 2048,
        "table": {
            0: {
                "block_coverage_scores": torch.tensor(
                    [[10.0, 9.0], [8.0, 7.0], [1.0, 0.5]]
                )
            }
        },
        "route_counts": {0: torch.tensor([4.0, 5.0, 1.0])},
    }
    torch.save(channel, channel_path)
    for path, counts in zip(
        fold_paths,
        (torch.tensor([4.0, 5.0, 1.0]), torch.tensor([2.0, 3.0, 5.0])),
    ):
        torch.save(
            {
                "split": "train",
                "sequence_length": 2048,
                "route_counts": {0: counts},
            },
            path,
        )
    base = {
        "schema_version": 1,
        "method": "static_expert_route_rms",
        "mode": "route_rms",
        "model_path": "/models/test",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0],
        "num_layers": 1,
        "num_experts": 3,
        "num_blocks": 2,
        "channel_block_size": 64,
        "target_pruning_ratio": 2.0 / 3.0,
        "total_blocks": 2,
        "maximum_blocks": 6,
        "min_blocks_per_expert": 0,
        "cache_provenance": {
            "channel": {
                "path": str(channel_path),
                "sha256": build_compute_calibrated_profile.file_sha256(channel_path),
            }
        },
    }
    torch.save({**base, "profile_widths": torch.tensor([[2, 0, 0]])}, source_path)
    torch.save({**base, "profile_widths": torch.tensor([[1, 1, 0]])}, reference_path)
    args = Namespace(
        source_profile=source_path,
        output_profile=output_path,
        target_routed_pruning_ratio=0.5,
        compute_route_cache=fold_paths,
        search_iterations=32,
        route_aggregation="mean",
        cvar_alpha=0.75,
        layer_entropy_gamma=0.0,
        compute_noninferiority_reference_profile=reference_path,
        fold_dual_iterations=256,
        fold_dual_step_size=2.0,
        fold_relative_tolerance=1.0e-10,
        route_envelope_expansion=0.0,
    )
    monkeypatch.setattr(build_compute_calibrated_profile, "parse_args", lambda: args)

    assert build_compute_calibrated_profile.main() == 0
    payload = torch.load(output_path, map_location="cpu", weights_only=True)

    assert payload["profile_widths"].tolist() == [[1, 1, 0]]
    audit = payload["compute_calibration"]["per_fold_noninferiority"]
    assert audit["route_envelope"]["expansion"] == pytest.approx(0.0)
    assert audit["route_envelope"]["constraint_relative_delta"] <= 1.0e-12
