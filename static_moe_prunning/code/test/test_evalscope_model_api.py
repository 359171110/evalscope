from __future__ import annotations

import hashlib

import pytest
import torch

from src.evalscope_model_api import (
    file_sha256,
    validate_static_merge_plan_artifact,
    validate_static_profile_artifacts,
)


def _write_artifacts(
    tmp_path,
    *,
    widths: torch.Tensor | None = None,
    sequence_length: int = 2048,
    calibration_free_purpose: str | None = None,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    profile_path = tmp_path / "profile.pt"
    channel_path = tmp_path / "channel.pt"
    profile_widths = torch.tensor([[1, 2]], dtype=torch.long) if widths is None else widths
    ranked = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    channel = {
        "model_path": str(model_path.resolve()),
        "split": "not_applicable" if calibration_free_purpose else "train",
        "sequence_length": 0 if calibration_free_purpose else sequence_length,
        "table": {
            0: {
                "ranked_indices": ranked,
                "block_relative_scores": torch.tensor([[1.0, 0.5], [1.0, 0.5]]),
                "block_coverage_scores": torch.tensor([[0.6, 0.4], [0.6, 0.4]]),
                "block_sizes": torch.tensor([2, 2], dtype=torch.long),
                "intermediate_size": 4,
            }
        },
    }
    if calibration_free_purpose:
        channel["purpose"] = calibration_free_purpose
    torch.save(channel, channel_path)
    profile = {
        "schema_version": 1,
        "method": "static_expert_test",
        "mode": "test",
        "model_path": str(model_path.resolve()),
        "profile_construction": "calibration_free" if calibration_free_purpose else "calibrated",
        "calibration_split": "not_applicable" if calibration_free_purpose else "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": 2,
        "total_blocks": int(profile_widths.sum().item()),
        "maximum_blocks": 4,
        "profile_widths": profile_widths,
        "profile_sha256": hashlib.sha256(
            profile_widths.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
        "cache_provenance": {"channel": {"sha256": file_sha256(channel_path)}},
    }
    if not calibration_free_purpose:
        profile["cache_provenance"]["calibration"] = {
            "split": "train",
            "sequence_length": sequence_length,
        }
    torch.save(profile, profile_path)
    return model_path, profile_path, channel_path


def test_artifact_preflight_accepts_audited_profile(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path)

    profile, channel, widths, table = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
        expected_profile_file_sha256=file_sha256(profile_path),
        expected_channel_file_sha256=file_sha256(channel_path),
    )

    assert widths.tolist() == [[1, 2]]
    assert set(table) == {0}
    assert profile["profile_file_sha256"] == file_sha256(profile_path)
    assert channel["channel_file_sha256"] == file_sha256(channel_path)


def test_artifact_preflight_accepts_matching_1024_sequence_length(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path, sequence_length=1024)

    _, channel, _, _ = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
    )

    assert channel["sequence_length"] == 1024


def test_artifact_preflight_accepts_calibration_free_channel_ranking(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path,
        calibration_free_purpose="aimer_weight_only_channel_ranking",
    )

    _, channel, _, _ = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
    )

    assert channel["purpose"] == "aimer_weight_only_channel_ranking"


def test_artifact_preflight_accepts_pure_pseudo_channel_ranking(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path,
        calibration_free_purpose="pure_pseudo_channel_ranking",
    )

    _, channel, _, _ = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
    )

    assert channel["purpose"] == "pure_pseudo_channel_ranking"


def test_artifact_preflight_rejects_unknown_calibration_free_channel_purpose(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path,
        calibration_free_purpose="unknown_weight_ranking",
    )

    with pytest.raises(ValueError, match="approved calibration-free"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def test_artifact_preflight_rejects_profile_channel_sequence_length_mismatch(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path, sequence_length=1024)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    profile["cache_provenance"]["calibration"]["sequence_length"] = 2048
    torch.save(profile, profile_path)

    with pytest.raises(ValueError, match="sequence lengths"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def test_artifact_preflight_rejects_fractional_profile_widths(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path, widths=torch.tensor([[1.0, 1.5]])
    )

    with pytest.raises(ValueError, match="integer"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def test_artifact_preflight_rejects_channel_hash_mismatch(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path)
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    channel["sequence_length"] = 1024
    torch.save(channel, channel_path)

    with pytest.raises(ValueError, match="SHA256"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def test_artifact_preflight_rejects_mismatched_per_layer_budget(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    profile["allocation_scope"] = "per_layer"
    profile["target_blocks_by_layer"] = [2]
    profile["actual_blocks_by_layer"] = [3]
    torch.save(profile, profile_path)

    with pytest.raises(ValueError, match="per-layer block budget"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def test_artifact_preflight_rejects_profile_channel_block_count_mismatch(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(tmp_path)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    profile["num_blocks"] = 3
    profile["maximum_blocks"] = 6
    torch.save(profile, profile_path)

    with pytest.raises(ValueError, match="block count"):
        validate_static_profile_artifacts(
            model_path=str(model_path),
            profile_path=profile_path,
            channel_cache_path=channel_path,
        )


def _write_merge_plan(
    model_path,
    profile_path,
    channel_path,
    *,
    retained: torch.Tensor | None = None,
):
    profile, channel, widths, table = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
    )
    merge_path = profile_path.parent / "merge.pt"
    merge = {
        "schema_version": 1,
        "purpose": "wick_down_projection_merge",
        "model_path": str(model_path.resolve()),
        "channel_file_sha256": channel["channel_file_sha256"],
        "layers": {
            0: {
                "retained_indices": (
                    torch.tensor([[0, 1], [3, 2]], dtype=torch.long)
                    if retained is None
                    else retained
                ),
                "pruned_indices": torch.tensor([[2, 3], [1, 0]], dtype=torch.long),
                "representative_indices": torch.tensor([[0, -1], [3, -1]], dtype=torch.long),
                "beta": torch.tensor([[0.25, 0.0], [0.5, 0.0]], dtype=torch.float32),
                "rejection_codes": torch.tensor([[0, 3], [0, 3]], dtype=torch.int8),
                "cumulative_relative_delta_norm": torch.tensor(
                    [[0.25, 0.0], [0.5, 0.0]], dtype=torch.float32
                ),
            }
        },
        "config": {
            "beta_max": 2.0,
            "representative_relative_delta_norm_max": 0.5,
        },
    }
    torch.save(merge, merge_path)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    profile.setdefault("cache_provenance", {})["merge_plan"] = {"sha256": file_sha256(merge_path)}
    torch.save(profile, profile_path)
    profile, channel, widths, table = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
    )
    return merge_path, profile, channel, widths, table


def test_merge_artifact_preflight_accepts_matching_wick_plan(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path,
        widths=torch.tensor([[1, 1]], dtype=torch.long),
        calibration_free_purpose="wick_weight_kernel_channel_ranking",
    )
    merge_path, profile, channel, widths, table = _write_merge_plan(
        model_path,
        profile_path,
        channel_path,
    )

    merge, merge_hash = validate_static_merge_plan_artifact(
        model_path=str(model_path),
        merge_plan_path=merge_path,
        profile=profile,
        channel=channel,
        widths=widths,
        table=table,
        expected_merge_plan_file_sha256=file_sha256(merge_path),
    )

    assert merge_hash == file_sha256(merge_path)
    assert merge["layers"][0]["retained_indices"].tolist() == [[0, 1], [3, 2]]


def test_merge_artifact_preflight_rejects_channel_prefix_mismatch(tmp_path) -> None:
    model_path, profile_path, channel_path = _write_artifacts(
        tmp_path,
        widths=torch.tensor([[1, 1]], dtype=torch.long),
        calibration_free_purpose="wick_weight_kernel_channel_ranking",
    )
    merge_path, profile, channel, widths, table = _write_merge_plan(
        model_path,
        profile_path,
        channel_path,
        retained=torch.tensor([[1, 0], [3, 2]], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="channel prefix"):
        validate_static_merge_plan_artifact(
            model_path=str(model_path),
            merge_plan_path=merge_path,
            profile=profile,
            channel=channel,
            widths=widths,
            table=table,
        )