from __future__ import annotations

from pathlib import Path

import torch

from scripts.build_aimer_profile import _build_topology_channel_payload, build_aimer_profile_payload
from src.evalscope_model_api import file_sha256, validate_static_profile_artifacts
from src.static_expert_pruning import validate_static_profile_payload


def test_aimer_profile_retains_highest_keep_scores_without_calibration() -> None:
    scores = torch.tensor([[4.0, 1.0, 3.0, 2.0], [1.0, 4.0, 2.0, 3.0]])
    profile = build_aimer_profile_payload(
        model_path=Path("/models/qwen3"),
        layer_ids=[0, 1],
        keep_scores=scores,
        target_pruning_ratio=0.5,
        num_blocks=12,
        top_k=2,
        score_file_sha256="a" * 64,
        channel_file_sha256="b" * 64,
        aimer_identity={"commit": "test"},
    )

    widths = validate_static_profile_payload(profile)
    assert widths.tolist() == [[12, 0, 12, 0], [0, 12, 0, 12]]
    assert profile["profile_construction"] == "calibration_free"
    assert profile["calibration_split"] == "not_applicable"
    assert profile["total_blocks"] == 48
    assert profile["actual_structural_pruning_ratio"] == 0.5


def test_aimer_profile_and_topology_channel_pass_artifact_preflight(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    channel_path = tmp_path / "aimer_topology_channels.pt"
    profile_path = tmp_path / "aimer_50pct_per_layer.pt"
    channel = _build_topology_channel_payload(
        model_path=model_path,
        layer_ids=[0],
        num_experts=4,
        intermediate_size=8,
        block_size=4,
    )
    torch.save(channel, channel_path)
    profile = build_aimer_profile_payload(
        model_path=model_path,
        layer_ids=[0],
        keep_scores=torch.tensor([[4.0, 1.0, 3.0, 2.0]]),
        target_pruning_ratio=0.5,
        num_blocks=2,
        top_k=2,
        score_file_sha256="a" * 64,
        channel_file_sha256=file_sha256(channel_path),
        aimer_identity={"commit": "test"},
    )
    torch.save(profile, profile_path)

    validated_profile, validated_channel, widths, table = validate_static_profile_artifacts(
        model_path=str(model_path),
        profile_path=profile_path,
        channel_cache_path=channel_path,
        expected_profile_file_sha256=file_sha256(profile_path),
        expected_channel_file_sha256=file_sha256(channel_path),
    )

    assert validated_profile["method"] == "aimer"
    assert validated_channel["purpose"] == "runtime_topology_only"
    assert widths.tolist() == [[2, 0, 2, 0]]
    assert table[0].block_sizes.tolist() == [4, 4]