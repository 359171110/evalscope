from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.export_uniform_enp_qwen3_moe import file_sha256, validate_enp_artifacts


def _artifacts(tmp_path: Path, widths: torch.Tensor, zero_token_policy: str = "prune_uniform") -> tuple[dict, dict, Path]:
    channel_path = tmp_path / "channels.pt"
    channel_path.write_bytes(b"frozen-enp-channel-cache")
    channel_cache = {
        "purpose": "enp_tenp_signed_projection_channel_ranking",
        "split": "train",
        "test_metrics_used": False,
    }
    profile = {
        "method": "enp",
        "mode": "uniform_expert_neuron_pruning",
        "profile_construction": "calibrated",
        "test_metrics_used_for_profile": False,
        "channel_block_size": 64,
        "profile_widths": widths,
        "cache_provenance": {
            "calibration": {"protocol_name": "c1_wikitext_train_128x2048_seed42_screening_v1"},
            "channel": {"sha256": file_sha256(channel_path)},
        },
        "enp": {"zero_token_policy": zero_token_policy},
    }
    return profile, channel_cache, channel_path


@pytest.mark.parametrize(("retained_channels", "width"), [(576, 9), (384, 6)])
def test_validate_enp_artifacts_accepts_wikitext_uniform_targets(
    tmp_path: Path,
    retained_channels: int,
    width: int,
) -> None:
    profile, channel_cache, channel_path = _artifacts(tmp_path, torch.full((48, 128), width))

    actual = validate_enp_artifacts(
        profile,
        channel_cache,
        retained_channels=retained_channels,
        expected_protocol_name="c1_wikitext_train_128x2048_seed42_screening_v1",
        channel_cache_path=channel_path,
    )

    assert bool((actual == width).all())


def test_validate_enp_artifacts_rejects_keep_full_profile(tmp_path: Path) -> None:
    widths = torch.full((2, 3), 6)
    widths[0, 0] = 12
    profile, channel_cache, channel_path = _artifacts(tmp_path, widths, zero_token_policy="keep_full")

    with pytest.raises(ValueError, match="same requested width"):
        validate_enp_artifacts(
            profile,
            channel_cache,
            retained_channels=384,
            expected_protocol_name="c1_wikitext_train_128x2048_seed42_screening_v1",
            channel_cache_path=channel_path,
        )