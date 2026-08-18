from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from Wanda.build_wanda_artifacts import clone_uniform_profile, file_sha256, main as build_main


def _profile(width: int, block_size: int, retained: int, channel_sha: str) -> dict:
    retained_blocks = retained // block_size
    widths = torch.full((2, 3), retained_blocks, dtype=torch.long)
    return {
        "schema_version": 1,
        "method": "wanda_grouped",
        "profile_construction": "calibrated",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [0, 1],
        "num_layers": 2,
        "num_experts": 3,
        "num_blocks": width // block_size,
        "channel_block_size": block_size,
        "intermediate_size": width,
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": int(widths.sum().item()),
        "maximum_blocks": 2 * 3 * (width // block_size),
        "target_pruning_ratio": 1.0 - retained / width,
        "actual_structural_pruning_ratio": 1.0 - retained / width,
        "retained_channels": retained,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "cache_provenance": {
            "channel": {
                "path": "/tmp/channels.pt",
                "sha256": channel_sha,
                "role": "wanda_ranking",
            }
        },
    }


def test_clone_uniform_profile_keeps_gemma4_512_alignment() -> None:
    source = _profile(704, 32, 352, "abc")
    cloned = clone_uniform_profile(source, 512, 0.25)
    assert cloned["retained_channels"] == 512
    assert cloned["target_pruning_ratio"] == 0.25
    assert abs(cloned["actual_structural_pruning_ratio"] - (1.0 - 512 / 704)) < 1e-12
    assert int(cloned["profile_widths"][0, 0].item()) == 16
    assert cloned["total_blocks"] == 2 * 3 * 16
    assert source["retained_channels"] == 352


def test_clone_from_profile_does_not_rewrite_channel_cache(tmp_path: Path, monkeypatch) -> None:
    channel_path = tmp_path / "channels.pt"
    channel_path.write_bytes(b"frozen-wanda-ranking")
    channel_sha = file_sha256(channel_path)
    source_path = tmp_path / "wanda_50pct_per_layer.pt"
    torch.save(_profile(768, 64, 384, channel_sha), source_path)
    output_path = tmp_path / "wanda_25pct_per_layer.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_wanda_artifacts",
            "--from-profile",
            str(source_path),
            "--channel-cache",
            str(channel_path),
            "--output-profile",
            str(output_path),
            "--retained-channels",
            "576",
            "--target-pruning-ratio",
            "0.25",
        ],
    )
    assert build_main() == 0
    assert channel_path.read_bytes() == b"frozen-wanda-ranking"
    assert file_sha256(channel_path) == channel_sha
    cloned = torch.load(output_path, map_location="cpu", weights_only=True)
    assert cloned["retained_channels"] == 576
    assert cloned["target_pruning_ratio"] == 0.25
    assert cloned["cache_provenance"]["channel"]["sha256"] == channel_sha
    summary = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["retained_channels"] == 576
