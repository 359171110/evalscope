from __future__ import annotations

import json
from pathlib import Path

import torch

from Product.build_product_artifacts import file_sha256, main as build_main
from Product.tests.helpers import expected_ranking, write_checkpoint


def run_build(tmp_path: Path, monkeypatch, family: str, retained: int) -> tuple[Path, Path]:
    model_path = tmp_path / f"{family}-model"
    write_checkpoint(model_path, family)
    artifact = tmp_path / f"{family}-artifacts"
    channel = artifact / "product_rankings.pt"
    profile = artifact / f"product_{retained}ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_artifacts",
            "--model-path",
            str(model_path),
            "--output-channel-cache",
            str(channel),
            "--output-profile",
            str(profile),
            "--retained-channels",
            str(retained),
        ],
    )
    assert build_main() == 0
    return channel, profile


def test_build_is_data_free_weight_only_and_ranks_by_product(tmp_path: Path, monkeypatch) -> None:
    channel_path, profile_path = run_build(tmp_path, monkeypatch, "qwen3", 64)
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    assert channel["purpose"] == "product_channel_ranking"
    assert channel["product"]["data_free"] is True
    assert channel["product"]["weight_only"] is True
    assert channel["product"]["down_used_for_ranking"] is False
    assert channel["split"] == "not_applicable"
    assert profile["profile_construction"] == "calibration_free"
    assert profile["method"] == "product"
    assert profile["retained_channels"] == 64
    ranked = channel["table"][0]["ranked_indices"]
    assert torch.equal(ranked[0], expected_ranking(256, 0))
    assert torch.equal(ranked[1], expected_ranking(256, 1))
    retained = json.loads(profile_path.with_name("product_retained_64ch.json").read_text(encoding="utf-8"))
    assert retained["retained_indices"]["0"][0] == expected_ranking(256, 0)[:64].tolist()


def test_25_and_50_share_the_same_product_cache(tmp_path: Path, monkeypatch) -> None:
    channel_50, _ = run_build(tmp_path, monkeypatch, "qwen3", 128)
    original = channel_50.read_bytes()
    original_sha = file_sha256(channel_50)
    artifact = tmp_path / "qwen3-artifacts"
    profile_25 = artifact / "product_192ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_artifacts",
            "--model-path",
            str(tmp_path / "qwen3-model"),
            "--output-channel-cache",
            str(channel_50),
            "--output-profile",
            str(profile_25),
            "--retained-channels",
            "192",
        ],
    )
    assert build_main() == 0
    assert channel_50.read_bytes() == original
    assert file_sha256(channel_50) == original_sha
    profile = torch.load(profile_25, map_location="cpu", weights_only=True)
    assert profile["retained_channels"] == 192
    assert profile["cache_provenance"]["channel"]["sha256"] == original_sha


def test_deepseek_profile_skips_the_dense_first_layer(tmp_path: Path, monkeypatch) -> None:
    channel_path, profile_path = run_build(tmp_path, monkeypatch, "deepseek", 32)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    assert profile["layer_ids"] == [1]
    assert profile["num_layers"] == 1
    assert set(map(int, channel["table"])) == {1}
    assert 0 not in channel["table"]
