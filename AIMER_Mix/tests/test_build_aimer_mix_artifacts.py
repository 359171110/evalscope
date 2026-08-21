from __future__ import annotations

import json
from pathlib import Path

import torch

from AIMER_Mix.build_aimer_mix_artifacts import file_sha256, main as build_main
from AIMER_Mix.tests.helpers import expected_ranking, write_checkpoint


def run_build(tmp_path: Path, monkeypatch, family: str, retained: int) -> tuple[Path, Path]:
    model_path = tmp_path / f"{family}-model"
    write_checkpoint(model_path, family)
    artifact = tmp_path / f"{family}-artifacts"
    channel = artifact / "aimer_mix_rankings.pt"
    profile = artifact / f"aimer_mix_{retained}ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aimer_mix_artifacts",
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


def test_build_is_data_free_and_ranks_by_mix(tmp_path: Path, monkeypatch) -> None:
    channel_path, profile_path = run_build(tmp_path, monkeypatch, "qwen3", 64)
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    assert channel["purpose"] == "aimer_mix_ranking"
    assert channel["method"] == "aimer_mix"
    assert channel["aimer_mix"]["data_free"] is True
    assert channel["aimer_mix"]["weight_only"] is True
    assert channel["aimer_mix"]["energy_mode"] == "geom"
    assert channel["aimer_mix"]["compensation"] == "none"
    assert channel["split"] == "not_applicable"
    assert profile["profile_construction"] == "calibration_free"
    assert profile["method"] == "aimer_mix"
    assert profile["retained_channels"] == 64
    assert channel["table"][0]["mean_alpha"] == 1.0
    assert channel["aimer_mix"]["mean_alpha"] == 1.0
    model_path = tmp_path / "qwen3-model"
    from safetensors import safe_open
    with safe_open(model_path / "model.safetensors", framework="pt", device="cpu") as handle:
        gate = handle.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
        up = handle.get_tensor("model.layers.0.mlp.experts.0.up_proj.weight")
        down = handle.get_tensor("model.layers.0.mlp.experts.0.down_proj.weight")
        gate1 = handle.get_tensor("model.layers.0.mlp.experts.1.gate_proj.weight")
        up1 = handle.get_tensor("model.layers.0.mlp.experts.1.up_proj.weight")
        down1 = handle.get_tensor("model.layers.0.mlp.experts.1.down_proj.weight")
    ranked = channel["table"][0]["ranked_indices"]
    assert torch.equal(ranked[0], expected_ranking(gate, up, down))
    assert torch.equal(ranked[1], expected_ranking(gate1, up1, down1))
    retained = json.loads(profile_path.with_name("aimer_mix_retained_64ch.json").read_text(encoding="utf-8"))
    assert retained["retained_indices"]["0"][0] == expected_ranking(gate, up, down)[:64].tolist()


def test_25_and_50_share_the_same_mix_cache(tmp_path: Path, monkeypatch) -> None:
    channel_50, _ = run_build(tmp_path, monkeypatch, "qwen3", 128)
    original = channel_50.read_bytes()
    original_sha = file_sha256(channel_50)
    artifact = tmp_path / "qwen3-artifacts"
    profile_25 = artifact / "aimer_mix_192ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aimer_mix_artifacts",
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
    assert "mean_alpha" in channel["aimer_mix"]
