from __future__ import annotations

import json
from pathlib import Path

import torch

from Random.build_random_artifacts import file_sha256, main as build_main


def write_index_only_checkpoint(model_path: Path, family: str) -> None:
    model_path.mkdir()
    if family == "qwen3":
        config = {
            "model_type": "qwen3_moe",
            "hidden_size": 4,
            "hidden_act": "silu",
            "moe_intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        weight_map = {
            "model.layers.0.mlp.gate.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.1.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.1.up_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.1.down_proj.weight": "model.safetensors",
        }
    else:
        config = {
            "model_type": "deepseek_v2",
            "hidden_size": 4,
            "hidden_act": "silu",
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "n_shared_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
        }
        weight_map = {
            "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
            "model.layers.1.mlp.gate.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.up_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.0.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.1.gate_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.1.up_proj.weight": "model.safetensors",
            "model.layers.1.mlp.experts.1.down_proj.weight": "model.safetensors",
            "model.layers.1.mlp.shared_experts.gate_proj.weight": "model.safetensors",
        }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def run_build(tmp_path: Path, monkeypatch, family: str, retained: int) -> tuple[Path, Path]:
    model_path = tmp_path / f"{family}-model"
    write_index_only_checkpoint(model_path, family)
    artifact = tmp_path / f"{family}-artifacts"
    channel = artifact / "random_rankings.pt"
    profile = artifact / f"random_{retained}ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_random_artifacts",
            "--model-path",
            str(model_path),
            "--output-channel-cache",
            str(channel),
            "--output-profile",
            str(profile),
            "--retained-channels",
            str(retained),
            "--seed",
            "42",
        ],
    )
    assert build_main() == 0
    return channel, profile


def test_build_is_data_free_and_calibration_free(tmp_path: Path, monkeypatch) -> None:
    channel_path, profile_path = run_build(tmp_path, monkeypatch, "qwen3", 64)
    channel = torch.load(channel_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    assert channel["purpose"] == "random_channel_ranking"
    assert channel["random"]["data_free"] is True
    assert channel["split"] == "not_applicable"
    assert profile["profile_construction"] == "calibration_free"
    assert profile["calibration_split"] == "not_applicable"
    assert profile["retained_channels"] == 64
    assert int(profile["profile_widths"][0, 0].item()) == 1
    retained = json.loads(profile_path.with_name("random_retained_64ch.json").read_text(encoding="utf-8"))
    assert retained["seed"] == 42
    assert len(retained["retained_indices"]["0"][0]) == 64


def test_25_and_50_share_the_same_permutation_cache(tmp_path: Path, monkeypatch) -> None:
    channel_50, _ = run_build(tmp_path, monkeypatch, "qwen3", 128)
    original = channel_50.read_bytes()
    original_sha = file_sha256(channel_50)
    artifact = tmp_path / "qwen3-artifacts"
    profile_25 = artifact / "random_192ch.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_random_artifacts",
            "--model-path",
            str(tmp_path / "qwen3-model"),
            "--output-channel-cache",
            str(channel_50),
            "--output-profile",
            str(profile_25),
            "--retained-channels",
            "192",
            "--seed",
            "42",
        ],
    )
    assert build_main() == 0
    assert channel_50.read_bytes() == original
    assert file_sha256(channel_50) == original_sha
    profile = torch.load(profile_25, map_location="cpu", weights_only=True)
    assert profile["retained_channels"] == 192
    assert profile["cache_provenance"]["channel"]["sha256"] == original_sha


def test_deepseek_profile_skips_the_dense_first_layer(tmp_path: Path, monkeypatch) -> None:
    _, profile_path = run_build(tmp_path, monkeypatch, "deepseek", 32)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    assert profile["layer_ids"] == [1]
    assert profile["num_layers"] == 1
    assert profile["num_experts"] == 2
