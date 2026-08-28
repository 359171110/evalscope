from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from CSP.build_csp_artifacts import main as build_main
from CSP.export_csp_checkpoint import main as export_main
from CSP.tests.helpers import write_checkpoint


def build_and_export(tmp_path: Path, monkeypatch, family: str) -> tuple[Path, Path, Path, dict[str, torch.Tensor]]:
    model = tmp_path / f"{family}-model"
    source = write_checkpoint(model, family)
    artifact = tmp_path / f"{family}-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "profile.pt"
    retained = 64 if family in {"qwen3", "qwen3.6"} else 32
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model), "--output-channel-cache", str(cache),
        "--output-profile", str(profile), "--retained-channels", str(retained),
    ])
    assert build_main() == 0
    output = tmp_path / f"{family}-output"
    monkeypatch.setattr("sys.argv", [
        "export_csp_checkpoint", "--model-path", str(model), "--profile", str(profile),
        "--channel-cache", str(cache), "--output-dir", str(output),
    ])
    assert export_main() == 0
    return output, cache, profile, source


@pytest.mark.parametrize("family", ["qwen3", "gemma4", "qwen3.6", "deepseek"])
def test_build_and_export_all_requested_families(tmp_path: Path, monkeypatch, family: str) -> None:
    output, cache_path, profile_path, _ = build_and_export(tmp_path, monkeypatch, family)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert cache["purpose"] == "csp_channel_ranking"
    assert cache["csp"]["data_free"] is True
    assert cache["csp"]["weight_only"] is True
    assert cache["csp"]["canonicalization"] is False
    assert profile["method"] == "csp"
    assert profile["retained_channels"] == (64 if family in {"qwen3", "qwen3.6"} else 32)
    assert manifest["method"] == "csp"
    assert manifest["retained_channels"] == (64 if family in {"qwen3", "qwen3.6"} else 32)


def test_gemma4_build_uses_expert_input_scale_and_preserves_non_routed_norm(tmp_path: Path, monkeypatch) -> None:
    output, cache_path, _, source = build_and_export(tmp_path, monkeypatch, "gemma4")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    assert cache["csp"]["input_scale_mode"] == "gemma4_pre_feedforward_layernorm_2"
    norm_name = "model.language_model.layers.0.pre_feedforward_layernorm_2.weight"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor(norm_name), source[norm_name])
        assert handle.get_tensor("model.language_model.layers.0.experts.gate_up_proj").shape == (2, 64, 4)


def test_qwen36_shared_and_deepseek_dense_shared_tensors_are_preserved(tmp_path: Path, monkeypatch) -> None:
    qwen_output, _, _, _ = build_and_export(tmp_path, monkeypatch, "qwen3.6")
    with safe_open(qwen_output / "model.safetensors", framework="pt", device="cpu") as handle:
        assert handle.get_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj").shape == (2, 128, 4)
    deep_output, _, _, deep_source = build_and_export(tmp_path, monkeypatch, "deepseek")
    with safe_open(deep_output / "model.safetensors", framework="pt", device="cpu") as handle:
        dense = "model.layers.0.mlp.gate_proj.weight"
        shared = "model.layers.1.mlp.shared_experts.gate_proj.weight"
        assert torch.equal(handle.get_tensor(dense), deep_source[dense])
        assert torch.equal(handle.get_tensor(shared), deep_source[shared])
    config = json.loads((deep_output / "config.json").read_text(encoding="utf-8"))
    assert config["moe_intermediate_size"] == 32
    assert config["shared_expert_intermediate_size"] == 128
