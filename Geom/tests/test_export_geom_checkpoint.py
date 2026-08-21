from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from Geom.build_geom_artifacts import file_sha256, main as build_main
from Geom.export_geom_checkpoint import main as export_checkpoint
from Geom.tests.helpers import expected_ranking, write_checkpoint


def run_export(tmp_path: Path, monkeypatch, family: str, retained: int) -> tuple[Path, dict[str, torch.Tensor]]:
    model_path = tmp_path / f"{family}-model"
    tensors = write_checkpoint(model_path, family)
    artifact = tmp_path / f"{family}-artifacts"
    channel = artifact / "geom_rankings.pt"
    profile = artifact / "profile.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_geom_artifacts",
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
    output = tmp_path / f"{family}-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_geom_checkpoint",
            "--model-path",
            str(model_path),
            "--profile",
            str(profile),
            "--channel-cache",
            str(channel),
            "--output-dir",
            str(output),
        ],
    )
    assert export_checkpoint() == 0
    return output, tensors


def test_export_qwen3_keeps_highest_geom_channels(tmp_path: Path, monkeypatch) -> None:
    output, source = run_export(tmp_path, monkeypatch, "qwen3", 64)
    retained = expected_ranking(256, 0)[:64]
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight"),
            source["model.layers.0.mlp.experts.0.gate_proj.weight"].index_select(0, retained),
        )
        assert torch.equal(
            handle.get_tensor("model.layers.0.mlp.experts.0.down_proj.weight"),
            source["model.layers.0.mlp.experts.0.down_proj.weight"].index_select(1, retained),
        )


def test_export_gemma4_slices_packed_experts(tmp_path: Path, monkeypatch) -> None:
    output, source = run_export(tmp_path, monkeypatch, "gemma4", 32)
    retained = expected_ranking(64, 0)[:32]
    name = "model.language_model.layers.0.experts.gate_up_proj"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        exported = handle.get_tensor(name)
        assert torch.equal(exported[0, :32], source[name][0, :64].index_select(0, retained))
        assert torch.equal(exported[0, 32:], source[name][0, 64:].index_select(0, retained))


def test_export_qwen36_preserves_shared_expert(tmp_path: Path, monkeypatch) -> None:
    output, source = run_export(tmp_path, monkeypatch, "qwen3.6", 64)
    shared_name = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor(shared_name), source[shared_name])
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["text_config"]["moe_intermediate_size"] == 64


def test_export_deepseek_slices_routed_and_keeps_fused_shared_width(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output, source = run_export(tmp_path, monkeypatch, "deepseek", 32)
    retained = expected_ranking(64, 0)[:32]
    dense_name = "model.layers.0.mlp.gate_proj.weight"
    shared_name = "model.layers.1.mlp.shared_experts.gate_proj.weight"
    routed_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        exported_routed = handle.get_tensor(routed_name)
        assert torch.equal(handle.get_tensor(dense_name), source[dense_name])
        assert torch.equal(handle.get_tensor(shared_name), source[shared_name])
        assert tuple(exported_routed.shape) == (32, 4)
        assert torch.equal(exported_routed, source[routed_name].index_select(0, retained))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert config["moe_intermediate_size"] == 32
    assert config["n_shared_experts"] == 2
    assert config["shared_expert_intermediate_size"] == 128
    assert manifest["method"] == "geom"
    assert manifest["export_layout"] == "slice_uniform_width"
    assert "self.shared_expert_intermediate_size = shared_expert_intermediate_size" in (
        output / "configuration_deepseek.py"
    ).read_text(encoding="utf-8")
    assert file_sha256(output / "pruning_export_manifest.json")
