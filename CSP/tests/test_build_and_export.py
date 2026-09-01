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
    retained = 64 if family in {"qwen3", "qwen3.6", "olmoe", "mixtral"} else 32
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


@pytest.mark.parametrize("family", ["qwen3", "gemma4", "qwen3.6", "deepseek", "olmoe", "mixtral"])
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
    assert profile["retained_channels"] == (64 if family in {"qwen3", "qwen3.6", "olmoe", "mixtral"} else 32)
    assert manifest["method"] == "csp"
    assert manifest["retained_channels"] == (64 if family in {"qwen3", "qwen3.6", "olmoe", "mixtral"} else 32)
    if family in {"olmoe", "mixtral"}:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert config["intermediate_size"] == 64
        assert "moe_intermediate_size" not in config


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


def test_qwen3_heterogeneous_profile_has_fixed_layer_budget(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "qwen3-model"
    write_checkpoint(model, "qwen3", large=True)
    artifact = tmp_path / "qwen3-heterogeneous-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "profile.pt"
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model),
        "--output-channel-cache", str(cache), "--output-profile", str(profile),
        "--heterogeneous-widths", "320", "384", "448", "--budget-width", "384",
    ])
    assert build_main() == 0
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    assert payload["allocation_scope"] == "per_layer_expert_sp_quantiles"
    assert payload["mode"] == "hsp_hetero_raw_expert_sp_quantiles"
    assert payload["profile_widths"].shape == (1, 4)
    assert int(payload["profile_widths"].sum()) == 4 * 6
    assert payload["width_options"] == [320, 384, 448]


@pytest.mark.parametrize(
    ("family", "widths", "budget"),
    [
        ("qwen3", (320, 384, 448), 384),
        ("qwen3.6", (192, 256, 320), 256),
        ("gemma4", (288, 352, 416), 352),
        ("deepseek", (576, 704, 832), 704),
        ("olmoe", (448, 512, 576), 512),
        ("mixtral", (448, 512, 576), 512),
    ],
)
def test_hsp_heterogeneous_profile_supports_all_model_families(
    tmp_path: Path,
    monkeypatch,
    family: str,
    widths: tuple[int, int, int],
    budget: int,
) -> None:
    model = tmp_path / f"{family}-hsp-model"
    write_checkpoint(model, family, large=True)
    artifact = tmp_path / f"{family}-hsp-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "hsp_profile.pt"
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model),
        "--output-channel-cache", str(cache), "--output-profile", str(profile),
        "--heterogeneous-widths", *(str(width) for width in widths),
        "--budget-width", str(budget), "--apply-input-scale", "never",
    ])
    assert build_main() == 0
    payload = torch.load(profile, map_location="cpu", weights_only=True)
    logical_widths = payload["profile_widths"] * payload["channel_block_size"]
    assert payload["method"] == "hsp"
    assert payload["mode"] == "hsp_hetero_raw_expert_sp_quantiles"
    assert sorted(torch.unique(logical_widths).tolist()) == list(widths)
    assert logical_widths.sum(dim=1).tolist() == [payload["num_experts"] * budget] * payload["num_layers"]
    assert payload["csp"]["canonicalization"] is False


@pytest.mark.parametrize(
    ("family", "widths", "budget", "physical_width"),
    [
        ("qwen3.6", (192, 256, 320), 256, 320),
        ("gemma4", (288, 352, 416), 352, 416),
        ("deepseek", (576, 704, 832), 704, 832),
        ("olmoe", (448, 512, 576), 512, 576),
        ("mixtral", (448, 512, 576), 512, 576),
    ],
)
def test_hsp_heterogeneous_export_supports_packed_and_shared_layouts(
    tmp_path: Path,
    monkeypatch,
    family: str,
    widths: tuple[int, int, int],
    budget: int,
    physical_width: int,
) -> None:
    model = tmp_path / f"{family}-hsp-export-model"
    write_checkpoint(model, family, large=True)
    artifact = tmp_path / f"{family}-hsp-export-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "hsp_profile.pt"
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model),
        "--output-channel-cache", str(cache), "--output-profile", str(profile),
        "--heterogeneous-widths", *(str(width) for width in widths),
        "--budget-width", str(budget), "--apply-input-scale", "never",
    ])
    assert build_main() == 0
    output = tmp_path / f"{family}-hsp-export-output"
    monkeypatch.setattr("sys.argv", [
        "export_csp_checkpoint", "--model-path", str(model), "--profile", str(profile),
        "--channel-cache", str(cache), "--output-dir", str(output),
    ])
    assert export_main() == 0

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    if family in {"olmoe", "mixtral"}:
        assert text_config["intermediate_size"] == physical_width
        assert "moe_intermediate_size" not in text_config
    else:
        assert text_config["moe_intermediate_size"] == physical_width
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "hsp"
    assert manifest["padded_intermediate_size"] == physical_width
    assert manifest["allocation_scope"] == "per_layer_expert_sp_quantiles"

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    routed_name = next(name for name in index["weight_map"] if "experts" in name and (
        name.endswith("gate_up_proj") or name.endswith("gate_proj.weight") or name.endswith("w1.weight")
    ))
    with safe_open(output / index["weight_map"][routed_name], framework="pt", device="cpu") as handle:
        assert handle.get_tensor(routed_name).shape[-2 if family == "gemma4" else 0] > 0


@pytest.mark.parametrize("family", ["qwen3.6", "gemma4"])
def test_hsp_packed_export_pads_gate_and_up_halves_separately(
    tmp_path: Path,
    monkeypatch,
    family: str,
) -> None:
    widths, budget, physical_width = (
        ((192, 256, 320), 256, 320) if family == "qwen3.6" else ((288, 352, 416), 352, 416)
    )
    model = tmp_path / f"{family}-hsp-packed-model"
    write_checkpoint(model, family, large=True)
    artifact = tmp_path / f"{family}-hsp-packed-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "hsp_profile.pt"
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model),
        "--output-channel-cache", str(cache), "--output-profile", str(profile),
        "--heterogeneous-widths", *(str(width) for width in widths),
        "--budget-width", str(budget), "--apply-input-scale", "never",
    ])
    assert build_main() == 0
    output = tmp_path / f"{family}-hsp-packed-output"
    monkeypatch.setattr("sys.argv", [
        "export_csp_checkpoint", "--model-path", str(model), "--profile", str(profile),
        "--channel-cache", str(cache), "--output-dir", str(output),
    ])
    assert export_main() == 0

    payload = torch.load(profile, map_location="cpu", weights_only=True)
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    routed_name = next(name for name in index["weight_map"] if name.endswith("gate_up_proj"))
    with safe_open(output / index["weight_map"][routed_name], framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(routed_name)
    assert packed.shape[1] == 2 * physical_width
    block = int(payload["channel_block_size"])
    for expert_id in range(int(payload["num_experts"])):
        logical = int(payload["profile_widths"][0, expert_id].item()) * block
        gate_slot = packed[expert_id, :physical_width]
        up_slot = packed[expert_id, physical_width:]
        assert torch.count_nonzero(gate_slot[logical:]) == 0
        assert torch.count_nonzero(up_slot[logical:]) == 0
        assert torch.count_nonzero(gate_slot[:logical]) > 0
        assert torch.count_nonzero(up_slot[:logical]) > 0


def test_qwen3_heterogeneous_export_pads_logical_experts_to_common_width(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "qwen3-model"
    source = write_checkpoint(model, "qwen3", large=True)
    artifact = tmp_path / "qwen3-heterogeneous-export-artifact"
    cache = artifact / "csp_rankings.pt"
    profile = artifact / "profile.pt"
    monkeypatch.setattr("sys.argv", [
        "build_csp_artifacts", "--model-path", str(model),
        "--output-channel-cache", str(cache), "--output-profile", str(profile),
        "--heterogeneous-widths", "320", "384", "448", "--budget-width", "384",
    ])
    assert build_main() == 0
    output = tmp_path / "qwen3-heterogeneous-output"
    monkeypatch.setattr("sys.argv", [
        "export_csp_checkpoint", "--model-path", str(model), "--profile", str(profile),
        "--channel-cache", str(cache), "--output-dir", str(output),
    ])
    assert export_main() == 0

    payload = torch.load(profile, map_location="cpu", weights_only=True)
    manifest = json.loads((output / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["padded_intermediate_size"] == 448
    assert manifest["logical_widths_by_layer"] == (payload["profile_widths"] * 64).tolist()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["moe_intermediate_size"] == 448

    with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
        for expert_id in range(4):
            gate_name = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_name = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            down_name = f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
            assert handle.get_tensor(gate_name).shape == (448, 4)
            assert handle.get_tensor(up_name).shape == (448, 4)
            assert handle.get_tensor(down_name).shape == (4, 448)
            logical = int(payload["profile_widths"][0, expert_id].item()) * 64
            assert torch.count_nonzero(handle.get_tensor(gate_name)[logical:]) == 0
            assert torch.count_nonzero(handle.get_tensor(up_name)[logical:]) == 0
            assert torch.count_nonzero(handle.get_tensor(down_name)[:, logical:]) == 0
            indices = torch.load(cache, map_location="cpu", weights_only=True)["table"][0]["ranked_indices"][expert_id]
            assert torch.equal(handle.get_tensor(gate_name)[:logical], source[gate_name].index_select(0, indices[:logical]))
