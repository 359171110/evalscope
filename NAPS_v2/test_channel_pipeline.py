from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from NAPS_v2.build_channel_artifacts import main as build_artifacts
from NAPS_v2.build_channel_merge_artifacts import main as build_merge_artifacts
from NAPS_v2.build_channel_profile import main as build_profile
from NAPS_v2.build_naps_v2_artifacts import file_sha256, load_weight_map
from NAPS_v2.capture_routed_tokens import weights_only_safe
from NAPS_v2.channel_merge import apply_channel_merge_plan
from NAPS_v2.export_naps_v2_heterogeneous_checkpoint import main as export_checkpoint
from NAPS_v2.model_adapter import PurePseudoModelAdapter


def build_tiny_gemma4_checkpoint(model_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    model_path.mkdir()
    config = {
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "intermediate_size": 8,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "top_k_experts": 1,
        },
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    gate_up = torch.arange(2 * 128 * 4, dtype=torch.float32).reshape(2, 128, 4) / 100.0 + 0.01
    down = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(2, 4, 64) / 100.0 + 0.01
    tensors = {
        "model.language_model.layers.0.router.proj.weight": torch.ones(2, 4),
        "model.language_model.layers.0.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.experts.down_proj": down,
    }
    shard_name = "model.safetensors"
    save_file(tensors, model_path / shard_name)
    index = {
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())},
        "weight_map": {name: shard_name for name in tensors},
    }
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return gate_up, down


def routed_record(inputs: torch.Tensor, route_weights: torch.Tensor) -> dict:
    return {
        "inputs": inputs.to(torch.bfloat16),
        "route_weights": route_weights,
        "captured_token_count": int(inputs.shape[0]),
        "captured_route_mass": float(route_weights.sum().item()),
        "total_route_count": 10,
        "total_route_mass": 2.0,
    }


def test_tiny_channel_pipeline_exports_ranked_prefixes(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model"
    gate_up, down = build_tiny_gemma4_checkpoint(model_path)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.channel_architecture
    fit_inputs = torch.tensor([[1.0, 0.5, -0.5, 2.0], [0.5, -1.0, 1.5, 0.25]])
    holdout_inputs = torch.tensor([[0.25, 1.0, -0.25, 0.5], [1.5, 0.5, 0.25, -0.5]])
    route_weights = torch.tensor([0.75, 0.5])
    capture_path = tmp_path / "capture.pt"
    capture = {
        "schema_version": 2,
        "model_path": str(model_path.resolve()),
        "model_family": "gemma4",
        "architecture": weights_only_safe(asdict(architecture)),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "calibration": {"protocol_name": "tiny-label-free"},
        "layers": [0],
        "max_tokens_per_expert": 2,
        "max_length": 8,
        "input_storage_dtype": "bfloat16",
        "splits": {
            "fit": {
                "prompt_count": 2,
                "input_storage_dtype": "bfloat16",
                "response_statistic_scope": "all_routed_tokens",
                "route_weighted_response_energy": torch.stack((
                    torch.arange(1, 65, dtype=torch.float32),
                    torch.arange(65, 129, dtype=torch.float32),
                )).unsqueeze(0),
                "layers": {0: {
                    0: routed_record(fit_inputs, route_weights),
                    1: routed_record(fit_inputs.flip(0), route_weights.flip(0)),
                }},
            },
            "holdout": {
                "prompt_count": 2,
                "input_storage_dtype": "bfloat16",
                "response_statistic_scope": "all_routed_tokens",
                "route_weighted_response_energy": torch.ones(1, 2, 64),
                "layers": {0: {
                    0: routed_record(holdout_inputs, route_weights),
                    1: routed_record(holdout_inputs.flip(0), route_weights.flip(0)),
                }},
            },
        },
    }
    torch.save(capture, capture_path)

    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr("sys.argv", [
        "build_channel_artifacts",
        "--model-path", str(model_path),
        "--capture", str(capture_path),
        "--output-dir", str(artifact_dir),
        "--widths", "32", "64",
    ])
    assert build_artifacts() == 0
    monkeypatch.setattr("sys.argv", [
        "build_channel_profile",
        "--artifact-dir", str(artifact_dir),
        "--uniform-width", "32",
    ])
    assert build_profile() == 0

    checkpoint_dir = tmp_path / "checkpoint"
    monkeypatch.setattr("sys.argv", [
        "export_naps_v2_heterogeneous_checkpoint",
        "--model-path", str(model_path),
        "--artifact-dir", str(artifact_dir),
        "--output-dir", str(checkpoint_dir),
    ])
    assert export_checkpoint() == 0

    rankings = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    retained = rankings["table"][0]["ranked_indices_by_width"][:, 0, :32]
    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        exported_gate_up = handle.get_tensor("model.language_model.layers.0.experts.gate_up_proj")
        exported_down = handle.get_tensor("model.language_model.layers.0.experts.down_proj")
    for expert_id in range(2):
        expected_gate = gate_up[expert_id, :64].index_select(0, retained[expert_id])
        expected_up = gate_up[expert_id, 64:].index_select(0, retained[expert_id])
        assert torch.equal(exported_gate_up[expert_id, :32], expected_gate)
        assert torch.equal(exported_gate_up[expert_id, 32:], expected_up)
        assert torch.equal(exported_down[expert_id], down[expert_id].index_select(1, retained[expert_id]))

    manifest = json.loads((checkpoint_dir / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "channel_calibrated_nested_mask_padded"
    assert manifest["calibration"]["protocol_name"] == "tiny-label-free"


def test_tiny_channel_pipeline_applies_optional_sparse_merge(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model"
    gate_up, down = build_tiny_gemma4_checkpoint(model_path)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    capture_path = tmp_path / "capture.pt"
    route_weights = torch.tensor([0.75, 0.5])
    inputs = torch.tensor([[1.0, 0.5, -0.5, 2.0], [0.5, -1.0, 1.5, 0.25]])
    capture = {
        "schema_version": 2,
        "model_path": str(model_path.resolve()),
        "model_family": "gemma4",
        "architecture": weights_only_safe(asdict(adapter.channel_architecture)),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "calibration": {"protocol_name": "tiny-label-free"},
        "layers": [0],
        "max_tokens_per_expert": 2,
        "max_length": 8,
        "input_storage_dtype": "bfloat16",
        "splits": {
            split: {
                "prompt_count": 2,
                "input_storage_dtype": "bfloat16",
                "response_statistic_scope": "all_routed_tokens",
                "route_weighted_response_energy": torch.ones(1, 2, 64),
                "layers": {0: {
                    0: routed_record(inputs, route_weights),
                    1: routed_record(inputs.flip(0), route_weights.flip(0)),
                }},
            }
            for split in ("fit", "holdout")
        },
    }
    torch.save(capture, capture_path)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr("sys.argv", [
        "build_channel_artifacts",
        "--model-path", str(model_path),
        "--capture", str(capture_path),
        "--output-dir", str(artifact_dir),
        "--widths", "32", "64",
    ])
    assert build_artifacts() == 0
    monkeypatch.setattr("sys.argv", [
        "build_channel_profile",
        "--artifact-dir", str(artifact_dir),
        "--uniform-width", "32",
    ])
    assert build_profile() == 0
    rankings = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    retained = rankings["table"][0]["ranked_indices_by_width"][:, 0, :32]
    generated_merge_dir = tmp_path / "generated_merge"
    monkeypatch.setattr("sys.argv", [
        "build_channel_merge_artifacts",
        "--model-path", str(model_path),
        "--artifact-dir", str(artifact_dir),
        "--capture", str(capture_path),
        "--output-dir", str(generated_merge_dir),
        "--target-cap", "2",
        "--min-fit-rows", "2",
        "--min-holdout-rows", "1",
        "--min-abs-correlation", "0.0",
    ])
    assert build_merge_artifacts() == 0
    generated_merge = torch.load(
        generated_merge_dir / "channel_merge_plan.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert generated_merge["retained_width"] == 32
    assert generated_merge["physical_width"] == 32
    assert generated_merge["holdout_used_for_acceptance"] is True
    assert generated_merge["summary"]["total_experts"] == 2
    plans = {}
    for expert_id in range(2):
        retained_set = set(retained[expert_id].tolist())
        target = next(channel for channel in range(64) if channel not in retained_set)
        representative = int(retained[expert_id, 0].item())
        plans[expert_id] = {
            "accepted": True,
            "retained_width": 32,
            "target_channels": [target],
            "representative_channels": [representative],
            "coefficients": [0.5],
            "trust_region_scale": 1.0,
        }
    merge_path = tmp_path / "channel_merge_plan.pt"
    torch.save({
        "schema_version": 1,
        "model_path": str(model_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "rankings_sha256": file_sha256(artifact_dir / "rankings.pt"),
        "profile_sha256": file_sha256(artifact_dir / "profile.pt"),
        "capture_path": str(capture_path.resolve()),
        "capture_sha256": file_sha256(capture_path),
        "holdout_used_for_acceptance": True,
        "benchmark_metrics_used": False,
        "layers": {0: plans},
        "summary": {"accepted_experts": 2},
    }, merge_path)
    checkpoint_dir = tmp_path / "checkpoint"
    monkeypatch.setattr("sys.argv", [
        "export_naps_v2_heterogeneous_checkpoint",
        "--model-path", str(model_path),
        "--artifact-dir", str(artifact_dir),
        "--output-dir", str(checkpoint_dir),
        "--channel-merge-plan", str(merge_path),
    ])
    assert export_checkpoint() == 0

    with safe_open(checkpoint_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        exported_gate_up = handle.get_tensor("model.language_model.layers.0.experts.gate_up_proj")
        exported_down = handle.get_tensor("model.language_model.layers.0.experts.down_proj")
    for expert_id in range(2):
        expected_down = apply_channel_merge_plan(down[expert_id], retained[expert_id], plans[expert_id])
        assert torch.equal(exported_gate_up[expert_id, :32], gate_up[expert_id, :64].index_select(0, retained[expert_id]))
        assert torch.equal(exported_gate_up[expert_id, 32:], gate_up[expert_id, 64:].index_select(0, retained[expert_id]))
        assert torch.equal(exported_down[expert_id], expected_down)
    manifest = json.loads((checkpoint_dir / "pruning_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "channel_calibrated_nested_mask_sparse_merge_padded"
    assert manifest["channel_merge"]["accepted_experts"] == 2