from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from Random.build_random_artifacts import file_sha256
from Random.export_random_checkpoint import main as export_checkpoint
from Random.random_core import random_channel_order


def write_checkpoint(model_path: Path, family: str) -> dict[str, torch.Tensor]:
    model_path.mkdir()
    if family == "qwen3":
        width = 128
        config = {
            "model_type": "qwen3_moe",
            "hidden_size": 4,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        tensors = {"model.layers.0.mlp.gate.weight": torch.ones(2, 4)}
        for expert_id in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert_id}"
            tensors[f"{prefix}.gate_proj.weight"] = torch.arange(width * 4).reshape(width, 4).float() + expert_id
            tensors[f"{prefix}.up_proj.weight"] = torch.arange(width * 4).reshape(width, 4).float() + 1000 + expert_id
            tensors[f"{prefix}.down_proj.weight"] = torch.arange(width * 4).reshape(4, width).float() + 2000 + expert_id
    elif family == "gemma4":
        width = 64
        text_config = {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "intermediate_size": 128,
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "top_k_experts": 1,
        }
        config = {"model_type": "gemma4", "text_config": text_config}
        prefix = "model.language_model.layers.0"
        tensors = {
            f"{prefix}.router.proj.weight": torch.ones(2, 4),
            f"{prefix}.experts.gate_up_proj": torch.arange(2 * width * 2 * 4).reshape(2, width * 2, 4).float(),
            f"{prefix}.experts.down_proj": torch.arange(2 * 4 * width).reshape(2, 4, width).float(),
        }
    elif family == "qwen3.6":
        width = 128
        text_config = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 4,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "shared_expert_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        config = {"model_type": "qwen3_5_moe", "text_config": text_config}
        prefix = "model.language_model.layers.0.mlp"
        tensors = {
            f"{prefix}.gate.weight": torch.ones(2, 4),
            f"{prefix}.shared_expert.gate_proj.weight": torch.full((width, 4), -7.0),
            f"{prefix}.experts.gate_up_proj": torch.arange(2 * width * 2 * 4).reshape(2, width * 2, 4).float(),
            f"{prefix}.experts.down_proj": torch.arange(2 * 4 * width).reshape(2, 4, width).float(),
        }
    else:
        width = 64
        config = {
            "model_type": "deepseek_v2",
            "hidden_size": 4,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "n_shared_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
        }
        tensors = {
            "model.layers.0.mlp.gate_proj.weight": torch.full((8, 4), 9.0),
            "model.layers.1.mlp.gate.weight": torch.ones(2, 4),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.full((width * 2, 4), -3.0),
            "model.layers.1.mlp.shared_experts.up_proj.weight": torch.full((width * 2, 4), -4.0),
            "model.layers.1.mlp.shared_experts.down_proj.weight": torch.full((4, width * 2), -5.0),
        }
        for expert_id in range(2):
            prefix = f"model.layers.1.mlp.experts.{expert_id}"
            tensors[f"{prefix}.gate_proj.weight"] = torch.arange(width * 4).reshape(width, 4).float() + expert_id
            tensors[f"{prefix}.up_proj.weight"] = torch.arange(width * 4).reshape(width, 4).float() + 100 + expert_id
            tensors[f"{prefix}.down_proj.weight"] = torch.arange(width * 4).reshape(4, width).float() + 200 + expert_id
        (model_path / "configuration_deepseek.py").write_text(
            "class DeepseekV2Config:\n"
            "    def __init__(\n"
            "        self,\n"
            "        moe_intermediate_size = 1407,\n"
            "        n_shared_experts = None,\n"
            "    ):\n"
            "        self.moe_intermediate_size = moe_intermediate_size\n"
            "        self.n_shared_experts = n_shared_experts\n",
            encoding="utf-8",
        )
        (model_path / "modeling_deepseek.py").write_text(
            "if config.n_shared_experts is not None:\n"
            "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
            "            self.shared_experts = None\n",
            encoding="utf-8",
        )
    save_file(tensors, model_path / "model.safetensors")
    index = {
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return tensors


def write_artifacts(model_path: Path, artifact_dir: Path, family: str, block_size: int) -> tuple[Path, Path]:
    width = 64 if family in {"gemma4", "deepseek"} else 128
    moe_layer = 1 if family == "deepseek" else 0
    ranking = torch.stack([
        random_channel_order(width, seed=42, layer_id=moe_layer, expert_id=0),
        random_channel_order(width, seed=42, layer_id=moe_layer, expert_id=1),
    ])
    table = {
        moe_layer: {
            "ranked_indices": ranking,
            "block_relative_scores": torch.ones(2, width // block_size),
            "block_coverage_scores": torch.full((2, width // block_size), block_size / width),
            "block_sizes": torch.full((width // block_size, ), block_size),
            "intermediate_size": width,
        }
    }
    channel = {
        "schema_version": 1,
        "purpose": "random_channel_ranking",
        "model_path": str(model_path.resolve()),
        "model_family": "deepseek_v2" if family == "deepseek" else family,
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "block_size": block_size,
        "table": table,
        "random": {"seed": 42, "data_free": True},
    }
    artifact_dir.mkdir()
    channel_path = artifact_dir / "random_rankings.pt"
    torch.save(channel, channel_path)
    widths = torch.ones((1, 2), dtype=torch.long)
    profile = {
        "schema_version": 1,
        "method": "random",
        "model_path": str(model_path.resolve()),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [moe_layer],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": width // block_size,
        "channel_block_size": block_size,
        "total_blocks": 2,
        "maximum_blocks": 2 * (width // block_size),
        "profile_widths": widths,
        "retained_expert_mask": None,
        "cache_provenance": {"channel": {"sha256": file_sha256(channel_path)}},
        "random": {"seed": 42},
    }
    profile_path = artifact_dir / "profile.pt"
    torch.save(profile, profile_path)
    return channel_path, profile_path


def run_export(tmp_path: Path, monkeypatch, family: str, block_size: int) -> tuple[Path, dict[str, torch.Tensor]]:
    model_path = tmp_path / f"{family}-model"
    tensors = write_checkpoint(model_path, family)
    channel, profile = write_artifacts(model_path, tmp_path / f"{family}-artifacts", family, block_size)
    output = tmp_path / f"{family}-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_random_checkpoint",
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


def test_export_qwen3_slices_coupled_separate_tensors(tmp_path: Path, monkeypatch) -> None:
    output, source = run_export(tmp_path, monkeypatch, "qwen3", 64)
    retained = random_channel_order(128, seed=42, layer_id=0, expert_id=0)[:64]
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
    retained = random_channel_order(64, seed=42, layer_id=0, expert_id=0)[:32]
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
    retained = random_channel_order(64, seed=42, layer_id=1, expert_id=0)[:32]
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
    assert config["shared_expert_intermediate_size"] == source[shared_name].shape[0]
    assert config["shared_expert_intermediate_size"] == 128
    assert manifest["export_layout"] == "slice_uniform_width"
    assert manifest["exported_moe_intermediate_size"] == 32
    assert manifest["exported_shared_expert_intermediate_size"] == 128
    config_py = (output / "configuration_deepseek.py").read_text(encoding="utf-8")
    modeling_py = (output / "modeling_deepseek.py").read_text(encoding="utf-8")
    assert "self.shared_expert_intermediate_size = shared_expert_intermediate_size" in config_py
    assert "shared_width = int(getattr(config, \"shared_expert_intermediate_size\", 0) or 0)" in modeling_py
