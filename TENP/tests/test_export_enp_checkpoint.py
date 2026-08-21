from __future__ import annotations

import json
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

from TENP.build_enp_artifacts import file_sha256
from TENP.export_enp_checkpoint import main as export_checkpoint


def write_checkpoint(model_path: Path, family: str) -> tuple[dict[str, torch.Tensor], dict]:
    model_path.mkdir()
    width = 64 if family == "gemma4" else 128
    if family == "qwen3":
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
    elif family == "deepseek":
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
            tensors[f"{prefix}.up_proj.weight"] = torch.arange(width * 4).reshape(width, 4).float() + 1000 + expert_id
            tensors[f"{prefix}.down_proj.weight"] = torch.arange(width * 4).reshape(4, width).float() + 2000 + expert_id
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
    else:
        model_type = "gemma4_text" if family == "gemma4" else "qwen3_5_moe_text"
        text_config = {
            "model_type": model_type,
            "hidden_size": 4,
            "hidden_activation": "gelu_pytorch_tanh" if family == "gemma4" else "silu",
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
        }
        if family == "gemma4":
            text_config.update({"intermediate_size": 128, "top_k_experts": 1})
            prefix = "model.language_model.layers.0"
            tensors = {f"{prefix}.router.proj.weight": torch.ones(2, 4)}
            expert_prefix = f"{prefix}.experts"
        else:
            text_config.update({"shared_expert_intermediate_size": width, "num_experts_per_tok": 1})
            prefix = "model.language_model.layers.0.mlp"
            tensors = {f"{prefix}.gate.weight": torch.ones(2, 4)}
            expert_prefix = f"{prefix}.experts"
            tensors[f"{prefix}.shared_expert.gate_proj.weight"] = torch.full((width, 4), -7.0)
        config = {"model_type": "gemma4" if family == "gemma4" else "qwen3_5_moe", "text_config": text_config}
        tensors[f"{expert_prefix}.gate_up_proj"] = torch.arange(2 * width * 2 * 4).reshape(2, width * 2, 4).float()
        tensors[f"{expert_prefix}.down_proj"] = torch.arange(2 * 4 * width).reshape(2, 4, width).float()
    save_file(tensors, model_path / "model.safetensors")
    index = {
        "metadata": {
            "total_size": sum(value.numel() * value.element_size() for value in tensors.values())
        },
        "weight_map": {
            name: "model.safetensors"
            for name in tensors
        },
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return tensors, config


def write_artifacts(model_path: Path, artifact_dir: Path, family: str, block_size: int) -> tuple[Path, Path]:
    width = 64 if family in {"gemma4", "deepseek"} else 128
    layer_id = 1 if family == "deepseek" else 0
    ranking = torch.stack((torch.arange(width - 1, -1, -1), torch.arange(width)))
    table = {
        layer_id: {
            "ranked_indices": ranking,
            "block_relative_scores": torch.ones(2, width // block_size),
            "block_coverage_scores": torch.full((2, width // block_size), block_size / width),
            "block_sizes": torch.full((width // block_size, ), block_size),
            "intermediate_size": width,
        }
    }
    channel = {
        "schema_version": 1,
        "purpose": "enp_cos_channel_ranking",
        "model_path": str(model_path.resolve()),
        "model_family": family,
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "block_size": block_size,
        "table": table,
    }
    artifact_dir.mkdir()
    channel_path = artifact_dir / "channels.pt"
    torch.save(channel, channel_path)
    widths = torch.ones((1, 2), dtype=torch.long)
    profile = {
        "schema_version": 1,
        "method": "enp",
        "model_path": str(model_path.resolve()),
        "profile_construction": "calibrated",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": [1 if family == "deepseek" else 0],
        "num_layers": 1,
        "num_experts": 2,
        "num_blocks": width // block_size,
        "channel_block_size": block_size,
        "total_blocks": 2,
        "maximum_blocks": 2 * (width // block_size),
        "profile_widths": widths,
        "retained_expert_mask": None,
        "cache_provenance": {
            "channel": {
                "sha256": file_sha256(channel_path)
            }
        },
    }
    profile_path = artifact_dir / "profile.pt"
    torch.save(profile, profile_path)
    return channel_path, profile_path


def run_export(tmp_path: Path, monkeypatch, family: str, block_size: int) -> tuple[Path, dict[str, torch.Tensor]]:
    model_path = tmp_path / f"{family}-model"
    tensors, _ = write_checkpoint(model_path, family)
    channel, profile = write_artifacts(model_path, tmp_path / f"{family}-artifacts", family, block_size)
    output = tmp_path / f"{family}-output"
    monkeypatch.setattr(
        "sys.argv", [
            "export_enp_checkpoint",
            "--model-path",
            str(model_path),
            "--profile",
            str(profile),
            "--channel-cache",
            str(channel),
            "--output-dir",
            str(output),
        ]
    )
    assert export_checkpoint() == 0
    return output, tensors


def test_export_qwen3_slices_coupled_separate_tensors(tmp_path: Path, monkeypatch) -> None:
    output, source = run_export(tmp_path, monkeypatch, "qwen3", 64)
    retained = torch.arange(127, 63, -1)
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
    retained = torch.arange(63, 31, -1)
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
    retained = torch.arange(63, 31, -1)
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
    assert manifest["method"] == "enp"
    assert manifest["export_layout"] == "slice_uniform_width"
    assert manifest["exported_shared_expert_intermediate_size"] == 128
    assert "self.shared_expert_intermediate_size = shared_expert_intermediate_size" in (
        output / "configuration_deepseek.py"
    ).read_text(encoding="utf-8")
