from __future__ import annotations

import json
from pathlib import Path

from Magnitude.model_adapter import MagnitudeModelAdapter


def write_config(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_qwen3_adapter_uses_separate_routed_experts(tmp_path: Path) -> None:
    write_config(
        tmp_path, {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "hidden_act": "silu",
            "moe_intermediate_size": 768,
            "num_hidden_layers": 48,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        }
    )
    weight_map = {
        "model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
    }

    adapter = MagnitudeModelAdapter.from_checkpoint(tmp_path, weight_map)

    assert adapter.architecture.model_family == "qwen3"
    assert adapter.architecture.tensor_codec == "separate"
    assert adapter.architecture.width_for_pruning(0.5) == 384
    assert adapter.architecture.width_for_pruning(0.25) == 576
    assert adapter.gate_name(1, 2) == "model.layers.1.mlp.experts.2.gate_proj.weight"


def test_gemma4_adapter_preserves_dense_plus_sparse_topology(tmp_path: Path) -> None:
    write_config(
        tmp_path, {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "hidden_activation": "gelu_pytorch_tanh",
                "intermediate_size": 2112,
                "moe_intermediate_size": 704,
                "num_hidden_layers": 30,
                "num_experts": 128,
                "top_k_experts": 8,
            },
        }
    )
    weight_map = {
        "model.language_model.layers.0.router.proj.weight": "model.safetensors",
        "model.language_model.layers.0.experts.gate_up_proj": "model.safetensors",
        "model.language_model.layers.0.experts.down_proj": "model.safetensors",
    }

    adapter = MagnitudeModelAdapter.from_checkpoint(tmp_path, weight_map)

    assert adapter.architecture.activation == "gelu_pytorch_tanh"
    assert adapter.architecture.branch_topology == "dense_plus_sparse"
    assert adapter.architecture.channel_alignment == 32
    assert adapter.architecture.width_for_pruning(0.5) == 352
    assert adapter.architecture.width_for_pruning(0.25) == 512


def test_qwen36_adapter_identifies_only_packed_routed_tensors(tmp_path: Path) -> None:
    write_config(
        tmp_path, {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 2048,
                "hidden_act": "silu",
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "num_hidden_layers": 40,
                "num_experts": 256,
                "num_experts_per_tok": 8,
            },
        }
    )
    weight_map = {
        "model.language_model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "model.safetensors",
    }

    adapter = MagnitudeModelAdapter.from_checkpoint(tmp_path, weight_map)

    assert adapter.architecture.model_family == "qwen3.6"
    assert adapter.architecture.branch_topology == "gated_shared"
    assert adapter.architecture.tensor_codec == "packed"
    assert adapter.architecture.width_for_pruning(0.5) == 256
    assert adapter.architecture.width_for_pruning(0.25) == 384


def test_deepseek_v2_adapter_skips_dense_first_layer_and_aligns_25_50(tmp_path: Path) -> None:
    write_config(
        tmp_path, {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "hidden_act": "silu",
            "moe_intermediate_size": 1408,
            "num_hidden_layers": 27,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 1,
        }
    )
    weight_map = {
        "model.layers.1.mlp.gate.weight": "model.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.1.mlp.experts.0.up_proj.weight": "model.safetensors",
        "model.layers.1.mlp.experts.0.down_proj.weight": "model.safetensors",
    }

    adapter = MagnitudeModelAdapter.from_checkpoint(tmp_path, weight_map)

    assert adapter.architecture.model_family == "deepseek_v2"
    assert adapter.architecture.moe_layer_ids()[0] == 1
    assert adapter.architecture.channel_alignment == 32
    assert adapter.architecture.width_for_pruning(0.25) == 1056
    assert adapter.architecture.width_for_pruning(0.5) == 704
    assert adapter.gate_name(1, 2) == "model.layers.1.mlp.experts.2.gate_proj.weight"
