from __future__ import annotations

import json
from pathlib import Path

from NAPS_v2.model_adapter import (
    BranchTopology,
    ExpertInputSemantics,
    ExpertTensorCodec,
    PurePseudoModelAdapter,
)


def write_config(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_gemma4_channel_architecture_contract(tmp_path: Path) -> None:
    write_config(tmp_path, {
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
    })
    weight_map = {
        "model.language_model.layers.0.router.proj.weight": "model.safetensors",
        "model.language_model.layers.0.experts.gate_up_proj": "model.safetensors",
        "model.language_model.layers.0.experts.down_proj": "model.safetensors",
    }

    architecture = PurePseudoModelAdapter.from_checkpoint(tmp_path, weight_map).channel_architecture

    assert architecture.activation == "gelu_pytorch_tanh"
    assert architecture.expert_tensor_codec is ExpertTensorCodec.PACKED
    assert architecture.branch_topology is BranchTopology.DENSE_PLUS_SPARSE
    assert architecture.expert_input_semantics is ExpertInputSemantics.INDEPENDENT_FROM_RAW_RESIDUAL
    assert architecture.channel_alignment == 32
    assert architecture.dense_intermediate_size == 2112
    assert architecture.shared_expert_intermediate_size is None
    assert architecture.router_has_global_scale
    assert architecture.router_has_per_expert_scale
    assert architecture.aligned_width(470) == 480
    assert architecture.width_for_retention(0.5) == 352


def test_qwen3_channel_architecture_contract(tmp_path: Path) -> None:
    write_config(tmp_path, {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "hidden_act": "silu",
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    })
    weight_map = {
        "model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
    }

    architecture = PurePseudoModelAdapter.from_checkpoint(tmp_path, weight_map).channel_architecture

    assert architecture.activation == "silu"
    assert architecture.expert_tensor_codec is ExpertTensorCodec.SEPARATE
    assert architecture.branch_topology is BranchTopology.ROUTED_ONLY
    assert architecture.expert_input_semantics is ExpertInputSemantics.SHARED_ROUTER_INPUT
    assert architecture.channel_alignment == 64
    assert architecture.shared_expert_intermediate_size is None
    assert not architecture.router_has_global_scale
    assert architecture.width_for_retention(0.5) == 384


def test_qwen36_channel_architecture_contract(tmp_path: Path) -> None:
    write_config(tmp_path, {
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
    })
    weight_map = {
        "model.language_model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.language_model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
        "model.language_model.layers.0.mlp.experts.down_proj": "model.safetensors",
    }

    architecture = PurePseudoModelAdapter.from_checkpoint(tmp_path, weight_map).channel_architecture

    assert architecture.expert_tensor_codec is ExpertTensorCodec.PACKED
    assert architecture.branch_topology is BranchTopology.GATED_SHARED
    assert architecture.shared_expert_intermediate_size == 512
    assert architecture.channel_alignment == 64
    assert architecture.width_for_retention(0.5) == 256


def test_channel_width_validation_rejects_unaligned_width(tmp_path: Path) -> None:
    write_config(tmp_path, {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "hidden_act": "silu",
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    })
    weight_map = {
        "model.layers.0.mlp.gate.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
        "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
    }
    architecture = PurePseudoModelAdapter.from_checkpoint(tmp_path, weight_map).channel_architecture

    try:
        architecture.validate_width(352)
    except ValueError as error:
        assert str(error) == "Channel width must be divisible by 64"
    else:
        raise AssertionError("Unaligned Qwen3 width must be rejected")