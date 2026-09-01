from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_config(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def _rows(width: int, hidden: int, expert_id: int) -> torch.Tensor:
    values = torch.ones(width, hidden)
    scale = torch.arange(1, width + 1, dtype=torch.float32)
    if expert_id:
        scale = scale.flip(0)
    values[:, 0] = scale
    return values


def write_checkpoint(path: Path, family: str, large: bool = False) -> dict[str, torch.Tensor]:
    """Write a small safetensors fixture for one supported model family."""

    path.mkdir()
    if family == "qwen3":
        width, hidden, experts = (768, 4, 4) if large else (128, 4, 2)
        config = {
            "model_type": "qwen3_moe", "hidden_size": hidden, "hidden_act": "silu",
            "moe_intermediate_size": width, "num_hidden_layers": 1,
            "num_experts": experts, "num_experts_per_tok": 1,
        }
        tensors = {"model.layers.0.mlp.gate.weight": torch.ones(experts, hidden)}
        for expert in range(experts):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            tensors[f"{prefix}.gate_proj.weight"] = _rows(width, hidden, expert)
            tensors[f"{prefix}.up_proj.weight"] = torch.ones(width, hidden)
            tensors[f"{prefix}.down_proj.weight"] = torch.ones(hidden, width)
    elif family == "gemma4":
        width, hidden, experts = (704, 4, 4) if large else (64, 4, 2)
        text = {
            "model_type": "gemma4_text", "hidden_size": hidden, "hidden_activation": "gelu_pytorch_tanh",
            "intermediate_size": 128, "moe_intermediate_size": width, "num_hidden_layers": 1,
            "num_experts": experts, "top_k_experts": 1,
        }
        config = {"model_type": "gemma4", "text_config": text}
        gate_up = torch.zeros(experts, 2 * width, hidden)
        for expert in range(experts):
            gate_up[expert, :width] = _rows(width, hidden, expert)
            gate_up[expert, width:] = 1.0
        tensors = {
            "model.language_model.layers.0.router.proj.weight": torch.ones(experts, hidden),
            "model.language_model.layers.0.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.experts.down_proj": torch.ones(experts, hidden, width),
            "model.language_model.layers.0.pre_feedforward_layernorm_2.weight": torch.tensor([2.0, 1.0, 1.0, 1.0]),
        }
    elif family == "qwen3.6":
        width, hidden, experts = (512, 4, 4) if large else (128, 4, 2)
        text = {
            "model_type": "qwen3_5_moe_text", "hidden_size": hidden, "hidden_act": "silu",
            "moe_intermediate_size": width, "shared_expert_intermediate_size": width,
            "num_hidden_layers": 1, "num_experts": experts, "num_experts_per_tok": 1,
        }
        config = {"model_type": "qwen3_5_moe", "text_config": text}
        gate_up = torch.zeros(experts, 2 * width, hidden)
        for expert in range(experts):
            gate_up[expert, :width] = _rows(width, hidden, expert)
            gate_up[expert, width:] = 1.0
        tensors = {
            "model.language_model.layers.0.mlp.gate.weight": torch.ones(experts, hidden),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.down_proj": torch.ones(experts, hidden, width),
        }
    elif family == "deepseek":
        width, hidden, experts = (1408, 4, 4) if large else (64, 4, 2)
        config = {
            "model_type": "deepseek_v2", "hidden_size": hidden, "hidden_act": "silu",
            "moe_intermediate_size": width, "num_hidden_layers": 2, "n_routed_experts": experts,
            "n_shared_experts": 2, "num_experts_per_tok": 1, "first_k_dense_replace": 1,
        }
        tensors = {"model.layers.0.mlp.gate_proj.weight": torch.full((8, hidden), 9.0)}
        tensors.update({
            "model.layers.1.mlp.gate.weight": torch.ones(experts, hidden),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.full((width * 2, hidden), -3.0),
            "model.layers.1.mlp.shared_experts.up_proj.weight": torch.full((width * 2, hidden), -4.0),
            "model.layers.1.mlp.shared_experts.down_proj.weight": torch.full((hidden, width * 2), -5.0),
        })
        for expert in range(experts):
            prefix = f"model.layers.1.mlp.experts.{expert}"
            tensors[f"{prefix}.gate_proj.weight"] = _rows(width, hidden, expert)
            tensors[f"{prefix}.up_proj.weight"] = torch.ones(width, hidden)
            tensors[f"{prefix}.down_proj.weight"] = torch.ones(hidden, width)
        (path / "configuration_deepseek.py").write_text(
            "class DeepseekV2Config:\n"
            "    def __init__(\n"
            "        self,\n"
            "        moe_intermediate_size = 1407,\n"
            "        n_shared_experts = None,\n"
            "    ):\n"
            "        self.moe_intermediate_size = moe_intermediate_size\n"
            "        self.n_shared_experts = n_shared_experts\n", encoding="utf-8")
        (path / "modeling_deepseek.py").write_text(
            "if config.n_shared_experts is not None:\n"
            "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
            "            self.shared_experts = None\n", encoding="utf-8")
    elif family == "olmoe":
        width, hidden, experts = (1024, 4, 4) if large else (128, 4, 2)
        config = {
            "model_type": "olmoe", "hidden_size": hidden, "hidden_act": "silu",
            "intermediate_size": width, "num_hidden_layers": 1,
            "num_experts": experts, "num_experts_per_tok": 1,
        }
        tensors = {"model.layers.0.mlp.gate.weight": torch.ones(experts, hidden)}
        for expert in range(experts):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            tensors[f"{prefix}.gate_proj.weight"] = _rows(width, hidden, expert)
            tensors[f"{prefix}.up_proj.weight"] = torch.ones(width, hidden)
            tensors[f"{prefix}.down_proj.weight"] = torch.ones(hidden, width)
    elif family == "mixtral":
        width, hidden, experts = (1024, 4, 4) if large else (128, 4, 2)
        config = {
            "model_type": "mixtral", "hidden_size": hidden, "hidden_act": "silu",
            "intermediate_size": width, "num_hidden_layers": 1,
            "num_local_experts": experts, "num_experts_per_tok": 1,
        }
        tensors = {"model.layers.0.block_sparse_moe.gate.weight": torch.ones(experts, hidden)}
        for expert in range(experts):
            prefix = f"model.layers.0.block_sparse_moe.experts.{expert}"
            tensors[f"{prefix}.w1.weight"] = _rows(width, hidden, expert)
            tensors[f"{prefix}.w3.weight"] = torch.ones(width, hidden)
            tensors[f"{prefix}.w2.weight"] = torch.ones(hidden, width)
    else:
        raise ValueError(family)
    save_file(tensors, path / "model.safetensors")
    index = {
        "metadata": {"total_size": sum(value.numel() * value.element_size() for value in tensors.values())},
        "weight_map": {name: "model.safetensors" for name in tensors},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return tensors
