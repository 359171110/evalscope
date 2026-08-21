from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_checkpoint(model_path: Path, family: str) -> dict[str, torch.Tensor]:
    """Write a tiny checkpoint whose channel c has magnitude c+1 on expert 0."""

    model_path.mkdir()
    if family == "qwen3":
        width = 256
        hidden = 4
        config = {
            "model_type": "qwen3_moe",
            "hidden_size": hidden,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        tensors = {"model.layers.0.mlp.gate.weight": torch.ones(2, hidden)}
        for expert_id in range(2):
            prefix = f"model.layers.0.mlp.experts.{expert_id}"
            tensors[f"{prefix}.gate_proj.weight"] = _increasing_rows(width, hidden, expert_id)
            tensors[f"{prefix}.up_proj.weight"] = torch.zeros(width, hidden)
            tensors[f"{prefix}.down_proj.weight"] = torch.zeros(hidden, width)
    elif family == "gemma4":
        width = 64
        hidden = 4
        text_config = {
            "model_type": "gemma4_text",
            "hidden_size": hidden,
            "hidden_activation": "gelu_pytorch_tanh",
            "intermediate_size": 128,
            "moe_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "top_k_experts": 1,
        }
        config = {"model_type": "gemma4", "text_config": text_config}
        prefix = "model.language_model.layers.0"
        gate_up = torch.zeros(2, width * 2, hidden)
        down = torch.zeros(2, hidden, width)
        for expert_id in range(2):
            gate_up[expert_id, :width] = _increasing_rows(width, hidden, expert_id)
        tensors = {
            f"{prefix}.router.proj.weight": torch.ones(2, hidden),
            f"{prefix}.experts.gate_up_proj": gate_up,
            f"{prefix}.experts.down_proj": down,
        }
    elif family == "qwen3.6":
        width = 128
        hidden = 4
        text_config = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": hidden,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "shared_expert_intermediate_size": width,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        config = {"model_type": "qwen3_5_moe", "text_config": text_config}
        prefix = "model.language_model.layers.0.mlp"
        gate_up = torch.zeros(2, width * 2, hidden)
        down = torch.zeros(2, hidden, width)
        for expert_id in range(2):
            gate_up[expert_id, :width] = _increasing_rows(width, hidden, expert_id)
        tensors = {
            f"{prefix}.gate.weight": torch.ones(2, hidden),
            f"{prefix}.shared_expert.gate_proj.weight": torch.full((width, hidden), -7.0),
            f"{prefix}.experts.gate_up_proj": gate_up,
            f"{prefix}.experts.down_proj": down,
        }
    else:
        width = 64
        hidden = 4
        config = {
            "model_type": "deepseek_v2",
            "hidden_size": hidden,
            "hidden_act": "silu",
            "moe_intermediate_size": width,
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "n_shared_experts": 2,
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
        }
        tensors = {
            "model.layers.0.mlp.gate_proj.weight": torch.full((8, hidden), 9.0),
            "model.layers.1.mlp.gate.weight": torch.ones(2, hidden),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.full((width * 2, hidden), -3.0),
            "model.layers.1.mlp.shared_experts.up_proj.weight": torch.full((width * 2, hidden), -4.0),
            "model.layers.1.mlp.shared_experts.down_proj.weight": torch.full((hidden, width * 2), -5.0),
        }
        for expert_id in range(2):
            prefix = f"model.layers.1.mlp.experts.{expert_id}"
            tensors[f"{prefix}.gate_proj.weight"] = _increasing_rows(width, hidden, expert_id)
            tensors[f"{prefix}.up_proj.weight"] = torch.zeros(width, hidden)
            tensors[f"{prefix}.down_proj.weight"] = torch.zeros(hidden, width)
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


def _increasing_rows(width: int, hidden: int, expert_id: int) -> torch.Tensor:
    rows = torch.zeros(width, hidden)
    values = torch.arange(1, width + 1, dtype=torch.float32)
    if expert_id == 1:
        values = values.flip(0)
    rows[:, 0] = values
    return rows


def expected_ranking(width: int, expert_id: int) -> torch.Tensor:
    order = torch.arange(width - 1, -1, -1)
    if expert_id == 1:
        order = torch.arange(width)
    return order
