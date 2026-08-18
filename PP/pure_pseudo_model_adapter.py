from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PurePseudoModelAdapter:
    model_family: str
    text_config: dict[str, Any]
    layer_prefix_template: str
    router_template: str
    expert_gate_up_template: str
    expert_gate_template: str | None
    expert_up_template: str | None
    expert_down_template: str
    norm_templates: tuple[str, ...]
    num_layers: int
    num_experts: int
    intermediate_size: int
    router_top_k: int

    @classmethod
    def from_checkpoint(cls, model_path: Path, weight_map: dict[str, str]) -> "PurePseudoModelAdapter":
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        model_type = str(text_config.get("model_type", config.get("model_type", ""))).lower()
        if model_type == "qwen3_5_moe_text":
            adapter = cls(
                model_family="qwen3.6",
                text_config=text_config,
                layer_prefix_template="model.language_model.layers.{layer}",
                router_template="model.language_model.layers.{layer}.mlp.gate.weight",
                expert_gate_up_template="model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
                expert_gate_template=None,
                expert_up_template=None,
                expert_down_template="model.language_model.layers.{layer}.mlp.experts.down_proj",
                norm_templates=(
                    "model.language_model.layers.{layer}.post_attention_layernorm.weight",
                    "model.language_model.layers.{layer}.input_layernorm.weight",
                ),
                num_layers=int(text_config["num_hidden_layers"]),
                num_experts=int(text_config["num_experts"]),
                intermediate_size=int(text_config["moe_intermediate_size"]),
                router_top_k=int(text_config["num_experts_per_tok"]),
            )
        elif model_type == "gemma4_text":
            adapter = cls(
                model_family="gemma4",
                text_config=text_config,
                layer_prefix_template="model.language_model.layers.{layer}",
                router_template="model.language_model.layers.{layer}.router.proj.weight",
                expert_gate_up_template="model.language_model.layers.{layer}.experts.gate_up_proj",
                expert_gate_template=None,
                expert_up_template=None,
                expert_down_template="model.language_model.layers.{layer}.experts.down_proj",
                norm_templates=(
                    "model.language_model.layers.{layer}.post_attention_layernorm.weight",
                    "model.language_model.layers.{layer}.input_layernorm.weight",
                ),
                num_layers=int(text_config["num_hidden_layers"]),
                num_experts=int(text_config["num_experts"]),
                intermediate_size=int(text_config["moe_intermediate_size"]),
                router_top_k=int(text_config["top_k_experts"]),
            )
        elif model_type == "qwen3_moe":
            adapter = cls(
                model_family="qwen3",
                text_config=text_config,
                layer_prefix_template="model.layers.{layer}",
                router_template="model.layers.{layer}.mlp.gate.weight",
                expert_gate_up_template="model.layers.{layer}.mlp.experts.gate_up_proj",
                expert_gate_template="model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                expert_up_template="model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                expert_down_template="model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                norm_templates=(
                    "model.layers.{layer}.post_attention_layernorm.weight",
                    "model.layers.{layer}.input_layernorm.weight",
                ),
                num_layers=int(text_config["num_hidden_layers"]),
                num_experts=int(text_config["num_experts"]),
                intermediate_size=int(text_config["moe_intermediate_size"]),
                router_top_k=int(text_config["num_experts_per_tok"]),
            )
        else:
            raise ValueError(f"Unsupported Pure-Pseudo model type: {model_type!r}.")
        adapter.validate_weight_map(weight_map)
        return adapter

    def validate_weight_map(self, weight_map: dict[str, str]) -> None:
        required = [self.router_template.format(layer=0)]
        down_name = self.expert_down_template.format(layer=0, expert=0)
        required.append(down_name)
        if self.expert_gate_template is None:
            required.append(self.expert_gate_up_template.format(layer=0))
        else:
            required.extend((self.expert_gate_template.format(layer=0, expert=0), self.expert_up_template.format(layer=0, expert=0)))
        missing = [name for name in required if name not in weight_map]
        if missing:
            raise KeyError(f"Missing {self.model_family} Pure-Pseudo tensors: {missing}")

    def layer_prefix(self, layer_id: int) -> str:
        return self.layer_prefix_template.format(layer=layer_id)

    def router_name(self, layer_id: int) -> str:
        return self.router_template.format(layer=layer_id)

    def expert_gate_up_name(self, layer_id: int) -> str:
        return self.expert_gate_up_template.format(layer=layer_id)

    def expert_down_name(self, layer_id: int) -> str:
        if "{expert}" in self.expert_down_template:
            raise ValueError("This model uses per-expert down_proj weights.")
        return self.expert_down_template.format(layer=layer_id)

    def expert_down_expert_name(self, layer_id: int, expert_id: int) -> str:
        if "{expert}" not in self.expert_down_template:
            raise ValueError("This model uses fused down_proj weights.")
        return self.expert_down_template.format(layer=layer_id, expert=expert_id)

    def expert_gate_name(self, layer_id: int, expert_id: int) -> str:
        if self.expert_gate_template is None:
            raise ValueError("This model uses fused gate_up_proj weights.")
        return self.expert_gate_template.format(layer=layer_id, expert=expert_id)

    def expert_up_name(self, layer_id: int, expert_id: int) -> str:
        if self.expert_up_template is None:
            raise ValueError("This model uses fused gate_up_proj weights.")
        return self.expert_up_template.format(layer=layer_id, expert=expert_id)

    def norm_names(self, layer_id: int) -> list[str]:
        return [template.format(layer=layer_id) for template in self.norm_templates]
