from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class ExpertTensorCodec(str, Enum):
    SEPARATE = "separate"
    PACKED = "packed"


class BranchTopology(str, Enum):
    ROUTED_ONLY = "routed_only"
    GATED_SHARED = "gated_shared"
    DENSE_PLUS_SPARSE = "dense_plus_sparse"


class ExpertInputSemantics(str, Enum):
    SHARED_ROUTER_INPUT = "shared_router_input"
    INDEPENDENT_FROM_RAW_RESIDUAL = "independent_from_raw_residual"


@dataclass(frozen=True)
class ChannelArchitecture:
    model_family: str
    hidden_size: int
    source_intermediate_size: int
    num_layers: int
    num_experts: int
    router_top_k: int
    activation: str
    expert_tensor_codec: ExpertTensorCodec
    branch_topology: BranchTopology
    expert_input_semantics: ExpertInputSemantics
    channel_alignment: int
    dense_intermediate_size: int | None
    shared_expert_intermediate_size: int | None
    router_has_global_scale: bool
    router_has_per_expert_scale: bool

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.source_intermediate_size <= 0:
            raise ValueError("Channel architecture dimensions must be positive")
        if self.num_layers <= 0 or self.num_experts <= 0 or self.router_top_k <= 0:
            raise ValueError("Channel architecture routing dimensions must be positive")
        if self.router_top_k > self.num_experts:
            raise ValueError("Router top-k cannot exceed the expert count")
        if self.channel_alignment <= 0 or self.source_intermediate_size % self.channel_alignment:
            raise ValueError("Source intermediate size must be channel-aligned")

    def aligned_width(
        self,
        requested_width: int,
        rounding: Literal["floor", "nearest", "ceil"] = "nearest",
    ) -> int:
        if requested_width <= 0:
            raise ValueError("Requested channel width must be positive")
        if rounding not in {"floor", "nearest", "ceil"}:
            raise ValueError(f"Unsupported channel-width rounding mode: {rounding!r}")
        requested_width = min(int(requested_width), self.source_intermediate_size)
        lower = max(self.channel_alignment, requested_width // self.channel_alignment * self.channel_alignment)
        upper = min(
            self.source_intermediate_size,
            ((requested_width + self.channel_alignment - 1) // self.channel_alignment) * self.channel_alignment,
        )
        if rounding == "floor":
            return lower
        if rounding == "ceil":
            return upper
        return lower if requested_width - lower <= upper - requested_width else upper

    def width_for_retention(
        self,
        retention: float,
        rounding: Literal["floor", "nearest", "ceil"] = "nearest",
    ) -> int:
        if not 0.0 < retention <= 1.0:
            raise ValueError("Retention must be in the interval (0, 1]")
        return self.aligned_width(round(self.source_intermediate_size * retention), rounding)

    def validate_width(self, width: int) -> None:
        if not 0 < width <= self.source_intermediate_size:
            raise ValueError("Channel width must be positive and no larger than the source width")
        if width % self.channel_alignment:
            raise ValueError(f"Channel width must be divisible by {self.channel_alignment}")


@dataclass(frozen=True)
class PurePseudoModelAdapter:
    model_family: str
    text_config: dict[str, Any]
    router_template: str
    router_scale_template: str | None
    router_per_expert_scale_template: str | None
    expert_input_norm_template: str | None
    expert_gate_up_template: str
    expert_gate_template: str | None
    expert_up_template: str | None
    expert_down_template: str
    norm_templates: tuple[str, ...]
    num_layers: int
    num_experts: int
    intermediate_size: int
    router_top_k: int

    @property
    def channel_architecture(self) -> ChannelArchitecture:
        activation = str(self.text_config.get("hidden_activation", self.text_config.get("hidden_act", "silu")))
        if self.model_family == "gemma4":
            branch_topology = BranchTopology.DENSE_PLUS_SPARSE
            input_semantics = ExpertInputSemantics.INDEPENDENT_FROM_RAW_RESIDUAL
            alignment = 32
            dense_width = int(self.text_config["intermediate_size"])
            shared_width = None
        elif self.model_family == "qwen3.6":
            branch_topology = BranchTopology.GATED_SHARED
            input_semantics = ExpertInputSemantics.SHARED_ROUTER_INPUT
            alignment = 64
            dense_width = None
            shared_width = int(self.text_config["shared_expert_intermediate_size"])
        elif self.model_family == "qwen3":
            branch_topology = BranchTopology.ROUTED_ONLY
            input_semantics = ExpertInputSemantics.SHARED_ROUTER_INPUT
            alignment = 64
            dense_width = None
            shared_width = None
        else:
            raise ValueError(f"Unsupported channel architecture family: {self.model_family!r}")
        return ChannelArchitecture(
            model_family=self.model_family,
            hidden_size=int(self.text_config["hidden_size"]),
            source_intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            router_top_k=self.router_top_k,
            activation=activation,
            expert_tensor_codec=(
                ExpertTensorCodec.SEPARATE if self.expert_gate_template is not None else ExpertTensorCodec.PACKED
            ),
            branch_topology=branch_topology,
            expert_input_semantics=input_semantics,
            channel_alignment=alignment,
            dense_intermediate_size=dense_width,
            shared_expert_intermediate_size=shared_width,
            router_has_global_scale=self.router_scale_template is not None,
            router_has_per_expert_scale=self.router_per_expert_scale_template is not None,
        )

    @classmethod
    def from_checkpoint(cls, model_path: Path, weight_map: dict[str, str]) -> "PurePseudoModelAdapter":
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        model_type = str(text_config.get("model_type", config.get("model_type", ""))).lower()
        common = {
            "text_config": text_config,
            "norm_templates": (
                "model.language_model.layers.{layer}.post_attention_layernorm.weight",
                "model.language_model.layers.{layer}.input_layernorm.weight",
            ),
            "num_layers": int(text_config["num_hidden_layers"]),
            "num_experts": int(text_config["num_experts"]),
            "intermediate_size": int(text_config["moe_intermediate_size"]),
        }
        if model_type == "qwen3_5_moe_text":
            adapter = cls(
                model_family="qwen3.6",
                router_template="model.language_model.layers.{layer}.mlp.gate.weight",
                router_scale_template=None,
                router_per_expert_scale_template=None,
                expert_input_norm_template=None,
                expert_gate_up_template="model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
                expert_gate_template=None,
                expert_up_template=None,
                expert_down_template="model.language_model.layers.{layer}.mlp.experts.down_proj",
                router_top_k=int(text_config["num_experts_per_tok"]),
                **common
            )
        elif model_type == "gemma4_text":
            adapter = cls(
                model_family="gemma4",
                router_template="model.language_model.layers.{layer}.router.proj.weight",
                router_scale_template="model.language_model.layers.{layer}.router.scale",
                router_per_expert_scale_template="model.language_model.layers.{layer}.router.per_expert_scale",
                expert_input_norm_template="model.language_model.layers.{layer}.pre_feedforward_layernorm_2.weight",
                expert_gate_up_template="model.language_model.layers.{layer}.experts.gate_up_proj",
                expert_gate_template=None,
                expert_up_template=None,
                expert_down_template="model.language_model.layers.{layer}.experts.down_proj",
                router_top_k=int(text_config["top_k_experts"]),
                **common
            )
        elif model_type == "qwen3_moe":
            adapter = cls(
                model_family="qwen3",
                router_template="model.layers.{layer}.mlp.gate.weight",
                router_scale_template=None,
                router_per_expert_scale_template=None,
                expert_input_norm_template=None,
                expert_gate_up_template="model.layers.{layer}.mlp.experts.gate_up_proj",
                expert_gate_template="model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                expert_up_template="model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                expert_down_template="model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                norm_templates=(
                    "model.layers.{layer}.post_attention_layernorm.weight",
                    "model.layers.{layer}.input_layernorm.weight"
                ),
                router_top_k=int(text_config["num_experts_per_tok"]),
                **{
                    k: v
                    for k, v in common.items()
                    if k != "norm_templates"
                }
            )
        else:
            raise ValueError(f"Unsupported Pure-Pseudo model type: {model_type!r}")
        adapter.validate_weight_map(weight_map)
        return adapter

    def validate_weight_map(self, weight_map: dict[str, str]) -> None:
        required = [
            self.router_template.format(layer=0),
            self.expert_down_template.format(layer=0, expert=0)
            if "{expert}" in self.expert_down_template else self.expert_down_template.format(layer=0)
        ]
        if self.expert_gate_template is None:
            required.append(self.expert_gate_up_template.format(layer=0))
        else:
            required.extend((
                self.expert_gate_template.format(layer=0, expert=0), self.expert_up_template.format(layer=0, expert=0)
            ))
        missing = [name for name in required if name not in weight_map]
        if missing:
            raise KeyError(f"Missing {self.model_family} tensors: {missing}")

    def router_name(self, layer_id: int) -> str:
        return self.router_template.format(layer=layer_id)

    def router_scale_name(self, layer_id: int) -> str:
        if self.router_scale_template is None:
            raise ValueError("Model does not use a router scale tensor")
        return self.router_scale_template.format(layer=layer_id)

    def router_per_expert_scale_name(self, layer_id: int) -> str:
        if self.router_per_expert_scale_template is None:
            raise ValueError("Model does not use per-expert router scales")
        return self.router_per_expert_scale_template.format(layer=layer_id)

    def expert_input_norm_name(self, layer_id: int) -> str:
        if self.expert_input_norm_template is None:
            raise ValueError("Model does not use a separate expert-input norm")
        return self.expert_input_norm_template.format(layer=layer_id)

    def expert_gate_up_name(self, layer_id: int) -> str:
        return self.expert_gate_up_template.format(layer=layer_id)

    def expert_down_name(self, layer_id: int) -> str:
        if "{expert}" in self.expert_down_template:
            raise ValueError("Model uses per-expert down projection weights")
        return self.expert_down_template.format(layer=layer_id)

    def expert_down_expert_name(self, layer_id: int, expert_id: int) -> str:
        if "{expert}" not in self.expert_down_template:
            raise ValueError("Model uses fused down projection weights")
        return self.expert_down_template.format(layer=layer_id, expert=expert_id)

    def expert_gate_name(self, layer_id: int, expert_id: int) -> str:
        if self.expert_gate_template is None:
            raise ValueError("Model uses fused gate_up projection weights")
        return self.expert_gate_template.format(layer=layer_id, expert=expert_id)

    def expert_up_name(self, layer_id: int, expert_id: int) -> str:
        if self.expert_up_template is None:
            raise ValueError("Model uses fused gate_up projection weights")
        return self.expert_up_template.format(layer=layer_id, expert=expert_id)

    def norm_names(self, layer_id: int) -> list[str]:
        return [template.format(layer=layer_id) for template in self.norm_templates]
