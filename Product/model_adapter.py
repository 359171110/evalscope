from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ProductArchitecture:
    """Frozen routed-expert structure needed by uniform Product pruning."""

    model_family: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_experts: int
    router_top_k: int
    activation: str
    tensor_codec: Literal["separate", "packed"]
    channel_alignment: int
    branch_topology: Literal["routed_only", "gated_shared", "dense_plus_sparse"]
    first_k_dense_replace: int = 0

    def moe_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(int(self.first_k_dense_replace), int(self.num_layers)))

    def aligned_width(self, requested_width: int, rounding: str = "nearest") -> int:
        if requested_width <= 0:
            raise ValueError("Requested channel width must be positive.")
        if rounding not in {"floor", "nearest", "ceil"}:
            raise ValueError("rounding must be floor, nearest, or ceil.")
        requested = min(int(requested_width), self.intermediate_size)
        lower = max(self.channel_alignment, requested // self.channel_alignment * self.channel_alignment)
        upper = min(
            self.intermediate_size,
            ((requested + self.channel_alignment - 1) // self.channel_alignment) * self.channel_alignment,
        )
        if rounding == "floor":
            return lower
        if rounding == "ceil":
            return upper
        return lower if requested - lower <= upper - requested else upper

    def width_for_pruning(self, pruning_ratio: float, rounding: str = "nearest") -> int:
        if not 0.0 <= pruning_ratio < 1.0:
            raise ValueError("pruning_ratio must be in [0, 1).")
        requested = round(self.intermediate_size * (1.0 - float(pruning_ratio)))
        return self.aligned_width(requested, rounding)

    def validate_width(self, width: int) -> None:
        if not 0 < int(width) < self.intermediate_size:
            raise ValueError("Retained width must be positive and smaller than the source width.")
        if int(width) % self.channel_alignment:
            raise ValueError(f"Retained width must be divisible by {self.channel_alignment}.")


@dataclass(frozen=True)
class ProductModelAdapter:
    """Resolve routed-expert tensor layouts without touching shared or dense branches."""

    architecture: ProductArchitecture
    text_config: dict[str, Any]
    router_template: str
    gate_up_template: str | None
    gate_template: str | None
    up_template: str | None
    down_template: str

    @classmethod
    def from_checkpoint(cls, model_path: Path, weight_map: dict[str, str]) -> "ProductModelAdapter":
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config", config)
        model_type = str(text_config.get("model_type", config.get("model_type", ""))).lower()
        num_experts = int(text_config.get("num_experts", text_config.get("n_routed_experts", 0)))
        common = {
            "hidden_size": int(text_config["hidden_size"]),
            "intermediate_size": int(text_config["moe_intermediate_size"]),
            "num_layers": int(text_config["num_hidden_layers"]),
            "num_experts": num_experts,
            "activation": str(text_config.get("hidden_activation", text_config.get("hidden_act", "silu"))),
        }
        if model_type == "qwen3_moe":
            architecture = ProductArchitecture(
                model_family="qwen3",
                model_type=model_type,
                router_top_k=int(text_config["num_experts_per_tok"]),
                tensor_codec="separate",
                channel_alignment=64,
                branch_topology="routed_only",
                **common,
            )
            adapter = cls(
                architecture=architecture,
                text_config=text_config,
                router_template="model.layers.{layer}.mlp.gate.weight",
                gate_up_template=None,
                gate_template="model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                up_template="model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                down_template="model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
            )
        elif model_type == "gemma4_text":
            architecture = ProductArchitecture(
                model_family="gemma4",
                model_type=model_type,
                router_top_k=int(text_config["top_k_experts"]),
                tensor_codec="packed",
                channel_alignment=32,
                branch_topology="dense_plus_sparse",
                **common,
            )
            adapter = cls(
                architecture=architecture,
                text_config=text_config,
                router_template="model.language_model.layers.{layer}.router.proj.weight",
                gate_up_template="model.language_model.layers.{layer}.experts.gate_up_proj",
                gate_template=None,
                up_template=None,
                down_template="model.language_model.layers.{layer}.experts.down_proj",
            )
        elif model_type == "qwen3_5_moe_text":
            architecture = ProductArchitecture(
                model_family="qwen3.6",
                model_type=model_type,
                router_top_k=int(text_config["num_experts_per_tok"]),
                tensor_codec="packed",
                channel_alignment=64,
                branch_topology="gated_shared",
                **common,
            )
            adapter = cls(
                architecture=architecture,
                text_config=text_config,
                router_template="model.language_model.layers.{layer}.mlp.gate.weight",
                gate_up_template="model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
                gate_template=None,
                up_template=None,
                down_template="model.language_model.layers.{layer}.mlp.experts.down_proj",
            )
        elif model_type == "deepseek_v2":
            architecture = ProductArchitecture(
                model_family="deepseek_v2",
                model_type=model_type,
                router_top_k=int(text_config["num_experts_per_tok"]),
                tensor_codec="separate",
                channel_alignment=32,
                branch_topology="gated_shared",
                first_k_dense_replace=int(text_config.get("first_k_dense_replace", 0)),
                **common,
            )
            adapter = cls(
                architecture=architecture,
                text_config=text_config,
                router_template="model.layers.{layer}.mlp.gate.weight",
                gate_up_template=None,
                gate_template="model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                up_template="model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                down_template="model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
            )
        else:
            raise ValueError(f"Unsupported Product model type: {model_type!r}.")
        adapter.validate_weight_map(weight_map)
        return adapter

    def validate_weight_map(self, weight_map: dict[str, str]) -> None:
        first_layer = self.architecture.moe_layer_ids()[0]
        required = [self.router_name(first_layer), self.down_name(first_layer, 0)]
        if self.architecture.tensor_codec == "packed":
            required.append(self.gate_up_name(first_layer))
        else:
            required.extend((self.gate_name(first_layer, 0), self.up_name(first_layer, 0)))
        missing = [name for name in required if name not in weight_map]
        if missing:
            raise KeyError(f"Missing {self.architecture.model_family} Product tensors: {missing}")

    def router_name(self, layer_id: int) -> str:
        return self.router_template.format(layer=layer_id)

    def gate_up_name(self, layer_id: int) -> str:
        if self.gate_up_template is None:
            raise ValueError("This model uses separate gate and up tensors.")
        return self.gate_up_template.format(layer=layer_id)

    def gate_name(self, layer_id: int, expert_id: int) -> str:
        if self.gate_template is None:
            raise ValueError("This model uses a packed gate_up tensor.")
        return self.gate_template.format(layer=layer_id, expert=expert_id)

    def up_name(self, layer_id: int, expert_id: int) -> str:
        if self.up_template is None:
            raise ValueError("This model uses a packed gate_up tensor.")
        return self.up_template.format(layer=layer_id, expert=expert_id)

    def down_name(self, layer_id: int, expert_id: int | None = None) -> str:
        if "{expert}" in self.down_template:
            if expert_id is None:
                raise ValueError("expert_id is required for separate expert tensors.")
            return self.down_template.format(layer=layer_id, expert=expert_id)
        return self.down_template.format(layer=layer_id)

    def routed_tensor_names(self, layer_id: int) -> list[str]:
        architecture = self.architecture
        if architecture.tensor_codec == "packed":
            return [self.gate_up_name(layer_id), self.down_name(layer_id)]
        names = []
        for expert_id in range(architecture.num_experts):
            names.extend((
                self.gate_name(layer_id, expert_id),
                self.up_name(layer_id, expert_id),
                self.down_name(layer_id, expert_id),
            ))
        return names

    def metadata(self) -> dict[str, Any]:
        return asdict(self.architecture)
