"""Native routing and expert-layout adapters for Qwen3, Qwen3.6, and Gemma4."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .core import build_router_probes, route_topk


@dataclass(frozen=True)
class AdapterMetadata:
    """Static architecture information needed by the pruning pipeline."""

    family: str
    hidden_size: int
    num_experts: int
    top_k: int
    intermediate_size: int
    activation: str
    packed_experts: bool
    channel_multiple: int


class MoEAdapter(ABC):
    """Small model-facing contract used by the data-free pipeline."""

    metadata: AdapterMetadata

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @abstractmethod
    def layers(self) -> tuple[torch.nn.Module, ...]:
        """Return decoder layers that contain routed MoE blocks."""

    @abstractmethod
    def router_effective_direction(self, layer: torch.nn.Module) -> torch.Tensor:
        """Return expert directions in raw residual coordinates."""

    @abstractmethod
    def native_route(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the native router and return top-k indices and weights."""

    def route_from_captured(
        self,
        layer: torch.nn.Module,
        raw_residual: torch.Tensor,
        expert_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route a captured norm input/output pair without recursively calling the norm."""

        del expert_input
        return self.native_route(layer, raw_residual)

    @abstractmethod
    def expert_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        """Return the exact routed expert input for a raw residual."""

    @abstractmethod
    def expert_weights(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed gate/up and down tensors as [experts, ...]."""

    def router_probe_bank(self, layer: torch.nn.Module, variants: int, scale: float) -> torch.Tensor:
        """Build local probes in the raw residual coordinate system."""

        directions = self.router_effective_direction(layer)
        return build_router_probes(directions, variants=variants, scale=scale)

    def local_rows(
        self,
        layer: torch.nn.Module,
        *,
        variants: int,
        scale: float,
        max_rows_per_expert: int,
    ) -> dict[int, torch.Tensor]:
        """Score a target layer from deterministic router-region probes."""

        from .core import collect_routed_rows

        probes = self.router_probe_bank(layer, variants, scale)
        raw = probes.reshape(-1, probes.shape[-1])
        top_k_index, top_k_weights = self.native_route(layer, raw)
        expert_input = self.expert_input(layer, raw)
        gate_up, _down = self.expert_weights(layer)
        return collect_routed_rows(
            expert_input,
            top_k_index,
            top_k_weights,
            gate_up,
            activation=self.metadata.activation,
            max_rows_per_expert=max_rows_per_expert,
        )

    def module_for_layer(self, layer: torch.nn.Module) -> torch.nn.Module:
        """Return the decoder submodule whose pre-hook receives raw residuals."""

        return layer

    @abstractmethod
    def capture_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        """Return the norm whose input is the raw post-attention residual."""

    def refresh_attention_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        """Return the attention module used to inject a Qwen refresh bank."""

        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise AttributeError("Decoder layer does not expose self_attn")
        return attention

    @property
    def supports_refresh_propagation(self) -> bool:
        """Whether stateless intermediate refresh injection is supported."""

        return self.metadata.family != "gemma4"

    def text_config(self) -> Any:
        """Return a nested text config when the checkpoint is multimodal."""

        config = getattr(self.model, "config", object())
        return getattr(config, "text_config", config)

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = MoEAdapter._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _flatten_hidden(raw_residual: torch.Tensor) -> torch.Tensor:
        if raw_residual.ndim < 2:
            raise ValueError("raw residual must have shape [tokens, hidden] or [batch, tokens, hidden]")
        return raw_residual.reshape(-1, raw_residual.shape[-1])

    @staticmethod
    def _norm_input(
        raw_residual: torch.Tensor,
        norm: torch.nn.Module,
        *,
        add_one_to_weight: bool = False,
    ) -> torch.Tensor:
        value = raw_residual.float()
        weight = getattr(norm, "weight", None)
        if weight is None:
            return norm(raw_residual)
        gamma = weight.float()
        if add_one_to_weight:
            gamma = gamma + 1.0
        epsilon = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1.0e-6)))
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
        return (normalized * gamma).to(dtype=raw_residual.dtype)


class Qwen3MoeAdapter(MoEAdapter):
    """Adapter for Qwen3's separate gate/up/down routed experts."""

    metadata = AdapterMetadata("qwen3", 0, 0, 0, 0, "silu", False, 64)

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        layers = self.layers()
        first = layers[0]
        mlp = self._mlp(first)
        experts = getattr(mlp, "experts")
        first_expert = experts[0]
        hidden_size = int(first_expert.gate_proj.weight.shape[1])
        intermediate_size = int(first_expert.gate_proj.weight.shape[0])
        gate = getattr(mlp, "gate")
        top_k = int(getattr(self.text_config(), "num_experts_per_tok", 1))
        self.metadata = AdapterMetadata(
            "qwen3", hidden_size, len(experts), top_k, intermediate_size, "silu", False, 64
        )

    def layers(self) -> tuple[torch.nn.Module, ...]:
        root = getattr(self.model, "model", self.model)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise AttributeError("Qwen3 model does not expose model.layers")
        return tuple(layers)

    @staticmethod
    def _mlp(layer: torch.nn.Module) -> torch.nn.Module:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise AttributeError("Qwen3 decoder layer does not expose mlp")
        return mlp

    def _router_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        return self._norm_input(raw_residual, layer.post_attention_layernorm)

    def _route_logits(self, layer: torch.nn.Module, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = self._mlp(layer).gate
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, indices = torch.topk(probabilities, self.metadata.top_k, dim=-1)
        norm_topk_prob = bool(
            getattr(self._mlp(layer), "norm_topk_prob", getattr(router, "norm_topk_prob", True))
        )
        if norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        weights = weights * float(getattr(router, "routed_scaling_factor", 1.0))
        return indices, weights.to(dtype=logits.dtype)

    def _native_router(self, layer: torch.nn.Module, expert_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = self._mlp(layer).gate
        result = router(expert_input)
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            return result[2].long(), result[1].to(dtype=expert_input.dtype)
        logits = result[0] if isinstance(result, (tuple, list)) else result
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Qwen3 router must return logits or (logits, weights, indices)")
        return self._route_logits(layer, logits.float())

    def router_effective_direction(self, layer: torch.nn.Module) -> torch.Tensor:
        mlp = self._mlp(layer)
        direction = mlp.gate.weight.detach().float()
        gamma = layer.post_attention_layernorm.weight.detach().float().clamp_min(1.0e-6)
        return direction / gamma.unsqueeze(0)

    def native_route(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._native_router(layer, self._flatten_hidden(self._router_input(layer, raw_residual)))

    def route_from_captured(
        self,
        layer: torch.nn.Module,
        raw_residual: torch.Tensor,
        expert_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del raw_residual
        return self._native_router(layer, self._flatten_hidden(expert_input))

    def expert_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        return self._flatten_hidden(self._router_input(layer, raw_residual))

    def expert_weights(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        experts = self._mlp(layer).experts
        gate_up = []
        down = []
        for expert in experts:
            gate_up.append(torch.cat((expert.gate_proj.weight.detach(), expert.up_proj.weight.detach()), dim=0))
            down.append(expert.down_proj.weight.detach())
        return torch.stack(gate_up), torch.stack(down)

    def capture_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        return layer.post_attention_layernorm


class Qwen35MoeAdapter(MoEAdapter):
    """Adapter for the current Qwen3.6 ``qwen3_5_moe`` implementation."""

    metadata = AdapterMetadata("qwen3.6", 0, 0, 0, 0, "silu", True, 64)

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        layers = self.layers()
        experts = self._mlp(layers[0]).experts
        gate_up = experts.gate_up_proj
        hidden_size = int(gate_up.shape[-1])
        intermediate_size = int(gate_up.shape[1] // 2)
        config = self.text_config()
        self.metadata = AdapterMetadata(
            "qwen3.6",
            hidden_size,
            int(gate_up.shape[0]),
            int(getattr(config, "num_experts_per_tok", 1)),
            intermediate_size,
            "silu",
            True,
            64,
        )

    def layers(self) -> tuple[torch.nn.Module, ...]:
        root = getattr(self.model, "model", self.model)
        root = getattr(root, "language_model", root)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise AttributeError("Qwen3.6 model does not expose decoder layers")
        return tuple(layers)

    @staticmethod
    def _mlp(layer: torch.nn.Module) -> torch.nn.Module:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise AttributeError("Qwen3.6 decoder layer does not expose mlp")
        return mlp

    def _router_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        return self._norm_input(raw_residual, layer.post_attention_layernorm, add_one_to_weight=True)

    def _route_logits(self, layer: torch.nn.Module, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = self._mlp(layer).gate
        scores = torch.sigmoid(logits.float())
        weights, indices = torch.topk(scores, self.metadata.top_k, dim=-1)
        norm_topk_prob = bool(
            getattr(self._mlp(layer), "norm_topk_prob", getattr(router, "norm_topk_prob", True))
        )
        if norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        weights = weights * float(getattr(router, "routed_scaling_factor", 1.0))
        return indices, weights.to(dtype=logits.dtype)

    def _native_router(self, layer: torch.nn.Module, expert_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = self._mlp(layer).gate
        result = router(expert_input)
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            return result[2].long(), result[1].to(dtype=expert_input.dtype)
        logits = result[0] if isinstance(result, (tuple, list)) else result
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Qwen3.6 router must return logits or (logits, weights, indices)")
        return self._route_logits(layer, logits.float())

    def router_effective_direction(self, layer: torch.nn.Module) -> torch.Tensor:
        mlp = self._mlp(layer)
        direction = mlp.gate.weight.detach().float()
        gamma = (1.0 + layer.post_attention_layernorm.weight.detach().float()).clamp_min(1.0e-6)
        return direction / gamma.unsqueeze(0)

    def native_route(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._native_router(layer, self._flatten_hidden(self._router_input(layer, raw_residual)))

    def route_from_captured(
        self,
        layer: torch.nn.Module,
        raw_residual: torch.Tensor,
        expert_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del raw_residual
        return self._native_router(layer, self._flatten_hidden(expert_input))

    def expert_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        return self._flatten_hidden(self._router_input(layer, raw_residual))

    def expert_weights(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        experts = self._mlp(layer).experts
        return experts.gate_up_proj.detach(), experts.down_proj.detach()

    def capture_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        return layer.post_attention_layernorm


class Gemma4MoeAdapter(MoEAdapter):
    """Adapter for Gemma4's raw-residual router and separate expert norm."""

    metadata = AdapterMetadata("gemma4", 0, 0, 0, 0, "gelu_pytorch_tanh", True, 32)

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        layers = self.layers()
        layer = layers[0]
        experts = self._experts(layer)
        gate_up = experts.gate_up_proj
        config = self.text_config()
        self.metadata = AdapterMetadata(
            "gemma4",
            int(gate_up.shape[-1]),
            int(gate_up.shape[0]),
            int(getattr(config, "top_k_experts", 1)),
            int(gate_up.shape[1] // 2),
            "gelu_pytorch_tanh",
            True,
            32,
        )

    def layers(self) -> tuple[torch.nn.Module, ...]:
        root = getattr(self.model, "model", self.model)
        root = getattr(root, "language_model", root)
        layers = getattr(root, "layers", None)
        if layers is None:
            raise AttributeError("Gemma4 model does not expose language decoder layers")
        return tuple(layers)

    @staticmethod
    def _experts(layer: torch.nn.Module) -> torch.nn.Module:
        experts = getattr(layer, "experts", None)
        if experts is None:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None:
            raise AttributeError("Gemma4 layer does not expose packed experts")
        return experts

    def _router(self, layer: torch.nn.Module) -> torch.nn.Module:
        router = getattr(layer, "router", None)
        if router is None:
            raise AttributeError("Gemma4 layer does not expose router")
        return router

    def router_effective_direction(self, layer: torch.nn.Module) -> torch.Tensor:
        router = self._router(layer)
        direction = router.proj.weight.detach().float()
        scale = getattr(router, "scale", None)
        if scale is not None:
            direction = direction * scale.detach().float().reshape(1, -1)
        return direction

    def native_route(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = self._flatten_hidden(raw_residual)
        result = self._router(layer)(flat)
        tensors = list(result) if isinstance(result, (tuple, list)) else [result]
        indices = next((item for item in tensors if item.dtype in {torch.int32, torch.int64} and item.ndim == 2), None)
        weights = next((item for item in tensors if item.is_floating_point() and item.ndim == 2 and item.shape[1] == self.metadata.top_k), None)
        if indices is not None and weights is not None:
            return indices.long(), weights.float().to(dtype=flat.dtype)
        logits = next((item for item in tensors if item.is_floating_point() and item.shape[-1] == self.metadata.num_experts), None)
        if logits is None:
            raise RuntimeError("Could not recover Gemma4 native router logits or top-k outputs")
        return route_topk(logits.float(), self.metadata.top_k, scoring="sigmoid")

    def expert_input(self, layer: torch.nn.Module, raw_residual: torch.Tensor) -> torch.Tensor:
        norm = getattr(layer, "pre_feedforward_layernorm_2", None)
        if norm is None:
            raise AttributeError("Gemma4 layer does not expose pre_feedforward_layernorm_2")
        return self._flatten_hidden(norm(raw_residual))

    def expert_weights(self, layer: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        experts = self._experts(layer)
        return experts.gate_up_proj.detach(), experts.down_proj.detach()

    def capture_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        norm = getattr(layer, "pre_feedforward_layernorm_2", None)
        if norm is None:
            raise AttributeError("Gemma4 layer does not expose pre_feedforward_layernorm_2")
        return norm


def adapter_for_model(model: torch.nn.Module) -> MoEAdapter:
    """Select an adapter from the Transformers model/config without importing Transformers."""

    config = getattr(model, "config", object())
    text_config = getattr(config, "text_config", config)
    model_type = str(getattr(text_config, "model_type", getattr(config, "model_type", ""))).lower()
    if model_type == "qwen3_moe":
        return Qwen3MoeAdapter(model)
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return Qwen35MoeAdapter(model)
    if model_type in {"gemma4_text", "gemma4"}:
        return Gemma4MoeAdapter(model)
    raise ValueError(f"Unsupported model_type: {model_type!r}")
