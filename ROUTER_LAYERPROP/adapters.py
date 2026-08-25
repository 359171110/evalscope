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
        probes = build_router_probes(directions, variants=variants, scale=scale)
        gate_up, _down = self.expert_weights(layer)
        return probes.to(device=gate_up.device, dtype=gate_up.dtype)

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

    def _text_root(self) -> torch.nn.Module:
        """Return the native text decoder root shared by Qwen wrappers."""

        root = getattr(self.model, "model", self.model)
        return getattr(root, "language_model", root)

    def _refresh_context(self, hidden_states: torch.Tensor) -> dict[str, Any]:
        """Prepare native masks and rotary embeddings for a stateless Qwen segment."""

        if self.metadata.family == "qwen3":
            from transformers.models.qwen3_moe import modeling_qwen3_moe as modeling

            root = self._text_root()
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            attention_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device
            )
            causal_mask = modeling.create_causal_mask(
                config=root.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = root.rotary_emb(hidden_states, position_ids=position_ids)
            return {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
            }
        if self.metadata.family == "qwen3.6":
            from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

            root = self._text_root()
            batch, sequence = hidden_states.shape[:2]
            all_position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            all_position_ids = all_position_ids.view(1, 1, -1).expand(4, batch, -1)
            text_position_ids = all_position_ids[0]
            rotary_position_ids = all_position_ids[1:]
            attention_mask = torch.ones((batch, sequence), dtype=torch.long, device=hidden_states.device)
            causal_mask = modeling.create_causal_mask(
                config=root.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            position_embeddings = root.rotary_emb(hidden_states, rotary_position_ids)
            return {
                "attention_mask": causal_mask,
                "linear_attention_mask": None,
                "position_ids": text_position_ids,
                "position_embeddings": position_embeddings,
            }
        raise NotImplementedError(f"Refresh propagation is not implemented for {self.metadata.family}")

    def run_refresh_window(
        self,
        bank: torch.Tensor,
        source_layer_id: int,
        target_layer_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run one Qwen refresh bank once and capture every target in its horizon."""

        if not self.supports_refresh_propagation:
            raise NotImplementedError(f"Refresh propagation is not implemented for {self.metadata.family}")
        targets = tuple(sorted(set(int(layer_id) for layer_id in target_layer_ids)))
        if not targets:
            raise ValueError("target_layer_ids must not be empty")
        if int(source_layer_id) > targets[0]:
            raise ValueError("source_layer_id must not exceed any target layer")
        root = self._text_root()
        source_layer = root.layers[int(source_layer_id)]
        context = self._refresh_context(bank)
        captured_raw: dict[int, torch.Tensor] = {}
        handles = []
        if int(source_layer_id) in targets:
            captured_raw[int(source_layer_id)] = bank.detach()
        for target_layer_id in targets:
            if target_layer_id == int(source_layer_id):
                continue
            target_module = self.capture_module(root.layers[target_layer_id])

            def capture_hook(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                layer_id: int = target_layer_id,
            ) -> None:
                del module
                if args and isinstance(args[0], torch.Tensor) and layer_id not in captured_raw:
                    captured_raw[layer_id] = args[0].detach()

            handles.append(target_module.register_forward_pre_hook(capture_hook))
        normalized = source_layer.post_attention_layernorm(bank)
        try:
            source_output = source_layer.mlp(normalized)
            if isinstance(source_output, (tuple, list)):
                source_output = source_output[0]
            hidden_states = bank + source_output
            for layer_id in range(int(source_layer_id) + 1, targets[-1] + 1):
                layer = root.layers[layer_id]
                if self.metadata.family == "qwen3":
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=context["attention_mask"],
                        position_ids=context["position_ids"],
                        past_key_values=None,
                        use_cache=False,
                        position_embeddings=context["position_embeddings"],
                    )
                else:
                    layer_mask = (
                        context["linear_attention_mask"]
                        if getattr(layer, "layer_type", "full_attention") == "linear_attention"
                        else context["attention_mask"]
                    )
                    hidden_states = layer(
                        hidden_states,
                        position_embeddings=context["position_embeddings"],
                        attention_mask=layer_mask,
                        position_ids=context["position_ids"],
                        past_key_values=None,
                    )
        finally:
            for handle in handles:
                handle.remove()
        missing = set(targets) - set(captured_raw)
        if missing:
            raise RuntimeError(f"Refresh window did not capture target layers: {sorted(missing)}")
        return hidden_states, captured_raw

    def run_refresh_segment(
        self,
        bank: torch.Tensor,
        source_layer_id: int,
        target_layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compatibility wrapper for one target within a refresh window."""

        hidden_states, captured = self.run_refresh_window(bank, source_layer_id, (target_layer_id,))
        return hidden_states, captured.get(int(target_layer_id))

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
    """Adapter for Qwen3's packed or separate gate/up/down routed experts."""

    metadata = AdapterMetadata("qwen3", 0, 0, 0, 0, "silu", False, 64)

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        layers = self.layers()
        first = layers[0]
        mlp = self._mlp(first)
        experts = getattr(mlp, "experts")
        packed = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
        if packed:
            hidden_size = int(experts.gate_up_proj.shape[-1])
            intermediate_size = int(experts.gate_up_proj.shape[1] // 2)
            num_experts = int(experts.gate_up_proj.shape[0])
        else:
            first_expert = experts[0]
            hidden_size = int(first_expert.gate_proj.weight.shape[1])
            intermediate_size = int(first_expert.gate_proj.weight.shape[0])
            num_experts = len(experts)
        gate = getattr(mlp, "gate")
        top_k = int(getattr(self.text_config(), "num_experts_per_tok", 1))
        self.metadata = AdapterMetadata(
            "qwen3", hidden_size, num_experts, top_k, intermediate_size, "silu", packed, 64
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
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            return experts.gate_up_proj.detach(), experts.down_proj.detach()
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
        probabilities = torch.softmax(logits.float(), dim=-1)
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
