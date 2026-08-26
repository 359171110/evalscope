"""Architecture adapter contract and a Hugging Face model implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import torch


@dataclass
class LayerTrace:
    """Native routing observations for one decoder layer and one token batch."""

    expert_input: torch.Tensor | None
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    expert_samples: dict[int, torch.Tensor] | None = None


class ArchitectureAdapter(ABC):
    """Small architecture boundary used by the model-independent algorithm."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of routed MoE layers."""

    @property
    @abstractmethod
    def num_experts(self) -> int:
        """Return the routed expert count per layer."""

    @property
    @abstractmethod
    def source_width(self) -> int:
        """Return the unpruned expert intermediate width."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device used for model statistics."""

    @abstractmethod
    def collect(self, input_ids: torch.Tensor, max_samples_per_expert: int = 128) -> tuple[LayerTrace, ...]:
        """Run native forward and return one trace per MoE layer."""

    @abstractmethod
    def expert_weights(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed gate/up and down weights on the active device."""

    def guided_input_ids(self, layer_id: int, expert_id: int, count: int, length: int) -> torch.Tensor:
        """Create guided sequences; concrete model adapters may override this."""

        del layer_id, expert_id, count, length
        raise NotImplementedError("guided sequence generation is not implemented by this adapter")

    def counterfactual_damage(
        self,
        layer_id: int,
        expert_id: int,
        trace: LayerTrace,
        activation: torch.Tensor,
        down: torch.Tensor,
        retained: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample native MoE output damage for one retained set."""

        del layer_id
        selected = trace.selected_experts == int(expert_id)
        if not bool(selected.any()):
            return torch.zeros((), device=activation.device)
        full = torch.nn.functional.linear(activation.float(), down.float())
        kept = activation.float().index_select(1, retained.to(activation.device)).matmul(
            down.float().index_select(1, retained.to(activation.device)).transpose(0, 1)
        )
        slot_weights = trace.routing_weights.masked_select(selected).float()
        if slot_weights.numel() != activation.shape[0]:
            raise ValueError("trace must contain exactly one selected slot per conditional sample")
        return ((slot_weights.unsqueeze(-1) * (full - kept)).square().sum(dim=-1)).mean()


class HuggingFaceAdapter(ArchitectureAdapter):
    """Capture native traces from the tested repository MoE adapters."""

    def __init__(self, model: torch.nn.Module, native_adapter: Any) -> None:
        self.model = model
        self.native_adapter = native_adapter
        metadata = native_adapter.metadata
        self._num_layers = len(native_adapter.layers())
        self._num_experts = int(metadata.num_experts)
        self._source_width = int(metadata.intermediate_size)
        self._device = next(model.parameters()).device
        self._traces: list[LayerTrace] = []

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def source_width(self) -> int:
        return self._source_width

    @property
    def device(self) -> torch.device:
        return self._device

    def collect(self, input_ids: torch.Tensor, max_samples_per_expert: int = 128) -> tuple[LayerTrace, ...]:
        """Capture all layer expert inputs during one native forward."""

        self._traces = []
        handles = []
        layers = self.native_adapter.layers()
        for layer_id, layer in enumerate(layers):
            module = self.native_adapter.capture_module(layer)

            def capture_hook(
                _module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                current_layer: int = layer_id,
            ) -> None:
                raw = args[0]
                expert_input = self.native_adapter.expert_input(layers[current_layer], raw)
                indices, weights = self.native_adapter.route_from_captured(layers[current_layer], raw, expert_input)
                compact_samples = self._compact_expert_samples(indices, expert_input, max_samples_per_expert)
                self._traces.append(
                    (
                        current_layer,
                        LayerTrace(
                            None,
                            indices.detach(),
                            weights.detach(),
                            compact_samples,
                        ),
                    )
                )

            handles.append(module.register_forward_pre_hook(capture_hook))
        try:
            with torch.inference_mode():
                backbone = getattr(self.model, "model", None)
                if backbone is None:
                    raise AttributeError("Hugging Face causal LM does not expose a transformer backbone as .model")
                try:
                    backbone(input_ids=input_ids, use_cache=False, return_dict=False)
                except TypeError:
                    backbone(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        traces = dict(self._traces)
        return tuple(traces[layer_id] for layer_id in range(self.num_layers))

    def _compact_expert_samples(
        self,
        selected_experts: torch.Tensor,
        expert_input: torch.Tensor,
        maximum: int,
    ) -> dict[int, torch.Tensor]:
        """Group routed token rows with one GPU sort instead of one scan per expert."""

        flat_experts = selected_experts.reshape(-1).long()
        top_k = int(selected_experts.shape[1])
        token_ids = torch.arange(
            selected_experts.shape[0], device=selected_experts.device, dtype=torch.long
        ).repeat_interleave(top_k)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts.index_select(0, order)
        sorted_tokens = token_ids.index_select(0, order)
        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        samples: dict[int, torch.Tensor] = {}
        offset = 0
        for expert_id_tensor, count_tensor in zip(unique_experts, counts):
            count = int(count_tensor.item())
            expert_id = int(expert_id_tensor.item())
            token_rows = sorted_tokens[offset : offset + min(count, maximum)]
            samples[expert_id] = expert_input.index_select(0, token_rows).detach()
            offset += count
        return samples

    def expert_weights(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native packed expert weights."""

        return self.native_adapter.expert_weights(self.native_adapter.layers()[int(layer_id)])

    def guided_input_ids(self, layer_id: int, expert_id: int, count: int, length: int) -> torch.Tensor:
        """Generate model-native sequences from a router-aligned embedding anchor."""

        if count <= 0 or length <= 0:
            raise ValueError("count and length must be positive")
        layer = self.native_adapter.layers()[int(layer_id)]
        direction = self.native_adapter.router_effective_direction(layer)[int(expert_id)].to(self.device).float()
        embeddings = self.model.get_input_embeddings().weight.detach().to(self.device).float()
        candidates = torch.topk(
            torch.nn.functional.cosine_similarity(embeddings, direction.unsqueeze(0), dim=-1),
            k=min(16, embeddings.shape[0]),
        ).indices
        anchor = candidates[0].reshape(1, 1)
        prompt = anchor.expand(count, 1)
        attention_mask = torch.ones_like(prompt)
        generated = self.model.generate(
            input_ids=prompt,
            attention_mask=attention_mask,
            max_new_tokens=max(1, int(length) - 1),
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            use_cache=True,
            pad_token_id=getattr(self.model.config, "pad_token_id", None),
        )
        return generated[:, :length]


def adapter_from_model(model: torch.nn.Module) -> HuggingFaceAdapter:
    """Build the repository's native Qwen3/Qwen3.6/Gemma4 adapter."""

    from ROUTER_LAYERPROP.adapters import adapter_for_model

    return HuggingFaceAdapter(model, adapter_for_model(model))