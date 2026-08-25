"""Native synthetic propagation and per-layer response collection."""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Any

import torch

from .adapters import MoEAdapter
from .config import LayerPropConfig
from .core import collect_routed_rows


def supported_layer_ids(adapter: MoEAdapter) -> tuple[int, ...]:
    """Return layers whose routed expert tensors match the adapter contract."""

    supported: list[int] = []
    for layer_id, layer in enumerate(adapter.layers()):
        try:
            gate_up, down = adapter.expert_weights(layer)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            continue
        if gate_up.ndim == 3 and down.ndim == 3:
            supported.append(layer_id)
    if not supported:
        raise ValueError(f"No routed MoE layers found for model family {adapter.metadata.family}")
    return tuple(supported)


class _LayerResponseCapture(AbstractContextManager["_LayerResponseCapture"]):
    """Capture expert responses at the native expert-input normalization boundary."""

    def __init__(
        self,
        adapter: MoEAdapter,
        layer_ids: tuple[int, ...],
        max_rows_per_expert: int,
    ) -> None:
        self.adapter = adapter
        self.layer_ids = layer_ids
        self.max_rows_per_expert = max_rows_per_expert
        self.phase = "train"
        self.rows: dict[str, dict[int, dict[int, list[torch.Tensor]]]] = {
            "train": {layer_id: defaultdict(list) for layer_id in layer_ids},
            "valid": {layer_id: defaultdict(list) for layer_id in layer_ids},
        }
        self.handles: list[Any] = []

    def __enter__(self) -> "_LayerResponseCapture":
        for layer_id in self.layer_ids:
            layer = self.adapter.layers()[layer_id]
            module = self.adapter.capture_module(layer)
            self.handles.append(module.register_forward_hook(self._hook(layer_id)))
        return self

    def _hook(self, layer_id: int) -> Any:
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
            del module
            if not args or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                return
            raw = args[0].detach()
            expert_input = output.detach()
            raw_flat = raw.reshape(-1, raw.shape[-1])
            expert_flat = expert_input.reshape(-1, expert_input.shape[-1])
            layer = self.adapter.layers()[layer_id]
            top_k_index, top_k_weights = self.adapter.route_from_captured(layer, raw_flat, expert_flat)
            gate_up, _down = self.adapter.expert_weights(layer)
            collected = collect_routed_rows(
                expert_flat,
                top_k_index,
                top_k_weights,
                gate_up,
                activation=self.adapter.metadata.activation,
                max_rows_per_expert=self.max_rows_per_expert,
            )
            for expert, rows in collected.items():
                if rows.shape[0]:
                    self.rows[self.phase][layer_id][expert].append(rows.cpu())

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
        return None

    def finalize(self) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
        """Concatenate and cap captured train/validation rows per layer and expert."""

        finalized: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
        for phase in ("train", "valid"):
            finalized[phase] = {}
            for layer_id in self.layer_ids:
                finalized[phase][layer_id] = {}
                expert_rows = self.rows[phase][layer_id]
                for expert in range(self.adapter.metadata.num_experts):
                    chunks = expert_rows.get(expert, [])
                    if not chunks:
                        hidden = self.adapter.metadata.intermediate_size
                        finalized[phase][layer_id][expert] = torch.empty((0, hidden), dtype=torch.float32)
                        continue
                    combined = torch.cat(chunks, dim=0)[: self.max_rows_per_expert]
                    finalized[phase][layer_id][expert] = combined.float()
        return finalized["train"], finalized["valid"]


def run_source0_propagation(
    model: torch.nn.Module,
    adapter: MoEAdapter,
    input_ids: torch.Tensor,
    *,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    device: torch.device,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]]]:
    """Propagate deterministic token lattice through the untouched native model."""

    if input_ids.ndim != 2 or input_ids.shape[1] != config.sequence_length:
        raise ValueError("input_ids must have shape [sequences, config.sequence_length]")
    split = max(1, int(input_ids.shape[0]) // 2)
    split = min(split, max(1, int(input_ids.shape[0]) - 1)) if input_ids.shape[0] > 1 else int(input_ids.shape[0])
    embedding_device = model.get_input_embeddings().weight.device
    model.eval()
    with _LayerResponseCapture(adapter, layer_ids, config.max_rows_per_expert_per_origin) as capture:
        with torch.inference_mode():
            for sequence_id, row in enumerate(input_ids):
                capture.phase = "train" if sequence_id < split else "valid"
                batch = row.unsqueeze(0).to(device=embedding_device)
                attention_mask = torch.ones_like(batch)
                model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
        train_rows, valid_rows = capture.finalize()
    return train_rows, valid_rows


def collect_local_rows(
    adapter: MoEAdapter,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    scale: float,
) -> dict[int, dict[int, torch.Tensor]]:
    """Collect target-local router-region rows without running a corpus."""

    local: dict[int, dict[int, torch.Tensor]] = {}
    for layer_id in layer_ids:
        layer = adapter.layers()[layer_id]
        local[layer_id] = adapter.local_rows(
            layer,
            variants=config.probe_variants,
            scale=scale,
            max_rows_per_expert=config.max_rows_per_expert_per_origin,
        )
    return local
