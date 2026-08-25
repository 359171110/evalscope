"""Native synthetic propagation and per-layer response collection."""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from math import ceil
from typing import Any

import torch

from .adapters import MoEAdapter
from .config import LayerPropConfig
from .core import collect_routed_rows


def refresh_source_targets(
    layer_ids: tuple[int, ...],
    *,
    stride: int,
    horizon: int,
) -> dict[int, tuple[int, ...]]:
    """Map each refresh source to targets that select it as their nearest short-timescale origin."""

    if not layer_ids:
        return {}
    if stride <= 0 or horizon <= 0:
        raise ValueError("stride and horizon must be positive")
    ordered = tuple(sorted(set(int(layer_id) for layer_id in layer_ids)))
    first_layer = ordered[0]
    sources = tuple(
        layer_id
        for layer_id in ordered
        if layer_id > first_layer and (layer_id - first_layer) % int(stride) == 0
    )
    selected: dict[int, list[int]] = {source: [] for source in sources}
    for target in ordered:
        eligible = [source for source in sources if source <= target and target - source <= int(horizon)]
        if eligible:
            selected[max(eligible)].append(target)
    return {source: tuple(targets) for source, targets in selected.items() if targets}


def pack_refresh_probes(
    adapter: MoEAdapter,
    layer: torch.nn.Module,
    config: LayerPropConfig,
    scale: float,
) -> torch.Tensor:
    """Pack a fixed pseudo-token budget as expert-interleaved refresh sequences."""

    experts = adapter.metadata.num_experts
    variants = max(config.probe_variants, ceil(config.num_pseudo_tokens / experts))
    probes = adapter.router_probe_bank(layer, variants=variants, scale=scale)
    rows = []
    expert_ids = torch.arange(experts, device=probes.device)
    for variant in range(variants):
        shift = (variant * 17) % experts
        order = (expert_ids + shift) % experts
        rows.append(probes[:, variant].index_select(0, order))
    packed = torch.cat(rows, dim=0)[: config.num_pseudo_tokens]
    return packed.view(config.num_sequences, config.sequence_length, packed.shape[-1])


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
        self.scales: dict[str, dict[int, list[float]]] = {
            "train": {layer_id: [] for layer_id in layer_ids},
            "valid": {layer_id: [] for layer_id in layer_ids},
        }
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
            self.scales[self.phase][layer_id].append(float(raw.float().square().mean().sqrt().item()))
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

    def finalize(
        self,
    ) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[int, float]]:
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
        scales = {}
        for layer_id in self.layer_ids:
            values = self.scales["train"][layer_id] + self.scales["valid"][layer_id]
            scales[layer_id] = float(torch.tensor(values).median().item()) if values else 1.0
        return finalized["train"], finalized["valid"], scales


def run_source0_propagation(
    model: torch.nn.Module,
    adapter: MoEAdapter,
    input_ids: torch.Tensor,
    *,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    device: torch.device,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[int, float]]:
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
        train_rows, valid_rows, scales = capture.finalize()
    return train_rows, valid_rows, scales


def run_router_long_propagation(
    adapter: MoEAdapter,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    source_scale: float,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, dict[int, torch.Tensor]], dict[int, float]]:
    """Propagate a source-0 router bank through every Qwen target layer once."""

    if not adapter.supports_refresh_propagation:
        raise NotImplementedError(f"Router long propagation is not implemented for {adapter.metadata.family}")
    if not layer_ids:
        raise ValueError("layer_ids must not be empty")
    source_layer = int(layer_ids[0])
    bank = pack_refresh_probes(
        adapter,
        adapter.layers()[source_layer],
        config,
        scale=float(source_scale),
    )
    with torch.inference_mode():
        _hidden, raw_by_target = adapter.run_refresh_window(bank, source_layer, layer_ids)
    split = max(1, int(bank.shape[0]) // 2)
    split = min(split, max(1, int(bank.shape[0]) - 1)) if bank.shape[0] > 1 else int(bank.shape[0])
    train_rows: dict[int, dict[int, torch.Tensor]] = {}
    valid_rows: dict[int, dict[int, torch.Tensor]] = {}
    scales: dict[int, float] = {}
    for layer_id in layer_ids:
        raw = raw_by_target[layer_id]
        train_rows[layer_id] = collect_rows_from_raw(
            adapter,
            layer_id,
            raw[:split],
            config.max_rows_per_expert_per_origin,
        )
        valid_rows[layer_id] = collect_rows_from_raw(
            adapter,
            layer_id,
            raw[split:],
            config.max_rows_per_expert_per_origin,
        )
        scales[layer_id] = float(raw.float().square().mean().sqrt().clamp_min(config.epsilon).item())
    return train_rows, valid_rows, scales


def collect_local_rows(
    adapter: MoEAdapter,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    scales: dict[int, float],
    fallback_scale: float,
) -> dict[int, dict[int, torch.Tensor]]:
    """Collect target-local router-region rows without running a corpus."""

    local: dict[int, dict[int, torch.Tensor]] = {}
    with torch.inference_mode():
        for layer_id in layer_ids:
            layer = adapter.layers()[layer_id]
            local[layer_id] = adapter.local_rows(
                layer,
                variants=config.probe_variants,
                scale=float(scales.get(layer_id, fallback_scale)),
                max_rows_per_expert=config.max_rows_per_expert_per_origin,
            )
    return local


def collect_rows_from_raw(
    adapter: MoEAdapter,
    layer_id: int,
    raw_residual: torch.Tensor,
    max_rows_per_expert: int,
) -> dict[int, torch.Tensor]:
    """Collect routed expert rows from a synthetic raw residual bank."""

    layer = adapter.layers()[int(layer_id)]
    flat_raw = raw_residual.reshape(-1, raw_residual.shape[-1])
    expert_input = adapter.expert_input(layer, flat_raw)
    top_k_index, top_k_weights = adapter.native_route(layer, flat_raw)
    gate_up, _down = adapter.expert_weights(layer)
    return collect_routed_rows(
        expert_input,
        top_k_index,
        top_k_weights,
        gate_up,
        activation=adapter.metadata.activation,
        max_rows_per_expert=max_rows_per_expert,
    )


def run_refresh_propagation(
    adapter: MoEAdapter,
    layer_ids: tuple[int, ...],
    config: LayerPropConfig,
    scales: dict[int, float],
    fallback_scale: float,
) -> dict[str, dict[int, dict[int, torch.Tensor]]]:
    """Build refresh origins and propagate each only across its configured horizon."""

    if not adapter.supports_refresh_propagation:
        return {}
    schedule = refresh_source_targets(
        layer_ids,
        stride=config.refresh_stride,
        horizon=config.refresh_horizon,
    )
    refresh_rows: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
    for source_layer, target_layers in schedule.items():
        source_name = f"refresh_{source_layer}"
        with torch.inference_mode():
            bank = pack_refresh_probes(
                adapter,
                adapter.layers()[source_layer],
                config,
                scale=float(scales.get(source_layer, fallback_scale)),
            )
            refresh_rows[source_name] = {}
            _hidden, raw_by_target = adapter.run_refresh_window(bank, source_layer, target_layers)
            for target_layer in target_layers:
                refresh_rows[source_name][target_layer] = collect_rows_from_raw(
                    adapter,
                    target_layer,
                    raw_by_target[target_layer],
                    config.max_rows_per_expert_per_origin,
                )
    return refresh_rows
