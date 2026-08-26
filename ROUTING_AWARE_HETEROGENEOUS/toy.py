"""Tiny deterministic adapter used for smoke tests and development."""

from __future__ import annotations

import torch

from .adapter import ArchitectureAdapter, LayerTrace


class ToyAdapter(ArchitectureAdapter):
    """A GPU-compatible two-layer SwiGLU MoE with synthetic native routing."""

    def __init__(self, *, device: str = "cpu", hidden: int = 8, channels: int = 8, experts: int = 3) -> None:
        if hidden <= 0 or channels <= 0 or experts < 2:
            raise ValueError("toy dimensions are invalid")
        self._device = torch.device(device)
        generator = torch.Generator(device=self._device).manual_seed(17)
        self._gate = torch.randn(2, experts, hidden, device=self._device, generator=generator)
        self._gate_up = torch.randn(2, experts, 2 * channels, hidden, device=self._device, generator=generator)
        self._down = torch.randn(2, experts, hidden, channels, device=self._device, generator=generator)

    @property
    def num_layers(self) -> int:
        return 2

    @property
    def num_experts(self) -> int:
        return int(self._gate.shape[1])

    @property
    def source_width(self) -> int:
        return int(self._down.shape[-1])

    @property
    def device(self) -> torch.device:
        return self._device

    def collect(self, input_ids: torch.Tensor) -> tuple[LayerTrace, ...]:
        """Convert token ids into hidden vectors and run deterministic top-k routing."""

        values = input_ids.to(self._device, dtype=torch.float32)
        hidden = torch.stack([torch.sin(values + index) for index in range(self._gate.shape[-1])], dim=-1)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        traces = []
        for layer_id in range(self.num_layers):
            logits = hidden @ self._gate[layer_id].transpose(0, 1)
            weights, selected = torch.topk(torch.softmax(logits.float(), dim=-1), 1, dim=-1)
            traces.append(LayerTrace(hidden, selected, weights.to(hidden.dtype)))
            hidden = hidden + 0.01 * (weights * selected.float()).sum(dim=-1, keepdim=True)
        return tuple(traces)

    def expert_weights(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return toy packed gate/up and down weights."""

        return self._gate_up[int(layer_id)], self._down[int(layer_id)]

    def guided_input_ids(self, layer_id: int, expert_id: int, count: int, length: int) -> torch.Tensor:
        """Return deterministic token ids biased toward a target router direction."""

        del layer_id, expert_id
        return torch.arange(count * length, device=self._device, dtype=torch.long).reshape(count, length)