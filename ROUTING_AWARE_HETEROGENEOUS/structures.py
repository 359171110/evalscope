"""Small tensor containers used by the pruning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CalibrationPools:
    """Natural and guided expert-conditioned samples kept on one device."""

    natural: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    guided: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    natural_mass: torch.Tensor | None = None
    natural_visitation: torch.Tensor | None = None
    guided_sequences_used: int = 0

    def combined(self, key: tuple[int, int], minimum: int, maximum: int) -> torch.Tensor:
        """Return natural samples, completed with guided samples when needed."""

        natural = self.natural.get(key)
        guided = self.guided.get(key)
        natural_count = 0 if natural is None else int(natural.shape[0])
        if natural_count >= minimum:
            return natural[:maximum]
        parts = [part for part in (natural, guided) if part is not None and part.shape[0] > 0]
        if not parts:
            return torch.empty((0, 0), device=self._device())
        return torch.cat(parts, dim=0)[:maximum]

    def _device(self) -> torch.device:
        """Infer the pool device from any stored tensor."""

        for table in (self.natural, self.guided):
            for value in table.values():
                return value.device
        if self.natural_mass is not None:
            return self.natural_mass.device
        return torch.device("cpu")


@dataclass
class PruningResult:
    """Artifacts produced by one method execution."""

    natural_mass: torch.Tensor
    natural_visitation: torch.Tensor
    channel_scores: torch.Tensor
    rankings: torch.Tensor
    distortions: torch.Tensor
    costs: torch.Tensor
    widths: torch.Tensor
    compensated_down: dict[tuple[int, int], torch.Tensor]
    diagnostics: dict[str, object] = field(default_factory=dict)