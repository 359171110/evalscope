"""Configuration for the routing-aware pruning method."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodConfig:
    """Runtime and allocation settings for one pruning run."""

    natural_sequences: int = 96
    guided_sequences: int = 32
    sequence_length: int = 2048
    calibration_batch_size: int = 1
    generation_batch_size: int = 1
    guided_batch_size: int = 1
    max_samples_per_expert: int = 128
    min_samples_per_expert: int = 32
    safe_samples_per_expert: int = 8
    anchor_candidates: int = 16
    retention: float = 0.5
    ridge: float = 1.0e-4
    epsilon: float = 1.0e-8
    device: str = "cuda"
    dtype: str = "float16"
    width_levels: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)

    def validate(self, *, source_width: int | None = None) -> None:
        """Validate settings and optionally check the source channel width."""

        if self.natural_sequences <= 0 or self.guided_sequences < 0:
            raise ValueError("sequence budgets must be positive/non-negative")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.calibration_batch_size <= 0 or self.generation_batch_size <= 0 or self.guided_batch_size <= 0:
            raise ValueError("calibration and generation batch sizes must be positive")
        if not 0 < self.safe_samples_per_expert <= self.min_samples_per_expert <= self.max_samples_per_expert:
            raise ValueError("sample limits must satisfy safe <= min <= max")
        if self.anchor_candidates <= 0 or not 0.0 < self.retention <= 1.0:
            raise ValueError("anchor_candidates and retention are invalid")
        if self.ridge < 0.0 or self.epsilon <= 0.0:
            raise ValueError("ridge must be non-negative and epsilon must be positive")
        if not self.width_levels or any(not 0.0 < level <= 1.0 for level in self.width_levels):
            raise ValueError("width_levels must contain values in (0, 1]")
        if tuple(sorted(set(self.width_levels), reverse=True)) != self.width_levels:
            raise ValueError("width_levels must be unique and descending")
        if source_width is not None and any(round(source_width * level) <= 0 for level in self.width_levels):
            raise ValueError("width_levels produce an empty channel set")