"""Configuration for Router-conditioned Multi-origin LayerProp."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerPropConfig:
    """Data-free pruning settings shared by all supported model families."""

    propagation_mode: str = "long_short"
    num_pseudo_tokens: int = 2048
    sequence_length: int = 32
    probe_variants: int = 8
    probe_scale: float = 1.0
    probe_sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    refresh_stride: int = 4
    refresh_horizon: int = 8
    max_rows_per_expert_per_origin: int = 128
    min_train_rows: int = 16
    min_valid_rows: int = 8
    recoverability_band: int = 32
    max_swaps_ratio: float = 0.05
    ridge_grid: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
    trust_ratio_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    channel_multiple: int = 64
    seed: int = 1729
    epsilon: float = 1.0e-8
    extra: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate settings before synthetic propagation or pruning."""

        if self.propagation_mode not in {"stable", "long_short"}:
            raise ValueError("propagation_mode must be 'stable' or 'long_short'")
        positive_ints = {
            "num_pseudo_tokens": self.num_pseudo_tokens,
            "sequence_length": self.sequence_length,
            "probe_variants": self.probe_variants,
            "refresh_stride": self.refresh_stride,
            "refresh_horizon": self.refresh_horizon,
            "max_rows_per_expert_per_origin": self.max_rows_per_expert_per_origin,
            "min_train_rows": self.min_train_rows,
            "min_valid_rows": self.min_valid_rows,
            "recoverability_band": self.recoverability_band,
            "channel_multiple": self.channel_multiple,
        }
        if any(int(value) <= 0 for value in positive_ints.values()):
            raise ValueError(f"All integer settings must be positive: {positive_ints}")
        if self.num_pseudo_tokens % self.sequence_length:
            raise ValueError("num_pseudo_tokens must be divisible by sequence_length")
        if self.probe_scale <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("probe_scale and epsilon must be positive")
        if not self.probe_sigmas or any(float(value) < 0.0 for value in self.probe_sigmas):
            raise ValueError("probe_sigmas must contain non-negative values")
        if not self.ridge_grid or any(float(value) <= 0.0 for value in self.ridge_grid):
            raise ValueError("ridge_grid must contain positive values")
        if not self.trust_ratio_grid or any(float(value) <= 0.0 for value in self.trust_ratio_grid):
            raise ValueError("trust_ratio_grid must contain positive values")
        if not 0.0 < self.max_swaps_ratio <= 1.0:
            raise ValueError("max_swaps_ratio must be in (0, 1]")

    @property
    def num_sequences(self) -> int:
        """Return the deterministic batch size implied by the token budget."""

        return self.num_pseudo_tokens // self.sequence_length
