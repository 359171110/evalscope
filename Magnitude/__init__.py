"""Calibration-free uniform magnitude channel pruning for routed MoE experts."""

from Magnitude.magnitude_core import coupled_channel_magnitude, rank_channels_by_magnitude
from Magnitude.model_adapter import MagnitudeModelAdapter

__all__ = ["MagnitudeModelAdapter", "coupled_channel_magnitude", "rank_channels_by_magnitude"]
