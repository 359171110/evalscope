"""Calibration-free uniform Product channel pruning for routed MoE experts."""

from Product.model_adapter import ProductModelAdapter
from Product.product_core import coupled_channel_product, rank_channels_by_product

__all__ = ["ProductModelAdapter", "coupled_channel_product", "rank_channels_by_product"]
