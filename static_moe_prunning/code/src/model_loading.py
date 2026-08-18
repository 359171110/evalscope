from __future__ import annotations

from .model_adapter import load_qwen3_moe
from .model_families import resolve_model_family


def load_supported_moe(model_path: str, device_map=None, model_family: str | None = None):
    """Load a supported MoE checkpoint using path/config family inference."""

    family = resolve_model_family(model_path=model_path, model_family=model_family)
    return load_qwen3_moe(
        model_path,
        device_map=device_map,
        model_family=family,
    )
