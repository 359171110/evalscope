from __future__ import annotations

from .model_adapter import load_qwen3_moe
from .model_families import resolve_model_family


def load_supported_moe(
    model_path: str,
    device_map=None,
    model_family: str | None = None,
    device: str | None = None,
):
    """Load a supported MoE checkpoint using path/config family inference."""

    family = resolve_model_family(model_path=model_path, model_family=model_family)
    loader_kwargs = {
        "device_map": device_map,
        "model_family": family,
    }
    if device is not None:
        loader_kwargs["device"] = device
    return load_qwen3_moe(model_path, **loader_kwargs)
