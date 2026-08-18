from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_FAMILY_ALIASES = {
    "qwen3": "qwen3",
    "qwen3moe": "qwen3",
    "qwen3-moe": "qwen3",
    "qwen3.6": "qwen3.6",
    "qwen3_6": "qwen3.6",
    "qwen3.5": "qwen3.6",
    "qwen3_5": "qwen3.6",
    "qwen3.5-moe": "qwen3.6",
    "qwen3_5_moe": "qwen3.6",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "gemma-4-it": "gemma4",
    "gemma-4-26b-a4b-it": "gemma4",
}


def normalize_model_family(model_family: str | None, default: str = "qwen3") -> str:
    raw = default if model_family is None else str(model_family).strip()
    family = MODEL_FAMILY_ALIASES.get((raw or default).lower())
    if family is None:
        raise ValueError(f"Unsupported model family: {model_family!r}.")
    return family


def _load_raw_config_json(model_path: str | None) -> dict[str, Any] | None:
    if not model_path:
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def detect_model_family_from_path(model_path: str | None) -> str | None:
    if not model_path:
        return None
    model_name = Path(model_path).name.lower()
    if "gemma-4" in model_name or "gemma4" in model_name:
        return "gemma4"
    if "qwen3.6" in model_name or "qwen3.5" in model_name or "qwen3_5" in model_name:
        return "qwen3.6"
    if "qwen3" in model_name:
        return "qwen3"

    payload = _load_raw_config_json(model_path)
    if payload is None:
        return None
    model_type = str(payload.get("model_type", "")).lower()
    text_model_type = str(payload.get("text_config", {}).get("model_type", "")).lower()
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} or text_model_type == "qwen3_5_moe_text":
        return "qwen3.6"
    if model_type in {"gemma4", "gemma4_text"} or text_model_type == "gemma4_text":
        return "gemma4"
    if model_type in {"qwen3_moe", "qwen3_moe_text"} or text_model_type == "qwen3_moe_text":
        return "qwen3"
    return None


def resolve_model_family(
    *,
    model_path: str | None = None,
    model_family: str | None = None,
    default: str = "qwen3",
) -> str:
    if model_family is not None:
        return normalize_model_family(model_family, default=default)
    return detect_model_family_from_path(model_path) or normalize_model_family(default)