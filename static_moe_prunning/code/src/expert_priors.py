from __future__ import annotations

from collections.abc import Mapping

import torch


def build_prior_payload(
    *,
    method: str,
    model_path: str,
    table: Mapping[int, torch.Tensor],
) -> dict:
    """Build a shape-auditable expert-prior cache payload."""

    normalized = {
        int(layer_idx): values.detach().float().cpu()
        for layer_idx, values in table.items()
    }
    if not normalized:
        raise ValueError("expert prior table must not be empty.")
    if any(values.ndim != 1 for values in normalized.values()):
        raise ValueError("each expert prior layer must be one-dimensional.")
    expert_counts = {int(values.numel()) for values in normalized.values()}
    if len(expert_counts) != 1:
        raise ValueError("expert count must be consistent across layers.")
    return {
        "schema_version": 1,
        "method": str(method),
        "model_path": str(model_path),
        "num_layers": len(normalized),
        "num_experts": next(iter(expert_counts)),
        "table": normalized,
    }
