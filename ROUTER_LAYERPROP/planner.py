"""Channel-set construction and held-out residual compensation."""

from __future__ import annotations

from typing import Any

import torch

from .config import LayerPropConfig
from .core import (
    CompensationResult,
    fit_ridge_down,
    output_energy_scores,
    recoverability_swap_refinement,
)


def _split_rows(rows: torch.Tensor, minimum_train: int, minimum_valid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split rows deterministically while preserving a usable validation side."""

    if rows.ndim != 2:
        raise ValueError("rows must have shape [samples, channels]")
    count = rows.shape[0]
    if count < minimum_train + minimum_valid:
        return rows, rows[:0]
    split = max(minimum_train, count // 2)
    split = min(split, count - minimum_valid)
    return rows[:split], rows[split:]


def _fallback_rows(down_proj: torch.Tensor, count: int) -> torch.Tensor:
    """Create a deterministic identity-like response for an uncovered expert."""

    channels = down_proj.shape[1]
    rows = torch.zeros((max(int(count), channels), channels), dtype=torch.float32, device=down_proj.device)
    diagonal = torch.arange(channels, device=down_proj.device)
    rows[diagonal, diagonal] = 1.0
    return rows


def choose_keep_channels(
    train_rows: torch.Tensor,
    down_proj: torch.Tensor,
    retained_channels: int,
    config: LayerPropConfig,
) -> torch.Tensor:
    """Initialize by output energy and refine locally by recoverability."""

    channels = int(down_proj.shape[1])
    if not 0 < retained_channels < channels:
        raise ValueError("retained_channels must be positive and smaller than source width")
    if train_rows.shape[0] == 0:
        scores = down_proj.float().square().sum(dim=0)
    else:
        scores = output_energy_scores(train_rows, down_proj)
    initial = torch.topk(scores, int(retained_channels), largest=True, sorted=False).indices
    if train_rows.shape[0] == 0:
        return torch.sort(initial).values
    return recoverability_swap_refinement(
        train_rows,
        down_proj,
        initial,
        ridge=float(config.ridge_grid[1] if len(config.ridge_grid) > 1 else config.ridge_grid[0]),
        band=int(config.recoverability_band),
        max_swaps=min(32, max(1, int(round(config.max_swaps_ratio * retained_channels))),),
    )


def build_expert_plan(
    *,
    source_rows: dict[str, torch.Tensor],
    down_proj: torch.Tensor,
    retained_channels: int,
    config: LayerPropConfig,
    validation_rows: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build one expert's keep set and accepted compensation tensor."""

    down_proj = down_proj.detach().float().cpu()
    if down_proj.ndim != 2:
        raise ValueError("down_proj must have shape [hidden, channels]")
    sources = {
        name: rows.detach().float().cpu()
        for name, rows in source_rows.items()
        if rows.ndim == 2 and rows.shape[0]
    }
    channels = int(down_proj.shape[1])
    if not sources:
        train_rows = _fallback_rows(down_proj, config.min_train_rows)
        valid_rows = train_rows[:0]
        train_source = "fallback_identity"
    else:
        train_parts = []
        valid_parts = []
        for name in sorted(sources):
            train, valid = _split_rows(sources[name], config.min_train_rows, config.min_valid_rows)
            if train.shape[0]:
                train_parts.append(train / max(float(train.shape[0]) ** 0.5, 1.0))
            if valid.shape[0]:
                valid_parts.append(valid / max(float(valid.shape[0]) ** 0.5, 1.0))
        train_rows = torch.cat(train_parts, dim=0) if train_parts else _fallback_rows(down_proj, config.min_train_rows)
        valid_rows = torch.cat(valid_parts, dim=0) if valid_parts else train_rows[:0]
        train_source = "+".join(sorted(sources))
    if validation_rows is not None and validation_rows.ndim == 2 and validation_rows.shape[0]:
        valid_rows = validation_rows.detach().float().cpu()
    keep = choose_keep_channels(train_rows, down_proj, retained_channels, config)
    if valid_rows.shape[0] < config.min_valid_rows:
        compensation = CompensationResult(
            down=down_proj.float().index_select(1, keep),
            accepted=False,
            ridge=None,
            trust_ratio=None,
            baseline_error=0.0,
            compensated_error=0.0,
            update_ratio=0.0,
        )
    else:
        compensation = fit_ridge_down(
            train_rows,
            valid_rows,
            down_proj,
            keep,
            ridge_grid=config.ridge_grid,
            trust_ratio_grid=config.trust_ratio_grid,
            epsilon=config.epsilon,
        )
    return {
        "source_width": channels,
        "retained_width": int(retained_channels),
        "retained_channels": keep.cpu(),
        "compensation_accepted": bool(compensation.accepted),
        "compensated_down": compensation.down.cpu() if compensation.accepted else None,
        "ridge": compensation.ridge,
        "trust_ratio": compensation.trust_ratio,
        "baseline_error": float(compensation.baseline_error),
        "compensated_error": float(compensation.compensated_error),
        "update_ratio": float(compensation.update_ratio),
        "train_source": train_source,
    }


def build_layer_plan(
    *,
    source_rows: dict[str, dict[int, torch.Tensor]],
    down_proj: torch.Tensor,
    retained_channels: int,
    config: LayerPropConfig,
    validation_rows: dict[int, torch.Tensor] | None = None,
) -> dict[int, dict[str, Any]]:
    """Build plans for all experts in one packed routed layer."""

    if down_proj.ndim != 3:
        raise ValueError("down_proj must have shape [experts, hidden, channels]")
    plans: dict[int, dict[str, Any]] = {}
    for expert in range(down_proj.shape[0]):
        expert_sources = {
            source: rows.get(expert, torch.empty((0, down_proj.shape[-1]), device=down_proj.device))
            for source, rows in source_rows.items()
        }
        plans[expert] = build_expert_plan(
            source_rows=expert_sources,
            down_proj=down_proj[expert],
            retained_channels=retained_channels,
            config=config,
            validation_rows=None if validation_rows is None else validation_rows.get(expert),
        )
    return plans


def plan_summary(layers: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Summarize accepted compensation and coverage counts."""

    plans = [plan for layer in layers.values() for plan in layer.values()]
    accepted = sum(bool(plan["compensation_accepted"]) for plan in plans)
    return {
        "experts": len(plans),
        "accepted_compensations": accepted,
        "compensation_acceptance_rate": accepted / max(len(plans), 1),
    }
