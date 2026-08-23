from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from AIMER_Mix.mix_core import descending_unit_ranks
from AIMER_MIX_PLUS.plus_core import PseudoSource, layerprop_mix_lambda, rank_percentiles_from_order


@dataclass(frozen=True)
class UnifyConfig:
    """Agreement-gated Mix fused with FFN-space LayerProp and PRP.

    There is no locked Mix core and no model-specific branch. Mix is a prior
    whose weight is the keep-set overlap with the FFN-space ranking:

        S = α Mix + (1 - α) S_pseudo
        S_pseudo = λ LayerProp + (1 - λ) PRP
        λ = N / (N + τ)
        α = |Top-K(Mix) ∩ Top-K(S_pseudo)| / K
    """

    layerprop_tau: float = 8.0
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.layerprop_tau <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("layerprop_tau and epsilon must be positive")


def keep_set_overlap(left: torch.Tensor, right: torch.Tensor, keep: int) -> float:
    """Return |Top-K(left) ∩ Top-K(right)| / K for one expert ranking."""

    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("left and right must be 1-D rankings of equal length")
    channels = int(left.numel())
    if not 1 <= keep <= channels:
        raise ValueError("keep must be in [1, channels]")
    left_idx = torch.topk(left, keep, largest=True, sorted=False).indices
    right_idx = torch.topk(right, keep, largest=True, sorted=False).indices
    mask = torch.zeros(channels, dtype=torch.bool, device=left.device)
    mask[left_idx] = True
    return float(mask[right_idx].to(dtype=torch.float32).mean().item())


def _hit_count(source: PseudoSource, layer_id: int, expert_id: int) -> float:
    if source.hit_counts is not None:
        return float(source.hit_counts[layer_id, expert_id].clamp_min(0.0).item())
    if source.coverage is None:
        return 0.0
    coverage = float(source.coverage[layer_id, expert_id].item())
    if coverage <= 0.0:
        return 0.0
    if coverage <= 1.0:
        return float("inf")
    return coverage


def _source_by_name(sources: list[PseudoSource], name: str) -> PseudoSource | None:
    matches = [source for source in sources if source.name == name]
    if len(matches) > 1:
        raise ValueError(f"Duplicate pseudo source: {name}")
    return matches[0] if matches else None


def _fuse_one_expert(
    mix_rank: torch.Tensor,
    retained_channels: int,
    layerprop_rank: torch.Tensor | None,
    prp_rank: torch.Tensor | None,
    layerprop_hits: float,
    config: UnifyConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channels = int(mix_rank.numel())
    mix_order = torch.argsort(mix_rank, descending=True, stable=True)

    lambda_lp = 0.0
    if layerprop_rank is not None and prp_rank is not None:
        lambda_lp = layerprop_mix_lambda(layerprop_hits, config.layerprop_tau)
        pseudo_rank = lambda_lp * layerprop_rank + (1.0 - lambda_lp) * prp_rank
        active = ["layerprop", "prp"]
    elif layerprop_rank is not None:
        lambda_lp = 1.0
        pseudo_rank = layerprop_rank
        active = ["layerprop"]
    elif prp_rank is not None:
        lambda_lp = 0.0
        pseudo_rank = prp_rank
        active = ["prp"]
    else:
        return mix_order, {
            "mix_alpha": 1.0,
            "overlap": 1.0,
            "layerprop_lambda": None,
            "layerprop_hits": None,
            "active_sources": [],
        }

    overlap = keep_set_overlap(mix_rank, pseudo_rank, retained_channels)
    alpha = float(max(0.0, min(1.0, overlap)))
    fused = alpha * mix_rank + (1.0 - alpha) * pseudo_rank
    order = torch.argsort(fused, descending=True, stable=True)
    hits = None if not math.isfinite(float(layerprop_hits)) else float(layerprop_hits)
    return order, {
        "mix_alpha": alpha,
        "overlap": overlap,
        "layerprop_lambda": float(lambda_lp),
        "layerprop_hits": hits,
        "active_sources": active,
    }


def build_unify_ranking(
    aimer_mix_scores: torch.Tensor,
    retained_channels: int,
    pseudo_sources: list[PseudoSource] | None = None,
    config: UnifyConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fuse Mix with LayerProp/PRP using keep-set overlap as the Mix gate."""

    config = config or UnifyConfig()
    if aimer_mix_scores.ndim != 3:
        raise ValueError("aimer_mix_scores must have shape [layers, experts, channels]")
    layers, experts, channels = map(int, aimer_mix_scores.shape)
    if not 1 < int(retained_channels) < channels:
        raise ValueError("retained_channels must be in (1, channels)")
    sources = pseudo_sources or []
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Pseudo source names must be unique")
    for source in sources:
        source.validate(layers, experts, channels)
        if source.name == "pp":
            raise ValueError("Unified fusion does not use PP; router space is not FFN evidence")

    layerprop = _source_by_name(sources, "layerprop")
    prp = _source_by_name(sources, "prp")
    extra = [name for name in names if name not in {"layerprop", "prp"}]
    if extra:
        raise ValueError(f"Unsupported unified sources: {extra}")

    device = aimer_mix_scores.device
    mix_ranks = descending_unit_ranks(aimer_mix_scores.reshape(layers * experts, channels)).reshape(
        layers, experts, channels
    )
    layerprop_ranks = (
        None if layerprop is None else rank_percentiles_from_order(layerprop.order.to(device))
    )
    prp_ranks = None if prp is None else rank_percentiles_from_order(prp.order.to(device))

    orders: list[torch.Tensor] = []
    diagnostics: list[list[dict[str, Any]]] = []
    for layer_id in range(layers):
        layer_orders: list[torch.Tensor] = []
        layer_diag: list[dict[str, Any]] = []
        for expert_id in range(experts):
            hits = 0.0 if layerprop is None else _hit_count(layerprop, layer_id, expert_id)
            order, info = _fuse_one_expert(
                mix_ranks[layer_id, expert_id],
                int(retained_channels),
                None if layerprop_ranks is None else layerprop_ranks[layer_id, expert_id],
                None if prp_ranks is None else prp_ranks[layer_id, expert_id],
                hits,
                config,
            )
            layer_orders.append(order)
            layer_diag.append({"layer_id": layer_id, "expert_id": expert_id, **info})
        orders.append(torch.stack(layer_orders))
        diagnostics.append(layer_diag)

    flat = [record for layer in diagnostics for record in layer]
    alphas = torch.tensor([float(record["mix_alpha"]) for record in flat])
    return torch.stack(orders), {
        "schema_version": 1,
        "method": "aimer_unify",
        "base": "aimer_mix",
        "retained_channels": int(retained_channels),
        "sources": [source.name for source in sources],
        "config": {
            "layerprop_tau": float(config.layerprop_tau),
        },
        "diagnostic_summary": {
            "mix_alpha_mean": float(alphas.mean().item()) if flat else 1.0,
            "mix_alpha_min": float(alphas.min().item()) if flat else 1.0,
            "mix_alpha_max": float(alphas.max().item()) if flat else 1.0,
        },
        "diagnostics": diagnostics,
    }


def build_unify_ranking_from_order(
    aimer_mix_order: torch.Tensor,
    retained_channels: int,
    pseudo_sources: list[PseudoSource] | None = None,
    config: UnifyConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Same fusion, taking a Mix permutation cache instead of raw scores."""

    if aimer_mix_order.ndim != 3:
        raise ValueError("aimer_mix_order must have shape [layers, experts, channels]")
    return build_unify_ranking(
        rank_percentiles_from_order(aimer_mix_order),
        retained_channels=retained_channels,
        pseudo_sources=pseudo_sources,
        config=config,
    )
