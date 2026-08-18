from __future__ import annotations

import torch


def slice_contiguous_token_span(
    token_ids: torch.Tensor,
    *,
    total_tokens: int,
    token_offset: int = 0,
) -> torch.Tensor:
    """Select a reproducible contiguous calibration interval."""

    if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
        raise ValueError("token_ids must have shape [1, tokens].")
    count = int(total_tokens)
    offset = int(token_offset)
    if count <= 0:
        raise ValueError("total_tokens must be positive.")
    if offset < 0:
        raise ValueError("token_offset must be non-negative.")
    end = offset + count
    if end > int(token_ids.shape[1]):
        raise ValueError("dataset does not contain enough calibration tokens.")
    return token_ids[:, offset:end]


def build_rare_event_risk_floors(
    expert_risk: torch.Tensor,
    *,
    early_layer_count: int,
    global_quantile: float,
    relative_to_global_max: float,
    minimum_width: int,
    num_blocks: int,
) -> tuple[torch.Tensor, dict]:
    """Select sparse train-only expert floors from a global rare-event tail.

    The risk threshold is computed over every physical expert, while eligibility
    is restricted to the first ``early_layer_count`` MoE layers.  This prevents
    late-layer scale outliers from changing the reference distribution without
    allowing them into the protected set.
    """

    if expert_risk.ndim != 2 or expert_risk.numel() == 0:
        raise ValueError("expert_risk must have shape [layers, experts].")
    if not bool(torch.isfinite(expert_risk).all()) or bool((expert_risk < 0).any()):
        raise ValueError("expert_risk must be finite and non-negative.")
    layers, experts = (int(size) for size in expert_risk.shape)
    early = int(early_layer_count)
    if not 1 <= early <= layers:
        raise ValueError("early_layer_count must be in [1, num_layers].")
    quantile = float(global_quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("global_quantile must be in [0, 1].")
    relative = float(relative_to_global_max)
    if not 0.0 <= relative <= 1.0:
        raise ValueError("relative_to_global_max must be in [0, 1].")
    blocks = int(num_blocks)
    width = int(minimum_width)
    if blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if not 0 <= width <= blocks:
        raise ValueError("minimum_width must be in [0, num_blocks].")

    risk = expert_risk.to(dtype=torch.float64)
    quantile_threshold = torch.quantile(risk.flatten(), quantile)
    relative_threshold = risk.max() * relative
    threshold = torch.maximum(quantile_threshold, relative_threshold)
    selected_mask = risk >= threshold
    if early < layers:
        selected_mask[early:] = False
    floors = torch.zeros((layers, experts), dtype=torch.long, device=risk.device)
    floors[selected_mask] = width
    selected = [
        {
            "layer": int(layer),
            "expert": int(expert),
            "risk": float(risk[layer, expert].item()),
            "min_width": width,
        }
        for layer, expert in torch.nonzero(selected_mask, as_tuple=False).tolist()
    ]
    metadata = {
        "early_layer_count": early,
        "global_quantile": quantile,
        "relative_to_global_max": relative,
        "minimum_width": width,
        "quantile_threshold": float(quantile_threshold.item()),
        "relative_threshold": float(relative_threshold.item()),
        "threshold": float(threshold.item()),
        "selected_count": len(selected),
        "selected_experts": selected,
    }
    return floors, metadata


def build_consensus_rare_event_risk_floors(
    expert_risks: torch.Tensor,
    *,
    early_layer_count: int,
    global_quantile: float,
    relative_to_global_max: float,
    minimum_width: int,
    num_blocks: int,
    minimum_votes: int,
) -> tuple[torch.Tensor, dict]:
    """Require a rare-risk expert to recur across calibration intervals."""

    if expert_risks.ndim != 3 or int(expert_risks.shape[0]) <= 0:
        raise ValueError("expert_risks must have shape [folds, layers, experts].")
    fold_count = int(expert_risks.shape[0])
    votes_required = int(minimum_votes)
    if not 1 <= votes_required <= fold_count:
        raise ValueError("minimum_votes must be in [1, fold_count].")
    width = int(minimum_width)
    blocks = int(num_blocks)
    if blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if not 0 <= width <= blocks:
        raise ValueError("minimum_width must be in [0, num_blocks].")
    fold_metadata = []
    vote_count = torch.zeros(expert_risks.shape[1:], dtype=torch.long)
    for fold_idx in range(fold_count):
        floors, metadata = build_rare_event_risk_floors(
            expert_risks[fold_idx],
            early_layer_count=early_layer_count,
            global_quantile=global_quantile,
            relative_to_global_max=relative_to_global_max,
            minimum_width=1,
            num_blocks=num_blocks,
        )
        vote_count += (floors > 0).to(torch.long).cpu()
        fold_metadata.append(metadata)
    selected_mask = vote_count >= votes_required
    floors = torch.zeros_like(vote_count)
    floors[selected_mask] = width
    selected = [
        {
            "layer": int(layer),
            "expert": int(expert),
            "votes": int(vote_count[layer, expert].item()),
            "min_width": width,
        }
        for layer, expert in torch.nonzero(selected_mask, as_tuple=False).tolist()
    ]
    metadata = {
        "fold_count": fold_count,
        "minimum_votes": votes_required,
        "early_layer_count": int(early_layer_count),
        "global_quantile": float(global_quantile),
        "relative_to_global_max": float(relative_to_global_max),
        "minimum_width": width,
        "selected_count": len(selected),
        "selected_experts": selected,
        "fold_thresholds": [float(item["threshold"]) for item in fold_metadata],
        "fold_selected_counts": [
            int(item["selected_count"]) for item in fold_metadata
        ],
    }
    return floors, metadata


def blend_typical_and_tail_score(
    typical_score: torch.Tensor,
    tail_score: torch.Tensor,
    *,
    tail_lambda: float,
    eps: float = 1.0e-16,
) -> torch.Tensor:
    """Geometrically blend average channel utility with rare-event utility."""

    if typical_score.shape != tail_score.shape:
        raise ValueError("typical_score and tail_score must have the same shape.")
    if not 0.0 <= float(tail_lambda) <= 1.0:
        raise ValueError("tail_lambda must be in [0, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if not bool(torch.isfinite(typical_score).all()) or not bool(
        torch.isfinite(tail_score).all()
    ):
        raise ValueError("channel scores must be finite.")
    if bool((typical_score < 0).any()) or bool((tail_score < 0).any()):
        raise ValueError("channel scores must be non-negative.")
    weight = float(tail_lambda)
    if weight == 0.0:
        return typical_score.clone()
    if weight == 1.0:
        return tail_score.clone()
    return typical_score.clamp_min(eps).pow(1.0 - weight) * tail_score.clamp_min(
        eps
    ).pow(weight)


def expert_tail_risk_from_channels(
    max_abs_activation: torch.Tensor,
    down_column_norm: torch.Tensor,
) -> torch.Tensor:
    """Collapse channel tail contributions into one expert-level risk proxy."""

    if max_abs_activation.shape != down_column_norm.shape or max_abs_activation.ndim != 2:
        raise ValueError("tail activation and down norms must share [experts, channels].")
    if not bool(torch.isfinite(max_abs_activation).all()) or not bool(
        torch.isfinite(down_column_norm).all()
    ):
        raise ValueError("tail activation and down norms must be finite.")
    if bool((max_abs_activation < 0).any()) or bool((down_column_norm < 0).any()):
        raise ValueError("tail activation and down norms must be non-negative.")
    return (max_abs_activation * down_column_norm).amax(dim=-1)
