from __future__ import annotations

import torch


def compute_co_route_uniqueness(
    co_route_context: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Measure how unlike each expert's train-only routing committee is.

    ``co_route_context`` stores gate-weighted expert co-occurrence matrices.  The
    diagonal is excluded because it measures an expert's own routing mass, not
    its committee context.  Experts without any routed partner receive zero
    uniqueness so missing evidence cannot create a safety constraint.
    """

    if co_route_context.ndim != 3 or co_route_context.shape[1] != co_route_context.shape[2]:
        raise ValueError("co-route context must be [layers, experts, experts].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if not bool(torch.isfinite(co_route_context).all()) or bool(
        (co_route_context < 0).any()
    ):
        raise ValueError("co-route context must be finite and non-negative.")

    context = co_route_context.to(torch.float64).clone()
    expert_count = int(context.shape[1])
    diagonal = torch.arange(expert_count, device=context.device)
    context[:, diagonal, diagonal] = 0.0
    norms = context.norm(dim=2, keepdim=True)
    active = norms.squeeze(2) > eps
    normalized = context / norms.clamp_min(eps)
    similarity = normalized @ normalized.transpose(1, 2)
    similarity[:, diagonal, diagonal] = -1.0
    nearest = similarity.max(dim=2).values.clamp(min=0.0, max=1.0)
    uniqueness = (1.0 - nearest) * active.to(nearest.dtype)
    return uniqueness.to(co_route_context.dtype)


def aggregate_unique_contribution_folds(
    output_saliency_folds: torch.Tensor,
    co_route_context_folds: torch.Tensor,
    *,
    aggregation: str = "mean",
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine actual output contribution with committee non-redundancy.

    Output contribution is normalized independently per fold and layer before
    multiplication.  Uniqueness is computed within each fold, preventing a
    high-volume calibration interval from dominating the cross-fitted score.
    """

    if output_saliency_folds.ndim != 3:
        raise ValueError("output saliency folds must be [folds, layers, experts].")
    if co_route_context_folds.ndim != 4:
        raise ValueError(
            "co-route context folds must be [folds, layers, experts, experts]."
        )
    if output_saliency_folds.shape[:3] != co_route_context_folds.shape[:3]:
        raise ValueError("output saliency and co-route fold shapes do not match.")
    if co_route_context_folds.shape[2] != co_route_context_folds.shape[3]:
        raise ValueError("co-route context matrices must be square.")
    normalized_output = []
    uniqueness_folds = []
    for fold_index in range(int(output_saliency_folds.shape[0])):
        normalized_output.append(
            aggregate_output_saliency_folds(
                output_saliency_folds[fold_index : fold_index + 1], eps=eps
            )
        )
        uniqueness_folds.append(
            compute_co_route_uniqueness(
                co_route_context_folds[fold_index], eps=eps
            )
        )
    normalized = torch.stack(normalized_output)
    uniqueness = torch.stack(uniqueness_folds)
    fold_scores = normalized * uniqueness
    if aggregation == "mean":
        score = fold_scores.mean(dim=0)
    elif aggregation == "minimum":
        score = fold_scores.min(dim=0).values
    else:
        raise ValueError("unique contribution aggregation must be mean or minimum.")
    return score.to(output_saliency_folds.dtype), uniqueness.to(
        output_saliency_folds.dtype
    )


def aggregate_output_saliency_folds(
    output_saliency_folds: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Build a scale-invariant consensus from train-only saliency folds.

    Each fold is normalized independently within every layer before averaging, so
    a fold with globally larger activation norms cannot dominate the consensus.
    """

    if output_saliency_folds.ndim != 3:
        raise ValueError("output saliency folds must be [folds, layers, experts].")
    if output_saliency_folds.shape[0] < 1:
        raise ValueError("at least one output saliency fold is required.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    if not bool(torch.isfinite(output_saliency_folds).all()) or bool(
        (output_saliency_folds < 0).any()
    ):
        raise ValueError("output saliency folds must be finite and non-negative.")

    folds = output_saliency_folds.to(torch.float64)
    layer_means = folds.mean(dim=2, keepdim=True)
    if bool((layer_means <= 0).any()):
        raise ValueError("every fold/layer must contain positive output saliency.")
    normalized = folds / layer_means.clamp_min(eps)
    return normalized.mean(dim=0).to(output_saliency_folds.dtype)


def rebind_expert_utility_to_coverage(
    old_block_values: torch.Tensor,
    old_coverage: torch.Tensor,
    new_coverage: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold expert utility fixed while replacing its channel-prefix coverage."""

    if old_block_values.shape != old_coverage.shape or old_coverage.shape != new_coverage.shape:
        raise ValueError("old values and old/new coverage tensors must share one shape.")
    if old_block_values.ndim != 3:
        raise ValueError("utility and coverage tensors must be [layers, experts, blocks].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    for name, tensor in (
        ("old_block_values", old_block_values),
        ("old_coverage", old_coverage),
        ("new_coverage", new_coverage),
    ):
        if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
            raise ValueError(f"{name} must be finite and non-negative.")
    if bool((new_coverage[..., :-1] < new_coverage[..., 1:]).any()):
        raise ValueError("new coverage must contain non-increasing prefix marginals.")

    dtype = torch.float64
    old_values = old_block_values.to(dtype=dtype)
    old_cov = old_coverage.to(dtype=dtype)
    new_cov = new_coverage.to(dtype=dtype)
    expert_utility = old_values.sum(dim=-1) / (old_cov + eps).sum(dim=-1).clamp_min(eps)
    rebound = expert_utility.unsqueeze(-1) * (new_cov + eps)
    return rebound.to(dtype=old_block_values.dtype), expert_utility.to(
        dtype=old_block_values.dtype
    )


def fuse_expert_utility_with_output_saliency(
    expert_utility: torch.Tensor,
    output_saliency: torch.Tensor,
    *,
    beta: float,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometrically fuse conditional utility with actual expert contribution."""

    if expert_utility.shape != output_saliency.shape or expert_utility.ndim != 2:
        raise ValueError("expert utility and output saliency must share [layers, experts].")
    if not bool(torch.isfinite(expert_utility).all()) or bool((expert_utility < 0).any()):
        raise ValueError("expert utility must be finite and non-negative.")
    if not bool(torch.isfinite(output_saliency).all()) or bool((output_saliency < 0).any()):
        raise ValueError("output saliency must be finite and non-negative.")
    beta_f = float(beta)
    if beta_f < 0.0:
        raise ValueError("beta must be non-negative.")
    normalized = output_saliency.to(torch.float64)
    normalized = normalized / normalized.mean(dim=1, keepdim=True).clamp_min(eps)
    factor = normalized.clamp_min(eps).pow(beta_f)
    fused = expert_utility.to(torch.float64) * factor
    return fused.to(expert_utility.dtype), factor.to(expert_utility.dtype)
