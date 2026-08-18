from __future__ import annotations

import torch


def _validate_inputs(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    keep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    cov = covariance.detach().to(dtype=torch.float64, device="cpu")
    keep = keep_indices.detach().to(dtype=torch.long, device="cpu").flatten()
    if weights.ndim != 2 or cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("down_proj must be [hidden, channels] and covariance must be square.")
    if weights.shape[1] != cov.shape[0]:
        raise ValueError("down_proj channel count must match covariance size.")
    if keep.numel() == 0 or bool((keep < 0).any()) or bool((keep >= cov.shape[0]).any()):
        raise ValueError("keep_indices must contain valid channel IDs.")
    if int(torch.unique(keep).numel()) != int(keep.numel()):
        raise ValueError("keep_indices must not contain duplicates.")
    return weights, cov, keep


def fit_ridge_compensation(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a residual compensation matrix and return effective kept columns."""

    weights, cov, keep = _validate_inputs(down_proj, covariance, keep_indices)
    if float(regularization) < 0.0:
        raise ValueError("regularization must be non-negative.")
    pruned = torch.tensor(
        [index for index in range(cov.shape[0]) if index not in set(keep.tolist())],
        dtype=torch.long,
    )
    cov_kk = cov.index_select(0, keep).index_select(1, keep)
    if pruned.numel() == 0:
        delta = torch.zeros((weights.shape[0], keep.numel()), dtype=weights.dtype)
    else:
        cov_pk = cov.index_select(0, pruned).index_select(1, keep)
        system = cov_kk + float(regularization) * torch.eye(keep.numel(), dtype=cov.dtype)
        target = weights.index_select(1, pruned).matmul(cov_pk)
        delta = torch.linalg.solve(system.transpose(0, 1), target.transpose(0, 1)).transpose(0, 1)
    effective = weights.index_select(1, keep) + delta
    return effective, delta


def fit_rank_limited_compensation(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    keep_indices: torch.Tensor,
    *,
    regularization: float,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ridge compensation and truncate it in the kept-feature covariance metric."""

    if int(rank) <= 0:
        raise ValueError("rank must be positive.")
    weights, cov, keep = _validate_inputs(down_proj, covariance, keep_indices)
    effective, delta = fit_ridge_compensation(
        weights,
        cov,
        keep,
        regularization=regularization,
    )
    keep_covariance = cov.index_select(0, keep).index_select(1, keep)
    keep_covariance = 0.5 * (keep_covariance + keep_covariance.transpose(0, 1))
    eigvals, eigvecs = torch.linalg.eigh(keep_covariance)
    safe_eigvals = eigvals.clamp_min(1.0e-12)
    coloring = (eigvecs * safe_eigvals.sqrt()[None, :]).matmul(eigvecs.transpose(0, 1))
    whitening = (eigvecs * safe_eigvals.rsqrt()[None, :]).matmul(eigvecs.transpose(0, 1))
    whitened_delta = delta.matmul(coloring)
    u, singular_values, vh = torch.linalg.svd(whitened_delta, full_matrices=False)
    effective_rank = min(int(rank), int(singular_values.numel()))
    truncated = (u[:, :effective_rank] * singular_values[:effective_rank]).matmul(vh[:effective_rank])
    truncated_delta = truncated.matmul(whitening)
    return weights.index_select(1, keep) + truncated_delta, truncated_delta


def normalized_output_error(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    keep_indices: torch.Tensor,
    effective_keep_proj: torch.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> float:
    """Compute gate-weighted normalized expert-output SSE from covariance."""

    weights, cov, keep = _validate_inputs(down_proj, covariance, keep_indices)
    effective = effective_keep_proj.detach().to(dtype=torch.float64, device="cpu")
    if effective.shape != (weights.shape[0], keep.numel()):
        raise ValueError("effective_keep_proj has an incompatible shape.")
    error_weights = torch.zeros_like(weights)
    error_weights.index_copy_(1, keep, weights.index_select(1, keep) - effective)
    pruned = torch.tensor(
        [index for index in range(cov.shape[0]) if index not in set(keep.tolist())],
        dtype=torch.long,
    )
    if pruned.numel() > 0:
        error_weights.index_copy_(1, pruned, weights.index_select(1, pruned))
    numerator = (error_weights.matmul(cov) * error_weights).sum()
    denominator = (weights.matmul(cov) * weights).sum()
    return float((numerator / denominator.clamp_min(float(epsilon))).item())


def rank_rms_channels(
    down_proj: torch.Tensor,
    unweighted_square_sum: torch.Tensor,
    *,
    route_count: int,
) -> torch.Tensor:
    """Rank channels by fit-only RMS activation times down-projection norm."""

    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    square_sum = unweighted_square_sum.detach().to(dtype=torch.float64, device="cpu").flatten()
    if int(route_count) <= 0 or square_sum.numel() != weights.shape[1]:
        raise ValueError("route_count and activation statistics have incompatible values.")
    score = square_sum.div(float(route_count)).clamp_min(0.0).sqrt() * weights.square().sum(dim=0).sqrt()
    return torch.argsort(-score, stable=True)


def rank_tail_channels(
    down_proj: torch.Tensor,
    unweighted_square_sum: torch.Tensor,
    max_abs: torch.Tensor,
    *,
    route_count: int,
    tail_lambda: float,
) -> torch.Tensor:
    """Rank channels by a fit-only RMS/Tail geometric blend."""

    if not 0.0 <= float(tail_lambda) <= 1.0:
        raise ValueError("tail_lambda must be in [0, 1].")
    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    square_sum = unweighted_square_sum.detach().to(dtype=torch.float64, device="cpu").flatten()
    maximum = max_abs.detach().to(dtype=torch.float64, device="cpu").flatten()
    if int(route_count) <= 0 or square_sum.numel() != weights.shape[1] or maximum.numel() != weights.shape[1]:
        raise ValueError("activation statistics have incompatible values.")
    down_norm = weights.square().sum(dim=0).sqrt().clamp_min(1.0e-16)
    rms = square_sum.div(float(route_count)).clamp_min(1.0e-16).sqrt() * down_norm
    tail = maximum.clamp_min(1.0e-16) * down_norm
    score = rms.pow(1.0 - float(tail_lambda)) * tail.pow(float(tail_lambda))
    return torch.argsort(-score, stable=True)


def ramp_conditional_residual_selection(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    *,
    keep_count: int,
    anchor_count: int = 0,
    regularization: float,
) -> torch.Tensor:
    """Select channels by greedy fit-only compensated output reconstruction."""

    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    cov = covariance.detach().to(dtype=torch.float64, device="cpu")
    if weights.ndim != 2 or cov.ndim != 2 or cov.shape != (weights.shape[1], weights.shape[1]):
        raise ValueError("down_proj and covariance have incompatible shapes.")
    if not 0 <= int(anchor_count) <= int(keep_count) <= int(weights.shape[1]) or int(keep_count) == 0:
        raise ValueError("keep_count and anchor_count must satisfy 0 <= anchor <= keep <= channels.")
    if float(regularization) < 0.0:
        raise ValueError("regularization must be non-negative.")

    output_energy = cov.diag() * weights.square().sum(dim=0)
    anchor_order = torch.argsort(-output_energy, stable=True)[: int(anchor_count)].tolist()
    selected = [int(index) for index in anchor_order]
    selected_mask = torch.zeros(weights.shape[1], dtype=torch.bool)
    residual_covariance = cov.clone()
    output_residual = weights.matmul(residual_covariance)

    def condition_on(channel_idx: int) -> None:
        nonlocal residual_covariance, output_residual
        covariance_column = residual_covariance[:, int(channel_idx)].clone()
        denominator = float(residual_covariance[int(channel_idx), int(channel_idx)].item()) + float(
            regularization
        )
        if denominator <= 0.0:
            selected_mask[int(channel_idx)] = True
            return
        output_column = output_residual[:, int(channel_idx)].clone()
        residual_covariance.sub_(
            covariance_column[:, None].matmul(covariance_column[None, :]).div_(denominator)
        )
        residual_covariance.copy_(0.5 * (residual_covariance + residual_covariance.transpose(0, 1)))
        output_residual.sub_(output_column[:, None].matmul(covariance_column[None, :]).div_(denominator))
        selected_mask[int(channel_idx)] = True

    for channel_idx in selected:
        condition_on(channel_idx)
    while len(selected) < int(keep_count):
        denominators = residual_covariance.diag().add(float(regularization))
        gains = output_residual.square().sum(dim=0).div(denominators.clamp_min(1.0e-30))
        gains[selected_mask] = float("-inf")
        channel_idx = int(torch.argmax(gains).item())
        if not torch.isfinite(gains[channel_idx]):
            remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
            channel_idx = int(remaining[0].item())
        selected.append(channel_idx)
        condition_on(channel_idx)
    return torch.tensor(selected, dtype=torch.long)


def conditional_activation_selection(
    covariance: torch.Tensor,
    *,
    keep_count: int,
    regularization: float,
) -> torch.Tensor:
    """Select a set that explains activation covariance without output weighting."""

    cov = covariance.detach().to(dtype=torch.float64, device="cpu")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square.")
    identity = torch.eye(cov.shape[0], dtype=cov.dtype)
    return ramp_conditional_residual_selection(
        identity,
        cov,
        keep_count=keep_count,
        anchor_count=0,
        regularization=regularization,
    )


def pairwise_output_correlation_selection(
    down_proj: torch.Tensor,
    covariance: torch.Tensor,
    *,
    keep_count: int,
) -> torch.Tensor:
    """Greedily retain high-energy channels while penalizing pairwise output redundancy."""

    weights = down_proj.detach().to(dtype=torch.float64, device="cpu")
    cov = covariance.detach().to(dtype=torch.float64, device="cpu")
    if weights.ndim != 2 or cov.shape != (weights.shape[1], weights.shape[1]):
        raise ValueError("down_proj and covariance have incompatible shapes.")
    if not 0 < int(keep_count) <= int(weights.shape[1]):
        raise ValueError("keep_count must be in [1, channels].")

    output_gram = cov * weights.transpose(0, 1).matmul(weights)
    energy = output_gram.diag().clamp_min(0.0)
    scale = energy.sqrt().clamp_min(1.0e-30)
    correlation = output_gram / (scale[:, None] * scale[None, :])
    selected: list[int] = []
    selected_mask = torch.zeros(weights.shape[1], dtype=torch.bool)
    max_redundancy = torch.zeros(weights.shape[1], dtype=torch.float64)
    while len(selected) < int(keep_count):
        scores = energy * (1.0 - max_redundancy.clamp(max=1.0))
        scores[selected_mask] = float("-inf")
        channel_idx = int(torch.argmax(scores).item())
        selected.append(channel_idx)
        selected_mask[channel_idx] = True
        max_redundancy = torch.maximum(max_redundancy, correlation[channel_idx].abs())
    return torch.tensor(selected, dtype=torch.long)