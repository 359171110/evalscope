from __future__ import annotations

import torch
import torch.nn.functional as F


def functional_product_kernel(
    responses: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    importance: torch.Tensor | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if responses.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("responses and down weight must be two-dimensional.")
    channel_count = int(responses.shape[1])
    if int(down_weight.shape[1]) != channel_count:
        raise ValueError("down weight columns must align with response channels.")
    normalized_response = F.normalize(responses.float(), p=2, dim=0, eps=eps)
    normalized_down = F.normalize(down_weight.float(), p=2, dim=0, eps=eps)
    kernel = (normalized_response.transpose(0, 1) @ normalized_response) * (
        normalized_down.transpose(0, 1) @ normalized_down
    )
    if importance is not None:
        if importance.shape != (channel_count,) or not torch.isfinite(importance).all() or (importance < 0).any():
            raise ValueError("importance must be a finite non-negative score per channel.")
        positive = importance.float()[importance > 0]
        scale = positive.median() if positive.numel() else torch.tensor(1.0, device=importance.device)
        weights = torch.sqrt(importance.float() / scale.clamp_min(eps))
        kernel = weights[:, None] * kernel * weights[None, :]
    if not torch.isfinite(kernel).all():
        raise ValueError("functional product kernel must contain only finite values.")
    return (kernel + kernel.transpose(0, 1)) * 0.5


def conditional_residual_order(
    kernel: torch.Tensor,
    importance_order: torch.Tensor,
    pseudo_order: torch.Tensor,
    *,
    retained_channels: int,
    protected_channels: int,
    ridge_relative: float = 1.0e-6,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix.")
    channel_count = int(kernel.shape[0])
    if importance_order.shape != (channel_count,) or pseudo_order.shape != (channel_count,):
        raise ValueError("importance and pseudo orders must contain every channel exactly once.")
    retained = int(retained_channels)
    protected = int(protected_channels)
    if not 0 <= protected <= retained <= channel_count:
        raise ValueError("channel budgets must satisfy 0 <= protected <= retained <= channel_count.")
    diagonal = kernel.diagonal().float().clamp_min(0.0)
    ridge = float(ridge_relative) * float(diagonal.mean().clamp_min(eps).item())
    residual = diagonal.clone()
    factors = torch.empty((channel_count, 0), dtype=torch.float32, device=kernel.device)
    selected = []
    selected_mask = torch.zeros(channel_count, dtype=torch.bool, device=kernel.device)
    importance_rank = torch.empty(channel_count, dtype=torch.long, device=kernel.device)
    importance_rank[importance_order.to(device=kernel.device)] = torch.arange(channel_count, device=kernel.device)

    def add_channel(channel_id: int) -> None:
        nonlocal factors, residual
        column = kernel[:, channel_id].float()
        if factors.shape[1]:
            column = column - factors @ factors[channel_id]
        denominator = torch.sqrt((column[channel_id] + ridge).clamp_min(eps))
        factor = column / denominator
        factors = torch.cat((factors, factor.unsqueeze(1)), dim=1)
        residual = (residual - factor.square()).clamp_min(0.0)
        selected.append(channel_id)
        selected_mask[channel_id] = True
        residual[channel_id] = -torch.inf

    for channel_id in pseudo_order[:protected].tolist():
        add_channel(int(channel_id))
    while len(selected) < retained:
        maximum = residual.max()
        tied = torch.nonzero(residual == maximum, as_tuple=False).flatten()
        chosen = tied[importance_rank[tied].argmin()]
        add_channel(int(chosen.item()))
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=importance_order.device)
    remaining = importance_order[~selected_mask.to(device=importance_order.device)[importance_order]]
    return torch.cat((selected_tensor, remaining)), residual