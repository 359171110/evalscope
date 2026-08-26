"""Exact discrete heterogeneous width allocation."""

from __future__ import annotations

import torch


def allocate_widths(
    costs: torch.Tensor,
    width_options: torch.Tensor,
    *,
    budget: int,
) -> torch.Tensor:
    """Solve the per-layer multiple-choice knapsack on the requested device."""

    if costs.ndim != 2 or width_options.ndim != 1 or costs.shape[1] != width_options.numel():
        raise ValueError("costs must be [experts, widths] and match width_options")
    if not bool(torch.all(width_options > 0)) or not bool(torch.all(width_options[1:] < width_options[:-1])):
        raise ValueError("width_options must be positive and strictly descending")
    experts, levels = (int(size) for size in costs.shape)
    options = width_options.to(device=costs.device, dtype=torch.long)
    target = int(budget)
    if target < int(options.min().item()) * experts or target > int(options.max().item()) * experts:
        raise ValueError("budget is outside the achievable width range")
    inf = torch.tensor(float("inf"), device=costs.device, dtype=torch.float32)
    dp = torch.full((target + 1,), inf, device=costs.device)
    dp[0] = 0.0
    choices: list[torch.Tensor] = []
    for expert_id in range(experts):
        next_dp = torch.full_like(dp, inf)
        parent = torch.full((target + 1,), -1, device=costs.device, dtype=torch.long)
        level_choices = torch.full((target + 1,), -1, device=costs.device, dtype=torch.long)
        for level_id in range(levels):
            width = int(options[level_id].item())
            if width > target:
                continue
            candidate = torch.full_like(dp, inf)
            candidate[width:] = dp[:-width] + costs[expert_id, level_id].float()
            improved = candidate < next_dp
            next_dp = torch.where(improved, candidate, next_dp)
            parent = torch.where(improved, torch.arange(target + 1, device=costs.device) - width, parent)
            level_parent = torch.where(improved, torch.full_like(parent, level_id), torch.full_like(parent, -1))
            level_choices = torch.where(improved, level_parent, level_choices)
        dp = next_dp
        choices.append(torch.stack((parent, level_choices)))
    if not torch.isfinite(dp[target]):
        raise ValueError("budget cannot be represented by width options")
    widths = torch.empty(experts, device=costs.device, dtype=torch.long)
    current = target
    for expert_id in range(experts - 1, -1, -1):
        parent, level = choices[expert_id][:, current]
        if int(level.item()) < 0:
            raise RuntimeError("allocator backtracking failed")
        widths[expert_id] = options[level]
        current = int(parent.item())
    return widths