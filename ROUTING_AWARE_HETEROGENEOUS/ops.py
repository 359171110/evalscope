"""GPU-friendly tensor operations for SwiGLU/GELU-gated experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gated_activation(value: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the architecture-native gate activation."""

    if activation in {"silu", "swish"}:
        return F.silu(value)
    if activation in {"gelu", "gelu_pytorch_tanh", "gelu_tanh"}:
        return F.gelu(value, approximate="tanh")
    raise ValueError(f"Unsupported gated activation: {activation!r}")


def channel_activation(
    expert_input: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Compute gated channel activations in float32 while retaining GPU residency."""

    if expert_input.ndim != 2 or gate.ndim != 2 or up.shape != gate.shape:
        raise ValueError("expert input and gate/up projections must be matrices")
    if expert_input.shape[1] != gate.shape[1]:
        raise ValueError("expert input hidden size does not match projections")
    value = expert_input.float()
    return gated_activation(F.linear(value, gate.float()), activation) * F.linear(value, up.float())


def expert_output(activation: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Project channel activations back to hidden space."""

    if activation.ndim != 2 or down.ndim != 2 or activation.shape[1] != down.shape[1]:
        raise ValueError("activation and down projection are not channel-aligned")
    return F.linear(activation.float(), down.float())


def output_energy_scores(activation: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Compute individual channel output-energy saliency."""

    if activation.shape[1] != down.shape[1]:
        raise ValueError("activation and down projection are not channel-aligned")
    return activation.float().square().mean(dim=0) * down.float().square().sum(dim=0)


def ridge_fold_down(
    activation: torch.Tensor,
    down: torch.Tensor,
    retained: torch.Tensor,
    *,
    ridge: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fold deleted channel outputs into retained columns with dual ridge solve."""

    keep = retained.to(device=activation.device, dtype=torch.long).flatten()
    channels = int(activation.shape[1])
    if keep.numel() == 0 or int(torch.unique(keep).numel()) != int(keep.numel()):
        raise ValueError("retained must be a non-empty set of unique channels")
    if bool((keep < 0).any()) or bool((keep >= channels).any()):
        raise ValueError("retained contains an invalid channel")
    all_channels = torch.arange(channels, device=activation.device)
    keep_mask = torch.zeros(channels, dtype=torch.bool, device=activation.device)
    keep_mask[keep] = True
    deleted = all_channels[~keep_mask]
    base_down = down.to(device=activation.device, dtype=torch.float32)
    retained_activation = activation.float().index_select(1, keep)
    retained_down = base_down.index_select(1, keep)
    if deleted.numel() == 0:
        return retained_down.to(dtype=down.dtype, device=down.device), {"error_before": 0.0, "error_after": 0.0}
    deleted_activation = activation.float().index_select(1, deleted)
    gram = retained_activation @ retained_activation.transpose(0, 1)
    scale = gram.diagonal().mean().clamp_min(float(epsilon))
    regularization = float(ridge) * scale
    system = gram + regularization * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(system, deleted_activation)
    delta = retained_activation.transpose(0, 1) @ coefficients
    effective_down = retained_down + base_down.index_select(1, deleted) @ delta.transpose(0, 1)
    lost = deleted_activation @ base_down.index_select(1, deleted).transpose(0, 1)
    residual = lost - retained_activation @ delta @ base_down.index_select(1, deleted).transpose(0, 1)
    denominator = (activation.float() @ base_down.transpose(0, 1)).square().sum().clamp_min(float(epsilon))
    before = lost.square().sum().div(denominator).sqrt()
    after = residual.square().sum().div(denominator).sqrt()
    diagnostics = {
        "error_before": float(before.item()),
        "error_after": float(after.item()),
        "recovery_ratio": float((1.0 - after / before.clamp_min(float(epsilon))).item()),
        "regularization": float(regularization.item()),
    }
    return effective_down.to(dtype=down.dtype, device=down.device), diagnostics