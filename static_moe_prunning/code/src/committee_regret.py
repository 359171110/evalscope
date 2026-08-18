from __future__ import annotations

import torch


def build_frontier_regret_floors(
    block_regret_folds: torch.Tensor,
    reference_widths: torch.Tensor,
    *,
    global_quantile: float,
    width_increment: int = 1,
    aggregation: str = "minimum",
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Protect high-regret blocks immediately beyond a frozen profile frontier."""

    if block_regret_folds.ndim != 4:
        raise ValueError("block regret folds must be [folds, layers, experts, blocks].")
    folds, layers, experts, blocks = block_regret_folds.shape
    if folds < 1 or reference_widths.shape != (layers, experts):
        raise ValueError("reference widths must match block-regret layers and experts.")
    if not bool(torch.isfinite(block_regret_folds).all()) or bool(
        (block_regret_folds < 0).any()
    ):
        raise ValueError("block regret folds must be finite and non-negative.")
    widths = reference_widths.to(torch.long)
    if bool((widths < 0).any()) or bool((widths > blocks).any()):
        raise ValueError("reference widths are outside the available block range.")
    quantile = float(global_quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("global quantile must be in [0, 1].")
    increment = int(width_increment)
    if increment < 1:
        raise ValueError("width increment must be positive.")
    if aggregation not in {"mean", "minimum"}:
        raise ValueError("frontier aggregation must be mean or minimum.")

    values = block_regret_folds.to(torch.float64)
    eligible = widths < blocks
    fold_frontier = torch.zeros(
        folds, layers, experts, dtype=torch.float64, device=values.device
    )
    for layer in range(layers):
        layer_eligible = torch.nonzero(eligible[layer], as_tuple=False).flatten()
        if layer_eligible.numel() == 0:
            continue
        layer_widths = widths[layer, layer_eligible].to(values.device)
        for fold in range(folds):
            raw = values[fold, layer, layer_eligible, layer_widths]
            normalized = raw / raw.mean().clamp_min(eps)
            fold_frontier[fold, layer, layer_eligible] = normalized
    if aggregation == "mean":
        score = fold_frontier.mean(dim=0)
    else:
        score = fold_frontier.min(dim=0).values
    eligible_scores = score[eligible]
    if eligible_scores.numel() == 0:
        raise ValueError("reference profile has no pruned frontier blocks.")
    threshold = torch.quantile(eligible_scores, quantile)
    selected = eligible & (score >= threshold) & (score > 0)
    min_widths = torch.zeros_like(widths)
    min_widths[selected] = (widths[selected] + increment).clamp_max(blocks)
    selected_experts = [
        {
            "layer": int(layer),
            "expert": int(expert),
            "score": float(score[layer, expert]),
            "reference_width": int(widths[layer, expert]),
            "frontier_block": int(widths[layer, expert]),
            "min_width": int(min_widths[layer, expert]),
        }
        for layer, expert in torch.nonzero(selected, as_tuple=False).tolist()
    ]
    audit = {
        "fold_count": int(folds),
        "aggregation": aggregation,
        "global_quantile": quantile,
        "quantile_threshold": float(threshold),
        "width_increment": increment,
        "eligible_count": int(eligible.sum()),
        "selected_count": len(selected_experts),
        "selected_experts": selected_experts,
    }
    return min_widths.cpu(), audit, score.to(block_regret_folds.dtype).cpu()


def diagonal_block_committee_residual(
    middle: torch.Tensor,
    down_weight: torch.Tensor,
    other_output: torch.Tensor,
    *,
    routing_weights: torch.Tensor,
    ranked_indices: torch.Tensor,
    block_sizes: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Estimate routed block-output energy not aligned with peer experts.

    The estimator keeps the diagonal of the down-projection Gram matrix.  This
    avoids materializing one hidden-size output per token and block while still
    moving the redundancy test to the channel blocks that are actually pruned.
    """

    if middle.ndim != 2 or down_weight.ndim != 2 or other_output.ndim != 2:
        raise ValueError("middle, down weight, and other output must be matrices.")
    token_count, intermediate_size = middle.shape
    hidden_size, down_intermediate = down_weight.shape
    if down_intermediate != intermediate_size:
        raise ValueError("down weight intermediate dimension does not match middle.")
    if other_output.shape != (token_count, hidden_size):
        raise ValueError("other output shape does not match token and hidden dimensions.")
    if routing_weights.shape != (token_count,):
        raise ValueError("routing weights must contain one value per token.")
    if ranked_indices.ndim != 1 or ranked_indices.numel() != intermediate_size:
        raise ValueError("ranked indices must contain every intermediate channel.")
    if block_sizes.ndim != 1 or int(block_sizes.sum().item()) != intermediate_size:
        raise ValueError("block sizes must partition the intermediate channels.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    for name, tensor in (
        ("middle", middle),
        ("down weight", down_weight),
        ("other output", other_output),
        ("routing weights", routing_weights),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite.")

    order = ranked_indices.to(device=middle.device, dtype=torch.long)
    if int(order.min().item()) < 0 or int(order.max().item()) >= intermediate_size:
        raise ValueError("ranked indices are out of bounds.")
    if int(torch.unique(order).numel()) != intermediate_size:
        raise ValueError("ranked indices must be a permutation.")

    compute_dtype = torch.float32
    middle_f = middle.to(compute_dtype)
    down_f = down_weight.to(device=middle.device, dtype=compute_dtype)
    other_f = other_output.to(device=middle.device, dtype=compute_dtype)
    other_norm = other_f.norm(dim=1, keepdim=True)
    other_unit = other_f / other_norm.clamp_min(eps)
    peer_projection = other_unit @ down_f
    down_norm_square = down_f.square().sum(dim=0).unsqueeze(0)
    orthogonal_column_norm_square = (
        down_norm_square - peer_projection.square()
    ).clamp_min(0.0)
    channel_energy = (
        routing_weights.to(device=middle.device, dtype=compute_dtype)
        .abs()
        .square()
        .unsqueeze(1)
        * middle_f.square()
        * orthogonal_column_norm_square
    )
    ranked_energy = channel_energy.index_select(1, order)
    block_values = []
    begin = 0
    for raw_size in block_sizes.tolist():
        size = int(raw_size)
        block_values.append(
            ranked_energy[:, begin : begin + size].sum(dim=1).clamp_min(0.0).sqrt()
        )
        begin += size
    return torch.stack(block_values, dim=1).to(middle.dtype)
