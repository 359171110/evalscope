from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F


@dataclass
class LayerChannelTable:
    ranked_indices: torch.Tensor
    block_relative_scores: torch.Tensor
    block_coverage_scores: torch.Tensor
    block_sizes: torch.Tensor
    intermediate_size: int


ChannelTable = Dict[int, LayerChannelTable]


def _channel_path_score(
    gamma: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Score aligned SwiGLU channels with coupled gate/up/down path norms."""

    gamma_float = gamma.detach().float()
    gate_norm = torch.linalg.vector_norm(
        gate_weight.detach().float() * gamma_float[None, :], dim=1
    )
    up_norm = torch.linalg.vector_norm(
        up_weight.detach().float() * gamma_float[None, :], dim=1
    )
    down_norm = torch.linalg.vector_norm(down_weight.detach().float(), dim=0)
    return (gate_norm * up_norm * down_norm).clamp_min(eps).pow(1.0 / 3.0)


def _build_layer_channel_table_from_raw_scores(
    raw_scores: torch.Tensor,
    block_size: int,
    eps: float = 1.0e-8,
) -> LayerChannelTable:
    if raw_scores.ndim != 2:
        raise ValueError("raw_scores must have shape [num_experts, intermediate_size].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    intermediate_size = int(raw_scores.shape[1])
    block_sizes = torch.tensor(
        [
            min(int(block_size), intermediate_size - start)
            for start in range(0, intermediate_size, int(block_size))
        ],
        dtype=torch.int64,
    )
    ranked_rows = []
    relative_rows = []
    coverage_rows = []
    for raw in raw_scores:
        ranked_score, ranked_idx = torch.sort(raw.float().clamp_min(eps), descending=True)
        block_score = torch.stack(
            [
                ranked_score[start : start + int(block_size)].sum()
                for start in range(0, intermediate_size, int(block_size))
            ]
        )
        ranked_rows.append(ranked_idx.cpu())
        relative_rows.append((block_score / (block_score.max() + eps)).cpu())
        coverage_rows.append((block_score / (block_score.sum() + eps)).cpu())
    return LayerChannelTable(
        ranked_indices=torch.stack(ranked_rows),
        block_relative_scores=torch.stack(relative_rows),
        block_coverage_scores=torch.stack(coverage_rows),
        block_sizes=block_sizes,
        intermediate_size=intermediate_size,
    )


def channel_table_to_payload(table: ChannelTable) -> Dict[int, Dict[str, object]]:
    return {
        int(layer_idx): {
            "ranked_indices": layer.ranked_indices,
            "block_relative_scores": layer.block_relative_scores,
            "block_coverage_scores": layer.block_coverage_scores,
            "block_sizes": layer.block_sizes,
            "intermediate_size": int(layer.intermediate_size),
        }
        for layer_idx, layer in table.items()
    }


def channel_table_from_payload(payload: Mapping[int, Mapping[str, object]]) -> ChannelTable:
    return {
        int(layer_idx): LayerChannelTable(
            ranked_indices=values["ranked_indices"].cpu(),
            block_relative_scores=values["block_relative_scores"].cpu(),
            block_coverage_scores=values["block_coverage_scores"].cpu(),
            block_sizes=values["block_sizes"].cpu(),
            intermediate_size=int(values["intermediate_size"]),
        )
        for layer_idx, values in payload.items()
    }


def channel_layer_to_device(
    channel_layer: LayerChannelTable,
    device: torch.device | str,
) -> LayerChannelTable:
    return LayerChannelTable(
        ranked_indices=channel_layer.ranked_indices.to(device=device),
        block_relative_scores=channel_layer.block_relative_scores.to(device=device),
        block_coverage_scores=channel_layer.block_coverage_scores.to(device=device),
        block_sizes=channel_layer.block_sizes.to(device=device),
        intermediate_size=channel_layer.intermediate_size,
    )


def _expert_activation(expert, gate_hidden: torch.Tensor, up_hidden: torch.Tensor) -> torch.Tensor:
    return getattr(expert, "act_fn", F.silu)(gate_hidden) * up_hidden


def compute_expert_outputs_with_channel_prefixes(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    block_keep_mask: torch.Tensor,
    channel_layer: LayerChannelTable,
) -> torch.Tensor:
    tokens, top_k = selected_experts.shape
    outputs = hidden_states.new_zeros((tokens, top_k, hidden_states.shape[-1]))
    block_counts = block_keep_mask.sum(dim=-1)
    active = block_counts > 0
    if not bool(active.any()):
        return outputs
    fused = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj") and not isinstance(
        experts, torch.nn.ModuleList
    )
    active_positions = torch.nonzero(active, as_tuple=False)
    active_experts = selected_experts[active]
    for expert_idx in torch.unique(active_experts).tolist():
        expert_positions = active_positions[active_experts == expert_idx]
        expert_block_counts = block_counts[expert_positions[:, 0], expert_positions[:, 1]]
        for block_count in torch.unique(expert_block_counts).tolist():
            positions = expert_positions[expert_block_counts == block_count]
            token_idx = positions[:, 0]
            slot_idx = positions[:, 1]
            channel_count = min(
                channel_layer.intermediate_size,
                int(channel_layer.block_sizes[: int(block_count)].sum().item()),
            )
            channel_idx = channel_layer.ranked_indices[expert_idx, :channel_count].to(hidden_states.device)
            current_state = hidden_states[token_idx]
            if fused:
                gate_weight, up_weight = experts.gate_up_proj[expert_idx].chunk(2, dim=0)
                gate_hidden = F.linear(current_state, gate_weight.index_select(0, channel_idx))
                up_hidden = F.linear(current_state, up_weight.index_select(0, channel_idx))
                current_hidden = experts.act_fn(gate_hidden) * up_hidden
                current_hidden = F.linear(current_hidden, experts.down_proj[expert_idx].index_select(1, channel_idx))
            else:
                expert = experts[expert_idx]
                gate_hidden = F.linear(current_state, expert.gate_proj.weight.index_select(0, channel_idx))
                up_hidden = F.linear(current_state, expert.up_proj.weight.index_select(0, channel_idx))
                current_hidden = _expert_activation(expert, gate_hidden, up_hidden)
                current_hidden = F.linear(current_hidden, expert.down_proj.weight.index_select(1, channel_idx))
            outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)
    return outputs