from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal

RouteWeighting = Literal["none", "mass", "square"]


def route_sample_weights(route_weights: torch.Tensor, mode: RouteWeighting) -> torch.Tensor:
    """Convert native router probabilities into Wanda observation weights."""

    values = route_weights.detach().float()
    if mode == "none":
        return torch.ones_like(values)
    if mode == "mass":
        return values
    if mode == "square":
        return values.square()
    raise ValueError(f"Unsupported route weighting: {mode!r}.")


def expert_channel_response(experts: torch.nn.Module, inputs: torch.Tensor, expert_id: int) -> torch.Tensor:
    """Evaluate the native gated intermediate response of one routed expert."""

    if hasattr(experts, "gate_up_proj"):
        gate, up = F.linear(inputs, experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
        return experts.act_fn(gate) * up
    if hasattr(experts, "experts"):
        return expert_channel_response(experts.experts, inputs, expert_id)
    expert = experts[expert_id]
    gate = expert.gate_proj(inputs)
    up = expert.up_proj(inputs)
    activation = getattr(expert, "act_fn", getattr(experts, "act_fn", None))
    if activation is None:
        raise ValueError("Could not resolve the expert activation function.")
    return activation(gate) * up


@dataclass
class WandaStatistics:
    """Accumulate routed input and intermediate second moments on their native devices."""

    layer_ids: tuple[int, ...]
    num_experts: int
    hidden_size: int
    intermediate_size: int
    route_weighting: RouteWeighting = "mass"

    def __post_init__(self) -> None:
        self.input_square_sums: dict[int, torch.Tensor] = {}
        self.middle_square_sums: dict[int, torch.Tensor] = {}
        self.weight_sums: dict[int, torch.Tensor] = {}
        self.route_counts: dict[int, torch.Tensor] = {}

    def update(
        self,
        layer_id: int,
        expert_inputs: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        experts: torch.nn.Module,
    ) -> None:
        if int(layer_id) not in self.layer_ids:
            return
        inputs = expert_inputs.detach().reshape(-1, expert_inputs.shape[-1])
        indices = top_k_index.detach().reshape(-1, top_k_index.shape[-1]).long()
        weights = top_k_weights.detach().reshape(-1, top_k_weights.shape[-1]).float()
        if inputs.shape[0] != indices.shape[0] or indices.shape != weights.shape:
            raise ValueError("Expert inputs and native route outputs are not row-aligned.")
        if inputs.shape[1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {inputs.shape[1]}.")
        if layer_id not in self.input_square_sums:
            device = inputs.device
            self.input_square_sums[layer_id] = torch.zeros(
                self.num_experts, self.hidden_size, dtype=torch.float32, device=device
            )
            self.middle_square_sums[layer_id] = torch.zeros(
                self.num_experts, self.intermediate_size, dtype=torch.float32, device=device
            )
            self.weight_sums[layer_id] = torch.zeros(self.num_experts, dtype=torch.float64, device=device)
            self.route_counts[layer_id] = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        with torch.no_grad():
            for expert_id in torch.unique(indices).tolist():
                row_ids, slot_ids = torch.where(indices == int(expert_id))
                current_inputs = inputs.index_select(0, row_ids)
                sample_weights = route_sample_weights(weights[row_ids, slot_ids], self.route_weighting)
                middle = expert_channel_response(experts, current_inputs, int(expert_id))
                self.input_square_sums[layer_id][expert_id].add_(
                    (current_inputs.float().square() * sample_weights[:, None]).sum(dim=0)
                )
                self.middle_square_sums[layer_id][expert_id].add_(
                    (middle.float().square() * sample_weights[:, None]).sum(dim=0)
                )
                self.weight_sums[layer_id][expert_id].add_(sample_weights.double().sum())
                self.route_counts[layer_id][expert_id].add_(int(row_ids.numel()))

    def payload(self) -> dict[str, object]:
        missing = sorted(set(self.layer_ids) - set(self.input_square_sums))
        if missing:
            raise ValueError(f"No Wanda statistics were captured for layers: {missing}")
        return {
            "route_weighting": self.route_weighting,
            "input_square_sums": {
                layer: value.cpu()
                for layer, value in self.input_square_sums.items()
            },
            "middle_square_sums": {
                layer: value.cpu()
                for layer, value in self.middle_square_sums.items()
            },
            "weight_sums": {
                layer: value.cpu()
                for layer, value in self.weight_sums.items()
            },
            "route_counts": {
                layer: value.cpu()
                for layer, value in self.route_counts.items()
            },
        }


def grouped_wanda_score(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_square_sum: torch.Tensor,
    middle_square_sum: torch.Tensor,
    normalizer: float | torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Score one coupled gate-row/up-row/down-column channel group.

    The three branch energies are added in squared L2 space. This preserves the
    classic Wanda factor ``abs(weight) * input_norm`` while making the complete
    gated-MLP dependency group structurally removable.
    """

    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate and up weights must be aligned matrices.")
    if down_weight.ndim != 2 or down_weight.shape[1] != gate_weight.shape[0]:
        raise ValueError("down columns must align with gate/up rows.")
    if input_square_sum.ndim != 1 or input_square_sum.numel() != gate_weight.shape[1]:
        raise ValueError("input statistics do not match the expert hidden size.")
    if middle_square_sum.ndim != 1 or middle_square_sum.numel() != gate_weight.shape[0]:
        raise ValueError("middle statistics do not match the expert width.")
    denominator = max(float(torch.as_tensor(normalizer).item()), eps)
    input_rms = (input_square_sum.float() / denominator).clamp_min(0).sqrt()
    middle_rms = (middle_square_sum.float() / denominator).clamp_min(0).sqrt()
    gate_energy = (gate_weight.float() * input_rms[None, :]).square().sum(dim=1)
    up_energy = (up_weight.float() * input_rms[None, :]).square().sum(dim=1)
    down_energy = down_weight.float().square().sum(dim=0) * middle_rms.square()
    return (gate_energy + up_energy + down_energy).clamp_min(eps).sqrt()


def weight_only_group_score(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Deterministic fallback for experts unseen in train-only calibration."""

    gate = gate_weight.float().square().sum(dim=1)
    up = up_weight.float().square().sum(dim=1)
    down = down_weight.float().square().sum(dim=0)
    return (gate + up + down).clamp_min(eps).sqrt()


def build_channel_table(raw_scores: torch.Tensor, block_size: int, eps: float = 1.0e-12) -> dict[str, object]:
    """Build the framework-compatible complete channel permutation table."""

    if raw_scores.ndim != 2:
        raise ValueError("raw_scores must have shape [experts, channels].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    width = int(raw_scores.shape[1])
    block_sizes = torch.tensor([min(int(block_size), width - begin) for begin in range(0, width, int(block_size))],
                               dtype=torch.long)
    ranked_indices = []
    relative_scores = []
    coverage_scores = []
    for expert_scores in raw_scores:
        order = torch.argsort(expert_scores.float(), descending=True, stable=True)
        ordered = expert_scores.float().index_select(0, order).clamp_min(eps)
        blocks = torch.stack([
            ordered[begin:begin + int(block_size)].sum() for begin in range(0, width, int(block_size))
        ])
        ranked_indices.append(order.cpu())
        relative_scores.append((blocks / blocks.max().clamp_min(eps)).cpu())
        coverage_scores.append((blocks / blocks.sum().clamp_min(eps)).cpu())
    return {
        "ranked_indices": torch.stack(ranked_indices),
        "block_relative_scores": torch.stack(relative_scores),
        "block_coverage_scores": torch.stack(coverage_scores),
        "block_sizes": block_sizes,
        "intermediate_size": width,
    }


def validate_rankings(
    table: dict[int, dict[str, object]],
    num_layers: int,
    num_experts: int,
    width: int,
    layer_ids: tuple[int, ...] | list[int] | None = None,
) -> None:
    """Require a full, duplicate-free channel permutation for every expert."""

    expected_ids = list(range(int(num_layers)) if layer_ids is None else [int(layer_id) for layer_id in layer_ids])
    if set(map(int, table)) != set(expected_ids):
        raise ValueError("Ranking table does not cover every requested MoE layer.")
    expected = torch.arange(width)
    for layer_id in expected_ids:
        ranking = table[layer_id]["ranked_indices"]
        if not isinstance(ranking, torch.Tensor) or tuple(ranking.shape) != (num_experts, width):
            raise ValueError(f"Layer {layer_id} ranking has an invalid shape.")
        if not torch.equal(torch.sort(ranking.long(), dim=1).values, expected.expand(num_experts, -1)):
            raise ValueError(f"Layer {layer_id} ranking rows must be complete channel permutations.")