from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

ENP_COS_EPS = 1.0e-8


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


def expert_down_weight(experts: torch.nn.Module, expert_id: int) -> torch.Tensor:
    """Return the down-projection matrix of one routed expert, shape [hidden, channels]."""

    if hasattr(experts, "down_proj") and not isinstance(experts, torch.nn.ModuleList):
        return experts.down_proj[expert_id]
    if hasattr(experts, "experts"):
        return expert_down_weight(experts.experts, expert_id)
    return experts[expert_id].down_proj.weight


@torch.no_grad()
def enp_cos_token_scores(
    middle: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    eps: float = ENP_COS_EPS,
) -> torch.Tensor:
    """ENP-COS token-sum: P_c,t = <m[t,c] W_down[:,c], y_t> / (||y_t||_2 + eps).

    ``middle`` is SwiGLU ``m`` with shape [tokens, channels]. ``down_weight`` is
    ``W_down`` with shape [hidden, channels]. The returned vector is ``sum_t P_c,t``.
    Divide by the token count to obtain the paper mean.
    """

    if middle.ndim != 2:
        raise ValueError("middle must have shape [tokens, channels].")
    if down_weight.ndim != 2 or int(down_weight.shape[1]) != int(middle.shape[1]):
        raise ValueError("down_weight must have shape [hidden, channels] aligned with middle.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    middle_float = middle.float()
    down_float = down_weight.float()
    output = middle_float @ down_float.T
    projection = output @ down_float
    denominator = output.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    return (middle_float * projection / denominator).sum(dim=0)


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


def expert_importance_scores(
    score_sum: torch.Tensor,
    route_count: int,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Mean ENP-COS over routed tokens, or coupled L2 if the expert saw zero tokens."""

    if int(route_count) > 0:
        return score_sum.float() / float(route_count), False
    return weight_only_group_score(gate_weight, up_weight, down_weight), True


@dataclass
class EnpStatistics:
    """Accumulate ENP-COS projection sums on routed tokens only."""

    layer_ids: tuple[int, ...]
    num_experts: int
    hidden_size: int
    intermediate_size: int
    eps: float = ENP_COS_EPS

    def __post_init__(self) -> None:
        self.channel_score_sums: dict[int, torch.Tensor] = {}
        self.route_counts: dict[int, torch.Tensor] = {}

    def update(
        self,
        layer_id: int,
        expert_inputs: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        experts: torch.nn.Module,
    ) -> None:
        del top_k_weights
        if int(layer_id) not in self.layer_ids:
            return
        inputs = expert_inputs.detach().reshape(-1, expert_inputs.shape[-1])
        indices = top_k_index.detach().reshape(-1, top_k_index.shape[-1]).long()
        if inputs.shape[0] != indices.shape[0]:
            raise ValueError("Expert inputs and native route outputs are not row-aligned.")
        if inputs.shape[1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {inputs.shape[1]}.")
        if layer_id not in self.channel_score_sums:
            device = inputs.device
            self.channel_score_sums[layer_id] = torch.zeros(
                self.num_experts, self.intermediate_size, dtype=torch.float32, device=device
            )
            self.route_counts[layer_id] = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        with torch.no_grad():
            for expert_id in torch.unique(indices).tolist():
                row_ids = torch.unique(torch.where(indices == int(expert_id))[0])
                current_inputs = inputs.index_select(0, row_ids)
                middle = expert_channel_response(experts, current_inputs, int(expert_id))
                down = expert_down_weight(experts, int(expert_id))
                self.channel_score_sums[layer_id][expert_id].add_(
                    enp_cos_token_scores(middle, down, eps=self.eps)
                )
                self.route_counts[layer_id][expert_id].add_(int(row_ids.numel()))

    def payload(self) -> dict[str, object]:
        missing = sorted(set(self.layer_ids) - set(self.channel_score_sums))
        if missing:
            raise ValueError(f"No ENP statistics were captured for layers: {missing}")
        return {
            "score_mode": "enp_cos",
            "score_formula": "mean_t <m[t,c] W_down[:,c], y_t> / (||y_t||_2 + 1e-8)",
            "token_aggregation": "unique_routed_mean",
            "eps": float(self.eps),
            "channel_score_sums": {
                layer: value.cpu()
                for layer, value in self.channel_score_sums.items()
            },
            "route_counts": {
                layer: value.cpu()
                for layer, value in self.route_counts.items()
            },
        }


def build_channel_table(raw_scores: torch.Tensor, block_size: int, eps: float = 1.0e-12) -> dict[str, object]:
    """Build the framework-compatible complete channel permutation table."""

    if raw_scores.ndim != 2:
        raise ValueError("raw_scores must have shape [experts, channels].")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive.")
    width = int(raw_scores.shape[1])
    block_sizes = torch.tensor(
        [min(int(block_size), width - begin) for begin in range(0, width, int(block_size))],
        dtype=torch.long,
    )
    ranked_indices = []
    relative_scores = []
    coverage_scores = []
    for expert_scores in raw_scores:
        order = torch.argsort(expert_scores.float(), descending=True, stable=True)
        ordered = expert_scores.float().index_select(0, order)
        shifted = ordered - ordered.min() + float(eps)
        blocks = torch.stack([
            shifted[begin:begin + int(block_size)].sum() for begin in range(0, width, int(block_size))
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
