from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def route_qwen3_topk(
    router,
    hidden_states: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    retained_expert_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_output = router(hidden_states)
    if isinstance(router_output, tuple) and len(router_output) == 3:
        return router_output
    router_logits = router_output[0] if isinstance(router_output, tuple) else router_output
    routing_logits = router_logits
    if retained_expert_mask is not None:
        mask = retained_expert_mask.to(device=router_logits.device, dtype=torch.bool)
        if mask.ndim != 1 or int(mask.numel()) != int(router_logits.shape[-1]):
            raise ValueError("retained_expert_mask must have shape [num_experts].")
        if int(mask.sum().item()) < int(top_k):
            raise ValueError("retained_expert_mask must retain at least top_k experts.")
        routing_logits = router_logits.masked_fill(~mask.view(1, -1), float("-inf"))
    routing_weights = F.softmax(routing_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return router_logits, routing_weights.to(hidden_states.dtype), selected_experts


def compute_expert_outputs(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tokens, top_k = selected_experts.shape
    outputs = hidden_states.new_zeros((tokens, top_k, hidden_states.shape[-1]))
    active_mask = torch.ones_like(selected_experts, dtype=torch.bool) if keep_mask is None else keep_mask.bool()
    if active_mask.shape != selected_experts.shape:
        raise ValueError("keep_mask shape must match selected_experts shape.")
    active_positions = torch.nonzero(active_mask, as_tuple=False)
    if active_positions.numel() == 0:
        return outputs
    active_experts = selected_experts[active_mask]
    fused = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj") and not isinstance(
        experts, torch.nn.ModuleList
    )
    for expert_idx in torch.unique(active_experts).tolist():
        positions = active_positions[active_experts == expert_idx]
        token_idx = positions[:, 0]
        slot_idx = positions[:, 1]
        current_state = hidden_states[token_idx]
        if fused:
            gate_branch, up_branch = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden = experts.act_fn(gate_branch) * up_branch
            current_hidden = F.linear(current_hidden, experts.down_proj[expert_idx])
        else:
            current_hidden = experts[expert_idx](current_state)
        outputs[token_idx, slot_idx] = current_hidden.to(outputs.dtype)
    return outputs


def compute_fused_weighted_hidden_states_index_add(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    final_hidden = torch.zeros_like(hidden_states)
    num_experts = int(getattr(experts, "num_experts", experts.gate_up_proj.shape[0]))
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    for expert_idx in torch.nonzero(expert_mask.sum(dim=(-1, -2)) > 0, as_tuple=False).flatten().tolist():
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        active = keep_mask[token_idx, top_k_pos]
        if not bool(active.any()):
            continue
        top_k_pos = top_k_pos[active]
        token_idx = token_idx[active]
        current_state = hidden_states[token_idx]
        gate_branch, up_branch = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden = experts.act_fn(gate_branch) * up_branch
        current_hidden = F.linear(current_hidden, experts.down_proj[expert_idx])
        current_hidden = current_hidden * routing_weights[token_idx, top_k_pos, None]
        final_hidden.index_add_(0, token_idx, current_hidden.to(final_hidden.dtype))
    return final_hidden


def compute_moe_weighted_hidden_states(
    hidden_states: torch.Tensor,
    experts,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    keep_mask: Optional[torch.Tensor] = None,
    moe_backend: str = "torch",
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if keep_mask is None:
        keep_mask = torch.ones_like(selected_experts, dtype=torch.bool)
    if keep_mask.shape != selected_experts.shape or routing_weights.shape != selected_experts.shape:
        raise ValueError("routing tensors must match selected_experts shape.")
    backend = str(moe_backend).lower()
    if backend not in {"torch", "triton", "torch_index_add"}:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}.")
    if backend == "torch_index_add" and hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        final_hidden = compute_fused_weighted_hidden_states_index_add(
            hidden_states, experts, selected_experts, routing_weights, keep_mask
        )
        return final_hidden, None, final_hidden.detach() if bool(keep_mask.all()) else None
    expert_outputs = compute_expert_outputs(hidden_states, experts, selected_experts, keep_mask=keep_mask)
    final_hidden = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    return final_hidden, expert_outputs, final_hidden.detach() if bool(keep_mask.all()) else None


def compute_optional_shared_expert_output(
    hidden_states: torch.Tensor,
    shared_expert=None,
    shared_expert_gate=None,
) -> Optional[torch.Tensor]:
    if shared_expert is None or shared_expert_gate is None:
        return None
    return torch.sigmoid(shared_expert_gate(hidden_states)) * shared_expert(hidden_states)