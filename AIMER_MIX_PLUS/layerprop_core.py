"""Data-free LayerProp scoring from native decoder residuals.

PP builds probes from router rows. Gemma4's router RMS/scale path is not the
expert FFN input, so LayerProp instead ranks channels from hidden states that
have already gone through the native decoder block, then through
``pre_feedforward_layernorm_2``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from AIMER_MIX_PLUS.build_pseudo_source_artifacts import activation_response


def synthetic_input_ids(
    *,
    vocab_size: int,
    num_sequences: int,
    sequence_length: int,
    bos_token_id: int,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Build a deterministic vocab lattice. This is not a calibration corpus."""

    if vocab_size < 3:
        raise ValueError("vocab_size must be at least 3")
    if num_sequences < 1 or sequence_length < 1:
        raise ValueError("num_sequences and sequence_length must be positive")
    total = int(num_sequences) * int(sequence_length)
    usable = torch.tensor(
        [idx for idx in range(vocab_size) if idx not in {int(pad_token_id), int(bos_token_id)}],
        dtype=torch.long,
    )
    if usable.numel() == 0:
        raise ValueError("vocab has no non-special token ids")
    lattice = torch.linspace(0, usable.numel() - 1, total).round().long().clamp(0, usable.numel() - 1)
    ids = usable[lattice].view(int(num_sequences), int(sequence_length))
    ids[:, 0] = int(bos_token_id)
    return ids


def pack_separate_expert_weights(experts: torch.nn.ModuleList) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ModuleList experts into packed ``gate_up`` / ``down`` tensors.

    Shared-expert modules are not included; callers must pass only routed experts.
    """

    gate_up_rows: list[torch.Tensor] = []
    down_rows: list[torch.Tensor] = []
    for expert in experts:
        if expert is None:
            raise ValueError("LayerProp cannot pack a missing routed expert slot")
        gate_up_rows.append(
            torch.cat([expert.gate_proj.weight.detach(), expert.up_proj.weight.detach()], dim=0)
        )
        down_rows.append(expert.down_proj.weight.detach())
    if not gate_up_rows:
        raise ValueError("No routed experts to pack for LayerProp")
    return torch.stack(gate_up_rows), torch.stack(down_rows)


def accumulate_routed_channel_scores(
    hidden_ffn: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    activation: str,
    score_mode: str,
    scores: torch.Tensor,
    mass: torch.Tensor,
    hit_counts: torch.Tensor,
) -> None:
    """Add route-weighted expert channel scores from one native expert call.

    ``hidden_ffn`` must already be in the expert-FFN space (Gemma4:
    ``pre_feedforward_layernorm_2``), not the router-normalized space.
    """

    if hidden_ffn.ndim != 2:
        raise ValueError("hidden_ffn must have shape [tokens, hidden]")
    if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden_ffn.shape[0]:
        raise ValueError("top_k_index/top_k_weights must align with hidden_ffn tokens")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert weights must be packed [experts, ...]")
    num_experts, packed_width, hidden = gate_up_proj.shape
    channels = int(down_proj.shape[-1])
    if packed_width != 2 * channels or down_proj.shape[0] != num_experts or down_proj.shape[1] != hidden:
        raise ValueError("packed gate_up/down shapes do not match")
    if tuple(scores.shape) != (num_experts, channels) or mass.numel() != num_experts:
        raise ValueError("scores/mass must cover every expert")
    if score_mode not in {"activation", "output"}:
        raise ValueError("score_mode must be 'activation' or 'output'")

    tokens = int(hidden_ffn.shape[0])
    top_k = int(top_k_index.shape[1])
    flat_index = top_k_index.reshape(-1).to(dtype=torch.long)
    flat_weight = top_k_weights.reshape(-1).to(dtype=torch.float32)
    token_index = torch.arange(tokens, device=hidden_ffn.device).repeat_interleave(top_k)
    unique_experts = torch.unique(flat_index)
    hidden_ffn = hidden_ffn.to(dtype=torch.float32)
    for expert_id in unique_experts.tolist():
        expert = int(expert_id)
        if expert < 0 or expert >= num_experts:
            continue
        chosen = flat_index == expert
        if not bool(chosen.any()):
            continue
        rows = token_index[chosen]
        weights = flat_weight[chosen].clamp_min(0.0)
        gate = gate_up_proj[expert, :channels]
        up = gate_up_proj[expert, channels:]
        activated = activation_response(hidden_ffn.index_select(0, rows), gate, up, activation).abs()
        contrib = (activated * weights.unsqueeze(1)).sum(0)
        if score_mode == "output":
            contrib = contrib * torch.linalg.vector_norm(down_proj[expert].float(), dim=0)
        scores[expert] += contrib.to(device=scores.device, dtype=scores.dtype)
        mass[expert] += weights.sum().to(device=mass.device, dtype=mass.dtype)
        hit_counts[expert] += chosen.to(dtype=torch.float32).sum().to(device=hit_counts.device)


def finalize_layerprop_scores(
    scores: torch.Tensor,
    mass: torch.Tensor,
    hit_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert accumulated mass into mean scores, coverage, and stability."""

    safe_mass = mass.clamp_min(1.0e-12)
    mean_scores = scores / safe_mass.unsqueeze(1)
    uncovered = mass <= 0
    mean_scores = torch.where(uncovered.unsqueeze(1), torch.zeros_like(mean_scores), mean_scores)
    coverage = (~uncovered).to(dtype=torch.float32)
    expected = hit_counts.float().clamp_min(1.0)
    mean_weight = mass / expected
    stability = torch.where(uncovered, torch.zeros_like(mass), mean_weight.clamp(0.0, 1.0).sqrt())
    return mean_scores, coverage, stability


def layerprop_payload_metadata(
    *,
    num_sequences: int,
    sequence_length: int,
    score_mode: str,
    family: str,
) -> dict[str, Any]:
    return {
        "name": "layerprop",
        "data_free": True,
        "probe_source": "native_decoder_synthetic_token_lattice",
        "calibration_corpus": None,
        "num_sequences": int(num_sequences),
        "sequence_length": int(sequence_length),
        "score_mode": score_mode,
        "model_family": family,
    }
