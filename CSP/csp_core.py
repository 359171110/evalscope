from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD = 1.0e-12


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_input_scale(input_scale: torch.Tensor | None, hidden_size: int) -> torch.Tensor | None:
    """Validate an optional fixed element-wise input transform."""

    if input_scale is None:
        return None
    scale = input_scale.detach().to(dtype=torch.float32).reshape(-1)
    if int(scale.numel()) != int(hidden_size):
        raise ValueError(f"input_scale must contain {hidden_size} values.")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("input_scale must contain only finite values.")
    return scale


def _effective_rows(rows: torch.Tensor, input_scale: torch.Tensor | None) -> torch.Tensor:
    result = rows.detach().to(dtype=torch.float32)
    if input_scale is not None:
        result = result * input_scale.reshape(1, -1)
    return result


def _log_l1(rows: torch.Tensor) -> torch.Tensor:
    """Compute log L1 row norms with max scaling."""

    maximum = rows.abs().amax(dim=1)
    normalized = rows.abs() / maximum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(1)
    return maximum.clamp_min(torch.finfo(torch.float32).tiny).to(torch.float64).log() + normalized.sum(dim=1).to(torch.float64).log()


def _log_l2_squared(rows: torch.Tensor) -> torch.Tensor:
    """Compute log squared L2 row norms with max scaling."""

    maximum = rows.abs().amax(dim=1)
    normalized = rows / maximum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(1)
    return 2.0 * maximum.clamp_min(torch.finfo(torch.float32).tiny).to(torch.float64).log() + normalized.square().sum(dim=1).to(torch.float64).log()


def _log_l2(rows: torch.Tensor) -> torch.Tensor:
    return 0.5 * _log_l2_squared(rows)


def canonical_structural_score(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    functional_viability_threshold: float = DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
    canonicalize: bool = False,
) -> torch.Tensor:
    """Compute CSP saliency for separate SwiGLU projection tensors.

    If ``canonicalize`` is true, CSP canonicalizes the up/down gauge with
    ``alpha=sqrt(||d||_2 / ||u||_2)`` before scoring. By default it scores the
    raw signature ``[g; u; d]``. In either mode the absolute-mass distribution
    uses ``log(N * ||theta||_2^2 / ||theta||_1^2)`` without allocating a
    concatenated tensor.
    """

    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("gate, up, and down must be rank-2 tensors.")
    if gate.shape != up.shape:
        raise ValueError("gate and up must share the same shape.")
    channels, hidden_size = (int(size) for size in gate.shape)
    if tuple(down.shape) != (hidden_size, channels):
        raise ValueError("down must have shape [hidden, channels].")
    if float(functional_viability_threshold) < 0.0:
        raise ValueError("functional_viability_threshold must be non-negative.")

    scale = effective_input_scale(input_scale, hidden_size)
    gate_f = _effective_rows(gate, scale)
    up_f = _effective_rows(up, scale)
    down_f = down.detach().to(dtype=torch.float32).transpose(0, 1)
    if not bool(torch.isfinite(gate_f).all() and torch.isfinite(up_f).all() and torch.isfinite(down_f).all()):
        raise ValueError("CSP inputs must contain only finite values.")

    log_gate_l1 = _log_l1(gate_f)
    log_up_l1 = _log_l1(up_f)
    log_down_l1 = _log_l1(down_f)
    log_gate_l2 = _log_l2(gate_f)
    log_up_l2 = _log_l2(up_f)
    log_down_l2 = _log_l2(down_f)
    positive = torch.isfinite(log_gate_l1) & torch.isfinite(log_up_l1) & torch.isfinite(log_down_l1)

    if canonicalize:
        # alpha is represented in log space. This is the unique positive
        # minimum-energy representative on the up/down gauge orbit.
        log_alpha = 0.5 * (log_down_l2 - log_up_l2)
        log_signature_l1 = torch.logsumexp(
            torch.stack((log_gate_l1, log_alpha + log_up_l1, log_down_l1 - log_alpha), dim=0), dim=0
        )
        log_signature_l2_sq = torch.logaddexp(
            2.0 * log_gate_l2,
            math.log(2.0) + log_up_l2 + log_down_l2,
        )
    else:
        log_signature_l1 = torch.logsumexp(
            torch.stack((log_gate_l1, log_up_l1, log_down_l1), dim=0), dim=0
        )
        log_signature_l2_sq = torch.logaddexp(
            torch.logaddexp(2.0 * log_gate_l2, 2.0 * log_up_l2),
            2.0 * log_down_l2,
        )
    signature_size = float(3 * hidden_size)
    score = math.log(signature_size) + log_signature_l2_sq - 2.0 * log_signature_l1

    # V=||g||_2||u||_2||d||_2 is gauge invariant. Comparing logs avoids
    # overflow and gives exact zero/degenerate handling for arbitrary scales.
    log_viability = log_gate_l2 + log_up_l2 + log_down_l2
    if functional_viability_threshold > 0.0:
        viable = log_viability > math.log(float(functional_viability_threshold))
    else:
        viable = torch.isfinite(log_viability)
    viable &= positive
    return torch.where(viable, score, torch.full_like(score, float("-inf"))).to(dtype=torch.float32)


def canonical_structural_score_packed(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    input_scale: torch.Tensor | None = None,
    functional_viability_threshold: float = DEFAULT_FUNCTIONAL_VIABILITY_THRESHOLD,
    canonicalize: bool = False,
) -> torch.Tensor:
    """Compute CSP saliency for one packed expert with ``gate_up=[2C,H]``."""

    if gate_up.ndim != 2 or down.ndim != 2:
        raise ValueError("packed gate_up and down must be rank-2 tensors.")
    if gate_up.shape[0] % 2:
        raise ValueError("packed gate_up must have an even number of rows.")
    width = int(gate_up.shape[0] // 2)
    if tuple(down.shape) != (int(gate_up.shape[1]), width):
        raise ValueError("packed gate_up and down shapes do not match.")
    return canonical_structural_score(
        gate_up[:width], gate_up[width:], down,
        input_scale=input_scale,
        functional_viability_threshold=functional_viability_threshold,
        canonicalize=canonicalize,
    )


def rank_channels_by_csp(scores: torch.Tensor) -> torch.Tensor:
    """Sort CSP scores descending with stable lower-index tie breaking."""

    if scores.ndim not in {1, 2}:
        raise ValueError("scores must be [channels] or [experts, channels].")
    return torch.argsort(scores, dim=-1, descending=True, stable=True)


def ranking_table(scores: torch.Tensor, block_size: int) -> dict[str, torch.Tensor | int]:
    """Build complete rankings and aligned block summaries."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [experts, channels].")
    width = int(scores.shape[1])
    block_size = int(block_size)
    if block_size <= 0 or width % block_size:
        raise ValueError("channel count must be divisible by block_size.")
    ranked = rank_channels_by_csp(scores)
    finite_scores = torch.nan_to_num(scores.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ranked_scores = torch.gather(finite_scores, 1, ranked)
    block_scores = ranked_scores.reshape(int(scores.shape[0]), width // block_size, block_size).mean(dim=2)
    coverage = block_scores / block_scores.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return {
        "ranked_indices": ranked.long().cpu(),
        "channel_scores": scores.to(dtype=torch.float32).cpu(),
        "block_relative_scores": block_scores.cpu(),
        "block_coverage_scores": coverage.cpu(),
        "block_sizes": torch.full((width // block_size,), block_size, dtype=torch.long),
        "intermediate_size": width,
    }


def retained_prefix(order: torch.Tensor, retained_channels: int) -> torch.Tensor:
    """Return the highest-ranked CSP channels."""

    if order.ndim != 1 or not 0 < int(retained_channels) <= int(order.numel()):
        raise ValueError("order must be 1-D and retained_channels must be in range.")
    return order[:int(retained_channels)].long()


def validate_rankings(
    table: dict[int, dict[str, object]],
    num_layers: int,
    num_experts: int,
    width: int,
    layer_ids: tuple[int, ...] | list[int] | None = None,
) -> None:
    """Validate that every requested layer contains full permutations."""

    expected_ids = list(range(int(num_layers)) if layer_ids is None else [int(item) for item in layer_ids])
    normalized = {int(layer_id): values for layer_id, values in table.items()}
    if set(normalized) != set(expected_ids):
        raise ValueError("Ranking table does not cover every requested MoE layer.")
    expected = torch.arange(width)
    for layer_id in expected_ids:
        ranking = normalized[layer_id]["ranked_indices"]
        if not isinstance(ranking, torch.Tensor) or tuple(ranking.shape) != (num_experts, width):
            raise ValueError(f"Layer {layer_id} ranking has an invalid shape.")
        if not torch.equal(torch.sort(ranking.long(), dim=1).values, expected.expand(num_experts, -1)):
            raise ValueError(f"Layer {layer_id} ranking rows must be complete channel permutations.")
