from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from AIMER_Mix.mix_core import descending_unit_ranks


SOURCE_NAMES = ("pp", "prp", "layerprop")


@dataclass(frozen=True)
class PseudoSource:
    """A per-expert pseudo-token ranking source.

    ``order`` is a complete channel permutation with shape
    ``[layers, experts, channels]``. ``coverage`` and ``stability`` are optional
    ``[layers, experts]`` confidence signals in
    ``[0, 1]``. ``hit_counts`` is an optional non-negative
    ``[layers, experts]`` LayerProp routing count used for adaptive LP+PRP
    mixing. Missing confidence defaults to one; a missing source is represented
    by ``None`` at the caller and is not treated as a failure.
    """

    name: str
    order: torch.Tensor
    coverage: torch.Tensor | None = None
    stability: torch.Tensor | None = None
    hit_counts: torch.Tensor | None = None
    base_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, layers: int, experts: int, channels: int) -> None:
        if self.name not in SOURCE_NAMES:
            raise ValueError(f"Unsupported pseudo source: {self.name!r}")
        if self.order.shape != (layers, experts, channels):
            raise ValueError(
                f"{self.name} order must have shape {(layers, experts, channels)}, "
                f"got {tuple(self.order.shape)}"
            )
        expected = torch.arange(channels, dtype=torch.long, device=self.order.device)
        if not torch.equal(
            torch.sort(self.order.to(torch.long), dim=2).values,
            expected.expand(layers, experts, -1),
        ):
            raise ValueError(f"{self.name} order rows must be complete channel permutations")
        for label, values in (("coverage", self.coverage), ("stability", self.stability)):
            if values is not None:
                if values.shape != (layers, experts):
                    raise ValueError(f"{self.name} {label} must have shape {(layers, experts)}")
                if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
                    raise ValueError(f"{self.name} {label} must be finite and in [0, 1]")
        if self.hit_counts is not None:
            if self.hit_counts.shape != (layers, experts):
                raise ValueError(f"{self.name} hit_counts must have shape {(layers, experts)}")
            if not bool(torch.isfinite(self.hit_counts).all()) or bool((self.hit_counts < 0).any()):
                raise ValueError(f"{self.name} hit_counts must be finite and non-negative")
        if self.base_weight < 0:
            raise ValueError("Pseudo source base_weight must be non-negative")


@dataclass(frozen=True)
class AIMERMixPlusConfig:
    """Fusion policy shared by Qwen3, Qwen3.6, Gemma4, and DeepSeek-V2."""

    # Preserve a high-confidence AIMER-Mix core, but allow pseudo evidence to
    # replace part of the boundary. This is a budget on positions, not a veto.
    boundary_fraction: float = 0.20
    minimum_boundary_channels: int = 32
    maximum_boundary_fraction: float = 0.35
    base_boundary_weight: float = 0.75
    pseudo_weight: float = 1.0
    pseudo_floor: float = 0.70
    agreement_bonus: float = 0.15
    disagreement_penalty: float = 0.05
    rank_temperature: float = 1.0
    epsilon: float = 1.0e-8
    # When True the keep-set is ranked only by pseudo sources (LayerProp/PRP/PP).
    # AIMER-Mix still supplies a fallback order if no source is active.
    ignore_base: bool = False
    # Mix LayerProp and PRP per expert as λ S_LP + (1-λ) S_PRP,
    # λ = N / (N + tau), where N is LayerProp hit count.
    adaptive_lp_prp: bool = False
    layerprop_tau: float = 8.0
    source_weights: tuple[tuple[str, float], ...] = (
        ("pp", 1.0),
        ("prp", 1.0),
        ("layerprop", 1.0),
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must be in (0, 1]")
        if not 0.0 < self.maximum_boundary_fraction <= 1.0:
            raise ValueError("maximum_boundary_fraction must be in (0, 1]")
        if self.boundary_fraction > self.maximum_boundary_fraction:
            raise ValueError("boundary_fraction cannot exceed maximum_boundary_fraction")
        if self.minimum_boundary_channels < 1:
            raise ValueError("minimum_boundary_channels must be positive")
        if not 0.0 <= self.pseudo_floor <= 1.0:
            raise ValueError("pseudo_floor must be in [0, 1]")
        if self.base_boundary_weight < 0.0 or self.pseudo_weight < 0.0:
            raise ValueError("base_boundary_weight and pseudo_weight must be non-negative")
        if self.agreement_bonus < 0.0 or self.disagreement_penalty < 0.0:
            raise ValueError("agreement_bonus and disagreement_penalty must be non-negative")
        if self.rank_temperature <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("rank_temperature and epsilon must be positive")
        if self.layerprop_tau <= 0.0:
            raise ValueError("layerprop_tau must be positive")

    def weight_for(self, name: str) -> float:
        return dict(self.source_weights).get(name, 0.0)


def rank_percentiles_from_order(order: torch.Tensor) -> torch.Tensor:
    """Convert complete descending permutations to per-row importance ranks."""

    if order.ndim != 3:
        raise ValueError("order must have shape [layers, experts, channels]")
    layers, experts, channels = order.shape
    expected = torch.arange(channels, dtype=torch.long, device=order.device)
    if not torch.equal(
        torch.sort(order.to(torch.long), dim=2).values,
        expected.expand(layers, experts, -1),
    ):
        raise ValueError("order rows must be complete channel permutations")
    ranks = torch.zeros((layers, experts, channels), dtype=torch.float32, device=order.device)
    if channels == 1:
        ranks.fill_(1.0)
        return ranks
    values = torch.linspace(1.0, 0.0, channels, device=order.device, dtype=torch.float32)
    ranks.scatter_(2, order.to(torch.long), values.expand(layers, experts, -1))
    return ranks


def layerprop_mix_lambda(hit_count: float, tau: float) -> float:
    """Coverage-gated LayerProp share: λ = N / (N + τ). Uncovered experts get λ = 0."""

    if tau <= 0.0:
        raise ValueError("tau must be positive")
    count = float(hit_count)
    if not math.isfinite(count):
        return 1.0
    count = max(count, 0.0)
    return count / (count + tau)


def _layerprop_hit_count(source: PseudoSource, layer_id: int, expert_id: int) -> float:
    if source.hit_counts is not None:
        return float(source.hit_counts[layer_id, expert_id].clamp_min(0.0).item())
    if source.coverage is None:
        return 0.0
    coverage = float(source.coverage[layer_id, expert_id].item())
    if coverage <= 0.0:
        return 0.0
    # Old caches only stored binary coverage. Treat a covered expert as
    # well-supported so λ → 1, and an uncovered expert as λ = 0.
    if coverage <= 1.0:
        return float("inf")
    return coverage


def _confidence(source: PseudoSource, layer_id: int, expert_id: int, device: torch.device) -> float:
    confidence = torch.tensor(1.0, dtype=torch.float32, device=device)
    if source.coverage is not None:
        confidence = confidence * source.coverage[layer_id, expert_id].to(device=device, dtype=torch.float32)
    if source.stability is not None:
        confidence = confidence * source.stability[layer_id, expert_id].to(device=device, dtype=torch.float32)
    return float(confidence.clamp_min(0.0).clamp_max(1.0).item())


def _rescue_size(retained_channels: int, config: AIMERMixPlusConfig) -> int:
    if config.ignore_base:
        return retained_channels
    requested = max(config.minimum_boundary_channels, round(retained_channels * config.boundary_fraction))
    maximum = max(config.minimum_boundary_channels, round(retained_channels * config.maximum_boundary_fraction))
    return min(retained_channels - 1, maximum, requested)


def _fuse_one_expert(
    base_rank: torch.Tensor,
    retained_channels: int,
    sources: list[tuple[PseudoSource, torch.Tensor, float]],
    config: AIMERMixPlusConfig,
    *,
    layer_id: int = 0,
    expert_id: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channels = int(base_rank.numel())
    rescue = _rescue_size(retained_channels, config)
    core_count = retained_channels - rescue
    base_order = torch.argsort(base_rank, descending=True, stable=True)
    core = base_order[:core_count]
    base_rescue = base_order[core_count:retained_channels]
    if not sources:
        return base_order, {
            "rescue_channels": rescue,
            "core_channels": core_count,
            "active_sources": [],
            "pseudo_mass": 0.0,
            "agreement_mass": 0.0,
        }

    adaptive_weights: dict[str, float] | None = None
    layerprop_lambda = None
    layerprop_hits = None
    if config.adaptive_lp_prp:
        present = {source.name for source, _ranks, _confidence in sources}
        if "layerprop" in present and "prp" in present:
            layerprop_source = next(source for source, _ranks, _confidence in sources if source.name == "layerprop")
            layerprop_hits = _layerprop_hit_count(layerprop_source, layer_id, expert_id)
            layerprop_lambda = layerprop_mix_lambda(layerprop_hits, config.layerprop_tau)
            adaptive_weights = {
                "layerprop": float(layerprop_lambda),
                "prp": float(1.0 - layerprop_lambda),
            }

    # Pseudo sources compete only for the boundary budget. Their score remains
    # capable of selecting channels outside the AIMER-Mix boundary, which is
    # the intended Gemma4 recovery path.
    pseudo_scores = torch.zeros(channels, dtype=torch.float32, device=base_rank.device)
    nominal_weight_total = 0.0
    source_ranks: list[torch.Tensor] = []
    source_confidences: list[float] = []
    active_names: list[str] = []
    for source, ranks, confidence in sources:
        if adaptive_weights is not None and source.name in adaptive_weights:
            nominal_weight = float(adaptive_weights[source.name])
            if nominal_weight <= 0.0:
                continue
            weight = nominal_weight
            confidence = 1.0
        else:
            nominal_weight = float(config.weight_for(source.name) * source.base_weight)
            if nominal_weight <= 0.0 or confidence <= 0.0:
                continue
            weight = nominal_weight * float(confidence)
        ranks = ranks.clamp_min(0.0).pow(1.0 / config.rank_temperature)
        source_ranks.append(ranks)
        source_confidences.append(float(confidence))
        active_names.append(source.name)
        pseudo_scores = pseudo_scores + weight * ranks
        nominal_weight_total += nominal_weight

    if not source_ranks:
        return base_order, {
            "rescue_channels": rescue,
            "core_channels": core_count,
            "active_sources": [],
            "pseudo_mass": 0.0,
            "agreement_mass": 0.0,
        }

    # Divide by the maximum weight available from the sources that are
    # actually present, not by their confidence-reduced effective weight.
    # Consequently low coverage/stability weakens pseudo evidence, while a
    # missing PP/PRP/LayerProp source does not penalize the remaining sources.
    pseudo_scores = pseudo_scores / max(nominal_weight_total, config.epsilon)
    # Agreement is a bonus, not a gate. A channel supported by one strong
    # source can still win; agreement only improves its rescue score.
    agreement = torch.zeros(channels, dtype=torch.float32, device=base_rank.device)
    if len(source_ranks) > 1:
        pair_count = 0
        for left_index in range(len(source_ranks)):
            for right_index in range(left_index + 1, len(source_ranks)):
                pair_confidence = (source_confidences[left_index] * source_confidences[right_index]) ** 0.5
                agreement = agreement + pair_confidence * (
                    (source_ranks[left_index] > config.pseudo_floor)
                    & (source_ranks[right_index] > config.pseudo_floor)
                ).to(torch.float32)
                pair_count += 1
        agreement = agreement / float(max(pair_count, 1))
    if len(source_ranks) > 1:
        if config.ignore_base:
            stacked = torch.stack(source_ranks)
        else:
            stacked = torch.stack([
                confidence * ranks + (1.0 - confidence) * base_rank
                for ranks, confidence in zip(source_ranks, source_confidences)
            ])
        disagreement = stacked.std(dim=0, unbiased=False)
    else:
        disagreement = torch.zeros_like(base_rank)
    base_w = 0.0 if config.ignore_base else config.base_boundary_weight
    fused_scores = (
        base_w * base_rank
        + config.pseudo_weight * pseudo_scores
        + config.agreement_bonus * agreement
        - config.disagreement_penalty * disagreement
    )

    # Keep the AIMER core fixed unless ignore_base, then let remaining channels
    # compete for the rescue budget. LayerProp/PRP can recover channels that
    # AIMER-Mix placed below the original Top-K boundary.
    core_mask = torch.zeros(channels, dtype=torch.bool, device=base_rank.device)
    if core.numel() > 0:
        core_mask[core] = True
    candidates = torch.where(~core_mask)[0]
    candidate_order = candidates[torch.argsort(fused_scores[candidates], descending=True, stable=True)]
    selected_rescue = candidate_order[:rescue]
    selected = selected_rescue if core.numel() == 0 else torch.cat((core, selected_rescue))
    selected_mask = torch.zeros(channels, dtype=torch.bool, device=base_rank.device)
    selected_mask[selected] = True
    tail = base_order[~selected_mask[base_order]]
    order = torch.cat((selected, tail))
    return order, {
        "rescue_channels": rescue,
        "core_channels": core_count,
        "active_sources": active_names,
        "pseudo_mass": float(pseudo_scores[selected_rescue].sum().item()),
        "agreement_mass": float(agreement[selected_rescue].sum().item()),
        "selected_rescue": selected_rescue,
        "base_rescue": base_rescue,
        "swap_count": int((~torch.isin(selected_rescue, base_rescue)).sum().item()),
        "layerprop_lambda": layerprop_lambda,
        "layerprop_hits": None if layerprop_hits is None or not math.isfinite(float(layerprop_hits)) else float(layerprop_hits),
    }


def build_plus_ranking(
    aimer_mix_scores: torch.Tensor,
    retained_channels: int,
    pseudo_sources: list[PseudoSource] | None = None,
    config: AIMERMixPlusConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build AMP rankings from AIMER-Mix scores and optional pseudo sources.

    Inputs are ``[experts, channels]``. A source may provide only a ranking
    permutation; it is converted to a rank percentile before fusion. Missing
    or conflicting sources never force a baseline fallback.
    """

    config = config or AIMERMixPlusConfig()
    if aimer_mix_scores.ndim != 3:
        raise ValueError("aimer_mix_scores must have shape [layers, experts, channels]")
    layers, experts, channels = map(int, aimer_mix_scores.shape)
    if not 1 < int(retained_channels) < channels:
        raise ValueError("retained_channels must be in (1, channels)")
    sources = pseudo_sources or []
    source_names = [source.name for source in sources]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Pseudo source names must be unique")
    for source in sources:
        source.validate(layers, experts, channels)
    source_ranks = {
        source.name: rank_percentiles_from_order(source.order.to(aimer_mix_scores.device))
        for source in sources
    }
    orders: list[torch.Tensor] = []
    diagnostics: list[list[dict[str, Any]]] = []
    base_ranks = descending_unit_ranks(aimer_mix_scores.reshape(layers * experts, channels)).reshape(
        layers, experts, channels
    )
    for layer_id in range(layers):
        layer_orders: list[torch.Tensor] = []
        layer_diagnostics: list[dict[str, Any]] = []
        for expert_id in range(experts):
            expert_sources = [
                (
                    source,
                    source_ranks[source.name][layer_id, expert_id],
                    _confidence(source, layer_id, expert_id, aimer_mix_scores.device),
                )
                for source in sources
            ]
            order, info = _fuse_one_expert(
                base_ranks[layer_id, expert_id],
                int(retained_channels),
                expert_sources,
                config,
                layer_id=layer_id,
                expert_id=expert_id,
            )
            layer_orders.append(order)
            layer_diagnostics.append({"layer_id": layer_id, "expert_id": expert_id, **info})
        orders.append(torch.stack(layer_orders))
        diagnostics.append(layer_diagnostics)
    return torch.stack(orders), {
        "schema_version": 1,
        "method": "aimer_mix_plus",
        "base": "aimer_mix",
        "retained_channels": int(retained_channels),
        "sources": [source.name for source in sources],
        "config": {
            "boundary_fraction": config.boundary_fraction,
            "minimum_boundary_channels": config.minimum_boundary_channels,
            "maximum_boundary_fraction": config.maximum_boundary_fraction,
            "pseudo_floor": config.pseudo_floor,
            "base_boundary_weight": config.base_boundary_weight,
            "pseudo_weight": config.pseudo_weight,
            "agreement_bonus": config.agreement_bonus,
            "disagreement_penalty": config.disagreement_penalty,
            "source_weights": dict(config.source_weights),
            "ignore_base": bool(config.ignore_base),
            "adaptive_lp_prp": bool(config.adaptive_lp_prp),
            "layerprop_tau": float(config.layerprop_tau),
        },
        "diagnostics": diagnostics,
    }


def build_plus_ranking_from_order(
    aimer_mix_order: torch.Tensor,
    retained_channels: int,
    pseudo_sources: list[PseudoSource] | None = None,
    config: AIMERMixPlusConfig | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fuse pseudo rankings while preserving the exact AIMER-Mix base order.

    Ranking-only AIMER-Mix caches are common in the repository and may not
    retain the original continuous score tensor. Converting their complete
    permutation to unique percentile values makes the no-source result exactly
    equal to the cached order, while still exposing the same boundary-rescue
    policy as :func:`build_plus_ranking`.
    """

    if aimer_mix_order.ndim != 3:
        raise ValueError("aimer_mix_order must have shape [layers, experts, channels]")
    base_scores = rank_percentiles_from_order(aimer_mix_order)
    return build_plus_ranking(
        base_scores,
        retained_channels=retained_channels,
        pseudo_sources=pseudo_sources,
        config=config,
    )
