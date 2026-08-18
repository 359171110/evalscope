from __future__ import annotations

from dataclasses import dataclass

import torch

from NAPS_v2.build_naps_v2_artifacts import rms_norm_rows
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import swiglu_response


@dataclass(frozen=True)
class RouteNCRConfig:
    probes_per_expert: int = 16
    candidates_per_attempt: int = 16
    max_attempts: int = 4
    seed: int = 42
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.probes_per_expert <= 0 or self.candidates_per_attempt <= 0 or self.max_attempts <= 0:
            raise ValueError("RouteNCR probe counts must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("RouteNCR epsilon must be positive")


def native_probe_spaces(
    adapter: PurePseudoModelAdapter,
    latent_rows: torch.Tensor,
    pre_ffw_norm: torch.Tensor | None,
    expert_input_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map data-free latent rows to native router and routed-expert input spaces."""

    epsilon = float(adapter.text_config["rms_norm_eps"])
    if adapter.model_family == "gemma4":
        if expert_input_norm is None:
            raise ValueError("Gemma4 RouteNCR probes require the native sparse-expert input norm")
        route_rows = rms_norm_rows(
            latent_rows,
            torch.ones(latent_rows.shape[-1], dtype=latent_rows.dtype, device=latent_rows.device),
            epsilon,
        )
        expert_rows = rms_norm_rows(latent_rows, expert_input_norm, epsilon)
        return route_rows, expert_rows
    if pre_ffw_norm is None:
        raise ValueError("Qwen RouteNCR probes require the native pre-FFW norm")
    normalized = rms_norm_rows(latent_rows, pre_ffw_norm, epsilon)
    return normalized, normalized


def native_router_forward(
    adapter: PurePseudoModelAdapter,
    route_rows: torch.Tensor,
    router: torch.Tensor,
    router_scale: torch.Tensor | None = None,
    per_expert_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the architecture-native offline router used by RouteNCR probes."""

    if route_rows.ndim != 2 or router.ndim != 2 or route_rows.shape[1] != router.shape[1]:
        raise ValueError("RouteNCR route rows and router weights are not aligned")
    if adapter.model_family == "gemma4":
        if router_scale is None or per_expert_scale is None:
            raise ValueError("Gemma4 RouteNCR routing requires global and per-expert scales")
        router_input = route_rows.float() * router_scale.float() * (router.shape[1] ** -0.5)
        logits = router_input @ router.float().transpose(0, 1)
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, selected = torch.topk(probabilities, k=adapter.router_top_k, dim=-1)
        normalized = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)
        weights = normalized * per_expert_scale.float()[selected]
        return logits, selected, weights
    logits = route_rows.float() @ router.float().transpose(0, 1)
    top_logits, selected = torch.topk(logits, k=adapter.router_top_k, dim=-1)
    return logits, selected, torch.softmax(top_logits, dim=-1)


def build_isotropic_probes(
    adapter: PurePseudoModelAdapter,
    hidden_size: int,
    pre_ffw_norm: torch.Tensor | None,
    expert_input_norm: torch.Tensor | None,
    config: RouteNCRConfig,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    latent_rows = torch.randn(
        (config.probes_per_expert, hidden_size),
        generator=generator,
        device=device,
    )
    _, expert_rows = native_probe_spaces(adapter, latent_rows, pre_ffw_norm, expert_input_norm)
    return expert_rows.detach()


def build_router_conditioned_probes(
    adapter: PurePseudoModelAdapter,
    router: torch.Tensor,
    pre_ffw_norm: torch.Tensor | None,
    expert_input_norm: torch.Tensor | None,
    router_scale: torch.Tensor | None,
    per_expert_scale: torch.Tensor | None,
    config: RouteNCRConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
    """Condition a shared isotropic synthetic prior through native Top-k rejection sampling."""

    generator = torch.Generator(device=router.device)
    generator.manual_seed(config.seed)
    probes_by_expert: list[list[torch.Tensor]] = [[] for _ in range(adapter.num_experts)]
    weights_by_expert: list[list[torch.Tensor]] = [[] for _ in range(adapter.num_experts)]
    routed_candidate_counts = torch.zeros(adapter.num_experts, dtype=torch.long, device=router.device)
    candidate_batch_size = adapter.num_experts * config.candidates_per_attempt
    candidates_attempted_total = 0
    for _ in range(config.max_attempts):
        if all(len(rows) >= config.probes_per_expert for rows in probes_by_expert):
            break
        latent_rows = torch.randn(
            (candidate_batch_size, router.shape[1]),
            generator=generator,
            device=router.device,
        )
        route_rows, expert_rows = native_probe_spaces(
            adapter,
            latent_rows,
            pre_ffw_norm,
            expert_input_norm,
        )
        _, selected, route_weights = native_router_forward(
            adapter,
            route_rows,
            router,
            router_scale,
            per_expert_scale,
        )
        candidates_attempted_total += candidate_batch_size
        for expert_id in range(adapter.num_experts):
            rows, slots = torch.where(selected == expert_id)
            routed_candidate_counts[expert_id] += rows.numel()
            if len(probes_by_expert[expert_id]) >= config.probes_per_expert:
                continue
            needed = config.probes_per_expert - len(probes_by_expert[expert_id])
            for row, slot in zip(rows[:needed].tolist(), slots[:needed].tolist()):
                probes_by_expert[expert_id].append(expert_rows[row].detach())
                weights_by_expert[expert_id].append(route_weights[row, slot].detach())

    counts = torch.tensor([len(rows) for rows in probes_by_expert], dtype=torch.long, device=router.device)
    if bool((counts < config.probes_per_expert).any()):
        missing = torch.where(counts < config.probes_per_expert)[0].tolist()
        raise RuntimeError(
            "RouteNCR shared-prior rejection sampling did not fill every expert budget: "
            f"experts={missing[:16]}, minimum_count={int(counts.min().item())}"
        )
    probes = torch.stack([torch.stack(rows[:config.probes_per_expert]) for rows in probes_by_expert])
    weights = torch.stack([torch.stack(rows[:config.probes_per_expert]) for rows in weights_by_expert])
    diagnostics: dict[str, torch.Tensor | int] = {
        "prior_coverage": routed_candidate_counts > 0,
        "routed_candidate_counts": routed_candidate_counts,
        "accepted_counts": counts,
        "candidate_batch_size": candidate_batch_size,
        "candidates_attempted_total": candidates_attempted_total,
    }
    return probes, weights, diagnostics


def synthetic_channel_score(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    activation: str,
    route_weights: torch.Tensor | None = None,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    if epsilon <= 0.0:
        raise ValueError("RouteNCR epsilon must be positive")
    responses = swiglu_response(probes, gate, up, activation=activation)
    if route_weights is not None:
        if route_weights.ndim != 1 or route_weights.shape[0] != responses.shape[0]:
            raise ValueError("RouteNCR route weights must align with synthetic response rows")
        responses = responses * route_weights.float().unsqueeze(1)
        response_normalizer = route_weights.float().square().sum().clamp_min(epsilon)
    else:
        response_normalizer = torch.tensor(
            responses.shape[0],
            dtype=torch.float32,
            device=responses.device,
        ).clamp_min(epsilon)
    response_energy = responses.float().square().sum(0) / response_normalizer
    return response_energy * down.float().square().sum(0)