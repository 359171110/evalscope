"""End-to-end routing-aware self-calibrated pruning algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adapter import ArchitectureAdapter, LayerTrace
from .allocation import allocate_widths
from .config import MethodConfig
from .ops import channel_activation, output_energy_scores, ridge_fold_down
from .structures import CalibrationPools, PruningResult


def _append_capped(table: dict[tuple[int, int], torch.Tensor], key: tuple[int, int], value: torch.Tensor, maximum: int) -> None:
    """Append samples and keep a GPU reservoir-sized prefix."""

    if value.numel() == 0:
        return
    old = table.get(key)
    merged = value if old is None else torch.cat((old, value), dim=0)
    if merged.shape[0] > maximum:
        order = torch.randperm(merged.shape[0], device=merged.device)[:maximum]
        merged = merged.index_select(0, order)
    table[key] = merged.detach()


class RoutingAwarePruner:
    """Implement natural prevalence plus conditional coverage completion."""

    def __init__(self, adapter: ArchitectureAdapter, config: MethodConfig | None = None) -> None:
        self.adapter = adapter
        self.config = config or MethodConfig()
        self.config.validate(source_width=adapter.source_width)

    def collect_calibration(self, natural_input_ids: torch.Tensor) -> CalibrationPools:
        """Collect natural traces, then guided traces only for coverage completion."""

        if natural_input_ids.ndim != 2:
            raise ValueError("natural_input_ids must have shape [sequences, tokens]")
        device = self.adapter.device
        natural_mass = torch.zeros((self.adapter.num_layers, self.adapter.num_experts), device=device, dtype=torch.float64)
        visitation = torch.zeros_like(natural_mass)
        pools = CalibrationPools(natural_mass=natural_mass, natural_visitation=visitation)
        self._collect_into(pools, natural_input_ids.to(device), natural=True)
        natural_tokens = int(natural_input_ids.numel())
        pools.natural_mass.div_(natural_tokens)
        pools.natural_visitation.div_(natural_tokens)
        for _ in range(self.config.guided_sequences):
            deficits = self._deficits(pools)
            if not deficits:
                break
            layer_id, expert_id = max(deficits, key=lambda item: item[2])[:2]
            guided_ids = self.adapter.guided_input_ids(
                layer_id, expert_id, count=1, length=natural_input_ids.shape[1]
            )
            self._collect_into(pools, guided_ids.to(device), natural=False)
        return pools

    def _collect_into(self, pools: CalibrationPools, input_ids: torch.Tensor, *, natural: bool) -> None:
        traces = self.adapter.collect(input_ids)
        target = pools.natural if natural else pools.guided
        for layer_id, trace in enumerate(traces):
            selected = trace.selected_experts.long()
            if selected.ndim != 2:
                raise ValueError("selected_experts must have shape [tokens, top_k]")
            weights = trace.routing_weights.float()
            if weights.shape != selected.shape:
                raise ValueError("routing weights and selected experts must align")
            if natural:
                flat = selected.reshape(-1)
                flat_weights = weights.reshape(-1)
                pools.natural_mass[layer_id].index_add_(0, flat, flat_weights.square().to(pools.natural_mass.dtype))
                pools.natural_visitation[layer_id].index_add_(
                    0, flat, torch.ones_like(flat, dtype=pools.natural_visitation.dtype)
                )
            for expert_id in range(self.adapter.num_experts):
                positions = torch.nonzero(selected == expert_id, as_tuple=False)
                if positions.numel() == 0:
                    continue
                _append_capped(
                    target,
                    (layer_id, expert_id),
                    trace.expert_input.index_select(0, positions[:, 0]),
                    self.config.max_samples_per_expert,
                )

    def _deficits(self, pools: CalibrationPools) -> list[tuple[int, int, int]]:
        deficits = []
        for layer_id in range(self.adapter.num_layers):
            for expert_id in range(self.adapter.num_experts):
                natural = pools.natural.get((layer_id, expert_id))
                guided = pools.guided.get((layer_id, expert_id))
                count = (0 if natural is None else natural.shape[0]) + (0 if guided is None else guided.shape[0])
                if count < self.config.min_samples_per_expert:
                    deficits.append((layer_id, expert_id, self.config.min_samples_per_expert - count))
        return deficits

    def run(self, natural_input_ids: torch.Tensor) -> PruningResult:
        """Run calibration, scoring, native damage estimation, allocation, and folding."""

        pools = self.collect_calibration(natural_input_ids)
        layers, experts, channels = self.adapter.num_layers, self.adapter.num_experts, self.adapter.source_width
        channel_scores = torch.zeros((layers, experts, channels), device=self.adapter.device, dtype=torch.float32)
        rankings = torch.zeros_like(channel_scores, dtype=torch.long)
        width_options = torch.tensor(
            [max(1, round(channels * level)) for level in self.config.width_levels],
            device=self.adapter.device,
            dtype=torch.long,
        )
        distortions = torch.zeros((layers, experts, width_options.numel()), device=self.adapter.device)
        activation_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id in range(layers):
            gate_up, down = self.adapter.expert_weights(layer_id)
            gate_up = gate_up.to(self.adapter.device)
            down = down.to(self.adapter.device)
            for expert_id in range(experts):
                samples = pools.combined(
                    (layer_id, expert_id), self.config.min_samples_per_expert, self.config.max_samples_per_expert
                )
                if samples.shape[0] < self.config.safe_samples_per_expert:
                    rankings[layer_id, expert_id] = torch.arange(channels, device=self.adapter.device)
                    distortions[layer_id, expert_id, 0] = 0.0
                    distortions[layer_id, expert_id, 1:] = float("inf")
                    continue
                gate, up = gate_up[expert_id].chunk(2, dim=0)
                activation = channel_activation(samples, gate, up, self._activation_name())
                activation_cache[layer_id, expert_id] = (activation, down[expert_id])
                scores = output_energy_scores(activation, down[expert_id])
                channel_scores[layer_id, expert_id] = scores
                ranking = torch.argsort(scores, descending=True, stable=True)
                rankings[layer_id, expert_id] = ranking
                for width_id, width in enumerate(width_options):
                    keep = ranking[: int(width.item())]
                    distortion = self._conditional_damage(activation, down[expert_id], keep, layer_id, expert_id, samples)
                    distortions[layer_id, expert_id, width_id] = distortion
        costs = distortions * pools.natural_mass.float().unsqueeze(-1)
        widths = torch.empty((layers, experts), device=self.adapter.device, dtype=torch.long)
        target_budget = round(self.config.retention * experts * channels)
        for layer_id in range(layers):
            unsafe = torch.zeros(experts, device=self.adapter.device, dtype=torch.bool)
            for expert_id in range(experts):
                key = (layer_id, expert_id)
                samples = pools.combined(key, self.config.min_samples_per_expert, self.config.max_samples_per_expert)
                unsafe[expert_id] = samples.shape[0] < self.config.safe_samples_per_expert
            layer_costs = costs[layer_id].clone()
            layer_costs[unsafe] = float("inf")
            if bool(unsafe.any()):
                layer_costs[unsafe, 0] = 0.0
            widths[layer_id] = allocate_widths(layer_costs, width_options, budget=target_budget)
        compensated: dict[tuple[int, int], torch.Tensor] = {}
        compensation_diagnostics: dict[str, dict[str, float]] = {}
        for (layer_id, expert_id), (activation, down) in activation_cache.items():
            width = int(widths[layer_id, expert_id].item())
            ranking = rankings[layer_id, expert_id]
            compensated[layer_id, expert_id], record = ridge_fold_down(
                activation,
                down,
                ranking[:width],
                ridge=self.config.ridge,
                epsilon=self.config.epsilon,
            )
            compensation_diagnostics[f"{layer_id}:{expert_id}"] = record
        diagnostics = {
            "natural_only_routing_statistics": True,
            "guided_sequences_used": sum(1 for value in pools.guided.values() if value.shape[0] > 0),
            "uncovered_experts": [key for key in pools.natural if pools.combined(key, self.config.min_samples_per_expert, self.config.max_samples_per_expert).shape[0] < self.config.safe_samples_per_expert],
            "width_options": width_options,
            "compensation": compensation_diagnostics,
        }
        return PruningResult(
            natural_mass=pools.natural_mass,
            natural_visitation=pools.natural_visitation,
            channel_scores=channel_scores,
            rankings=rankings,
            distortions=distortions,
            costs=costs,
            widths=widths,
            compensated_down=compensated,
            diagnostics=diagnostics,
        )

    def _conditional_damage(
        self,
        activation: torch.Tensor,
        down: torch.Tensor,
        retained: torch.Tensor,
        layer_id: int,
        expert_id: int,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate conditional expert damage without mixing unrelated traces."""

        del samples
        full = activation.float().matmul(down.float().transpose(0, 1))
        retained_activation = activation.float().index_select(1, retained)
        retained_down = down.float().index_select(1, retained)
        candidate = retained_activation.matmul(retained_down.transpose(0, 1))
        damage = (full - candidate).square().sum(dim=-1).mean()
        del layer_id, expert_id
        return damage

    def _activation_name(self) -> str:
        """Read activation metadata from the repository adapter when available."""

        native_adapter = getattr(self.adapter, "native_adapter", None)
        metadata = getattr(native_adapter, "metadata", None)
        return str(getattr(metadata, "activation", "silu") or "silu")