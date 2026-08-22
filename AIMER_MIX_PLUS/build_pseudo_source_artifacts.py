from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors import safe_open

from AIMER_Mix.mix_core import file_sha256
from AIMER_Mix.model_adapter import AIMERMixModelAdapter


SourceMethod = Literal["pp", "prp"]
ScoreMode = Literal["activation", "output"]


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json is missing weight_map")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_first_tensor(
    model_path: Path,
    weight_map: dict[str, str],
    names: Iterable[str],
) -> tuple[str, torch.Tensor]:
    tried = []
    for name in names:
        tried.append(name)
        if name in weight_map:
            return name, load_tensor(model_path, weight_map, name)
    raise KeyError(f"Missing checkpoint tensor; tried: {tried}")


def rms_norm_rows(rows: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    if rows.ndim != 2 or norm_weight.ndim != 1 or rows.shape[1] != norm_weight.numel():
        raise ValueError("rows and norm_weight have incompatible RMSNorm shapes")
    variance = rows.float().square().mean(dim=-1, keepdim=True)
    return rows.float() * torch.rsqrt(variance + float(eps)) * norm_weight.float().unsqueeze(0)


def activation_response(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    gate_hidden = F.linear(probes.float(), gate.float())
    normalized = activation.lower()
    if normalized in {"silu", "swiglu"}:
        activated = F.silu(gate_hidden)
    elif normalized in {"gelu_pytorch_tanh", "gelu_tanh"}:
        activated = F.gelu(gate_hidden, approximate="tanh")
    elif normalized in {"gelu", "gelu_pytorch"}:
        activated = F.gelu(gate_hidden)
    else:
        raise ValueError(f"Unsupported routed-expert activation: {activation!r}")
    return activated * F.linear(probes.float(), up.float())


def aggregate_channel_scores(
    probes: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    activation: str,
    top_q: int,
    score_mode: ScoreMode,
) -> torch.Tensor:
    if probes.ndim != 2 or probes.shape[0] == 0:
        raise ValueError("At least one pseudo probe is required")
    response = activation_response(probes, gate, up, activation).abs()
    selected = min(int(top_q), int(response.shape[0]))
    if selected < 1:
        raise ValueError("top_q must be positive")
    scores = torch.topk(response, k=selected, dim=0, largest=True, sorted=False).values.mean(0)
    if score_mode == "output":
        scores = scores * torch.linalg.vector_norm(down.float(), dim=0)
    elif score_mode != "activation":
        raise ValueError("score_mode must be 'activation' or 'output'")
    return scores


def router_neighbors(router: torch.Tensor, neighbor_count: int, eps: float = 1.0e-12) -> torch.Tensor:
    if router.ndim != 2:
        raise ValueError("router must have shape [experts, hidden]")
    count = int(neighbor_count)
    if not 0 <= count < router.shape[0]:
        raise ValueError("neighbor_count must be in [0, num_experts)")
    normalized = F.normalize(router.float(), dim=1, eps=eps)
    gram = normalized @ normalized.transpose(0, 1)
    gram.fill_diagonal_(-torch.inf)
    adjacent = torch.topk(gram, k=count, dim=1, largest=True, sorted=True).indices
    self_ids = torch.arange(router.shape[0]).unsqueeze(1)
    return torch.cat((self_ids, adjacent.cpu()), dim=1)


def expert_norm_candidates(adapter: AIMERMixModelAdapter, layer_id: int) -> tuple[str, ...]:
    family = adapter.architecture.model_family
    if family == "gemma4":
        prefix = f"model.language_model.layers.{layer_id}"
        return (
            f"{prefix}.pre_feedforward_layernorm_2.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.input_layernorm.weight",
        )
    if family == "qwen3.6":
        prefix = f"model.language_model.layers.{layer_id}"
    else:
        prefix = f"model.layers.{layer_id}"
    return (
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.pre_feedforward_layernorm.weight",
        f"{prefix}.input_layernorm.weight",
    )


def previous_write_gamma_candidates(adapter: AIMERMixModelAdapter, layer_id: int) -> tuple[tuple[str, ...], ...]:
    if adapter.architecture.model_family != "gemma4":
        return ()
    prefix = f"model.language_model.layers.{layer_id}"
    return (
        (
            f"{prefix}.post_feedforward_layernorm.weight",
            f"{prefix}.post_feedforward_layernorm_2.weight",
        ),
    )


def load_previous_write_gamma(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
    hidden_size: int,
) -> torch.Tensor:
    for names in previous_write_gamma_candidates(adapter, layer_id):
        if all(name in weight_map for name in names):
            gamma = torch.ones(hidden_size, dtype=torch.float32)
            for name in names:
                gamma = gamma * load_tensor(model_path, weight_map, name).float()
            return gamma
    return torch.ones(hidden_size, dtype=torch.float32)


def dense_down_candidates(adapter: AIMERMixModelAdapter, layer_id: int) -> tuple[str, ...]:
    family = adapter.architecture.model_family
    if family in {"gemma4", "qwen3.6"}:
        prefix = f"model.language_model.layers.{layer_id}"
    else:
        prefix = f"model.layers.{layer_id}"
    return (
        f"{prefix}.mlp.down_proj.weight",
        f"{prefix}.mlp.shared_experts.down_proj.weight",
        f"{prefix}.mlp.shared_expert.down_proj.weight",
    )


def iter_expert_weights(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    architecture = adapter.architecture
    if architecture.tensor_codec == "packed":
        gate_up = load_tensor(model_path, weight_map, adapter.gate_up_name(layer_id))
        down = load_tensor(model_path, weight_map, adapter.down_name(layer_id))
        width = architecture.intermediate_size
        for expert_id in range(architecture.num_experts):
            yield expert_id, gate_up[expert_id, :width], gate_up[expert_id, width:], down[expert_id]
        return
    for expert_id in range(architecture.num_experts):
        yield (
            expert_id,
            load_tensor(model_path, weight_map, adapter.gate_name(layer_id, expert_id)),
            load_tensor(model_path, weight_map, adapter.up_name(layer_id, expert_id)),
            load_tensor(model_path, weight_map, adapter.down_name(layer_id, expert_id)),
        )


def iter_previous_write_chunks(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
) -> Iterator[torch.Tensor]:
    architecture = adapter.architecture
    if layer_id in set(architecture.moe_layer_ids()):
        if architecture.tensor_codec == "packed":
            down = load_tensor(model_path, weight_map, adapter.down_name(layer_id))
            for expert_id in range(architecture.num_experts):
                yield down[expert_id].transpose(0, 1).float()
        else:
            for expert_id in range(architecture.num_experts):
                yield load_tensor(
                    model_path,
                    weight_map,
                    adapter.down_name(layer_id, expert_id),
                ).transpose(0, 1).float()
        return
    for name in dense_down_candidates(adapter, layer_id):
        if name in weight_map:
            down = load_tensor(model_path, weight_map, name)
            if down.ndim != 2 or down.shape[0] != architecture.hidden_size:
                continue
            yield down.transpose(0, 1).float()
            return


def select_previous_write_probes(
    router: torch.Tensor,
    write_chunks: Iterable[torch.Tensor],
    *,
    previous_gamma: torch.Tensor,
    probe_count: int,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select signed previous-write directions nearest to each current router row."""

    count = int(probe_count)
    if count < 1:
        raise ValueError("probe_count must be positive")
    router_unit = F.normalize(router.float(), dim=1, eps=eps)
    best_scores: torch.Tensor | None = None
    best_probes: torch.Tensor | None = None
    hidden_size = int(router.shape[1])
    if previous_gamma.shape != (hidden_size,):
        raise ValueError("previous write gamma must match hidden size")
    for chunk in write_chunks:
        if chunk.ndim != 2 or chunk.shape[1] != hidden_size:
            raise ValueError("previous write chunks must have shape [candidates, hidden]")
        candidates = F.normalize(chunk.float() * previous_gamma.unsqueeze(0), dim=1, eps=eps)
        logits = router_unit @ candidates.transpose(0, 1)
        keep = min(count, int(candidates.shape[0]))
        values, indices = torch.topk(logits.abs(), k=keep, dim=1, largest=True, sorted=True)
        signed = candidates[indices] * torch.where(
            torch.gather(logits, 1, indices) >= 0,
            1.0,
            -1.0,
        ).unsqueeze(2)
        if best_scores is None or best_probes is None:
            best_scores = values
            best_probes = signed
            continue
        combined_scores = torch.cat((best_scores, values), dim=1)
        combined_probes = torch.cat((best_probes, signed), dim=1)
        keep = min(count, int(combined_scores.shape[1]))
        best_scores, positions = torch.topk(combined_scores, k=keep, dim=1, largest=True, sorted=True)
        best_probes = torch.gather(
            combined_probes,
            1,
            positions.unsqueeze(2).expand(-1, -1, hidden_size),
        )
    if best_scores is None or best_probes is None:
        raise ValueError("No previous-layer write directions are available")
    return best_probes, best_scores


def route_residual_candidates(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
    residuals: torch.Tensor,
    router: torch.Tensor,
    expert_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the model-family router approximation used for data-free probes."""

    architecture = adapter.architecture
    eps = float(adapter.text_config.get("rms_norm_eps", 1.0e-6))
    if architecture.model_family == "gemma4":
        prefix = f"model.language_model.layers.{layer_id}.router"
        scale = load_tensor(model_path, weight_map, f"{prefix}.scale").float()
        per_expert_scale = load_tensor(model_path, weight_map, f"{prefix}.per_expert_scale").float().reshape(-1)
        route_rows = residuals.float() * torch.rsqrt(
            residuals.float().square().mean(-1, keepdim=True) + eps
        )
        router_input = route_rows * scale * (architecture.hidden_size ** -0.5)
        logits = router_input @ router.float().transpose(0, 1)
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, selected = torch.topk(
            probabilities,
            k=architecture.router_top_k,
            dim=-1,
        )
        normalized = top_probabilities / top_probabilities.sum(-1, keepdim=True).clamp_min(1.0e-12)
        weights = normalized * per_expert_scale[selected]
        return selected, weights
    route_rows = rms_norm_rows(residuals, expert_norm, eps)
    logits = route_rows.float() @ router.float().transpose(0, 1)
    top_logits, selected = torch.topk(logits, k=architecture.router_top_k, dim=-1)
    return selected, torch.softmax(top_logits, dim=-1)


def select_routed_previous_write_probes(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
    router: torch.Tensor,
    expert_norm: torch.Tensor,
    write_chunks: Iterable[torch.Tensor],
    previous_gamma: torch.Tensor,
    probe_count: int,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route signed previous-write directions and retain each expert's strongest bucket."""

    architecture = adapter.architecture
    count = int(probe_count)
    if count < 1:
        raise ValueError("probe_count must be positive")
    hidden_size = architecture.hidden_size
    best_scores = torch.full((architecture.num_experts, count), -torch.inf, dtype=torch.float32)
    best_probes = torch.zeros((architecture.num_experts, count, hidden_size), dtype=torch.float32)
    routed_counts = torch.zeros(architecture.num_experts, dtype=torch.long)
    found_any = False
    for chunk in write_chunks:
        if chunk.ndim != 2 or chunk.shape[1] != hidden_size:
            raise ValueError("previous write chunks must have shape [candidates, hidden]")
        positive = F.normalize(chunk.float() * previous_gamma.unsqueeze(0), dim=1, eps=eps)
        candidates = torch.cat((positive, -positive), dim=0)
        selected, route_weights = route_residual_candidates(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            residuals=candidates,
            router=router,
            expert_norm=expert_norm,
        )
        found_any = True
        for expert_id in range(architecture.num_experts):
            rows, slots = torch.where(selected == expert_id)
            if rows.numel() == 0:
                continue
            routed_counts[expert_id] += rows.numel()
            values = route_weights[rows, slots].float()
            probes = candidates.index_select(0, rows)
            combined_scores = torch.cat((best_scores[expert_id], values), dim=0)
            combined_probes = torch.cat((best_probes[expert_id], probes), dim=0)
            keep = min(count, int(combined_scores.numel()))
            scores, positions = torch.topk(combined_scores, k=keep, largest=True, sorted=True)
            best_scores[expert_id, :keep] = scores
            best_probes[expert_id, :keep] = combined_probes.index_select(0, positions)
    if not found_any:
        raise ValueError("No previous-layer write directions are available")
    valid_counts = torch.isfinite(best_scores).sum(1)
    return best_probes, best_scores, torch.minimum(valid_counts, routed_counts)


def scores_to_table(scores: torch.Tensor, block_size: int) -> dict[str, torch.Tensor | int]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [experts, channels]")
    width = int(scores.shape[1])
    if width % int(block_size):
        raise ValueError("channel width must be divisible by block_size")
    ranked = torch.argsort(scores.float(), dim=1, descending=True, stable=True)
    ranked_scores = torch.gather(scores.float(), 1, ranked)
    num_blocks = width // int(block_size)
    block_scores = ranked_scores.reshape(scores.shape[0], num_blocks, int(block_size)).mean(2)
    coverage = block_scores / block_scores.sum(1, keepdim=True).clamp_min(1.0e-12)
    return {
        "ranked_indices": ranked.long().cpu(),
        "channel_scores": scores.float().cpu(),
        "block_relative_scores": block_scores.cpu(),
        "block_coverage_scores": coverage.cpu(),
        "block_sizes": torch.full((num_blocks,), int(block_size), dtype=torch.long),
        "intermediate_size": width,
    }


def collect_pp_scores(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
    neighbor_count: int,
    top_q: int,
    probe_signs: str,
    score_mode: ScoreMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    architecture = adapter.architecture
    router = load_tensor(model_path, weight_map, adapter.router_name(layer_id)).float()
    norm_name, norm = load_first_tensor(
        model_path,
        weight_map,
        expert_norm_candidates(adapter, layer_id),
    )
    neighbors = router_neighbors(router, neighbor_count)
    scores = []
    for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id):
        raw = router.index_select(0, neighbors[expert_id])
        probes = rms_norm_rows(raw, norm.float(), float(adapter.text_config.get("rms_norm_eps", 1.0e-6)))
        if probe_signs == "positive-negative":
            probes = torch.cat((probes, -probes), dim=0)
        elif probe_signs != "positive":
            raise ValueError("probe_signs must be 'positive' or 'positive-negative'")
        scores.append(
            aggregate_channel_scores(
                probes,
                gate,
                up,
                down,
                activation=architecture.activation,
                top_q=top_q,
                score_mode=score_mode,
            ).cpu()
        )
    confidence = torch.ones(architecture.num_experts, dtype=torch.float32)
    return torch.stack(scores), confidence, confidence.clone(), {
        "probe_source": "current_router_self_plus_cosine_neighbors",
        "neighbor_count": int(neighbor_count),
        "probe_signs": probe_signs,
        "expert_norm_tensor": norm_name,
    }


def collect_prp_scores(
    *,
    model_path: Path,
    weight_map: dict[str, str],
    adapter: AIMERMixModelAdapter,
    layer_id: int,
    probe_count: int,
    top_q: int,
    score_mode: ScoreMode,
    fallback_pp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    architecture = adapter.architecture
    previous_layer = int(layer_id) - 1
    norm_name, norm = load_first_tensor(
        model_path,
        weight_map,
        expert_norm_candidates(adapter, layer_id),
    )
    router = load_tensor(model_path, weight_map, adapter.router_name(layer_id)).float()
    previous_gamma = load_previous_write_gamma(
        model_path,
        weight_map,
        adapter,
        previous_layer,
        architecture.hidden_size,
    )
    try:
        probes, affinities, probe_counts = select_routed_previous_write_probes(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            router=router,
            expert_norm=norm.float(),
            write_chunks=iter_previous_write_chunks(model_path, weight_map, adapter, previous_layer),
            previous_gamma=previous_gamma,
            probe_count=probe_count,
        )
    except ValueError as error:
        if "No previous-layer write directions" not in str(error):
            raise
        zeros = torch.zeros(architecture.num_experts, dtype=torch.float32)
        return fallback_pp.clone(), zeros, torch.ones_like(zeros), {
            "probe_source": "pp_fallback_no_previous_write_catalog",
            "previous_layer": previous_layer,
            "expert_norm_tensor": norm_name,
        }
    scores = []
    eps = float(adapter.text_config.get("rms_norm_eps", 1.0e-6))
    for expert_id, gate, up, down in iter_expert_weights(model_path, weight_map, adapter, layer_id):
        count = int(probe_counts[expert_id].item())
        if count == 0:
            scores.append(fallback_pp[expert_id].cpu())
            continue
        expert_probes = rms_norm_rows(probes[expert_id, :count], norm.float(), eps)
        scores.append(
            aggregate_channel_scores(
                expert_probes,
                gate,
                up,
                down,
                activation=architecture.activation,
                top_q=top_q,
                score_mode=score_mode,
            ).cpu()
        )
    coverage = probe_counts.float().div(max(probe_count, 1)).clamp(0.0, 1.0).cpu()
    finite_affinities = torch.where(torch.isfinite(affinities), affinities, torch.zeros_like(affinities))
    stability = finite_affinities.sum(1).div(probe_counts.clamp_min(1)).clamp(0.0, 1.0).sqrt().cpu()
    return torch.stack(scores), coverage, stability, {
        "probe_source": "previous_layer_signed_write_directions_native_topk_filtered",
        "previous_layer": previous_layer,
        "probe_count_max": int(probes.shape[1]),
        "probe_count_mean": float(probe_counts.float().mean().item()),
        "mean_route_weight": float(stability.square().mean().item()),
        "expert_norm_tensor": norm_name,
        "previous_write_gamma": architecture.model_family == "gemma4",
    }


def build_source_payload(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    method: SourceMethod,
    tables: dict[int, dict[str, torch.Tensor | int]],
    coverage: torch.Tensor,
    stability: torch.Tensor,
    layer_metadata: list[dict[str, Any]],
    score_mode: ScoreMode,
    top_q: int,
) -> dict[str, Any]:
    architecture = adapter.architecture
    return {
        "schema_version": 1,
        "purpose": "aimer_mix_plus_pseudo_source",
        "method": f"aimer_mix_plus_{method}",
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "architecture": adapter.metadata(),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": architecture.channel_alignment,
        "table": tables,
        "pseudo_source": {
            "name": method,
            "data_free": True,
            "score_mode": score_mode,
            "top_q": int(top_q),
            "coverage": coverage.float().cpu(),
            "stability": stability.float().cpu(),
            "layers": layer_metadata,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-model PP or PRP rankings for AIMER-Mix-Plus.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--method", choices=("pp", "prp"), required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--router-neighbors", type=int, default=8)
    parser.add_argument("--prp-probe-count", type=int, default=32)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--probe-signs", choices=("positive", "positive-negative"), default="positive")
    parser.add_argument("--score-mode", choices=("activation", "output"), default="activation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_cache.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    weight_map = load_weight_map(model_path)
    adapter = AIMERMixModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    tables: dict[int, dict[str, torch.Tensor | int]] = {}
    coverage_rows = []
    stability_rows = []
    layer_metadata = []
    for layer_id in architecture.moe_layer_ids():
        pp_scores, pp_coverage, pp_stability, pp_metadata = collect_pp_scores(
            model_path=model_path,
            weight_map=weight_map,
            adapter=adapter,
            layer_id=layer_id,
            neighbor_count=int(args.router_neighbors),
            top_q=int(args.top_q),
            probe_signs=args.probe_signs,
            score_mode=args.score_mode,
        )
        if args.method == "pp":
            scores, coverage, stability, metadata = pp_scores, pp_coverage, pp_stability, pp_metadata
        else:
            scores, coverage, stability, metadata = collect_prp_scores(
                model_path=model_path,
                weight_map=weight_map,
                adapter=adapter,
                layer_id=layer_id,
                probe_count=int(args.prp_probe_count),
                top_q=int(args.top_q),
                score_mode=args.score_mode,
                fallback_pp=pp_scores,
            )
        tables[layer_id] = scores_to_table(scores, architecture.channel_alignment)
        coverage_rows.append(coverage)
        stability_rows.append(stability)
        layer_metadata.append({"layer_id": layer_id, **metadata})
        print(f"scored_source={args.method} layer={layer_id}", flush=True)
    payload = build_source_payload(
        model_path=model_path,
        adapter=adapter,
        method=args.method,
        tables=tables,
        coverage=torch.stack(coverage_rows),
        stability=torch.stack(stability_rows),
        layer_metadata=layer_metadata,
        score_mode=args.score_mode,
        top_q=int(args.top_q),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    summary = {
        "schema_version": 1,
        "method": payload["method"],
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "layer_ids": list(architecture.moe_layer_ids()),
        "coverage_mean": float(payload["pseudo_source"]["coverage"].mean().item()),
        "stability_mean": float(payload["pseudo_source"]["stability"].mean().item()),
        "cache_sha256": file_sha256(output_path),
        "pseudo_source": {
            key: value
            for key, value in payload["pseudo_source"].items()
            if key not in {"coverage", "stability"}
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())