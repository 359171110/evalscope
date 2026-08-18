from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from NAPS_v2.build_channel_artifacts import nested_order
from NAPS_v2.build_naps_v2_artifacts import (
    file_sha256,
    iter_expert_weights,
    load_norm,
    load_tensor,
    load_weight_map,
)
from NAPS_v2.model_adapter import PurePseudoModelAdapter
from NAPS_v2.naps_v2_core import NapsV2Config, effective_zero_mask, stable_concat_score
from NAPS_v2.prism import (
    RouteNCRConfig,
    build_isotropic_probes,
    build_router_conditioned_probes,
    synthetic_channel_score,
)


DEFAULT_WIDTHS = {
    "qwen3": (256, 384, 512),
    "qwen3.6": (192, 256, 320),
    "gemma4": (352, 384, 416, 448, 480),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict data-free RouteNCR channel-ranking artifacts.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--widths", type=int, nargs="+")
    parser.add_argument("--probes-per-expert", type=int, default=16)
    parser.add_argument("--candidates-per-attempt", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--effective-zero-threshold", type=float, default=1.0e-12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_probe_tensors(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    router = load_tensor(model_path, weight_map, adapter.router_name(layer_id)).to(device)
    if adapter.model_family != "gemma4":
        _, pre_ffw_norm = load_norm(model_path, weight_map, adapter, layer_id)
        return router, pre_ffw_norm.to(device), None, None, None
    expert_input_norm = load_tensor(model_path, weight_map, adapter.expert_input_norm_name(layer_id)).to(device)
    router_scale = load_tensor(model_path, weight_map, adapter.router_scale_name(layer_id)).to(device)
    per_expert_scale = load_tensor(
        model_path,
        weight_map,
        adapter.router_per_expert_scale_name(layer_id),
    ).to(device)
    return router, None, expert_input_norm, router_scale, per_expert_scale


def build_layer_table(
    model_path: Path,
    weight_map: dict[str, str],
    adapter: PurePseudoModelAdapter,
    layer_id: int,
    route_config: RouteNCRConfig,
    score_config: NapsV2Config,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    router, pre_ffw_norm, expert_input_norm, router_scale, per_expert_scale = load_probe_tensors(
        model_path,
        weight_map,
        adapter,
        layer_id,
        device,
    )
    isotropic_probes = build_isotropic_probes(
        adapter,
        adapter.channel_architecture.hidden_size,
        pre_ffw_norm,
        expert_input_norm,
        route_config,
        device,
    )
    route_probes, route_weights, route_diagnostics = build_router_conditioned_probes(
        adapter,
        router,
        pre_ffw_norm,
        expert_input_norm,
        router_scale,
        per_expert_scale,
        route_config,
    )

    aimer_scores = []
    isotropic_scores = []
    route_ncr_scores = []
    down_energies = []
    zero_masks = []
    aimer_orders = []
    isotropic_orders = []
    route_ncr_orders = []
    aimer_nonfinite_counts = []
    with torch.inference_mode():
        for expert_id, gate, up, down in iter_expert_weights(
            model_path,
            weight_map,
            adapter,
            layer_id,
            device,
        ):
            raw_aimer = stable_concat_score(gate, up, down, score_config)
            zero_mask = effective_zero_mask(gate, up, down, score_config.effective_zero_threshold)
            finite_aimer = torch.isfinite(raw_aimer)
            if not bool(finite_aimer.any()):
                aimer = torch.zeros_like(raw_aimer)
            else:
                aimer = raw_aimer.masked_fill(~finite_aimer, raw_aimer[finite_aimer].min())
            isotropic = synthetic_channel_score(
                isotropic_probes,
                gate,
                up,
                down,
                adapter.channel_architecture.activation,
                epsilon=route_config.epsilon,
            )
            route_ncr = synthetic_channel_score(
                route_probes[expert_id],
                gate,
                up,
                down,
                adapter.channel_architecture.activation,
                route_weights[expert_id],
                route_config.epsilon,
            )
            down_energy = down.float().square().sum(0)
            aimer_scores.append(aimer.detach().cpu())
            isotropic_scores.append(isotropic.detach().cpu())
            route_ncr_scores.append(route_ncr.detach().cpu())
            down_energies.append(down_energy.detach().cpu())
            zero_masks.append(zero_mask.detach().cpu())
            aimer_nonfinite_counts.append(int((~finite_aimer).sum().item()))
            aimer_orders.append(nested_order(aimer, zero_mask, aimer).detach().cpu())
            isotropic_orders.append(nested_order(isotropic, zero_mask, aimer).detach().cpu())
            route_ncr_orders.append(nested_order(route_ncr, zero_mask, aimer).detach().cpu())

    routed_counts = route_diagnostics["routed_candidate_counts"].detach().cpu()
    candidates_attempted = int(route_diagnostics["candidates_attempted_total"])
    diagnostics = {
        "layer_id": layer_id,
        "prior_coverage_count": int(route_diagnostics["prior_coverage"].sum().item()),
        "prior_coverage_rate": float(route_diagnostics["prior_coverage"].float().mean().item()),
        "routed_candidate_count_min": int(routed_counts.min().item()),
        "routed_candidate_count_mean": float(routed_counts.float().mean().item()),
        "routing_acceptance_rate_mean": float((routed_counts.float() / max(1, candidates_attempted)).mean().item()),
        "stable_aimer_nonfinite_count": sum(aimer_nonfinite_counts),
        "candidate_batch_size": int(route_diagnostics["candidate_batch_size"]),
        "candidates_attempted_total": candidates_attempted,
    }
    table = {
        "stable_aimer_scores": torch.stack(aimer_scores),
        "isotropic_gaussian_response_scores": torch.stack(isotropic_scores),
        "route_ncr_scores": torch.stack(route_ncr_scores),
        "down_channel_energy": torch.stack(down_energies),
        "effective_zero_masks": torch.stack(zero_masks),
        "stable_aimer_ranked_indices": torch.stack(aimer_orders),
        "isotropic_gaussian_ranked_indices": torch.stack(isotropic_orders),
        "route_ncr_ranked_indices": torch.stack(route_ncr_orders),
        "route_ncr_route_weights": route_weights.detach().cpu(),
        "route_ncr_prior_coverage": route_diagnostics["prior_coverage"].detach().cpu(),
        "route_ncr_routed_candidate_counts": routed_counts,
        "intermediate_size": adapter.intermediate_size,
    }
    return table, diagnostics


def validate_widths(adapter: PurePseudoModelAdapter, widths: tuple[int, ...]) -> None:
    if not widths:
        raise ValueError("RouteNCR requires at least one diagnostic width")
    for width in widths:
        adapter.channel_architecture.validate_width(width)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    layer_ids = list(range(adapter.num_layers)) if args.layers is None else sorted(set(args.layers))
    if not layer_ids or layer_ids[0] < 0 or layer_ids[-1] >= adapter.num_layers:
        raise ValueError("Requested RouteNCR layers are outside the model")
    widths = tuple(sorted(set(args.widths or DEFAULT_WIDTHS[adapter.model_family])))
    validate_widths(adapter, widths)
    route_config = RouteNCRConfig(
        probes_per_expert=args.probes_per_expert,
        candidates_per_attempt=args.candidates_per_attempt,
        max_attempts=args.max_attempts,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    score_config = NapsV2Config(effective_zero_threshold=args.effective_zero_threshold)
    device = torch.device(args.device)
    tables = {}
    diagnostics = []
    for layer_id in layer_ids:
        table, layer_diagnostics = build_layer_table(
            model_path,
            weight_map,
            adapter,
            layer_id,
            route_config,
            score_config,
            device,
        )
        tables[layer_id] = table
        diagnostics.append(layer_diagnostics)
        print(
            f"Built RouteNCR layer {layer_id + 1}/{adapter.num_layers}: "
            f"prior_coverage_rate={layer_diagnostics['prior_coverage_rate']:.4f}, "
            f"minimum_routed_candidates={layer_diagnostics['routed_candidate_count_min']}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model_provenance = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
    }
    payload = {
        "schema_version": 1,
        "purpose": "strict_data_free_channel_ranking",
        "method": "route_ncr",
        "method_name": "Router-Conditioned Native Channel Response",
        "model_path": str(model_path),
        "model_family": adapter.model_family,
        "source_intermediate_size": adapter.intermediate_size,
        "channel_alignment": adapter.channel_architecture.channel_alignment,
        "activation": adapter.channel_architecture.activation,
        "layer_ids": layer_ids,
        "width_options": widths,
        "ranking_is_nested": True,
        "construction": {
            "data_free": True,
            "real_tokens_used": False,
            "text_used": False,
            "datasets_used": False,
            "labels_used": False,
            "isotropic_baseline": "native_normalized_gaussian_response",
            "router_conditioning": "shared_isotropic_prior_native_top_k_rejection",
            "route_weighting": "native_route_coefficient_squared",
        },
        "config": asdict(route_config),
        "effective_zero_threshold": args.effective_zero_threshold,
        "model_provenance": model_provenance,
        "table": tables,
        "diagnostics": diagnostics,
    }
    torch.save(payload, output_dir / "rankings.pt")
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"table", "diagnostics"}
    }
    summary["layer_count"] = len(layer_ids)
    summary["expert_count"] = adapter.num_experts
    summary["diagnostics"] = diagnostics
    (output_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())