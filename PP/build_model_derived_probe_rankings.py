from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from PP.build_gate_hybrid_protection import (
    _load_first_tensor,
    _load_model_config,
    _load_tensor,
    _load_weight_map,
)
from PP.build_protected_rankings import build_protected_artifacts, cache_orders
from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload
from WICK.build_wick_profile import file_sha256, rms_norm_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESP or PWRP model-derived probe rankings.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-cache", type=Path, required=True)
    parser.add_argument("--pp-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--method", choices=("ESP", "PWRP"), required=True)
    parser.add_argument("--probe-count", type=int, default=9)
    parser.add_argument("--top-q", type=int, default=4)
    parser.add_argument("--protection-ratio", type=float, default=0.10)
    parser.add_argument("--retained-blocks", type=int, required=True)
    parser.add_argument("--spectral-oversample", type=int, default=4)
    parser.add_argument("--spectral-power-iterations", type=int, default=4)
    parser.add_argument("--spectral-seed", type=int, default=42)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def swiglu_probe_importance(
    probes: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    top_q: int,
) -> torch.Tensor:
    """Aggregate absolute SwiGLU channel responses without down-projection weighting."""

    if probes.ndim != 2 or gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("probes and gate/up weights must be aligned two-dimensional tensors.")
    if int(probes.shape[1]) != int(gate_weight.shape[1]):
        raise ValueError("probe hidden size must match gate/up input size.")
    selected = int(top_q)
    if not 1 <= selected <= int(probes.shape[0]):
        raise ValueError("top_q must be in [1, number of probes].")
    if not torch.isfinite(probes).all() or not torch.isfinite(gate_weight).all() or not torch.isfinite(up_weight).all():
        raise ValueError("probe and expert weights must contain only finite values.")
    activation = F.silu(F.linear(probes.float(), gate_weight.float())) * F.linear(probes.float(), up_weight.float())
    return torch.topk(activation.abs(), k=selected, dim=0, largest=True, sorted=False).values.mean(dim=0)


def _joint_input_operator(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    gate_projection = gate_weight.float() @ vectors
    up_projection = up_weight.float() @ vectors
    return gate_weight.float().transpose(0, 1) @ gate_projection + up_weight.float().transpose(0, 1) @ up_projection


def expert_spectral_probes(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    router_row: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    probe_count: int,
    oversample: int,
    power_iterations: int,
    seed: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Approximate dominant joint gate/up right singular vectors and orient their signs."""

    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up weights must be aligned two-dimensional tensors.")
    hidden_size = int(gate_weight.shape[1])
    if router_row.shape != (hidden_size,) or norm_weight.shape != (hidden_size,):
        raise ValueError("router and RMSNorm weights must match the expert input size.")
    rank = int(probe_count)
    subspace_size = min(hidden_size, rank + int(oversample))
    if not 1 <= rank <= subspace_size:
        raise ValueError("probe_count and oversample define an invalid spectral subspace.")
    if int(power_iterations) < 1:
        raise ValueError("power_iterations must be positive.")

    generator = torch.Generator(device=gate_weight.device)
    generator.manual_seed(int(seed))
    subspace = torch.randn(
        hidden_size,
        subspace_size,
        dtype=torch.float32,
        device=gate_weight.device,
        generator=generator,
    )
    subspace = torch.linalg.qr(subspace, mode="reduced").Q
    for _ in range(int(power_iterations)):
        subspace = torch.linalg.qr(_joint_input_operator(gate_weight, up_weight, subspace), mode="reduced").Q

    gate_projection = gate_weight.float() @ subspace
    up_projection = up_weight.float() @ subspace
    reduced_operator = gate_projection.transpose(0, 1) @ gate_projection
    reduced_operator += up_projection.transpose(0, 1) @ up_projection
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced_operator)
    descending = torch.argsort(eigenvalues, descending=True, stable=True)[:rank]
    eigenvalues = eigenvalues.index_select(0, descending).clamp_min(0.0)
    directions = subspace @ eigenvectors.index_select(1, descending)
    probes = rms_norm_rows(directions.transpose(0, 1), norm_weight, eps)
    affinities = probes @ router_row.float()
    signs = torch.where(affinities >= 0.0, 1.0, -1.0).unsqueeze(1)
    probes = probes * signs
    total_energy = gate_weight.float().square().sum() + up_weight.float().square().sum()
    concentration = float((eigenvalues.sum() / total_energy.clamp_min(1.0e-12)).item())
    return probes, eigenvalues, concentration


def update_previous_write_topk(
    router_weight: torch.Tensor,
    normalized_candidates: torch.Tensor,
    *,
    probe_count: int,
    best_scores: torch.Tensor | None = None,
    best_probes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one candidate chunk into target-expert previous-write Top-k probes."""

    if router_weight.ndim != 2 or normalized_candidates.ndim != 2:
        raise ValueError("router and candidates must be two-dimensional tensors.")
    if int(router_weight.shape[1]) != int(normalized_candidates.shape[1]):
        raise ValueError("router and candidate hidden sizes must match.")
    selected = min(int(probe_count), int(normalized_candidates.shape[0]))
    if selected < 1:
        raise ValueError("probe_count and candidate count must be positive.")
    logits = router_weight.float() @ normalized_candidates.float().transpose(0, 1)
    chunk_scores, chunk_indices = torch.topk(logits.abs(), k=selected, dim=1, largest=True, sorted=True)
    chunk_logits = torch.gather(logits, 1, chunk_indices)
    chunk_signs = torch.where(chunk_logits >= 0.0, 1.0, -1.0).unsqueeze(2)
    chunk_probes = normalized_candidates[chunk_indices] * chunk_signs
    if best_scores is None or best_probes is None:
        return chunk_scores, chunk_probes

    combined_scores = torch.cat((best_scores, chunk_scores), dim=1)
    combined_probes = torch.cat((best_probes, chunk_probes), dim=1)
    keep = min(int(probe_count), int(combined_scores.shape[1]))
    merged_scores, merged_indices = torch.topk(combined_scores, k=keep, dim=1, largest=True, sorted=True)
    gather_indices = merged_indices.unsqueeze(2).expand(-1, -1, int(combined_probes.shape[2]))
    merged_probes = torch.gather(combined_probes, 1, gather_indices)
    return merged_scores, merged_probes


def previous_write_probes(
    router_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    previous_down_weights: Iterable[torch.Tensor],
    *,
    probe_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select oriented previous-layer down-column probes with streaming Top-k."""

    best_scores = None
    best_probes = None
    for down_weight in previous_down_weights:
        if down_weight.ndim != 2 or int(down_weight.shape[0]) != int(router_weight.shape[1]):
            raise ValueError("previous down weights must have shape [hidden_size, channels].")
        candidates = rms_norm_rows(down_weight.transpose(0, 1), norm_weight, eps)
        best_scores, best_probes = update_previous_write_topk(
            router_weight,
            candidates,
            probe_count=int(probe_count),
            best_scores=best_scores,
            best_probes=best_probes,
        )
    if best_scores is None or best_probes is None:
        raise ValueError("previous_down_weights must be non-empty.")
    return best_probes, best_scores


def order_to_scores(order: torch.Tensor) -> torch.Tensor:
    if order.ndim != 1 or not torch.equal(torch.sort(order.to(torch.long)).values.cpu(), torch.arange(order.numel())):
        raise ValueError("order must be a permutation of all channel indices.")
    scores = torch.empty(order.numel(), dtype=torch.float32, device=order.device)
    scores[order] = torch.arange(order.numel(), 0, -1, dtype=torch.float32, device=order.device)
    return scores


def protection_overlap(pp_order: torch.Tensor, probe_scores: torch.Tensor, protected_channels: int) -> float:
    protected = int(protected_channels)
    probe_order = torch.argsort(probe_scores.float(), descending=True, stable=True)
    return float(torch.isin(pp_order[:protected].to(probe_order.device), probe_order[:protected]).sum().item() / protected)


def aimer_filled_order(
    aimer_order: torch.Tensor,
    pseudo_scores: torch.Tensor,
    protected_channels: int,
) -> torch.Tensor:
    """Place pseudo-protected channels first and preserve the AIMER fill order."""

    if aimer_order.shape != pseudo_scores.shape or aimer_order.ndim != 1:
        raise ValueError("AIMER order and pseudo scores must be aligned one-dimensional tensors.")
    protected = torch.argsort(pseudo_scores.float(), descending=True, stable=True)[: int(protected_channels)]
    mask = torch.zeros(int(aimer_order.numel()), dtype=torch.bool, device=aimer_order.device)
    mask[protected] = True
    return torch.cat((protected, aimer_order[~mask[aimer_order]]))


def summarize_records(records: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in keys:
        values = torch.tensor([record[key] for record in records], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean().item()),
            "p10": float(torch.quantile(values, 0.10).item()),
            "median": float(values.median().item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }
    return summary


def collect_probe_scores(
    *,
    model_path: Path,
    config: dict,
    pp_orders: torch.Tensor,
    method: str,
    probe_count: int,
    top_q: int,
    protection_ratio: float,
    spectral_oversample: int,
    spectral_power_iterations: int,
    spectral_seed: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], list[dict[str, float]]]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    channel_count = int(config["moe_intermediate_size"])
    expected_shape = (num_layers, num_experts, channel_count)
    if tuple(pp_orders.shape) != expected_shape:
        raise ValueError("PP cache dimensions must match the model dimensions.")
    protected_channels = int(round(channel_count * float(protection_ratio)))
    scores_by_layer: dict[int, torch.Tensor] = {}
    records: list[dict[str, float]] = []

    for layer_id in range(num_layers):
        layer_prefix = f"model.layers.{layer_id}"
        router = _load_tensor(model_path, weight_map, f"{layer_prefix}.mlp.gate.weight").to(device=device)
        norm_weight = _load_first_tensor(
            model_path,
            weight_map,
            [
                f"{layer_prefix}.post_attention_layernorm.weight",
                f"{layer_prefix}.pre_feedforward_layernorm.weight",
                f"{layer_prefix}.input_layernorm.weight",
            ],
        ).to(device=device)

        write_probes = None
        write_affinities = None
        if method == "PWRP" and layer_id > 0:
            previous_prefix = f"model.layers.{layer_id - 1}"

            def previous_down_weights() -> Iterable[torch.Tensor]:
                for source_expert_id in range(num_experts):
                    tensor_name = f"{previous_prefix}.mlp.experts.{source_expert_id}.down_proj.weight"
                    yield _load_tensor(model_path, weight_map, tensor_name).to(device=device)

            write_probes, write_affinities = previous_write_probes(
                router,
                norm_weight,
                previous_down_weights(),
                probe_count=int(probe_count),
                eps=float(config["rms_norm_eps"]),
            )

        layer_scores = []
        for expert_id in range(num_experts):
            if method == "PWRP" and layer_id == 0:
                scores = order_to_scores(pp_orders[layer_id, expert_id].to(device=device))
                record = {
                    "pp10_probe10_overlap": 1.0,
                    "selected_affinity_mean": 0.0,
                    "fallback_pp": 1.0,
                }
            else:
                expert_prefix = f"{layer_prefix}.mlp.experts.{expert_id}"
                gate = _load_tensor(model_path, weight_map, f"{expert_prefix}.gate_proj.weight").to(device=device)
                up = _load_tensor(model_path, weight_map, f"{expert_prefix}.up_proj.weight").to(device=device)
                if method == "ESP":
                    probes, eigenvalues, concentration = expert_spectral_probes(
                        gate,
                        up,
                        router[expert_id],
                        norm_weight,
                        probe_count=int(probe_count),
                        oversample=int(spectral_oversample),
                        power_iterations=int(spectral_power_iterations),
                        seed=int(spectral_seed) + layer_id * num_experts + expert_id,
                        eps=float(config["rms_norm_eps"]),
                    )
                    record = {
                        "spectral_concentration": concentration,
                        "smallest_selected_eigenvalue": float(eigenvalues[-1].item()),
                    }
                else:
                    assert write_probes is not None and write_affinities is not None
                    probes = write_probes[expert_id]
                    record = {
                        "selected_affinity_mean": float(write_affinities[expert_id].mean().item()),
                        "fallback_pp": 0.0,
                    }
                scores = swiglu_probe_importance(probes, gate, up, min(int(top_q), int(probes.shape[0])))
                record["pp10_probe10_overlap"] = protection_overlap(
                    pp_orders[layer_id, expert_id], scores, protected_channels
                )
                del gate, up
            record.update({"layer_id": float(layer_id), "expert_id": float(expert_id)})
            records.append(record)
            layer_scores.append(scores.cpu())
        scores_by_layer[layer_id] = torch.stack(layer_scores)
        del router, norm_weight, write_probes, write_affinities
        print(f"Scored {method} layer {layer_id + 1}/{num_layers}", flush=True)
    return scores_by_layer, records


def build_channel_cache(
    *,
    model_path: Path,
    scores_by_layer: dict[int, torch.Tensor],
    method: str,
    block_size: int,
    metadata: dict,
) -> dict:
    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(scores, int(block_size))
        for layer_id, scores in sorted(scores_by_layer.items())
    }
    return {
        "schema_version": 1,
        "purpose": f"{method.lower()}_model_derived_probe_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": int(block_size),
        "table": channel_table_to_payload(tables),
        "model_derived_probes": metadata,
    }


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    aimer_path = args.aimer_cache.expanduser().resolve()
    pp_path = args.pp_cache.expanduser().resolve()
    config = _load_model_config(model_path)
    aimer_orders = cache_orders(torch.load(aimer_path, map_location="cpu", weights_only=True))
    pp_orders = cache_orders(torch.load(pp_path, map_location="cpu", weights_only=True))
    if aimer_orders.shape != pp_orders.shape:
        raise ValueError("AIMER and PP cache dimensions must match.")
    scores_by_layer, records = collect_probe_scores(
        model_path=model_path,
        config=config,
        pp_orders=pp_orders,
        method=args.method,
        probe_count=int(args.probe_count),
        top_q=int(args.top_q),
        protection_ratio=float(args.protection_ratio),
        spectral_oversample=int(args.spectral_oversample),
        spectral_power_iterations=int(args.spectral_power_iterations),
        spectral_seed=int(args.spectral_seed),
        device=torch.device(args.device),
    )
    channel_count = int(aimer_orders.shape[-1])
    protected_channels = int(round(channel_count * float(args.protection_ratio)))
    orders_by_layer = []
    for layer_id in range(int(aimer_orders.shape[0])):
        layer_orders = []
        for expert_id in range(int(aimer_orders.shape[1])):
            layer_orders.append(
                aimer_filled_order(
                    aimer_orders[layer_id, expert_id],
                    scores_by_layer[layer_id][expert_id],
                    protected_channels,
                )
            )
        orders_by_layer.append(torch.stack(layer_orders))
    orders = torch.stack(orders_by_layer)
    summary_keys = ("pp10_probe10_overlap", "spectral_concentration", "smallest_selected_eigenvalue")
    if args.method == "PWRP":
        summary_keys = ("pp10_probe10_overlap", "selected_affinity_mean", "fallback_pp")
    metadata = {
        "method": args.method,
        "probe_count": int(args.probe_count),
        "top_q": int(args.top_q),
        "protection_ratio": float(args.protection_ratio),
        "ranking_criterion": "top_q_mean_absolute_swiglu_response_no_down_norm",
        "pp_cache_sha256": file_sha256(pp_path),
        "aimer_cache_sha256": file_sha256(aimer_path),
        "protected_channels": protected_channels,
        "checkpoint_identity": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "diagnostics": summarize_records(records, summary_keys),
    }
    if args.method == "ESP":
        metadata.update(
            {
                "probe_source": "joint_gate_up_dominant_right_singular_subspace",
                "sign_orientation": "target_router_logit",
                "spectral_oversample": int(args.spectral_oversample),
                "spectral_power_iterations": int(args.spectral_power_iterations),
                "spectral_seed": int(args.spectral_seed),
            }
        )
    else:
        metadata.update(
            {
                "probe_source": "previous_layer_routed_expert_down_columns",
                "candidate_orientation": "absolute_target_router_logit",
                "first_layer_fallback": "PP-Frozen-v1 Top10 protection ranking",
                "shared_expert_candidates": False,
            }
        )
    profile_channel, profile = build_protected_artifacts(
        model_path=model_path,
        orders=orders,
        method=args.method.lower(),
        backbone=f"{args.method.lower()}_protection",
        retained_blocks=int(args.retained_blocks),
        protection_ratio=float(args.protection_ratio),
        block_size=int(args.channel_block_size),
        backbone_cache_sha256=file_sha256(aimer_path),
        pseudo_cache_sha256=file_sha256(pp_path),
    )
    profile["model_derived_probes"] = metadata
    profile_channel["model_derived_probes"] = metadata
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile_channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {"sha256": file_sha256(args.output_channel_cache), "role": args.method.lower()}
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    profile_summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    profile_summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(profile_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_output.write_text(
        json.dumps(
            {
                "method": args.method,
                "expert_count": len(records),
                "summary": summarize_records(records, summary_keys),
                "per_expert": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_channel_cache.resolve())
    print(args.diagnostics_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())