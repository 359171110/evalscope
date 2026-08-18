from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Mapping

import torch
import torch.nn.functional as F

from src.amp_proxy import split_gate_up_proj
from src.calibration_data import validate_calibration_token_cache_payload
from src.channel_runtime import _expert_activation, channel_table_to_payload
from src.enp_tenp import (
    build_enp_widths,
    build_signed_projection_channel_table,
    build_tenp_widths,
    gate_norm_direction_score_sum,
    signed_projection_scores,
)
from src.model_adapter import maybe_bf16_autocast
from src.model_loading import load_supported_moe
from src.model_structure import iter_moe_layer_bindings
from src.runtime_pruner import compute_optional_shared_expert_output, route_qwen3_topk
from src.static_expert_pruning import validate_static_profile_payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio_tag(routed_param_retention: float) -> str:
    pruning_percent = (1.0 - float(routed_param_retention)) * 100.0
    rounded = round(pruning_percent)
    if abs(pruning_percent - rounded) < 1.0e-9:
        return f"{rounded}pct"
    return f"{pruning_percent:g}pct".replace(".", "p")


def _expert_shape(experts) -> tuple[int, int]:
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj") and not isinstance(
        experts, torch.nn.ModuleList
    ):
        return int(experts.gate_up_proj.shape[0]), int(experts.gate_up_proj.shape[1] // 2)
    if len(experts) <= 0:
        raise ValueError("MoE expert collection is empty.")
    return len(experts), int(experts[0].gate_proj.weight.shape[0])


def apply_enp_zero_token_policy(
    widths: torch.Tensor,
    *,
    zero_token_mask: torch.Tensor,
    num_blocks: int,
    policy: str,
) -> torch.Tensor:
    adjusted = widths.clone()
    if policy == "keep_full":
        adjusted[zero_token_mask] = int(num_blocks)
    elif policy not in {"error", "prune_uniform"}:
        raise ValueError(f"Unsupported ENP zero-token policy: {policy}")
    return adjusted


@dataclass
class EnpTenpAccumulator:
    eps: float = 1.0e-8
    current_domain: str = "all"
    expert_score_sums: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    channel_score_sums: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    route_counts: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)
    gate_weight_sums: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    def set_domain(self, domain: str) -> None:
        resolved = str(domain).strip()
        self.current_domain = resolved or "all"

    def _ensure_layer(self, layer_idx: int, experts, device: torch.device) -> None:
        domain = self.current_domain
        layer = int(layer_idx)
        num_experts, intermediate_size = _expert_shape(experts)
        for storage in (
            self.expert_score_sums,
            self.channel_score_sums,
            self.route_counts,
            self.gate_weight_sums,
        ):
            storage.setdefault(domain, {})
        if layer in self.expert_score_sums[domain]:
            return
        self.expert_score_sums[domain][layer] = torch.zeros(
            num_experts, device=device, dtype=torch.float32
        )
        self.channel_score_sums[domain][layer] = torch.zeros(
            num_experts, intermediate_size, device=device, dtype=torch.float32
        )
        self.route_counts[domain][layer] = torch.zeros(
            num_experts, device=device, dtype=torch.int64
        )
        self.gate_weight_sums[domain][layer] = torch.zeros(
            num_experts, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def update_and_compute_output(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        experts,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_dim].")
        if selected_experts.shape != routing_weights.shape or selected_experts.ndim != 2:
            raise ValueError("selected_experts and routing_weights must have shape [tokens, top_k].")
        self._ensure_layer(layer_idx, experts, hidden_states.device)
        layer = int(layer_idx)
        domain = self.current_domain
        weighted_output = torch.zeros_like(hidden_states)
        fused = (
            hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
            and not isinstance(experts, torch.nn.ModuleList)
        )
        for expert_idx in torch.unique(selected_experts).tolist():
            positions = torch.nonzero(selected_experts == expert_idx, as_tuple=False)
            token_idx = positions[:, 0]
            slot_idx = positions[:, 1]
            current_state = hidden_states.index_select(0, token_idx)
            if fused:
                gate_weight, up_weight = split_gate_up_proj(experts.gate_up_proj[expert_idx])
                gate_hidden = F.linear(current_state, gate_weight)
                up_hidden = F.linear(current_state, up_weight)
                middle = experts.act_fn(gate_hidden) * up_hidden
                down_weight = experts.down_proj[expert_idx]
                expert_output = F.linear(middle, down_weight)
            else:
                expert = experts[expert_idx]
                gate_hidden = F.linear(
                    current_state,
                    expert.gate_proj.weight,
                    getattr(expert.gate_proj, "bias", None),
                )
                up_hidden = F.linear(
                    current_state,
                    expert.up_proj.weight,
                    getattr(expert.up_proj, "bias", None),
                )
                middle = _expert_activation(expert, gate_hidden, up_hidden)
                down_weight = expert.down_proj.weight
                expert_output = F.linear(
                    middle,
                    down_weight,
                    getattr(expert.down_proj, "bias", None),
                )
            gate = routing_weights[token_idx, slot_idx]
            self.expert_score_sums[domain][layer][expert_idx] += gate_norm_direction_score_sum(
                current_state,
                expert_output,
                gate,
                eps=self.eps,
            )
            self.channel_score_sums[domain][layer][expert_idx] += signed_projection_scores(
                middle,
                down_weight,
                eps=self.eps,
            )
            self.route_counts[domain][layer][expert_idx] += int(token_idx.numel())
            self.gate_weight_sums[domain][layer][expert_idx] += gate.float().sum()
            weighted_output.index_add_(
                0,
                token_idx,
                (gate.unsqueeze(-1) * expert_output).to(weighted_output.dtype),
            )
        return weighted_output

    def aggregate(self) -> dict[str, object]:
        domains = sorted(self.expert_score_sums)
        if not domains:
            raise ValueError("No ENP/TENP calibration observations were collected.")
        layer_ids = sorted({layer for domain in domains for layer in self.expert_score_sums[domain]})
        expert_scores: dict[int, torch.Tensor] = {}
        channel_scores: dict[int, torch.Tensor] = {}
        route_counts: dict[int, torch.Tensor] = {}
        gate_weight_means: dict[int, torch.Tensor] = {}
        multi_domain = len(domains) > 1
        for layer_idx in layer_ids:
            template_expert = next(
                self.expert_score_sums[domain][layer_idx]
                for domain in domains
                if layer_idx in self.expert_score_sums[domain]
            )
            template_channel = next(
                self.channel_score_sums[domain][layer_idx]
                for domain in domains
                if layer_idx in self.channel_score_sums[domain]
            )
            mixed_expert = torch.zeros_like(template_expert, device="cpu")
            mixed_channel = torch.zeros_like(template_channel, device="cpu")
            total_routes = torch.zeros_like(template_expert, device="cpu", dtype=torch.int64)
            total_gate = torch.zeros_like(template_expert, device="cpu")
            for domain in domains:
                if layer_idx not in self.expert_score_sums[domain]:
                    continue
                expert = self.expert_score_sums[domain][layer_idx].detach().float().cpu()
                channel = self.channel_score_sums[domain][layer_idx].detach().float().cpu()
                routes = self.route_counts[domain][layer_idx].detach().cpu()
                gate_sum = self.gate_weight_sums[domain][layer_idx].detach().float().cpu()
                if multi_domain:
                    expert_norm = expert.norm().clamp_min(self.eps)
                    mixed_expert += expert / expert_norm
                    observed = routes > 0
                    for expert_idx in torch.nonzero(observed, as_tuple=False).flatten().tolist():
                        values = channel[expert_idx]
                        mixed_channel[expert_idx] += values / values.norm().clamp_min(self.eps)
                else:
                    mixed_expert += expert
                    mixed_channel += channel / routes.clamp_min(1).float().unsqueeze(1)
                total_routes += routes
                total_gate += gate_sum
            expert_scores[layer_idx] = mixed_expert
            channel_scores[layer_idx] = mixed_channel
            route_counts[layer_idx] = total_routes
            gate_weight_means[layer_idx] = total_gate / total_routes.clamp_min(1).float()
        return {
            "domains": domains,
            "domain_balance": "layer_l2_equal_weight" if multi_domain else "single_domain_direct",
            "expert_scores": expert_scores,
            "channel_scores": channel_scores,
            "route_counts": route_counts,
            "gate_weight_means": gate_weight_means,
            "expert_score_sums_by_domain": {
                domain: {
                    layer_idx: values.detach().float().cpu()
                    for layer_idx, values in layers.items()
                }
                for domain, layers in self.expert_score_sums.items()
            },
            "channel_score_sums_by_domain": {
                domain: {
                    layer_idx: values.detach().float().cpu()
                    for layer_idx, values in layers.items()
                }
                for domain, layers in self.channel_score_sums.items()
            },
            "route_counts_by_domain": {
                domain: {
                    layer_idx: values.detach().cpu()
                    for layer_idx, values in layers.items()
                }
                for domain, layers in self.route_counts.items()
            },
        }


@contextmanager
def patch_enp_tenp_collection(model, accumulator: EnpTenpAccumulator):
    originals = []
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        target = binding.patch_target
        original = target.forward
        if binding.kind == "mlp":
            top_k = int(binding.top_k)
            norm_topk_prob = bool(binding.norm_topk_prob)

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _top_k=top_k,
                _norm=norm_topk_prob,
            ):
                batch, sequence, hidden_dim = hidden_states.shape
                flat = hidden_states.reshape(-1, hidden_dim)
                router_logits, gate, selected = route_qwen3_topk(
                    self.gate,
                    flat,
                    top_k=_top_k,
                    norm_topk_prob=_norm,
                )
                output = accumulator.update_and_compute_output(
                    _layer_idx,
                    flat,
                    self.experts,
                    selected,
                    gate,
                )
                shared = compute_optional_shared_expert_output(
                    flat,
                    shared_expert=getattr(self, "shared_expert", None),
                    shared_expert_gate=getattr(self, "shared_expert_gate", None),
                )
                if shared is not None:
                    output = output + shared
                return output.reshape(batch, sequence, hidden_dim), router_logits

        else:

            def _forward(
                self,
                hidden_states,
                top_k_index,
                top_k_weights,
                _layer_idx=layer_idx,
            ):
                return accumulator.update_and_compute_output(
                    _layer_idx,
                    hidden_states,
                    self,
                    top_k_index,
                    top_k_weights,
                )

        originals.append((target, original))
        target.forward = MethodType(_forward, target)
    if not originals:
        raise ValueError("No supported MoE layers were found for ENP/TENP calibration.")
    try:
        yield model
    finally:
        for target, original in originals:
            target.forward = original


def _profile_payload(
    *,
    method: str,
    mode: str,
    widths: torch.Tensor,
    layer_ids: list[int],
    num_blocks: int,
    block_size: int,
    intermediate_size: int,
    model_path: Path,
    calibration_payload: Mapping[str, object],
    calibration_path: Path,
    calibration_file_sha256: str,
    channel_path: Path,
    channel_file_sha256: str,
    routed_param_retention: float,
    method_metadata: Mapping[str, object],
) -> dict[str, object]:
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    payload: dict[str, object] = {
        "schema_version": 1,
        "method": method,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibrated",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": int(widths.shape[1]),
        "num_blocks": int(num_blocks),
        "channel_block_size": int(block_size),
        "intermediate_size": int(intermediate_size),
        "allocation_scope": "per_layer" if method in {"dense", "enp"} else "linear_trapezoid",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "routed_param_retention": float(routed_param_retention),
        "target_pruning_ratio": 1.0 - float(routed_param_retention),
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": None,
        "profile_widths": widths.detach().cpu().to(torch.long),
        "profile_sha256": hashlib.sha256(
            widths.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
        "cache_provenance": {
            "calibration": {
                "path": str(calibration_path),
                "sha256": calibration_file_sha256,
                "input_ids_sha256": calibration_payload.get("input_ids_sha256"),
                "protocol_name": calibration_payload.get("protocol_name"),
                "split": calibration_payload.get("split"),
                "sequence_length": calibration_payload.get("sequence_length"),
                "calibration_sequences": calibration_payload.get("calibration_sequences"),
                "calibration_tokens": calibration_payload.get("calibration_tokens"),
            },
            "channel": {
                "path": str(channel_path),
                "sha256": channel_file_sha256,
                "score_mode": "signed_projection",
                "split": "train",
                "sequence_length": calibration_payload.get("sequence_length"),
                "calibration_sequences": calibration_payload.get("calibration_sequences"),
                "calibration_tokens": calibration_payload.get("calibration_tokens"),
            },
        },
        method: dict(method_metadata),
    }
    validate_static_profile_payload(payload)
    return payload


def _write_profile(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    summary = {key: value for key, value in payload.items() if key != "profile_widths"}
    widths = payload["profile_widths"]
    if not isinstance(widths, torch.Tensor):
        raise TypeError("profile_widths must be a tensor.")
    unique, counts = torch.unique(widths, return_counts=True)
    summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(unique.tolist(), counts.tolist())
    }
    summary["profile_file_sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TENP statistics and freeze Dense, ENP, and TENP static profiles."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-family", default="qwen3")
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--output-statistics", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile-dir", type=Path, required=True)
    parser.add_argument("--routed-param-retention", type=float, nargs="+", default=[0.60])
    parser.add_argument("--important-expert-ratio", type=float, default=0.30)
    parser.add_argument("--shallow-weight", type=float, default=1.0)
    parser.add_argument("--deep-weight", type=float, default=2.0)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--calibration-sequences", type=int, default=512)
    parser.add_argument("--min-tokens-per-expert", type=int, default=32)
    parser.add_argument("--allow-undercovered-experts", action="store_true")
    parser.add_argument(
        "--zero-token-policy",
        choices=("error", "prune_uniform", "keep_full"),
        default="error",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--device",
        default=None,
        help="Used when --device-map none, to load then move the model without accelerate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routed_param_retentions = [float(value) for value in args.routed_param_retention]
    if not routed_param_retentions or any(not 0.0 < value <= 1.0 for value in routed_param_retentions):
        raise ValueError("routed-param-retention values must be in (0, 1].")
    if len(set(routed_param_retentions)) != len(routed_param_retentions):
        raise ValueError("routed-param-retention values must be unique.")
    if any(not 0.0 <= float(args.important_expert_ratio) <= value for value in routed_param_retentions):
        raise ValueError("important-expert-ratio must be in [0, routed-param-retention] for every target.")
    if int(args.channel_block_size) <= 0:
        raise ValueError("channel-block-size must be positive.")
    if int(args.min_tokens_per_expert) < 0:
        raise ValueError("min-tokens-per-expert must be non-negative.")

    model_path = args.model_path.expanduser().resolve()
    calibration_path = args.calibration_cache.expanduser().resolve()
    calibration_payload = torch.load(calibration_path, map_location="cpu", weights_only=True)
    input_ids = validate_calibration_token_cache_payload(
        calibration_payload,
        required_sequence_length=int(args.sequence_length),
        model_path=model_path,
        require_identity=True,
    )
    if int(calibration_payload.get("calibration_sequences", -1)) != int(args.calibration_sequences):
        raise ValueError("calibration cache sequence count does not match --calibration-sequences.")
    sequence_order = calibration_payload.get("source", {}).get("sequence_order")
    if not isinstance(sequence_order, list) or len(sequence_order) != int(args.calibration_sequences):
        sequence_order = ["all"] * int(args.calibration_sequences)

    model, _ = load_supported_moe(
        str(model_path),
        device_map=args.device_map,
        model_family=args.model_family,
        device=args.device,
    )
    bindings = list(iter_moe_layer_bindings(model))
    if not bindings:
        raise ValueError("No supported MoE layers were found in the model.")
    device = next(model.parameters()).device
    accumulator = EnpTenpAccumulator()
    with patch_enp_tenp_collection(model, accumulator):
        for sequence_idx in range(int(args.calibration_sequences)):
            begin = sequence_idx * int(args.sequence_length)
            end = begin + int(args.sequence_length)
            accumulator.set_domain(str(sequence_order[sequence_idx]))
            tokens = input_ids[:, begin:end].to(device)
            with torch.inference_mode(), maybe_bf16_autocast():
                model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
            completed = sequence_idx + 1
            if completed == 1 or completed % 8 == 0 or completed == int(args.calibration_sequences):
                print(f"enp_tenp_progress={completed}/{args.calibration_sequences}", flush=True)

    aggregate = accumulator.aggregate()
    layer_ids = sorted(int(layer_idx) for layer_idx in aggregate["expert_scores"])
    route_counts = aggregate["route_counts"]
    zero_coverage = {
        layer_idx: torch.nonzero(route_counts[layer_idx] == 0, as_tuple=False).flatten().tolist()
        for layer_idx in layer_ids
    }
    zero_coverage = {layer_idx: experts for layer_idx, experts in zero_coverage.items() if experts}
    if zero_coverage and args.zero_token_policy == "error":
        preview = {layer: experts[:8] for layer, experts in list(zero_coverage.items())[:8]}
        raise ValueError(
            "ENP/TENP calibration contains zero-token experts; increase train-only calibration data "
            f"or use --zero-token-policy keep_full. Preview: {preview}"
        )
    if zero_coverage and args.zero_token_policy == "keep_full":
        preview = {layer: experts[:8] for layer, experts in list(zero_coverage.items())[:8]}
        print(
            "WARNING: keeping zero-token experts full width because no channel score was observed. "
            f"Preview: {preview}",
            flush=True,
        )
    elif zero_coverage:
        preview = {layer: experts[:8] for layer, experts in list(zero_coverage.items())[:8]}
        print(
            "WARNING: pruning zero-token experts to the common ENP width using their deterministic "
            f"unobserved-channel order. Preview: {preview}",
            flush=True,
        )
    undercovered_route_counts = {
        layer_idx: {
            int(expert_idx): int(route_counts[layer_idx][expert_idx].item())
            for expert_idx in torch.nonzero(
                (route_counts[layer_idx] > 0)
                & (route_counts[layer_idx] < int(args.min_tokens_per_expert)),
                as_tuple=False,
            ).flatten().tolist()
        }
        for layer_idx in layer_ids
    }
    undercovered_route_counts = {
        layer_idx: counts for layer_idx, counts in undercovered_route_counts.items() if counts
    }
    if undercovered_route_counts and not bool(args.allow_undercovered_experts):
        preview = {
            layer: dict(list(counts.items())[:8])
            for layer, counts in list(undercovered_route_counts.items())[:8]
        }
        raise ValueError(
            "ENP/TENP calibration coverage is below min-tokens-per-expert; increase train-only calibration "
            "data or pass --allow-undercovered-experts to record and accept nonzero low coverage. "
            f"Preview: {preview}"
        )
    if undercovered_route_counts:
        preview = {
            layer: dict(list(counts.items())[:8])
            for layer, counts in list(undercovered_route_counts.items())[:8]
        }
        print(
            "WARNING: accepting nonzero ENP/TENP expert coverage below "
            f"{args.min_tokens_per_expert} tokens. Preview: {preview}",
            flush=True,
        )

    channel_table = build_signed_projection_channel_table(
        aggregate["channel_scores"],
        block_size=int(args.channel_block_size),
    )
    shapes = {
        tuple(aggregate["channel_scores"][layer_idx].shape)
        for layer_idx in layer_ids
    }
    if len(shapes) != 1:
        raise ValueError("ENP/TENP requires uniform expert and channel dimensions across layers.")
    num_experts, intermediate_size = next(iter(shapes))
    num_blocks = int(channel_table[layer_ids[0]].block_sizes.numel())
    expert_scores = torch.stack([aggregate["expert_scores"][layer_idx] for layer_idx in layer_ids])

    calibration_hash = file_sha256(calibration_path)
    statistics_payload = {
        "schema_version": 1,
        "method": "enp_tenp_statistics",
        "model_path": str(model_path),
        "split": "train",
        "calibration_cache_path": str(calibration_path),
        "calibration_cache_file_sha256": calibration_hash,
        "calibration_input_ids_sha256": calibration_payload.get("input_ids_sha256"),
        "protocol_name": calibration_payload.get("protocol_name"),
        "sequence_length": int(args.sequence_length),
        "calibration_sequences": int(args.calibration_sequences),
        "calibration_tokens": int(args.sequence_length) * int(args.calibration_sequences),
        "expert_score_formula": "sum(gate * ||expert_output||_2 * (1 - cosine(input, expert_output)))",
        "channel_score_formula": "mean signed projection; M * ((M @ W_down.T) @ W_down) / ||M @ W_down.T||_2",
        "accumulation_dtype": "float32",
        "min_tokens_per_expert": int(args.min_tokens_per_expert),
        "undercovered_expert_route_counts": undercovered_route_counts,
        "undercovered_experts_explicitly_allowed": bool(args.allow_undercovered_experts),
        "zero_token_policy": args.zero_token_policy,
        "zero_token_experts_by_layer": zero_coverage,
        "test_metrics_used": False,
        **aggregate,
    }
    args.output_statistics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(statistics_payload, args.output_statistics)

    channel_payload = {
        "schema_version": 1,
        "purpose": "enp_tenp_signed_projection_channel_ranking",
        "model_path": str(model_path),
        "score_mode": "signed_projection",
        "score_formula": statistics_payload["channel_score_formula"],
        "split": "train",
        "calibration_source": calibration_payload.get("source"),
        "calibration_cache_file_sha256": calibration_hash,
        "calibration_input_ids_sha256": calibration_payload.get("input_ids_sha256"),
        "sequence_length": int(args.sequence_length),
        "calibration_sequences": int(args.calibration_sequences),
        "calibration_tokens": int(args.sequence_length) * int(args.calibration_sequences),
        "block_size": int(args.channel_block_size),
        "route_counts": route_counts,
        "min_tokens_per_expert": int(args.min_tokens_per_expert),
        "undercovered_expert_route_counts": undercovered_route_counts,
        "undercovered_experts_explicitly_allowed": bool(args.allow_undercovered_experts),
        "zero_token_policy": args.zero_token_policy,
        "zero_token_experts_by_layer": zero_coverage,
        "expert_scores": aggregate["expert_scores"],
        "table": channel_table_to_payload(channel_table),
        "test_metrics_used": False,
    }
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel_payload, args.output_channel_cache)
    channel_hash = file_sha256(args.output_channel_cache)

    dense_widths = torch.full(
        (len(layer_ids), int(num_experts)),
        num_blocks,
        dtype=torch.long,
    )
    zero_token_mask = torch.stack([route_counts[layer_idx] == 0 for layer_idx in layer_ids])
    profiles = {
        args.output_profile_dir / "dense_full_width.pt": _profile_payload(
            method="dense",
            mode="full_width",
            widths=dense_widths,
            layer_ids=layer_ids,
            num_blocks=num_blocks,
            block_size=int(args.channel_block_size),
            intermediate_size=int(intermediate_size),
            model_path=model_path,
            calibration_payload=calibration_payload,
            calibration_path=calibration_path,
            calibration_file_sha256=calibration_hash,
            channel_path=args.output_channel_cache.resolve(),
            channel_file_sha256=channel_hash,
            routed_param_retention=1.0,
            method_metadata={"runtime_role": "matched_dense_reference"},
        ),
    }
    for routed_param_retention in routed_param_retentions:
        enp_widths = build_enp_widths(
            num_layers=len(layer_ids),
            num_experts=int(num_experts),
            num_blocks=num_blocks,
            routed_param_retention=routed_param_retention,
        )
        enp_widths = apply_enp_zero_token_policy(
            enp_widths,
            zero_token_mask=zero_token_mask,
            num_blocks=num_blocks,
            policy=args.zero_token_policy,
        )
        tenp_widths, important_mask, tenp_audit = build_tenp_widths(
            expert_scores,
            num_blocks=num_blocks,
            routed_param_retention=routed_param_retention,
            important_expert_ratio=float(args.important_expert_ratio),
            shallow_weight=float(args.shallow_weight),
            deep_weight=float(args.deep_weight),
            forced_full_mask=zero_token_mask if args.zero_token_policy == "keep_full" else None,
        )
        tag = _ratio_tag(routed_param_retention)
        profiles[args.output_profile_dir / f"enp_{tag}_per_layer.pt"] = _profile_payload(
            method="enp",
            mode="uniform_expert_neuron_pruning",
            widths=enp_widths,
            layer_ids=layer_ids,
            num_blocks=num_blocks,
            block_size=int(args.channel_block_size),
            intermediate_size=int(intermediate_size),
            model_path=model_path,
            calibration_payload=calibration_payload,
            calibration_path=calibration_path,
            calibration_file_sha256=calibration_hash,
            channel_path=args.output_channel_cache.resolve(),
            channel_file_sha256=channel_hash,
            routed_param_retention=routed_param_retention,
            method_metadata={
                "neuron_score": "signed_projection",
                "common_width_blocks": int(enp_widths[0, 0].item()),
                "zero_token_policy": args.zero_token_policy,
                "forced_full_expert_ids_by_layer": {
                    str(layer_id): torch.nonzero(
                        zero_token_mask[row], as_tuple=False
                    ).flatten().tolist()
                    for row, layer_id in enumerate(layer_ids)
                    if bool(zero_token_mask[row].any())
                },
                "router_topology_changed": False,
                "shared_experts_pruned": False,
            },
        )
        profiles[args.output_profile_dir / f"tenp_{tag}_trapezoid.pt"] = _profile_payload(
            method="tenp",
            mode="trapezoidal_important_expert_neuron_pruning",
            widths=tenp_widths,
            layer_ids=layer_ids,
            num_blocks=num_blocks,
            block_size=int(args.channel_block_size),
            intermediate_size=int(intermediate_size),
            model_path=model_path,
            calibration_payload=calibration_payload,
            calibration_path=calibration_path,
            calibration_file_sha256=calibration_hash,
            channel_path=args.output_channel_cache.resolve(),
            channel_file_sha256=channel_hash,
            routed_param_retention=routed_param_retention,
            method_metadata={
                "important_expert_ratio": float(args.important_expert_ratio),
                "expert_score": "gate_norm_direction",
                "neuron_score": "signed_projection",
                "schedule": "linear_largest_remainder",
                "shallow_weight": float(args.shallow_weight),
                "deep_weight": float(args.deep_weight),
                "zero_token_policy": args.zero_token_policy,
                "forced_full_expert_ids_by_layer": {
                    str(layer_id): torch.nonzero(
                        zero_token_mask[row], as_tuple=False
                    ).flatten().tolist()
                    for row, layer_id in enumerate(layer_ids)
                    if bool(zero_token_mask[row].any())
                },
                "full_expert_ids_by_layer": {
                    str(layer_id): torch.nonzero(
                        important_mask[row], as_tuple=False
                    ).flatten().tolist()
                    for row, layer_id in enumerate(layer_ids)
                },
                "router_topology_changed": False,
                "shared_experts_pruned": False,
                **tenp_audit,
            },
        )
    for path, profile in profiles.items():
        _write_profile(path, profile)
        print(path.resolve())
    print(args.output_statistics.resolve())
    print(args.output_channel_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())