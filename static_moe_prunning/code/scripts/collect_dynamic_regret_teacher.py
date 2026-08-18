from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from src.amp_proxy import split_gate_up_proj
from src.channel_runtime import _expert_activation, channel_table_from_payload
from src.model_adapter import maybe_bf16_autocast
from src.model_structure import iter_moe_layer_bindings
from src.runtime_pruner import (
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    route_qwen3_topk,
)
from src.calibration_data import (
    collect_contiguous_text_tokens,
    load_calibration_text_dataset,
    load_shared_calibration_tokens,
)
from src.dynamic_regret import compute_dynamic_regret_batch
from src.committee_regret import diagonal_block_committee_residual
from src.model_loading import load_supported_moe


def _load_score_table(path: Path) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {int(layer): values.float().cpu() for layer, values in payload["table"].items()}


@dataclass
class DynamicRegretAccumulator:
    block_values: dict[int, torch.Tensor] = field(default_factory=dict)
    unconditional_block_values: dict[int, torch.Tensor] = field(default_factory=dict)
    block_demands: dict[int, torch.Tensor] = field(default_factory=dict)
    route_counts: dict[int, torch.Tensor] = field(default_factory=dict)
    width_histogram: dict[int, int] = field(default_factory=dict)
    output_saliency_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    co_route_context_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    block_committee_residual_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    block_committee_residual_counts: dict[int, torch.Tensor] = field(default_factory=dict)

    def update(self, layer_idx: int, batch) -> None:
        layer = int(layer_idx)
        values = batch.block_values.detach().double().cpu()
        unconditional = batch.unconditional_block_values.detach().double().cpu()
        demands = batch.block_demands.detach().double().cpu()
        routes = batch.route_counts.detach().double().cpu()
        if layer not in self.block_values:
            self.block_values[layer] = torch.zeros_like(values)
            self.unconditional_block_values[layer] = torch.zeros_like(unconditional)
            self.block_demands[layer] = torch.zeros_like(demands)
            self.route_counts[layer] = torch.zeros_like(routes)
        self.block_values[layer] += values
        self.unconditional_block_values[layer] += unconditional
        self.block_demands[layer] += demands
        self.route_counts[layer] += routes
        unique, counts = torch.unique(batch.widths.detach().cpu(), return_counts=True)
        for width, count in zip(unique.tolist(), counts.tolist()):
            self.width_histogram[int(width)] = (
                self.width_histogram.get(int(width), 0) + int(count)
            )

    def update_output_saliency(
        self,
        layer_idx: int,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_outputs: torch.Tensor,
        *,
        num_experts: int,
    ) -> None:
        layer = int(layer_idx)
        contributions = (
            routing_weights.float() * expert_outputs.float().norm(dim=-1)
        ).reshape(-1)
        selected = selected_experts.to(contributions.device, torch.long).reshape(-1)
        sums = torch.zeros(num_experts, device=contributions.device, dtype=torch.float64)
        sums.index_add_(0, selected, contributions.to(torch.float64))
        if layer not in self.output_saliency_sums:
            self.output_saliency_sums[layer] = torch.zeros_like(sums.cpu())
        self.output_saliency_sums[layer] += sums.cpu()

        token_weights = torch.zeros(
            selected_experts.numel() // selected_experts.shape[-1],
            num_experts,
            device=routing_weights.device,
            dtype=torch.float64,
        )
        token_weights.scatter_add_(
            1,
            selected_experts.to(token_weights.device, torch.long).reshape(
                token_weights.shape[0], -1
            ),
            routing_weights.to(torch.float64).reshape(token_weights.shape[0], -1),
        )
        context = token_weights.transpose(0, 1) @ token_weights
        if layer not in self.co_route_context_sums:
            self.co_route_context_sums[layer] = torch.zeros_like(context.cpu())
        self.co_route_context_sums[layer] += context.cpu()

    def update_block_committee_regret(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        experts,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_outputs: torch.Tensor,
        channel_layer,
    ) -> None:
        layer = int(layer_idx)
        num_experts = int(channel_layer.ranked_indices.shape[0])
        num_blocks = int(channel_layer.block_sizes.numel())
        if layer not in self.block_committee_residual_sums:
            self.block_committee_residual_sums[layer] = torch.zeros(
                num_experts, num_blocks, dtype=torch.float64
            )
            self.block_committee_residual_counts[layer] = torch.zeros(
                num_experts, dtype=torch.int64
            )
        committee_output = (
            routing_weights.float().unsqueeze(-1) * expert_outputs.float()
        ).sum(dim=1)
        fused = (
            hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
            and not isinstance(experts, torch.nn.ModuleList)
        )
        for expert_idx in torch.unique(selected_experts).tolist():
            positions = torch.nonzero(
                selected_experts == expert_idx, as_tuple=False
            )
            token_idx = positions[:, 0]
            slot_idx = positions[:, 1]
            current_state = hidden_states[token_idx]
            if fused:
                gate_weight, up_weight = split_gate_up_proj(
                    experts.gate_up_proj[expert_idx]
                )
                gate_hidden = F.linear(current_state, gate_weight)
                up_hidden = F.linear(current_state, up_weight)
                middle = experts.act_fn(gate_hidden) * up_hidden
                down_weight = experts.down_proj[expert_idx]
            else:
                expert = experts[expert_idx]
                gate_hidden = F.linear(current_state, expert.gate_proj.weight)
                up_hidden = F.linear(current_state, expert.up_proj.weight)
                middle = _expert_activation(expert, gate_hidden, up_hidden)
                down_weight = expert.down_proj.weight
            slot_weights = routing_weights[token_idx, slot_idx]
            own_output = (
                slot_weights.float().unsqueeze(1)
                * expert_outputs[token_idx, slot_idx].float()
            )
            other_output = committee_output[token_idx] - own_output
            residual = diagonal_block_committee_residual(
                middle,
                down_weight,
                other_output,
                routing_weights=slot_weights,
                ranked_indices=channel_layer.ranked_indices[expert_idx],
                block_sizes=channel_layer.block_sizes,
            )
            self.block_committee_residual_sums[layer][expert_idx] += (
                residual.detach().double().sum(dim=0).cpu()
            )
            self.block_committee_residual_counts[layer][expert_idx] += int(
                residual.shape[0]
            )


@contextmanager
def patch_dynamic_regret_collection(
    model,
    amp_table,
    aimer_table,
    channel_table,
    accumulator: DynamicRegretAccumulator,
    target_pruning_ratio: float,
    parent_mode: str,
    collect_output_saliency: bool = False,
    collect_block_committee_regret: bool = False,
):
    originals = []
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        if not all(layer_idx in table for table in (amp_table, aimer_table, channel_table)):
            continue
        target = binding.patch_target
        original = target.forward
        amp = amp_table[layer_idx]
        aimer = aimer_table[layer_idx]
        channels = channel_table[layer_idx]

        if binding.kind == "mlp":
            top_k = int(binding.top_k)
            norm = binding.norm_topk_prob

            def _forward(
                self,
                hidden_states,
                _layer_idx=layer_idx,
                _amp=amp,
                _aimer=aimer,
                _channels=channels,
                _top_k=top_k,
                _norm=norm,
            ):
                batch_size, sequence, hidden_dim = hidden_states.shape
                flat = hidden_states.reshape(-1, hidden_dim)
                router_logits, gate, selected = route_qwen3_topk(
                    self.gate, flat, top_k=_top_k, norm_topk_prob=_norm
                )
                total_blocks = int(
                    round(
                        _top_k
                        * int(_channels.block_sizes.numel())
                        * (1.0 - target_pruning_ratio)
                    )
                )
                teacher = compute_dynamic_regret_batch(
                    gate=gate,
                    selected_experts=selected,
                    amp_layer=_amp,
                    aimer_layer=_aimer,
                    block_coverage_layer=_channels.block_coverage_scores,
                    total_blocks=total_blocks,
                    num_experts=int(_channels.ranked_indices.shape[0]),
                    parent_mode=parent_mode,
                )
                accumulator.update(_layer_idx, teacher)
                output, expert_outputs, _ = compute_moe_weighted_hidden_states(
                    flat,
                    self.experts,
                    selected,
                    gate,
                    moe_backend=(
                        "torch"
                        if collect_output_saliency or collect_block_committee_regret
                        else "torch_index_add"
                    ),
                )
                if collect_output_saliency:
                    if expert_outputs is None:
                        raise RuntimeError("output saliency requires materialized expert outputs.")
                    accumulator.update_output_saliency(
                        _layer_idx,
                        selected,
                        gate,
                        expert_outputs,
                        num_experts=int(_channels.ranked_indices.shape[0]),
                    )
                if collect_block_committee_regret:
                    if expert_outputs is None:
                        raise RuntimeError(
                            "block committee regret requires materialized expert outputs."
                        )
                    accumulator.update_block_committee_regret(
                        _layer_idx,
                        flat,
                        self.experts,
                        selected,
                        gate,
                        expert_outputs,
                        _channels,
                    )
                shared = compute_optional_shared_expert_output(
                    flat,
                    shared_expert=getattr(self, "shared_expert", None),
                    shared_expert_gate=getattr(self, "shared_expert_gate", None),
                )
                if shared is not None:
                    output = output + shared
                return output.reshape(batch_size, sequence, hidden_dim), router_logits

        else:

            def _forward(
                self,
                hidden_states,
                top_k_index,
                top_k_weights,
                _layer_idx=layer_idx,
                _amp=amp,
                _aimer=aimer,
                _channels=channels,
            ):
                top_k = int(top_k_index.shape[-1])
                total_blocks = int(
                    round(
                        top_k
                        * int(_channels.block_sizes.numel())
                        * (1.0 - target_pruning_ratio)
                    )
                )
                teacher = compute_dynamic_regret_batch(
                    gate=top_k_weights,
                    selected_experts=top_k_index,
                    amp_layer=_amp,
                    aimer_layer=_aimer,
                    block_coverage_layer=_channels.block_coverage_scores,
                    total_blocks=total_blocks,
                    num_experts=int(_channels.ranked_indices.shape[0]),
                    parent_mode=parent_mode,
                )
                accumulator.update(_layer_idx, teacher)
                output, expert_outputs, _ = compute_moe_weighted_hidden_states(
                    hidden_states,
                    self,
                    top_k_index,
                    top_k_weights,
                    moe_backend=(
                        "torch"
                        if collect_output_saliency or collect_block_committee_regret
                        else "torch_index_add"
                    ),
                )
                if collect_output_saliency:
                    if expert_outputs is None:
                        raise RuntimeError("output saliency requires materialized expert outputs.")
                    accumulator.update_output_saliency(
                        _layer_idx,
                        top_k_index,
                        top_k_weights,
                        expert_outputs,
                        num_experts=int(_channels.ranked_indices.shape[0]),
                    )
                if collect_block_committee_regret:
                    if expert_outputs is None:
                        raise RuntimeError(
                            "block committee regret requires materialized expert outputs."
                        )
                    accumulator.update_block_committee_regret(
                        _layer_idx,
                        hidden_states,
                        self,
                        top_k_index,
                        top_k_weights,
                        expert_outputs,
                        _channels,
                    )
                return output

        originals.append((target, original))
        target.forward = MethodType(_forward, target)
    if not originals:
        raise ValueError("No Qwen3 MoE layers matched the teacher caches.")
    try:
        yield model
    finally:
        for target, original in originals:
            target.forward = original


def continuous_train_tokens(tokenizer, total_tokens: int, device) -> torch.Tensor:
    dataset, _ = load_calibration_text_dataset(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        text_field="text",
    )
    tokens, _ = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field="text",
        total_tokens=total_tokens,
    )
    return tokens.to(device)


def continuous_calibration_tokens(tokenizer, args, total_tokens: int, device):
    if getattr(args, "calibration_token_cache", None) is not None:
        tokens, source = load_shared_calibration_tokens(
            args.calibration_token_cache,
            required_sequence_length=int(args.sequence_length),
            model_path=getattr(args, "model_path", None),
            device=device,
        )
        if int(tokens.shape[1]) != int(total_tokens):
            raise ValueError("shared calibration cache token count does not match collector arguments.")
        return tokens, source
    dataset, source = load_calibration_text_dataset(
        dataset_name=args.calibration_dataset,
        dataset_config=args.calibration_config,
        split=args.calibration_split,
        text_field=args.calibration_text_field,
        arrow_files=args.calibration_arrow_file,
    )
    tokens, stream = collect_contiguous_text_tokens(
        tokenizer,
        dataset,
        text_field=args.calibration_text_field,
        total_tokens=total_tokens,
        token_offset=args.calibration_token_offset,
        row_batch_size=args.calibration_row_batch_size,
    )
    return tokens.to(device), {**source, "token_stream": stream}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect frozen APA teacher regret for static expert distillation."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--amp-score-cache", type=Path, required=True)
    parser.add_argument("--aimer-score-cache", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, default=0.50)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--calibration-sequences", type=int, default=128)
    parser.add_argument("--calibration-token-offset", type=int, default=0)
    parser.add_argument("--calibration-dataset", default="wikitext")
    parser.add_argument("--calibration-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-text-field", default="text")
    parser.add_argument(
        "--calibration-arrow-file",
        action="append",
        type=Path,
        default=[],
        help="Offline Arrow shard in deterministic order; repeatable.",
    )
    parser.add_argument("--calibration-row-batch-size", type=int, default=1024)
    parser.add_argument("--calibration-token-cache", type=Path, default=None)
    parser.add_argument(
        "--parent-mode",
        choices=("gate", "top_p", "dual", "combined"),
        required=True,
        help="Required explicitly to prevent silent teacher-objective drift.",
    )
    parser.add_argument(
        "--collect-output-saliency",
        action="store_true",
        help="Collect REAP-style mean gate-weighted expert output norms.",
    )
    parser.add_argument(
        "--collect-block-committee-regret",
        action="store_true",
        help=(
            "Collect gate-weighted 64-channel diagonal output residuals "
            "orthogonal to the other routed experts' committee output."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("Dynamic-regret calibration requires a positive sequence length.")
    if not 0.0 <= args.target_pruning_ratio <= 1.0:
        raise ValueError("target-pruning-ratio must be in [0, 1].")
    channel_payload = torch.load(
        args.channel_cache, map_location="cpu", weights_only=True
    )
    if channel_payload.get("split") != "train":
        raise ValueError("channel cache must be calibrated on the train split.")
    if int(channel_payload.get("sequence_length", -1)) != int(args.sequence_length):
        raise ValueError("channel cache sequence length does not match teacher calibration.")
    channel_table = channel_table_from_payload(channel_payload["table"])
    amp_table = _load_score_table(args.amp_score_cache)
    aimer_table = _load_score_table(args.aimer_score_cache)
    model, tokenizer = load_supported_moe(args.model_path)
    bindings = list(iter_moe_layer_bindings(model))
    if not bindings:
        raise ValueError("No supported MoE layers found in the model.")
    top_k_values = {int(binding.top_k) for binding in bindings}
    block_counts = {
        int(channel_table[int(binding.layer_idx)].block_sizes.numel())
        for binding in bindings
        if int(binding.layer_idx) in channel_table
    }
    if len(top_k_values) != 1 or len(block_counts) != 1:
        raise ValueError("Teacher metadata requires uniform top-k and blocks per expert.")
    model_top_k = next(iter(top_k_values))
    model_blocks_per_expert = next(iter(block_counts))
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    total_tokens = args.sequence_length * args.calibration_sequences
    tokens, calibration_source = continuous_calibration_tokens(
        tokenizer, args, total_tokens, device
    )
    accumulator = DynamicRegretAccumulator()
    with patch_dynamic_regret_collection(
        model,
        amp_table,
        aimer_table,
        channel_table,
        accumulator,
        target_pruning_ratio=args.target_pruning_ratio,
        parent_mode=args.parent_mode,
        collect_output_saliency=bool(args.collect_output_saliency),
        collect_block_committee_regret=bool(args.collect_block_committee_regret),
    ):
        for sequence_idx in range(args.calibration_sequences):
            begin = sequence_idx * args.sequence_length
            with torch.inference_mode(), maybe_bf16_autocast():
                model(tokens[:, begin : begin + args.sequence_length], use_cache=False)
            completed = sequence_idx + 1
            if completed == 1 or completed % 8 == 0 or completed == args.calibration_sequences:
                print(
                    f"teacher_progress={completed}/{args.calibration_sequences}",
                    flush=True,
                )

    output_saliency_mean = {}
    if args.collect_output_saliency:
        for layer_idx, sums in accumulator.output_saliency_sums.items():
            routes = accumulator.route_counts[layer_idx].clamp_min(1.0)
            output_saliency_mean[layer_idx] = (sums / routes).float()

    block_committee_residual_mean = {}
    if args.collect_block_committee_regret:
        for layer_idx, sums in accumulator.block_committee_residual_sums.items():
            counts = accumulator.block_committee_residual_counts[layer_idx].clamp_min(1)
            block_committee_residual_mean[layer_idx] = (
                sums / counts.to(torch.float64).unsqueeze(1)
            ).float()

    payload = {
        "schema_version": 1,
        "method": "dynamic_regret_distillation_teacher",
        "teacher": "floor_free_apa_dual_prior",
        "model_path": args.model_path,
        "dataset": calibration_source["dataset"],
        "dataset_name": calibration_source["dataset"],
        "dataset_config": calibration_source["config"],
        "split": calibration_source["split"],
        "text_field": calibration_source["text_field"],
        "calibration_source": calibration_source,
        "calibration_cache_file_sha256": calibration_source.get("cache_file_sha256"),
        "calibration_input_ids_sha256": calibration_source.get("input_ids_sha256"),
        "sequence_length": args.sequence_length,
        "calibration_sequences": args.calibration_sequences,
        "calibration_tokens": total_tokens,
        "calibration_token_offset": int(args.calibration_token_offset),
        "calibration_token_end": int(args.calibration_token_offset) + total_tokens,
        "target_dynamic_pruning_ratio": float(args.target_pruning_ratio),
        "top_k": model_top_k,
        "blocks_per_expert": model_blocks_per_expert,
        "total_blocks_per_token_per_layer": int(
            round(
                model_top_k
                * model_blocks_per_expert
                * (1.0 - args.target_pruning_ratio)
            )
        ),
        "min_blocks_per_slot": 0,
        "parent_mode": args.parent_mode,
        "parent_score": (
            "gate"
            if args.parent_mode == "gate"
            else f"{args.parent_mode}; normalized components; top1=1"
        ),
        "regret_value": "sum_t I(teacher selects block) * parent_score * RMS block coverage",
        "block_values": accumulator.block_values,
        "unconditional_block_values": accumulator.unconditional_block_values,
        "block_demands": accumulator.block_demands,
        "route_counts": accumulator.route_counts,
        "expert_output_saliency_mean": output_saliency_mean,
        "expert_co_route_context": (
            accumulator.co_route_context_sums
            if args.collect_output_saliency
            else {}
        ),
        "output_saliency_formula": (
            "mean_active_token(gate * l2_norm(expert_output))"
            if args.collect_output_saliency
            else None
        ),
        "co_route_formula": (
            "sum_token(dense_topk_gate_outer_product); diagonal retained in cache"
            if args.collect_output_saliency
            else None
        ),
        "expert_block_committee_residual_mean": block_committee_residual_mean,
        "block_committee_residual_counts": (
            accumulator.block_committee_residual_counts
            if args.collect_block_committee_regret
            else {}
        ),
        "block_committee_regret_formula": (
            "mean_active_route(sqrt(sum_channel_in_ranked_block(" 
            "gate^2 * middle^2 * (down_column_norm^2 - "
            "projection_on_other_committee_unit^2))))"
            if args.collect_block_committee_regret
            else None
        ),
        "block_committee_regret_approximation": (
            "diagonal_down_gram; exact committee direction; 64-channel ranked blocks"
            if args.collect_block_committee_regret
            else None
        ),
        "teacher_width_histogram": accumulator.width_histogram,
        "test_metrics_used": False,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_cache)
    print(args.output_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
