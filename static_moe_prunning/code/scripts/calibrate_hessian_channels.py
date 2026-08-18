from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from src.amp_proxy import split_gate_up_proj
from src.channel_runtime import (
    _build_layer_channel_table_from_raw_scores,
    _channel_path_score,
    _expert_activation,
    channel_table_to_payload,
)
from src.model_adapter import maybe_bf16_autocast
from src.model_structure import (
    get_layer_gamma_weight,
    iter_moe_layer_bindings,
)
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
from src.tail_risk import (
    blend_typical_and_tail_score,
    expert_tail_risk_from_channels,
)
from src.model_loading import load_supported_moe


@dataclass
class DownInputHessianAccumulator:
    """Collect E[z^2] for the SwiGLU activation feeding each down projection."""

    square_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    abs_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    max_abs: dict[int, torch.Tensor] = field(default_factory=dict)
    counts: dict[int, torch.Tensor] = field(default_factory=dict)
    route_counts: dict[int, torch.Tensor] = field(default_factory=dict)

    def update(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        experts,
        selected_experts: torch.Tensor,
        route_selected_experts: torch.Tensor | None = None,
    ) -> None:
        fused = (
            hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
            and not isinstance(experts, torch.nn.ModuleList)
        )
        if fused:
            num_experts = int(experts.gate_up_proj.shape[0])
            intermediate_size = int(experts.gate_up_proj.shape[1] // 2)
        else:
            num_experts = len(experts)
            intermediate_size = int(experts[0].gate_proj.weight.shape[0])
        if layer_idx not in self.square_sums:
            self.square_sums[layer_idx] = torch.zeros(
                (num_experts, intermediate_size),
                device=hidden_states.device,
                dtype=torch.float64,
            )
            self.abs_sums[layer_idx] = torch.zeros(
                (num_experts, intermediate_size),
                device=hidden_states.device,
                dtype=torch.float64,
            )
            self.max_abs[layer_idx] = torch.zeros(
                (num_experts, intermediate_size),
                device=hidden_states.device,
                dtype=torch.float64,
            )
            self.counts[layer_idx] = torch.zeros(
                num_experts,
                device=hidden_states.device,
                dtype=torch.int64,
            )
            self.route_counts[layer_idx] = torch.zeros(
                num_experts,
                device=hidden_states.device,
                dtype=torch.int64,
            )

        routed = selected_experts if route_selected_experts is None else route_selected_experts
        self.route_counts[layer_idx] += torch.bincount(
            routed.reshape(-1), minlength=num_experts
        ).to(device=hidden_states.device, dtype=torch.int64)

        for expert_idx in torch.unique(selected_experts).tolist():
            positions = torch.nonzero(selected_experts == expert_idx, as_tuple=False)
            current_state = hidden_states[positions[:, 0]]
            if fused:
                gate_weight, up_weight = split_gate_up_proj(experts.gate_up_proj[expert_idx])
                gate_hidden = F.linear(current_state, gate_weight)
                up_hidden = F.linear(current_state, up_weight)
                middle = experts.act_fn(gate_hidden) * up_hidden
            else:
                expert = experts[expert_idx]
                gate_hidden = F.linear(current_state, expert.gate_proj.weight)
                up_hidden = F.linear(current_state, expert.up_proj.weight)
                middle = _expert_activation(expert, gate_hidden, up_hidden)
            middle_abs = middle.detach().double().abs()
            self.square_sums[layer_idx][expert_idx] += middle_abs.square().sum(dim=0)
            self.abs_sums[layer_idx][expert_idx] += middle_abs.sum(dim=0)
            self.max_abs[layer_idx][expert_idx] = torch.maximum(
                self.max_abs[layer_idx][expert_idx],
                middle_abs.amax(dim=0),
            )
            self.counts[layer_idx][expert_idx] += int(middle.shape[0])


@contextmanager
def patch_hessian_collection(
    model,
    accumulator: DownInputHessianAccumulator,
    *,
    expert_coverage: str = "routed",
):
    """Collect activations while using a Blackwell-safe dense-equivalent MoE path."""

    if expert_coverage not in {"routed", "all"}:
        raise ValueError("expert_coverage must be 'routed' or 'all'.")

    originals = []
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        target = binding.patch_target
        original = target.forward
        if binding.kind == "mlp":
            top_k = binding.top_k
            norm_topk_prob = binding.norm_topk_prob

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
                    self.gate, flat, top_k=_top_k, norm_topk_prob=_norm
                )
                calibration_selected = selected
                if expert_coverage == "all":
                    num_experts = (
                        int(self.experts.gate_up_proj.shape[0])
                        if hasattr(self.experts, "gate_up_proj")
                        else len(self.experts)
                    )
                    calibration_selected = torch.arange(
                        num_experts, device=flat.device
                    ).expand(int(flat.shape[0]), -1)
                accumulator.update(
                    _layer_idx,
                    flat,
                    self.experts,
                    calibration_selected,
                    route_selected_experts=selected,
                )
                output, _, _ = compute_moe_weighted_hidden_states(
                    flat,
                    self.experts,
                    selected,
                    gate,
                    moe_backend="torch_index_add",
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
                calibration_selected = top_k_index
                if expert_coverage == "all":
                    num_experts = (
                        int(self.gate_up_proj.shape[0])
                        if hasattr(self, "gate_up_proj")
                        else len(self)
                    )
                    calibration_selected = torch.arange(
                        num_experts, device=hidden_states.device
                    ).expand(int(hidden_states.shape[0]), -1)
                accumulator.update(
                    _layer_idx,
                    hidden_states,
                    self,
                    calibration_selected,
                    route_selected_experts=top_k_index,
                )
                output, _, _ = compute_moe_weighted_hidden_states(
                    hidden_states,
                    self,
                    top_k_index,
                    top_k_weights,
                    moe_backend="torch_index_add",
                )
                return output

        originals.append((target, original))
        target.forward = MethodType(_forward, target)
    try:
        yield model
    finally:
        for target, original in originals:
            target.forward = original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate diagonal-Hessian proxy rankings for MoE down-input channels."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
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
        "--expert-coverage",
        choices=("routed", "all"),
        default="routed",
        help="Collect channel statistics from routed experts or from all experts for every token.",
    )
    parser.add_argument("--hybrid-output-dir", type=Path, default=None)
    parser.add_argument("--tail-output-dir", type=Path, default=None)
    parser.add_argument(
        "--moment-lambdas",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--tail-lambdas",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 1.0),
    )
    return parser.parse_args()


def continuous_train_tokens(
    tokenizer,
    total_tokens: int,
    device,
    *,
    token_offset: int = 0,
) -> torch.Tensor:
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
        token_offset=token_offset,
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


@torch.no_grad()
def build_moment_table(
    model,
    accumulator: DownInputHessianAccumulator,
    block_size: int,
    moment_lambda: float,
    tail_lambda: float = 0.0,
):
    if not 0.0 <= moment_lambda <= 1.0:
        raise ValueError("moment_lambda must be in [0, 1].")
    table = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        if layer_idx not in accumulator.square_sums:
            continue
        counts = accumulator.counts[layer_idx].clamp_min(1).to(torch.float64)
        mean_square_middle = accumulator.square_sums[layer_idx] / counts[:, None]
        mean_abs_middle = accumulator.abs_sums[layer_idx] / counts[:, None]
        max_abs_middle = accumulator.max_abs[layer_idx]
        experts = binding.experts
        down_square_norms = []
        if hasattr(experts, "down_proj") and not isinstance(experts, torch.nn.ModuleList):
            for expert_idx in range(experts.down_proj.shape[0]):
                down_square_norms.append(
                    experts.down_proj[expert_idx].detach().float().square().sum(dim=0)
                )
        else:
            for expert in experts:
                down_square_norms.append(
                    expert.down_proj.weight.detach().float().square().sum(dim=0)
                )
        down_hessian_diag = torch.stack(down_square_norms).to(
            mean_square_middle.device, dtype=torch.float64
        )
        down_norm = down_hessian_diag.clamp_min(1.0e-16).sqrt()
        rms_middle = mean_square_middle.clamp_min(1.0e-16).sqrt()
        activation_score = mean_abs_middle.clamp_min(1.0e-16) * down_norm
        rms_score = rms_middle * down_norm
        typical_score = activation_score.pow(1.0 - moment_lambda) * rms_score.pow(
            moment_lambda
        )
        tail_score = max_abs_middle * down_norm
        raw_scores = blend_typical_and_tail_score(
            typical_score,
            tail_score,
            tail_lambda=tail_lambda,
        )
        unseen = accumulator.counts[layer_idx] == 0
        if bool(unseen.any()):
            gamma = get_layer_gamma_weight(binding.layer).detach()
            for expert_idx in torch.nonzero(unseen, as_tuple=False).flatten().tolist():
                if hasattr(experts, "gate_up_proj") and not isinstance(
                    experts, torch.nn.ModuleList
                ):
                    gate_weight, up_weight = split_gate_up_proj(
                        experts.gate_up_proj[expert_idx]
                    )
                    down_weight = experts.down_proj[expert_idx]
                else:
                    expert = experts[expert_idx]
                    gate_weight = expert.gate_proj.weight
                    up_weight = expert.up_proj.weight
                    down_weight = expert.down_proj.weight
                fallback = _channel_path_score(
                    gamma,
                    gate_weight,
                    up_weight,
                    down_weight,
                    eps=1.0e-8,
                ).to(raw_scores.device, dtype=raw_scores.dtype)
                raw_scores[expert_idx] = fallback
        table[layer_idx] = _build_layer_channel_table_from_raw_scores(
            raw_scores.cpu(), block_size=block_size, eps=1.0e-8
        )
    return table


@torch.no_grad()
def build_expert_tail_risk(model, accumulator: DownInputHessianAccumulator):
    risk = {}
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        if layer_idx not in accumulator.max_abs:
            continue
        experts = binding.experts
        down_norms = []
        if hasattr(experts, "down_proj") and not isinstance(
            experts, torch.nn.ModuleList
        ):
            for expert_idx in range(experts.down_proj.shape[0]):
                down_norms.append(
                    experts.down_proj[expert_idx]
                    .detach()
                    .float()
                    .square()
                    .sum(dim=0)
                    .sqrt()
                )
        else:
            for expert in experts:
                down_norms.append(
                    expert.down_proj.weight
                    .detach()
                    .float()
                    .square()
                    .sum(dim=0)
                    .sqrt()
                )
        down_norm = torch.stack(down_norms).to(
            accumulator.max_abs[layer_idx].device, dtype=torch.float64
        )
        risk[layer_idx] = expert_tail_risk_from_channels(
            accumulator.max_abs[layer_idx], down_norm
        ).cpu()
    return risk


def main() -> int:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("Hessian channel calibration requires a positive sequence length.")
    model, tokenizer = load_supported_moe(args.model_path)
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    total_tokens = args.sequence_length * args.calibration_sequences
    tokens, calibration_source = continuous_calibration_tokens(
        tokenizer, args, total_tokens, device
    )
    accumulator = DownInputHessianAccumulator()
    with patch_hessian_collection(
        model, accumulator, expert_coverage=args.expert_coverage
    ):
        for sequence_idx in range(args.calibration_sequences):
            begin = sequence_idx * args.sequence_length
            with torch.inference_mode(), maybe_bf16_autocast():
                model(tokens[:, begin : begin + args.sequence_length], use_cache=False)
            completed = sequence_idx + 1
            if completed == 1 or completed % 8 == 0 or completed == args.calibration_sequences:
                print(
                    f"calibration_progress={completed}/{args.calibration_sequences}",
                    flush=True,
                )
    expert_tail_risk = build_expert_tail_risk(model, accumulator)
    def payload(
        table,
        score_mode: str,
        score_formula: str,
        moment_lambda: float,
        tail_lambda: float = 0.0,
    ):
        return {
            "model_path": args.model_path,
            "block_size": args.channel_block_size,
            "score_mode": score_mode,
            "score_formula": score_formula,
            "moment_lambda": moment_lambda,
            "tail_lambda": tail_lambda,
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
            "expert_coverage": args.expert_coverage,
            "expert_activation_counts": {
                int(layer_idx): counts.cpu()
                for layer_idx, counts in accumulator.counts.items()
            },
            "route_counts": {
                int(layer_idx): counts.cpu()
                for layer_idx, counts in accumulator.route_counts.items()
            },
            "unobserved_experts": int(
                sum(
                    (counts == 0).sum().item()
                    for counts in accumulator.route_counts.values()
                )
            ),
            "unseen_expert_fallback": "weight_path_score",
            "expert_tail_risk_proxy": expert_tail_risk,
            "expert_tail_risk_formula": "max_c(max_t|z_tc| * ||W_down[:,c]||_2)",
            "table": channel_table_to_payload(table),
        }

    hessian_table = build_moment_table(
        model, accumulator, args.channel_block_size, moment_lambda=1.0
    )
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        payload(
            hessian_table,
            "down_input_rms_hessian",
            "sqrt(E[z_c^2]) * ||W_down[:,c]||_2",
            1.0,
        ),
        args.output_cache,
    )
    if args.hybrid_output_dir is not None:
        args.hybrid_output_dir.mkdir(parents=True, exist_ok=True)
        for moment_lambda in args.moment_lambdas:
            table = build_moment_table(
                model,
                accumulator,
                args.channel_block_size,
                moment_lambda=float(moment_lambda),
            )
            label = f"{float(moment_lambda):.2f}".replace(".", "p")
            path = args.hybrid_output_dir / f"qwen3_channels_b64_lambda_{label}.pt"
            torch.save(
                payload(
                    table,
                    "activation_rms_moment_hybrid",
                    "(E|z_c|)^(1-lambda) * sqrt(E[z_c^2])^lambda * ||W_down[:,c]||_2",
                    float(moment_lambda),
                ),
                path,
            )
            print(path)
    if args.tail_output_dir is not None:
        args.tail_output_dir.mkdir(parents=True, exist_ok=True)
        for tail_lambda in args.tail_lambdas:
            value = float(tail_lambda)
            table = build_moment_table(
                model,
                accumulator,
                args.channel_block_size,
                moment_lambda=1.0,
                tail_lambda=value,
            )
            label = f"{value:.2f}".replace(".", "p")
            path = (
                args.tail_output_dir
                / f"qwen3_channels_b{args.channel_block_size}_tail_{label}.pt"
            )
            torch.save(
                payload(
                    table,
                    f"down_input_rms_tail_{value:.2f}",
                    "(sqrt(E[z_c^2])*||W_down[:,c]||)^(1-lambda) * "
                    "(max|z_c|*||W_down[:,c]||)^lambda",
                    1.0,
                    value,
                ),
                path,
            )
            print(path)
    print(args.output_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
