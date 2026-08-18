from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch

CODE_ROOT = Path(__file__).resolve().parents[2] / "static_moe_prunning" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ramp_protocol import (  # noqa: E402
    DEFAULT_SPLIT_QUOTAS,
    E1_SPLIT_QUOTAS,
    build_stratified_split_indices,
    index_tensor_sha256,
    select_representative_experts,
)
from ramp_statistics import RoutedExpertCovarianceAccumulator  # noqa: E402
from src.calibration_data import load_shared_calibration_tokens  # noqa: E402
from src.model_adapter import maybe_bf16_autocast  # noqa: E402
from src.model_loading import load_supported_moe  # noqa: E402
from src.model_structure import iter_moe_layer_bindings  # noqa: E402
from src.runtime_pruner import (  # noqa: E402
    compute_moe_weighted_hidden_states,
    compute_optional_shared_expert_output,
    route_qwen3_topk,
)


def file_sha256(path: Path) -> str:
    """Hash a file in streaming chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RAMP routed-expert covariance statistics.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-family", default="qwen3")
    parser.add_argument("--calibration-token-cache", type=Path, required=True)
    parser.add_argument("--rms-channel-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--experiment", choices=("RAMP-E0", "RAMP-E1"), default="RAMP-E0")
    parser.add_argument("--split", choices=("fit", "validation", "fit_validation"), default="fit_validation")
    parser.add_argument(
        "--max-sequences-per-split",
        type=int,
        default=None,
        help="Smoke-only cap applied after frozen split construction; omit for formal collection.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a dictionary payload: {path}")
    return payload


def build_manifest(
    token_payload: dict,
    token_cache_path: Path,
    rms_payload: dict,
    rms_cache_path: Path,
    splits: dict[str, torch.Tensor],
    representatives: list[dict[str, int | str]],
    args: argparse.Namespace,
) -> dict:
    """Create the provenance manifest embedded in the covariance artifact."""

    return {
        "schema_version": 1,
        "experiment": str(args.experiment),
        "model_path": str(args.model_path.resolve()),
        "model_family": str(args.model_family),
        "calibration_token_cache": {
            "path": str(token_cache_path.resolve()),
            "sha256": file_sha256(token_cache_path),
            "input_ids_sha256": token_payload.get("input_ids_sha256"),
            "protocol_name": token_payload.get("protocol_name"),
            "source": token_payload.get("source"),
        },
        "rms_channel_cache": {
            "path": str(rms_cache_path.resolve()),
            "sha256": file_sha256(rms_cache_path),
            "input_ids_sha256": rms_payload.get("calibration_input_ids_sha256"),
        },
        "sequence_length": int(args.sequence_length),
        "seed": int(args.seed),
        "split_indices": {
            name: {
                "count": int(indices.numel()),
                "sha256": index_tensor_sha256(indices),
                "indices": indices.tolist(),
            }
            for name, indices in splits.items()
        },
        "representative_experts": representatives,
        "test_metrics_used_for_selection": False,
        "audit_collected": False,
    }


@contextmanager
def patch_ramp_collection(model, accumulator: RoutedExpertCovarianceAccumulator):
    """Patch MoE layers while preserving the dense-equivalent forward result."""

    originals = []
    for binding in iter_moe_layer_bindings(model):
        layer_idx = int(binding.layer_idx)
        target = binding.patch_target
        accumulator.initialize_layer(layer_idx, binding.experts)
        original = target.forward
        if binding.kind != "mlp":
            raise NotImplementedError("RAMP collector currently supports Qwen3 MLP experts only.")
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
            accumulator.update(_layer_idx, flat, self.experts, selected, gate)
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

        originals.append((target, target.forward))
        target.forward = MethodType(_forward, target)
    try:
        yield model
    finally:
        for target, original in originals:
            target.forward = original


def collect_split(
    model,
    tokens: torch.Tensor,
    sequence_indices: torch.Tensor,
    sequence_length: int,
    target_experts: dict[int, tuple[int, ...]],
    *,
    device: torch.device,
) -> dict[int, dict[int, dict[str, object]]]:
    """Collect one non-audit split and return CPU statistics."""

    accumulator = RoutedExpertCovarianceAccumulator(target_experts)
    with patch_ramp_collection(model, accumulator):
        for progress, sequence_idx in enumerate(sequence_indices.tolist(), start=1):
            begin = int(sequence_idx) * int(sequence_length)
            input_ids = tokens[:, begin : begin + int(sequence_length)].to(device)
            with torch.inference_mode(), maybe_bf16_autocast():
                model(input_ids, use_cache=False)
            if progress == 1 or progress % 8 == 0 or progress == int(sequence_indices.numel()):
                print(f"split_progress={progress}/{sequence_indices.numel()}", flush=True)
    return accumulator.to_payload()


def merge_split_payloads(
    split_payloads: dict[str, dict[int, dict[int, dict[str, object]]]],
) -> dict[int, dict[int, dict[str, object]]]:
    """Keep split-specific statistics under one serializable payload."""

    merged: dict[int, dict[int, dict[str, object]]] = {}
    layer_ids = sorted({layer for stats in split_payloads.values() for layer in stats})
    for layer_idx in layer_ids:
        merged[int(layer_idx)] = {}
        expert_ids = sorted({expert for stats in split_payloads.values() for expert in stats.get(int(layer_idx), {})})
        for expert_idx in expert_ids:
            down_proj = None
            splits = {}
            for split_name, statistics in split_payloads.items():
                values = dict(statistics.get(int(layer_idx), {}).get(int(expert_idx), {}))
                current_down = values.pop("down_proj", None)
                if current_down is not None:
                    if down_proj is None:
                        down_proj = current_down
                    elif not torch.equal(down_proj, current_down):
                        raise ValueError(f"down_proj changed across splits for layer {layer_idx}, expert {expert_idx}.")
                splits[split_name] = values
            if down_proj is None:
                raise ValueError(f"missing down_proj for layer {layer_idx}, expert {expert_idx}.")
            merged[int(layer_idx)][int(expert_idx)] = {
                "down_proj": down_proj,
                "splits": splits,
            }
    return merged


def main() -> int:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("sequence length must be positive.")
    if args.max_sequences_per_split is not None and int(args.max_sequences_per_split) <= 0:
        raise ValueError("max-sequences-per-split must be positive when provided.")
    if args.split == "fit":
        requested_splits = ("fit",)
    elif args.split == "validation":
        requested_splits = ("validation",)
    else:
        requested_splits = ("fit", "validation")

    token_payload = load_payload(args.calibration_token_cache)
    tokens, token_source = load_shared_calibration_tokens(
        args.calibration_token_cache,
        required_sequence_length=int(args.sequence_length),
        model_path=str(args.model_path),
        device="cpu",
    )
    del token_source
    sequence_order = token_payload.get("source", {}).get("sequence_order")
    if not isinstance(sequence_order, list) or len(sequence_order) != int(token_payload["calibration_sequences"]):
        raise ValueError("calibration cache must include source.sequence_order for RAMP-E0.")
    quotas = E1_SPLIT_QUOTAS if args.experiment == "RAMP-E1" else DEFAULT_SPLIT_QUOTAS
    splits = build_stratified_split_indices(sequence_order, seed=int(args.seed), quotas=quotas)
    rms_payload = load_payload(args.rms_channel_cache)
    representatives = select_representative_experts(rms_payload["route_counts"])
    target_experts: dict[int, tuple[int, ...]] = {}
    for item in representatives:
        target_experts.setdefault(int(item["layer"]), tuple())
        target_experts[int(item["layer"])] = tuple(
            sorted(set(target_experts[int(item["layer"])] + (int(item["expert"]),)))
        )

    model, _ = load_supported_moe(
        str(args.model_path),
        device_map=args.device_map,
        model_family=args.model_family,
    )
    device = next(model.parameters()).device
    split_payloads = {}
    collected_indices = {}
    for split_name in requested_splits:
        sequence_indices = splits[split_name]
        if args.max_sequences_per_split is not None:
            sequence_indices = sequence_indices[: int(args.max_sequences_per_split)]
        collected_indices[split_name] = sequence_indices
        print(f"starting_split={split_name}", flush=True)
        split_payloads[split_name] = collect_split(
            model,
            tokens,
            sequence_indices,
            int(args.sequence_length),
            target_experts,
            device=device,
        )

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": str(args.experiment),
        "model_path": str(args.model_path.resolve()),
        "model_family": str(args.model_family),
        "sequence_length": int(args.sequence_length),
        "calibration_sequences": int(token_payload["calibration_sequences"]),
        "calibration_input_ids_sha256": token_payload["input_ids_sha256"],
        "calibration_cache_file_sha256": file_sha256(args.calibration_token_cache),
        "rms_channel_cache_file_sha256": file_sha256(args.rms_channel_cache),
        "split": args.split,
        "split_indices": {
            name: {
                "frozen_count": int(splits[name].numel()),
                "frozen_sha256": index_tensor_sha256(splits[name]),
                "frozen_indices": splits[name].tolist(),
                "collected_count": int(collected_indices[name].numel()),
                "collected_sha256": index_tensor_sha256(collected_indices[name]),
                "collected_indices": collected_indices[name].tolist(),
            }
            for name in requested_splits
        },
        "representative_experts": representatives,
        "target_experts": {layer: list(experts) for layer, experts in target_experts.items()},
        "statistics": merge_split_payloads(split_payloads),
        "manifest": build_manifest(
            token_payload,
            args.calibration_token_cache,
            rms_payload,
            args.rms_channel_cache,
            splits,
            representatives,
            args,
        ),
        "test_metrics_used_for_selection": False,
        "audit_collected": False,
        "smoke_only": args.max_sequences_per_split is not None,
        "max_sequences_per_split": args.max_sequences_per_split,
    }
    torch.save(payload, args.output_cache)
    print(args.output_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())