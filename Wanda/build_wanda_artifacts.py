from __future__ import annotations

import argparse
import hashlib
import json
import torch
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from safetensors import safe_open
from typing import Any

from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload
from Wanda.model_adapter import WandaModelAdapter
from Wanda.wanda_core import build_channel_table, grouped_wanda_score, validate_rankings, weight_only_group_score


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


class CheckpointReader:

    def __init__(self, model_path: Path, weight_map: dict[str, str]) -> None:
        self.model_path = model_path
        self.weight_map = weight_map
        self.stack = ExitStack()
        self.handles: dict[str, Any] = {}

    def __enter__(self) -> "CheckpointReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.stack.close()

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"Missing checkpoint tensor: {name}")
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(
                safe_open(self.model_path / shard, framework="pt", device="cpu")
            )
        return self.handles[shard].get_tensor(name)


def validate_statistics(payload: dict[str, Any], model_path: Path, adapter: WandaModelAdapter) -> None:
    if int(payload.get("schema_version", -1)) != 1 or payload.get("purpose") != "structured_moe_wanda_statistics":
        raise ValueError("Unsupported Wanda statistics payload.")
    if Path(str(payload.get("model_path", ""))).resolve() != model_path:
        raise ValueError("Wanda statistics were built for a different model path.")
    if payload.get("model_family") != adapter.architecture.model_family:
        raise ValueError("Wanda statistics model family does not match the checkpoint.")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("split") != "train":
        raise ValueError("Wanda statistics must come from a train-only calibration cache.")
    if payload["model_provenance"]["config_sha256"] != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after Wanda calibration.")
    if payload["model_provenance"]["weight_index_sha256"] != file_sha256(model_path / "model.safetensors.index.json"):
        raise ValueError("Checkpoint weight index changed after Wanda calibration.")


def collect_scores(
    model_path: Path,
    adapter: WandaModelAdapter,
    weight_map: dict[str, str],
    statistics: dict[str, Any],
) -> tuple[dict[int, torch.Tensor], int]:
    architecture = adapter.architecture
    input_sums = {int(key): value for key, value in statistics["input_square_sums"].items()}
    middle_sums = {int(key): value for key, value in statistics["middle_square_sums"].items()}
    normalizers = {int(key): value for key, value in statistics["weight_sums"].items()}
    scores: dict[int, torch.Tensor] = {}
    unseen_experts = 0
    with CheckpointReader(model_path, weight_map) as reader:
        for layer_id in range(architecture.num_layers):
            if layer_id not in input_sums or layer_id not in middle_sums or layer_id not in normalizers:
                raise ValueError(f"Wanda statistics are missing layer {layer_id}.")
            if architecture.tensor_codec == "packed":
                gate_up = reader.tensor(adapter.gate_up_name(layer_id))
                down = reader.tensor(adapter.down_name(layer_id))
            layer_scores = []
            for expert_id in range(architecture.num_experts):
                if architecture.tensor_codec == "packed":
                    gate, up = gate_up[expert_id].chunk(2, dim=0)
                    expert_down = down[expert_id]
                else:
                    gate = reader.tensor(adapter.gate_name(layer_id, expert_id))
                    up = reader.tensor(adapter.up_name(layer_id, expert_id))
                    expert_down = reader.tensor(adapter.down_name(layer_id, expert_id))
                normalizer = float(normalizers[layer_id][expert_id].item())
                if normalizer > 0:
                    score = grouped_wanda_score(
                        gate,
                        up,
                        expert_down,
                        input_sums[layer_id][expert_id],
                        middle_sums[layer_id][expert_id],
                        normalizer,
                    )
                else:
                    score = weight_only_group_score(gate, up, expert_down)
                    unseen_experts += 1
                layer_scores.append(score.cpu())
            scores[layer_id] = torch.stack(layer_scores)
            print(f"wanda_scoring_progress={layer_id + 1}/{architecture.num_layers}", flush=True)
    return scores, unseen_experts


def clone_uniform_profile(
    profile: dict[str, Any],
    retained_channels: int,
    target_pruning_ratio: float | None = None,
) -> dict[str, Any]:
    cloned = {key: value for key, value in profile.items()}
    block_size = int(cloned["channel_block_size"])
    intermediate_size = int(cloned["intermediate_size"])
    if retained_channels <= 0 or retained_channels >= intermediate_size:
        raise ValueError(
            f"retained_channels must be in (0, {intermediate_size}); got {retained_channels}."
        )
    if retained_channels % block_size != 0:
        raise ValueError(
            f"retained_channels={retained_channels} is not aligned to block_size={block_size}."
        )
    retained_blocks = retained_channels // block_size
    widths = torch.full(tuple(cloned["profile_widths"].shape), retained_blocks, dtype=torch.long)
    actual_ratio = 1.0 - retained_channels / intermediate_size
    requested_ratio = actual_ratio if target_pruning_ratio is None else float(target_pruning_ratio)
    cloned["profile_widths"] = widths
    cloned["retained_channels"] = retained_channels
    cloned["target_pruning_ratio"] = requested_ratio
    cloned["actual_structural_pruning_ratio"] = actual_ratio
    cloned["total_blocks"] = int(widths.sum().item())
    cloned["target_blocks_by_layer"] = widths.sum(dim=1).tolist()
    cloned["actual_blocks_by_layer"] = widths.sum(dim=1).tolist()
    cloned["profile_sha256"] = hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest()
    cloned["created_at"] = datetime.now(timezone.utc).isoformat()
    validate_static_profile_payload(cloned)
    return cloned


def write_profile(profile: dict[str, Any], profile_path: Path) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, profile_path)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["profile_file_sha256"] = file_sha256(profile_path)
    profile_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_artifacts(
    *,
    model_path: Path,
    adapter: WandaModelAdapter,
    statistics_path: Path,
    statistics: dict[str, Any],
    raw_scores: dict[int, torch.Tensor],
    retained_channels: int,
    target_pruning_ratio: float,
    unseen_experts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    architecture = adapter.architecture
    block_size = architecture.channel_alignment
    tables = {layer_id: build_channel_table(raw_scores[layer_id], block_size) for layer_id in raw_scores}
    validate_rankings(tables, architecture.num_layers, architecture.num_experts, architecture.intermediate_size)
    num_blocks = architecture.intermediate_size // block_size
    retained_blocks = retained_channels // block_size
    widths = torch.full((architecture.num_layers, architecture.num_experts), retained_blocks, dtype=torch.long)
    actual_ratio = 1.0 - retained_channels / architecture.intermediate_size
    calibration = statistics["calibration"]
    channel_payload = {
        "schema_version": 1,
        "purpose": "wanda_grouped_channel_ranking",
        "method": "wanda_grouped",
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "architecture": adapter.metadata(),
        "model_provenance": statistics["model_provenance"],
        "split": "train",
        "test_metrics_used": False,
        "block_size": block_size,
        "score_mode": "grouped_wanda_gate_up_down_l2",
        "score_formula": "sqrt(||Wg*RMS(x)||_F,row^2 + ||Wu*RMS(x)||_F,row^2 + RMS(z)^2*||Wd||_2,col^2)",
        "route_weighting": statistics["route_weighting"],
        "unseen_expert_fallback": "weight_only_group_l2",
        "unseen_experts": unseen_experts,
        "calibration": calibration,
        "statistics_path": str(statistics_path),
        "statistics_sha256": file_sha256(statistics_path),
        "table": tables,
    }
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    profile = {
        "schema_version": 1,
        "method": "wanda_grouped",
        "mode": "per_expert_fixed_grouped_wanda",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "profile_construction": "calibrated",
        "calibration_split": "train",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(range(architecture.num_layers)),
        "num_layers": architecture.num_layers,
        "num_experts": architecture.num_experts,
        "num_blocks": num_blocks,
        "channel_block_size": block_size,
        "intermediate_size": architecture.intermediate_size,
        "allocation_scope": "per_expert_fixed",
        "target_blocks_by_layer": widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": float(target_pruning_ratio),
        "actual_structural_pruning_ratio": actual_ratio,
        "retained_channels": retained_channels,
        "retained_expert_mask": None,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "wanda": {
            "score_mode": channel_payload["score_mode"],
            "route_weighting": statistics["route_weighting"],
            "unseen_expert_fallback": channel_payload["unseen_expert_fallback"],
            "unseen_experts": unseen_experts,
            "architecture": adapter.metadata(),
        },
    }
    validate_static_profile_payload(profile)
    return channel_payload, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Wanda channel rankings and a static profile.")
    parser.add_argument("--from-profile", type=Path)
    parser.add_argument("--channel-cache", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--output-channel-cache", type=Path)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float)
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    return parser.parse_args()


def clone_from_existing_profile(args: argparse.Namespace) -> int:
    source_path = args.from_profile.expanduser().resolve()
    profile_path = args.output_profile.expanduser().resolve()
    if args.retained_channels is None:
        raise ValueError("--retained-channels is required with --from-profile.")
    if profile_path == source_path:
        raise ValueError("Refusing to overwrite the source Wanda profile.")
    profile = torch.load(source_path, map_location="cpu", weights_only=True)
    cloned = clone_uniform_profile(profile, int(args.retained_channels), args.target_pruning_ratio)
    if args.channel_cache is not None:
        channel_path = args.channel_cache.expanduser().resolve()
        expected = cloned["cache_provenance"]["channel"]["sha256"]
        actual = file_sha256(channel_path)
        if expected != actual:
            raise ValueError("Wanda channel cache SHA256 does not match the source profile.")
        cloned["cache_provenance"] = {
            **cloned["cache_provenance"],
            "channel": {
                **cloned["cache_provenance"]["channel"],
                "path": str(channel_path),
                "sha256": actual,
            },
        }
    write_profile(cloned, profile_path)
    if abs(cloned["actual_structural_pruning_ratio"] - cloned["target_pruning_ratio"]) > 1.0e-12:
        print(
            f"WARNING: requested pruning ratio {cloned['target_pruning_ratio']:.8f} was aligned to "
            f"{cloned['actual_structural_pruning_ratio']:.8f} ({cloned['retained_channels']} channels).",
            flush=True,
        )
    print(profile_path)
    return 0


def main() -> int:
    args = parse_args()
    if args.from_profile is not None:
        return clone_from_existing_profile(args)
    if args.model_path is None or args.statistics is None or args.output_channel_cache is None:
        raise ValueError("--model-path, --statistics, and --output-channel-cache are required unless --from-profile is set.")
    if (args.retained_channels is None) == (args.target_pruning_ratio is None):
        raise ValueError("Provide exactly one of --retained-channels or --target-pruning-ratio.")
    model_path = args.model_path.expanduser().resolve()
    statistics_path = args.statistics.expanduser().resolve()
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
    weight_map = load_weight_map(model_path)
    adapter = WandaModelAdapter.from_checkpoint(model_path, weight_map)
    validate_statistics(statistics, model_path, adapter)
    if args.retained_channels is not None:
        retained_channels = int(args.retained_channels)
        target_ratio = 1.0 - retained_channels / adapter.architecture.intermediate_size
    else:
        target_ratio = float(args.target_pruning_ratio)
        retained_channels = adapter.architecture.width_for_pruning(target_ratio, args.rounding)
    adapter.architecture.validate_width(retained_channels)
    raw_scores, unseen_experts = collect_scores(model_path, adapter, weight_map, statistics)
    channel, profile = build_artifacts(
        model_path=model_path,
        adapter=adapter,
        statistics_path=statistics_path,
        statistics=statistics,
        raw_scores=raw_scores,
        retained_channels=retained_channels,
        target_pruning_ratio=target_ratio,
        unseen_experts=unseen_experts,
    )
    channel_path = args.output_channel_cache.expanduser().resolve()
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, channel_path)
    profile["cache_provenance"] = {
        "calibration": {
            "path": statistics["calibration"].get("path"),
            "sha256": statistics["calibration"].get("cache_file_sha256"),
            "input_ids_sha256": statistics["calibration"].get("input_ids_sha256"),
            "protocol_name": statistics["calibration"].get("protocol_name"),
            "split": "train",
            "sequence_length": statistics["calibration"].get("sequence_length"),
            "calibration_sequences": statistics["calibration"].get("calibration_sequences"),
            "calibration_tokens": statistics["calibration"].get("calibration_tokens"),
        },
        "statistics": {
            "path": str(statistics_path),
            "sha256": file_sha256(statistics_path)
        },
        "channel": {
            "path": str(channel_path),
            "sha256": file_sha256(channel_path),
            "role": "wanda_ranking"
        },
    }
    profile_path = args.output_profile.expanduser().resolve()
    write_profile(profile, profile_path)
    if abs(profile["actual_structural_pruning_ratio"] - target_ratio) > 1.0e-12:
        print(
            f"WARNING: requested pruning ratio {target_ratio:.8f} was aligned to "
            f"{profile['actual_structural_pruning_ratio']:.8f} ({retained_channels} channels).",
            flush=True,
        )
    print(channel_path)
    print(profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())