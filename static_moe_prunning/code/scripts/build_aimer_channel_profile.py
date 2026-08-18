from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from src.channel_runtime import _build_layer_channel_table_from_raw_scores, channel_table_to_payload
from src.static_expert_pruning import allocate_static_prefix_widths_per_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a calibration-free channel-wise AIMER profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aimer-root", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
    parser.add_argument(
        "--score-variant",
        choices=("original", "gauge_balanced", "shape", "stable_concat"),
        default="original",
    )
    parser.add_argument("--effective-zero-threshold", type=float, default=1.0e-12)
    return parser.parse_args()


def _git_identity(repository: Path) -> dict[str, object]:
    root = repository.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"repository": str(root), "commit": commit, "tree_dirty": bool(status.strip())}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_aimer_importance(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float = 1.0e-8,
    score_variant: str = "original",
    effective_zero_threshold: float = 1.0e-12,
) -> torch.Tensor:
    """Return inverse AIMER ratios for aligned SwiGLU structural channels."""

    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate/up weights must have the same two-dimensional shape.")
    if down_weight.ndim != 2 or int(down_weight.shape[1]) != int(gate_weight.shape[0]):
        raise ValueError("down weight columns must align with gate/up output channels.")
    gate = gate_weight.detach().float()
    up = up_weight.detach().float()
    down = down_weight.detach().float().transpose(0, 1)
    if effective_zero_threshold < 0:
        raise ValueError("effective-zero threshold must be non-negative.")
    if score_variant == "shape":
        component_scores = []
        for weight in (gate, up, down):
            mean_abs = weight.abs().mean(dim=1)
            rms = weight.square().mean(dim=1).sqrt()
            component_scores.append(rms / mean_abs.clamp_min(eps))
        return torch.stack(component_scores).mean(dim=0).clamp_min(eps)
    if score_variant == "gauge_balanced":
        up_norm = torch.linalg.vector_norm(up, dim=1)
        down_norm = torch.linalg.vector_norm(down, dim=1)
        scale = torch.sqrt(down_norm.clamp_min(eps) / up_norm.clamp_min(eps))
        up = up * scale[:, None]
        down = down / scale[:, None]
    elif score_variant not in {"original", "stable_concat"}:
        raise ValueError(f"Unsupported AIMER score variant: {score_variant}")
    abs_mean = (gate.abs().sum(dim=1) + up.abs().sum(dim=1) + down.abs().sum(dim=1))
    numel = int(gate.shape[1] + up.shape[1] + down.shape[1])
    abs_mean = abs_mean / float(numel)
    root_mean_square = (
        (gate.square().sum(dim=1) + up.square().sum(dim=1) + down.square().sum(dim=1)) / float(numel)
    ).sqrt()
    ratio = abs_mean / root_mean_square.clamp_min(eps)
    importance = ratio.clamp_min(eps).reciprocal()
    if score_variant == "stable_concat":
        max_projection_abs = torch.stack(
            (
                gate.abs().max(dim=1).values,
                up.abs().max(dim=1).values,
                down.abs().max(dim=1).values,
            )
        ).max(dim=0).values
        importance[max_projection_abs < effective_zero_threshold] = -torch.inf
    return importance


def build_aimer_channel_artifacts(
    *,
    model_path: Path,
    raw_scores_by_layer: dict[int, torch.Tensor],
    target_pruning_ratio: float,
    block_size: int,
    source_identity: dict[str, object],
    score_variant: str = "original",
    effective_zero_threshold: float = 1.0e-12,
) -> tuple[dict, dict]:
    if not 0.0 <= float(target_pruning_ratio) <= 1.0:
        raise ValueError("target pruning ratio must be in [0, 1].")
    if not raw_scores_by_layer:
        raise ValueError("raw_scores_by_layer must not be empty.")
    layer_ids = sorted(raw_scores_by_layer)
    tables = {
        layer_id: _build_layer_channel_table_from_raw_scores(raw_scores_by_layer[layer_id], block_size)
        for layer_id in layer_ids
    }
    shapes = {tuple(raw_scores_by_layer[layer_id].shape) for layer_id in layer_ids}
    if len(shapes) != 1:
        raise ValueError("all channel score layers must have the same shape.")
    num_experts, intermediate_size = next(iter(shapes))
    num_blocks = int(tables[layer_ids[0]].block_sizes.numel())
    coverage = torch.stack([tables[layer_id].block_coverage_scores for layer_id in layer_ids])
    maximum_blocks_per_layer = int(num_experts * num_blocks)
    retained_blocks_per_layer = int(round(maximum_blocks_per_layer * (1.0 - float(target_pruning_ratio))))
    budgets = torch.full((len(layer_ids),), retained_blocks_per_layer, dtype=torch.long)
    widths = allocate_static_prefix_widths_per_layer(coverage, total_blocks_by_layer=budgets)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    channel_payload = {
        "schema_version": 1,
        "purpose": f"aimer_{score_variant}_weight_only_channel_ranking",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": int(block_size),
        "table": channel_table_to_payload(tables),
    }
    profile = {
        "schema_version": 1,
        "method": "aimer_channel",
        "mode": "aimer_weight_only_channel",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": int(num_experts),
        "num_blocks": num_blocks,
        "channel_block_size": int(block_size),
        "intermediate_size": int(intermediate_size),
        "allocation_scope": "per_layer",
        "target_blocks_by_layer": budgets.tolist(),
        "actual_blocks_by_layer": widths.sum(dim=1).tolist(),
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": float(target_pruning_ratio),
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": None,
        "profile_widths": widths.cpu(),
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "aimer": {
            "criterion": "inverse_channel_normalized_absolute_mean_over_root_mean_square",
            "score_variant": score_variant,
            "gauge_normalization": (
                "balance_up_down_l2_norm_per_channel" if score_variant == "gauge_balanced" else "none"
            ),
            "shape_definition": (
                "mean_of_per_projection_rms_over_mean_absolute_value_scores"
                if score_variant == "shape" else "none"
            ),
            "effective_zero_policy": (
                {
                    "definition": "max_projection_linf_below_threshold",
                    "threshold": float(effective_zero_threshold),
                    "importance": "negative_infinity",
                }
                if score_variant == "stable_concat" else "none"
            ),
            "higher_importance_is_retained": True,
            "coupled_projections": ["gate_proj", "up_proj", "down_proj"],
            "source_identity": source_identity,
        },
    }
    return channel_payload, profile


def _load_model_config(model_path: Path) -> dict:
    payload = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config = payload.get("text_config", payload)
    if config.get("model_type") not in {"qwen3_moe", "qwen3_5_moe_text"}:
        raise ValueError("Channel-wise AIMER currently supports Qwen3 MoE checkpoints only.")
    return config


def _load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json is missing weight_map.")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def _load_tensor(model_path: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(f"Missing checkpoint tensor: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def collect_channel_scores(
    model_path: Path,
    config: dict,
    score_variant: str = "original",
    effective_zero_threshold: float = 1.0e-12,
) -> dict[int, torch.Tensor]:
    weight_map = _load_weight_map(model_path)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    scores: dict[int, torch.Tensor] = {}
    for layer_id in range(num_layers):
        rows = []
        fused_prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
        fused_gate_up_name = f"{fused_prefix}.gate_up_proj"
        if fused_gate_up_name in weight_map:
            gate_up = _load_tensor(model_path, weight_map, fused_gate_up_name)
            down = _load_tensor(model_path, weight_map, f"{fused_prefix}.down_proj")
            if int(gate_up.shape[0]) != num_experts or int(down.shape[0]) != num_experts:
                raise ValueError("Fused expert tensors do not match the configured expert count.")
            for expert_id in range(num_experts):
                gate, up = gate_up[expert_id].chunk(2, dim=0)
                rows.append(
                    channel_aimer_importance(
                        gate,
                        up,
                        down[expert_id],
                        score_variant=score_variant,
                        effective_zero_threshold=effective_zero_threshold,
                    ).cpu()
                )
        else:
            for expert_id in range(num_experts):
                prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
                gate = _load_tensor(model_path, weight_map, f"{prefix}.gate_proj.weight")
                up = _load_tensor(model_path, weight_map, f"{prefix}.up_proj.weight")
                down = _load_tensor(model_path, weight_map, f"{prefix}.down_proj.weight")
                rows.append(
                    channel_aimer_importance(
                        gate,
                        up,
                        down,
                        score_variant=score_variant,
                        effective_zero_threshold=effective_zero_threshold,
                    ).cpu()
                )
        scores[layer_id] = torch.stack(rows)
        print(f"Scored layer {layer_id + 1}/{num_layers}", flush=True)
    return scores


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    config = _load_model_config(model_path)
    scores = collect_channel_scores(
        model_path,
        config,
        score_variant=args.score_variant,
        effective_zero_threshold=float(args.effective_zero_threshold),
    )
    channel, profile = build_aimer_channel_artifacts(
        model_path=model_path,
        raw_scores_by_layer=scores,
        target_pruning_ratio=float(args.target_pruning_ratio),
        block_size=int(args.channel_block_size),
        source_identity=_git_identity(args.aimer_root),
        score_variant=args.score_variant,
        effective_zero_threshold=float(args.effective_zero_threshold),
    )
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel, args.output_channel_cache)
    profile["cache_provenance"] = {
        "channel": {
            "sha256": _file_sha256(args.output_channel_cache),
            "role": "weight_only_channel_ranking",
        }
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["width_histogram"] = {
        str(int(width)): int(count)
        for width, count in zip(*torch.unique(profile["profile_widths"], return_counts=True))
    }
    args.output_profile.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())