from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.evalscope_model_api import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a calibration-free AIMER whole-expert profile.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--aimer-score-cache", type=Path, required=True)
    parser.add_argument("--aimer-root", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float, required=True)
    parser.add_argument("--channel-block-size", type=int, default=64)
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


def _load_model_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_moe":
        raise ValueError("AIMER profile builder currently supports Qwen3 MoE checkpoints only.")
    return payload


def _load_keep_scores(path: Path, model_path: Path) -> tuple[dict, list[int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("method") != "top_p_aimer":
        raise ValueError("AIMER score cache method must be 'top_p_aimer'.")
    cached_model = Path(str(payload.get("model_path", ""))).expanduser().resolve()
    if cached_model != model_path:
        raise ValueError("AIMER score cache model_path does not match the requested checkpoint.")
    table = payload.get("table")
    if not isinstance(table, dict) or not table:
        raise ValueError("AIMER score cache table is missing.")
    layer_ids = sorted(int(layer_id) for layer_id in table)
    rows = [table[layer_id].detach().float().cpu() for layer_id in layer_ids]
    if any(row.ndim != 1 or not bool(torch.isfinite(row).all()) for row in rows):
        raise ValueError("AIMER scores must be finite one-dimensional tensors.")
    if len({int(row.numel()) for row in rows}) != 1:
        raise ValueError("AIMER score layers must have a uniform expert count.")
    return payload, layer_ids, torch.stack(rows)


def _build_topology_channel_payload(
    *,
    model_path: Path,
    layer_ids: list[int],
    num_experts: int,
    intermediate_size: int,
    block_size: int,
) -> dict:
    if block_size <= 0 or intermediate_size <= 0 or intermediate_size % block_size != 0:
        raise ValueError("MoE intermediate size must be divisible by channel block size.")
    block_sizes = torch.full((intermediate_size // block_size,), block_size, dtype=torch.long)
    ranked = torch.arange(intermediate_size, dtype=torch.long).view(1, -1).expand(num_experts, -1).clone()
    relative = torch.ones((num_experts, block_sizes.numel()), dtype=torch.float32)
    coverage = torch.full_like(relative, 1.0 / float(block_sizes.numel()))
    return {
        "schema_version": 1,
        "purpose": "runtime_topology_only",
        "model_path": str(model_path),
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "table": {
            layer_id: {
                "ranked_indices": ranked.clone(),
                "block_relative_scores": relative.clone(),
                "block_coverage_scores": coverage.clone(),
                "block_sizes": block_sizes.clone(),
                "intermediate_size": intermediate_size,
            }
            for layer_id in layer_ids
        },
    }


def build_aimer_profile_payload(
    *,
    model_path: Path,
    layer_ids: list[int],
    keep_scores: torch.Tensor,
    target_pruning_ratio: float,
    num_blocks: int,
    top_k: int,
    score_file_sha256: str,
    channel_file_sha256: str,
    aimer_identity: dict[str, object],
) -> dict:
    if not 0.0 <= target_pruning_ratio <= 1.0:
        raise ValueError("target pruning ratio must be in [0, 1].")
    num_layers, num_experts = (int(size) for size in keep_scores.shape)
    pruned_per_layer = int(round(num_experts * target_pruning_ratio))
    if pruned_per_layer > num_experts - top_k:
        raise ValueError("AIMER pruning must retain at least top_k experts per layer.")
    retained_per_layer = num_experts - pruned_per_layer
    retained_mask = torch.zeros_like(keep_scores, dtype=torch.bool)
    for row_idx in range(num_layers):
        ranked = sorted(range(num_experts), key=lambda expert_id: (-float(keep_scores[row_idx, expert_id]), expert_id))
        retained_mask[row_idx, ranked[:retained_per_layer]] = True
    widths = retained_mask.to(torch.long) * int(num_blocks)
    actual_blocks_by_layer = widths.sum(dim=1).tolist()
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel()) * int(num_blocks)
    return {
        "schema_version": 1,
        "method": "aimer",
        "mode": "aimer_weight_only_whole_expert",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_blocks": int(num_blocks),
        "top_k": int(top_k),
        "allocation_scope": "per_layer",
        "experts_to_prune_per_layer": pruned_per_layer,
        "retained_experts_by_layer": retained_mask.sum(dim=1).tolist(),
        "target_blocks_by_layer": actual_blocks_by_layer,
        "actual_blocks_by_layer": actual_blocks_by_layer,
        "total_blocks": total_blocks,
        "maximum_blocks": maximum_blocks,
        "target_pruning_ratio": float(target_pruning_ratio),
        "actual_structural_pruning_ratio": 1.0 - total_blocks / maximum_blocks,
        "retained_expert_mask": retained_mask,
        "profile_widths": widths,
        "profile_sha256": hashlib.sha256(widths.numpy().tobytes(order="C")).hexdigest(),
        "aimer_keep_scores": keep_scores,
        "aimer": {
            "criterion": "inverse_normalized_absolute_mean_over_root_mean_square",
            "higher_keep_score_is_retained": True,
            "source_identity": aimer_identity,
        },
        "cache_provenance": {
            "channel": {"sha256": channel_file_sha256, "role": "runtime_topology_only"},
            "aimer_scores": {"sha256": score_file_sha256, "role": "weight_only_expert_ranking"},
        },
    }


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    score_path = args.aimer_score_cache.expanduser().resolve()
    config = _load_model_config(model_path)
    _, layer_ids, keep_scores = _load_keep_scores(score_path, model_path)
    if len(layer_ids) != int(config["num_hidden_layers"]) or int(keep_scores.shape[1]) != int(config["num_experts"]):
        raise ValueError("AIMER score shape does not match the model configuration.")
    channel_payload = _build_topology_channel_payload(
        model_path=model_path,
        layer_ids=layer_ids,
        num_experts=int(config["num_experts"]),
        intermediate_size=int(config["moe_intermediate_size"]),
        block_size=int(args.channel_block_size),
    )
    args.output_channel_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(channel_payload, args.output_channel_cache)
    profile = build_aimer_profile_payload(
        model_path=model_path,
        layer_ids=layer_ids,
        keep_scores=keep_scores,
        target_pruning_ratio=float(args.target_pruning_ratio),
        num_blocks=int(config["moe_intermediate_size"]) // int(args.channel_block_size),
        top_k=int(config["num_experts_per_tok"]),
        score_file_sha256=file_sha256(score_path),
        channel_file_sha256=file_sha256(args.output_channel_cache),
        aimer_identity=_git_identity(args.aimer_root),
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, args.output_profile)
    print(args.output_channel_cache.resolve())
    print(args.output_profile.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())