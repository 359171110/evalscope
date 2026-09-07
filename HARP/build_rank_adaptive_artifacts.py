"""Build a HARP-RankAdaptive profile from an existing HARP ranking cache."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from CSP.csp_core import file_sha256
from CSP.model_adapter import CSPModelAdapter
from CSP.build_csp_artifacts_impl import load_weight_map
from HARP.rank_adaptive_core import allocate_v2_rank_adaptive_widths
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def build_profile(
    model_path: Path,
    channel_cache: Path,
    output_profile: Path,
    *,
    low_width: int,
    budget_width: int,
    high_width: int,
    gamma: float = 2.0,
    min_fraction: float = 0.15,
    relative_noise: float = 0.005,
    repeats: int = 32,
) -> dict[str, Any]:
    """Build and save an independent HARP-v2-RankAdaptive profile."""

    model = model_path.expanduser().resolve()
    cache_path = channel_cache.expanduser().resolve()
    profile_path = output_profile.expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"HARP ranking cache does not exist: {cache_path}")
    weight_map = load_weight_map(model)
    adapter = CSPModelAdapter.from_checkpoint(model, weight_map)
    architecture = adapter.architecture
    if budget_width - low_width != high_width - budget_width:
        raise ValueError("RankAdaptive widths must be symmetric around budget_width.")
    for width in (low_width, budget_width, high_width):
        architecture.validate_width(width)

    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("purpose") != "harp_channel_ranking":
        raise ValueError("Expected a HARP channel ranking cache.")
    if Path(str(cache.get("model_path", ""))).resolve() != model:
        raise ValueError("HARP ranking cache was built for a different model path.")
    provenance = cache.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model / "config.json"):
        raise ValueError("Checkpoint config changed after HARP ranking construction.")
    if provenance.get("weight_index_sha256") != file_sha256(model / "model.safetensors.index.json"):
        raise ValueError("Checkpoint weight index changed after HARP ranking construction.")

    layer_ids = list(architecture.moe_layer_ids())
    if sorted(int(layer_id) for layer_id in cache["table"]) != sorted(layer_ids):
        raise ValueError("HARP ranking cache does not cover the model MoE layers.")
    layer_scores = torch.tensor(cache["layer_scores"], dtype=torch.float64)
    expert_scores = [cache["table"][int(layer_id)]["expert_structural_scores"].to(dtype=torch.float64)
                     for layer_id in layer_ids]
    logical_widths, diagnostics = allocate_v2_rank_adaptive_widths(
        layer_scores,
        expert_scores,
        low_width=low_width,
        mid_width=budget_width,
        high_width=high_width,
        gamma=gamma,
        min_fraction=min_fraction,
        relative_noise=relative_noise,
        repeats=repeats,
    )
    block = architecture.channel_alignment
    if bool(((logical_widths % block) != 0).any()):
        raise RuntimeError("RankAdaptive widths must stay aligned.")
    profile_widths = logical_widths // block
    actual_mean = float(logical_widths.to(dtype=torch.float64).mean().item())
    profile: dict[str, Any] = {
        "schema_version": 1,
        "method": "harp",
        "mode": "harp_v2_rank_adaptive_layer_expert_channel_sp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "model_family": architecture.model_family,
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": layer_ids,
        "num_layers": len(layer_ids),
        "num_experts": architecture.num_experts,
        "num_blocks": architecture.intermediate_size // block,
        "channel_block_size": block,
        "intermediate_size": architecture.intermediate_size,
        "allocation_scope": "per_layer_expert_harp_layer_expert_channel_sp",
        "allocation_objective": (
            "harp_v2_layer_sp_waterfill_then_expert_sp_combo_search_then_channel_sp_prefix"
        ),
        "target_blocks_by_layer": profile_widths.sum(dim=1).tolist(),
        "actual_blocks_by_layer": profile_widths.sum(dim=1).tolist(),
        "total_blocks": int(profile_widths.sum().item()),
        "maximum_blocks": int(profile_widths.numel() * (architecture.intermediate_size // block)),
        "target_pruning_ratio": 1.0 - budget_width / architecture.intermediate_size,
        "actual_structural_pruning_ratio": 1.0 - actual_mean / architecture.intermediate_size,
        "retained_channels": None,
        "budget_reference_width": budget_width,
        "width_options": [low_width, budget_width, high_width],
        "padded_intermediate_size": high_width,
        "retained_expert_mask": None,
        "profile_widths": profile_widths,
        "profile_sha256": hashlib.sha256(profile_widths.numpy().tobytes(order="C")).hexdigest(),
        "csp": {
            "data_free": True,
            "weight_only": True,
            "accumulator_dtype": "float32",
            "input_scale_mode": "none",
            "canonicalization": False,
            "architecture": adapter.metadata(),
            "harp": {
                "allocator": "v2_rank_adaptive",
                "layer_scores": [float(value) for value in layer_scores.tolist()],
                "low_width": low_width,
                "budget_width": budget_width,
                "high_width": high_width,
                "rank_adaptive": diagnostics,
            },
        },
        "cache_provenance": {
            "channel": {
                "path": str(cache_path),
                "sha256": file_sha256(cache_path),
                "role": "harp_ranking",
            }
        },
    }
    validate_static_profile_payload(profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, profile_path)
    print(profile_path)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--low-width", type=int, required=True)
    parser.add_argument("--budget-width", type=int, required=True)
    parser.add_argument("--high-width", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--min-fraction", type=float, default=0.15)
    parser.add_argument("--relative-noise", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    build_profile(
        args.model_path,
        args.channel_cache,
        args.output_profile,
        low_width=args.low_width,
        budget_width=args.budget_width,
        high_width=args.high_width,
        gamma=args.gamma,
        min_fraction=args.min_fraction,
        relative_noise=args.relative_noise,
        repeats=args.repeats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())