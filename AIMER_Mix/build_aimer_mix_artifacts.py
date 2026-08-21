from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from AIMER_Mix.mix_core import (
    EnergyMode,
    file_sha256,
    mix_channel_importance,
    packed_mix_channel_importance,
    ranking_table,
    validate_rankings,
)
from AIMER_Mix.model_adapter import AIMERMixModelAdapter
from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload


def load_weight_map(model_path: Path) -> dict[str, str]:
    payload = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(name): str(shard) for name, shard in payload["weight_map"].items()}


def _load_named_tensors(model_path: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, torch.Tensor]:
    shards: dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"Missing routed tensor {name}.")
        shards.setdefault(weight_map[name], []).append(name)
    loaded: dict[str, torch.Tensor] = {}
    for shard_name, shard_names in shards.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in shard_names:
                loaded[name] = handle.get_tensor(name)
    return loaded


def score_separate_layer(
    adapter: AIMERMixModelAdapter,
    tensors: dict[str, torch.Tensor],
    layer_id: int,
    energy_mode: EnergyMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    architecture = adapter.architecture
    rows = []
    alphas = []
    for expert_id in range(architecture.num_experts):
        mix, alpha = mix_channel_importance(
            tensors[adapter.gate_name(layer_id, expert_id)],
            tensors[adapter.up_name(layer_id, expert_id)],
            tensors[adapter.down_name(layer_id, expert_id)],
            energy_mode=energy_mode,
        )
        rows.append(mix)
        alphas.append(alpha)
    return torch.stack(rows), torch.tensor(alphas, dtype=torch.float32)


def score_packed_layer(
    adapter: AIMERMixModelAdapter,
    tensors: dict[str, torch.Tensor],
    layer_id: int,
    energy_mode: EnergyMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = tensors[adapter.gate_up_name(layer_id)]
    down = tensors[adapter.down_name(layer_id)]
    if gate_up.ndim != 3 or down.ndim != 3:
        raise ValueError("Packed expert tensors must have a leading expert axis.")
    width = adapter.architecture.intermediate_size
    rows = []
    alphas = []
    for expert_id in range(adapter.architecture.num_experts):
        mix, alpha = packed_mix_channel_importance(gate_up[expert_id], down[expert_id], energy_mode=energy_mode)
        rows.append(mix)
        alphas.append(alpha)
    scores = torch.stack(rows)
    if scores.shape[1] != width:
        raise ValueError("Packed AIMER-Mix width does not match moe_intermediate_size.")
    return scores, torch.tensor(alphas, dtype=torch.float32)


def build_layer_rankings(
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    weight_map: dict[str, str],
    energy_mode: EnergyMode,
) -> dict[int, dict[str, torch.Tensor | int | float]]:
    architecture = adapter.architecture
    tables: dict[int, dict[str, torch.Tensor | int | float]] = {}
    for layer_id in architecture.moe_layer_ids():
        names = adapter.routed_tensor_names(layer_id)
        tensors = _load_named_tensors(model_path, weight_map, names)
        if architecture.tensor_codec == "packed":
            scores, alphas = score_packed_layer(adapter, tensors, layer_id, energy_mode)
        else:
            scores, alphas = score_separate_layer(adapter, tensors, layer_id, energy_mode)
        del tensors
        tables[int(layer_id)] = ranking_table(scores, architecture.channel_alignment, expert_alpha=alphas)
        mean_alpha = float(alphas.mean().item())
        print(f"scored_layer={layer_id} mean_alpha={mean_alpha:.4f}", flush=True)
    return tables


def _mix_metadata(energy_mode: EnergyMode) -> dict[str, Any]:
    return {
        "data_free": True,
        "weight_only": True,
        "accumulator_dtype": "float32",
        "eps": 1.0e-8,
        "effective_zero_threshold": 1.0e-12,
        "aimer_criterion": "rms_over_mean_abs",
        "energy_mode": energy_mode,
        "alpha": "min(mean_path_l2) / max(mean_path_l2)",
        "mix": "alpha * rank(AIMER) + (1-alpha) * rank(energy)",
        "compensation": "none",
    }


def build_channel_payload(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    tables: dict[int, dict[str, torch.Tensor | int | float]],
    energy_mode: EnergyMode,
) -> dict[str, Any]:
    architecture = adapter.architecture
    moe_ids = architecture.moe_layer_ids()
    validate_rankings(
        tables,
        len(moe_ids),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=moe_ids,
    )
    layer_means = [float(tables[layer_id]["mean_alpha"]) for layer_id in moe_ids]
    return {
        "schema_version": 1,
        "purpose": "aimer_mix_ranking",
        "method": "aimer_mix",
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "architecture": adapter.metadata(),
        "model_provenance": {
            "config_sha256": file_sha256(model_path / "config.json"),
            "weight_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        },
        "split": "not_applicable",
        "sequence_length": 0,
        "calibration_sequences": 0,
        "test_metrics_used": False,
        "block_size": architecture.channel_alignment,
        "score_mode": "per_expert_aimer_geom_rank_mix" if energy_mode == "geom" else "per_expert_aimer_l2_rank_mix",
        "table": tables,
        "aimer_mix": {
            **_mix_metadata(energy_mode),
            "mean_alpha_by_layer": layer_means,
            "mean_alpha": float(sum(layer_means) / max(len(layer_means), 1)),
        },
    }


def build_profile(
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    retained_channels: int,
    target_pruning_ratio: float,
    energy_mode: EnergyMode,
    mean_alpha: float | None = None,
) -> dict[str, Any]:
    architecture = adapter.architecture
    moe_ids = architecture.moe_layer_ids()
    block_size = architecture.channel_alignment
    num_blocks = architecture.intermediate_size // block_size
    retained_blocks = retained_channels // block_size
    widths = torch.full((len(moe_ids), architecture.num_experts), retained_blocks, dtype=torch.long)
    total_blocks = int(widths.sum().item())
    maximum_blocks = int(widths.numel() * num_blocks)
    actual_ratio = 1.0 - retained_channels / architecture.intermediate_size
    profile = {
        "schema_version": 1,
        "method": "aimer_mix",
        "mode": "per_expert_fixed_aimer_energy_rank_mix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_family": architecture.model_family,
        "profile_construction": "calibration_free",
        "calibration_split": "not_applicable",
        "calibration_frozen_before_evaluation": True,
        "test_metrics_used_for_profile": False,
        "layer_ids": list(moe_ids),
        "num_layers": len(moe_ids),
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
        "aimer_mix": {
            **_mix_metadata(energy_mode),
            "mean_alpha": mean_alpha,
            "architecture": adapter.metadata(),
        },
    }
    validate_static_profile_payload(profile)
    return profile


def write_profile(profile: dict[str, Any], profile_path: Path) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(profile, profile_path)
    summary = {key: value for key, value in profile.items() if key != "profile_widths"}
    summary["profile_file_sha256"] = file_sha256(profile_path)
    profile_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_retained_indices(channel: dict[str, Any], retained_channels: int, output_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "method": "aimer_mix",
        "retained_channels": int(retained_channels),
        "layer_ids": sorted(int(layer_id) for layer_id in channel["table"]),
        "retained_indices": {
            str(int(layer_id)): table["ranked_indices"][:, :int(retained_channels)].tolist()
            for layer_id, table in channel["table"].items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_existing_cache(
    channel: dict[str, Any],
    *,
    model_path: Path,
    adapter: AIMERMixModelAdapter,
    energy_mode: EnergyMode,
) -> None:
    if int(channel.get("schema_version", -1)) != 1 or channel.get("purpose") != "aimer_mix_ranking":
        raise ValueError("Unsupported AIMER-Mix channel ranking payload.")
    if Path(str(channel.get("model_path", ""))).resolve() != model_path:
        raise ValueError("AIMER-Mix rankings were built for a different model path.")
    if channel.get("model_family") != adapter.architecture.model_family:
        raise ValueError("AIMER-Mix rankings model family does not match the checkpoint.")
    mix_meta = channel.get("aimer_mix", {})
    if mix_meta.get("data_free") is not True:
        raise ValueError("AIMER-Mix rankings must be data-free.")
    if mix_meta.get("energy_mode") != energy_mode:
        raise ValueError(
            f"Existing mix cache uses energy_mode={mix_meta.get('energy_mode')!r}, requested {energy_mode!r}."
        )
    provenance = channel.get("model_provenance", {})
    if provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after AIMER-Mix ranking construction.")
    if provenance.get("weight_index_sha256") != file_sha256(model_path / "model.safetensors.index.json"):
        raise ValueError("Checkpoint weight index changed after AIMER-Mix ranking construction.")
    architecture = adapter.architecture
    validate_rankings(
        channel["table"],
        len(architecture.moe_layer_ids()),
        architecture.num_experts,
        architecture.intermediate_size,
        layer_ids=architecture.moe_layer_ids(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-free AIMER-Mix rankings and a static profile.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-channel-cache", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--target-pruning-ratio", type=float)
    parser.add_argument("--retained-channels", type=int)
    parser.add_argument("--rounding", choices=("floor", "nearest", "ceil"), default="nearest")
    parser.add_argument("--energy-mode", choices=("geom", "l2"), default="geom")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.retained_channels is None) == (args.target_pruning_ratio is None):
        raise ValueError("Provide exactly one of --retained-channels or --target-pruning-ratio.")
    energy_mode: EnergyMode = args.energy_mode
    model_path = args.model_path.expanduser().resolve()
    channel_path = args.output_channel_cache.expanduser().resolve()
    profile_path = args.output_profile.expanduser().resolve()
    weight_map = load_weight_map(model_path)
    adapter = AIMERMixModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if args.retained_channels is not None:
        retained_channels = int(args.retained_channels)
        target_ratio = 1.0 - retained_channels / architecture.intermediate_size
    else:
        target_ratio = float(args.target_pruning_ratio)
        retained_channels = architecture.width_for_pruning(target_ratio, args.rounding)
    architecture.validate_width(retained_channels)

    if channel_path.exists():
        channel = torch.load(channel_path, map_location="cpu", weights_only=True)
        validate_existing_cache(channel, model_path=model_path, adapter=adapter, energy_mode=energy_mode)
    else:
        tables = build_layer_rankings(model_path, adapter, weight_map, energy_mode)
        channel = build_channel_payload(
            model_path=model_path,
            adapter=adapter,
            tables=tables,
            energy_mode=energy_mode,
        )
        channel_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(channel, channel_path)

    profile = build_profile(
        model_path=model_path,
        adapter=adapter,
        retained_channels=retained_channels,
        target_pruning_ratio=target_ratio,
        energy_mode=energy_mode,
        mean_alpha=channel.get("aimer_mix", {}).get("mean_alpha"),
    )
    profile["cache_provenance"] = {
        "channel": {
            "path": str(channel_path),
            "sha256": file_sha256(channel_path),
            "role": "aimer_mix_ranking",
        }
    }
    write_profile(profile, profile_path)
    write_retained_indices(
        channel,
        retained_channels,
        profile_path.with_name(f"aimer_mix_retained_{retained_channels}ch.json"),
    )
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
