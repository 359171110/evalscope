from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from NAPS_v2.build_naps_v2_artifacts import load_weight_map
from NAPS_v2.model_adapter import PurePseudoModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NAPS-v2 Mask or ExpertComp checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("mask", "expertcomp"), required=True)
    return parser.parse_args()


def retained_table(cache: dict[str, Any], retained_channels: int) -> dict[tuple[int, int], torch.Tensor]:
    return {(int(layer_id), expert_id): row[:retained_channels].to(torch.long)
            for layer_id, values in cache["table"].items()
            for expert_id, row in enumerate(values["ranked_indices"])}


def apply_compensation_plan(down: torch.Tensor, retained: torch.Tensor, plan: dict[str, Any]) -> torch.Tensor:
    output = down.index_select(1, retained).float()
    if plan.get("fallback_reason") is not None:
        return output.to(down.dtype)
    targets = [int(value) for value in plan.get("target_channels", [])]
    representatives = plan.get("representative_channels", [])
    coefficients = plan.get("coefficients", [])
    scale = float(plan.get("trust_region_scale", 0.0))
    retained_positions = {int(channel): position for position, channel in enumerate(retained.tolist())}
    if not (len(targets) == len(representatives) == len(coefficients)):
        raise ValueError("Compensation target, representative, and coefficient lengths do not align")
    for target, target_representatives, target_coefficients in zip(targets, representatives, coefficients):
        if len(target_representatives) != len(target_coefficients):
            raise ValueError("Compensation representative and coefficient lengths do not align")
        for representative, coefficient in zip(target_representatives, target_coefficients):
            position = retained_positions.get(int(representative))
            if position is None:
                raise ValueError(f"Compensation representative {representative} is not retained")
            output[:, position] += scale * float(coefficient) * down[:, target].float()
    return output.to(down.dtype)


def layer_id_from_name(name: str) -> int:
    marker = ".layers."
    if marker not in name:
        raise ValueError(f"Cannot parse layer from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def expert_id_from_name(name: str) -> int:
    marker = ".experts."
    if marker not in name:
        raise ValueError(f"Cannot parse expert from tensor name: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(artifact_dir / "rankings.pt", map_location="cpu", weights_only=True)
    profile = torch.load(artifact_dir / "profile.pt", map_location="cpu", weights_only=True)
    compensation = torch.load(artifact_dir / "compensation_plan.pt", map_location="cpu", weights_only=True)
    retained_channels = int(profile["profile_widths"].flatten()[0].item() * profile["channel_block_size"])
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    retained = retained_table(cache, retained_channels)
    source_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    total_size = 0
    shape_changes: dict[str, Any] = {}
    changed_expert_tensors = 0

    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                changed = tensor
                if adapter.expert_gate_template is not None and name.endswith(
                    ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
                ) and ".experts." in name:
                    layer_id = layer_id_from_name(name)
                    expert_id = expert_id_from_name(name)
                    ids = retained[(layer_id, expert_id)]
                    if name.endswith(("gate_proj.weight", "up_proj.weight")):
                        changed = tensor.index_select(0, ids)
                    elif args.variant == "expertcomp":
                        changed = apply_compensation_plan(tensor, ids, compensation["layers"][layer_id][expert_id])
                    else:
                        changed = tensor.index_select(1, ids)
                    changed_expert_tensors += 1
                elif (
                    adapter.expert_gate_template is None and ".layers." in name
                    and name == adapter.expert_gate_up_name(layer_id_from_name(name))
                ):
                    layer_id = layer_id_from_name(name)
                    selected = []
                    for expert_id in range(adapter.num_experts):
                        ids = retained[(layer_id, expert_id)]
                        selected.append(
                            tensor[expert_id].index_select(0, torch.cat((ids, ids + adapter.intermediate_size)))
                        )
                    changed = torch.stack(selected)
                    changed_expert_tensors += 1
                elif (
                    adapter.expert_gate_template is None and ".layers." in name
                    and name == adapter.expert_down_name(layer_id_from_name(name))
                ):
                    layer_id = layer_id_from_name(name)
                    selected = []
                    for expert_id in range(adapter.num_experts):
                        ids = retained[(layer_id, expert_id)]
                        selected.append(
                            apply_compensation_plan(
                                tensor[expert_id], ids, compensation["layers"][layer_id][expert_id]
                            ) if args.variant == "expertcomp" else tensor[expert_id].index_select(1, ids)
                        )
                    changed = torch.stack(selected)
                    changed_expert_tensors += 1
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(tensor.shape):
                    shape_changes[name] = {"source": list(tensor.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})

    expected_changed = adapter.num_layers * (3 * adapter.num_experts if adapter.expert_gate_template is not None else 2)
    if changed_expert_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_expert_tensors} expert tensors, expected {expected_changed}")
    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {
            ".git",
            "config.json",
            "model.safetensors.index.json",
        } or source.name.startswith("."):
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    text_config = source_config.get("text_config", source_config)
    text_config["moe_intermediate_size"] = retained_channels
    (output_dir
     / "config.json").write_text(json.dumps(source_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json"
     ).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "method": f"naps_v2_{args.variant}",
        "source_model": str(model_path),
        "artifact_dir": str(artifact_dir),
        "retained_channels": retained_channels,
        "changed_expert_tensors": changed_expert_tensors,
        "shape_changes": shape_changes
    }
    (output_dir / "pruning_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
