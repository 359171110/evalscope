from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from NAPS.build_naps_artifacts import load_weight_map
from PP.pure_pseudo_model_adapter import PurePseudoModelAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a NAPS-Mask or NAPS-Bounded-Merge checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("mask", "bounded-merge"), required=True)
    return parser.parse_args()


def retained_table(cache: dict, retained_channels: int) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (int(layer_id), expert_id): row[:retained_channels].to(torch.long)
        for layer_id, layer_values in cache["table"].items()
        for expert_id, row in enumerate(layer_values["ranked_indices"])
    }


def merge_columns(
    down: torch.Tensor,
    layer_id: int,
    expert_id: int,
    retained: torch.Tensor,
    merge_payload: dict,
) -> torch.Tensor:
    output = down.index_select(1, retained).float()
    for pair in merge_payload.get("layers", {}).get(layer_id, {}).get(expert_id, {}).get("pairs", []):
        representative = int(pair["representative"])
        position = (retained == representative).nonzero(as_tuple=False)
        if position.numel() == 0:
            continue
        output[:, int(position[0].item())] += float(pair["beta"]) * down[:, int(pair["pruned"])].float()
    return output.to(dtype=down.dtype)


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
    merge = torch.load(artifact_dir / "merge_plan.pt", map_location="cpu", weights_only=True)
    retained_channels = int(profile["profile_widths"].flatten()[0].item() * profile["channel_block_size"])
    weight_map = load_weight_map(model_path)
    adapter = PurePseudoModelAdapter.from_checkpoint(model_path, weight_map)
    retained = retained_table(cache, retained_channels)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    total_size = 0
    shape_changes = {}

    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                changed = tensor
                if adapter.expert_gate_template is not None and ".mlp.experts." in name:
                    parts = name.split(".")
                    layer_id = int(parts[2])
                    expert_id = int(parts[5])
                    ids = retained[(layer_id, expert_id)]
                    if name.endswith(("gate_proj.weight", "up_proj.weight")):
                        changed = tensor.index_select(0, ids)
                    elif name.endswith("down_proj.weight"):
                        changed = merge_columns(tensor, layer_id, expert_id, ids, merge) if args.variant == "bounded-merge" else tensor.index_select(1, ids)
                elif adapter.expert_gate_template is None and ".mlp.experts.gate_up_proj" in name:
                    layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                    ids = [retained[(layer_id, expert_id)] for expert_id in range(adapter.num_experts)]
                    selected = []
                    for expert_id, expert_ids in enumerate(ids):
                        index = torch.cat((expert_ids, expert_ids + adapter.intermediate_size))
                        selected.append(tensor[expert_id].index_select(0, index))
                    changed = torch.stack(selected)
                elif adapter.expert_gate_template is None and ".mlp.experts.down_proj" in name:
                    layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                    selected = []
                    for expert_id in range(adapter.num_experts):
                        ids = retained[(layer_id, expert_id)]
                        selected.append(merge_columns(tensor[expert_id], layer_id, expert_id, ids, merge) if args.variant == "bounded-merge" else tensor[expert_id].index_select(1, ids))
                    changed = torch.stack(selected)
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(tensor.shape):
                    shape_changes[name] = {"source": list(tensor.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})

    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {"config.json", "model.safetensors.index.json"}:
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    text_config = config.get("text_config", config)
    text_config["moe_intermediate_size"] = retained_channels
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "pruning_export_manifest.json").write_text(
        json.dumps({"schema_version": 1, "method": f"naps_{args.variant}", "source_model": str(model_path), "artifact_dir": str(artifact_dir), "retained_channels": retained_channels, "shape_changes": shape_changes}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())