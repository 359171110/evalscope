from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Any

from static_moe_prunning.code.src.static_expert_pruning import validate_static_profile_payload
from Wanda.model_adapter import WandaModelAdapter
from Wanda.wanda_core import validate_rankings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def retained_table(cache: dict[str, Any], retained_channels: int) -> dict[tuple[int, int], torch.Tensor]:
    return {(int(layer_id), expert_id): row[:retained_channels].long()
            for layer_id, values in cache["table"].items()
            for expert_id, row in enumerate(values["ranked_indices"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a standard uniformly Wanda-pruned MoE checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--channel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    channel_path = args.channel_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = torch.load(profile_path, map_location="cpu", weights_only=True)
    cache = torch.load(channel_path, map_location="cpu", weights_only=True)
    validate_static_profile_payload(profile)
    if (profile.get("method") != "wanda_grouped" or cache.get("purpose") != "wanda_grouped_channel_ranking"):
        raise ValueError("Expected a Wanda profile and Wanda channel ranking cache.")
    if profile["cache_provenance"]["channel"]["sha256"] != file_sha256(channel_path):
        raise ValueError("Wanda channel cache SHA256 does not match the profile.")
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = {str(name): str(shard) for name, shard in index["weight_map"].items()}
    adapter = WandaModelAdapter.from_checkpoint(model_path, weight_map)
    architecture = adapter.architecture
    if (
        Path(str(profile["model_path"])).resolve() != model_path
        or Path(str(cache["model_path"])).resolve() != model_path
    ):
        raise ValueError("Wanda artifacts were built for a different model path.")
    model_provenance = cache.get("model_provenance", {})
    if model_provenance.get("config_sha256") != file_sha256(model_path / "config.json"):
        raise ValueError("Checkpoint config changed after Wanda ranking construction.")
    if model_provenance.get("weight_index_sha256") != file_sha256(index_path):
        raise ValueError("Checkpoint weight index changed after Wanda ranking construction.")
    widths = profile["profile_widths"].long()
    if not bool((widths == widths.flatten()[0]).all()):
        raise ValueError("A standard HF checkpoint requires one uniform expert width.")
    retained_channels = int(widths.flatten()[0].item()) * int(profile["channel_block_size"])
    architecture.validate_width(retained_channels)
    validate_rankings(
        cache["table"],
        architecture.num_layers,
        architecture.num_experts,
        architecture.intermediate_size,
    )
    retained = retained_table(cache, retained_channels)
    changed_tensors = 0
    shape_changes: dict[str, Any] = {}
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                source = handle.get_tensor(name)
                changed = source
                if architecture.tensor_codec == "separate" and ".experts." in name:
                    layer_id = layer_id_from_name(name)
                    expert_id = expert_id_from_name(name)
                    target_names = {
                        adapter.gate_name(layer_id, expert_id): 0,
                        adapter.up_name(layer_id, expert_id): 0,
                        adapter.down_name(layer_id, expert_id): 1,
                    }
                    if name in target_names:
                        indices = retained[(layer_id, expert_id)]
                        changed = source.index_select(target_names[name], indices)
                        changed_tensors += 1
                elif architecture.tensor_codec == "packed" and ".layers." in name:
                    layer_id = layer_id_from_name(name)
                    if name == adapter.gate_up_name(layer_id):
                        selected = []
                        for expert_id in range(architecture.num_experts):
                            indices = retained[(layer_id, expert_id)]
                            packed_indices = torch.cat((indices, indices + architecture.intermediate_size))
                            selected.append(source[expert_id].index_select(0, packed_indices))
                        changed = torch.stack(selected)
                        changed_tensors += 1
                    elif name == adapter.down_name(layer_id):
                        changed = torch.stack([
                            source[expert_id].index_select(1, retained[(layer_id, expert_id)])
                            for expert_id in range(architecture.num_experts)
                        ])
                        changed_tensors += 1
                tensors[name] = changed.contiguous()
                total_size += changed.numel() * changed.element_size()
                if list(changed.shape) != list(source.shape):
                    shape_changes[name] = {"source": list(source.shape), "exported": list(changed.shape)}
        save_file(tensors, output_dir / shard_name, metadata={"format": "pt"})
        print(f"exported_shard={shard_name}", flush=True)
    tensors_per_layer = 3 * architecture.num_experts if architecture.tensor_codec == "separate" else 2
    expected_changed = architecture.num_layers * tensors_per_layer
    if changed_tensors != expected_changed:
        raise RuntimeError(f"Changed {changed_tensors} routed-expert tensors, expected {expected_changed}.")
    for source in model_path.iterdir():
        if source.suffix == ".safetensors" or source.name in {
            "config.json",
            "model.safetensors.index.json",
            ".git",
        }:
            continue
        target = output_dir / source.name
        shutil.copytree(source, target) if source.is_dir() else shutil.copy2(source, target)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    config.get("text_config", config)["moe_intermediate_size"] = retained_channels
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index.setdefault("metadata", {})["total_size"] = total_size
    (output_dir / "model.safetensors.index.json"
     ).write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": "wanda_grouped",
        "model_family": architecture.model_family,
        "source_model": str(model_path),
        "source_config_sha256": file_sha256(model_path / "config.json"),
        "source_weight_index_sha256": file_sha256(index_path),
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "channel_cache": str(channel_path),
        "channel_cache_sha256": file_sha256(channel_path),
        "retained_channels": retained_channels,
        "source_expert_width": architecture.intermediate_size,
        "actual_structural_pruning_ratio": 1.0 - retained_channels / architecture.intermediate_size,
        "changed_routed_expert_tensors": changed_tensors,
        "exported_weight_bytes": total_size,
        "shape_changes": shape_changes,
        "preserved_scope": "routers, shared experts, dense MLPs, multimodal modules, and auxiliary/MTP tensors",
        "validation_status": {
            "transformers_greedy_smoke": "pending",
            "vllm_health_and_chat": "pending",
            "quick9": "pending",
        },
    }
    (output_dir / "pruning_export_manifest.json"
     ).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())